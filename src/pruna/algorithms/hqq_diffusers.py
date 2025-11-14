# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, Type

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from ConfigSpace import OrdinalHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.target_modules import (
    TARGET_MODULES_TYPE,
    TargetModules,
    get_skipped_submodules,
    is_leaf_module,
    map_targeted_nn_roots,
)
from pruna.engine.model_checks import (
    get_diffusers_transformer_models,
    get_diffusers_unet_models,
)
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.utils import load_json_config
from pruna.logging.filter import SuppressOutput
from pruna.logging.logger import pruna_logger


class HQQDiffusers(PrunaAlgorithmBase):
    """
    Implement HQQ for Image-Gen models.

    Half-Quadratic Quantization (HQQ) leverages fast, robust optimization techniques for on-the-fly quantization,
    eliminating the need for calibration data and making it applicable to any model. This algorithm is specifically
    adapted for diffusers models.
    """

    algorithm_name: str = "hqq_diffusers"
    group_tags: list[str] = [tags.QUANTIZER]
    references: dict[str, str] = {
        "GitHub": "https://github.com/mobiusml/hqq",
        "Article": "https://mobiusml.github.io/hqq_blog/",
    }
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.hqq_diffusers
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda"]
    dataset_required: bool = False
    compatible_before: Iterable[str] = ["qkv_diffusers"]
    compatible_after: Iterable[str] = ["deepcache", "fastercache", "fora", "pab", "torch_compile"]

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            OrdinalHyperparameter(
                "weight_bits",
                sequence=[2, 4, 8],
                default_value=8,
                meta=dict(desc="Number of bits to use for quantization."),
            ),
            OrdinalHyperparameter(
                "group_size",
                sequence=[8, 16, 32, 64, 128],
                default_value=64,
                meta=dict(desc="Group size for quantization."),
            ),
            OrdinalHyperparameter(
                "backend",
                sequence=["gemlite", "bitblas", "torchao_int4", "marlin"],
                default_value="torchao_int4",
                meta=dict(desc="Backend to use for quantization."),
            ),
            TargetModules(
                "target_modules",
                default_value=None,
                meta=dict(
                    desc="Precise choices of which modules to quantize, "
                    "e.g. {include: ['transformer.*']} to quantize only the transformer in a diffusion pipeline. "
                    f"See the {TargetModules.documentation_name_with_link} documentation for more details."
                ),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a unet-based or transformer-based diffusion model.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a diffusion model, False otherwise.
        """
        transformer_and_unet_models = get_diffusers_transformer_models() + get_diffusers_unet_models()

        if isinstance(model, tuple[Any, ...](transformer_and_unet_models)):
            return True

        return any(isinstance(attr_value, tuple(transformer_and_unet_models)) for attr_value in model.__dict__.values())

    def get_model_dependent_hyperparameter_defaults(
        self, model: Any, smash_config: SmashConfigPrefixWrapper
    ) -> TARGET_MODULES_TYPE:
        """
        Get default values for the target_modules based on the model and configuration.

        Parameters
        ----------
        model : Any
            The model to get the default hyperparameters from.
        smash_config : SmashConfig
            The SmashConfig object.

        Returns
        -------
        TARGET_MODULES_TYPE
            The default target_modules for the algorithm.
        """
        include: list[str] = []
        exclude: list[str] = []
        transformer_and_unet_models = get_diffusers_transformer_models() + get_diffusers_unet_models()
        for attr_name, attr_value in model.__dict__.items():
            if isinstance(attr_value, tuple(transformer_and_unet_models)):
                include.append(f"{attr_name}.*")
        if not include:
            include = ["*"]
        return {"include": include, "exclude": exclude}

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Quantize the model.

        Parameters
        ----------
        model : Any
            The model to quantize.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the quantization.

        Returns
        -------
        Any
            The quantized model.
        """
        pruna_logger.debug(
            "HQQ can only save linear layers. So models (e.g. Sana) with separate torch.nn.Parameters or "
            "buffers can not be saved correctly. If some parameters are not saved, handle them manually or "
            "consider selecting a different quantizer."
        )
        imported_modules = self.import_algorithm_packages()

        if smash_config["target_modules"] is None:
            target_modules = self.get_model_dependent_hyperparameter_defaults(model, smash_config)
        else:
            target_modules = smash_config["target_modules"]

        config = imported_modules["HqqConfig"](nbits=smash_config["weight_bits"], group_size=smash_config["group_size"])

        def quantize_component(attr_name: str | None, module: torch.nn.Module, subpaths: list[str]) -> torch.nn.Module:
            """
            Quantize the model if it itself is a transformer or unet, or its components.

            Parameters
            ----------
            attr_name : str | None
                The name of the attribute in the model pointing to the component to quantize.
            module : torch.nn.Module
                The component to quantize.
            subpaths : list[str]
                The subpaths of the component to quantize.

            Returns
            -------
            torch.nn.Module
                The quantized component.
            """
            module.layers = find_module_layers_type(module, nn.Linear)

            if module._class_name == "SD3Transformer2DModel":
                pruna_logger.info(
                    "Your are using SD3Transformer2DModel, please be aware that this transformer is not savable for now."
                )

            ignored_leaf_modules = get_skipped_submodules(module, subpaths, filter_fn=is_leaf_module)
            auto_hqq_hf_diffusers_model = construct_base_class(
                imported_modules, extra_ignore_modules=ignored_leaf_modules
            )

            compute_dtype = next(iter(module.parameters())).dtype
            auto_hqq_hf_diffusers_model.quantize_model(
                module,
                quant_config=config,
                compute_dtype=compute_dtype,
                device=smash_config["device"],
            )

            # skipped layers are not casted to device and compute dtype so we need to do it manually
            for name, submodule in module.named_modules():
                if name in ignored_leaf_modules:
                    submodule.to(smash_config["device"])
                    submodule.to(compute_dtype)

            # Prepare the module for fast inference based on the backend
            # we use the conditions from the hqq documentation
            param_dtype = next(iter(module.parameters())).dtype
            if (
                smash_config["backend"] == "torchao_int4"
                and smash_config["weight_bits"] == 4
                and param_dtype == torch.bfloat16
            ):
                imported_modules["prepare_for_inference"](module, backend="torchao_int4")
            elif (
                smash_config["backend"] == "gemlite"
                and smash_config["weight_bits"] in [4, 2, 1]
                and param_dtype == torch.float16
            ):
                imported_modules["prepare_for_inference"](module, backend="gemlite")
            elif (
                smash_config["backend"] == "bitblas"
                and smash_config["weight_bits"] in [4, 2]
                and param_dtype == torch.float16
            ):
                imported_modules["prepare_for_inference"](module, backend="bitblas")
            else:
                # We default to the torch backend if the input backend is not applicable
                imported_modules["prepare_for_inference"](module)

            unet_types = get_diffusers_unet_models()
            if isinstance(module, tuple(unet_types)):
                for layer in module.up_blocks:
                    if layer.upsamplers is not None:
                        layer.upsamplers[0].name = "conv"

            return module

        model = map_targeted_nn_roots(quantize_component, model, target_modules)
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a method packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        with SuppressOutput():
            from hqq.core.quantize import BaseQuantizeConfig
            from hqq.models.base import BaseHQQModel, BasePatch
            from hqq.utils.patching import prepare_for_inference

        import diffusers

        return dict(
            prepare_for_inference=prepare_for_inference,
            HqqConfig=BaseQuantizeConfig,
            BaseHQQModel=BaseHQQModel,
            BasePatch=BasePatch,
            diffusers=diffusers,
        )


def construct_base_class(imported_modules: Dict[str, Any], extra_ignore_modules: list[str]) -> Type[Any]:
    """
    Construct and return the AutoHQQHFDiffusersModel base class.

    Parameters
    ----------
    imported_modules : Dict[str, Any]
        Dictionary containing imported modules needed for the base class construction.
    extra_ignore_modules : None | list[str], optional
        The names of modules to ignore for quantization in addition to the ones already ignored by BaseHQQModel.

    Returns
    -------
    Type[AutoHQQHFDiffusersModel]
        The constructed AutoHQQHFDiffusersModel class.
    """

    class AutoHQQHFDiffusersModel(imported_modules["BaseHQQModel"], imported_modules["BasePatch"]):  # type: ignore
        """Base class for HQQ Hugging Face Diffusers models."""

        # Save model architecture
        @classmethod
        def cache_model(cls, model: Any, save_dir: str) -> None:
            """
            Cache the model configuration by saving it to disk.

            Parameters
            ----------
            model : Any
                The model whose configuration should be cached.
            save_dir : str
                Directory path where the model configuration will be saved.
            """
            model.save_config(save_dir)

        @classmethod
        def create_model(cls, save_dir: str, kwargs: dict) -> Any:
            """
            Create an empty model from the cached configuration.

            Parameters
            ----------
            save_dir : str
                Directory path where the model configuration is cached.
            kwargs : dict
                Additional keyword arguments for the model creation.

            Returns
            -------
            Any
                The created model.
            """
            model_kwargs: Dict[str, Any] = {}

            with init_empty_weights():
                # recover class from save_dir
                config = load_json_config(save_dir, "config.json")
                model_class = getattr(imported_modules["diffusers"], config["_class_name"])
                model = model_class.from_config(save_dir, **model_kwargs)

            return model

        @classmethod
        def get_ignore_layers(cls, model) -> list[str]:
            """
            Get the layers which should be ignored for quantization.

            Parameters
            ----------
            model : Any
                The model to get the ignore layers from.

            Returns
            -------
            list
                The layers which should be ignored for quantization.
            """
            ignore_layers = super().get_ignore_layers(model)
            return list(set(ignore_layers + extra_ignore_modules))

    return AutoHQQHFDiffusersModel


def find_module_layers_type(model: Any, layer_type: type, exclude_module_names: list[str] = []) -> list:
    """
    Find all layers of a specific type in a model.

    Parameters
    ----------
    model : Any
        The model to search through.
    layer_type : type
        The type of layer to find.
    exclude_module_names : list[str], optional
        The names of the modules to exclude from the search.

    Returns
    -------
    list
        List of found layers matching the specified type.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, layer_type) and name not in exclude_module_names:
            layers.append(module)
    return layers
