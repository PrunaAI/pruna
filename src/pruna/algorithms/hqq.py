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

import shutil
import tempfile
from collections.abc import Iterable
from typing import Any, Dict, Type, cast

import torch
from ConfigSpace import CategoricalHyperparameter, Constant, OrdinalHyperparameter
from transformers import AutoModelForCausalLM, Pipeline

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.target_modules import (
    TARGET_MODULES_TYPE,
    TargetModules,
    expand_list_of_targeted_paths,
    get_skipped_submodules,
    is_leaf_module,
    map_targeted_nn_roots,
    target_backbone,
)
from pruna.engine.model_checks import is_causal_lm, is_janus_llamagen_ar, is_transformers_pipeline_with_causal_lm
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.utils import move_to_device, safe_memory_cleanup
from pruna.logging.filter import SuppressOutput
from pruna.logging.logger import pruna_logger


class HQQ(PrunaAlgorithmBase):
    """
    Implement HQQ using huggingface transformers and the HQQ package.

    Half-Quadratic Quantization (HQQ) leverages fast, robust optimization techniques for on-the-fly quantization,
    eliminating the need for calibration data.
    """

    algorithm_name: str = "hqq"
    group_tags: list[str] = [tags.QUANTIZER]
    references: dict[str, str] = {
        "GitHub": "https://github.com/mobiusml/hqq",
        "Article": "https://mobiusml.github.io/hqq_blog/",
    }
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.hqq
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda"]
    dataset_required: bool = False
    compatible_before: Iterable[str] = ["torch_structured"]
    compatible_after: Iterable[str] = ["torch_compile", "sage_attn"]

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
            Constant("backend", value="torchao_int4"),
            CategoricalHyperparameter(
                "compute_dtype",
                choices=["torch.bfloat16", "torch.float16"],
                default_value="torch.float16",
                meta=dict(desc="Compute dtype for quantization."),
            ),
            Boolean(
                "use_torchao_kernels",
                default=True,
                meta=dict(desc="Whether to use the torchaoint4 kernels for inference."),
            ),
            Boolean(
                "force_hf_implementation",
                default=False,
                meta=dict(desc="Whether or not to bypass the HQQ quantization and use the generic HF quantization."),
            ),
            TargetModules(
                "target_modules",
                default_value=None,
                meta=dict(
                    desc="Precise choices of which modules to quantize, "
                    "e.g. {include: ['model.*']} to quantize the whole language model in a pipeline. "
                    f"See the {TargetModules.documentation_name_with_link} documentation for more details."
                ),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a causal language model.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a causal language model or a Janus LlamaGen AR model, False otherwise.
        """
        return is_causal_lm(model) or is_janus_llamagen_ar(model) or is_transformers_pipeline_with_causal_lm(model)

    def get_model_dependent_hyperparameter_defaults(
        self, model: Any, smash_config: SmashConfigPrefixWrapper
    ) -> dict[str, Any]:
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
        dict[str, Any]
            A dictionary containing the default target_modules for the algorithm.
        """
        target_modules: TARGET_MODULES_TYPE = target_backbone(model)
        return {"target_modules": target_modules}

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
        imported_modules = self.import_algorithm_packages()

        weight_quantization_bits = smash_config["weight_bits"]
        group_size = smash_config["group_size"]
        compute_dtype = torch.float16 if smash_config["compute_dtype"] == "torch.float16" else torch.bfloat16

        quant_config_hqq = imported_modules["BaseQuantizeConfig"](nbits=weight_quantization_bits, group_size=group_size)
        move_to_device(model, "cpu")
        safe_memory_cleanup()

        target_modules: None | TARGET_MODULES_TYPE = smash_config["target_modules"]
        if target_modules is None:
            defaults = self.get_model_dependent_hyperparameter_defaults(model, smash_config)
            target_modules = cast(TARGET_MODULES_TYPE, defaults["target_modules"])
        self.verify_target_modules(model, target_modules)

        def quantize_component(attr_name: str | None, module: torch.nn.Module, subpaths: list[str]) -> torch.nn.Module:
            """
            Quantize the model itself if it's a transformer, or its language model if it's a pipeline or Janus model.

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
            if is_janus_llamagen_ar(module):
                # dispatch to the language model because HQQ expects a LLM as input
                lm_path = "model.language_model"
                supported_targets = [
                    path
                    for path in subpaths
                    # make sure we're recognizing the full module name
                    if (path == lm_path or path.startswith(f"{lm_path}."))
                ]
                if len(supported_targets) != len(subpaths):
                    pruna_logger.warning(
                        f"HQQ on Janus is only supported for the language model `{lm_path}`, skipping other submodules."
                    )

                # relative path from the language model: remove lm_path and the dot
                lm_subpaths = {path[len(lm_path) + 1 :] for path in supported_targets}
                submodule = module.get_submodule(lm_path)
                lm_attr_name = lm_path if attr_name is None else f"{attr_name}.{lm_path}"
                submodule = quantize_component(lm_attr_name, submodule, lm_subpaths)

                module.set_submodule(lm_path, submodule)
                return module

            skipped_leaf_modules = get_skipped_submodules(module, subpaths, filter_fn=is_leaf_module)

            try:  # Try to quantize the model using HQQ
                if smash_config["force_hf_implementation"]:
                    raise Exception(
                        "AutoHQQHFModel is bypassed, defaulting to generic HF quantization. "
                        "Set force_hf_implementation to False to (try to) use AutoHQQHFModel."
                    )
                auto_targeted_hqq_hf_model = construct_base_class(
                    imported_modules, extra_ignore_modules=skipped_leaf_modules
                )
                module = auto_targeted_hqq_hf_model.quantize_model(
                    module,
                    quant_config=quant_config_hqq,
                    device=smash_config["device"],
                    compute_dtype=compute_dtype,
                )

                # skipped layers are not casted to device and compute dtype so we need to do it manually
                for name, submodule in module.named_modules():
                    if name in skipped_leaf_modules:
                        submodule.to(smash_config["device"])
                        submodule.to(compute_dtype)
            except Exception as e:  # Default to generic HF quantization if it fails or if default_to_hf is True
                if not smash_config["force_hf_implementation"]:
                    pruna_logger.debug(f"HQQ pipeline error: {e}")
                    pruna_logger.info(
                        f"Could not quantize model{'' if attr_name is None else f'.{attr_name}'} "
                        "using specialized HQQ pipeline, trying implementation from transformers library... "
                        "See debug logs for more details."
                    )

                # define config with skipped layers
                quant_config_hf = imported_modules["HqqConfig"](
                    nbits=weight_quantization_bits, group_size=group_size, skip_modules=skipped_leaf_modules
                )

                # Create a temporary directory in a specific location
                base_temp_dir = smash_config["cache_dir"]
                temp_dir = tempfile.mkdtemp(dir=base_temp_dir)
                module.save_pretrained(temp_dir)

                module = AutoModelForCausalLM.from_pretrained(
                    temp_dir,
                    quantization_config=quant_config_hf,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16 if smash_config["compute_dtype"] == "torch.float16" else torch.bfloat16,
                )

                # Delete the temporary directory and its contents
                shutil.rmtree(temp_dir)

            # Prepare the model for fast inference
            try:
                if weight_quantization_bits == 4 and smash_config["use_torchao_kernels"]:
                    pruna_logger.info(
                        "Patching model for fast inference with torchaoint4 kernels. "
                        "This operation can make the model incompatible with re-load. "
                        "If you plan to save and re-load the model, set use_torchao_kernels to False."
                    )
                    imported_modules["prepare_for_inference"](module, backend=smash_config["backend"])
            except Exception as e:
                pruna_logger.error(f"Error: {e}")
                pass

            return module

        smashed_model = map_targeted_nn_roots(quantize_component, model, target_modules)
        # as we have moved the model to cpu for cleaning, but only one of its attribute was put back on cuda.
        move_to_device(smashed_model, smash_config["device"])
        return smashed_model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        with SuppressOutput():
            from hqq.core.quantize import BaseQuantizeConfig
            from hqq.engine.hf import HQQModelForCausalLM
            from hqq.models.hf.base import AutoHQQHFModel
            from hqq.utils.patching import prepare_for_inference
            from transformers import (
                HqqConfig,  # we do isolate this because this statement will import HQQ (transformers' lazy import)
            )

        return dict(
            BaseQuantizeConfig=BaseQuantizeConfig,
            AutoHQQHFModel=AutoHQQHFModel,
            prepare_for_inference=prepare_for_inference,
            HqqConfig=HqqConfig,
            HQQModelForCausalLM=HQQModelForCausalLM,
        )

    def verify_target_modules(self, model: Any, target_modules: TARGET_MODULES_TYPE) -> None:
        """
        Warn the user if non-saveable modules are targeted.

        Parameters
        ----------
        model : Any
            The model to verify the target modules of.
        target_modules : TARGET_MODULES_TYPE
            The target modules to verify.
        """
        if isinstance(model, Pipeline):
            saveable = "model"
        elif is_janus_llamagen_ar(model):
            saveable = "model.language_model"
        else:
            return

        targeted_paths = expand_list_of_targeted_paths(target_modules, model)
        non_saveable_attributes = [
            path.split(".")[0] for path in targeted_paths if not (path == saveable or path.startswith(f"{saveable}."))
        ]
        if non_saveable_attributes:
            pruna_logger.warning(
                f"HQQ saving/loading is not implemented for the following attributes: {non_saveable_attributes}"
            )


def construct_base_class(imported_modules: Dict[str, Any], extra_ignore_modules: list[str]) -> Type[Any]:
    """
    Construct and return the AutoTargetedHQQHFModel class.

    AutoTargetedHQQHFModel is a subclass of AutoHQQHFModel that extends the set of layers ignored by HQQ quantization.

    Parameters
    ----------
    imported_modules : Dict[str, Any]
        Dictionary containing imported modules needed for the base class construction.
    extra_ignore_modules : None | list[str], optional
        The paths to the modules to ignore for quantization in addition to the ones already ignored by BaseHQQModel.

    Returns
    -------
    Type[AutoTargetedHQQHFModel]
        The constructed AutoTargetedHQQHFModel class.
    """

    class AutoTargetedHQQHFModel(imported_modules["AutoHQQHFModel"]):
        """Base class for HQQ Hugging Face models with targeted quantization."""

        @classmethod
        def get_ignore_layers(cls, model) -> list[str]:
            """
            Get the layers which should be ignored for quantization.

            This method extends the set of layers ignored by AutoHQQHFModel by adding the layers
            specified in extra_ignore_modules.

            Parameters
            ----------
            model : Any
                The model to get the ignore layers from.

            Returns
            -------
            list
                The layers which should be ignored for quantization.
            """
            ignore_layers = super().get_ignore_layers(model)  # we only add new layers to ignore
            return list(set(ignore_layers + extra_ignore_modules))  # avoid duplicates

    return AutoTargetedHQQHFModel
