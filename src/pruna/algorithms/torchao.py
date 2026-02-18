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

import importlib
from collections.abc import Iterable
from functools import partial
from typing import Any, Dict, cast

import torch
from ConfigSpace import CategoricalHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.smash_config import SmashConfig, SmashConfigPrefixWrapper
from pruna.config.target_modules import TARGET_MODULES_TYPE, TargetModules, map_targeted_nn_roots, target_backbone
from pruna.engine.model_checks import (
    get_diffusers_transformer_models,
    get_diffusers_unet_models,
    is_causal_lm,
    is_transformers_pipeline_with_causal_lm,
)
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger
from pruna.logging.utils import suppress_logging

# Based on common diffusers transformer architectures
NORM_MODULES: list[str] = [
    "adaln_single",
    "caption_norm",
    "norm",
    "norm1",
    "norm1_context",
    "norm2",
    "norm2_context",
    "norm3",
    "norm_k",
    "norm_q",
    "norm_v",
    "norm_final",
    "norm_out",
    "skip_norm",
]
EMBEDDING_MODULES: list[str] = [
    "caption_projection",
    "condition_embedder",
    "context_embedder",
    "guidance_condition_pro",
    "guidance_embedder",
    "ofs_embedding",
    "ofs_proj",
    "patch_embed",
    "pos_embed",
    "rope",
    "time_embedding",
    "time_proj",
    "time_text_embed",
    "timestep_embedder",
    "text_embedder",
    "x_embedder",
]


class Torchao(PrunaAlgorithmBase):
    """
    Implement quantization using torchao.

    This replaces each nn.Linear in-place with a low-precision Tensor subclass via
    ``torchao.quantization.quantize``. It uses per-channel uniform affine
    ("linear") quantization for weights (e.g. symmetric int8 or int4) and dynamic
    per-tensor affine quantization for activations (8-bit at runtime). When combined
    with torch.compile, this can yield substantial inference speedups over
    full-precision model.
    """

    algorithm_name: str = "torchao"
    group_tags: list[str] = [tags.QUANTIZER]
    references: dict[str, str] = {"GitHub": "https://huggingface.co/docs/diffusers/quantization/torchao"}
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.save_before_apply
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda", "accelerate"]
    dataset_required: bool = False
    compatible_before: Iterable[str] = ["qkv_diffusers", "torch_structured"]
    compatible_after: Iterable[str] = ["flash_attn3", "fora", "torch_compile", "sage_attn"]
    disjointly_compatible_before: Iterable[str] = ["hqq", "hqq_diffusers"]
    disjointly_compatible_after: Iterable[str] = []

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            CategoricalHyperparameter(
                "quant_type",
                choices=["int4dq", "int4wo", "int8dq", "int8wo", "fp8wo", "fp8dq", "fp8dqrow"],
                default_value="int8dq",
                meta=dict(
                    desc=(
                        "Quantization type: prefix selects data format (int4/int8/fp8); "
                        "`wo` quantizes only the weights (activations remain in full precision); "
                        "`dq` fully quantizes and dequantizes both weights and activations; "
                        "`dqrow` also does full quantize-dequantize but computes a separate scale for each row"
                    )
                ),
            ),
            CategoricalHyperparameter(
                "excluded_modules",
                choices=["none", "norm", "embedding", "norm+embedding"],
                default_value="none",
                meta=dict(desc="Which types of modules to omit when applying quantization."),
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
        Check if the model is a torch.nn.Module or a diffusers pipeline with a transformer model.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is suitable for torchao quantization, False otherwise.
        """
        pruna_logger.warning(
            "torchao has strict version compatibility requirements with torch. "
            "If you encounter crashes when using torchao, ensure that your torch and torchao "
            "versions are compatible, as documented in the torchao compatibility table: "
            "https://github.com/pytorch/ao/issues/2919#issue-3375688762"
        )

        transformer_models = get_diffusers_transformer_models()
        unet_models = get_diffusers_unet_models()
        if isinstance(model, tuple(transformer_models)):
            return True
        if isinstance(model, tuple(unet_models)):
            return True
        if hasattr(model, "unet") and isinstance(model.unet, tuple(unet_models)):
            return True
        if hasattr(model, "transformer") and isinstance(model.transformer, tuple(transformer_models)):
            return True
        if is_causal_lm(model) or is_transformers_pipeline_with_causal_lm(model):
            return True
        return isinstance(model, torch.nn.Module)

    def get_model_dependent_hyperparameter_defaults(
        self, model: Any, smash_config: SmashConfig | SmashConfigPrefixWrapper
    ) -> TARGET_MODULES_TYPE:  # ty: ignore[invalid-method-override]
        """
        Provide default `target_modules` using `target_backbone`, with additional exclusions.

        Extends the base backbone targets by optionally excluding norm and embedding modules
        based on the `excluded_modules` hyperparameter in `smash_config`.

        Parameters
        ----------
        model : Any
            The model to derive defaults from.
        smash_config : SmashConfigPrefixWrapper
            The algorithm-prefixed configuration. Used to read `excluded_modules`
            to determine which module types to exclude.

        Returns
        -------
        dict[str, Any]
            A dictionary with a `target_modules` key mapping to include/exclude patterns.
        """
        target_modules: TARGET_MODULES_TYPE = target_backbone(model)

        # exclude norm and embedding modules based on the excluded modules hyperparameter
        if "norm" in smash_config["excluded_modules"]:
            for norm_module in NORM_MODULES:
                target_modules["exclude"].extend(_get_patterns_for_exact_attribute_match(norm_module))
        if "embedding" in smash_config["excluded_modules"]:
            for embedding_module in EMBEDDING_MODULES:
                target_modules["exclude"].extend(_get_patterns_for_exact_attribute_match(embedding_module))
        return {"target_modules": target_modules}

    def _validate_config(self, smash_config: SmashConfigPrefixWrapper) -> None:
        """
        Validate the configuration for torchao quantization and throw warnings for invalid configurations.

        Parameters
        ----------
        smash_config : SmashConfigPrefixWrapper
            The configuration for the quantization.
        """
        if (
            smash_config["torch_compile"]
            and smash_config._base_config["torch_compile_mode"] != "max-autotune-no-cudagraphs"
        ):
            pruna_logger.warning(
                "You are using torchao with torch.compile. "
                "Please set `smash_config['torch_compile_mode']='max-autotune-no-cudagraphs'` for best results; "
                "otherwise you may encounter undesirable outcomes."
            )

        if "fp8" in smash_config["quant_type"] and not (
            torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
        ):
            pruna_logger.warning(
                "Float8 quantization requires an NVIDIA GPU with compute capability ≥ 8.9. "
                "Your device does not meet this requirement."
            )

        if smash_config["quant_type"] == "fp8dqrow":
            pruna_logger.warning(
                "Row wise float8 dynamic quantization is still experimental and might not work on your hardware."
            )

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Quantize the model with torchao.

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
        self._validate_config(smash_config)

        # Suppress torchao INFO messages (e.g., about skipping small layers) during quantization
        target_modules: None | TARGET_MODULES_TYPE = smash_config["target_modules"]
        if target_modules is None:
            defaults = self.get_model_dependent_hyperparameter_defaults(model, smash_config)
            target_modules = cast(TARGET_MODULES_TYPE, defaults["target_modules"])

        with suppress_logging("torchao.quantization.quant_api"):
            imported_modules = self.import_algorithm_packages()
            is_linear = imported_modules["_is_linear"]

            def quantize_nn_module(
                attr_name: str | None, module: torch.nn.Module, subpaths: list[str]
            ) -> torch.nn.Module:
                """
                Quantize a nn.Module.

                Parameters
                ----------
                attr_name : str | None
                    The name of the attribute in the model pointing to the nn.Module to quantize.
                module : torch.nn.Module
                    The nn.Module to quantize.
                subpaths : list[str]
                    The subpaths of the module to quantize.

                Returns
                -------
                torch.nn.Module
                    The quantized nn.Module.
                """

                def filter_linear_and_targeted_fn(submodule: torch.nn.Module, subpath: str, prefix: str = None) -> bool:
                    true_subpath = subpath if prefix is None else f"{prefix}.{subpath}"
                    is_lin = is_linear(submodule, subpath)
                    return is_lin and true_subpath in subpaths

                # Only apply quantization on module list level if torch compile is also applied at that level
                if smash_config["torch_compile"] and smash_config._base_config["torch_compile_target"] == "module_list":
                    # Apply quantization to individual submodules in ModuleLists
                    for module_list_name, module_list in module.named_modules():
                        if isinstance(module_list, torch.nn.ModuleList):
                            for i, submodule in enumerate(module_list):
                                imported_modules["quantize"](
                                    submodule,
                                    imported_modules[smash_config["quant_type"]],
                                    filter_fn=partial(filter_linear_and_targeted_fn, prefix=f"{module_list_name}.{i}"),
                                )
                else:
                    # Apply quantization to the entire model
                    imported_modules["quantize"](
                        module, imported_modules[smash_config["quant_type"]], filter_fn=filter_linear_and_targeted_fn
                    )
                return module

            model = map_targeted_nn_roots(quantize_nn_module, model, target_modules)
            return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the packages needed for torchao quantization.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        from torchao.quantization import (
            float8_dynamic_activation_float8_weight,
            float8_weight_only,
            int4_weight_only,
            int8_dynamic_activation_int4_weight,
            int8_dynamic_activation_int8_weight,
            int8_weight_only,
            quantize_,
        )
        from torchao.quantization.quant_api import PerRow

        _is_linear = importlib.import_module("torchao.quantization.quant_api")._is_linear
        return dict(
            quantize=quantize_,
            int4dq=int8_dynamic_activation_int4_weight(),
            int4wo=int4_weight_only(),
            int8dq=int8_dynamic_activation_int8_weight(),
            int8wo=int8_weight_only(),
            fp8wo=float8_weight_only(),
            fp8dq=float8_dynamic_activation_float8_weight(),
            fp8dqrow=float8_dynamic_activation_float8_weight(
                activation_dtype=torch.float8_e4m3fn,
                weight_dtype=torch.float8_e4m3fn,
                granularity=PerRow(),
            ),
            _is_linear=_is_linear,
        )


def _get_patterns_for_exact_attribute_match(attribute_name: str) -> list[str]:
    """
    Get the patterns for an exact attribute match in a model.

    Parameters
    ----------
    attribute_name : str
        The name of the attribute to match.

    Returns
    -------
    list[str]
        Patterns for detecting paths containing the attribute name between dots.
    """
    # we want to return "*.{attribute_name}.*" but that would miss paths which start or finish with the name
    return [
        attribute_name,  # exact match from the root
        f"*.{attribute_name}",  # match at the end
        f"{attribute_name}.*",  # match at the start
        f"*.{attribute_name}.*",  # match in the middle
    ]
