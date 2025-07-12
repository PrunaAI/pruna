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

import importlib
from typing import Any, Dict

import torch

from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.smash_space import CategoricalHyperparameter
from pruna.engine.model_checks import get_diffusers_transformer_models, get_diffusers_unet_models, is_causal_lm
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger

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


class TorchaoQuantizer(PrunaQuantizer):
    """
    Implement quantization using torchao.

    This replaces each nn.Linear in-place with a low-precision Tensor subclass via
    torchao.quantization.quantize_. It uses per-channel uniform affine
    (“linear”) quantization for weights (e.g. symmetric int8 or int4) and dynamic
    per-tensor affine quantization for activations (8-bit at runtime). When combined
    with torch.compile, this can yield substantial inference speedups over
    full-precision model.
    """

    algorithm_name: str = "torchao"
    references: dict[str, str] = {"GitHub": "https://huggingface.co/docs/diffusers/quantization/torchao"}
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.save_before_apply
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda", "accelerate"]
    dataset_required: bool = False
    compatible_algorithms: dict[str, list[str]] = dict(
        cacher=["fora"],
        compiler=["torch_compile"],
        factorizer=["qkv_diffusers"],
        pruner=["torch_structured"],
    )

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
                choices=[
                    "int4dq",
                    "int4wo",
                    "int8dq",
                    "int8wo",
                    "fp8wo",
                    "fp8dq",
                    "fp8dqrow",
                ],
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
                choices=[
                    "none",
                    "norm",
                    "embedding",
                    "norm+embedding",
                ],
                default_value="none",
                meta=dict(desc="Which types of modules to omit when applying quantization."),
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
        if is_causal_lm(model):
            return True
        return isinstance(model, torch.nn.Module)

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
        if hasattr(model, "unet"):
            working_model = model.unet
        elif hasattr(model, "transformer"):
            working_model = model.transformer
        else:
            working_model = model

        excluded_modules = []
        if "norm" in smash_config["excluded_modules"]:
            excluded_modules.extend(NORM_MODULES)
        if "embedding" in smash_config["excluded_modules"]:
            excluded_modules.extend(EMBEDDING_MODULES)

        imported_modules = self.import_algorithm_packages()
        is_linear = imported_modules["_is_linear"]

        def filter_fn(module: torch.nn.Module, fqn: str) -> bool:
            if not is_linear(module, fqn):
                return False
            return all(name not in excluded_modules for name in fqn.split("."))

        if (
            smash_config["compiler"] == "torch_compile"
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
        # Only apply quantization on module list level if torch compile is also applied at that level
        if (
            smash_config["compiler"] == "torch_compile"
            and smash_config._base_config["torch_compile_target"] == "module_list"
        ):
            # Apply quantization to the entire model
            imported_modules["quantize"](
                working_model, imported_modules[smash_config["quant_type"]], filter_fn=filter_fn
            )
        else:
            # Apply quantization to individual submodules in ModuleLists
            for name, module in working_model.named_modules():
                if isinstance(module, torch.nn.ModuleList):
                    for i, submodule in enumerate(module):
                        if isinstance(submodule, torch.nn.Module):
                            imported_modules["quantize"](
                                submodule, imported_modules[smash_config["quant_type"]], filter_fn=filter_fn
                            )
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

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
