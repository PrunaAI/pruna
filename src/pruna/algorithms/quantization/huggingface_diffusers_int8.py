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

import tempfile
from typing import Any, Dict, cast

import diffusers
import torch.nn as nn
from ConfigSpace import CategoricalHyperparameter, Constant, OrdinalHyperparameter
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfig, SmashConfigPrefixWrapper
from pruna.config.target_modules import TARGET_MODULES_TYPE, TargetModules, get_skipped_submodules, map_targeted_nn_roots
from pruna.engine.model_checks import (
    get_diffusers_transformer_models,
    get_diffusers_unet_models,
)
from pruna.engine.utils import determine_dtype, get_device_map, move_to_device


class DiffusersInt8Quantizer(PrunaQuantizer):
    """
    Implement Int8 quantization for Image-Gen models.

    BitsAndBytes offers a simple method to quantize models to 8-bit or 4-bit precision.
    The 8-bit mode blends outlier fp16 values with int8 non-outliers to mitigate performance degradation,
    while 4-bit quantization further compresses the model and is often used with QLoRA for fine-tuning.
    This algorithm is specifically adapted for diffusers models.
    """

    algorithm_name: str = "diffusers_int8"
    references: dict[str, str] = {"GitHub": "https://github.com/bitsandbytes-foundation/bitsandbytes"}
    tokenizer_required: bool = False
    processor_required: bool = False
    dataset_required: bool = False
    runs_on: list[str] = ["cuda", "accelerate"]
    compatible_algorithms: dict[str, list[str]] = dict(
        factorizer=["qkv_diffusers"], cacher=["deepcache", "fastercache", "fora", "pab"], compiler=["torch_compile"]
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
            OrdinalHyperparameter(
                "weight_bits",
                sequence=[4, 8],
                default_value=8,
                meta=dict(desc="Number of bits to use for quantization."),
            ),
            Boolean("double_quant", meta=dict(desc="Whether to enable double quantization.")),
            Boolean("enable_fp32_cpu_offload", meta=dict(desc="Whether to enable fp32 cpu offload.")),
            Constant("has_fp16_weight", value=False),
            Constant("compute_dtype", value="bfloat16"),
            Constant("threshold", value=6.0),
            CategoricalHyperparameter(
                "quant_type",
                choices=["fp4", "nf4"],
                default_value="fp4",
                meta=dict(desc="Quantization type to use."),
            ),
            TargetModules(
                name="target_modules",
                default_value=None,
                meta=dict(desc="Precise choices of which modules to quantize."),
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

        if isinstance(model, tuple(transformer_and_unet_models)):
            return True

        if hasattr(model, "transformer") and isinstance(model.transformer, tuple(transformer_and_unet_models)):
            return True

        return hasattr(model, "unet") and isinstance(model.unet, tuple(transformer_and_unet_models))

    def get_unconstrained_hyperparameter_defaults(
        self, model: Any, smash_config: SmashConfig | SmashConfigPrefixWrapper
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
        prefix: str
        if hasattr(model, "transformer"):
            prefix = "transformer."
        elif hasattr(model, "unet"):
            prefix = "unet."
        else:
            prefix = ""
        return {"include": [prefix + "*"], "exclude": [prefix + "lm_head"]}

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
        target_modules = smash_config["target_modules"]
        if target_modules is None:
            target_modules = self.get_unconstrained_hyperparameter_defaults(model, smash_config)
        target_modules = cast(TARGET_MODULES_TYPE, target_modules)

        def quantize_latent_model(attr_name: str | None, latent_model: nn.Module, subpaths: list[str]) -> Any:
            """
            Quantize a working model with bitsandbytes.

            Parameters
            ----------
            attr_name : str | None
                The name of the attribute in the model pointing to the working model to quantize.
            latent_model : torch.nn.Module
                The latent model to quantize.
            subpaths : list[str]
                The subpaths of the working model to quantize.
            """
            skipped_modules = get_skipped_submodules(latent_model, subpaths, leaf_modules=True)

            with tempfile.TemporaryDirectory(prefix=str(smash_config["cache_dir"])) as temp_dir:
                # Only the full model contains the device map, so we get it using the attribute name
                # attr_name can be None, then get_device_map defaults to the whole model, which is the expected behavior
                device_map = get_device_map(model, subset_key=attr_name)

                # save the latent model (to be quantized) in a temp directory
                move_to_device(latent_model, "cpu")
                latent_model.save_pretrained(temp_dir)
                latent_class = getattr(diffusers, type(latent_model).__name__)
                compute_dtype = determine_dtype(latent_model)

                bnb_config = DiffusersBitsAndBytesConfig(
                    load_in_8bit=smash_config["weight_bits"] == 8,
                    load_in_4bit=smash_config["weight_bits"] == 4,
                    llm_int8_threshold=float(smash_config["threshold"]),
                    llm_int8_skip_modules=skipped_modules,
                    llm_int8_enable_fp32_cpu_offload=smash_config["enable_fp32_cpu_offload"],
                    llm_int8_has_fp16_weight=smash_config["has_fp16_weight"],
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=smash_config["quant_type"],
                    bnb_4bit_use_double_quant=smash_config["double_quant"],
                )

                # re-load the latent model (with the quantization config)
                quantized_latent_model = latent_class.from_pretrained(
                    temp_dir,
                    quantization_config=bnb_config,
                    torch_dtype=compute_dtype,
                    device_map=device_map,
                )
            return quantized_latent_model

        quantized_model = map_targeted_nn_roots(quantize_latent_model, model, target_modules, leaf_modules=True)
        return quantized_model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        return dict()
