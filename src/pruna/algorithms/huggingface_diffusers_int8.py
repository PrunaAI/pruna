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
from collections.abc import Iterable
from typing import Any, cast

import diffusers
import torch.nn as nn
from ConfigSpace import CategoricalHyperparameter, Constant, OrdinalHyperparameter
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfig, SmashConfigPrefixWrapper
from pruna.config.target_modules import (
    TARGET_MODULES_TYPE,
    TargetModules,
    get_skipped_submodules,
    is_leaf_module,
    map_targeted_nn_roots,
    target_backbone,
)
from pruna.engine.model_checks import (
    get_diffusers_transformer_models,
    get_diffusers_unet_models,
)
from pruna.engine.utils import determine_dtype, get_device_map, move_to_device
from pruna.logging.logger import pruna_logger


class DiffusersInt8(PrunaAlgorithmBase):
    """
    Implement Int8 quantization for Image-Gen models.

    BitsAndBytes offers a simple method to quantize models to 8-bit or 4-bit precision.
    The 8-bit mode blends outlier fp16 values with int8 non-outliers to mitigate performance degradation,
    while 4-bit quantization further compresses the model and is often used with QLoRA for fine-tuning.
    This algorithm is specifically adapted for diffusers models.
    """

    algorithm_name: str = "diffusers_int8"
    group_tags: list[str] = [tags.QUANTIZER]
    references: dict[str, str] = {"GitHub": "https://github.com/bitsandbytes-foundation/bitsandbytes"}
    tokenizer_required: bool = False
    processor_required: bool = False
    dataset_required: bool = False
    runs_on: list[str] = ["cuda", "accelerate"]
    save_fn: None = None
    compatible_before: Iterable[str] = ["qkv_diffusers"]
    compatible_after: Iterable[str] = ["deepcache", "fastercache", "fora", "pab", "torch_compile", "sage_attn"]

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

        if isinstance(model, tuple(transformer_and_unet_models)):
            return True

        if hasattr(model, "transformer") and isinstance(model.transformer, tuple(transformer_and_unet_models)):
            return True

        return hasattr(model, "unet") and isinstance(model.unet, tuple(transformer_and_unet_models))

    def get_model_dependent_hyperparameter_defaults(
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
        return target_backbone(model)

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
            target_modules = self.get_model_dependent_hyperparameter_defaults(model, smash_config)
        target_modules = cast(TARGET_MODULES_TYPE, target_modules)

        def quantize_working_model(attr_name: str | None, working_model: nn.Module, subpaths: list[str]) -> Any:
            """
            Quantize a working model with bitsandbytes.

            Parameters
            ----------
            attr_name : str | None
                The name of the attribute in the model pointing to the working model to quantize.
            working_model : torch.nn.Module
                The working model to quantize, i.e. a nn.Module component of the model.
            subpaths : list[str]
                The subpaths of the working model to quantize.
            """
            if not hasattr(working_model, "save_pretrained") or not callable(working_model.save_pretrained):
                raise ValueError(
                    "diffusers-int8 was applied to a module which didn't have a callable save_pretrained method."
                )

            # only include leaf modules since the bnb quantizer skips all submodules
            # within a skipped module. Only Linear and Conv1d layers can be quantized anyway.
            skipped_modules = get_skipped_submodules(working_model, subpaths, filter_fn=is_leaf_module)
            pruna_logger.debug(
                f"Skipping {self.algorithm_name} quantization for the following "
                f"leaf modules within {attr_name or 'the model'} : {skipped_modules}"
            )

            with tempfile.TemporaryDirectory(prefix=str(smash_config["cache_dir"])) as temp_dir:
                # Only the full model contains the device map, so we get it using the attribute name
                # attr_name can be None, then get_device_map defaults to the whole model, which is the expected behavior
                device_map = get_device_map(model, subset_key=attr_name)

                # save the latent model (to be quantized) in a temp directory
                move_to_device(working_model, "cpu")
                working_model.save_pretrained(temp_dir)
                working_class = getattr(diffusers, type(working_model).__name__)
                compute_dtype = determine_dtype(working_model)

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
                quantized_working_model = working_class.from_pretrained(
                    temp_dir,
                    quantization_config=bnb_config,
                    torch_dtype=compute_dtype,
                    device_map=device_map,
                )
            return quantized_working_model

        quantized_model = map_targeted_nn_roots(quantize_working_model, model, target_modules)
        return quantized_model
