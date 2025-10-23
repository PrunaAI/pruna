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

import shutil
import tempfile
from collections.abc import Iterable
from typing import Any, Dict

import torch
from ConfigSpace import CategoricalHyperparameter, Constant, OrdinalHyperparameter
from transformers import AutoModelForCausalLM

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_causal_lm, is_janus_llamagen_ar, is_transformers_pipeline_with_causal_lm
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.utils import ModelContext, move_to_device, safe_memory_cleanup
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
    compatible_after: Iterable[str] = ["torch_compile"]

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
        if is_transformers_pipeline_with_causal_lm(model):
            return self._apply_to_model_within_transformers_pipeline(model, smash_config)

        imported_modules = self.import_algorithm_packages()

        weight_quantization_bits = smash_config["weight_bits"]
        group_size = smash_config["group_size"]

        quant_config_hqq = imported_modules["BaseQuantizeConfig"](nbits=weight_quantization_bits, group_size=group_size)
        quant_config_hf = imported_modules["HqqConfig"](nbits=weight_quantization_bits, group_size=group_size)
        move_to_device(model, "cpu")
        safe_memory_cleanup()
        with ModelContext(model) as (mc, working_model):
            try:  # Try to quantize the model using HQQ
                if smash_config["force_hf_implementation"]:
                    raise Exception(
                        "AutoHQQHFModel is bypassed, defaulting to generic HF quantization. "
                        "Set force_hf_implementation to False to (try to) use AutoHQQHFModel."
                    )
                working_model = imported_modules["AutoHQQHFModel"].quantize_model(
                    working_model,
                    quant_config=quant_config_hqq,
                    device=smash_config["device"],
                    compute_dtype=torch.float16 if smash_config["compute_dtype"] == "torch.float16" else torch.bfloat16,
                )
            except Exception:  # Default to generic HF quantization if it fails or if default_to_hf is True
                if not smash_config["force_hf_implementation"]:
                    pruna_logger.info(
                        "Could not quantize model using specialized HQQ pipeline, "
                        "trying implementation from transformers library..."
                    )
                # Create a temporary directory in a specific location
                base_temp_dir = smash_config["cache_dir"]
                temp_dir = tempfile.mkdtemp(dir=base_temp_dir)
                working_model.save_pretrained(temp_dir)

                working_model = AutoModelForCausalLM.from_pretrained(
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
                    imported_modules["prepare_for_inference"](working_model, backend=smash_config["backend"])
            except Exception as e:
                pruna_logger.error(f"Error: {e}")
                pass

            mc.update_working_model(working_model)

        smashed_model = mc.get_updated_model()
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
