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
from typing import Any, Dict

import torch
from ConfigSpace import CategoricalHyperparameter, Constant, OrdinalHyperparameter
from transformers import AutoModelForCausalLM

from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_causal_lm, is_janus_llamagen_ar
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.utils import ModelContext, move_to_device, safe_memory_cleanup
from pruna.logging.filter import SuppressOutput
from pruna.logging.logger import pruna_logger


class HQQQuantizer(PrunaQuantizer):
    """
    Implement HQQ using huggingface transformers and the HQQ package.

    Half-Quadratic Quantization (HQQ) leverages fast, robust optimization techniques for on-the-fly quantization,
    eliminating the need for calibration data.
    """

    algorithm_name: str = "hqq"
    references: dict[str, str] = {
        "GitHub": "https://github.com/mobiusml/hqq",
        "Article": "https://mobiusml.github.io/hqq_blog/",
    }
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.hqq
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda"]
    dataset_required: bool = False
    compatible_algorithms: dict[str, list[str]] = dict(compiler=["torch_compile"], pruner=["torch_structured"])

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
        return is_causal_lm(model) or is_janus_llamagen_ar(model)

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

        quant_config_hqq = imported_modules["BaseQuantizeConfig"](nbits=weight_quantization_bits, group_size=group_size)
        quant_config_hf = imported_modules["HqqConfig"](nbits=weight_quantization_bits, group_size=group_size)
        move_to_device(model, "cpu")
        safe_memory_cleanup()
        with ModelContext(model) as (pipeline, working_model, denoiser_type):
            try:  # Try to quantize the model using HQQ
                working_model = imported_modules["AutoHQQHFModel"].quantize_model(
                    working_model,
                    quant_config=quant_config_hqq,
                    device=smash_config["device"],
                    compute_dtype=torch.float16 if smash_config["compute_dtype"] == "torch.float16" else torch.bfloat16,
                )
            except Exception:  # Default to generic HF quantization if it fails
                pruna_logger.info("Could not quantize model using specialized HQQ pipeline, trying generic interface...")
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
                if weight_quantization_bits == 4:
                    imported_modules["prepare_for_inference"](working_model, backend=smash_config["backend"])
            except Exception as e:
                pruna_logger.error(f"Error: {e}")
                pass
            # redefining the working_model breaks links with context manager
            # so we need to re-define the working_model as an attribute of the model.
            pipeline.working_model = working_model
            # as we have moved the model to cpu for cleaning, but only one of its attribute was put back on cuda.
            move_to_device(model, smash_config["device"])
            return model

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
