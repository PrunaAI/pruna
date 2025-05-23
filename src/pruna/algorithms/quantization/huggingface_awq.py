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

import tempfile
from typing import Any, Callable, Dict

from ConfigSpace import Constant, OrdinalHyperparameter

from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.data.utils import recover_text_from_dataloader
from pruna.engine.model_checks import is_causal_lm
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.utils import safe_memory_cleanup


class AWQQuantizer(PrunaQuantizer):
    """
    Implement AWQ using huggingface transformers.

    Activation-aware Weight Quantization (AWQ) selectively quantizes model weights using a calibration dataset,
    preserving a small fraction that are important for maintaining performance in LLMs. This minimizes quantization loss,
    allowing models to operate at 4-bit precision without significantly sacrificing accuracy.
    """

    algorithm_name: str = "awq"
    references: dict[str, str] = {"GitHub": "https://github.com/casper-hansen/AutoAWQ"}
    save_fn: Callable = SAVE_FUNCTIONS.awq_quantized
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda"]
    dataset_required: bool = True
    compatible_algorithms: dict[str, list[str]] = dict()
    required_install: str = "``pip install pruna[autoawq]``"

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            Constant("weight_bits", value=4),
            Constant("zero_point", value=True),
            OrdinalHyperparameter(
                "group_size",
                sequence=[8, 16, 32, 64, 128],
                default_value=128,
                meta=dict(desc="Group size for quantization."),
            ),
            Constant("version", value="gemm"),
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
            True if the model is a causal language model, False otherwise.
        """
        return is_causal_lm(model)

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

        # Create a temporary directory to save the model
        with tempfile.TemporaryDirectory(prefix=smash_config["cache_dir"]) as temp_dir:
            # cast original model to CPU to free memory for smashed model
            if hasattr(model, "to"):
                model.to("cpu")
                safe_memory_cleanup()
            model.save_pretrained(temp_dir)

            awq_config = {
                "quant_method": "awq",
                "w_bit": smash_config["weight_bits"],
                "q_group_size": smash_config["group_size"],
                "zero_point": smash_config["zero_point"],
                "version": smash_config["version"],
            }

            # Use "auto" instead of our standard "sequential" here as AWQ requires variable amount of memory on GPU:0
            smashed_model = imported_modules["AutoAWQForCausalLM"].from_pretrained(
                temp_dir, **{"low_cpu_mem_usage": True, "use_cache": False}, device_map="auto"
            )

            # presence of tokenizer and dataloader is ensured beforehand
            dataloader = smash_config.val_dataloader()
            tokenizer = smash_config.tokenizer
            calib_data = recover_text_from_dataloader(dataloader, tokenizer)  # type: ignore[arg-type]
            smashed_model.quantize(
                tokenizer=smash_config.tokenizer,  # type: ignore[arg-type]
                calib_data=calib_data,
                quant_config=awq_config,
            )

        return smashed_model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError(f"AWQ is not installed. Please install it using {self.required_install}.")

        return dict(AutoAWQForCausalLM=AutoAWQForCausalLM)
