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

"""Quantization using the `llmcompressor` library."""

from typing import Any, Dict

from ConfigSpace import Constant

from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.data.utils import recover_text_from_dataloader
from pruna.engine.model_checks import is_causal_lm


class LLMCompressorQuantizer(PrunaQuantizer):
    """Quantize causal language models with `llmcompressor`."""

    algorithm_name = "llm_compressor"
    references = {"GitHub": "https://github.com/vllm-project/llm-compressor"}
    tokenizer_required = True
    processor_required = False
    run_on_cpu = False
    run_on_cuda = True
    dataset_required = True
    compatible_algorithms: Dict[str, list[str]] = dict()
    # save_fn remains ``None`` as `llmcompressor` patches ``save_pretrained``
    required_install = "``pip install llmcompressor``"

    def get_hyperparameters(self) -> list:
        """Return the hyperparameters used for quantization."""
        return [Constant("scheme", value="W4A16_ASYM")]

    def model_check_fn(self, model: Any) -> bool:
        """Check that the model is a causal language model."""
        return is_causal_lm(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """Apply AWQ quantization using ``llmcompressor``."""
        imported = self.import_algorithm_packages()

        recipe = [
            imported["AWQModifier"](
                ignore=["lm_head"],
                scheme=smash_config["scheme"],
                targets=["Linear"],
            )
        ]

        dataloader = smash_config.val_dataloader()
        tokenizer = smash_config.tokenizer
        calib_data = recover_text_from_dataloader(dataloader, tokenizer)

        imported["oneshot"](model=model, recipe=recipe, calib_data=calib_data)
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """Import ``llmcompressor`` packages lazily."""
        try:
            from llmcompressor import oneshot
            from llmcompressor.modifiers.awq import AWQModifier
        except Exception as e:  # pragma: no cover - dependency missing
            raise ImportError(
                "llmcompressor is not installed. Please install it using `pip install llmcompressor`."
            ) from e
        return {"oneshot": oneshot, "AWQModifier": AWQModifier}
