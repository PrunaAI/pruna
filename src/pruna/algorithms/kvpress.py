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

import functools
from collections.abc import Iterable
from typing import Any, Dict

from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.hyperparameters import UnconstrainedHyperparameter
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_causal_lm, is_transformers_pipeline_with_causal_lm
from pruna.engine.save import SAVE_FUNCTIONS

PRESS_TYPES = [
    "CompactorPress",
    "CURPress",
    "ExpectedAttentionPress",
    "ExpectedAttentionStatsPress",
    "FastKVzipPress",
    "FinchPress",
    "KnormPress",
    "KVzapPress",
    "KVzipPress",
    "KeyDiffPress",
    "LagKVPress",
    "LeverageScorePress",
    "NonCausalAttnPress",
    "ObservedAttentionPress",
    "PyramidKVPress",
    "QFilterPress",
    "RandomPress",
    "SnapKVPress",
    "StreamingLLMPress",
    "TOVAPress",
]


class KVPress(PrunaAlgorithmBase):
    """
    Integrate KVPress for KV cache compression in causal language models.

    KVPress is a library by NVIDIA that compresses the key-value cache during the prefill phase
    of autoregressive generation, reducing memory usage for long-context inference. It provides
    over 30 compression strategies (presses) that score and prune KV pairs based on different
    importance criteria.
    """

    algorithm_name: str = "kvpress"
    group_tags: list[tags] = [tags.KV_CACHER]
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.reapply
    references: dict[str, str] = {
        "GitHub": "https://github.com/NVIDIA/kvpress",
        "Article": "https://huggingface.co/blog/nvidia/kvpress",
    }
    required_install: str = "pip install pruna[kvpress]"
    tokenizer_required: bool = False
    processor_required: bool = False
    dataset_required: bool = False
    runs_on: list[str] = ["cuda"]
    compatible_before: Iterable[str] = [
        "awq", "gptq", "half", "hqq", "llm_int8",
        "quanto", "sage_attn", "torchao", "moe_kernel_tuner",
    ]
    compatible_after: Iterable[str] = ["torch_compile", "moe_kernel_tuner"]

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
                "press_type",
                choices=PRESS_TYPES,
                default_value="ExpectedAttentionPress",
                meta={"desc": "The KV cache compression strategy to use."},
            ),
            UniformFloatHyperparameter(
                "compression_ratio",
                lower=0.0,
                upper=1.0,
                default_value=0.5,
                meta={"desc": "Fraction of KV pairs to remove. 0.0 means no compression."},
            ),
            UnconstrainedHyperparameter(
                "press_kwargs",
                default_value=None,
                meta={"desc": "Additional keyword arguments passed to the press constructor."},
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
            True if the model is a causal language model, False otherwise.
        """
        return is_causal_lm(model) or is_transformers_pipeline_with_causal_lm(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Apply KV cache compression to the model.

        Parameters
        ----------
        model : Any
            The model to apply KV cache compression to.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the compression.

        Returns
        -------
        Any
            The model with KV cache compression applied to its generate method.
        """
        imported_modules = self.import_algorithm_packages()

        press_type = smash_config["press_type"]
        compression_ratio = smash_config["compression_ratio"]
        press_kwargs = smash_config["press_kwargs"] or {}

        press_cls = imported_modules[press_type]
        press = press_cls(compression_ratio=compression_ratio, **press_kwargs)

        original_generate = model.generate

        @functools.wraps(original_generate)
        def generate_with_press(*args, **kwargs):
            with press(model):
                return original_generate(*args, **kwargs)

        model.generate = generate_with_press
        model._kvpress_original_generate = original_generate
        model._kvpress_press = press

        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the kvpress package lazily.

        Returns
        -------
        Dict[str, Any]
            A dictionary mapping press class names to their classes.
        """
        import kvpress

        return {name: getattr(kvpress, name) for name in PRESS_TYPES}
