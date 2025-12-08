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

from collections.abc import Iterable
from typing import Any

import torch
from diffusers import DiffusionPipeline

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.save import SAVE_FUNCTIONS


class SageAttn(PrunaAlgorithmBase):
    """
    Replace torch.nn.functional.scaled_dot_product_attention with sage_attn.

    SageAttention is a fast and memory-efficient attention mechanism. It applies the flash attention mechanism
    in combination with quantization and smoothing to speed up attention computations.
    """

    algorithm_name: str = "sage_attn"
    group_tags: list[str] = [tags.KERNEL]
    save_fn = SAVE_FUNCTIONS.reapply
    references: dict[str, str] = {
        "GitHub": "https://github.com/thu-ml/SageAttention",
        "Kernel Hub": "https://huggingface.co/kernels-community/sage_attention",
    }
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda", "accelerate"]
    dataset_required: bool = False
    compatible_before: Iterable[str] = []
    compatible_after: Iterable[str] = ["torch_compile"]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model has an attention mechanism that can be replaced with sage_attn.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a valid model for the algorithm, False otherwise.
        """
        if not isinstance(model, DiffusionPipeline) or not hasattr(model, "components"):
            return False

        return any(
            hasattr(component, "set_attention_backend") and component.dtype in [torch.bfloat16, torch.float16]
            for component in model.components.values()
        )

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Wrap the model to use SageAttention where possible.

        Parameters
        ----------
        model : Any
            The model to wrap.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the application of the algorithm.

        Returns
        -------
        Any
            The wrapped model.
        """
        # We simply apply the sage attention backend from diffusers
        # Furthermore, we use the sage attention kernel from the hub as the default sageattn function
        # is broken (at least at the moment)
        for component in model.components.values():
            if hasattr(component, "set_attention_backend") and component.dtype in [
                torch.bfloat16,
                torch.float16,
            ]:
                component.set_attention_backend("sage_hub")
        return model
