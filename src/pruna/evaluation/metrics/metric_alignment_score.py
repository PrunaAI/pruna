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

"""Alignment Score metric using VLM for image-text alignment evaluation."""

from __future__ import annotations

from typing import Any, Literal

import torch

from pruna.evaluation.metrics.metric_vlm_base import _DoesThisImageShowPromptMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.utils import SINGLE
from pruna.evaluation.metrics.vlm_base import BaseVLM
from pruna.evaluation.metrics.vlm_utils import VQAnswer


@MetricRegistry.register("alignment_score")
class AlignmentScoreMetric(_DoesThisImageShowPromptMetric):
    """
    Binary image-text alignment score using a VLM Yes/No question.

    Asks ``'Does this image show "{prompt}"?'`` (same template as VQAScore, arXiv:2404.01291)
    and scores the answer as 1.0 (Yes) or 0.0 (No) via structured output.

    Unlike :class:`~pruna.evaluation.metrics.metric_vqa.VQAMetric`, which uses soft
    ``P(Yes)`` probabilities from logprobs (VQAScore-style), this metric applies structured
    generation to produce a binary score. Use :class:`VQAMetric` for paper-aligned VQAScore
    evaluation; use this metric when you prefer strict pass/fail alignment checks or when
    logprobs are not available from the VLM backend.

    Higher scores indicate better image-text alignment.

    Parameters
    ----------
    vlm : BaseVLM | None, optional
        Custom VLM instance. If provided, ``vlm_type`` and ``model_name`` are ignored.
    vlm_type : {"litellm", "transformers"}, optional
        VLM backend. Default is "litellm".
    model_name : str | None, optional
        Litellm model id or HuggingFace checkpoint id. **Required** when ``vlm`` is not
        provided (e.g. ``openai/gpt-4o``).
    vlm_kwargs : dict, optional
        Forwarded by ``get_vlm`` to ``LitellmVLM`` or ``TransformersVLM``. For local models,
        set ``model_load_kwargs`` for ``from_pretrained``; for litellm, pass extra API options.
    structured_output : bool, optional
        Use structured generation (litellm pydantic; transformers outlines when applicable).
        Default is True.
    device : str | torch.device | None, optional
        Device for transformers VLM.
    api_key : str | None, optional
        API key for litellm.
    call_type : str, optional
        Call type for the metric.
    **kwargs : Any
        Additional arguments.

    Notes
    -----
    For VQAScore-style soft P(Yes) scores, use :class:`~pruna.evaluation.metrics.metric_vqa.VQAMetric`.
    Hosted vs local setup: :doc:`Evaluate a model </docs_pruna/user_manual/evaluate>` (Vision-language judge metrics).
    """

    scores: list[float]
    default_call_type: str = "y_x"
    higher_is_better: bool = True
    metric_name: str = "alignment_score"
    runs_on: list[str] = ["cuda", "cpu"]

    def __init__(
        self,
        vlm: BaseVLM | None = None,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str | None = None,
        vlm_kwargs: dict | None = None,
        structured_output: bool = True,
        device: str | torch.device | None = None,
        api_key: str | None = None,
        call_type: str = SINGLE,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device)
        self.use_probability = False
        self.response_format = VQAnswer if structured_output else None
        self._init_vlm_scores(
            vlm=vlm,
            vlm_type=vlm_type,
            model_name=model_name,
            vlm_kwargs=vlm_kwargs,
            structured_output=structured_output,
            device=device,
            api_key=api_key,
            call_type=call_type,
        )
