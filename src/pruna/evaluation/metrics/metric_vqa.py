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

"""
VQA (Visual Question Answering) metric.

Reference: VQAScore - Evaluating Text-to-Visual Generation with Image-to-Text Generation
https://arxiv.org/abs/2404.01291

Note: VQAScore uses P(Yes) (probability of "Yes" answer) for ranking. With litellm,
use_probability=True (default) requests logprobs for soft scores when the provider supports it.
Set use_probability=False for binary 0/1. With ``transformers``, ``use_probability=True``
uses next-token softmax mass on yes/no prefix tokens (VQAScore-style); ``False`` uses
generation plus binary matching.

For API keys, LiteLLM vs local ``transformers``, and hosted vs local construction, see
:doc:`Evaluate a model </docs_pruna/user_manual/evaluate>` (Vision-language judge metrics) and
:func:`~pruna.evaluation.metrics.vlm_base.get_vlm`.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from pruna.evaluation.metrics.metric_vlm_base import StatefulVLMMeanScoresMetric, prompts_from_y_x_inputs
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, metric_data_processor
from pruna.evaluation.metrics.vlm_base import BaseVLM
from pruna.evaluation.metrics.vlm_utils import VQAnswer, _process_images


@MetricRegistry.register("vqa")
class VQAMetric(StatefulVLMMeanScoresMetric):
    """
    VQA (Visual Question Answering) metric.

    Uses VLM to answer "Does this image show '{prompt}'?" and scores alignment.
    Higher scores indicate better image-text alignment.

    VQAScore (arXiv:2404.01291) uses P(Yes) for ranking. Default ``use_probability=True``
    with litellm requests logprobs for soft scores when supported.

    Parameters
    ----------
    vlm : BaseVLM | None, optional
        Custom VLM instance. If provided, ``vlm_type`` and ``model_name`` are ignored.
    vlm_type : {"litellm", "transformers"}, optional
        VLM backend to use. Default is "litellm".
    model_name : str | None, optional
        Litellm model id or HuggingFace checkpoint id. **Required** when ``vlm`` is not
        provided (e.g. ``openai/gpt-4o``).
    vlm_kwargs : dict, optional
        Forwarded by ``get_vlm`` to ``LitellmVLM`` or ``TransformersVLM``. For local models,
        set ``model_load_kwargs`` for ``from_pretrained``; for litellm, pass extra API options.
    structured_output : bool, optional
        Use structured generation for stable outputs (litellm pydantic; transformers outlines
        when a string format is used). Default is True.
    device : str | torch.device | None, optional
        Device for transformers VLM.
    api_key : str | None, optional
        API key for litellm.
    call_type : str, optional
        Call type for the metric.
    use_probability : bool, optional
        If True, use P(Yes) when backend supports logprobs (litellm). Otherwise binary 0/1.
        Default is True for paper alignment.
    **kwargs : Any
        Additional arguments.

    Notes
    -----
    For strict binary scoring without logprobs, pass ``use_probability=False``. Hosted vs
    local setup: :doc:`Evaluate a model </docs_pruna/user_manual/evaluate>` (Vision-language judge metrics).
    """

    scores: list[float]
    default_call_type: str = "y_x"
    higher_is_better: bool = True
    metric_name: str = "vqa"

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
        use_probability: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device)
        self.use_probability = use_probability
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

    def update(self, x: list[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : list[Any] | torch.Tensor
            The input data (prompts).
        gt : torch.Tensor
            The ground truth (unused; present for call-type compatibility).
        outputs : torch.Tensor
            The output images.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = prompts_from_y_x_inputs(inputs, len(images))
        for i, image in enumerate(images):
            prompt = prompts[i] if i < len(prompts) else ""
            question = f'Does this image show "{prompt}"?'
            score = self.vlm.score(
                [image],
                [question],
                ["Yes"],
                response_format=self.response_format,
                use_probability=self.use_probability,
            )[0]
            self.scores.append(score)

    def compute(self) -> MetricResult:
        """
        Compute the VQA score.

        Returns
        -------
        MetricResult
            The mean VQA score across all updates.
        """
        return self.compute_mean_of_scores()
