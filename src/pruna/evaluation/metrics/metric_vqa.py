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

Note: VQAScore uses P(Yes) (probability of "Yes" answer) for ranking. This implementation
defaults to binary (0/1) for compatibility. Set use_probability=True when using litellm
with a provider that supports logprobs to get soft scores.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

import numpy as np
import torch

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.metric_vlm_utils import YesNoAnswer, _process_images
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.evaluation.metrics.vlm_base import BaseVLM, get_vlm


@MetricRegistry.register("vqa")
class VQAMetric(StatefulMetric):
    """
    VQA (Visual Question Answering) metric.

    Uses VLM to answer "Does this image show '{prompt}'?" and scores alignment.
    Higher scores indicate better image-text alignment.

    VQAScore (arXiv:2404.01291) uses P(Yes) for ranking. Default is binary (0/1).
    Set use_probability=True with litellm + logprobs-capable provider for soft scores.

    Parameters
    ----------
    *args : Any
        Additional positional arguments.
    vlm : BaseVLM | None, optional
        Custom VLM instance. If provided, vlm_type and model_name are ignored.
    vlm_type : {"litellm", "transformers"}, optional
        VLM backend to use. Default is "litellm".
    model_name : str, optional
        Model name (gpt-4o for litellm, model path for transformers).
    vlm_kwargs : dict, optional
        Extra kwargs for VLM init (e.g. model_load_kwargs for transformers).
    structured_output : bool, optional
        Use structured generation for stable outputs. Default is True.
    use_outlines : bool, optional
        Use outlines for transformers. Default is False.
    device : str | torch.device | None, optional
        Device for transformers VLM.
    api_key : str | None, optional
        API key for litellm.
    call_type : str, optional
        Call type for the metric.
    use_probability : bool, optional
        If True, use P(Yes) when backend supports logprobs (litellm). Otherwise binary 0/1.
        Default is False for backward compatibility.
    **kwargs : Any
        Additional arguments.
    """

    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = "vqa"
    runs_on: List[str] = ["cpu"]

    def __init__(
        self,
        *args,
        vlm: Optional[BaseVLM] = None,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str = "gpt-4o",
        vlm_kwargs: Optional[dict] = None,
        structured_output: bool = True,
        use_outlines: bool = False,
        device=None,
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        use_probability: bool = False,
        **kwargs,
    ):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        self.structured_output = structured_output
        self.use_probability = use_probability

        self.vlm = get_vlm(
            vlm=vlm,
            vlm_type=vlm_type,
            model_name=model_name,
            device=device,
            api_key=api_key,
            use_outlines=use_outlines,
            **(vlm_kwargs or {}),
        )
        self.response_format = (
            YesNoAnswer if structured_output and vlm_type == "litellm" else
            ("yes_no" if structured_output and vlm_type == "transformers" else None)
        )

        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = x if isinstance(x, list) else [""] * len(images)

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
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))
