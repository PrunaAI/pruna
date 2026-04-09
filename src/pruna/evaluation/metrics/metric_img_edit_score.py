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
Image Edit Score metric.

VLM-based instruction-following score for image editing. Evaluates how well an edited image
follows the given editing instruction on a 0-10 scale. Related work: EditScore (arXiv:2509.23909),
ADIEE (ICCV 2025).
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

import numpy as np
import torch

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.vlm_utils import FloatOutput, _process_images, get_score_from_response
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import (
    SINGLE,
    get_call_type_for_single_metric,
    metric_data_processor,
)
from pruna.evaluation.metrics.vlm_base import BaseVLM, get_vlm


@MetricRegistry.register("img_edit_score")
class ImageEditScoreMetric(StatefulMetric):
    """
    Image Edit Score metric.

    VLM-based instruction-following score for image editing. Evaluates how well an edited image
    follows the given editing instruction. Higher scores indicate better editing quality.

    Related work: EditScore (arXiv:2509.23909), ADIEE (ICCV 2025).

    Parameters
    ----------
    *args : Any
        Additional positional arguments.
    vlm : BaseVLM | None, optional
        Custom VLM instance. If provided, vlm_type and model_name are ignored.
    vlm_type : {"litellm", "transformers"}, optional
        VLM backend. Default is "litellm".
    model_name : str, optional
        Model name. Default is "gpt-4o".
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
    """

    scores: List[float]
    default_call_type: str = "y_x"
    higher_is_better: bool = True
    metric_name: str = "img_edit_score"

    def __init__(
        self,
        *args,
        vlm: Optional[BaseVLM] = None,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str = "gpt-4o",
        vlm_kwargs: Optional[dict] = None,
        structured_output: bool = True,
        device=None,
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        **kwargs,
    ):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)

        self.vlm = get_vlm(
            vlm=vlm,
            vlm_type=vlm_type,
            model_name=model_name,
            device=device,
            api_key=api_key,
            structured_output=structured_output,
            **(vlm_kwargs or {}),
        )
        self.response_format = FloatOutput if structured_output else None

        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (editing instructions).
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output (edited) images.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = inputs[1] if len(inputs) > 1 and isinstance(inputs[1], list) else [""] * len(images)
        for i, image in enumerate(images):
            prompt = prompts[i] if i < len(prompts) else ""
            question = (
                f'On a scale of 0 to 10, how well does this edited image follow the instruction "{prompt}"? '
                "0 = instruction not followed at all, 10 = perfectly executed. Reply with a single number."
            )
            responses = self.vlm.generate([image], [question], response_format=self.response_format)
            self.scores.append(get_score_from_response(responses[0]))

    def compute(self) -> MetricResult:
        """
        Compute the image edit score.

        Returns
        -------
        MetricResult
            The mean image edit score across all updates.
        """
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))
