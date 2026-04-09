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

"""QA Accuracy metric using VLM for image understanding evaluation."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

import numpy as np
import torch

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import (
    SINGLE,
    get_call_type_for_single_metric,
    metric_data_processor,
)
from pruna.evaluation.metrics.vlm_base import BaseVLM, get_vlm
from pruna.evaluation.metrics.vlm_utils import VQAnswer, _process_images


@MetricRegistry.register("qa_accuracy")
class QAAccuracyMetric(StatefulMetric):
    """
    QA Accuracy metric.

    Uses VLM to answer questions about images.
    Higher scores indicate better image understanding.

    Parameters
    ----------
    *args : Any
        Additional positional arguments.
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
        Additional arguments. Supports ``aggregation`` (e.g. ``"all_or_nothing"`` for GenEval-style
        wiring); stored on the metric instance.
    """

    scores: List[float]
    default_call_type: str = "y_gt"
    higher_is_better: bool = True
    metric_name: str = "qa_accuracy"

    def __init__(
        self,
        *args,
        vlm: Optional[BaseVLM] = None,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str | None = None,
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
        self.response_format = VQAnswer if structured_output else None

        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])
        self.aggregation = kwargs.pop("aggregation", "mean")

    def _extract_questions(self, gt: Any, n: int) -> List[List[str]]:
        if isinstance(gt, (list, tuple)) and len(gt) >= n:
            out = []
            for i in range(n):
                v = gt[i]
                if isinstance(v, dict) and "questions" in v:
                    qs = v["questions"]
                    out.append(list(qs.values()) if isinstance(qs, dict) else list(qs))
                else:
                    out.append([])
            return out
        return [[]] * n

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data.
        gt : torch.Tensor
            The ground truth (questions per image).
        outputs : torch.Tensor
            The output images.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        auxiliaries = inputs[1] if len(inputs) > 1 else []
        questions_per_image = self._extract_questions(auxiliaries, len(images))
        for i, image in enumerate(images):
            questions = questions_per_image[i] if i < len(questions_per_image) else []
            if not questions:
                aux = auxiliaries[i] if i < len(auxiliaries) else {}
                raise ValueError(
                    "qa_accuracy requires 'questions' in auxiliaries. "
                    "Use a benchmark that provides it (e.g. GenEval, DPG, OneIG). "
                    f"Got aux keys: {list(aux.keys()) if isinstance(aux, dict) else 'not a dict'}."
                )
            scores = self.vlm.score(
                [image] * len(questions),
                questions,
                ["Yes"] * len(questions),
                response_format=self.response_format,
            )
            score = float(np.mean(scores))
            self.scores.append(score)

    def compute(self) -> MetricResult:
        """
        Compute the QA accuracy score.

        Returns
        -------
        MetricResult
            The mean QA accuracy across all updates.
        """
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))
