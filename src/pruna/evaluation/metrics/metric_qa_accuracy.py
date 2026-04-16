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

from typing import Any, Literal

import numpy as np
import torch

from pruna.evaluation.metrics.metric_vlm_base import StatefulVLMMeanScoresMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import (
    SINGLE,
    metric_data_processor,
)
from pruna.evaluation.metrics.vlm_base import BaseVLM
from pruna.evaluation.metrics.vlm_utils import VQAnswer, _process_images


@MetricRegistry.register("qa_accuracy")
class QAAccuracyMetric(StatefulVLMMeanScoresMetric):
    """
    QA Accuracy metric.

    Uses a VLM to score yes/no alignment between each question and the generated image.
    Higher scores indicate better image understanding.

    **Multiple questions** come from each auxiliary dict's ``questions`` mapping (e.g. GenEval
    atomic probes, OneIG items). Each question is scored independently via :meth:`BaseVLM.score`
    with expected answer ``"Yes"``.

    **Aggregation** (``aggregation`` kwarg):

    - ``mean`` (default): per image, average VLM scores over all questions; the metric's
      :meth:`compute` returns the mean of those per-image values across ``update`` calls.
    - ``all_or_nothing``: per image, ``1.0`` only if **every** question scores strictly above
      ``0.5`` (scores equal to ``0.5`` count as failure). This matches strict GenEval-style
      reporting (all atomic checks must pass per sample; see `GenEval
      <https://arxiv.org/abs/2310.11513>`_). :class:`~pruna.evaluation.task.Task` wires this for
      the GenEval benchmark.

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
        Supports ``aggregation``: ``"mean"`` or ``"all_or_nothing"``.

    Raises
    ------
    ValueError
        If ``aggregation`` is not ``"mean"`` or ``"all_or_nothing"``.

    Examples
    --------
    Same ``hosted`` / ``local`` pattern as :func:`~pruna.evaluation.metrics.vlm_base.get_vlm`:

    .. code-block:: python

        import torch

        from pruna.evaluation.metrics import QAAccuracyMetric

        hosted = QAAccuracyMetric(vlm_type="litellm", model_name="openai/gpt-4o")
        local = QAAccuracyMetric(
            vlm_type="transformers",
            model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            device="cpu",
            vlm_kwargs={"model_load_kwargs": {"torch_dtype": torch.float32}},
        )
    """

    scores: list[float]
    default_call_type: str = "y_gt"
    higher_is_better: bool = True
    metric_units: str = "accuracy"
    metric_name: str = "qa_accuracy"

    def __init__(
        self,
        *args,
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
        self.response_format = VQAnswer if structured_output else None
        self.aggregation = kwargs.pop("aggregation", "mean")
        if self.aggregation not in {"mean", "all_or_nothing"}:
            raise ValueError(
                f"qa_accuracy aggregation must be one of {{'mean', 'all_or_nothing'}}. Got: {self.aggregation!r}."
            )
        self.metric_units = type(self).metric_units
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

    def _extract_questions(self, gt: Any, n: int) -> list[list[str]]:
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
        return [[] for _ in range(n)]

    def update(self, x: list[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : list[Any] | torch.Tensor
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
            if self.aggregation == "all_or_nothing":
                score = 1.0 if all(s > 0.5 for s in scores) else 0.0
            else:
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
        return self.compute_mean_of_scores()
