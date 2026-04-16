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

"""Shared helpers and base class for VLM-backed stateful metrics with mean-over-``scores``.

Concrete metrics call :func:`~pruna.evaluation.metrics.vlm_base.get_vlm` via
:class:`StatefulVLMMeanScoresMetric` with ``vlm_type`` of ``"litellm"`` (hosted) or
``"transformers"`` (local Hugging Face). See :mod:`~pruna.evaluation.metrics.vlm_base`.

For **auxiliary image bytes** (editing benchmarks, ``pred`` tensors in
:mod:`~pruna.evaluation.vlm_benchmark_helpers`), use
:func:`~pruna.evaluation.metrics.vlm_utils.pil_rgb_from_aux_image_bytes` and
:data:`~pruna.evaluation.metrics.vlm_utils.VLM_AUX_IMAGE_BYTES_KEY_ORDER` instead of ad-hoc
key scans. User-facing overview:
:doc:`Evaluate a model (vision-language judge metrics) </docs_pruna/user_manual/evaluate>`.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import get_call_type_for_single_metric
from pruna.evaluation.metrics.vlm_base import BaseVLM, get_vlm


def auxiliary_dicts_from_gt(gt: Any, batch_size: int) -> list[dict[str, Any]]:
    """
    Map batch ``gt`` to per-row auxiliary dicts when using ``prompt_with_auxiliaries_collate``.

    For ``y_x`` metrics, :func:`~pruna.evaluation.metrics.utils.metric_data_processor` does not
    include ``gt`` in its output; pass the batch ``gt`` argument here so fields such as
    ``source_image_bytes`` are visible to editing metrics.

    Parameters
    ----------
    gt : Any
        Second element of the dataloader batch: typically a ``list[dict]`` of aux columns.
    batch_size : int
        Number of samples in the batch.

    Returns
    -------
    list[dict[str, Any]]
        One dict per row; empty dicts when ``gt`` is not a list of dicts (e.g. tensor placeholders
        in tests).
    """
    if batch_size <= 0:
        return []
    if isinstance(gt, (list, tuple)) and gt and isinstance(gt[0], dict):
        out: list[dict[str, Any]] = []
        for i in range(batch_size):
            row = gt[i] if i < len(gt) else {}
            out.append(row if isinstance(row, dict) else {})
        return out
    return [{} for _ in range(batch_size)]


class StatefulVLMMeanScoresMetric(StatefulMetric):
    """
    Base for VLM metrics that accumulate ``scores`` and report the batch mean in :meth:`compute`.

    Subclasses set ``default_call_type`` and ``metric_name``, then call :meth:`_init_vlm_scores`
    from ``__init__`` after any metric-specific attributes (e.g. ``use_probability``).

    Parameters
    ----------
    device : str | torch.device | None
        Device forwarded to :class:`~pruna.evaluation.metrics.metric_stateful.StatefulMetric`.
    **kwargs : Any
        Additional keyword arguments forwarded to the parent class.

    Notes
    -----
    User guide: :doc:`Evaluate a model </docs_pruna/user_manual/evaluate>` (Vision-language judge metrics).
    Registry metrics (``VQAMetric``, ``VieScoreMetric``, …) pass ``vlm_type`` and ``model_name``
    into :meth:`_init_vlm_scores`; see :func:`~pruna.evaluation.metrics.vlm_base.get_vlm`.
    """

    scores: list[float]
    default_call_type: str = "y_x"
    higher_is_better: bool = True
    metric_name: str = ""

    def _init_vlm_scores(
        self,
        *,
        vlm: BaseVLM | None,
        vlm_type: Literal["litellm", "transformers"],
        model_name: str | None,
        vlm_kwargs: dict[str, Any] | None,
        structured_output: bool,
        device: str | torch.device | None,
        api_key: str | None,
        call_type: str,
    ) -> None:
        """Attach ``self.vlm``, ``self.call_type``, and the ``scores`` state."""
        self.vlm = get_vlm(
            vlm=vlm,
            vlm_type=vlm_type,
            model_name=model_name,
            device=device,
            api_key=api_key,
            structured_output=structured_output,
            **(vlm_kwargs or {}),
        )
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])
        self.higher_is_better = type(self).higher_is_better

    def compute_mean_of_scores(self) -> MetricResult:
        """
        Return the mean of accumulated ``scores``, or ``0.0`` when empty.

        Returns
        -------
        MetricResult
            Aggregated result for this metric.
        """
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))
