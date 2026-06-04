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

"""OneIG alignment scoring with dependency masking (parent ``No`` gates children)."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import torch
from PIL import Image

from pruna.evaluation.metrics.metric_qa_accuracy import QAAccuracyMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.utils import metric_data_processor
from pruna.evaluation.metrics.vlm_utils import _process_images, split_mxn_grid

_DEFAULT_ONEIG_ALIGNMENT_VLM = "Qwen/Qwen2.5-VL-7B-Instruct"


def _int_dict_keys(mapping: Mapping[Any, Any]) -> dict[int, Any]:
    return {int(k): v for k, v in mapping.items()}


def _normalize_dependencies(deps: Any) -> dict[int, list[int]]:
    if not isinstance(deps, Mapping):
        return {}
    out: dict[int, list[int]] = {}
    for k, v in deps.items():
        key = int(k)
        if isinstance(v, list):
            out[key] = [int(p) for p in v]
        else:
            out[key] = []
    return out


def _active_oneig_question_ids(qmap: dict[int, Any]) -> list[int]:
    """Question ids with real prompt text (excludes HF ``datasets`` padding and empty slots)."""
    active: list[int] = []
    for qi in sorted(qmap):
        text = qmap[qi]
        if text is None:
            continue
        s = str(text).strip()
        if not s or s == "None":
            continue
        active.append(qi)
    return active


def apply_oneig_dependency_mask(
    raw_scores: Mapping[int, float],
    dependencies: Mapping[int, list[int]],
) -> dict[int, float]:
    """
    Apply OneIG ``filter_score`` logic per dependency graph (single grid cell).

    Parents with semantic answer ``No`` (score ``0``) force dependent question
    scores to ``0``. Parent id ``0`` is ignored, matching the reference script.

    Parameters
    ----------
    raw_scores : Mapping[int, float]
        Map question id → VLM score in ``{0, 1}`` (or float) before masking.
    dependencies : Mapping[int, list[int]]
        Map child question id → list of parent question ids (use ``[0]`` for roots).

    Returns
    -------
    dict[int, float]
        Copy of scores with dependent questions zeroed when any non-zero parent
        scored ``0``.
    """
    filtered = {int(k): float(v) for k, v in raw_scores.items()}
    deps = _normalize_dependencies(dependencies)
    raw = dict(filtered)
    for child_id, parent_ids in deps.items():
        if child_id not in filtered:
            continue
        any_parent_no = False
        for parent_id in parent_ids:
            if parent_id == 0:
                continue
            if parent_id not in raw:
                continue
            if raw[parent_id] == 0.0:
                any_parent_no = True
                break
        if any_parent_no:
            filtered[child_id] = 0.0
    return filtered


def aggregate_oneig_alignment_per_cell(filtered_scores: Mapping[int, float], question_ids: list[int]) -> float:
    """
    Mean filtered score over all questions in the prompt (one grid cell).

    Parameters
    ----------
    filtered_scores : Mapping[int, float]
        Post-mask scores for each question id.
    question_ids : list[int]
        Ordered ids (typically sorted ascending) defining the denominator.

    Returns
    -------
    float
        Average score in ``[0, 1]`` if inputs are binary; ``0.0`` if ``question_ids`` is empty.
    """
    if not question_ids:
        return 0.0
    s = sum(float(filtered_scores[qid]) for qid in question_ids)
    return s / float(len(question_ids))


def _aux_list_from_gt(aux_slot: Any, batch_size: int) -> list[dict[str, Any]]:
    if isinstance(aux_slot, torch.Tensor):
        raise ValueError(
            "oneig_alignment expects gt as list[dict] with 'questions' and optional 'dependencies'. "
            f"Got tensor with shape {tuple(aux_slot.shape)}."
        )
    if not isinstance(aux_slot, (list, tuple)):
        return [{} for _ in range(batch_size)]
    out: list[dict[str, Any]] = []
    for i in range(batch_size):
        row = aux_slot[i] if i < len(aux_slot) else {}
        if not isinstance(row, dict):
            raise ValueError(f"oneig_alignment requires aux[{i}] to be a dict. Got: {type(row)!r}.")
        out.append(row)
    return out


@MetricRegistry.register("oneig_alignment")
class OneIGAlignmentMetric(QAAccuracyMetric):
    """
    OneIG alignment with dependency-aware aggregation.

    Matches ``OneIG-Benchmark`` ``alignment_score.py``: split an ``m x n`` output grid
    (default ``2 x 2``), score **one question per VLM call** across all cells, apply
    dependency masking per cell, then average cell scores.

    Scoring semantics
    -----------------
    OneIG Q_D probes are phrased so **Yes = aligned**. Each call requests
    :meth:`~pruna.evaluation.metrics.vlm_base.BaseVLM.score` with expected answer
    ``"Yes"`` (probability of Yes). Low scores act as semantic **No** for dependency
    masking.

    Parameters
    ----------
    grid_size : tuple[int, int], optional
        ``(columns, rows)`` for :func:`~pruna.evaluation.metrics.vlm_utils.split_mxn_grid`.
        Default ``(2, 2)`` per OneIG. Use ``(1, 1)`` to score the full image without splitting.
    vlm : BaseVLM | None, optional
        Custom VLM instance. If provided, ``vlm_type`` and ``model_name`` are ignored.
    vlm_type : {"litellm", "transformers"}, optional
        VLM backend. Default is ``"transformers"`` (paper-faithful Qwen2.5-VL).
    model_name : str | None, optional
        HuggingFace or litellm model id. Default ``Qwen/Qwen2.5-VL-7B-Instruct``.
    vlm_kwargs : dict, optional
        Forwarded by ``get_vlm``.
    structured_output : bool, optional
        Use structured generation when applicable.
    device : str | torch.device | None, optional
        Device for transformers VLM.
    api_key : str | None, optional
        API key for litellm.
    call_type : str, optional
        Call type for the metric.
    aggregation : str, optional
        Unused; kept for registry compatibility with :class:`QAAccuracyMetric`.
    **kwargs : Any
        Additional keyword arguments for :class:`QAAccuracyMetric`.

    Examples
    --------
    .. code-block:: python

        from pruna.evaluation.metrics import OneIGAlignmentMetric

        paper = OneIGAlignmentMetric(device="cuda")
        api = OneIGAlignmentMetric(vlm_type="litellm", model_name="openai/gpt-4o")
    """

    metric_name: str = "oneig_alignment"
    metric_units: str = "alignment"

    def __init__(
        self,
        *args: Any,
        grid_size: tuple[int, int] = (2, 2),
        vlm: Any | None = None,
        vlm_type: Literal["litellm", "transformers"] = "transformers",
        model_name: str | None = _DEFAULT_ONEIG_ALIGNMENT_VLM,
        vlm_kwargs: dict | None = None,
        structured_output: bool = True,
        device: str | torch.device | None = None,
        api_key: str | None = None,
        call_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            vlm=vlm,
            vlm_type=vlm_type,
            model_name=model_name,
            vlm_kwargs=vlm_kwargs,
            structured_output=structured_output,
            device=device,
            api_key=api_key,
            call_type=call_type,
            **kwargs,
        )
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.metric_units = type(self).metric_units

    def _score_sample(self, image: Any, aux: dict[str, Any]) -> float:
        if not isinstance(image, Image.Image):
            if isinstance(image, torch.Tensor):
                from pruna.evaluation.metrics.vlm_utils import _tensor_to_pil

                image = _tensor_to_pil(image)
            else:
                image = Image.fromarray(image).convert("RGB")
        cells = split_mxn_grid(image, self.grid_size)
        qs = aux.get("questions")
        if not isinstance(qs, dict) or not qs:
            raise ValueError(
                f"oneig_alignment requires 'questions' as a non-empty dict on aux. Got keys: {list(aux.keys())}."
            )
        qmap = _int_dict_keys(qs)
        qids = _active_oneig_question_ids(qmap)
        if not qids:
            return 0.0
        deps = _normalize_dependencies(aux.get("dependencies", {}))
        per_question_cell_scores: dict[int, list[float]] = {}
        n_cells = len(cells)
        for qid in qids:
            qtext = str(qmap[qid])
            raw_scores_list = self.vlm.score(
                cells,
                [qtext] * n_cells,
                ["Yes"] * n_cells,
                response_format=self.response_format,
            )
            per_question_cell_scores[qid] = [float(s) for s in raw_scores_list]
        cell_means: list[float] = []
        for cell_i in range(n_cells):
            raw_map = {qid: per_question_cell_scores[qid][cell_i] for qid in qids}
            filtered = apply_oneig_dependency_mask(raw_map, deps)
            cell_means.append(aggregate_oneig_alignment_per_cell(filtered, qids))
        return float(sum(cell_means) / len(cell_means))

    def update(self, x: list[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Score each prompt image with OneIG alignment (grid split + per-question VLM calls).

        Parameters
        ----------
        x : list[Any] | torch.Tensor
            Unused batch metadata (kept for metric interface).
        gt : torch.Tensor
            Ground-truth slot holding per-sample aux dicts with ``questions`` and
            optionally ``dependencies``.
        outputs : torch.Tensor
            Model outputs (images) evaluated against the questions.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        aux_list = _aux_list_from_gt(inputs[1] if len(inputs) > 1 else [], len(images))
        for i, image in enumerate(images):
            self.scores.append(self._score_sample(image, aux_list[i]))
