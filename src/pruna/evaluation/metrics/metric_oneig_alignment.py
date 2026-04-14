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

from typing import Any, Mapping

import torch

from pruna.evaluation.metrics.metric_qa_accuracy import QAAccuracyMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.utils import metric_data_processor
from pruna.evaluation.metrics.vlm_utils import _process_images


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


@MetricRegistry.register("oneig_alignment")
class OneIGAlignmentMetric(QAAccuracyMetric):
    """
    OneIG alignment with dependency-aware aggregation.

    Reuses :class:`QAAccuracyMetric` VLM Yes/No scoring but aggregates like
    ``OneIG-Benchmark`` ``alignment_score.py`` for a **single** grid cell (no
    ``split_mxn_grid``): question ids are sorted numerically, raw scores are
    masked when any non-root parent is ``No``, then the mean over all questions
    is stored per image. Entries with null or blank question text (HF ``datasets``
    schema padding) are omitted from scoring.

    Numerical parity with upstream also depends on the VLM (e.g. ``openai/gpt-4o`` via
    litellm vs reference Qwen2.5-VL).

    Parameters
    ----------
    *args : Any
        Additional positional arguments for :class:`QAAccuracyMetric`.
    vlm : BaseVLM | None, optional
        Custom VLM instance. If provided, ``vlm_type`` and ``model_name`` are ignored.
    vlm_type : {"litellm", "transformers"}, optional
        VLM backend. Default is ``"litellm"``.
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
        Additional keyword arguments for :class:`QAAccuracyMetric`.
    """

    metric_name: str = "oneig_alignment"
    metric_units: str = "alignment"

    def update(self, x: list[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Score each question with the VLM, apply dependency masking, append per-cell mean.

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
        aux_list = inputs[1] if len(inputs) > 1 else []
        if isinstance(aux_list, torch.Tensor):
            aux_list = aux_list.tolist()
        for i, image in enumerate(images):
            aux = aux_list[i] if i < len(aux_list) else {}
            if not isinstance(aux, dict):
                raise ValueError(
                    "oneig_alignment requires aux[{}] to be a dict with 'questions'. Got: {!r}.".format(i, type(aux))
                )
            qs = aux.get("questions")
            if not isinstance(qs, dict) or not qs:
                raise ValueError(
                    f"oneig_alignment requires 'questions' as a non-empty dict on aux. Got keys: {list(aux.keys())}."
                )
            qmap = _int_dict_keys(qs)
            qids = _active_oneig_question_ids(qmap)
            if not qids:
                self.scores.append(0.0)
                continue
            question_texts = [str(qmap[qi]) for qi in qids]
            deps = _normalize_dependencies(aux.get("dependencies", {}))
            raw_scores_list = self.vlm.score(
                [image] * len(question_texts),
                question_texts,
                ["Yes"] * len(question_texts),
                response_format=self.response_format,
            )
            raw_map = {qid: float(raw_scores_list[j]) for j, qid in enumerate(qids)}
            filtered = apply_oneig_dependency_mask(raw_map, deps)
            self.scores.append(aggregate_oneig_alignment_per_cell(filtered, qids))
