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

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union, cast

from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric
from pruna.logging.logger import pruna_logger

T2I_COMPBENCH = "t2i_compbench"


@dataclass(frozen=True)
class _EvalFile:
    key: str
    relpath: str
    label: str


_EVAL_FILES: Tuple[_EvalFile, ...] = (
    _EvalFile("attribute_binding", "annotation_blip/vqa_result.json", "Attribute Binding (BLIP-VQA)"),
    _EvalFile("non_spatial", "annotation_clip/vqa_result.json", "Non-spatial (CLIPScore)"),
    _EvalFile("spatial_2d", "labels/annotation_obj_detection_2d/vqa_result.json", "2D Spatial (UniDet)"),
    _EvalFile("spatial_3d", "labels/annotation_obj_detection_3d/vqa_result.json", "3D Spatial (UniDet)"),
    _EvalFile("numeracy", "annotation_num/vqa_result.json", "Numeracy (UniDet)"),
    _EvalFile("complex", "annotation_3_in_1/vqa_result.json", "Complex Compositions (3-in-1)"),
)


def _mean_from_vqa_json(path: Path) -> Tuple[float, int]:
    """Return (mean_score, count) from a T2I-CompBench(++) vqa_result.json."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        scores = [float(x.get("answer", 0.0)) for x in data]
    elif isinstance(data, dict):
        scores = [float(v) for v in data.values()]
    else:
        raise ValueError(f"Unsupported JSON structure at {path}")

    return (float(mean(scores)), len(scores)) if scores else (0.0, 0)


def _normalized_weights(
    weights: Optional[Mapping[str, float]], available: Iterable[str]
) -> Dict[str, float]:
    if not weights:
        a = list(available)
        return {k: 1.0 / max(1, len(a)) for k in a}
    filtered = {k: float(v) for k, v in weights.items() if k in set(available) and v > 0}
    if not filtered:
        a = list(available)
        return {k: 1.0 / max(1, len(a)) for k in a}
    total = sum(filtered.values())
    return {k: filtered.get(k, 0.0) / total for k in available}


def _aggregate_dir(results_dir: Union[str, Path], weights: Optional[Mapping[str, float]]) -> Dict[str, Any]:
    base = Path(results_dir)
    per_category: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    available, missing = [], []

    for spec in _EVAL_FILES:
        path = base / spec.relpath
        if not path.is_file():
            pruna_logger.info("[T2I-CompBench++] Missing results for %s at %s", spec.label, path)
            missing.append(spec.key)
            continue
        try:
            m, n = _mean_from_vqa_json(path)
        except Exception as e:
            pruna_logger.warning("[T2I-CompBench++] Failed to parse %s at %s: %s", spec.label, path, e)
            missing.append(spec.key)
            continue
        per_category[spec.key] = m
        counts[spec.key] = n
        available.append(spec.key)

    if not available:
        return {
            "overall": 0.0,
            "per_category": {},
            "counts": {},
            "available_categories": [],
            "missing": [s.key for s in _EVAL_FILES],
        }

    w = _normalized_weights(weights, available)
    overall = sum(per_category[k] * w[k] for k in available)

    return {
        "overall": float(overall),
        "per_category": per_category,
        "counts": counts,
        "available_categories": available,
        "missing": missing,
        "weights_used": w,
    }


@MetricRegistry.register(T2I_COMPBENCH)
class T2ICompBench(StatefulMetric):
    """
    Aggregates T2I-CompBench(++) scores from JSON outputs produced by the official evaluation scripts.

    This metric **does not** run BLIP-VQA/UniDet/CLIPScore/3-in-1; it only reads their outputs.
    """

    metric_name: str = T2I_COMPBENCH
    higher_is_better: bool = True
    metric_units: str = "score"  # 0–1
    default_call_type: str = "y"  # IQA-like metric that only uses outputs

    def __init__(
        self,
        results_dir: Union[str, Path],
        weights: Optional[Mapping[str, float]] = None,
        call_type: str = SINGLE,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Handle/ignore unused kwargs gracefully
        if "device" in kwargs:
            pruna_logger.info("Device argument ignored for %s (file aggregation only).", self.metric_name)
            kwargs.pop("device")
        if kwargs:
            pruna_logger.info("Ignoring unused kwargs for %s: %s", self.metric_name, list(kwargs.keys()))

        # Validate results_dir
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory does not exist: {self.results_dir}")
        if not self.results_dir.is_dir():
            raise ValueError(f"Results path is not a directory: {self.results_dir}")

        # Validate/keep weights
        if weights is not None:
            valid_categories = {spec.key for spec in _EVAL_FILES}
            invalid = set(weights.keys()) - valid_categories
            if invalid:
                pruna_logger.warning(
                    "Invalid category names in weights: %s. Valid categories are: %s",
                    invalid,
                    valid_categories,
                )
                weights = {k: v for k, v in weights.items() if k in valid_categories}
        self.weights = dict(weights) if weights is not None else None

        # Resolve call type once; we don't actually branch on it.
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)

        pruna_logger.info(
            "Initialized %s with results_dir=%s, call_type=%s",
            self.metric_name,
            self.results_dir,
            self.call_type,
        )

    def update(self, inputs: Any, ground_truths: Any, predictions: Any) -> None:
        """No-op. File-based aggregator does not use batch data."""
        return

    def compute(self) -> MetricResult:
        """
        Read category JSON files, aggregate with optional weights, and return the score.
        """
        try:
            agg = _aggregate_dir(self.results_dir, self.weights)

            # Keep scalar in value; stash everything else under params["details"].
            params: Dict[str, Any] = {
                "results_dir": str(self.results_dir),
                "weights": self.weights,
                "call_type": self.call_type,
                "details": {k: v for k, v in agg.items() if k != "overall"},
            }
            return MetricResult(self.metric_name, params, cast(float, agg["overall"]))
        except Exception as e:
            pruna_logger.error("Failed to compute %s: %s", self.metric_name, e)
            return self._get_default_result(error=str(e))

    def _get_default_result(self, error: Optional[str] = None) -> MetricResult:
        details: Dict[str, Any] = {
            "per_category": {},
            "counts": {},
            "available_categories": [],
            "missing": [s.key for s in _EVAL_FILES],
        }
        if error:
            details["error"] = error

        params: Dict[str, Any] = {
            "results_dir": str(getattr(self, "results_dir", "")),
            "weights": getattr(self, "weights", None),
            "call_type": getattr(self, "call_type", self.default_call_type),
            "details": details,
        }
        return MetricResult(self.metric_name, params, 0.0)
