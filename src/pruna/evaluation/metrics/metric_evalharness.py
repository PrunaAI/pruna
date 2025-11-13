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

from typing import Any, List, Tuple

import numpy as np
from lm_eval.api import metrics  # noqa: F401  # needed to register lm-eval metrics
from lm_eval.api import registry as lm_registry

from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.logging.logger import pruna_logger

METRIC_EVALHARNESS = "lm_eval_metric"


@MetricRegistry.register(METRIC_EVALHARNESS)
class LMEvalMetric(StatefulMetric):
    """
    Generic Pruna wrapper for lm-evaluation-harness metrics.

    This metric accumulates (reference, prediction) pairs and delegates
    computation to lm-eval’s registered metric and aggregation functions.

    Parameters
    ----------
    metric_name : str
        Name of the lm-eval metric (e.g., "acc", "f1", "bleu", "exact_match").
    call_type : str
        Type of metric call in Pruna (default: "y_gt").
    """

    pairs: List[Tuple[Any, Any]]  # dynamically added by add_state()

    def __init__(self, metric_name: str, call_type: str = "y_gt") -> None:
        super().__init__()
        self.metric_name = metric_name
        self.call_type = call_type

        # not using huggingface evaluate fallback
        if metric_name not in lm_registry.METRIC_REGISTRY:
            raise ValueError(f"Metric '{metric_name}' not found in lm-eval registry.")

        self.metric_fn = lm_registry.METRIC_REGISTRY[metric_name]
        self.agg_fn = lm_registry.get_metric_aggregation(metric_name)
        self.higher_is_better = lm_registry.is_higher_better(metric_name)

        if self.agg_fn is None:
            raise ValueError(f"No aggregation function registered for '{metric_name}'.")

        if self.higher_is_better is None:
            pruna_logger.warning(f"higher_is_better not specified for '{metric_name}', defaulting to True.")
            self.higher_is_better = True

        self.metric_units = metric_name

        self.add_state("pairs", [])

        pruna_logger.info(f"LMEvalMetric initialized: {metric_name} (higher_is_better={self.higher_is_better})")

    def update(self, preds, refs) -> None:
        """Accumulate predictions and references for later aggregation."""
        if len(preds) != len(refs):
            raise ValueError(f"Preds and refs length mismatch: {len(preds)} vs {len(refs)}")

        for ref, pred in zip(refs, preds):
            raw_item = self.metric_fn((ref, pred))
            self.pairs.append(raw_item)

    def compute(self) -> MetricResult:
        """
        Compute the lm-eval metric by applying its metric and aggregation functions.

        Returns
        -------
        MetricResult
            Pruna-compatible metric result object.
        """
        if not self.pairs:
            pruna_logger.warning(f"No data to compute {self.metric_name}, returning 0.0")
            return MetricResult(
                name=self.metric_name,
                params={
                    "num_samples": 0,
                    "higher_is_better": self.higher_is_better,
                    "metric_units": self.metric_units,
                },
                result=0.0,
            )

        try:
            score = self.agg_fn(self.pairs)
        except Exception as e:
            pruna_logger.error(f"Failed computing lm-eval metric {self.metric_name}: {e}")
            raise

        return MetricResult(
            name=self.metric_name,
            params={
                "num_samples": len(self.pairs),
                "higher_is_better": self.higher_is_better,
                "metric_units": self.metric_units,
            },
            result=score,
        )
