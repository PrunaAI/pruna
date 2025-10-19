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

from typing import Any, Dict, List, Union

import torch

from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.logging.logger import pruna_logger

METRIC_EVALHARNESS = "eval_harness"


@MetricRegistry.register(METRIC_EVALHARNESS)
class EvalHarnessMetric(BaseMetric):
    """
    Wraps the EleutherAI LM Evaluation Harness as a Pruna metric.

    This allows you to run benchmark tasks (HellaSwag, MMLU, GSM8k, etc.)
    and return a scalar score usable in Pruna’s pipeline.

    Parameters
    ----------
    tasks : list[str]
        List of eval harness tasks to run.
    model_args : dict
        Arguments for the backend model in eval harness.
    device : str | torch.device
        Device to run evaluation on (cuda, cpu, etc.).
    call_type : str
        The call type (default "y").
    """

    metric_name: str = METRIC_EVALHARNESS
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_units: str = "score"

    def __init__(
        self,
        tasks: List[str],
        model_args: Dict[str, Any],
        device: Union[str, torch.device] = "cuda:0",
        call_type: str = "y",
        **kwargs,
    ):
        super().__init__()
        self.tasks = tasks
        self.model_args = model_args
        self.device = torch.device(device) if isinstance(device, str) else device
        self.call_type = call_type

    def compute(self, model, dataloader) -> MetricResult:
        """Run LM Eval Harness and return the aggregated result."""
        try:
            from lm_eval import evaluator
        except ImportError:
            raise ImportError("lm_eval package is not installed. Please install it with `pip install lm-eval`.")

        try:
            eval_results = evaluator.simple_evaluate(
                model="hf",
                model_args=self.model_args,
                tasks=self.tasks,
                device=str(self.device),
                batch_size=1,
            )
        except Exception as e:
            pruna_logger.error(f"Eval Harness failed: {e}")
            raise

        scores, task_scores = [], {}

        preferred_keys = ["acc,none", "acc", "f1", "em"]

        for task in self.tasks:
            if task not in eval_results["results"]:
                pruna_logger.warning(f"Task {task} not found in eval results")
                continue

            task_results = eval_results["results"][task]
            metric_keys = [k for k in task_results if k not in {"alias", "samples", "higher_is_better"}]

            if not metric_keys:
                pruna_logger.warning(f"No metrics found for task {task}")
                continue

            key = next((k for k in preferred_keys if k in metric_keys), metric_keys[0])
            score = task_results[key]
            task_scores[task] = score
            scores.append(score)

        final_score = sum(scores) / len(scores) if scores else 0.0

        return MetricResult(
            self.metric_name,
            {**self.__dict__, "task_scores": task_scores},
            final_score,
            higher_is_better=self.higher_is_better,
            metric_units=self.metric_units,
        )
