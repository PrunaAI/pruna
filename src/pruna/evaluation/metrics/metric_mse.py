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

from typing import Any, List

import torch

from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import get_call_type, metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_MSE = "mse"


@MetricRegistry.register(METRIC_MSE)
class MSEMetric(StatefulMetric):
    """
    Mean Squared Error.
    
    MSE = (1/N) * sum((pred - target)^2)
    Lower is better (0 = perfect match).
    """

    metric_name = "mse"
    higher_is_better = False
    default_call_type = "pairwise_y_gt"
    runs_on = ["cuda", "cpu", "mps"]

    def __init__(self, device=None, call_type: str = ""):
        super().__init__(device=device)
        self.call_type = get_call_type(call_type, self.default_call_type)
        # accumulate sum of squared errors
        self.add_state("sum_sq_err", torch.tensor(0.0, device=self.device))
        # count total elements (not samples)
        self.add_state("n_elements", torch.tensor(0, dtype=torch.long, device=self.device))

    @torch.no_grad()
    def update(self, x: List[Any], gt: Any, outputs: Any) -> None:
        """Accumulate squared errors from a batch."""
        data = metric_data_processor(x, gt, outputs, self.call_type, self.device)
        preds = data[0]
        targets = data[1] if len(data) > 1 else data[0]

        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds, device=self.device)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=self.device)

        preds = preds.to(self.device)
        targets = targets.to(self.device)

        # squared errors for all elements
        sq_err = (preds - targets) ** 2
        
        self.sum_sq_err += sq_err.sum()
        self.n_elements += sq_err.numel()

        # sanity check if MSE is unusually high
        batch_mse = sq_err.mean().item()
        if batch_mse > 10000:
            pruna_logger.warning(f"MSE looks high ({batch_mse:.1f}), check data scaling")

    def compute(self) -> MetricResult:
        """Return mean of all squared errors."""
        if self.n_elements.item() == 0:
            pruna_logger.warning("MSE called with no data, returning 0")
            return MetricResult(self.metric_name, self.__dict__.copy(), 0.0)

        mse = (self.sum_sq_err / self.n_elements).item()
        return MetricResult(self.metric_name, self.__dict__.copy(), mse)
