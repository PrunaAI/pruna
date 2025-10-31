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
from torch import Tensor

from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_MSE = "mse"


@MetricRegistry.register(METRIC_MSE)
class MSEMetric(StatefulMetric):
    """
    Mean Squared Error metric. Accumulates sum of squared errors and sample count across batches.

    The MSE metric compares predictions against ground truth values by computing the mean of squared differences.
    Lower values indicate better performance.

    Parameters
    ----------
    *args : Any
        Additional arguments to pass to the StatefulMetric constructor.
    call_type : str, optional
        The call type to use for the metric. Default is SINGLE.
    **kwargs : Any
        Additional keyword arguments to pass to the StatefulMetric constructor.
    """

    squared_errors: List[Tensor]
    default_call_type: str = "gt_y"  # ground truth vs outputs
    higher_is_better: bool = False  # Lower MSE means better performance
    metric_name: str = METRIC_MSE

    def __init__(self, *args, call_type: str = SINGLE, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)

        # Register state variables - use empty list to accumulate squared errors
        self.add_state("squared_errors", [])

    def update(self, x: Any | Tensor, gt: Tensor, outputs: Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : Any | Tensor
            The input data (may be unused depending on call_type).
        gt : Tensor
            The ground truth values.
        outputs : Tensor
            The model predictions/outputs.
        """
        # Process inputs based on call_type (returns tuple of tensors)
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        gt_tensor = inputs[0]
        output_tensor = inputs[1]

        if gt_tensor is None or output_tensor is None:
            pruna_logger.debug("MSE.update received None for gt or outputs; skipping.")
            return

        # Ensure tensors are on the same device
        output_tensor = output_tensor.to(gt_tensor.device)

        # Flatten tensors for easier computation
        try:
            gt_flat = gt_tensor.view(-1)
            out_flat = output_tensor.view(-1)
        except RuntimeError:
            gt_flat = gt_tensor.flatten()
            out_flat = output_tensor.flatten()

        # Ensure same number of elements
        if gt_flat.numel() != out_flat.numel():
            pruna_logger.warning(
                f"MSE: Ground truth ({gt_flat.numel()} elements) and output "
                f"({out_flat.numel()} elements) have different sizes. Skipping batch."
            )
            return

        # Compute squared errors and append to list
        squared_errors = (out_flat - gt_flat) ** 2
        self.squared_errors.append(squared_errors)

    def compute(self) -> MetricResult:
        """
        Compute the final MSE metric value.

        Returns
        -------
        MetricResult
            The computed MSE value wrapped in a MetricResult object.
        """
        if not self.squared_errors:
            mse_value = float("nan")
            pruna_logger.warning("MSE: No samples accumulated. Returning NaN.")
        else:
            # Concatenate all squared errors and compute mean
            all_squared_errors = torch.cat(self.squared_errors)
            mse_value = float(all_squared_errors.mean().item())

        return MetricResult(self.metric_name, self.__dict__.copy(), mse_value)
