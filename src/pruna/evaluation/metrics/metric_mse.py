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

from pruna.engine.utils import device_to_string
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_MSE = "mse"


@MetricRegistry.register(METRIC_MSE)
class MSE(StatefulMetric):
    """
    Calculates the Mean Squared Error (MSE) between two tensors.

    The MSE is the average of the element-wise squared differences between two tensors.
    When comparing images, this is equivalent to overlaying the two images and averaging the squared
    pixel differences: identical images give an MSE of ``0.0`` and larger values indicate that the
    tensors are further apart.

    The error is accumulated over all batches seen during evaluation, so the reported value is the
    mean squared error over every element of every batch.

    Parameters
    ----------
    *args : Any
        Additional arguments to pass to the StatefulMetric constructor.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    call_type : str
        The call type to use for the metric. Determines which two tensors are compared,
        e.g. the model outputs against the ground truth. Defaults to single mode.
    **kwargs : Any
        Additional keyword arguments to pass to the StatefulMetric constructor.
    """

    sum_squared_error: torch.Tensor
    n_observations: torch.Tensor
    metric_name: str = METRIC_MSE
    higher_is_better: bool = False
    default_call_type: str = "gt_y"

    def __init__(
        self,
        *args,
        device: str | torch.device | None = None,
        call_type: str = SINGLE,
        **kwargs,
    ) -> None:
        super().__init__(device=device)
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)

        self.add_state("sum_squared_error", torch.tensor(0.0, device=self.device))
        self.add_state("n_observations", torch.tensor(0, device=self.device))

    @torch.no_grad()
    def update(self, x: List[Any] | torch.Tensor, gt: List[Any] | torch.Tensor, outputs: Any) -> None:
        """
        Update the metric with new batch data.

        The two tensors selected by the call type are compared element-wise and the sum of their
        squared differences is accumulated together with the number of compared elements.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data.
        gt : List[Any] | torch.Tensor
            The ground truth / cached images.
        outputs : Any
            The output images.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type, self.device)
        a = torch.as_tensor(inputs[0], dtype=torch.float32, device=self.device)
        b = torch.as_tensor(inputs[1], dtype=torch.float32, device=self.device)

        if a.shape != b.shape:
            msg = f"MSE requires matching tensor shapes, got {tuple(a.shape)} and {tuple(b.shape)}."
            pruna_logger.error(msg)
            raise ValueError(msg)

        self.sum_squared_error += torch.sum((a - b) ** 2)
        self.n_observations += a.numel()

    def compute(self) -> MetricResult:
        """
        Compute the mean squared error over all accumulated batches.

        Returns
        -------
        MetricResult
            The computed MSE metric result.
        """
        if self.n_observations == 0:
            pruna_logger.warning("MSE receives no samples, returning NaN.")
            return MetricResult(self.metric_name, self.__dict__.copy(), float("nan"))

        result = (self.sum_squared_error / self.n_observations).item()
        return MetricResult(self.metric_name, self.__dict__.copy(), result)

    def move_to_device(self, device: str | torch.device) -> None:
        """
        Move the metric to a specific device.

        Parameters
        ----------
        device : str | torch.device
            The device to move the metric to.
        """
        if not self.is_device_supported(device):
            raise ValueError(
                f"Metric {self.metric_name} does not support device {device}. Must be one of {self.runs_on}."
            )
        self.device = device_to_string(device)
        self.sum_squared_error = self.sum_squared_error.to(device)
        self.n_observations = self.n_observations.to(device)
