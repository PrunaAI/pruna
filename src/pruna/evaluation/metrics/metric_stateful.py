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

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List

from torch import Tensor

from pruna.logging.logger import pruna_logger


class StatefulMetric(ABC):
    """
    Base class for all metrics that have state functionality.

    A stateful metric maintains internal state variables that accumulate information
    across multiple batches or iterations. Unlike simple metrics that compute values
    independently for each input, stateful metrics track running statistics or
    aggregated values over time.
    """

    metric_name: str
    call_type: str

    def __init__(self) -> None:
        """Initialize the StatefulMetric class."""
        super().__init__()
        self._defaults: Dict[str, List | Tensor] = {}

    def add_state(self, name: str, default: List | Tensor) -> None:
        """
        Add state variables to the metric.

        Parameters
        ----------
        name : str
            The name of the state variable.
        default : List | Tensor
            The default value of the state variable.
        """
        # The states must be a tensor or a list.
        # If it's a tensor, it should have an initial value. If it's a list, it should be empty.
        if (not isinstance(default, (Tensor, List))) or (isinstance(default, List) and default):
            pruna_logger.error("State variable must be a tensor or any empty list (where you can append tensors)")
            raise ValueError("State variable must be a tensor or any empty list (where you can append tensors)")

        if isinstance(default, Tensor):
            default = default.contiguous()

        setattr(self, name, default)

        self._defaults[name] = deepcopy(default)

    def forward(self, *args, **kwargs) -> None:
        """
        Compute the metric value.

        Parameters
        ----------
        *args : Any
            The arguments to pass to the metric.
        **kwargs : Any
            The keyword arguments to pass to the metric.
        """
        pass

    def reset(self) -> None:
        """
        Reset the metric state.

        This method clears all stored values used in the metric's calculations.
        For tensor-based states (e.g., running sums, counts), it replaces them with fresh tensors with default values.
        For list-based states it simply clears the contents in place.
        """
        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, Tensor):
                setattr(self, attr, default.detach().clone().to(current_val.device))
            else:
                getattr(self, attr).clear()

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Override this method to update the state variables of your metric.

        Parameters
        ----------
        *args : Any
            The arguments to pass to the metric.
        **kwargs : Any
            The keyword arguments to pass to the metric.
        """

    @abstractmethod
    def compute(self) -> Any:
        """Override this method to compute the final metric value."""

    def is_pairwise(self) -> bool:
        """
        Check if a metric is pairwise.

        Returns
        -------
        bool
            True if the metric is pairwise, False otherwise.
        """
        return self.call_type.startswith("pairwise")
