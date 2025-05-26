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
from typing import Any

from torch.utils.data import DataLoader

from pruna.engine.pruna_model import PrunaModel


class BaseMetric(ABC):
    """The base class for all Pruna metrics."""

    metric_name: str
    metric_units: str
    higher_is_better: bool

    @abstractmethod
    def compute(
        self,
        model: PrunaModel,
        dataloader: DataLoader,
    ) -> Any:
        """
        Compute the metric value.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        dataloader : DataLoader
            The dataloader to use for the evaluation.

        Returns
        -------
        Any
            The computed metric value.
        """
        pass
