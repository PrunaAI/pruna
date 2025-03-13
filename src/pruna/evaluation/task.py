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

from typing import Any, List, cast

import torch

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.metric_pairwise_clip import PairwiseClipScore
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.metric_torch import TorchMetricWrapper
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.logging.logger import pruna_logger

AVAILABLE_REQUESTS = ("image_generation_quality",)


class Task:
    """
    Processes user requests and converts them into a format that the evaluation module can handle.

    Parameters
    ----------
    request : str | List[str | BaseMetric]
        The user request.
    datamodule : PrunaDataModule
        The dataloader to use for the evaluation.
    device : str | torch.device
        The device to use for the evaluation.
    """

    def __init__(
        self, request: str | List[str | BaseMetric], datamodule: PrunaDataModule, device: str | torch.device = "cuda"
    ) -> None:
        self.metrics = get_metrics(request)
        self.datamodule = datamodule
        self.dataloader = datamodule.test_dataloader()
        self.device = device

    def get_single_stateful_metrics(self) -> List[StatefulMetric]:
        """
        Get single stateful metrics.

        Returns
        -------
        List[StatefulMetric]
            The stateful metrics.
        """
        return [metric for metric in self.metrics if isinstance(metric, StatefulMetric) and not metric.is_pairwise()]

    def get_pairwise_stateful_metrics(self) -> List[StatefulMetric]:
        """
        Get pairwise stateful metrics.

        Returns
        -------
        List[StatefulMetric]
            The pairwise metrics.
        """
        return [metric for metric in self.metrics if isinstance(metric, StatefulMetric) and metric.is_pairwise()]

    def get_stateless_metrics(self) -> List[Any]:
        """
        Get stateless metrics.

        Returns
        -------
        List[Any]
            The stateless metrics.
        """
        return [metric for metric in self.metrics if not isinstance(metric, StatefulMetric)]

    def is_pairwise_evaluation(self) -> bool:
        """
        Check if the evaluation task is pairwise.

        Returns
        -------
        bool
            True if the task is pairwise, False otherwise.
        """
        return any(metric.is_pairwise() for metric in self.metrics if isinstance(metric, StatefulMetric))


def get_metrics(request: str | List[str | BaseMetric]) -> List[BaseMetric]:
    """
    Convert user requests into a list of metrics.

    Parameters
    ----------
    request : str | List[str]
        The user request. Right now, it only supports image generation quality.

    Returns
    -------
    List[BaseMetric]
        The list of metrics for the task.

    Raises
    ------
    NotImplementedError
        _description_
    ValueError
        _description_
    """
    if isinstance(request, List):
        if all(isinstance(item, BaseMetric) for item in request):
            pruna_logger.info("Using provided list of metric instances.")
            metrics: List[BaseMetric] = cast(List[BaseMetric], request)  # for mypy
            return metrics
        elif all(isinstance(item, str) for item in request):
            pruna_logger.info(f"Creating metrics from names: {request}")
            metric_names: List[str] = cast(List[str], request)
            return MetricRegistry.get_metrics(metric_names)
        else:
            pruna_logger.error("List must contain either all strings or all BaseMetric instances.")
            raise ValueError("List must contain either all strings or all BaseMetric instances.")
    else:
        if request == "image_generation_quality":
            pruna_logger.info("An evaluation task for image generation quality is being created.")
            return [
                TorchMetricWrapper("clip_score"),
                PairwiseClipScore(),
                TorchMetricWrapper("psnr"),
            ]
        else:
            pruna_logger.error(f"Metric {request} not found. Available requests: {AVAILABLE_REQUESTS}.")
            raise ValueError(f"Metric {request} not found. Available requests: {AVAILABLE_REQUESTS}.")
