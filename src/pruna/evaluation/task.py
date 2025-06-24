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

from typing import Any, List, cast
from warnings import warn

import torch

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.metric_cmmd import CMMD
from pruna.evaluation.metrics.metric_elapsed_time import LATENCY, THROUGHPUT, TOTAL_TIME
from pruna.evaluation.metrics.metric_energy import CO2_EMISSIONS, ENERGY_CONSUMED
from pruna.evaluation.metrics.metric_memory import DISK_MEMORY
from pruna.evaluation.metrics.metric_model_architecture import TOTAL_MACS, TOTAL_PARAMS
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.metric_torch import TorchMetricWrapper
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.utils import get_hyperparameters
from pruna.logging.logger import pruna_logger

AVAILABLE_REQUESTS = ("image_generation_quality",)
PARENT_METRICS = (
    "ModelArchitectureStats",
    "InferenceTimeStats",
    "EnvironmentalImpactStats",
)
DEPRECATION_TO_NEW_MAP = {
    "elapsed_time": [LATENCY, THROUGHPUT, TOTAL_TIME],
    "gpu_memory": [DISK_MEMORY],
    "energy": [ENERGY_CONSUMED, CO2_EMISSIONS],
    "model_architecture": [TOTAL_MACS, TOTAL_PARAMS],
}


class Task:
    """
    Processes user requests and converts them into a format that the evaluation module can handle.

    Parameters
    ----------
    request : str | List[str | BaseMetric | StatefulMetric]
        The user request.
    datamodule : PrunaDataModule
        The dataloader to use for the evaluation.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    """

    def __init__(
        self,
        request: str | List[str | BaseMetric | StatefulMetric],
        datamodule: PrunaDataModule,
        device: str | torch.device | None = None,
    ) -> None:
        device = set_to_best_available_device(device)
        self.metrics = get_metrics(request, device)
        self.datamodule = datamodule
        self.dataloader = datamodule.test_dataloader()
        if device not in ["cuda", "cpu", "mps"]:
            raise ValueError(f"Unsupported device: {device}. Must be one of: cuda, cpu, mps.")
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


def get_metrics(
    request: str | List[str | BaseMetric | StatefulMetric], device: str | torch.device | None = None
) -> List[BaseMetric | StatefulMetric]:
    """
    Convert user requests into a list of metrics.

    Parameters
    ----------
    request : str | List[str]
        The user request. Right now, it only supports image generation quality.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.

    Returns
    -------
    List[BaseMetric | StatefulMetric]
        The list of metrics for the task.

    Raises
    ------
    NotImplementedError
        _description_
    ValueError
        _description_
    """
    if isinstance(request, List):
        if all(isinstance(item, BaseMetric | StatefulMetric) for item in request):
            return _process_metric_instances(request=cast(List[BaseMetric | StatefulMetric], request))
        elif all(isinstance(item, str) for item in request):
            return _process_metric_names(request=cast(List[str], request), device=device)
        else:
            pruna_logger.error("List must contain either all strings or all [BaseMetric | StatefulMetric] instances.")
            raise ValueError("List must contain either all strings or all [BaseMetric | StatefulMetric] instances.")
    else:
        return _process_single_request(request, device)


def _process_metric_instances(request: List[BaseMetric | StatefulMetric]) -> List[BaseMetric | StatefulMetric]:
    pruna_logger.info("Using provided list of metric instances.")
    new_request_metrics: List[BaseMetric | StatefulMetric] = []
    for metric in request:
        if metric.__class__.__name__ in PARENT_METRICS:
            for child in metric.__class__.__subclasses__():
                child = cast(type[BaseMetric], child)
                hyperparameters = get_hyperparameters(metric, metric.__class__.__init__)
                new_request_metrics.append(MetricRegistry.get_metric(child.metric_name, **hyperparameters))
        else:
            new_request_metrics.append(cast(BaseMetric | StatefulMetric, metric))
    return new_request_metrics


def _process_metric_names(request: List[str], device: str | torch.device | None) -> List[BaseMetric | StatefulMetric]:
    pruna_logger.info(f"Creating metrics from names: {request}")
    new_requests: List[str] = []
    for metric_name in request:
        metric_name = cast(str, metric_name)
        if metric_name in DEPRECATION_TO_NEW_MAP:
            warn(
                f"Metric {metric_name} is deprecated and will be removed in 'v0.2.8' release. "
                f"Use {DEPRECATION_TO_NEW_MAP[metric_name]} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            for new_metric in DEPRECATION_TO_NEW_MAP[metric_name]:
                new_requests.append(cast(str, new_metric))
        else:
            new_requests.append(cast(str, metric_name))
    return MetricRegistry.get_metrics(names=new_requests, device=device)


def _process_single_request(request: str, device: str | torch.device | None) -> List[BaseMetric | StatefulMetric]:
    if request == "image_generation_quality":
        pruna_logger.info("An evaluation task for image generation quality is being created.")
        return [
            TorchMetricWrapper("clip_score"),
            TorchMetricWrapper("clip_score", call_type="pairwise"),
            CMMD(device=device),
        ]
    else:
        pruna_logger.error(f"Metric {request} not found. Available requests: {AVAILABLE_REQUESTS}.")
        raise ValueError(f"Metric {request} not found. Available requests: {AVAILABLE_REQUESTS}.")
