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

import torch

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.engine.utils import device_to_string, find_bytes_free_per_gpu, set_to_best_available_device
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.metric_cmmd import CMMD
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
    low_memory : bool, optional
        If True, we will run stateful metrics on cpu.
        If False, we will run stateful metrics on the best available device.
        Default is False.
    """

    def __init__(
        self,
        request: str | List[str | BaseMetric | StatefulMetric],
        datamodule: PrunaDataModule,
        device: str | torch.device | None = None,
        low_memory: bool = False,
    ) -> None:
        self.auto_device = device is None  # We set this for the compatibility checks later on.
        self.low_memory = low_memory
        self.device = set_to_best_available_device(
            device
        )  # The inference device is set as the task device for evaluation agent and optimization agent.
        self.stateful_metric_device = self._set_stateful_metric_device_from_task_device()
        self.metrics = _safe_build_metrics(request, self.device, self.stateful_metric_device)

        self.datamodule = datamodule
        self.dataloader = datamodule.test_dataloader()

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

    def _set_stateful_metric_device_from_task_device(self) -> str:
        """
        Return the device for stateful metrics based on the task device.

        Parameters
        ----------
        low_memory : bool
            If True, we will run stateful metrics on cpu.
            If False, we will run stateful metrics on the best available device.

        Returns
        -------
        str
            The device for the stateful metrics.
        """
        if self.low_memory:
            return "cpu"  # We will run stateful metrics on cpu
        elif self.device == "accelerate":
            bytes_free_per_gpu = find_bytes_free_per_gpu()
            return set_to_best_available_device("cuda", bytes_free_per_gpu)
        else:
            return self.device  # for when we pass a specific cuda device, or cpu or mps.


def _safe_build_metrics(
    request: str | List[str | BaseMetric | StatefulMetric], inference_device: str, stateful_metric_device: str
):
    try:
        return get_metrics(request, inference_device, stateful_metric_device)
    except torch.cuda.OutOfMemoryError as e:
        if stateful_metric_device == "cuda":
            pruna_logger.error(
                "Not enough GPU memory for metrics on %s. Please try initializing task with `low_memory=True`.",
                stateful_metric_device,
            )
        raise e


def get_metrics(
    request: str | List[str | BaseMetric | StatefulMetric], inference_device: str, stateful_metric_device: str
) -> List[BaseMetric | StatefulMetric]:
    """
    Convert user requests into a list of metrics.

    Parameters
    ----------
    request : str | List[str]
        The user request. Right now, it only supports image generation quality.
    inference_device : str | torch.device | None, optional
        The device to be used for inference, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    stateful_metric_device : str | torch.device | None, optional
        The device to be used for stateful metrics, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.

    Returns
    -------
    List[BaseMetric | StatefulMetric]
        The list of metrics for the task.
    """
    if isinstance(request, List):
        if all(isinstance(item, BaseMetric | StatefulMetric) for item in request):
            return _process_metric_instances(
                request=cast(List[BaseMetric | StatefulMetric], request),
                inference_device=inference_device,
                stateful_metric_device=stateful_metric_device,
            )
        elif all(isinstance(item, str) for item in request):
            return _process_metric_names(
                request=cast(List[str], request),
                inference_device=inference_device,
                stateful_metric_device=stateful_metric_device,
            )
        else:
            pruna_logger.error("List must contain either all strings or all [BaseMetric | StatefulMetric] instances.")
            raise ValueError("List must contain either all strings or all [BaseMetric | StatefulMetric] instances.")
    else:
        return _process_single_request(
            request=cast(str, request), inference_device=inference_device, stateful_metric_device=stateful_metric_device
        )


def _process_metric_instances(
    request: List[BaseMetric | StatefulMetric],
    inference_device: str,
    stateful_metric_device: str,
) -> List[BaseMetric | StatefulMetric]:
    pruna_logger.info("Using provided list of metric instances.")
    new_request_metrics: List[BaseMetric | StatefulMetric] = []
    for metric in request:
        if metric.__class__.__name__ in PARENT_METRICS:
            for child in metric.__class__.__subclasses__():
                child = cast(type[BaseMetric], child)
                hyperparameters = get_hyperparameters(metric, metric.__class__.__init__)
                if (
                    "device" in hyperparameters and hyperparameters["device"] != inference_device
                ):  # We check with inference device because everything in PARENT_METRICS is a BaseMetric.
                    pruna_logger.warning(
                        f"The task is requested to run on {inference_device}, but the metric {metric.metric_name} "
                        f"is configured to run on {hyperparameters['device']}."
                    )
                    pruna_logger.info(f"Setting the metric to run on {inference_device}.")
                    hyperparameters["device"] = inference_device
                new_request_metrics.append(MetricRegistry.get_metric(child.metric_name, **hyperparameters))
        else:
            if isinstance(metric, BaseMetric):
                metric.device = device_to_string(inference_device)
            else:
                metric.move_to_device(stateful_metric_device)
            new_request_metrics.append(cast(BaseMetric | StatefulMetric, metric))
    return new_request_metrics


def _process_metric_names(
    request: List[str], inference_device: str, stateful_metric_device: str
) -> List[BaseMetric | StatefulMetric]:
    pruna_logger.info(f"Creating metrics from names: {request}")
    new_requests: List[str] = []
    for metric_name in request:
        metric_name = cast(str, metric_name)
        new_requests.append(cast(str, metric_name))
    return MetricRegistry.get_metrics(
        names=new_requests, inference_device=inference_device, stateful_metric_device=stateful_metric_device
    )


def _process_single_request(
    request: str, stateful_metric_device: str, inference_device: str
) -> List[BaseMetric | StatefulMetric]:
    if request == "image_generation_quality":
        pruna_logger.info("An evaluation task for image generation quality is being created.")
        return [
            TorchMetricWrapper("clip_score", device=stateful_metric_device),
            TorchMetricWrapper("clip_score", call_type="pairwise", device=stateful_metric_device),
            CMMD(device=stateful_metric_device),
        ]
    else:
        pruna_logger.error(f"Metric {request} not found. Available requests: {AVAILABLE_REQUESTS}.")
        raise ValueError(f"Metric {request} not found. Available requests: {AVAILABLE_REQUESTS}.")
