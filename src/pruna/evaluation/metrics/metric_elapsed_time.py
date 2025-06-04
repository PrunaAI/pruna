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

import time
from typing import Any, Dict, cast
from warnings import warn

import torch
from torch.utils.data import DataLoader

from pruna.engine.pruna_model import PrunaModel
from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.logging.logger import pruna_logger

LATENCY = "latency"
THROUGHPUT = "throughput"
TOTAL_TIME = "total_time"


class InferenceTimeStats(BaseMetric):
    """
    Internal metric for shared inference time computation.

    This class is not intended for direct use by end users. Instead, it serves as a shared internal utility
    to compute timing-related statistics (latency, throughput, and total inference time) used by its
    specialized child metrics: `LatencyMetric`, `ThroughputMetric`, and `TotalTimeMetric`.

    The metric performs warmup and timed inference iterations to calculate:
    1. Total Time: Total execution time across all timed iterations (in milliseconds)
    2. Latency: Average processing time per batch (in milliseconds)
    3. Throughput: Number of samples processed per millisecond

    These values are returned as raw results for consumption by child metrics, which expose them through
    a standardized `MetricResult` interface.

    Parameters
    ----------
    n_iterations : int, default=100
        The number of batches to evaluate the model.
    n_warmup_iterations : int, default=10
        The number of warmup batches to evaluate the model.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    timing_type : str, default="sync"
        The type of timing to use.
    """

    def __init__(
        self,
        n_iterations: int = 100,
        n_warmup_iterations: int = 10,
        device: str | torch.device | None = None,
        timing_type: str = "sync",
    ) -> None:
        self.n_iterations = n_iterations
        self.n_warmup_iterations = n_warmup_iterations
        self.device = set_to_best_available_device(device)
        self.timing_type = timing_type

    def _measure(self, model: PrunaModel, dataloader: DataLoader, iterations: int, measure_fn) -> None:
        """Perform iterations and apply the measure function.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        dataloader : DataLoader
            The dataloader to evaluate the model on.
        iterations : int
            The number of iterations to run inference.
        measure_fn : Callable
            The function to apply to each batch.
        """
        c = 0
        while c < iterations:
            for batch in dataloader:
                batch = model.inference_handler.move_inputs_to_device(batch, self.device)
                x = model.inference_handler.prepare_inputs(batch)
                measure_fn(model, x)
                c += 1
                if c >= iterations:
                    break

    def _time_inference(self, model: PrunaModel, x: Any) -> float:
        """
        Time a single inference and return the elapsed time in milliseconds.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        x : Any
            The input to the model.

        Returns
        -------
        float
            The elapsed time in milliseconds.
        """
        if self.timing_type == "async" or self.device == "cpu":
            startevent_time = time.time()
            model.run_inference(x)
            endevent_time = time.time()
            return (endevent_time - startevent_time) * 1000  # in ms
        elif self.timing_type == "sync":
            try:
                torch_device_attr = getattr(torch, self.device)
            except AttributeError:
                raise ValueError(f"Device {self.device} not supported for sync timing. Using async timing instead.")
            startevent = torch_device_attr.Event(enable_timing=True)
            endevent = torch_device_attr.Event(enable_timing=True)
            startevent.record()
            if isinstance(x, dict):
                _ = model(**x, **model.inference_handler.model_args)
            else:
                _ = model(x, **model.inference_handler.model_args)
            endevent.record()
            torch_device_attr.synchronize()
            return startevent.elapsed_time(endevent)  # in ms
        else:
            raise ValueError(f"Timing type {self.timing_type} not supported.")

    @torch.no_grad()
    def compute(self, model: PrunaModel, dataloader: DataLoader) -> Dict[str, Any] | MetricResult:
        """
        Compute the inference time for model inference.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        dataloader : DataLoader
            The dataloader to evaluate the model on.

        Returns
        -------
        Dict[str, Any] | MetricResult
            The inference time for model inference.
        """
        if self.timing_type == "async" and self.device == "cpu":
            pruna_logger.warning("Async timing is not supported on CPU. Using sync timing instead.")

        model.set_to_eval()
        model.move_to_device(self.device)

        # Warmup
        self._measure(
            model,
            dataloader,
            self.n_warmup_iterations,
            lambda m, x: (
                m(**x, **m.inference_handler.model_args)  # x is a dict
                if isinstance(x, dict)
                else m(x, **m.inference_handler.model_args)  # x is tensor/list
            ),
        )

        # Measurement
        list_elapsed_times = []
        self._measure(
            model, dataloader, self.n_iterations, lambda m, x: list_elapsed_times.append(self._time_inference(m, x))
        )

        total_elapsed_time = sum(list_elapsed_times)
        self.batch_size = cast(int, dataloader.batch_size)

        raw_results = {
            TOTAL_TIME: total_elapsed_time,
            LATENCY: total_elapsed_time / self.n_iterations,
            THROUGHPUT: self.n_iterations * self.batch_size / total_elapsed_time,
        }

        return cast(Dict[str, Any], raw_results)


@MetricRegistry.register(LATENCY)
class LatencyMetric(InferenceTimeStats):
    """
    View over InferenceTimeStats with latency as primary metric.

    Parameters
    ----------
    n_iterations : int, default=100
        The number of batches to evaluate the model.
    n_warmup_iterations : int, default=10
        The number of warmup batches to evaluate the model.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    timing_type : str, default="sync"
        The type of timing to use.
    """

    higher_is_better: bool = False
    metric_name: str = LATENCY
    metric_units: str = "ms/num_iterations"

    def compute(self, model: PrunaModel, dataloader: DataLoader) -> MetricResult:
        """
        Compute the latency for model inference.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        dataloader : DataLoader
            The dataloader to evaluate the model on.

        Returns
        -------
        MetricResult
            The latency for model inference.
        """
        # Note: This runs separate inference if called directly.
        # Use EvaluationAgent to share computation across time metrics.
        raw_results = super().compute(model, dataloader)
        result = cast(Dict[str, Any], raw_results)[self.metric_name]
        return MetricResult(self.metric_name, self.__dict__.copy(), result)


@MetricRegistry.register(THROUGHPUT)
class ThroughputMetric(InferenceTimeStats):
    """
    View over InferenceTimeStats with throughput as primary metric.

    Parameters
    ----------
    n_iterations : int, default=100
        The number of batches to evaluate the model.
    n_warmup_iterations : int, default=10
        The number of warmup batches to evaluate the model.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    timing_type : str, default="sync"
        The type of timing to use.
    """

    higher_is_better: bool = True
    metric_units: str = "num_iterations/ms"
    metric_name: str = THROUGHPUT

    def compute(self, model: PrunaModel, dataloader: DataLoader) -> MetricResult:
        """
        Compute the throughput for model inference.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        dataloader : DataLoader
            The dataloader to evaluate the model on.

        Returns
        -------
        MetricResult
            The throughput for model inference.
        """
        # Note: This runs separate inference if called directly.
        # Use EvaluationAgent to share computation across time metrics.
        raw_results = super().compute(model, dataloader)
        result = cast(Dict[str, Any], raw_results)[self.metric_name]
        return MetricResult(self.metric_name, self.__dict__.copy(), result)


@MetricRegistry.register(TOTAL_TIME)
class TotalTimeMetric(InferenceTimeStats):
    """
    View over InferenceTimeStats with elapsed time as primary metric.

    Parameters
    ----------
    n_iterations : int, default=100
        The number of batches to evaluate the model.
    n_warmup_iterations : int, default=10
        The number of warmup batches to evaluate the model.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    timing_type : str, default="sync"
        The type of timing to use.
    """

    higher_is_better: bool = False
    metric_units: str = "ms"
    metric_name: str = TOTAL_TIME

    def compute(self, model: PrunaModel, dataloader: DataLoader) -> MetricResult:
        """
        Compute the total time for model inference.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        dataloader : DataLoader
            The dataloader to evaluate the model on.

        Returns
        -------
        MetricResult
            The total time for model inference.
        """
        # Note: This runs separate inference if called directly.
        # Use EvaluationAgent to share computation across time metrics.
        raw_results = super().compute(model, dataloader)
        result = cast(Dict[str, Any], raw_results)[self.metric_name]
        return MetricResult(self.metric_name, self.__dict__.copy(), result)


class ElapsedTimeMetric:
    """
    Deprecated class.

    Parameters
    ----------
    *args : Any
        Arguments for InferenceTimeStats.
    **kwargs : Any
        Keyword arguments for InferenceTimeStats.
    """

    def __new__(cls, *args, **kwargs):
        """Forwards to InferenceTimeStats."""
        warn(
            "Class ElapsedTimeMetric is deprecated and will be removed in 'v0.2.8' release. \n"
            "It has been replaced by InferenceTimeStats, \n"
            "which is a shared parent class for 'LatencyMetric', 'ThroughputMetric' and 'TotalTimeMetric'. \n"
            "In the future please use 'LatencyMetric', 'ThroughputMetric' or 'TotalTimeMetric' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return InferenceTimeStats(*args, **kwargs)
