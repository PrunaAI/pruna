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
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from pruna.engine.pruna_model import PrunaModel
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.logging.logger import pruna_logger


@MetricRegistry.register("elapsed_time")
class ElapsedTimeMetric(BaseMetric):
    """
    Time metrics for inference.

    Measures three key inference performance metrics:
    1. Elapsed Time: Total execution time in milliseconds
    2. Latency: Average processing time per batch in milliseconds
    3. Throughput: Number of batches processed per millisecond

    Parameters
    ----------
    n_iterations : int, default=100
        The number of batches to evaluate the model.
    n_warmup_iterations : int, default=10
        The number of warmup batches to evaluate the model.
    device : str | torch.device, default="cuda"
        The device to evaluate the model on.
    timing_type : str, default="sync"
        The type of timing to use.
    """

    def __init__(
        self,
        n_iterations: int = 100,
        n_warmup_iterations: int = 10,
        device: str | torch.device = "cuda",
        timing_type: str = "sync",
    ) -> None:
        super().__init__()
        self.n_iterations = n_iterations
        self.n_warmup_iterations = n_warmup_iterations
        self.device = device
        self.timing_type = timing_type

    @torch.no_grad()
    def compute(self, model: PrunaModel, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Compute the elapsed time, latency, and throughput for model inference.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        dataloader : DataLoader
            The dataloader to evaluate the model on.

        Returns
        -------
        dict
            The elapsed time, latency, and throughput for model inference.
        """
        if self.timing_type == "async" and self.device == "cpu":
            pruna_logger.warning("Async timing is not supported on CPU. Using sync timing instead.")

        model.set_to_eval()
        model.move_to_device(self.device)

        # Warmup
        c = 0
        while c < self.n_warmup_iterations:
            for batch in dataloader:
                batch = model.inference_handler.move_inputs_to_device(batch, self.device)
                x = model.inference_handler.prepare_inputs(batch)
                model(x)
                c += 1
                if c >= self.n_warmup_iterations:
                    break

        # Measurement
        list_elapsed_times = []
        c = 0
        while c < self.n_iterations:
            for batch in dataloader:
                batch = model.inference_handler.move_inputs_to_device(batch, self.device)
                x = model.inference_handler.prepare_inputs(batch)
                if self.timing_type == "async" or self.device == "cpu":
                    startevent_time = time.time()
                    _ = model(x)
                    endevent_time = time.time()
                    elapsed_time = (endevent_time - startevent_time) * 1000  # in ms
                elif self.timing_type == "sync":
                    startevent = torch.cuda.Event(enable_timing=True)
                    endevent = torch.cuda.Event(enable_timing=True)
                    startevent.record()
                    _ = model(x)
                    endevent.record()
                    torch.cuda.synchronize()
                    elapsed_time = startevent.elapsed_time(endevent)  # in ms
                else:
                    raise ValueError(f"Timing type {self.timing_type} not supported.")
                list_elapsed_times.append(elapsed_time)
                c += 1
                if c >= self.n_iterations:
                    break

        total_elapsed_time = sum(list_elapsed_times)
        batch_size = dataloader.batch_size

        return {
            f"inference_elapsed_time_ms_@{batch_size}": total_elapsed_time,
            f"inference_latency_ms_@{batch_size}": total_elapsed_time / self.n_iterations,
            f"inference_throughput_batches_per_ms_@{batch_size}": self.n_iterations / total_elapsed_time,
        }
