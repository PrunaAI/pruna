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

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Type, cast
from warnings import warn

import pynvml
import torch
from torch.utils.data import DataLoader

from pruna.engine.pruna_model import PrunaModel
from pruna.engine.utils import safe_memory_cleanup, set_to_best_available_device, set_to_train
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.logging.logger import pruna_logger

DISK_MEMORY = "disk_memory"
INFERENCE_MEMORY = "inference_memory"
TRAINING_MEMORY = "training_memory"
VALID_MODES = (DISK_MEMORY, INFERENCE_MEMORY, TRAINING_MEMORY)
DEPRECATED_MODES = ("disk", "inference", "training")
DEPRECATED_MODES_MAP = {"disk": DISK_MEMORY, "inference": INFERENCE_MEMORY, "training": TRAINING_MEMORY}
MEMORY_UNITS = "MB"


class GPUManager:
    """
    A manager class to handle GPU interactions using pynvml.

    Parameters
    ----------
    gpu_indices : Optional[List[int]]
        List of GPU indices to manage. If None, single GPU is assumed.
    """

    def __init__(self, gpu_indices: Optional[List[int]] = None) -> None:
        """Initialize the GPUManager."""
        self.device_count = 0
        self.gpu_indices = gpu_indices
        self.handles: List[Any] = []

    @contextmanager
    def manage_resources(self) -> Generator[None, None, None]:
        """
        Context manager to ensure pynvml is initialized and shut down properly.

        Yields
        ------
        None
        """
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            # if gpu_indices is None, monitor all GPUs
            self.gpu_indices = self.gpu_indices or list(range(self.device_count))
            for idx in self.gpu_indices:
                if idx >= self.device_count:
                    raise ValueError(f"GPU index {idx} is out of range. Only {self.device_count} GPUs available.")
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self.handles.append(handle)
            yield
        finally:
            pynvml.nvmlShutdown()

    def get_memory_usage(self) -> Dict[int, int]:
        """
        Get the current memory usage for each managed GPU.

        Returns
        -------
        Dict[int, int]
            Dictionary of memory usage in bytes for each GPU.
        """
        if self.gpu_indices is None or self.handles is None:
            raise ValueError("GPU indices and handles must be initialized")
        memory_usages = {}
        for idx, handle in zip(self.gpu_indices, self.handles):
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_usages[idx] = mem_info.used
        return memory_usages


class GPUMemoryStats(BaseMetric):
    """
    Evaluate the GPU memory usage of a model.

    Parameters
    ----------
    mode : str
        The mode for memory evaluation. Must be one of 'disk_memory', 'inference_memory', or 'training_memory'.
    gpu_indices : Optional[List[int]]
        List of GPU indices to monitor. If None, all GPUs are assumed.
    """

    def __init__(self, mode: str = DISK_MEMORY, gpu_indices: Optional[List[int]] = None) -> None:
        """
        Initialize the GPUMemoryStats.

        Raises
        ------
        ValueError
            If the provided mode is invalid.
        """
        if mode not in VALID_MODES:
            if mode in DEPRECATED_MODES:
                warn(
                    "GPUMemoryStats is no longer the preferred interface for GPU memory evaluation. "
                    "Please use 'DiskMemoryMetric', 'InferenceMemoryMetric' or 'TrainingMemoryMetric' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                warn(
                    f"Mode '{mode}' is deprecated and will be removed in 'v0.2.8' release. "
                    f"Please use '{DEPRECATED_MODES_MAP[mode]}' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                mode = DEPRECATED_MODES_MAP[mode]
            else:
                raise ValueError(f"Mode must be one of {VALID_MODES}, got '{mode}'.")

        self.mode = mode
        self.gpu_indices = gpu_indices
        self.device_map = None

    def compute(self, model: PrunaModel, dataloader: DataLoader) -> MetricResult:
        """
        Compute the peak GPU memory usage of the model.

        Parameters
        ----------
        model : PrunaModel
            The model instance.
        dataloader : DataLoader
            The DataLoader for model evaluation.

        Returns
        -------
        MetricResult
            The peak GPU memory usage in MB.
        """
        save_path = model.smash_config.cache_dir + "/metrics_save"
        model_cls = model.__class__
        model_device_indices = self._detect_model_gpus(model)
        if not model_device_indices:
            pruna_logger.warning("No GPUs found.")
            raise ValueError("No GPUs detected for the model. Memory metric is designed to measure the GPU usage.")
        model.save_pretrained(save_path)
        model.move_to_device("cpu")

        gpu_manager = GPUManager(self.gpu_indices)
        with gpu_manager.manage_resources():
            safe_memory_cleanup()
            memory_before_load = gpu_manager.get_memory_usage()

            # Load and prepare the model
            metric_model = self._load_and_prepare_model(save_path, model_cls)

            memory_after_load = gpu_manager.get_memory_usage()

            utilized_gpus = cast(List[int], self._detect_model_gpus(metric_model) or gpu_manager.gpu_indices)
            pruna_logger.info(f"Utilized GPUs: {utilized_gpus}")

            # Perform forward pass if required by the mode
            if self.mode in {INFERENCE_MEMORY, TRAINING_MEMORY}:
                self._perform_forward_pass(model, dataloader)

            memory_after_model_run = gpu_manager.get_memory_usage()

            pruna_logger.info(
                "Calculating memory usage...\n"
                "Current architecture assumes model parallelism, and sums memory usage of all GPUs."
            )
            before_sum = sum(memory_before_load[idx] for idx in utilized_gpus)
            load_sum = sum(memory_after_load[idx] for idx in utilized_gpus)
            after_sum = sum(memory_after_model_run[idx] for idx in utilized_gpus)

            peak_memory = (max(before_sum, load_sum, after_sum) - before_sum) / 1024**2

            safe_memory_cleanup()

        if len(model_device_indices) > 1 and self.device_map is None:
            pruna_logger.error("Multiple GPUs detected, but no device map found. Please check the model configuration.")
            raise
        else:
            model.move_to_device("cuda")
        return MetricResult(self.mode, self.__dict__.copy(), peak_memory)

    def _detect_model_gpus(self, model: PrunaModel) -> List[int]:
        """
        Detect which GPUs the model is utilizing.

        Parameters
        ----------
        model : PrunaModel
            The model instance.

        Returns
        -------
        List[int]
            List of GPU indices the model is using, or None if no GPUs detected.
        """
        # Check various device mapping attributes
        for device_map_attr in ["device_map", "hf_device_map"]:
            if indices := self._check_device_map(model, device_map_attr):
                return indices

        # Check model tensors and parameters
        for tensor_attr in ["state_dict", "parameters"]:
            if indices := self._check_tensor_locations(model, tensor_attr):
                return indices

        # Check model device attribute
        if indices := self._check_model_device(model):
            return indices

        pruna_logger.warning("No GPUs detected for the model. Will attempt to measure memory usage for all GPUs.")
        return []

    def _check_device_map(self, model: PrunaModel, attr_name: str) -> Optional[List[int]]:
        """
        Extract GPU indices from a device map attribute.

        Parameters
        ----------
        model : PrunaModel
            The model instance.
        attr_name : str
            The name of the attribute to check.

        Returns
        -------
        Optional[List[int]]
            The list of GPU indices the model is using.
        """
        device_map = getattr(model, attr_name, None)
        self.device_map = device_map
        if not isinstance(device_map, dict):
            return None

        indices = set()
        for device in device_map.values():
            if isinstance(device, str):
                if "cuda" in device:
                    try:
                        idx = int(device.split(":")[1])
                        indices.add(idx)
                    except (IndexError, ValueError):
                        indices.add(0)
                elif "mps" in device:
                    indices.add(-1)  # Use -1 as a placeholder for MPS

        return sorted(indices) if indices else None

    def _check_tensor_locations(self, model: PrunaModel, attr_name: str) -> Optional[List[int]]:
        """
        Extract GPU indices from model tensor locations.

        Parameters
        ----------
        model : PrunaModel
            The model instance.
        attr_name : str
            The name of the attribute to check.

        Returns
        -------
        Optional[List[int]]
            The list of GPU indices the model is using, or None if no GPUs detected.
        """
        try:
            tensors = model.state_dict().values() if attr_name == "state_dict" else model.parameters()
            indices = {
                tensor.device.index
                for tensor in tensors
                if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda"
            }
            return sorted(indices) if indices else None
        except Exception:
            return None

    def _check_model_device(self, model: PrunaModel) -> Optional[List[int]]:
        """
        Extract GPU index from model's device attribute.

        Parameters
        ----------
        model : PrunaModel
            The model instance.

        Returns
        -------
        Optional[List[int]]
            The list of GPU indices the model is using, or None if no GPUs detected.
        """
        if not hasattr(model, "device"):
            return None

        device = model.device
        if hasattr(device, "type"):
            if device.type == "cuda":
                return [device.index if device.index is not None else 0]
            elif device.type == "mps":
                return [-1]  # Placeholder for MPS

        elif isinstance(device, str):
            if "cuda" in device:
                return [0]
            elif "mps" in device:
                return [-1]

        return None

    def _load_and_prepare_model(self, model_path: str, model_cls: Type[PrunaModel]) -> PrunaModel:
        """
        Load the model and set it to the appropriate mode.

        Parameters
        ----------
        model_path : str
            The path to the pretrained model.
        model_cls : Type[PrunaModel]
            The class of the model to load.

        Returns
        -------
        PrunaModel
            The loaded and prepared model.
        """
        model = model_cls.from_pretrained(
            model_path,
        )
        model.move_to_device(set_to_best_available_device(None))
        if self.mode in {DISK_MEMORY, INFERENCE_MEMORY}:
            model.set_to_eval()
        elif self.mode == TRAINING_MEMORY:
            set_to_train(model)
        return model

    def _perform_forward_pass(self, model: PrunaModel, dataloader: DataLoader) -> None:
        """
        Perform a single forward pass through the model.

        Parameters
        ----------
        model : PrunaModel
            The model instance.
        dataloader : DataLoader
            The DataLoader for model evaluation.
        """
        with torch.no_grad() if self.mode == INFERENCE_MEMORY else torch.enable_grad():
            batch = next(iter(dataloader))
            model.run_inference(batch=batch, device=set_to_best_available_device(None))


@MetricRegistry.register(DISK_MEMORY)
class DiskMemoryMetric(BaseMetric):
    """
    Wrapper over GPUMemoryStats with disk memory as primary metric.

    Parameters
    ----------
    gpu_indices : Optional[List[int]]
        The GPU indices to monitor.
    """

    metric_name: str = DISK_MEMORY
    metric_units: str = MEMORY_UNITS
    higher_is_better: bool = False

    def __init__(self, gpu_indices: Optional[List[int]] = None) -> None:
        """Initialize the DiskMemoryMetric."""
        self.metric = GPUMemoryStats(DISK_MEMORY, gpu_indices)

    def compute(self, model: PrunaModel, dataloader: DataLoader) -> MetricResult:
        """
        Compute the disk memory usage of the model.

        Parameters
        ----------
        model : PrunaModel
            The model instance.
        dataloader : DataLoader
            The DataLoader for model evaluation.

        Returns
        -------
        MetricResult
            The disk memory usage of the model.
        """
        return self.metric.compute(model, dataloader)


@MetricRegistry.register(INFERENCE_MEMORY)
class InferenceMemoryMetric(BaseMetric):
    """
    Wrapper over GPUMemoryStats with inference memory as primary metric.

    Parameters
    ----------
    gpu_indices : Optional[List[int]]
        The GPU indices to monitor.
    """

    metric_name: str = INFERENCE_MEMORY
    metric_units: str = MEMORY_UNITS
    higher_is_better: bool = False

    def __init__(self, gpu_indices: Optional[List[int]] = None) -> None:
        """Initialize the InferenceMemoryMetric."""
        self.metric = GPUMemoryStats(INFERENCE_MEMORY, gpu_indices)

    def compute(self, model: PrunaModel, dataloader: DataLoader) -> MetricResult:
        """
        Compute the inference memory usage of the model.

        Parameters
        ----------
        model : PrunaModel
            The model instance.
        dataloader : DataLoader
            The DataLoader for model evaluation.

        Returns
        -------
        MetricResult
            The inference memory usage of the model.
        """
        return self.metric.compute(model, dataloader)


@MetricRegistry.register(TRAINING_MEMORY)
class TrainingMemoryMetric(BaseMetric):
    """
    Wrapper over GPUMemoryStats with training memory as primary metric.

    Parameters
    ----------
    gpu_indices : Optional[List[int]]
        The GPU indices to monitor.
    """

    metric_name: str = TRAINING_MEMORY
    metric_units: str = MEMORY_UNITS
    higher_is_better: bool = False

    def __init__(self, gpu_indices: Optional[List[int]] = None) -> None:
        """Initialize the TrainingMemoryMetric."""
        self.metric = GPUMemoryStats(TRAINING_MEMORY, gpu_indices)

    def compute(self, model: PrunaModel, dataloader: DataLoader) -> MetricResult:
        """
        Compute the training memory usage of the model.

        Parameters
        ----------
        model : PrunaModel
            The model instance.
        dataloader : DataLoader
            The DataLoader for model evaluation.

        Returns
        -------
        MetricResult
            The training memory usage of the model.
        """
        return self.metric.compute(model, dataloader)


class GPUMemoryMetric:
    """
    Deprecated class.

    Parameters
    ----------
    *args : Any
        Arguments for GPUMemoryStats.
    **kwargs : Any
        Keyword arguments for GPUMemoryStats.
    """

    def __new__(cls, *args, **kwargs):
        """Forwards to GPUMemoryStats."""
        warn(
            "GPUMemoryMetric is deprecated and will be removed in 'v0.2.8' release. \n "
            "It has been replaced by GPUMemoryStats, "
            "which is a shared parent class for 'DiskMemoryMetric', 'InferenceMemoryMetric' and 'TrainingMemoryMetric'. \n"  # noqa: E501
            "In the future, please use 'DiskMemoryMetric', 'InferenceMemoryMetric' or 'TrainingMemoryMetric' instead.\n",
            DeprecationWarning,
            stacklevel=2,
        )
        return GPUMemoryStats(*args, **kwargs)
