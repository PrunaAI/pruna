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

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Type

import pynvml
import torch
from torch.utils.data import DataLoader

from pruna.engine.pruna_model import PrunaModel
from pruna.engine.utils import safe_memory_cleanup, set_to_train
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.logging.logger import pruna_logger

VALID_MODES = ("disk", "inference", "training")


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
        Context manager to ensure pynvml is initialized andshut down properly.

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


@MetricRegistry.register("gpu_memory")
class GPUMemoryMetric(BaseMetric):
    """
    Evaluate the GPU memory usage of a model.

    Parameters
    ----------
    mode : str
        The mode for memory evaluation. Must be one of 'disk', 'inference', or 'training'.
    gpu_indices : Optional[List[int]]
        List of GPU indices to monitor. If None, all GPUs are assumed.
    """

    def __init__(self, mode: str = "disk", gpu_indices: Optional[List[int]] = None) -> None:
        """
        Initialize the GPUMemoryMetric.

        Raises
        ------
        ValueError
            If the provided mode is invalid.
        """
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"Mode must be one of {VALID_MODES}, got '{mode}'.")

        self.mode = mode
        self.gpu_indices = gpu_indices

    def compute(self, model: PrunaModel, dataloader: DataLoader) -> Dict[str, Any]:
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
        Dict[str, Any]
            The peak GPU memory usage in MB.
        """
        save_path = model.smash_config.cache_dir + "/metrics_save"
        model_cls = model.__class__
        model.save_pretrained(save_path)

        # results.update(stateless_metric.compute(save_path, self.task.dataloader, model.__class__))
        gpu_manager = GPUManager(self.gpu_indices)
        with gpu_manager.manage_resources():
            safe_memory_cleanup()
            memory_before_load = gpu_manager.get_memory_usage()

            # Load and prepare the model
            model = self._load_and_prepare_model(save_path, model_cls)

            memory_after_load = gpu_manager.get_memory_usage()

            utilized_gpus = self._detect_model_gpus(model) or gpu_manager.gpu_indices  # or [0]
            if not utilized_gpus:
                pruna_logger.warning("No GPUs found.")
                raise ValueError("No GPUs detected for the model.")

            pruna_logger.info(f"Utilized GPUs: {utilized_gpus}")

            # Perform forward pass if required by the mode
            if self.mode in {"inference", "training"}:
                self._perform_forward_pass(model, dataloader)

            memory_after_model_run = gpu_manager.get_memory_usage()
            # Tracking possible peak memory spike with torch
            memory_after_model_run_torch = torch.cuda.max_memory_allocated()  # TODO:test this

            pruna_logger.info(
                "Calculating memory usage...\n"
                "Current architecture assumes model parallelism, and sums memory usage of all GPUs."
            )
            before_sum = sum(memory_before_load[idx] for idx in utilized_gpus)
            load_sum = sum(memory_after_load[idx] for idx in utilized_gpus)
            after_sum = sum(memory_after_model_run[idx] for idx in utilized_gpus)

            peak_memory = (max(before_sum, load_sum, after_sum, memory_after_model_run_torch) - before_sum) / 1024**2

            safe_memory_cleanup()

            return {f"{self.mode}_memory": peak_memory}

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
            The list of GPU indices the model is using, or None if no GPUs detected.
        """
        device_map = getattr(model, attr_name, None)
        if not isinstance(device_map, dict):
            return None

        indices = set()
        for device in device_map.values():
            if isinstance(device, str) and "cuda" in device:
                try:
                    idx = int(device.split(":")[1])
                    indices.add(idx)
                except (IndexError, ValueError):
                    indices.add(0)

        return sorted(list(indices)) if indices else None

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
            if attr_name == "state_dict":
                tensors = model.state_dict().values()
            else:  # parameters
                tensors = model.parameters()

            indices = {
                tensor.device.index
                for tensor in tensors
                if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda"
            }
            return sorted(list(indices)) if indices else None
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
        if hasattr(device, "index") and device.type == "cuda":
            return [device.index]
        elif isinstance(device, str) and "cuda" in device:
            return [0]

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
        model.move_to_device("cuda")
        if self.mode in {"disk", "inference"}:
            model.set_to_eval()
        elif self.mode == "training":
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
        with torch.no_grad() if self.mode == "inference" else torch.enable_grad():
            batch = next(iter(dataloader))
            batch = model.inference_handler.move_inputs_to_device(batch, "cuda")
            inputs = model.inference_handler.prepare_inputs(batch)

            model(inputs)
