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

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as transforms_functional
from torch import Tensor

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_SHARPNESS = "sharpness"


@MetricRegistry.register(METRIC_SHARPNESS)
class SharpnessMetric(StatefulMetric):
    """
    Laplacian‑variance image‑sharpness metric.

    Parameters
    ----------
    *args : Any
        Additional arguments.
    kernel_size : int
        The size of the kernel for the Laplacian operator. Default is 3.
    call_type : str
        The type of call to use for the metric. IQA metrics are only supported for single mode.
    **kwargs : Any
        Additional keyword arguments.
    """

    scores: List[float]
    default_call_type: str = "y"  # Blind IQA metric
    higher_is_better: bool = True
    metric_name: str = METRIC_SHARPNESS
    runs_on: List[str] = ["cpu", "cuda"]

    def __init__(self, *args, kernel_size: int = 3, call_type: str = SINGLE, **kwargs) -> None:
        device = kwargs.pop("device", None)
        if device is not None and device not in self.runs_on:
            pruna_logger.error(f"SharpnessMetric: device {device} not supported. Supported devices: {self.runs_on}")
            raise
        super().__init__(*args, **kwargs)
        self.device = set_to_best_available_device(device)  # OpenCV only works on CPU
        self.kernel_size = kernel_size
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    @torch.no_grad()
    def update(self, x: List[Any] | Tensor, gt: List[Any] | Tensor, outputs: Any) -> None:
        """
        Accumulate the sharpness scores for each batch.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data.
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images.
        """
        images_list = metric_data_processor(x, gt, outputs, self.call_type)
        images = cast(Tensor, images_list[0])  # Since the processor returns the batch wrapped in a list

        # Batchify if single image
        if images.ndim == 3:
            images = images.unsqueeze(0)

        if images.ndim != 4:
            pruna_logger.error(f"Expected 4‑D tensor (B, C, H, W); got shape {tuple(images.shape)}")
            raise

        # Move to CPU OpenCV only works on numpy
        imgs = images.detach().cpu()

        # If range is 0‑1 → scale to 0‑255 for uint8
        if imgs.max() <= 1.0:
            imgs = imgs * 255.0
        imgs = imgs.to(torch.uint8)

        for img in imgs:
            if img.shape[0] == 3:
                img_gray = transforms_functional.rgb_to_grayscale(img.float()).squeeze(0).numpy().astype(np.uint8)
            elif img.shape[0] == 1:
                img_gray = img.squeeze(0).numpy().astype(np.uint8)
            else:
                pruna_logger.error("SharpnessMetric: unsupported channel count")
                raise

            lap = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=self.kernel_size)
            sharp = float(lap.var())
            self.scores.append(sharp)

    def compute(self) -> MetricResult:
        """
        Compute the mean sharpness score.

        Returns
        -------
        MetricResult
            The sharpness metric result.
        """
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)

        mean_val = float(np.mean(self.scores))
        return MetricResult(self.metric_name, self.__dict__, mean_val)
