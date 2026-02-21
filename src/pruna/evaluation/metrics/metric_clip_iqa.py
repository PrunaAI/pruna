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

"""
CLIP-IQA (Image Quality Assessment) Metric for Pruna.

This metric uses CLIP to assess image quality without reference images.
It evaluates various aspects of image quality like sharpness, brightness, etc.

Based on the InferBench implementation:
https://github.com/PrunaAI/InferBench
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

import torch
from torch import Tensor
from torchmetrics.multimodal import CLIPImageQualityAssessment

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import metric_data_processor
from pruna.logging.logger import pruna_logger


METRIC_CLIP_IQA = "clip_iqa"


@MetricRegistry.register(METRIC_CLIP_IQA)
class CLIPIQAMetric(StatefulMetric):
    """
    CLIP-based Image Quality Assessment (CLIP-IQA) metric.

    This metric uses CLIP to assess various aspects of image quality including:
    - Quality: Overall quality score
    - Brightness: Image brightness
    - Sharpness: Image sharpness
    - Colorfulness: Color vibrancy
    - Contrast: Image contrast

    Higher scores indicate better quality for the respective aspect.

    Parameters
    ----------
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    prompts : tuple of str, optional
        The prompts to use for quality assessment. Default is ("quality",).
    data_range : float, optional
        The data range of input images. Default is 255.0.
    call_type : str, optional
        The call type to use for the metric. Default is "y".

    References
    ----------
    CLIP-IQA: https://github.com/chaofengc/CLIP-IQA
    """

    total: torch.Tensor
    count: torch.Tensor
    higher_is_better: bool = True
    metric_name: str = METRIC_CLIP_IQA
    call_type: str = "y"

    def __init__(
        self,
        device: str | torch.device | None = None,
        prompts: tuple = ("quality",),
        data_range: float = 255.0,
        call_type: str = "y",
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        
        # Initialize CLIP-IQA metric
        self.metric = CLIPImageQualityAssessment(
            model_name_or_path="clip_iqa",
            data_range=data_range,
            prompts=prompts,
        )
        self.metric.to(self.device)
        
        self.call_type = call_type
        self.prompts = prompts
        
        # Add state for tracking scores
        self.add_state("total", torch.zeros(len(prompts)))
        self.add_state("count", torch.zeros(1))
        
        pruna_logger.info(f"Initialized CLIP-IQA metric with prompts: {prompts}")

    def _convert_to_tensor(self, images: Union[Tensor, List[Tensor]]) -> Tensor:
        """
        Convert input images to tensor format expected by CLIP-IQA.

        Parameters
        ----------
        images : Union[Tensor, List[Tensor]]
            The images to convert.

        Returns
        -------
        Tensor
            The converted tensor.
        """
        if isinstance(images, Tensor):
            # Handle tensor input
            if images.ndim == 3:
                images = images.unsqueeze(0)
            # Convert to float if needed
            if images.dtype == torch.uint8:
                images = images.float()
            return images
        elif isinstance(images, (list, tuple)):
            tensors = []
            for img in images:
                if isinstance(img, Tensor):
                    if img.ndim == 3:
                        img = img.unsqueeze(0)
                    if img.dtype == torch.uint8:
                        img = img.float()
                    tensors.append(img)
                else:
                    # PIL Image or other
                    import numpy as np
                    img_array = np.array(img)
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
                    tensors.append(img_tensor)
            return torch.cat(tensors, dim=0)
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")

    def update(
        self,
        x: Any,
        gt: Any,
        outputs: Union[Tensor, List[Tensor]],
    ) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : Any
            The input data (not used for CLIP-IQA).
        gt : Any
            The ground truth data (not used for CLIP-IQA).
        outputs : Union[Tensor, List[Tensor]]
            The generated images to evaluate.
        """
        # Get the generated images from outputs
        inputs = metric_data_processor(x, gt, outputs, self.call_type, self.device)
        images = inputs[0]  # Just use the outputs directly
        
        # Convert to tensor
        images_tensor = self._convert_to_tensor(images).to(self.device)
        
        # Compute CLIP-IQA scores
        with torch.no_grad():
            scores = self.metric(images_tensor)
            
            if scores.ndim == 0:
                # Single score
                scores = scores.unsqueeze(0)
            
            # Sum scores across batch
            self.total += scores.sum(dim=0)
            self.count += scores.shape[0]

    def compute(self) -> MetricResult:
        """
        Compute the CLIP-IQA score.

        Returns
        -------
        MetricResult
            The computed CLIP-IQA score(s).
        """
        if self.count.item() == 0:
            pruna_logger.warning("No samples to compute CLIP-IQA score")
            return MetricResult(self.metric_name, self.__dict__.copy(), 0.0)
        
        # Compute mean scores
        mean_scores = self.total / self.count
        
        # If single prompt, return scalar
        if len(self.prompts) == 1:
            result_value = mean_scores[0].item()
        else:
            # Return dict of scores for multiple prompts
            result_dict = {prompt: score.item() for prompt, score in zip(self.prompts, mean_scores)}
            result_value = result_dict
        
        return MetricResult(self.metric_name, self.__dict__.copy(), result_value)

    def move_to_device(self, device: str | torch.device) -> None:
        """
        Move the metric to a specific device.

        Parameters
        ----------
        device : str | torch.device
            The device to move the metric to.
        """
        if not self.is_device_supported(device):
            raise ValueError(
                f"Metric {self.metric_name} does not support device {device}. Must be one of {self.runs_on}."
            )
        self.device = set_to_best_available_device(device)
        self.metric = self.metric.to(device)
