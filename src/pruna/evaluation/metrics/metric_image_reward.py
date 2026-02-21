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
Image Reward Metric for Pruna.

This metric computes image reward scores using the ImageReward library.

Based on the InferBench implementation:
https://github.com/PrunaAI/InferBench
"""

from __future__ import annotations

from typing import Any, List

import torch
from PIL import Image

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_IMAGE_REWARD = "image_reward"


@MetricRegistry.register(METRIC_IMAGE_REWARD)
class ImageRewardMetric(StatefulMetric):
    """
    Image Reward metric for evaluating image-text alignment.

    This metric uses the ImageReward model to compute how well generated images
    match their text prompts based on learned human preferences.
    Higher scores indicate better alignment.

    Parameters
    ----------
    *args : Any
        Additional arguments to pass to the StatefulMetric constructor.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    model_name : str, optional
        The ImageReward model to use. Default is "ImageReward-v1.0".
    **kwargs : Any
        Additional keyword arguments to pass to the StatefulMetric constructor.

    References
    ----------
    ImageReward: https://github.com/thaosu/ImageReward
    """

    total: torch.Tensor
    count: torch.Tensor
    call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = METRIC_IMAGE_REWARD

    def __init__(
        self,
        *args,
        device: str | torch.device | None = None,
        model_name: str = "ImageReward-v1.0",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.device = set_to_best_available_device(device)
        self.model_name = model_name

        # Import ImageReward lazily
        try:
            import ImageReward as RM
        except ImportError:
            pruna_logger.error("ImageReward is not installed. Install with: pip install ImageReward")
            raise

        self.model = RM.load(self.model_name, device=str(self.device))

        self.add_state("total", torch.zeros(1))
        self.add_state("count", torch.zeros(1))

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        This computes the ImageReward scores for the given images and prompts.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (text prompts).
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images to score.
        """
        # Get images and prompts
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = inputs[0]  # Generated images
        prompts = x if isinstance(x, list) else [""] * len(images)

        with torch.no_grad():
            for i, image in enumerate(images):
                # Convert tensor to PIL Image if needed
                if isinstance(image, torch.Tensor):
                    image = self._tensor_to_pil(image)

                prompt = prompts[i] if i < len(prompts) else ""

                score = self.model.score(prompt, image)
                self.total += score
                self.count += 1

    def compute(self) -> MetricResult:
        """
        Compute the average ImageReward metric based on previous updates.

        Returns
        -------
        MetricResult
            The average ImageReward metric.
        """
        result = self.total / self.count if self.count.item() != 0 else torch.zeros(1)
        return MetricResult(self.metric_name, self.__dict__.copy(), result.item())

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert a tensor to a PIL Image.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to convert. Expected shape: (C, H, W) or (B, C, H, W).

        Returns
        -------
        Image.Image
            The converted PIL Image.
        """
        # Handle batch dimension
        if tensor.ndim == 4:
            tensor = tensor[0]

        # Ensure values are in [0, 1]
        if tensor.max() > 1:
            tensor = tensor / 255.0

        # Convert to numpy and then to PIL
        numpy_image = tensor.cpu().numpy()
        numpy_image = (numpy_image * 255).astype("uint8")
        return Image.fromarray(numpy_image.transpose(1, 2, 0))
