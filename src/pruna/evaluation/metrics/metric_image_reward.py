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
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_IMAGE_REWARD = "image_reward"


@MetricRegistry.register(METRIC_IMAGE_REWARD)
class ImageRewardMetric(StatefulMetric):
    """
    Image Reward metric for evaluating image-text alignment.

    This metric uses the ImageReward model to compute how well generated images
    match their text prompts based on learned human preferences.
    Higher scores indicate better alignment.

    Reference
    ----------
    ImageReward: https://github.com/thaosu/ImageReward

    Parameters
    ----------
    *args : Any
        Additional arguments.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    model_name : str, optional
        The ImageReward model to use. Default is "ImageReward-v1.0".
    call_type : str, optional
        The type of call to use for the metric.
    **kwargs : Any
        Additional keyword arguments.
    """

    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = METRIC_IMAGE_REWARD
    runs_on: List[str] = ["cpu", "cuda", "mps"]

    def __init__(
        self,
        *args,
        device: str | torch.device | None = None,
        model_name: str = "ImageReward-v1.0",
        call_type: str = SINGLE,
        **kwargs,
    ) -> None:
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        self.model_name = model_name
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)

        # Import ImageReward lazily
        try:
            import ImageReward as ImageRewardModule
        except ImportError:
            pruna_logger.error("ImageReward is not installed. Install with: pip install image-reward")
            raise

        self.model = ImageRewardModule.load(self.model_name, device=str(self.device))
        self.add_state("scores", [])

    @torch.no_grad()
    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (text prompts).
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images to score.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = inputs[0]
        prompts = x if isinstance(x, list) else [""] * len(images)

        for i, image in enumerate(images):
            if isinstance(image, torch.Tensor):
                image = self._tensor_to_pil(image)

            prompt = prompts[i] if i < len(prompts) else ""
            score = self.model.score(prompt, image)
            self.scores.append(score)

    def compute(self) -> MetricResult:
        """
        Compute the mean ImageReward metric.

        Returns
        -------
        MetricResult
            The mean ImageReward metric.
        """
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)

        import numpy as np

        mean_score = float(np.mean(self.scores))
        return MetricResult(self.metric_name, self.__dict__, mean_score)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.max() > 1:
            tensor = tensor / 255.0
        np_img = (tensor.cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(np_img.transpose(1, 2, 0))
