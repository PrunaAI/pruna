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

from typing import Any, List

import PIL
import torch
from torch import Tensor
from torchvision.transforms import ToPILImage

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

IMAGE_REWARD = "image_reward"


@MetricRegistry.register(IMAGE_REWARD)
class ImageRewardMetric(StatefulMetric):
    """
    ImageReward metric for evaluating text-to-image generation quality.

    ImageReward is a human preference reward model for text-to-image generation that
    outperforms existing methods like CLIP, Aesthetic, and BLIP in understanding
    human preferences.

    Parameters
    ----------
    device : str | torch.device | None, optional
        The device to use for the model. If None, the best available device will be used.
    model_name : str, optional
        The ImageReward model to use. Default is "ImageReward-v1.0".
    call_type : str
        The type of call to use for the metric. IQA metrics, like image_reward, are only supported for single mode.
    **kwargs : Any
        Additional keyword arguments for the metric.
    """

    higher_is_better: bool = True
    default_call_type: str = "y"
    metric_name: str = IMAGE_REWARD
    metric_units: str = "score"

    # Type annotations for dynamically added attributes
    scores: List[float]
    prompts: List[str]

    def __init__(
        self,
        device: str | torch.device | None = None,
        model_name: str = "ImageReward-v1.0",
        call_type: str = SINGLE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.device = set_to_best_available_device(device)
        self.model_name = model_name
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)

        # Import ImageReward here to avoid dependency issues
        import ImageReward as RM  # noqa: N814

        # Load the ImageReward model
        pruna_logger.info(f"Loading ImageReward model: {model_name}")
        self.model = RM.load(model_name, device=self.device)
        self.to_pil = ToPILImage()

        # Initialize state for accumulating scores
        self.add_state("scores", [])
        self.add_state("prompts", [])

    def update(self, x: List[str] | Tensor, gt: Tensor, outputs: Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : List[str] | Tensor
            The input prompts for text-to-image generation.
        gt : Tensor
            The ground truth images (not used for ImageReward).
        outputs : Tensor
            The generated images to evaluate.
        """
        # Prepare inputs
        metric_inputs = metric_data_processor(x, gt, outputs, self.call_type, device=self.device)
        prompts = self._extract_prompts(x)
        images = metric_inputs[1] if len(metric_inputs) > 1 else outputs

        # Format images as PIL Images
        formatted_images = [self._format_image(image) for image in images]

        # Score images with prompts
        for prompt, image in zip(prompts, formatted_images):
            score = self.model.score(prompt, image)
            self.scores.append(score)
            self.prompts.append(prompt)

    def compute(self) -> MetricResult:
        """
        Compute the final ImageReward metric.

        Returns
        -------
        MetricResult
            The computed ImageReward score.
        """
        if not self.scores:
            pruna_logger.warning("No scores available for ImageReward computation")
            return MetricResult(self.metric_name, self.__dict__.copy(), 0.0)

        # Calculate mean score
        mean_score = torch.mean(torch.tensor(self.scores)).item()

        return MetricResult(self.metric_name, self.__dict__.copy(), mean_score)

    def _extract_prompts(self, x: List[str] | Tensor) -> List[str]:
        """
        Extract prompts from input data.

        Parameters
        ----------
        x : List[str] | Tensor
            The input data containing prompts.

        Returns
        -------
        List[str]
            The extracted prompts.
        """
        if isinstance(x, list):
            return x
        elif isinstance(x, Tensor):
            # If x is a tensor, we need to handle it differently
            # This might be the case for some data formats
            pruna_logger.warning("Input x is a tensor, assuming it contains encoded prompts")
            return [f"prompt_{i}" for i in range(x.shape[0])]
        else:
            pruna_logger.error(f"Unexpected input type for prompts: {type(x)}")
            return []

    def _format_image(self, image: Tensor) -> PIL.Image.Image:
        """
        Format a single image with its prompt using ImageReward.

        Parameters
        ----------
        image : Tensor
            The image to score.

        Returns
        -------
        float
            The ImageReward score for the image.
        """
        # Convert tensor to PIL Image
        if image.dim() == 4:
            # Batch dimension, take first image
            image = image[0]

        # Ensure image is in the correct format (C, H, W) with values in [0, 1]
        if image.dim() == 3 and image.shape[0] in [1, 3, 4]:
            # Image is in CHW format
            if image.shape[0] == 1:
                # Grayscale, convert to RGB
                image = image.repeat(3, 1, 1)
            elif image.shape[0] == 4:
                # RGBA, take only RGB channels
                image = image[:3]

            # Normalize to [0, 1] if needed
            if image.max() > 1.0:
                image = image / 255.0

            # Convert to PIL Image
            pil_image = self.to_pil(image)
        return pil_image
