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
from pruna.evaluation.metrics.utils import (
    SINGLE,
    get_call_type_for_single_metric,
    metric_data_processor,
)
from pruna.logging.logger import pruna_logger

IMAGE_REWARD = "image_reward"
HPS_REWARD = "hps"
HPSv2_REWARD = "hpsv2"
VQA_REWARD = "vqa"


class BaseModelRewardMetric(StatefulMetric):
    """Base class for model reward metrics."""

    higher_is_better: bool = True
    default_call_type: str = "y"
    metric_units: str = "score"

    # Type annotations for dynamically added attributes
    scores: List[float]
    prompts: List[str]

    def __init__(
        self,
        device: str | torch.device | None = None,
        call_type: str = SINGLE,
        model_load_kwargs: dict = {},
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.device = set_to_best_available_device(device)
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.to_pil = ToPILImage()

        # Initialize state for accumulating scores
        self.add_state("scores", [])
        self.add_state("prompts", [])
        self._load(**model_load_kwargs)

    def _load(self, **kwargs: Any) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def _score_image(self, prompt: str, image: PIL.Image.Image) -> float:
        raise NotImplementedError("Subclasses must implement this method")

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
            score = self._score_image(prompt=prompt, image=image)
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
        PIL.Image.Image
            The formatted PIL image.
        """
        # Convert tensor to PIL Image
        if image.dim() == 4:
            # Batch dimension, take first image
            image = image[0]

        # Handle different tensor formats
        if image.dim() == 3:
            # Check if image is in HWC format (H, W, C)
            if image.shape[-1] in [1, 3, 4]:
                # Convert HWC to CHW format
                image = image.permute(2, 0, 1)
            elif image.shape[0] in [1, 3, 4]:
                # Image is already in CHW format
                pass
            else:
                # Unknown format, try to handle gracefully
                pruna_logger.warning(f"Unexpected image shape: {image.shape}")
                return PIL.Image.new("RGB", (224, 224))

            # Handle different channel counts
            if image.shape[0] == 1:
                # Grayscale, convert to RGB
                image = image.repeat(3, 1, 1)
            elif image.shape[0] == 4:
                # RGBA, take only RGB channels
                image = image[:3]

            # Normalize to [0, 1] if needed
            if image.max() > 1.0:
                image = image / 255.0

            # Ensure values are in valid range
            image = torch.clamp(image, 0.0, 1.0)

            # Convert to PIL Image
            pil_image = self.to_pil(image)
        return pil_image


@MetricRegistry.register(IMAGE_REWARD)
class ImageRewardMetric(BaseModelRewardMetric):
    """
    ImageReward metric for evaluating text-to-image generation quality.

    ImageReward is a human preference reward model for text-to-image generation that
    outperforms existing methods like CLIP, Aesthetic, and BLIP in understanding
    human preferences.

    References
    ----------
    - https://github.com/THUDM/ImageReward

    Parameters
    ----------
    device : str | torch.device | None, optional
        The device to use for the model. If None, the best available device will be used.
    call_type : str
        The type of call to use for the metric. IQA metrics, like image_reward, are only supported for single mode.
    **model_load_kwargs : Any
        Additional keyword arguments for the model.
    **kwargs : Any
        Additional keyword arguments for the metric.
    """

    metric_name: str = IMAGE_REWARD

    def _load(self, **kwargs: Any) -> None:
        model_name = kwargs.get("model_name", "ImageReward-v1.0")
        pruna_logger.info(f"Loading ImageReward model: {model_name}")

        import ImageReward as RM  # noqa: N814

        self.model = RM.load(
            name=model_name,
            device=kwargs.get("device", self.device),
            **kwargs,
        )

    def _score_image(self, prompt: str, image: PIL.Image.Image) -> float:
        """
        Score an image with a prompt using ImageReward.

        Parameters
        ----------
        prompt : str
            The prompt to score the image with.
        image : PIL.Image.Image
            The image to score.

        Returns
        -------
        float
            The score of the image.
        """
        return self.model.score(prompt, image)


@MetricRegistry.register(HPS_REWARD)
class HPSMetric(BaseModelRewardMetric):
    """
    HPS metric for evaluating text-to-image generation quality.

    References
    ----------
    - https://github.com/tgxs002/align_sd

    Parameters
    ----------
    device : str | torch.device | None, optional
        The device to use for the model. If None, the best available device will be used.
    call_type : str
        The type of call to use for the metric. IQA metrics, like image_reward, are only supported for single mode.
    **model_load_kwargs : Any
        Additional keyword arguments for the model.
    **kwargs : Any
        Additional keyword arguments for the metric.
    """

    metric_name: str = HPS_REWARD

    def _load(self, **kwargs: Any) -> None:
        import clip

        self.model, self.preprocess = clip.load("ViT-L/14", device=kwargs.get("device", self.device), **kwargs)

    def _score_image(self, prompt: str, image: PIL.Image.Image) -> float:
        """
        Score an image with a prompt using HPS.

        Parameters
        ----------
        prompt : str
            The prompt to score the image with.
        image : PIL.Image.Image
            The image to score.

        Returns
        -------
        float
            The score of the image.
        """
        model = self.model
        preprocess = self.preprocess
        device = self.device

        # Preprocess image and move to device
        images = preprocess(image).unsqueeze(0).to(device)
        # Tokenize prompt and move to device
        import clip

        text = clip.tokenize([prompt]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(text)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            hps = image_features @ text_features.T

        return hps.item()


@MetricRegistry.register(HPSv2_REWARD)
class HPSv2Metric(BaseModelRewardMetric):
    """
    HPSv2 metric for evaluating text-to-image generation quality.

    References
    ----------
    - https://github.com/tgxs002/HPSv2

    Parameters
    ----------
    device : str | torch.device | None, optional
        The device to use for the model. If None, the best available device will be used.
    call_type : str
        The type of call to use for the metric. IQA metrics, like image_reward, are only supported for single mode.
    **model_load_kwargs : Any
        Additional keyword arguments for the model.
    **kwargs : Any
        Additional keyword arguments for the metric.
    """

    metric_name: str = HPSv2_REWARD

    def _load(self, **kwargs: Any) -> None:
        import hpsv2

        self.model = hpsv2

    def _score_image(self, prompt: str, image: PIL.Image.Image) -> float:
        """
        Score an image with a prompt using HPS.

        Parameters
        ----------
        prompt : str
            The prompt to score the image with.
        image : PIL.Image.Image
            The image to score.

        Returns
        -------
        float
            The score of the image.
        """
        score = self.model.score(imgs_path=image, prompt=prompt, hps_version="v2.1")
        # Handle case where score might be a list or array
        if isinstance(score, (list, tuple)):
            return float(score[0])
        return float(score)


@MetricRegistry.register(VQA_REWARD)
class VQAMetric(BaseModelRewardMetric):
    """
    VQA metric for evaluating text-to-image generation quality.

    VQA (Visual Question Answering) metric evaluates the quality of text-to-image
    generation by assessing how well the generated images can answer questions about
    their content.

    References
    ----------
    - https://github.com/PrunaAI/t2v-metrics

    Parameters
    ----------
    device : str | torch.device | None, optional
        The device to use for the model. If None, the best available device will be used.
    call_type : str
        The type of call to use for the metric. IQA metrics, like image_reward, are only supported for single mode.
    **model_load_kwargs : Any
        Additional keyword arguments for the model.
    **kwargs : Any
        Additional keyword arguments for the metric.
    """

    metric_name: str = VQA_REWARD

    def _load(self, **kwargs: Any) -> None:
        try:
            import t2v_metrics

            self.model = t2v_metrics.VQAScore(
                model=kwargs.get("model", "clip-flant5-xxl"),
                device=str(self.device),
            )
        except ImportError:
            pruna_logger.warning("t2v-metrics not available. VQA metric will not work.")
            self.model = None

    def _score_image(self, prompt: str, image: PIL.Image.Image) -> float:
        """
        Score an image with a prompt using VQA.

        Parameters
        ----------
        prompt : str
            The prompt to score the image with.
        image : PIL.Image.Image
            The image to score.

        Returns
        -------
        float
            The VQA score for the image.
        """
        if self.model is None:
            return 0.0

        try:
            # Convert prompt to a question format for VQA
            question = f"What is shown in this image? Answer: {prompt}"
            score = self.model.score(image, question)

            # Handle case where score might be a list or array
            if isinstance(score, (list, tuple)):
                return float(score[0])
            return float(score)
        except Exception as e:
            pruna_logger.warning(f"Error computing VQA score: {e}")
            return 0.0
