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
CLIP Score Metric for Pruna.

This metric computes the CLIP score between images and text prompts.
It measures how well the generated images match the given text descriptions.

Based on the InferBench implementation:
https://github.com/PrunaAI/InferBench
"""

from __future__ import annotations

from typing import Any, List, cast

import torch
from torch import Tensor
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers.models.clip.modeling_clip import CLIPModel as _CLIPModel
from transformers.models.clip.processing_clip import CLIPProcessor as _CLIPProcessor

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger


METRIC_CLIP = "clip_score"


@MetricRegistry.register(METRIC_CLIP)
class CLIPScoreMetric(CLIPScore, StatefulMetric):
    """
    CLIP Score metric for evaluating image-text alignment.

    This metric computes the CLIP score between generated images and their corresponding text prompts.
    It measures how well the image matches the text description using CLIP embeddings.
    Higher scores indicate better image-text alignment.

    Parameters
    ----------
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    model_name_or_path : str, optional
        The name or path of the CLIP model to use. Default is "openai/clip-vit-large-patch14".
    call_type : str, optional
        The call type to use for the metric. Default is "gt_y".

    References
    ----------
    CLIP: https://openai.com/research/clip
    """

    higher_is_better: bool = True
    metric_name: str = METRIC_CLIP

    def __init__(
        self,
        device: str | torch.device | None = None,
        model_name_or_path: str = "openai/clip-vit-large-patch14",
        call_type: str = SINGLE,
        **kwargs: Any,
    ) -> None:
        device = set_to_best_available_device(device)
        if "call_type" in kwargs:
            pruna_logger.warning(f"call_type is not supported for {self.metric_name}. Using default call_type gt_y")
            kwargs.pop("call_type")
        
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        self.move_to_device(device)
        self.call_type = get_call_type_for_single_metric(SINGLE, "gt_y")
        pruna_logger.info(f"Using call_type: {self.call_type} for metric {self.metric_name}")

    def update(  # type: ignore[override]
        self,
        x: Tensor | List[Tensor],
        gt: Tensor | List[Tensor],
        outputs: Tensor | List[Tensor],
    ) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : Tensor | List[Tensor]
            The input data (text prompts).
        gt : Tensor | List[Tensor]
            The ground truth images.
        outputs : Tensor | List[Tensor]
            The generated images.
        """
        metric_inputs = metric_data_processor(x, gt, outputs, self.call_type, self.device)
        # For CLIP score, we need [images, text_prompts]
        # gt contains the text prompts, outputs contains the generated images
        images = metric_inputs[1]  # Generated images
        prompts = metric_inputs[0]  # Text prompts
        
        score, n_samples = _clip_score_update(
            cast(Tensor, images),
            prompts,
            cast(_CLIPModel, self.model),
            cast(_CLIPProcessor, self.processor),
        )
        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self) -> MetricResult:
        """
        Compute the CLIP score.

        Returns
        -------
        MetricResult
            The computed CLIP score.
        """
        clip_score = super().compute()  # type: ignore[safe-super]
        clip_score_item = clip_score.item() if isinstance(clip_score, Tensor) else clip_score
        return MetricResult(self.metric_name, self.__dict__.copy(), clip_score_item)

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
        self.to(device)


def _process_image_data(images: Tensor) -> List[Tensor]:
    """
    Helper function to process image data.

    CLIP expects a list of 3D images.

    Parameters
    ----------
    images : Tensor
        The images to process.

    Returns
    -------
    List[Tensor]
        The processed images.
    """
    list_images = [images] if images.ndim == 3 else list(images)
    if not all(i.ndim == 3 for i in list_images):
        pruna_logger.error("Expected all images to be 3d but found image that has either more or less")
        raise ValueError("Expected all images to be 3d but found image that has either more or less")
    return list_images


def _get_image_features(
    data: List[Tensor],
    device: torch.device,
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> Tensor:
    """
    Get the CLIP image features for the given images.

    Parameters
    ----------
    data : List[Tensor]
        The images to get the features for.
    device : torch.device
        The device to run the model on.
    model : _CLIPModel
        The CLIP model to use.
    processor : _CLIPProcessor
        The image processor to use.

    Returns
    -------
    Tensor
        The CLIP image features for the given images.
    """
    processed = processor(images=[i.cpu() for i in data], return_tensors="pt", padding=True)
    return model.get_image_features(processed["pixel_values"].to(device))


def _get_text_features(
    prompts: Any,
    device: torch.device,
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> Tensor:
    """
    Get the CLIP text features for the given prompts.

    Parameters
    ----------
    prompts : Any
        The text prompts to get the features for.
    device : torch.device
        The device to run the model on.
    model : _CLIPModel
        The CLIP model to use.
    processor : _CLIPProcessor
        The text processor to use.

    Returns
    -------
    Tensor
        The CLIP text features for the given prompts.
    """
    # Handle prompts - could be a string, list of strings, or tensor
    if isinstance(prompts, str):
        prompts = [prompts]
    elif isinstance(prompts, Tensor):
        # If tensor, assume it's a batch of strings or indices
        if prompts.dim() == 0:
            prompts = [prompts.item()] if prompts.numel() == 1 else [str(p) for p in prompts]
        else:
            prompts = [str(p) for p in prompts]
    
    processed = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    return model.get_text_features(processed["input_ids"].to(device))


def _clip_score_update(
    images: Tensor,
    prompts: Any,
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> tuple[Tensor, int]:
    """
    Update the CLIP score.

    Parameters
    ----------
    images : Tensor
        The generated images.
    prompts : Any
        The text prompts.
    model : _CLIPModel
        The CLIP model to use.
    processor : _CLIPProcessor
        The processor to use.

    Returns
    -------
    tuple[Tensor, int]
        The CLIP score and the number of samples.
    """
    image_data = _process_image_data(images)
    device = image_data[0].device
    model = cast(Any, model).to(device)

    image_features = _get_image_features(image_data, device, model, processor)
    text_features = _get_text_features(prompts, device, model, processor)

    # Normalize features
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # Calculate cosine similarity
    score = 100 * (image_features * text_features).sum(dim=-1)
    return score, len(image_data)
