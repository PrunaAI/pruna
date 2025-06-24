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

import torch
from torch import Tensor
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers.models.clip.modeling_clip import CLIPModel as _CLIPModel
from transformers.models.clip.processing_clip import CLIPProcessor as _CLIPProcessor

from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import PAIRWISE, metric_data_processor
from pruna.logging.logger import pruna_logger


@MetricRegistry.register("pairwise_clip_score")
class PairwiseClipScore(CLIPScore, StatefulMetric):
    """
    Pairwise CLIP Score metric.

    This metric is used to compute the CLIP score between generated images of smashed and base models.
    It is a pairwise metric, meaning it compares two images at a time.
    This feature is currently a PR in torchmetrics and not supported by the latest official CLIPScore version.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments for the CLIPScore metric.
    """

    higher_is_better: bool = True
    metric_name: str = "pairwise_clip_score"

    def __init__(self, **kwargs: Any) -> None:
        if "device" in kwargs:
            kwargs.pop("device")
        if "call_type" in kwargs:
            pruna_logger.error(f"Call type is not supported for {self.metric_name}. Using default call type {PAIRWISE}")
            kwargs.pop("call_type")
        super().__init__(**kwargs)
        self.call_type = "pairwise_y_gt"
        pruna_logger.info(f"Using call_type: {self.call_type} for metric {self.metric_name}")

    def update(  # type: ignore[override]
        self, x: Tensor | List[Tensor], outputs: Tensor | List[Tensor], gt: Tensor | List[Tensor]
    ) -> None:
        """
        Update the metric.

        Parameters
        ----------
        x : Tensor | List[Tensor]
            The input images.
        outputs : Tensor | List[Tensor]
            The generated images.
        gt : Tensor | List[Tensor]
            The ground truth images.
        """
        metric_inputs = metric_data_processor(x, gt, outputs, self.call_type)
        source, target = metric_inputs
        score, n_samples = _clip_score_update(
            cast(Tensor, source),
            cast(Tensor, target),
            cast(_CLIPModel, self.model),
            cast(_CLIPProcessor, self.processor),
        )
        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self) -> Any:
        """
        Compute the metric.

        Returns
        -------
        Any
            The metric result.
        """
        pairwise_score = super().compute()  # type: ignore[safe-super]
        pairwise_score_item = pairwise_score.item() if isinstance(pairwise_score, Tensor) else pairwise_score
        return MetricResult(self.metric_name, self.__dict__.copy(), pairwise_score_item)


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

    Raises
    ------
    ValueError
        If the images are not 3D.
    """
    list_images = [images] if images.ndim == 3 else list(images)
    if not all(i.ndim == 3 for i in list_images):
        pruna_logger.error("Expected all images to be 3d but found image that has either more or less")
        raise ValueError("Expected all images to be 3d but found image that has either more or less")
    return list_images


def _get_features(
    data: List[Tensor],
    device: torch.device,
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> Tensor:
    """
    Get the CLIP features for the given images.

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
        The CLIP features for the given images.
    """
    # The processor expects the images to be on the CPU
    processed = processor(images=[i.cpu() for i in data], return_tensors="pt", padding=True)
    return model.get_image_features(processed["pixel_values"].to(device))


def _clip_score_update(
    source: Tensor,
    target: Tensor,
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> tuple[Tensor, int]:
    """
    Update the CLIP score.

    Parameters
    ----------
    source : Tensor
        The source images.
    target : Tensor
        The target images.
    model : _CLIPModel
        The CLIP model to use.
    processor : _CLIPProcessor
        The image processor to use.

    Returns
    -------
    tuple[Tensor, int]
        The CLIP score and the number of samples.
    """
    source_data = _process_image_data(source)
    target_data = _process_image_data(target)

    device = source_data[0].device
    model = cast(Any, model).to(device)

    source_features = _get_features(source_data, device, model, processor)
    target_features = _get_features(target_data, device, model, processor)
    source_features = source_features / source_features.norm(p=2, dim=-1, keepdim=True)
    target_features = target_features / target_features.norm(p=2, dim=-1, keepdim=True)

    # Calculate cosine similarity
    score = 100 * (source_features * target_features).sum(dim=-1)
    return score, len(source_data)
