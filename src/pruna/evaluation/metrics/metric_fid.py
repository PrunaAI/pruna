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
FID (Fréchet Inception Distance) Metric for Pruna.

This metric computes the FID between generated images and reference images.
It measures the distance between the feature distributions of real and generated images.

Based on the InferBench implementation:
https://github.com/PrunaAI/InferBench
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import torch
from PIL import Image
from torch import Tensor
from torchmetrics.image.fid import FrechetInceptionDistance

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import metric_data_processor
from pruna.logging.logger import pruna_logger


METRIC_FID = "fid"


@MetricRegistry.register(METRIC_FID)
class FIDMetric(StatefulMetric):
    """
    Fréchet Inception Distance (FID) metric for evaluating image generation quality.

    FID compares the distribution of generated images with a reference distribution.
    Lower FID scores indicate better image quality, meaning the generated images
    are more similar to the reference images.

    Parameters
    ----------
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    feature_dim : int, optional
        The dimension of the Inception features to use. Default is 2048.
    call_type : str, optional
        The call type to use for the metric. Default is "gt_y".

    References
    ----------
    FID: https://arxiv.org/abs/1706.08500
    """

    higher_is_better: bool = False
    metric_name: str = METRIC_FID

    def __init__(
        self,
        device: str | torch.device | None = None,
        feature_dim: int = 2048,
        call_type: str = "gt_y",
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        
        # Initialize FID metric
        self.fid = FrechetInceptionDistance(feature=feature_dim, normalize=False, reset_real_features=False)
        self.fid.to(self.device)
        
        # Use float64 for better stability on CUDA, float32 for MPS
        if self.device == "cuda":
            self.fid.set_dtype(torch.float64)
        else:
            self.fid.set_dtype(torch.float32)
        
        self.call_type = call_type
        self._real_features_initialized = False
        
        # Add state for tracking
        self.add_state("score", torch.tensor(0.0))
        
        pruna_logger.info(f"Initialized FID metric on device: {self.device}")

    def _pil_to_tensor(self, image: Union[Image.Image, Tensor]) -> Tensor:
        """
        Convert PIL Image or Tensor to FID-compatible tensor format.

        Parameters
        ----------
        image : Union[Image.Image, Tensor]
            The image to convert.

        Returns
        -------
        Tensor
            The converted tensor with uint8 values in (N, C, H, W) format.
        """
        if isinstance(image, Tensor):
            # If already a tensor, ensure it's in the right format
            if image.dtype != torch.uint8:
                image = (image * 255).to(torch.uint8) if image.max() <= 1.0 else image.to(torch.uint8)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            elif image.ndim == 4 and image.shape[0] == 1:
                pass  # Already in (1, C, H, W) format
            return image
        else:
            # PIL Image
            img_array = torch.from_numpy(
                __import__('numpy').array(image.convert("RGB"))
            ).permute(2, 0, 1).to(torch.uint8)
            return img_array.unsqueeze(0)

    def _convert_batch_to_tensor(self, images: Union[Tensor, List, Image.Image]) -> Tensor:
        """
        Convert a batch of images to FID-compatible tensor format.

        Parameters
        ----------
        images : Union[Tensor, List, Image.Image]
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
            # Convert to uint8 if needed
            if images.dtype != torch.uint8:
                if images.max() <= 1.0:
                    images = (images * 255).to(torch.uint8)
                else:
                    images = images.to(torch.uint8)
            return images
        elif isinstance(images, (list, tuple)):
            tensors = []
            for img in images:
                tensors.append(self._pil_to_tensor(img))
            return torch.cat(tensors, dim=0)
        else:
            return self._pil_to_tensor(images)

    def update(
        self,
        x: Any,
        gt: Union[Tensor, List[Tensor]],
        outputs: Union[Tensor, List[Tensor]],
        reference_images: Optional[Union[Tensor, List[Tensor]]] = None,
    ) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : Any
            The input data (not used for FID).
        gt : Union[Tensor, List[Tensor]]
            The ground truth / reference images.
        outputs : Union[Tensor, List[Tensor]]
            The generated images.
        reference_images : Optional[Union[Tensor, List[Tensor]]], optional
            Additional reference images to use for FID computation.
        """
        # Get the generated images from outputs
        inputs = metric_data_processor(x, gt, outputs, self.call_type, self.device)
        generated_images = inputs[1]  # The generated images
        
        # Convert to tensor format
        gen_tensor = self._convert_batch_to_tensor(generated_images).to(self.device)
        
        # Load reference images if provided
        ref_tensors = None
        if reference_images is not None:
            ref_tensors = self._convert_batch_to_tensor(reference_images).to(self.device)
        elif gt is not None:
            # Use gt as reference images
            ref_tensors = self._convert_batch_to_tensor(gt).to(self.device)
        
        # Update FID with reference images if provided and not initialized
        if ref_tensors is not None and not self._real_features_initialized:
            self.fid.update(ref_tensors, real=True)
            self._real_features_initialized = True
        
        # Update FID with generated images
        self.fid.update(gen_tensor, real=False)

    def compute(self) -> MetricResult:
        """
        Compute the FID score.

        Returns
        -------
        MetricResult
            The computed FID score.
        """
        try:
            fid_score = self.fid.compute()
            if isinstance(fid_score, Tensor):
                fid_value = fid_score.item()
            else:
                fid_value = float(fid_score)
            
            # Handle edge case where FID can't be computed
            if fid_value == float("inf") or fid_value > 1e10:
                pruna_logger.warning("FID score is infinite or too large, returning maximum value")
                fid_value = float("inf")
                
            return MetricResult(self.metric_name, self.__dict__.copy(), fid_value)
        except Exception as e:
            pruna_logger.error(f"Error computing FID: {e}")
            return MetricResult(self.metric_name, self.__dict__.copy(), float("inf"))

    def reset(self) -> None:
        """
        Reset the metric state.
        """
        super().reset()
        self._real_features_initialized = False

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
        self.fid = self.fid.to(device)
