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
ARNIQA Metric for Pruna.

This metric computes image quality assessment scores using ARNIQA.

Based on the InferBench implementation:
https://github.com/PrunaAI/InferBench
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
from PIL import Image

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_ARNIQA = "arniqa"


@MetricRegistry.register(METRIC_ARNIQA)
class ARNIQAMetric(StatefulMetric):
    """
    ARNIQA (ARNI Quality Assessment) metric for evaluating image quality.

    This metric uses the ARNIQA model to assess image quality.
    Higher scores indicate better image quality.

    Parameters
    ----------
    *args : Any
        Additional arguments to pass to the StatefulMetric constructor.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    regressor_dataset : str, optional
        The dataset used for training the regressor. Default is "koniq10k".
    reduction : str, optional
        The reduction method for the scores. Default is "mean".
    normalize : bool, optional
        Whether to normalize the scores. Default is True.
    **kwargs : Any
        Additional keyword arguments to pass to the StatefulMetric constructor.

    References
    ----------
    ARNIQA: https://github.com/teichlab/ARNIQA
    """

    total: torch.Tensor
    count: torch.Tensor
    call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = METRIC_ARNIQA

    def __init__(
        self,
        *args,
        device: str | torch.device | None = None,
        regressor_dataset: str = "koniq10k",
        reduction: str = "mean",
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.device = set_to_best_available_device(device)
        self.regressor_dataset = regressor_dataset
        self.reduction = reduction
        self.normalize = normalize

        # Import torchmetrics ARNIQA
        try:
            from torchmetrics.image.arniqa import ARNIQA as TorchARNIQA
        except ImportError:
            pruna_logger.error("ARNIQA not available in torchmetrics. Installing torchmetrics with image extras may help.")
            raise

        self.metric = TorchARNIQA(
            regressor_dataset=self.regressor_dataset,
            reduction=self.reduction,
            normalize=self.normalize,
            autocast=False,
        )
        self.metric.to(self.device)

        self.add_state("total", torch.zeros(1))
        self.add_state("count", torch.zeros(1))

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        This computes the ARNIQA scores for the given images.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data.
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images to score.
        """
        # Get images
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = inputs[0]  # Generated images

        with torch.no_grad():
            for image in images:
                # Convert tensor to ARNIQA expected format
                if isinstance(image, torch.Tensor):
                    image_tensor = self._tensor_to_arniqa_format(image)
                else:
                    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                score = self.metric(image_tensor)
                self.total += score
                self.count += 1

    def compute(self) -> MetricResult:
        """
        Compute the average ARNIQA metric based on previous updates.

        Returns
        -------
        MetricResult
            The average ARNIQA metric.
        """
        result = self.total / self.count if self.count.item() != 0 else torch.zeros(1)
        return MetricResult(self.metric_name, self.__dict__.copy(), result.item())

    def _tensor_to_arniqa_format(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor to ARNIQA expected format (C, H, W) in [0, 1].

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to convert.

        Returns
        -------
        torch.Tensor
            The converted tensor.
        """
        # Handle batch dimension
        if tensor.ndim == 4:
            tensor = tensor[0]

        # Ensure values are in [0, 1]
        if tensor.max() > 1:
            tensor = tensor / 255.0

        return tensor
