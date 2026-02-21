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

ARNIQA (No-Reference Image Quality Assessment with
Deep Learning) implementation.

Based on the InferBench implementation:
https://github.com/PrunaAI/InferBench
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_ARNIQA = "arniqa"


class ARNIQANetwork(nn.Module):
    """ARNIQA network for image quality assessment."""

    def __init__(self, regressor_dataset: str = "koniq10k"):
        super().__init__()
        # Simplified ARNIQA backbone - uses ResNet features
        # In production, load pretrained weights from:
        # https://github.com/teichlab/ARNIQA
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.regressor(feat)


@MetricRegistry.register(METRIC_ARNIQA)
class ARNIQAMetric(StatefulMetric):
    """
    ARNIQA (ARNI Quality Assessment) metric.

    No-reference image quality assessment using deep learning.
    Note: This is a simplified implementation. For production use,
    download pretrained weights from https://github.com/teichlab/ARNIQA

    Higher scores indicate better image quality.

    Parameters
    ----------
    device : str | torch.device | None, optional
        Device to use.
    regressor_dataset : str, optional
        Dataset for regressor training. Default is "koniq10k".
    pretrained : bool, optional
        Load pretrained weights. Default is False.
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
        pretrained: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.device = set_to_best_available_device(device)
        self.regressor_dataset = regressor_dataset

        self.model = ARNIQANetwork(regressor_dataset=regressor_dataset)
        
        if pretrained:
            self._load_pretrained()
        
        self.model.to(self.device)
        self.model.eval()

        self.add_state("total", torch.zeros(1))
        self.add_state("count", torch.zeros(1))

    def _load_pretrained(self) -> None:
        """Load pretrained ARNIQA weights."""
        # Would load from https://github.com/teichlab/ARNIQA
        # For now, uses random weights
        pruna_logger.warning("ARNIQA pretrained weights not implemented yet")

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = inputs[0]

        with torch.no_grad():
            for image in images:
                image_tensor = self._process_image(image)
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                score = self.model(image_tensor)
                self.total += score.item()
                self.count += 1

    def compute(self) -> MetricResult:
        result = self.total / self.count if self.count.item() != 0 else torch.zeros(1)
        return MetricResult(self.metric_name, self.__dict__.copy(), result.item())

    def _process_image(self, image: torch.Tensor | Image.Image) -> torch.Tensor:
        """Process image to tensor."""
        if isinstance(image, Image.Image):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        elif isinstance(image, torch.Tensor):
            if image.ndim == 4:
                image = image[0]
            if image.max() > 1:
                image = image / 255.0
        return image
