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

import timm
import torch

# Ruff complains when we don't import functional as f, but common practice is to import it as F
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

DINO_SCORE = "dino_score"


@MetricRegistry.register(DINO_SCORE)
class DinoScore(StatefulMetric):
    """
    DINO Score metric.

    A similarity metric based on DINO (self-distillation with no labels),
    a self-supervised vision transformer trained to learn high-level image representations without annotations.
    DinoScore compares the embeddings of generated and reference images in this representation space,
    producing a value where higher scores indicate that the generated images preserve more of the semantic content of the
    reference images.

    Reference
    ----------
    https://github.com/facebookresearch/dino
    https://arxiv.org/abs/2104.14294

    Parameters
    ----------
    device : str | torch.device | None
        The device to use for the metric.
    call_type : str
        The call type to use for the metric.
    """

    similarities: List[Tensor]
    metric_name: str = DINO_SCORE
    higher_is_better: bool = True
    runs_on: List[str] = ["cuda", "cpu"]
    default_call_type: str = "x_y"

    def __init__(self, device: str | torch.device | None = None, call_type: str = SINGLE):
        super().__init__()
        self.device = set_to_best_available_device(device)
        if device is not None and not any(self.device.startswith(prefix) for prefix in self.runs_on):
            pruna_logger.error(f"DinoScore: device {device} not supported. Supported devices: {self.runs_on}")
            raise
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        # Load the DINO ViT-S/16 model once
        self.model = timm.create_model("vit_small_patch16_224.dino", pretrained=True)
        self.model.eval().to(self.device)
        # Add internal state to accumulate similarities
        self.add_state("similarities", default=[])

    @torch.no_grad()
    def update(self, x: List[Any] | Tensor, gt: Tensor, outputs: torch.Tensor) -> None:
        """
        Accumulate the DINO scores for each batch.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data.
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images.
        """
        metric_inputs = metric_data_processor(x, gt, outputs, self.call_type)
        inputs, preds = metric_inputs

        # Extract embeddings ([CLS] token)
        emb_x = self.model.forward_features(inputs)
        emb_y = self.model.forward_features(preds)

        # Normalize embeddings
        emb_x = F.normalize(emb_x, dim=1)
        emb_y = F.normalize(emb_y, dim=1)

        # Compute cosine similarity
        sim = (emb_x * emb_y).sum(dim=1)
        self.similarities.append(sim)

    def compute(self) -> MetricResult:
        """
        Compute the DINO score.

        Returns
        -------
        MetricResult
            The DINO score.
        """
        sims = torch.cat(self.similarities)
        mean_sim = sims.mean().item()
        return MetricResult(self.metric_name, self.__dict__, mean_sim)
