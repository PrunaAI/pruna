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

import torch
from huggingface_hub import model_info
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_CMMD = "cmmd"


@MetricRegistry.register(METRIC_CMMD)
class CMMD(StatefulMetric):
    """
    Calculates the CMMD metric, which is the Maximum Mean Discrepancy between CLIP embeddings of two sets of images.

    from https://arxiv.org/abs/2401.09603, Rethinking FID: Towards a Better Evaluation metric for Image Generation

    Parameters
    ----------
    *args : Any
        Additional arguments to pass to the StatefulMetric constructor.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    clip_model_name : str
        The name of the CLIP model to use.
    call_type : str
        The call type to use for the metric.
    **kwargs : Any
        Additional keyword arguments to pass to the StatefulMetric constructor.
    """

    ground_truth_embeddings: List[torch.Tensor]
    output_embeddings: List[torch.Tensor]
    default_call_type: str = "gt_y"
    higher_is_better: bool = False
    metric_name: str = METRIC_CMMD

    def __init__(
        self,
        *args,
        device: str | torch.device | None = None,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        call_type: str = SINGLE,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.device = set_to_best_available_device(device)
        try:
            model_info(clip_model_name)
        except EntryNotFoundError:
            pruna_logger.error(f"Model {clip_model_name} does not exist.")
            raise ValueError(f"Model {clip_model_name} does not exist.")

        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.sigma = 10  # Sigma parameter from the paper
        self.scale = 1000  # Scale parameter from the paper
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)

        self.add_state("ground_truth_embeddings", [])
        self.add_state("output_embeddings", [])

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Update computes the CLIP embeddings for the two sets of images and appends them to the internal state.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data.
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        gt_embeddings = self._get_embeddings(inputs[0])
        output_embeddings = self._get_embeddings(inputs[1])

        self.ground_truth_embeddings.append(gt_embeddings)
        self.output_embeddings.append(output_embeddings)

    def compute(self) -> MetricResult:
        """
        Compute the CMMD metric.

        Calculated the Maximum Mean Discrepancy between the collected CLIP embeddings.

        Returns
        -------
        torch.Tensor
            The computed CMMD metric.
        """
        x = torch.cat(self.ground_truth_embeddings, dim=0)
        y = torch.cat(self.output_embeddings, dim=0)
        result = self._mmd(x, y)
        result_f = result.item() if isinstance(result, torch.Tensor) else result

        return MetricResult(self.metric_name, self.__dict__.copy(), result_f)

    def _get_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get the CLIP embeddings for a set of images.

        Parameters
        ----------
        images : torch.Tensor
            The images to get the CLIP embeddings for.

        Returns
        -------
        torch.Tensor
            The CLIP embeddings for the images.
        """
        processed = self.clip_processor(images=images.cpu(), return_tensors="pt").to(self.device)
        embeddings = self.clip_model(**processed).image_embeds
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        return embeddings

    def _mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Maximum Mean Discrepancy between two sets of embeddings.

        Adapted from the JAX implementation in:
        https://github.com/google-research/google-research/blob/583d3178157a3dc1eaec04935387ec797004f09b/cmmd/distance.py

        Parameters
        ----------
        x : torch.Tensor
            The first set of embeddings.
        y : torch.Tensor
            The second set of embeddings.

        Returns
        -------
        torch.Tensor
            The Maximum Mean Discrepancy between the two sets of embeddings.
        """
        x_sqnorms = torch.diag(torch.matmul(x, x.T))
        y_sqnorms = torch.diag(torch.matmul(y, y.T))

        gamma = 1 / (2 * self.sigma**2)
        k_xx = torch.mean(
            torch.exp(
                -gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0))
            )
        )
        k_xy = torch.mean(
            torch.exp(
                -gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0))
            )
        )
        k_yy = torch.mean(
            torch.exp(
                -gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0))
            )
        )
        return self.scale * (k_xx + k_yy - 2 * k_xy)
