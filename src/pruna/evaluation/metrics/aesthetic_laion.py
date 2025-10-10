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

import pathlib
from typing import Any, Dict, List, Literal
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from huggingface_hub import model_info
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_AESTHETIC_LAION = "aesthetic_laion"


@MetricRegistry.register(METRIC_AESTHETIC_LAION)
class AestheticLAION(StatefulMetric):
    """
     Predicts an image aesthetic quality score using LAION-Aesthetics_Predictor V1.

    This metric computes CLIP image embeddings and feeds them into a pretrained
    linear head released by LAION (matched to the chosen CLIP variant). The model
    returns a scalar score per image (on a ~1–10 scale). The metric
    aggregates scores across updates by reporting their mean. Higher is better.

    Reference
    ---------
    LAION-Aesthetics_Predictor V1: https://github.com/LAION-AI/aesthetic-predictor

    Parameters
    ----------
    *args : Any
        Additional arguments to pass to the StatefulMetric constructor.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    clip_model_name : CLIPVariantAesthetics
        The variant of a CLIP model to be used.
    **kwargs : Any
        Additional keyword arguments to pass to the StatefulMetric constructor.
    """

    total: torch.Tensor
    count: torch.Tensor
    call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = METRIC_AESTHETIC_LAION

    model_name_to_aesthetic_head_name: Dict[str, str] = {
        "openai/clip-vit-large-patch14": "vit_l_14",
        "openai/clip-vit-base-patch32": "vit_b_32",
        "openai/clip-vit-base-patch16": "vit_b_16",
    }

    def __init__(
        self,
        *args,
        device: str | torch.device | None = None,
        model_name_or_path: Literal[
            "openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16"
        ] = "openai/clip-vit-large-patch14",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if model_name_or_path not in self.model_name_to_aesthetic_head_name:
            pruna_logger.error(f"Model {model_name_or_path} does not exist.")
            raise ValueError(f"Model {model_name_or_path} does not exist.")

        self.device = set_to_best_available_device(device)
        try:
            model_info(model_name_or_path)
        except (EntryNotFoundError, RepositoryNotFoundError):
            # Should never happen due to the previous check
            pruna_logger.error(f"Model {model_name_or_path} does not exist.")
            raise ValueError(f"Model {model_name_or_path} does not exist.")

        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(model_name_or_path).to(self.device)
        self.clip_processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
        self.aesthetic_model = self._get_aesthetic_model(model_name_or_path)

        self.add_state("total", torch.zeros(1))
        self.add_state("count", torch.zeros(1))

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        This computes the CLIP embeddings and aesthetic scores for the given inputs.

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
        image_features = self._get_embeddings(inputs[0])
        with torch.no_grad():
            prediction = self.aesthetic_model(image_features).cpu()
            self.total += torch.sum(prediction)
            self.count += prediction.shape[0]

    def compute(self) -> MetricResult:
        """
        Compute the average Aesthetic LAION metric based on previous updates.

        Returns
        -------
        float
            The average Aesthetic LAION metric.
        """
        result = self.total / self.count if self.count.item() != 0 else torch.zeros(1)
        return MetricResult(self.metric_name, self.__dict__.copy(), result.item())

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
        self.clip_model.to(self.device)
        embeddings = self.clip_model(**processed).image_embeds
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        return embeddings

    def _get_aesthetic_model(self, clip_model="openai/clip-vit-large-patch14"):
        """
        Load the aesthetic model.

        Parameters
        ----------
        clip_model : str
            CLIP variant name.

            see https://github.com/LAION-AI/aesthetic-predictor/tree/main for available variants

        Returns
        -------
        torch.nn.Linear
            A pretrained linear head corresponding to the given CLIP model name.
        """
        home = pathlib.Path("~").expanduser()
        cache_folder = home / ".cache/aesthetic_laion_linear_heads"
        clip_model = self.model_name_to_aesthetic_head_name[clip_model]
        path_to_model = cache_folder / ("sa_0_4_" + clip_model + "_linear.pth")
        if not path_to_model.exists():
            cache_folder.mkdir(exist_ok=True, parents=True)
            url_model = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_" + clip_model + "_linear.pth?raw=true"
            )
            urlretrieve(url_model, path_to_model)
        if clip_model == "vit_l_14":
            aesthetic_linear_head = nn.Linear(768, 1)
        elif clip_model == "vit_b_32" or clip_model == "vit_b_16":
            aesthetic_linear_head = nn.Linear(512, 1)
        else:
            # Should never happen due to the previous check
            pruna_logger.error(f"Model {clip_model} is not supported by aesthetic predictor.")
            raise ValueError(f"Model {clip_model} is not supported by aesthetic predictor.")
        aesthetic_linear_head.load_state_dict(torch.load(path_to_model, map_location=self.device))
        aesthetic_linear_head.eval()
        return aesthetic_linear_head.to(self.device)
