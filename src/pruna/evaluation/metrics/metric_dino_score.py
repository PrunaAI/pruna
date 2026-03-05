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

from typing import Any, List, Literal

import torch

# Ruff complains when we don't import functional as f, but common practice is to import it as F
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torchvision import transforms

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

DINO_SCORE = "dino_score"

DINO_PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


@MetricRegistry.register(DINO_SCORE)
class DinoScore(StatefulMetric):
    """
    DINO Score metric.

    A similarity metric based on DINO (self-distillation with no labels),
    a self-supervised vision transformer trained to learn high-level image representations without annotations.
    DinoScore compares the [CLS] token embeddings of generated and reference images in this representation space,
    producing a value where higher scores indicate that the generated images preserve more of the semantic content of the
    reference images.

    Supports DINO (v1), DINOv2, and DINOv3 backbones. DINOv3 models may require weights from Meta's download form.

    References
    ----------
    DINO: https://github.com/facebookresearch/dino, https://arxiv.org/abs/2104.14294
    DINOv2: https://github.com/facebookresearch/dinov2
    DINOv3: https://github.com/facebookresearch/dinov3

    Parameters
    ----------
    device : str | torch.device | None
        The device to use for the metric.
    model : {"dino", "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"}
        Backbone variant. "dino" uses timm vit_small_patch16_224.dino (DINO v1).
        "dinov2_*" uses torch.hub facebookresearch/dinov2. "dinov3_*" uses timm (requires timm>=1.0.20).
    call_type : str
        The call type to use for the metric.
    """

    similarities: List[Tensor]
    metric_name: str = DINO_SCORE
    higher_is_better: bool = True
    runs_on: List[str] = ["cuda", "cpu", "mps"]
    default_call_type: str = "gt_y"

    def __init__(
        self,
        device: str | torch.device | None = None,
        model: Literal[
            "dino", "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"
        ] = "dino",
        call_type: str = SINGLE,
    ):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        if device is not None and not any(self.device.startswith(prefix) for prefix in self.runs_on):
            pruna_logger.error(f"DinoScore: device {device} not supported. Supported devices: {self.runs_on}")
            raise
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.model_name = model
        self.model = self._load_model(model)
        self.model.eval().to(self.device)
        self.add_state("similarities", default=[])
        self.processor = DINO_PREPROCESS

    def _load_model(
        self,
        model: str,
    ) -> torch.nn.Module:
        if model == "dino":
            import timm
            return timm.create_model("vit_small_patch16_224.dino", pretrained=True)
        if model.startswith("dinov2_"):
            return torch.hub.load("facebookresearch/dinov2", model)
        if model.startswith("dinov3_"):
            import timm
            timm_map = {
                "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
                "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
                "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
            }
            timm_name = timm_map.get(model)
            if timm_name is None:
                raise ValueError(f"Unsupported DINOv3 model: {model}. Choose from {list(timm_map.keys())}")
            try:
                return timm.create_model(timm_name, pretrained=True)
            except Exception as e:
                raise ValueError(
                    f"DINOv3 requires timm>=1.0.20 and model weights from Meta. "
                    f"See https://github.com/facebookresearch/dinov3. Error: {e}"
                ) from e
        raise ValueError(f"Unsupported model: {model}")

    def _get_embeddings(self, x: Tensor) -> Tensor:
        if self.model_name == "dino":
            features = self.model.forward_features(x)
            return features[:, 0]
        if self.model_name.startswith("dinov2_"):
            out = self.model.forward_features(x)
            return out["x_norm_clstoken"]
        if self.model_name.startswith("dinov3_"):
            features = self.model.forward_features(x)
            return features[:, 0]
        features = self.model.forward_features(x)
        if isinstance(features, dict):
            return features["x_norm_clstoken"]
        return features[:, 0]

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
        inputs = self.processor(inputs)
        preds = self.processor(preds)
        emb_x = self._get_embeddings(inputs)
        emb_y = self._get_embeddings(preds)
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
