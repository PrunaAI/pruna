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
HPS (Human Preference Score) Metric for Pruna.

This metric computes the HPSv2 score measuring human preference for image-text alignment.

Based on the InferBench implementation:
https://github.com/PrunaAI/InferBench
"""

from __future__ import annotations

import os
from typing import Any, List

import huggingface_hub
import torch
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.utils import hps_version_map, root_path
from PIL import Image

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_HPS = "hps"


@MetricRegistry.register(METRIC_HPS)
class HPSMetric(StatefulMetric):
    """
    Human Preference Score v2 metric for evaluating image-text alignment.

    This metric uses the HPSv2 model to compute how well generated images
    match their text prompts based on human preferences.
    Higher scores indicate better alignment with human preferences.

    Reference
    ----------
    HPSv2: https://github.com/tgxs002/HPSv2

    Parameters
    ----------
    *args : Any
        Additional arguments.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    hps_version : str, optional
        The HPS version to use. Default is "v2.1".
    call_type : str, optional
        The type of call to use for the metric.
    **kwargs : Any
        Additional keyword arguments.
    """

    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = METRIC_HPS
    runs_on: List[str] = ["cpu", "cuda", "mps"]

    def __init__(
        self,
        *args,
        device: str | torch.device | None = None,
        hps_version: str = "v2.1",
        call_type: str = SINGLE,
        **kwargs,
    ) -> None:
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        self.hps_version = hps_version
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)

        try:
            import hpsv2  # noqa: F401
        except ImportError:
            pruna_logger.error("hpsv2 not installed. Install with: pip install hpsv2")
            raise

        self.model_dict = {}
        self._initialize_model()
        self.add_state("scores", [])

    def _initialize_model(self) -> None:
        if not self.model_dict:
            model, _, preprocess_val = create_model_and_transforms(
                "ViT-H-14", "laion2B-s32B-b79K", precision="amp",
                device=self.device, jit=False, force_quick_gelu=False,
                force_custom_text=False, force_patch_dropout=False,
                force_image_size=None, pretrained_image=False,
                image_mean=None, image_std=None, light_augmentation=True,
                aug_cfg={}, output_dict=True, with_score_predictor=False,
                with_region_predictor=False,
            )
            self.model_dict["model"] = model
            self.model_dict["preprocess_val"] = preprocess_val

            if not os.path.exists(root_path):
                os.makedirs(root_path)
            cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[self.hps_version])
            checkpoint = torch.load(cp, map_location=self.device)
            model.load_state_dict(checkpoint["state_dict"])
            self.tokenizer = get_tokenizer("ViT-H-14")
            model.to(self.device)
            model.eval()

    @torch.no_grad()
    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = inputs[0]
        prompts = x if isinstance(x, list) else [""] * len(images)

        model = self.model_dict["model"]
        preprocess_val = self.model_dict["preprocess_val"]

        for i, image in enumerate(images):
            if isinstance(image, torch.Tensor):
                image = self._tensor_to_pil(image)
            prompt = prompts[i] if i < len(prompts) else ""

            image_tensor = preprocess_val(image).unsqueeze(0).to(self.device)
            text = self.tokenizer([prompt]).to(self.device)

            with torch.amp.autocast(device_type=self.device.type):
                out = model(image_tensor, text)
                image_features = out["image_features"]
                text_features = out["text_features"]
                logits = image_features @ text_features.T
                hps_score = torch.diagonal(logits).cpu().detach().numpy()[0]

            self.scores.append(hps_score)

    def compute(self) -> MetricResult:
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)

        import numpy as np
        mean_score = float(np.mean(self.scores))
        return MetricResult(self.metric_name, self.__dict__, mean_score)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.max() > 1:
            tensor = tensor / 255.0
        np_img = (tensor.cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(np_img.transpose(1, 2, 0))
