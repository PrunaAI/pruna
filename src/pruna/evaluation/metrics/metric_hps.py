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
from pruna.evaluation.metrics.utils import metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_HPS = "hps"


@MetricRegistry.register(METRIC_HPS)
class HPSMetric(StatefulMetric):
    """
    Human Preference Score v2 metric for evaluating image-text alignment.

    This metric uses the HPSv2 model to compute how well generated images
    match their text prompts based on human preferences.
    Higher scores indicate better alignment with human preferences.

    Parameters
    ----------
    *args : Any
        Additional arguments to pass to the StatefulMetric constructor.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    hps_version : str, optional
        The HPS version to use. Default is "v2.1".
    **kwargs : Any
        Additional keyword arguments to pass to the StatefulMetric constructor.

    References
    ----------
    HPSv2: https://github.com/tgxs002/HPSv2
    """

    total: torch.Tensor
    count: torch.Tensor
    call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = METRIC_HPS

    def __init__(
        self,
        *args,
        device: str | torch.device | None = None,
        hps_version: str = "v2.1",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.device = set_to_best_available_device(device)
        self.hps_version = hps_version

        # Try to import hpsv2
        try:
            import hpsv2
        except ImportError:
            pruna_logger.error("hpsv2 is not installed. Install with: pip install hpsv2")
            raise

        self.model_dict = {}
        self._initialize_model()

        self.add_state("total", torch.zeros(1))
        self.add_state("count", torch.zeros(1))

    def _initialize_model(self) -> None:
        """Initialize the HPSv2 model."""
        if not self.model_dict:
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                "ViT-H-14",
                "laion2B-s32B-b79K",
                precision="amp",
                device=self.device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False,
            )
            self.model_dict["model"] = model
            self.model_dict["preprocess_val"] = preprocess_val

            # Load checkpoint
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[self.hps_version])
            checkpoint = torch.load(cp, map_location=self.device)
            model.load_state_dict(checkpoint["state_dict"])
            self.tokenizer = get_tokenizer("ViT-H-14")
            model = model.to(self.device)
            model.eval()

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        This computes the HPS scores for the given images and prompts.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (text prompts).
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images to score.
        """
        # Get images and prompts
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = inputs[0]  # Generated images
        prompts = x if isinstance(x, list) else [""] * len(images)

        model = self.model_dict["model"]
        preprocess_val = self.model_dict["preprocess_val"]

        with torch.inference_mode():
            for i, image in enumerate(images):
                # Convert tensor to PIL Image if needed
                if isinstance(image, torch.Tensor):
                    image = self._tensor_to_pil(image)

                prompt = prompts[i] if i < len(prompts) else ""

                # Process the image
                image_tensor = preprocess_val(image).unsqueeze(0).to(device=self.device, non_blocking=True)
                # Process the prompt
                text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                # Calculate the HPS
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = model(image_tensor, text)
                    image_features = outputs["image_features"]
                    text_features = outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image).cpu().detach().numpy()

                self.total += hps_score[0]
                self.count += 1

    def compute(self) -> MetricResult:
        """
        Compute the average HPS metric based on previous updates.

        Returns
        -------
        MetricResult
            The average HPS metric.
        """
        result = self.total / self.count if self.count.item() != 0 else torch.zeros(1)
        return MetricResult(self.metric_name, self.__dict__.copy(), result.item())

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert a tensor to a PIL Image.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to convert. Expected shape: (C, H, W) or (B, C, H, W).

        Returns
        -------
        Image.Image
            The converted PIL Image.
        """
        # Handle batch dimension
        if tensor.ndim == 4:
            tensor = tensor[0]

        # Ensure values are in [0, 1]
        if tensor.max() > 1:
            tensor = tensor / 255.0

        # Convert to numpy and then to PIL
        numpy_image = tensor.cpu().numpy()
        numpy_image = (numpy_image * 255).astype("uint8")
        return Image.fromarray(numpy_image.transpose(1, 2, 0))
