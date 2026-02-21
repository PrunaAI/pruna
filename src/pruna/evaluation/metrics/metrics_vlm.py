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
VLM-based metrics for Pruna.

Metrics using Vision-Language Models for evaluation.
Supports LitellmVLM (API-based) and TransformersVLM (local models).
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Literal, Optional

import numpy as np
import torch
from PIL import Image

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import get_call_type_for_single_metric, metric_data_processor, SINGLE
from pruna.evaluation.metrics.vlm_base import BaseVLM, LitellmVLM, TransformersVLM


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.max() > 1:
        tensor = tensor / 255.0
    np_img = (tensor.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(np_img.transpose(1, 2, 0))


def _process_images(images: torch.Tensor) -> List[Image.Image]:
    return [_tensor_to_pil(img) if isinstance(img, torch.Tensor) else img for img in images]


# VQA Metric
@MetricRegistry.register("vqa")
class VQAMetric(StatefulMetric):
    """VQA metric using VLM."""
    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = "vqa"
    runs_on: List[str] = ["cpu"]  # API-based, doesn't need GPU

    def __init__(self, *args, vlm_type: Literal["litellm", "transformers"] = "litellm",
                 model_name: str = "gpt-4o", device=None, api_key: Optional[str] = None,
                 call_type: str = SINGLE, **kwargs):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        self.vlm = self._create_vlm(vlm_type, model_name, device, api_key)
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def _create_vlm(self, vlm_type: str, model_name: str, device: Any, api_key: Optional[str]) -> BaseVLM:
        if vlm_type == "litellm":
            return LitellmVLM(model_name=model_name, api_key=api_key)
        return TransformersVLM(model_name=model_name, device=device)

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = x if isinstance(x, list) else [""] * len(images)
        for i, image in enumerate(images):
            prompt = prompts[i] if i < len(prompts) else ""
            question = f'Does this image show "{prompt}"? Answer Yes or No.'
            score = self.vlm.score([image], [question], ["Yes"])[0]
            self.scores.append(score)

    def compute(self) -> MetricResult:
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))


# Alignment Score Metric
@MetricRegistry.register("alignment_score")
class AlignmentScoreMetric(StatefulMetric):
    """Alignment Score metric using VLM."""
    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = "alignment_score"
    runs_on: List[str] = ["cpu"]

    def __init__(self, *args, vlm_type: Literal["litellm", "transformers"] = "litellm",
                 model_name: str = "gpt-4o", device=None, api_key: Optional[str] = None,
                 call_type: str = SINGLE, **kwargs):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        self.vlm = self._create_vlm(vlm_type, model_name, device, api_key)
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def _create_vlm(self, vlm_type: str, model_name: str, device: Any, api_key: Optional[str]) -> BaseVLM:
        if vlm_type == "litellm":
            return LitellmVLM(model_name=model_name, api_key=api_key)
        return TransformersVLM(model_name=model_name, device=device)

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = x if isinstance(x, list) else [""] * len(images)
        for i, image in enumerate(images):
            prompt = prompts[i] if i < len(prompts) else ""
            question = f'Does this image show "{prompt}"? Answer Yes or No.'
            score = self.vlm.score([image], [question], ["Yes"])[0]
            self.scores.append(score)

    def compute(self) -> MetricResult:
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))


# Image Edit Score Metric
@MetricRegistry.register("img_edit_score")
class ImageEditScoreMetric(StatefulMetric):
    """Image Edit Score metric using VLM."""
    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = "img_edit_score"
    runs_on: List[str] = ["cpu"]

    def __init__(self, *args, vlm_type: Literal["litellm", "transformers"] = "litellm",
                 model_name: str = "gpt-4o", device=None, api_key: Optional[str] = None,
                 call_type: str = SINGLE, **kwargs):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        self.vlm = self._create_vlm(vlm_type, model_name, device, api_key)
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def _create_vlm(self, vlm_type: str, model_name: str, device: Any, api_key: Optional[str]) -> BaseVLM:
        if vlm_type == "litellm":
            return LitellmVLM(model_name=model_name, api_key=api_key)
        return TransformersVLM(model_name=model_name, device=device)

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = x if isinstance(x, list) else [""] * len(images)
        for i, image in enumerate(images):
            prompt = prompts[i] if i < len(prompts) else ""
            question = f'Rate 0-10: Does this image show "{prompt}"? Reply with a number.'
            responses = self.vlm.generate([image], [question])
            score = self._parse_score(responses[0])
            self.scores.append(score)

    def _parse_score(self, response: str) -> float:
        numbers = re.findall(r'\d+', response)
        return min(float(numbers[0]), 10.0) / 10.0 if numbers else 0.0

    def compute(self) -> MetricResult:
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))


# QA Accuracy Metric
@MetricRegistry.register("qa_accuracy")
class QAAccuracyMetric(StatefulMetric):
    """QA Accuracy metric using VLM."""
    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = "qa_accuracy"
    runs_on: List[str] = ["cpu"]

    def __init__(self, *args, vlm_type: Literal["litellm", "transformers"] = "litellm",
                 model_name: str = "gpt-4o", device=None, api_key: Optional[str] = None,
                 call_type: str = SINGLE, **kwargs):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        self.vlm = self._create_vlm(vlm_type, model_name, device, api_key)
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def _create_vlm(self, vlm_type: str, model_name: str, device: Any, api_key: Optional[str]) -> BaseVLM:
        if vlm_type == "litellm":
            return LitellmVLM(model_name=model_name, api_key=api_key)
        return TransformersVLM(model_name=model_name, device=device)

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        for image in images:
            question = "What is in this image? Answer:"
            responses = self.vlm.generate([image], [question])
            score = 1.0 if responses[0].strip() else 0.0
            self.scores.append(score)

    def compute(self) -> MetricResult:
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))


# Text Score Metric
@MetricRegistry.register("text_score")
class TextScoreMetric(StatefulMetric):
    """Text Score metric for text rendering using VLM."""
    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = False  # Lower is better
    metric_name: str = "text_score"
    runs_on: List[str] = ["cpu"]

    def __init__(self, *args, vlm_type: Literal["litellm", "transformers"] = "litellm",
                 model_name: str = "gpt-4o", device=None, api_key: Optional[str] = None,
                 call_type: str = SINGLE, **kwargs):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        self.vlm = self._create_vlm(vlm_type, model_name, device, api_key)
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def _create_vlm(self, vlm_type: str, model_name: str, device: Any, api_key: Optional[str]) -> BaseVLM:
        if vlm_type == "litellm":
            return LitellmVLM(model_name=model_name, api_key=api_key)
        return TransformersVLM(model_name=model_name, device=device)

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        for image in images:
            prompt = "Extract all text from this image. If no text, say 'No text'."
            responses = self.vlm.generate([image], [prompt])
            score = 0.0 if responses[0].strip().lower() != "no text" else 10.0
            self.scores.append(score)

    def compute(self) -> MetricResult:
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))


# VieScore Metric
@MetricRegistry.register("viescore")
class VieScoreMetric(StatefulMetric):
    """VieScore metric for image quality using VLM."""
    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = "viescore"
    runs_on: List[str] = ["cpu"]

    def __init__(self, *args, vlm_type: Literal["litellm", "transformers"] = "litellm",
                 model_name: str = "gpt-4o", device=None, api_key: Optional[str] = None,
                 call_type: str = SINGLE, **kwargs):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        self.vlm = self._create_vlm(vlm_type, model_name, device, api_key)
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def _create_vlm(self, vlm_type: str, model_name: str, device: Any, api_key: Optional[str]) -> BaseVLM:
        if vlm_type == "litellm":
            return LitellmVLM(model_name=model_name, api_key=api_key)
        return TransformersVLM(model_name=model_name, device=device)

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = x if isinstance(x, list) else [""] * len(images)
        for i, image in enumerate(images):
            prompt = prompts[i] if i < len(prompts) else ""
            sem_prompt = f'Rate 0-10: Does this image show "{prompt}"?'
            sem_resp = self.vlm.generate([image], [sem_prompt])[0]
            sem_score = self._parse_score(sem_resp)
            qual_prompt = "Rate 0-10: How natural is this image? Any artifacts?"
            qual_resp = self.vlm.generate([image], [qual_prompt])[0]
            qual_score = self._parse_score(qual_resp)
            score = math.sqrt(sem_score * qual_score) / 10.0
            self.scores.append(score)

    def _parse_score(self, response: str) -> float:
        numbers = re.findall(r'\d+', response)
        return min(float(numbers[0]), 10.0) if numbers else 0.0

    def compute(self) -> MetricResult:
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))
