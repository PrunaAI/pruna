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

References
----------
VQAScore: https://arxiv.org/abs/2310.08868
VieScore: https://github.com/ByteDance/IEA-eval
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Literal, Optional

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.evaluation.metrics.vlm_base import LitellmVLM, TransformersVLM


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    from PIL import Image

    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.max() > 1:
        tensor = tensor / 255.0
    np_img = (tensor.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(np_img.transpose(1, 2, 0))


def _process_images(images: torch.Tensor) -> List[Any]:
    return [_tensor_to_pil(img) if isinstance(img, torch.Tensor) else img for img in images]


# Pydantic models for structured generation
class VQAnswer(BaseModel):
    """Structured output for VQA."""

    answer: str
    confidence: float = 1.0


class ScoreOutput(BaseModel):
    """Structured output for scoring metrics."""

    score: float
    reasoning: Optional[str] = None


# VQA Metric
@MetricRegistry.register("vqa")
class VQAMetric(StatefulMetric):
    """
    VQA (Visual Question Answering) metric.

    Uses VLM to answer questions about images and compare with expected answers.
    Higher scores indicate better image-text alignment.

    Reference
    ----------
    VQAScore: Uses VLM for VQA-based image evaluation
    https://arxiv.org/abs/2310.08868

    Parameters
    ----------
    vlm_type : {"litellm", "transformers"}, optional
        VLM backend to use. Default is "litellm".
    model_name : str, optional
        Model name (gpt-4o for litellm, model path for transformers).
    structured_output : bool, optional
        Use structured generation for stable outputs. Default is True.
    use_outlines : bool, optional
        Use outlines for transformers. Default is False.
    device : str | torch.device | None, optional
        Device for transformers VLM.
    api_key : str | None, optional
        API key for litellm.
    **kwargs : Any
        Additional arguments.
    """

    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = "vqa"
    runs_on: List[str] = ["cpu"]

    def __init__(
        self,
        *args,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str = "gpt-4o",
        structured_output: bool = True,
        use_outlines: bool = False,
        device=None,
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        **kwargs,
    ):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)
        self.structured_output = structured_output

        # Create VLM with structured generation support
        if vlm_type == "litellm":
            self.vlm = LitellmVLM(model_name=model_name, api_key=api_key)
            self.response_format = VQAnswer if structured_output else None
        else:
            self.vlm = TransformersVLM(model_name=model_name, device=device, use_outlines=use_outlines)
            self.response_format = "yes_no" if structured_output else None

        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (text prompts).
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images to score.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = x if isinstance(x, list) else [""] * len(images)

        for i, image in enumerate(images):
            prompt = prompts[i] if i < len(prompts) else ""
            question = f'Does this image show "{prompt}"? Answer Yes or No.'
            score = self.vlm.score([image], [question], ["Yes"], response_format=self.response_format)[0]
            self.scores.append(score)

    def compute(self) -> MetricResult:
        """
        Compute the metric result.

        Returns
        -------
        MetricResult
            The computed metric result.
        """
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))


# Alignment Score Metric
@MetricRegistry.register("alignment_score")
class AlignmentScoreMetric(StatefulMetric):
    """
    Alignment Score metric using VLM.

    Assesses how well generated images match text prompts through structured questioning.
    Higher scores indicate better alignment.

    Reference
    ----------
    Uses VLM for image-text alignment evaluation.

    Parameters
    ----------
    vlm_type : {"litellm", "transformers"}, optional
        VLM backend. Default is "litellm".
    structured_output : bool, optional
        Use structured generation. Default is True.
    **kwargs : Any
        Additional arguments.
    """

    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = "alignment_score"
    runs_on: List[str] = ["cpu"]

    def __init__(
        self,
        *args,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str = "gpt-4o",
        structured_output: bool = True,
        use_outlines: bool = False,
        device=None,
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        **kwargs,
    ):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)

        if vlm_type == "litellm":
            self.vlm = LitellmVLM(model_name=model_name, api_key=api_key)
            self.response_format = ScoreOutput if structured_output else None
        else:
            self.vlm = TransformersVLM(model_name=model_name, device=device, use_outlines=use_outlines)
            self.response_format = "integer" if structured_output else None

        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (text prompts).
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images to score.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = x if isinstance(x, list) else [""] * len(images)
        for i, image in enumerate(images):
            prompt = prompts[i] if i < len(prompts) else ""
            question = f'Does this image show "{prompt}"? Answer Yes or No.'
            score = self.vlm.score([image], [question], ["Yes"], response_format=self.response_format)[0]
            self.scores.append(score)

    def compute(self) -> MetricResult:
        """
        Compute the metric result.

        Returns
        -------
        MetricResult
            The computed metric result.
        """
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))


# Image Edit Score Metric
@MetricRegistry.register("img_edit_score")
class ImageEditScoreMetric(StatefulMetric):
    """
    Image Edit Score metric.

    Evaluates how well an image was edited based on editing instructions.
    Higher scores indicate better editing quality.

    Reference
    ----------
    VieScore: https://github.com/ByteDance/IEA-eval
    """

    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = "img_edit_score"
    runs_on: List[str] = ["cpu"]

    def __init__(
        self,
        *args,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str = "gpt-4o",
        structured_output: bool = True,
        use_outlines: bool = False,
        device=None,
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        **kwargs,
    ):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)

        if vlm_type == "litellm":
            self.vlm = LitellmVLM(model_name=model_name, api_key=api_key)
            self.response_format = ScoreOutput if structured_output else None
        else:
            self.vlm = TransformersVLM(model_name=model_name, device=device, use_outlines=use_outlines)
            self.response_format = "integer" if structured_output else None

        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (text prompts).
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images to score.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = x if isinstance(x, list) else [""] * len(images)
        for i, image in enumerate(images):
            prompt = prompts[i] if i < len(prompts) else ""
            question = f'Rate 0-10: Does this image show "{prompt}"? Reply with a number.'
            responses = self.vlm.generate([image], [question], response_format=self.response_format)
            score = self._parse_score(responses[0])
            self.scores.append(score)

    def _parse_score(self, response: str) -> float:
        if isinstance(response, str):
            numbers = re.findall(r"\d+", response)
            return min(float(numbers[0]), 10.0) / 10.0 if numbers else 0.0
        return 0.0

    def compute(self) -> MetricResult:
        """
        Compute the metric result.

        Returns
        -------
        MetricResult
            The computed metric result.
        """
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))


# QA Accuracy Metric
@MetricRegistry.register("qa_accuracy")
class QAAccuracyMetric(StatefulMetric):
    """
    QA Accuracy metric.

    Uses VLM to answer questions about images.
    Higher scores indicate better image understanding.
    """

    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = "qa_accuracy"
    runs_on: List[str] = ["cpu"]

    def __init__(
        self,
        *args,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str = "gpt-4o",
        structured_output: bool = True,
        use_outlines: bool = False,
        device=None,
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        **kwargs,
    ):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)

        if vlm_type == "litellm":
            self.vlm = LitellmVLM(model_name=model_name, api_key=api_key)
            self.response_format = VQAnswer if structured_output else None
        else:
            self.vlm = TransformersVLM(model_name=model_name, device=device, use_outlines=use_outlines)
            self.response_format = None  # No constraint for open QA

        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (text prompts).
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images to score.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        for image in images:
            question = "What is in this image? Answer:"
            responses = self.vlm.generate([image], [question], response_format=self.response_format)
            score = 1.0 if responses and responses[0].strip() else 0.0
            self.scores.append(score)

    def compute(self) -> MetricResult:
        """
        Compute the metric result.

        Returns
        -------
        MetricResult
            The computed metric result.
        """
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))


# Text Score Metric
@MetricRegistry.register("text_score")
class TextScoreMetric(StatefulMetric):
    """
    Text Score metric for evaluating text rendering in images.

    Uses VLM for OCR to extract text and compare with ground truth.
    Lower scores (edit distance) are better.
    """

    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = False
    metric_name: str = "text_score"
    runs_on: List[str] = ["cpu"]

    def __init__(
        self,
        *args,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str = "gpt-4o",
        structured_output: bool = True,
        use_outlines: bool = False,
        device=None,
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        **kwargs,
    ):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)

        if vlm_type == "litellm":
            self.vlm = LitellmVLM(model_name=model_name, api_key=api_key)
            self.response_format = None  # OCR is open-ended
        else:
            self.vlm = TransformersVLM(model_name=model_name, device=device, use_outlines=use_outlines)
            self.response_format = None

        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (text prompts).
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images to score.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        for image in images:
            prompt = "Extract all text from this image. If no text, say 'No text'."
            responses = self.vlm.generate([image], [prompt], response_format=self.response_format)
            score = 0.0 if responses and responses[0].strip().lower() != "no text" else 10.0
            self.scores.append(score)

    def compute(self) -> MetricResult:
        """
        Compute the metric result.

        Returns
        -------
        MetricResult
            The computed metric result.
        """
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))


# VieScore Metric
@MetricRegistry.register("viescore")
class VieScoreMetric(StatefulMetric):
    """
    VieScore metric for evaluating image quality (semantic + quality).

    Uses VLM to assess both semantic alignment and visual quality.
    Higher scores indicate better overall quality.

    Reference
    ----------
    VieScore: https://github.com/ByteDance/IEA-eval

    Computes:
    - Semantic score: How well image follows prompt
    - Quality score: Naturalness and artifacts
    - Overall: Geometric mean of semantic and quality
    """

    scores: List[float]
    default_call_type: str = "y"
    higher_is_better: bool = True
    metric_name: str = "viescore"
    runs_on: List[str] = ["cpu"]

    def __init__(
        self,
        *args,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str = "gpt-4o",
        structured_output: bool = True,
        use_outlines: bool = False,
        device=None,
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        **kwargs,
    ):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)

        if vlm_type == "litellm":
            self.vlm = LitellmVLM(model_name=model_name, api_key=api_key)
            self.response_format = ScoreOutput if structured_output else None
        else:
            self.vlm = TransformersVLM(model_name=model_name, device=device, use_outlines=use_outlines)
            self.response_format = "integer" if structured_output else None

        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    def update(self, x: List[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (text prompts).
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images to score.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = x if isinstance(x, list) else [""] * len(images)
        for i, image in enumerate(images):
            prompt = prompts[i] if i < len(prompts) else ""

            # Semantic score
            sem_prompt = f'Rate 0-10: Does this image show "{prompt}"?'
            sem_resp = self.vlm.generate([image], [sem_prompt], response_format=self.response_format)[0]
            sem_score = self._parse_score(sem_resp)

            # Quality score
            qual_prompt = "Rate 0-10: How natural is this image? Any artifacts?"
            qual_resp = self.vlm.generate([image], [qual_prompt], response_format=self.response_format)[0]
            qual_score = self._parse_score(qual_resp)

            # Overall = geometric mean
            score = math.sqrt(sem_score * qual_score) / 10.0
            self.scores.append(score)

    def _parse_score(self, response: str) -> float:
        if isinstance(response, str):
            numbers = re.findall(r"\d+", response)
            return min(float(numbers[0]), 10.0) if numbers else 0.0
        return 0.0

    def compute(self) -> MetricResult:
        """
        Compute the metric result.

        Returns
        -------
        MetricResult
            The computed metric result.
        """
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))
