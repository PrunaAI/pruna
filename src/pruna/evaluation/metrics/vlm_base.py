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
VLM (Vision-Language Model) base classes for metrics.

This module provides two VLM implementations:
1. LitellmVLM - Uses litellm for API-based VLM calls (supports 100+ providers)
2. TransformersVLM - Uses local VLM models from HuggingFace Transformers
"""

from __future__ import annotations

import base64
import io
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
from PIL import Image

from pruna.logging.logger import pruna_logger


class BaseVLM(ABC):
    """Base class for Vision-Language Models."""

    @abstractmethod
    def generate(self, images: List[Image.Image], prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for images and prompts."""
        pass

    @abstractmethod
    def score(self, images: List[Image.Image], questions: List[str], answers: List[str], **kwargs) -> List[float]:
        """Score how well answers match images for given questions."""
        pass


class LitellmVLM(BaseVLM):
    """
    VLM using litellm for API-based inference.
    Supports 100+ LLM providers (OpenAI, Anthropic, Azure, etc.)
    Default model is gpt-4o.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.extra_kwargs = kwargs

        try:
            import litellm
            litellm.drop_params = True
            self._litellm = litellm
        except ImportError:
            pruna_logger.error("litellm not installed. Install with: pip install litellm")
            raise

    def generate(self, images: List[Image.Image], prompts: List[str], **kwargs) -> List[str]:
        results = []
        for image, prompt in zip(images, prompts):
            try:
                # Use synchronous completion, not async
                response = self._litellm.completion(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                        ]
                    }],
                    api_key=self.api_key,
                    **self.extra_kwargs,
                    **kwargs,
                )
                results.append(response.choices[0].message.content)
            except Exception as e:
                pruna_logger.error(f"Litellm generation failed: {e}")
                results.append("")
        return results

    def score(self, images: List[Image.Image], questions: List[str], answers: List[str], **kwargs) -> List[float]:
        scores = []
        for image, question, answer in zip(images, questions, answers):
            prompt = f"{question} Answer with just Yes or No."
            response = self.generate([image], [prompt], **kwargs)[0].lower()
            score = 1.0 if answer.lower() in response else 0.0
            scores.append(score)
        return scores

    def _image_to_data_url(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"


class TransformersVLM(BaseVLM):
    """
    VLM using HuggingFace Transformers for local inference.
    Supports models like BLIP, LLaVA, etc.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: Optional[str | torch.device] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.extra_kwargs = kwargs
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessorForVision2Seq, AutoModelForVision2Seq
        except ImportError:
            pruna_logger.error("transformers not installed. Install with: pip install transformers")
            raise
        pruna_logger.info(f"Loading VLM model: {self.model_name}")
        self._processor = AutoProcessorForVision2Seq.from_pretrained(self.model_name)
        self._model = AutoModelForVision2Seq.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def generate(self, images: List[Image.Image], prompts: List[str], **kwargs) -> List[str]:
        self._load_model()
        results = []
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        with torch.inference_mode():
            for image, prompt in zip(images, prompts):
                inputs = self._processor(images=[image], text=prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                output = self._model.generate(**inputs, max_new_tokens=max_new_tokens, **self.extra_kwargs)
                response = self._processor.decode(output[0], skip_special_tokens=True)
                results.append(response)
        return results

    def score(self, images: List[Image.Image], questions: List[str], answers: List[str], **kwargs) -> List[float]:
        scores = []
        for image, question, answer in zip(images, questions, answers):
            prompt = f"Question: {question} Answer:"
            responses = self.generate([image], [prompt], **kwargs)
            response = responses[0].lower()
            score = 1.0 if answer.lower() in response else 0.0
            scores.append(score)
        return scores
