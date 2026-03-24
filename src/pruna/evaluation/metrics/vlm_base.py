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

Both support structured generation for stable outputs:
- LitellmVLM: Uses pydantic models with response_format
- TransformersVLM: Uses outlines for constrained decoding.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional, Type, TypeVar, Union

import torch
from PIL import Image
from pydantic import BaseModel

from pruna.logging.logger import pruna_logger

T = TypeVar("T", bound=BaseModel)


def get_vlm(
    vlm: Optional[BaseVLM] = None,
    vlm_type: Literal["litellm", "transformers"] = "litellm",
    model_name: str = "gpt-4o",
    device: Optional[str | torch.device] = None,
    api_key: Optional[str] = None,
    use_outlines: bool = False,
    **vlm_kwargs: Any,
) -> BaseVLM:
    """
    Create or return a VLM instance.

    Parameters
    ----------
    vlm : BaseVLM | None
        If provided, returned as-is. Otherwise a VLM is created.
    vlm_type : {"litellm", "transformers"}
        Backend when creating a VLM.
    model_name : str
        Model name for litellm or HuggingFace.
    device : str | torch.device | None
        Device for transformers VLM.
    api_key : str | None
        API key for litellm.
    use_outlines : bool
        Use outlines for transformers.
    **vlm_kwargs : Any
        Extra kwargs passed to LitellmVLM or TransformersVLM.
        For TransformersVLM, use model_load_kwargs={"torch_dtype": torch.bfloat16}
        to pass options to from_pretrained.

    Returns
    -------
    BaseVLM
        The VLM instance.
    """
    if vlm is not None:
        return vlm
    if vlm_type == "litellm":
        return LitellmVLM(model_name=model_name, api_key=api_key, **vlm_kwargs)
    model_load_kwargs = vlm_kwargs.pop("model_load_kwargs", {})
    return TransformersVLM(
        model_name=model_name,
        device=device,
        use_outlines=use_outlines,
        model_load_kwargs=model_load_kwargs,
        **vlm_kwargs,
    )


class BaseVLM(ABC):
    """Base class for Vision-Language Models."""

    @abstractmethod
    def generate(
        self,
        images: List[Image.Image],
        prompts: List[str],
        response_format: Optional[Union[Type[BaseModel], Literal["integer"], Literal["yes_no"], Literal["json"]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate responses for images and prompts.

        Parameters
        ----------
        images : List[Image.Image]
            List of PIL Images.
        prompts : List[str]
            List of text prompts.
        response_format : Type[BaseModel] | str | None
            Optional pydantic model (litellm) or format string: "integer", "yes_no", "json" (transformers/outlines).
        **kwargs : Any
            Additional arguments passed to the implementation.

        Returns
        -------
        List[str]
            Generated responses.
        """
        pass

    @abstractmethod
    def score(
        self,
        images: List[Image.Image],
        questions: List[str],
        answers: List[str],
        use_probability: bool = False,
        response_format: Optional[Union[Type[BaseModel], Literal["integer"], Literal["yes_no"], Literal["json"]]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Score how well answers match images for given questions.

        Parameters
        ----------
        images : List[Image.Image]
            List of PIL Images.
        questions : List[str]
            List of questions.
        answers : List[str]
            List of expected answers.
        use_probability : bool, optional
            If True and supported, return P(expected answer) instead of binary 0/1.
        response_format : Type[BaseModel] | str | None, optional
            Structured output format. When set, uses generate() with this format and
            extracts the answer field for comparison instead of raw string matching.
        **kwargs : Any
            Additional arguments passed to the implementation.

        Returns
        -------
        List[float]
            Scores for each image-question pair (0-1, or probability when use_probability).
        """
        pass


class LitellmVLM(BaseVLM):
    """
    VLM using litellm for API-based inference.

    Supports 100+ LLM providers (OpenAI, Anthropic, Azure, etc.)
    Default model is gpt-4o.

    Parameters
    ----------
    model_name : str, optional
        Model name (e.g., gpt-4o). Default is "gpt-4o".
    api_key : str | None, optional
        API key for the provider. Uses LITELLM_API_KEY or OPENAI_API_KEY env if None.
    **kwargs : Any
        Additional arguments passed to litellm.
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

    def generate(
        self,
        images: List[Image.Image],
        prompts: List[str],
        response_format: Optional[Union[Type[BaseModel], Literal["integer"], Literal["yes_no"], Literal["json"]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate responses for images and prompts.

        Parameters
        ----------
        images : List[Image.Image]
            List of PIL Images.
        prompts : List[str]
            List of text prompts.
        response_format : Type[BaseModel] | str | None
            Optional pydantic model for structured output (litellm uses BaseModel).
        **kwargs : Any
            Additional arguments passed to litellm completion.

        Returns
        -------
        List[str]
            Generated responses.
        """
        results = []
        for image, prompt in zip(images, prompts):
            try:
                # Prepare message content
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                ]
                # Prepare completion kwargs
                completion_kwargs = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": content}],
                    "api_key": self.api_key,
                    **self.extra_kwargs,
                    **kwargs,
                }
                # Add structured generation if requested (litellm uses pydantic models only)
                if response_format is not None and isinstance(response_format, type):
                    completion_kwargs["response_format"] = response_format
                # Use synchronous completion
                response = self._litellm.completion(**completion_kwargs)
                content_result = response.choices[0].message.content
                # If using pydantic, content is already parsed
                use_pydantic = (
                    response_format is not None
                    and isinstance(response_format, type)
                    and isinstance(content_result, response_format)
                )
                if use_pydantic:
                    # Return JSON string representation
                    results.append(content_result.model_dump_json())
                else:
                    results.append(content_result)
            except Exception as e:
                pruna_logger.error(f"Litellm generation failed: {e}")
                results.append("")
        return results

    def score(
        self,
        images: List[Image.Image],
        questions: List[str],
        answers: List[str],
        use_probability: bool = False,
        response_format: Optional[Union[Type[BaseModel], Literal["integer"], Literal["yes_no"], Literal["json"]]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Score how well answers match images for given questions.

        When use_probability=True, requests logprobs from the API and returns P(expected).
        When response_format is set, uses structured generation and extracts the answer field.
        Falls back to binary 0/1 if logprobs not available.

        Parameters
        ----------
        images : List[Image.Image]
            List of PIL Images.
        questions : List[str]
            List of questions.
        answers : List[str]
            List of expected answers.
        use_probability : bool, optional
            If True, return P(expected) from logprobs when available. Default is False.
        response_format : Type[BaseModel] | str | None, optional
            Structured output format for answer extraction.
        **kwargs : Any
            Additional arguments passed to litellm completion.

        Returns
        -------
        List[float]
            Scores for each image-question pair (0-1, or probability when use_probability).
        """
        from pruna.evaluation.metrics.metric_vlm_utils import get_answer_from_response

        scores = []
        for image, question, answer in zip(images, questions, answers):
            prompt = f"{question} Please answer yes or no."
            if use_probability:
                score = self._score_with_logprobs(image, prompt, answer, **kwargs)
            elif response_format is not None:
                raw = self.generate([image], [prompt], response_format=response_format, **kwargs)[0]
                response_answer = get_answer_from_response(raw)
                score = 1.0 if answer.lower() in response_answer.lower() else 0.0
            else:
                response = self.generate([image], [prompt], **kwargs)[0].lower()
                score = 1.0 if answer.lower() in response else 0.0
            scores.append(score)
        return scores

    def _score_with_logprobs(self, image: Image.Image, prompt: str, expected: str, **kwargs: Any) -> float:
        """
        Get P(expected) from logprobs when available.

        Parameters
        ----------
        image : Image.Image
            PIL Image to score.
        prompt : str
            Question prompt.
        expected : str
            Expected answer (e.g., "Yes").
        **kwargs : Any
            Additional arguments passed to litellm completion.

        Returns
        -------
        float
            Probability of expected answer (0-1), or binary 0/1 on fallback.
        """
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
        ]
        completion_kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "api_key": self.api_key,
            "logprobs": True,
            "top_logprobs": 5,
            **self.extra_kwargs,
            **kwargs,
        }
        try:
            response = self._litellm.completion(**completion_kwargs)
            choice = response.choices[0]
            logprobs = getattr(choice, "logprobs", None) or getattr(choice.message, "logprobs", None)
            if logprobs and hasattr(logprobs, "content"):
                for tok in logprobs.content or []:
                    top = getattr(tok, "top_logprobs", None) or []
                    for t in top:
                        token_str = getattr(t, "token", "") or str(t).lower()
                        if token_str and expected.lower() in token_str.lower():
                            logprob = float(getattr(t, "logprob", -1e9) or -1e9)
                            return min(1.0, max(0.0, math.exp(logprob)))
            content_str = (choice.message.content or "").lower()
            if expected.lower() in content_str:
                return 1.0
            return 0.0
        except Exception:
            response = self.generate([image], [prompt], **kwargs)[0].lower()
            return 1.0 if expected.lower() in response else 0.0

    def _image_to_data_url(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"


class TransformersVLM(BaseVLM):
    """
    VLM using HuggingFace Transformers for local inference.

    Supports models like BLIP, LLaVA, SmolVLM, etc.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model name. Default is "Salesforce/blip2-opt-2.7b".
    device : str | torch.device | None, optional
        Device for inference. Auto-detected if None.
    use_outlines : bool, optional
        Use outlines for constrained decoding. Default is False.
    model_load_kwargs : dict, optional
        Kwargs passed to from_pretrained (e.g. torch_dtype, attn_implementation).
    **kwargs : Any
        Additional arguments passed to model.generate.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: Optional[str | torch.device] = None,
        use_outlines: bool = False,
        model_load_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.use_outlines = use_outlines
        self.model_load_kwargs = model_load_kwargs or {}
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
        self._outlines_model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError:
            pruna_logger.error("transformers not installed. Install with: pip install transformers")
            raise
        pruna_logger.info(f"Loading VLM model: {self.model_name}")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(self.model_name, **self.model_load_kwargs)
        device = self.device
        self._model.to(device)  # type: ignore[invalid-argument-type]
        self._model.eval()

    def _load_outlines_model(self) -> None:
        """Lazily wrap the loaded multimodal model for Outlines structured generation."""
        if self._outlines_model is not None:
            return
        try:
            import outlines
        except ImportError:
            pruna_logger.warning("outlines not installed, using standard generation")
            return
        self._load_model()
        self._outlines_model = outlines.from_transformers(self._model, self._processor)

    def _get_outlines_output_type(
        self,
        response_format: Optional[Union[Type[BaseModel], Literal["integer"], Literal["yes_no"], Literal["json"]]],
    ) -> Any:
        """Map current response formats to an Outlines-compatible output type."""
        if response_format is None:
            return None
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format
        if response_format == "integer":
            return int
        if response_format == "yes_no":
            return Literal["Yes", "No"]
        if response_format == "json":
            return dict
        return None

    @staticmethod
    def _serialize_outlines_result(result: Any) -> str:
        """Normalize Outlines results so the existing response parsers still work."""
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        if isinstance(result, (dict, list)):
            return json.dumps(result)
        return str(result)

    @staticmethod
    def _to_outlines_input(image: Image.Image, prompt: str) -> list[Any]:
        """Build a minimal multimodal input payload for Outlines."""
        from outlines.inputs import Image as OutlinesImage

        return [prompt, OutlinesImage(image)]

    def generate(
        self,
        images: List[Image.Image],
        prompts: List[str],
        response_format: Optional[Union[Type[BaseModel], Literal["integer"], Literal["yes_no"], Literal["json"]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate responses using local VLM.

        Parameters
        ----------
        images : List[Image.Image]
            List of PIL Images.
        prompts : List[str]
            List of text prompts.
        response_format : Type[BaseModel] | str | None
            Format constraint for outlines ("integer", "yes_no") or None.
        **kwargs : Any
            Additional arguments passed to model generate.

        Returns
        -------
        List[str]
            Generated responses.
        """
        self._load_model()
        results = []
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        if self.use_outlines and response_format is not None:
            results = self._generate_with_outlines(images, prompts, response_format, max_new_tokens)
        else:
            results = self._generate_standard(images, prompts, max_new_tokens)
        return results

    def _generate_with_outlines(
        self,
        images: List[Image.Image],
        prompts: List[str],
        response_format: Optional[Union[Type[BaseModel], Literal["integer"], Literal["yes_no"], Literal["json"]]],
        max_new_tokens: int,
    ) -> List[str]:
        """Generate using outlines for constrained decoding."""
        self._load_outlines_model()
        if self._outlines_model is None:
            return self._generate_standard(images, prompts, max_new_tokens)
        output_type = self._get_outlines_output_type(response_format)
        if output_type is None:
            return self._generate_standard(images, prompts, max_new_tokens)
        results = []
        for image, prompt in zip(images, prompts):
            try:
                model_input = self._to_outlines_input(image, prompt)
                output = self._outlines_model(model_input, output_type, max_new_tokens=max_new_tokens)
                results.append(self._serialize_outlines_result(output))
            except Exception as e:
                pruna_logger.warning(f"Outlines generation failed: {e}, using standard")
                results.extend(self._generate_standard([image], [prompt], max_new_tokens))
        return results

    def _prepare_inputs(self, image: Image.Image, prompt: str) -> dict:
        """Prepare model inputs, supporting both BLIP-style and chat-template processors."""
        try:
            inputs = self._processor(images=[image], text=prompt, return_tensors="pt")
        except (ValueError, TypeError):
            conversation = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
            ]
            inputs = self._processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _decode_output(self, output_ids: torch.Tensor) -> str:
        """Decode model output to text."""
        if hasattr(self._processor, "batch_decode"):
            return self._processor.batch_decode([output_ids], skip_special_tokens=True)[0]
        return self._processor.decode(output_ids, skip_special_tokens=True)

    def _generate_standard(
        self,
        images: List[Image.Image],
        prompts: List[str],
        max_new_tokens: int,
    ) -> List[str]:
        """Standard generation without outlines."""
        results = []
        with torch.inference_mode():
            for image, prompt in zip(images, prompts):
                inputs = self._prepare_inputs(image, prompt)
                output = self._model.generate(**inputs, max_new_tokens=max_new_tokens, **self.extra_kwargs)
                response = self._decode_output(output[0])
                results.append(response)
        return results

    def score(
        self,
        images: List[Image.Image],
        questions: List[str],
        answers: List[str],
        use_probability: bool = False,
        response_format: Optional[Union[Type[BaseModel], Literal["integer"], Literal["yes_no"], Literal["json"]]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """
        Score how well answers match images for given questions.

        use_probability is not supported for TransformersVLM; uses binary 0/1.
        When response_format is set, uses structured generation and extracts the answer field.

        Parameters
        ----------
        images : List[Image.Image]
            List of PIL Images.
        questions : List[str]
            List of questions.
        answers : List[str]
            List of expected answers.
        use_probability : bool, optional
            Ignored; TransformersVLM always uses binary 0/1.
        response_format : Type[BaseModel] | str | None, optional
            Structured output format for answer extraction.
        **kwargs : Any
            Additional arguments passed to generate.

        Returns
        -------
        List[float]
            Scores for each image-question pair (0 or 1).
        """
        from pruna.evaluation.metrics.metric_vlm_utils import get_answer_from_response

        scores = []
        for image, question, answer in zip(images, questions, answers):
            prompt = f"{question} Please answer yes or no."
            responses = self.generate([image], [prompt], response_format=response_format, **kwargs)
            raw = responses[0] if responses else ""
            response_answer = get_answer_from_response(raw) if response_format is not None else raw.lower()
            score = 1.0 if answer.lower() in response_answer.lower() else 0.0
            scores.append(score)
        return scores
