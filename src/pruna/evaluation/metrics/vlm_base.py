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

Implementations
---------------
- **LitellmVLM** — API inference via ``litellm`` (many providers behind one client).
- **TransformersVLM** — local Hugging Face models on device.

Why LiteLLM for the default API path
--------------------------------------
Judge-style metrics need a capable vision-language model. Loading large VLMs locally is
expensive; routing through ``litellm`` keeps the default path lightweight and matches common
API-judge setups without bundling a full local VLM in every metric run.

API keys and environment
------------------------
For ``vlm_type="litellm"``, the key passed to the provider is resolved in this order:

1. The ``api_key`` argument on the metric or :func:`get_vlm`
2. ``LITELLM_API_KEY``
3. ``OPENAI_API_KEY``

Routes such as ``openai/gpt-4o`` use the OpenAI-compatible key. Other providers follow
LiteLLM’s environment conventions (for example ``ANTHROPIC_API_KEY`` for ``anthropic/...``).
The same ``OPENAI_API_KEY`` you use for other OpenAI-hosted judges (for example in pbench)
applies here. Replicate and similar tokens used by ``mine/`` demos or image backends are not
read by ``LitellmVLM``; configure those only for scripts that document them.

For a short user-facing summary of key order, hosted vs local, and a minimal ``transformers``
example, see :doc:`Evaluate a model </docs_pruna/user_manual/evaluate>` (Vision-language judge
metrics).

Choosing local vs API
---------------------
Metrics in :data:`VLM_METRIC_REGISTRY_NAMES` take ``vlm_type`` and ``model_name``:

- **API** (``vlm_type="litellm"``, default) — use a vision-capable route (e.g. ``openai/gpt-4o``;
  see :data:`~pruna.evaluation.vlm_benchmark_helpers.DEFAULT_LITELLM` in helpers).
- **Local** (``vlm_type="transformers"``) — e.g. SmolVLM for offline or CI.

The ``oneig_reasoning`` metric is separate: it runs the LLM2CLIP stack locally; see
``pruna.evaluation.metrics.metric_oneig_reasoning``.

Structured outputs
------------------
- LitellmVLM: pydantic ``response_format`` where applicable.
- TransformersVLM: Outlines 1.x constrained decoding via ``outlines.Generator`` and
  ``outlines.models.transformers.from_transformers`` (single- and multi-image ``Chat`` inputs).

Usage examples
----------------
Minimal LiteLLM and local ``transformers`` construction is shown under :func:`get_vlm`
(``Examples`` section). **Registry metrics** (``vqa``, ``qa_accuracy``, ``alignment_score``,
``img_edit_score``, OCR/text metrics, ``oneig_alignment``, ``vie_score``, …) take the same
``vlm_type``, ``model_name``, ``api_key``, and ``vlm_kwargs`` pattern; see
``StatefulVLMMeanScoresMetric`` in ``metric_vlm_base`` and each metric class docstring.

For VIEScore-style **text--image editing** metrics that pass two PIL images per prompt (source
then edited), call :meth:`LitellmVLM.generate_with_image_lists` or
:meth:`TransformersVLM.generate_with_image_lists` with ``image_lists[i]`` aligned to
``prompts[i]``.

Metric-level (hosted vs local)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``Task(request=["vqa"], ...)`` supplies ``model_name="openai/gpt-4o"`` for VLM registry names.
Override the backend by constructing a metric instance:

.. code-block:: python

    import torch

    from pruna.evaluation.metrics import VQAMetric

    hosted = VQAMetric(vlm_type="litellm", model_name="openai/gpt-4o")
    local = VQAMetric(
        vlm_type="transformers",
        model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        device="cpu",
        vlm_kwargs={"model_load_kwargs": {"torch_dtype": torch.float32}},
    )

``QAAccuracyMetric`` and other classes that call :func:`get_vlm` directly use the same
arguments (substitute the class name).
"""

from __future__ import annotations

import base64
import io
import math
import os
from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional, Type, TypeVar, Union

import torch
from PIL import Image
from pydantic import BaseModel

from pruna.logging.logger import pruna_logger

T = TypeVar("T", bound=BaseModel)

VLM_METRIC_REGISTRY_NAMES: frozenset[str] = frozenset(
    (
        "vqa",
        "qa_accuracy",
        "alignment_score",
        "img_edit_score",
        "text_score",
        "ocr_levenshtein",
        "ocr_text_score",
        "oneig_text_score",
        "oneig_alignment",
        "vie_score",
    )
)


def get_vlm(
    vlm: Optional[BaseVLM] = None,
    vlm_type: Literal["litellm", "transformers"] = "litellm",
    *,
    model_name: Optional[str] = None,
    device: Optional[str | torch.device] = None,
    api_key: Optional[str] = None,
    structured_output: bool = True,
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
    model_name : str | None
        Model name for litellm (e.g. ``openai/gpt-4o``) or HuggingFace ``from_pretrained`` id.
        **Required** when ``vlm`` is not provided. Ignored when ``vlm`` is provided.
    device : str | torch.device | None
        Device for transformers VLM.
    api_key : str | None
        API key for litellm.
    structured_output : bool
        When True, litellm uses pydantic ``response_format`` from the metric; for
        ``transformers``, enables outlines-based constrained decoding when a string
        format is passed to ``generate``/``score``.
    **vlm_kwargs : Any
        Same dict as ``vlm_kwargs`` on VLM metrics: forwarded to the backend chosen by
        ``vlm_type``. For ``"litellm"``, kwargs go to ``LitellmVLM`` (e.g. provider-specific
        options). For ``"transformers"``, use ``model_load_kwargs`` for
        ``AutoModelForImageTextToText.from_pretrained``; any other keys are passed to
        ``TransformersVLM`` after ``model_load_kwargs`` is popped.

    Returns
    -------
    BaseVLM
        The VLM instance.

    Notes
    -----
    When ``vlm_type`` is ``"litellm"`` and ``api_key`` is omitted, the key is taken from
    ``LITELLM_API_KEY`` or ``OPENAI_API_KEY``. See the module docstring above. User manual:
    :doc:`Evaluate a model </docs_pruna/user_manual/evaluate>` (Vision-language judge metrics).

    Examples
    --------
    Hosted (``litellm``) and local Hugging Face (``transformers``). API key for ``hosted`` from
    ``OPENAI_API_KEY`` or ``LITELLM_API_KEY`` if ``api_key`` is omitted.

    .. code-block:: python

        import torch

        from pruna.evaluation.metrics.vlm_base import get_vlm

        hosted = get_vlm(vlm_type="litellm", model_name="openai/gpt-4o")
        local = get_vlm(
            vlm_type="transformers",
            model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            device="cpu",
            model_load_kwargs={"torch_dtype": torch.float32},
        )

    Another LiteLLM provider route (set the env var that route expects, e.g.
    ``ANTHROPIC_API_KEY`` for ``anthropic/...``):

    .. code-block:: python

        from pruna.evaluation.metrics.vlm_base import get_vlm

        other_provider = get_vlm(
            vlm_type="litellm", model_name="anthropic/claude-3-5-sonnet-20241022"
        )
    """
    if vlm is not None:
        return vlm
    if not model_name:
        raise ValueError(
            "get_vlm requires model_name when vlm is not provided "
            '(pass model_name explicitly, e.g. model_name="openai/gpt-4o").'
        )
    if vlm_type == "litellm":
        return LitellmVLM(model_name=model_name, api_key=api_key, **vlm_kwargs)
    model_load_kwargs = vlm_kwargs.pop("model_load_kwargs", {})
    return TransformersVLM(
        model_name=model_name,
        device=device,
        use_outlines=structured_output,
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

    Supports many providers (OpenAI, Anthropic, Azure, and others) through a single client.

    Parameters
    ----------
    model_name : str
        Model name (e.g. ``openai/gpt-4o`` for litellm). Passed from :func:`get_vlm`.
    api_key : str | None, optional
        API key for the provider. If omitted, uses ``LITELLM_API_KEY`` then ``OPENAI_API_KEY``.
    **kwargs : Any
        Additional arguments passed to litellm.

    Notes
    -----
    LiteLLM is the default API backend so metric runs can use a hosted VLM judge without
    downloading large local checkpoints. Provider-specific environment variables are described
    in the LiteLLM documentation; OpenAI-compatible routes typically use ``OPENAI_API_KEY``.
    User manual: :doc:`Evaluate a model </docs_pruna/user_manual/evaluate>` (Vision-language
    judge metrics).

    Examples
    --------
    OpenAI-compatible route (pass ``api_key`` explicitly, or rely on ``OPENAI_API_KEY`` /
    ``LITELLM_API_KEY`` when omitted):

    >>> from pruna.evaluation.metrics.vlm_base import LitellmVLM
    >>> hosted = LitellmVLM(model_name="openai/gpt-4o", api_key="sk-placeholder")
    >>> hosted.api_key == "sk-placeholder"
    True

    Same naming as :func:`get_vlm` examples: ``other_provider`` for non-OpenAI LiteLLM routes
    (set ``ANTHROPIC_API_KEY``, etc.):

    .. code-block:: python

        from pruna.evaluation.metrics.vlm_base import LitellmVLM

        other_provider = LitellmVLM(model_name="anthropic/claude-3-5-sonnet-20241022")
    """

    def __init__(
        self,
        model_name: str,
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

    def _litellm_chat_completion(
        self,
        content: list[dict[str, Any]],
        response_format: Optional[Union[Type[BaseModel], Literal["integer"], Literal["yes_no"], Literal["json"]]] = None,
        **kwargs: Any,
    ) -> str:
        completion_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "api_key": self.api_key,
            **self.extra_kwargs,
            **kwargs,
        }
        if response_format is not None and isinstance(response_format, type):
            completion_kwargs["response_format"] = response_format
        response = self._litellm.completion(**completion_kwargs)
        content_result = response.choices[0].message.content
        use_pydantic = (
            response_format is not None
            and isinstance(response_format, type)
            and isinstance(content_result, response_format)
        )
        if use_pydantic:
            return content_result.model_dump_json()
        return str(content_result) if content_result is not None else ""

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
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                ]
                results.append(self._litellm_chat_completion(content, response_format, **kwargs))
            except Exception as e:
                pruna_logger.error(f"Litellm generation failed: {e}")
                results.append("")
        return results

    def generate_with_image_lists(
        self,
        image_lists: List[List[Image.Image]],
        prompts: List[str],
        response_format: Optional[Union[Type[BaseModel], Literal["integer"], Literal["yes_no"], Literal["json"]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate one response per (``image_list``, ``prompt``) pair.

        Each ``image_list`` contains one or more PIL images (e.g. source and edited for
        VIEScore ``tie``). Message content is built as text first, then each image as
        ``image_url``, matching common OpenAI-style multi-image chat layouts.

        Parameters
        ----------
        image_lists : list[list[PIL.Image.Image]]
            One list of images per prompt (same length as ``prompts``).
        prompts : list[str]
            User text for each row.
        response_format : optional
            Same as :meth:`generate`.
        **kwargs : Any
            Forwarded to litellm ``completion``.

        Returns
        -------
        list[str]
            One string (or JSON string for pydantic) per row.
        """
        if len(image_lists) != len(prompts):
            raise ValueError("image_lists and prompts must have the same length.")
        results: List[str] = []
        for imgs, prompt in zip(image_lists, prompts):
            try:
                content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
                for im in imgs:
                    content.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(im)}})
                results.append(self._litellm_chat_completion(content, response_format, **kwargs))
            except Exception as e:
                pruna_logger.error(f"Litellm multi-image generation failed: {e}")
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
        from pruna.evaluation.metrics.vlm_utils import get_answer_from_response

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
                # Match token if it starts with the yes/no word and the remainder is non-alphabetic
                # (e.g. "yes." or "no," match, but "yesterday" or "not" do not).
                def _word_matches(token_str: str, word: str) -> bool:
                    return token_str.startswith(word) and (
                        len(token_str) == len(word) or not token_str[len(word)].isalpha()
                    )

                yes_words = ("yes", " yes")
                no_words = ("no", " no")
                p_yes = 0.0
                p_no = 0.0
                for tok in logprobs.content or []:
                    top = getattr(tok, "top_logprobs", None) or []
                    for t in top:
                        token_str = (getattr(t, "token", "") or "").lower()
                        lp = float(getattr(t, "logprob", -1e9) or -1e9)
                        prob = math.exp(lp)
                        if any(_word_matches(token_str, w) for w in yes_words):
                            p_yes += prob
                        elif any(_word_matches(token_str, w) for w in no_words):
                            p_no += prob
                    break  # Only process the first output token's top_logprobs
                eps = 1e-12
                denom = p_yes + p_no
                if denom > eps:
                    ans = expected.strip().lower()
                    if ans == "yes":
                        return float(min(1.0, p_yes / denom))
                    if ans == "no":
                        return float(min(1.0, p_no / denom))
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
        Whether to use outlines for constrained decoding when the caller passes a string
        ``response_format``. Usually set from ``structured_output`` via :func:`get_vlm`.
    model_load_kwargs : dict, optional
        Kwargs passed to from_pretrained (e.g. dtype, attn_implementation).
    **kwargs : Any
        Additional arguments passed to model.generate.

    Notes
    -----
    Prefer :func:`get_vlm` from metrics so ``structured_output`` and ``vlm_kwargs`` match
    registry metrics. User manual: :doc:`Evaluate a model </docs_pruna/user_manual/evaluate>`
    (Vision-language judge metrics).

    Examples
    --------
    Local judge only (same ``local`` pattern as :func:`get_vlm`; prefer :func:`get_vlm` from
    metrics so ``structured_output`` and ``vlm_kwargs`` stay aligned):

    .. code-block:: python

        import torch

        from pruna.evaluation.metrics.vlm_base import TransformersVLM

        local = TransformersVLM(
            model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            device="cpu",
            model_load_kwargs={"torch_dtype": torch.float32},
        )
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
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.extra_kwargs = kwargs
        self._model = None
        self._processor = None
        self._yes_no_prefix_ids: Optional[tuple[list[int], list[int]]] = None
        self._outlines_wrapped_model: Any = None

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

    def _get_outlines_wrapped_model(self) -> Any:
        """Lazily wrap HF model + processor for Outlines 1.x steerable generation."""
        if self._outlines_wrapped_model is None:
            from outlines.models.transformers import from_transformers

            assert self._processor is not None, "_processor must be loaded before wrapping with outlines"
            self._outlines_wrapped_model = from_transformers(self._model, self._processor)
        return self._outlines_wrapped_model

    def _pil_for_outlines(self, image: Image.Image) -> Any:
        """Wrap a PIL image for ``outlines.inputs.Image`` (requires a concrete ``format``)."""
        from outlines.inputs import Image as OutlinesImage

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        pil = Image.open(buf)
        return OutlinesImage(pil)

    def _chat_user_with_images(self, images: List[Image.Image], prompt: str) -> Any:
        """Build an ``outlines.inputs.Chat`` with one or more images then text (HF multimodal dicts)."""
        from outlines.inputs import Chat

        parts: list[dict[str, Any]] = []
        for im in images:
            parts.append({"type": "image", "image": self._pil_for_outlines(im)})
        parts.append({"type": "text", "text": prompt})
        return Chat([{"role": "user", "content": parts}])

    def _outlines_output_term(self, response_format: Any) -> Any:
        """
        Map metric ``response_format`` to an Outlines output type, or None for unconstrained decode.

        Returns
        -------
        Any
            A term accepted by :class:`outlines.generator.Generator`, or None.
        """
        from outlines.types import json_schema, regex

        if isinstance(response_format, str):
            if response_format == "integer":
                return regex(r"\d+")
            if response_format == "yes_no":
                return regex(r"(Yes|No)")
            return None
        if isinstance(response_format, type):
            try:
                if issubclass(response_format, BaseModel):
                    return json_schema(response_format)
            except TypeError:
                return None
        return None

    def _generate_steered(self, chats: List[Any], output_term: Any, max_new_tokens: int) -> List[str]:
        """Run Outlines :class:`~outlines.generator.Generator` on prepared chat inputs."""
        from outlines import Generator

        om = self._get_outlines_wrapped_model()
        results: List[str] = []
        with torch.compiler.set_stance("force_eager"):
            gen = Generator(om, output_type=output_term)
            for chat in chats:
                try:
                    out = gen(chat, max_new_tokens=max_new_tokens)
                    results.append(out if isinstance(out, str) else str(out))
                except Exception as e:
                    pruna_logger.warning(f"Outlines generation failed: {e}, using empty string")
                    results.append("")
        return results

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
            When ``use_outlines`` is True: string ``integer`` / ``yes_no``, or a Pydantic model
            class for JSON-schema constrained decoding; otherwise unconstrained ``model.generate``.
        **kwargs : Any
            Additional arguments passed to model generate.

        Returns
        -------
        List[str]
            Generated responses.
        """
        self._load_model()
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        term = self._outlines_output_term(response_format) if self.use_outlines else None
        if term is not None:
            chats = [self._chat_user_with_images([image], prompt) for image, prompt in zip(images, prompts)]
            return self._generate_steered(chats, term, max_new_tokens)
        return self._generate_standard(images, prompts, max_new_tokens)

    def generate_with_image_lists(
        self,
        image_lists: List[List[Image.Image]],
        prompts: List[str],
        response_format: Optional[Union[Type[BaseModel], Literal["integer"], Literal["yes_no"], Literal["json"]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate with multiple PIL images per prompt (e.g. VIEScore source + edited).

        Uses the chat template path with several ``image`` parts then text. When
        ``use_outlines`` is True and ``response_format`` maps to an Outlines output type
        (string ``integer`` / ``yes_no`` or a Pydantic model class), uses the same
        Outlines 1.x steerable path as :meth:`generate` via ``outlines.inputs.Chat``.
        Otherwise uses unconstrained ``model.generate``.

        Parameters
        ----------
        image_lists : list[list[PIL.Image.Image]]
            One list of images per prompt.
        prompts : list[str]
            Prompts aligned with ``image_lists``.
        response_format : optional
            Same conventions as :meth:`generate` for structured decoding when outlines is enabled.
        **kwargs : Any
            Passed through (e.g. ``max_new_tokens``).

        Returns
        -------
        list[str]
            Decoded strings per row.
        """
        if len(image_lists) != len(prompts):
            raise ValueError("image_lists and prompts must have the same length.")
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        self._load_model()
        term = self._outlines_output_term(response_format) if self.use_outlines else None
        if term is not None:
            chats = [self._chat_user_with_images(imgs, prompt) for imgs, prompt in zip(image_lists, prompts)]
            return self._generate_steered(chats, term, max_new_tokens)
        results: List[str] = []
        with torch.inference_mode():
            for imgs, prompt in zip(image_lists, prompts):
                inputs = self._prepare_inputs_multi(imgs, prompt)
                input_len = inputs["input_ids"].shape[1]
                output = self._model.generate(**inputs, max_new_tokens=max_new_tokens, **self.extra_kwargs)
                response = self._decode_output(output[0][input_len:])
                results.append(response)
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

    def _prepare_inputs_multi(self, images: List[Image.Image], prompt: str) -> dict:
        """Chat-template inputs with multiple images then text (VIEScore ``tie``-style)."""
        parts: list[dict[str, Any]] = []
        for im in images:
            parts.append({"type": "image", "image": im})
        parts.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": parts}]
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
                input_len = inputs["input_ids"].shape[1]
                output = self._model.generate(**inputs, max_new_tokens=max_new_tokens, **self.extra_kwargs)
                # Decode only the newly generated tokens to avoid re-including the prompt text.
                response = self._decode_output(output[0][input_len:])
                results.append(response)
        return results

    def _get_tokenizer(self) -> Any:
        """Return the HF tokenizer used for yes/no prefix ids and decoding."""
        self._load_model()
        proc = self._processor
        tok = getattr(proc, "tokenizer", None) or getattr(proc, "text_tokenizer", None)
        if tok is None:
            raise ValueError(
                "Transformers VLM probability scoring requires a tokenizer on the processor; "
                "pass use_probability=False for binary scoring."
            )
        return tok

    def _score_yes_no_probability(self, image: Image.Image, question: str, answer: str) -> float:
        """Soft VQAScore-style score from next-token softmax over yes/no prefix token ids."""
        from pruna.evaluation.metrics.vlm_utils import yes_no_first_token_id_groups

        self._load_model()
        prompt = f"{question} Please answer yes or no."
        inputs = self._prepare_inputs(image, prompt)
        if self._yes_no_prefix_ids is None:
            self._yes_no_prefix_ids = yes_no_first_token_id_groups(self._get_tokenizer())
        yes_ids, no_ids = self._yes_no_prefix_ids
        if not yes_ids or not no_ids:
            pruna_logger.warning("Empty yes/no prefix token ids; install a tokenizer with standard Yes/No encodings.")
            return 0.0
        with torch.inference_mode():
            out = self._model(**inputs)
            if not hasattr(out, "logits") or out.logits is None:
                raise RuntimeError("Model forward did not return logits; cannot compute P(Yes).")
            logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        device = probs.device
        p_yes = probs[torch.tensor(yes_ids, device=device, dtype=torch.long)].sum()
        p_no = probs[torch.tensor(no_ids, device=device, dtype=torch.long)].sum()
        denom = p_yes + p_no
        ans = answer.strip().lower()
        eps = 1e-12
        if float(denom.item()) < eps:
            if ans == "yes":
                return float(p_yes.clamp(0.0, 1.0).item())
            if ans == "no":
                return float(p_no.clamp(0.0, 1.0).item())
            return 0.0
        if ans == "yes":
            return float((p_yes / (denom + eps)).item())
        if ans == "no":
            return float((p_no / (denom + eps)).item())
        return float((p_yes / (denom + eps)).item())

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

        When ``use_probability`` is True, computes a VQAScore-style score from the next-token
        distribution at the last context position (softmax mass on yes/no prefix token ids,
        normalized over their union). Otherwise uses generation and binary substring matching.

        Parameters
        ----------
        images : List[Image.Image]
            List of PIL Images.
        questions : List[str]
            List of questions.
        answers : List[str]
            List of expected answers.
        use_probability : bool, optional
            If True, return a soft score from logits (no ``generate`` call). If False, binary.
        response_format : Type[BaseModel] | str | None, optional
            Structured output format for answer extraction (only when ``use_probability`` is False).
        **kwargs : Any
            Additional arguments passed to ``generate`` when ``use_probability`` is False.

        Returns
        -------
        List[float]
            Scores for each image-question pair in ``[0, 1]``.
        """
        from pruna.evaluation.metrics.vlm_utils import get_answer_from_response

        scores = []
        for image, question, answer in zip(images, questions, answers):
            if use_probability:
                scores.append(self._score_yes_no_probability(image, question, answer))
                continue
            prompt = f"{question} Please answer yes or no."
            responses = self.generate([image], [prompt], response_format=response_format, **kwargs)
            raw = responses[0] if responses else ""
            response_answer = get_answer_from_response(raw) if response_format is not None else raw.lower()
            score = 1.0 if answer.lower() in response_answer.lower() else 0.0
            scores.append(score)
        return scores
