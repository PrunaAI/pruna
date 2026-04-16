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

"""Shared utilities and Pydantic models for VLM metrics."""

from __future__ import annotations

import json
import math
import re
from io import BytesIO
from typing import Any, List, Sequence

import torch
from PIL import Image
from pydantic import BaseModel, Field

VLM_AUX_IMAGE_BYTES_KEY_ORDER: tuple[str, ...] = (
    "source_image_bytes",
    "input_image_bytes",
    "reference_image_bytes",
    "image_bytes",
)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.max() > 1:
        tensor = tensor / 255.0
    np_img = (tensor.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(np_img.transpose(1, 2, 0))


def _process_images(images: torch.Tensor) -> List[Any]:
    return [_tensor_to_pil(img) if isinstance(img, torch.Tensor) else img for img in images]


def pil_rgb_from_aux_image_bytes(
    aux: dict[str, Any],
    *,
    min_bytes_in_value_scan: int = 0,
) -> Image.Image | None:
    """
    Decode a source / reference RGB image from auxiliary dict bytes.

    Tries :data:`VLM_AUX_IMAGE_BYTES_KEY_ORDER` first, then scans ``aux.values()`` for raw
    byte blobs. The value scan skips blobs shorter than ``min_bytes_in_value_scan`` (use
    ``100`` to match editing metrics that avoid tiny false positives).

    Parameters
    ----------
    aux : dict[str, Any]
        Per-sample auxiliary dict (e.g. from ``prompt_with_auxiliaries_collate``).
    min_bytes_in_value_scan : int, optional
        Minimum length for blobs discovered only in the generic ``aux.values()`` pass.
        Named keys use any non-empty ``bytes`` / ``bytearray``. Use ``0`` when any
        non-trivial blob in ``aux.values()`` should be tried (e.g. tests building preds from aux).

    Returns
    -------
    PIL.Image.Image | None
        RGB image if decoding succeeds; ``None`` if nothing decodable was found.
    """
    for key in VLM_AUX_IMAGE_BYTES_KEY_ORDER:
        raw = aux.get(key)
        if isinstance(raw, (bytes, bytearray)) and raw:
            try:
                return Image.open(BytesIO(raw)).convert("RGB")
            except Exception:
                continue
    for v in aux.values():
        if isinstance(v, (bytes, bytearray)) and v and len(v) >= min_bytes_in_value_scan:
            try:
                return Image.open(BytesIO(v)).convert("RGB")
            except Exception:
                continue
    return None


def yes_no_first_token_id_groups(tokenizer: Any) -> tuple[list[int], list[int]]:
    """
    Collect first subword token ids that start a yes/no answer for next-token softmax scoring.

    Used by :class:`~pruna.evaluation.metrics.vlm_base.TransformersVLM` for VQAScore-style
    P(Yes): sum softmax mass on these ids, normalized against yes+no for a stable [0, 1] score.

    Parameters
    ----------
    tokenizer : Any
        Hugging Face ``PreTrainedTokenizer`` (or compatible ``encode``).

    Returns
    -------
    list[int]
        Distinct token ids for yes-leaning first tokens (overlap with no-ids removed).
    list[int]
        Distinct token ids for no-leaning first tokens (overlap with yes-ids removed).
    """
    yes_prefixes = (
        "Yes",
        " Yes",
        " yes",
        "yes",
        "\nYes",
        "\n Yes",
        "Yes,",
        " Yes,",
    )
    no_prefixes = (
        "No",
        " No",
        " no",
        "no",
        "\nNo",
        "\n No",
        "No,",
        " No,",
    )
    yes_ids: set[int] = set()
    no_ids: set[int] = set()
    for s in yes_prefixes:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            yes_ids.add(ids[0])
    for s in no_prefixes:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            no_ids.add(ids[0])
    overlap = yes_ids & no_ids
    yes_only = sorted(yes_ids - overlap)
    no_only = sorted(no_ids - overlap)
    return yes_only, no_only


class VQAnswer(BaseModel):
    """
    Structured output for VQA questions (Yes/No or open-ended).

    Parameters
    ----------
    answer : str
        Answer to the question. Typically "Yes" or "No" for alignment metrics,
        but can be any string for open-ended questions.
    """

    answer: str = Field(description="Answer to the question")


class FloatOutput(BaseModel):
    """
    Structured output for numeric scoring (img_edit_score, VieScoreMetric).

    Parameters
    ----------
    score : float
        Score from 0 to 10.
    """

    score: float = Field(ge=0, le=10, description="Score from 0 to 10")


class VIEScoreJsonOutput(BaseModel):
    """
    Structured output matching VIEScore JSON (text-to-image / editing evaluation).

    Parameters
    ----------
    score : list[float]
        One or more sub-scores on a 0--10 scale (e.g. two criteria for editing).
    reasoning : str
        Short evaluator reasoning.
    """

    score: list[float] = Field(description="Sub-scores on 0-10 scale")
    reasoning: str = Field(default="", description="Brief reasoning")


def _json_dict_from_response_fragment(text: str) -> dict | None:
    """Parse a leading JSON object from a string response, or return None."""
    stripped = (text or "").strip()
    if not stripped.startswith("{"):
        return None
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


class TextOutput(BaseModel):
    """
    Structured output for text extraction (text_score).

    Parameters
    ----------
    text : str
        Extracted text from the image, or 'No text recognized' if empty.
    """

    text: str = Field(description="Extracted text from the image, or 'No text recognized' if empty")


def get_answer_from_response(response: str | BaseModel | dict) -> str:
    """
    Extract answer string from a VLM score() response (VQAnswer, dict, or raw string).

    Parameters
    ----------
    response : str | BaseModel | dict
        Raw response from vlm.generate() or vlm.score().

    Returns
    -------
    str
        Extracted answer string, or empty string.
    """
    if response is None:
        return ""
    if isinstance(response, VQAnswer):
        return response.answer
    if isinstance(response, dict):
        return response.get("answer", "")
    raw = str(response).strip()
    parsed = _json_dict_from_response_fragment(raw)
    if parsed is not None:
        return str(parsed.get("answer", raw))
    return raw


def get_text_from_response(response: str | BaseModel | dict) -> str:
    """
    Extract text from a VLM generate() response (str, pydantic, or dict).

    Parameters
    ----------
    response : str | BaseModel | dict
        Raw response from vlm.generate().

    Returns
    -------
    str
        Extracted text, or empty string.
    """
    if response is None:
        return ""
    if isinstance(response, TextOutput):
        text = response.text
    elif isinstance(response, dict):
        text = response.get("text", "")
    else:
        text = (response or "").strip()
        parsed = _json_dict_from_response_fragment(text)
        if parsed is not None:
            text = str(parsed.get("text", text))
        for phrase in ("No text recognized", "no text recognized", "No text"):
            text = text.replace(phrase, "").strip()
    return (text or "").strip()


def get_score_from_response(response: str | BaseModel | dict) -> float:
    """
    Extract numeric score (0-10) from a VLM generate() response.

    Handles:

    * ``FloatOutput`` instances (local / parsed Pydantic).
    * ``dict`` with a ``"score"`` key.
    * JSON **strings** (e.g. LitellmVLM returns ``model_dump_json()`` for structured output).
    * Plain text with a number (first decimal or integer matched).

    Parameters
    ----------
    response : str | BaseModel | dict
        Raw response from vlm.generate().

    Returns
    -------
    float
        Score in [0, 1] (normalized from 0-10). Always non-negative.
    """
    if response is None:
        return 0.0
    if isinstance(response, FloatOutput):
        return max(0.0, min(float(response.score), 10.0)) / 10.0
    if isinstance(response, dict):
        return max(0.0, min(float(response.get("score", 0)), 10.0)) / 10.0
    text = str(response or "").strip()
    parsed = _json_dict_from_response_fragment(text)
    if parsed is not None and "score" in parsed:
        try:
            return max(0.0, min(float(parsed["score"]), 10.0)) / 10.0
        except (TypeError, ValueError):
            pass
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        return min(float(match.group(0)), 10.0) / 10.0
    return 0.0


def viescore_min_scores_0_10(response: str | BaseModel | dict) -> list[float]:
    """
    Parse VIEScore-style JSON with a ``score`` list of values in ``[0, 10]``.

    Parameters
    ----------
    response : str | BaseModel | dict
        Model output (pydantic ``VIEScoreJsonOutput``, dict, or JSON string).

    Returns
    -------
    list[float]
        Sub-scores; empty if parsing fails.
    """
    if response is None:
        return []
    if isinstance(response, VIEScoreJsonOutput):
        return [float(x) for x in response.score]
    if isinstance(response, dict):
        raw = response.get("score", [])
        if isinstance(raw, (list, tuple)):
            return [float(x) for x in raw]
        return []
    text = str(response or "").strip()
    parsed = _json_dict_from_response_fragment(text)
    if parsed is not None and "score" in parsed:
        try:
            raw = parsed["score"]
            if isinstance(raw, (list, tuple)):
                return [float(x) for x in raw]
            return [float(raw)]
        except (TypeError, ValueError):
            return []
    return []


def viescore_tie_overall_unit(sc_scores: Sequence[float], pq_scores: Sequence[float]) -> float:
    """
    Overall VIEScore for text-image editing (``tie`` task): ``sqrt(min(SC)*min(PQ))/10`` in ``[0, 1]``.

    Matches the reference ``math.sqrt(SC_score * PQ_score)`` on a 0--10 scale with
    ``SC_score = min(...)``, ``PQ_score = min(...)`` (`VIEScore`_).

    .. _VIEScore: https://github.com/TIGER-AI-Lab/VIEScore

    Parameters
    ----------
    sc_scores : Sequence[float]
        Semantic / instruction sub-scores on 0--10 (e.g. editing success and over-editing).
    pq_scores : Sequence[float]
        Perceptual sub-scores on 0--10 (e.g. naturalness and artifacts).

    Returns
    -------
    float
        Overall score in ``[0, 1]`` (higher is better).
    """
    if not sc_scores or not pq_scores:
        return 0.0
    sc = min(float(x) for x in sc_scores)
    pq = min(float(x) for x in pq_scores)
    return math.sqrt(sc * pq) / 10.0
