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
import re
from typing import Any, List

import torch
from PIL import Image
from pydantic import BaseModel, Field


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.max() > 1:
        tensor = tensor / 255.0
    np_img = (tensor.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(np_img.transpose(1, 2, 0))


def _process_images(images: torch.Tensor) -> List[Any]:
    return [_tensor_to_pil(img) if isinstance(img, torch.Tensor) else img for img in images]


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
    Structured output for numeric scoring (img_edit_score, viescore).

    Parameters
    ----------
    score : float
        Score from 0 to 10.
    """

    score: float = Field(ge=0, le=10, description="Score from 0 to 10")


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
    if raw.startswith("{"):
        try:
            return json.loads(raw).get("answer", raw)
        except (json.JSONDecodeError, TypeError):
            pass
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
    text = _extract_text_payload(response)
    return _strip_no_text_markers(text)


def _extract_text_payload(response: str | BaseModel | dict) -> str:
    if response is None:
        return ""
    if isinstance(response, TextOutput):
        return response.text
    if isinstance(response, dict):
        return str(response.get("text", "") or "")
    return _parse_json_text(str(response or "").strip())


def _parse_json_text(text: str) -> str:
    if not text.startswith("{"):
        return text
    try:
        data = json.loads(text)
        return str(data.get("text", text))
    except (json.JSONDecodeError, TypeError):
        return text


def _strip_no_text_markers(text: str) -> str:
    cleaned = text or ""
    for phrase in ("No text recognized", "no text recognized", "No text"):
        cleaned = cleaned.replace(phrase, "").strip()
    return cleaned.strip()


def get_score_from_response(response: str | BaseModel | dict) -> float:
    """
    Extract numeric score (0-10) from a VLM generate() response.

    Parameters
    ----------
    response : str | BaseModel | dict
        Raw response from vlm.generate().

    Returns
    -------
    float
        Score in [0, 1] (normalized from 0-10).
    """
    if response is None:
        return 0.0
    if isinstance(response, FloatOutput):
        return min(response.score, 10.0) / 10.0
    if isinstance(response, dict):
        return min(float(response.get("score", 0)), 10.0) / 10.0
    numbers = re.findall(r"\d+", str(response or ""))
    return min(float(numbers[0]), 10.0) / 10.0 if numbers else 0.0
