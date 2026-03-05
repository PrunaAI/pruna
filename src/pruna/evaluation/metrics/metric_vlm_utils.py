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

from typing import Any, List, Literal

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
    Structured output for VQA (answer with optional confidence).

    Parameters
    ----------
    answer : str
        The VQA answer text.
    confidence : float, optional
        Confidence score. Default is 1.0.
    """

    answer: str
    confidence: float = 1.0


class YesNoAnswer(BaseModel):
    """
    Structured output for Yes/No questions (alignment, VQA, QA accuracy).

    Parameters
    ----------
    answer : Literal["Yes", "No"]
        Answer must be exactly Yes or No.
    """

    answer: Literal["Yes", "No"] = Field(description="Answer must be exactly Yes or No")


class ScoreOutput(BaseModel):
    """
    Structured output for numeric scoring (img_edit_score, viescore).

    Parameters
    ----------
    score : float
        Score from 0 to 10.
    reasoning : str | None, optional
        Optional reasoning for the score.
    """

    score: float = Field(ge=0, le=10, description="Score from 0 to 10")
    reasoning: str | None = None


class OCRText(BaseModel):
    """
    Structured output for OCR text extraction (text_score).

    Parameters
    ----------
    text : str
        Extracted text from the image, or 'No text recognized' if empty.
    """

    text: str = Field(description="Extracted text from the image, or 'No text recognized' if empty")
