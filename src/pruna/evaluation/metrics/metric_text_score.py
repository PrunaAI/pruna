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

"""Text Score metric for evaluating text rendering in images using VLM OCR."""

from __future__ import annotations

import re
from typing import Any, List, Literal, Optional

import numpy as np
import torch

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.metric_vlm_utils import TextOutput, _process_images, get_text_from_response
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import (
    SINGLE,
    get_call_type_for_single_metric,
    metric_data_processor,
)
from pruna.evaluation.metrics.vlm_base import BaseVLM, get_vlm

OCR_PROMPT = (
    "Extract all text visible in this image. Include logos, stylized fonts, handwritten text, "
    "and non-standard typography. Return only the extracted text, exactly as it appears—no preamble, "
    "explanation, or markdown. Preserve words, numbers, punctuation, and spacing. "
    "If no text is recognized, reply with exactly: No text recognized"
)


@MetricRegistry.register("text_score")
class TextScoreMetric(StatefulMetric):
    """
    Text Score metric for evaluating text rendering in images.

    Uses VLM for OCR to extract text and compare with ground truth.
    Lower scores (edit distance) are better.

    Parameters
    ----------
    *args : Any
        Additional positional arguments.
    vlm : BaseVLM | None, optional
        Custom VLM instance. If provided, vlm_type and model_name are ignored.
    vlm_type : {"litellm", "transformers"}, optional
        VLM backend. Default is "litellm".
    model_name : str, optional
        Model name. Default is "gpt-4o".
    vlm_kwargs : dict, optional
        Extra kwargs for VLM init (e.g. model_load_kwargs for transformers).
    structured_output : bool, optional
        Use structured generation. Default is True.
    use_outlines : bool, optional
        Use outlines for transformers. Default is False.
    device : str | torch.device | None, optional
        Device for transformers VLM.
    api_key : str | None, optional
        API key for litellm.
    call_type : str, optional
        Call type for the metric.
    **kwargs : Any
        Additional arguments.
    """

    scores: List[float]
    default_call_type: str = "y_gt"
    higher_is_better: bool = False
    metric_name: str = "text_score"
    runs_on: List[str] = ["cuda", "cpu"]

    def __init__(
        self,
        vlm: Optional[BaseVLM] = None,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str = "gpt-4o",
        vlm_kwargs: Optional[dict] = None,
        structured_output: bool = True,
        use_outlines: bool = True,
        device: str | torch.device = "cpu",
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        **kwargs,
    ):
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)

        self.vlm = get_vlm(
            vlm=vlm,
            vlm_type=vlm_type,
            model_name=model_name,
            device=device,
            api_key=api_key,
            use_outlines=use_outlines,
            **(vlm_kwargs or {}),
        )
        self.response_format = TextOutput if structured_output else None

        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("scores", [])

    @staticmethod
    def _normalize_text(s: str) -> str:
        cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", "", s or "")
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> float:
        if len(s1) < len(s2):
            return TextScoreMetric._levenshtein(s2, s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j] + (c1 != c2), prev[j + 1] + 1, curr[-1] + 1))
            prev = curr
        return float(prev[-1])

    def update(self, x: List[Any] | torch.Tensor, gt: List[str], outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (prompts).
        gt : List[dict] | List[str]
            Ground truth auxiliaries. Each item must have 'text_content' key (e.g. from
            LongTextBench, OneIG). Or a list of strings for backward compatibility.
        outputs : torch.Tensor
            The output images.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        auxiliaries = inputs[1] if len(inputs) > 1 and isinstance(inputs[1], (list, tuple)) else [{}] * len(images)
        for i, image in enumerate(images):
            responses = self.vlm.generate([image], [OCR_PROMPT], response_format=self.response_format)
            raw = responses[0] if responses else ""
            ocr_text = get_text_from_response(raw)
            aux = auxiliaries[i] if i < len(auxiliaries) else {}
            text_gt = aux.get("text_content") if isinstance(aux, dict) else (aux if isinstance(aux, str) else None)
            if text_gt is None:
                raise ValueError(
                    "text_score requires 'text_content' in auxiliaries. "
                    "Use a benchmark that provides it (e.g. LongTextBench, OneIG)."
                )
            norm_gt = self._normalize_text(text_gt)
            norm_ocr = self._normalize_text(ocr_text)
            score = self._levenshtein(norm_ocr, norm_gt)
            self.scores.append(score)

    def compute(self) -> MetricResult:
        """
        Compute the text score.

        Returns
        -------
        MetricResult
            The mean text score (edit distance) across all updates.
        """
        if not self.scores:
            return MetricResult(self.metric_name, self.__dict__, 0.0)
        return MetricResult(self.metric_name, self.__dict__, float(np.mean(self.scores)))
