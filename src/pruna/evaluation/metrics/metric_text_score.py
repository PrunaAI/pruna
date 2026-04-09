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

"""Text rendering via OCR: mean Levenshtein (``text_score`` / ``ocr_levenshtein``).

OneIG composite: ``oneig_text_score`` / ``ocr_text_score``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Literal, Optional

import numpy as np
import torch

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.metric_text_score_utils import (
    levenshtein,
    normalize_text_simple,
    oneig_mean_text_score,
    oneig_per_sample_contributions,
)
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


class _BaseVLMOCRTextMetric(StatefulMetric):
    """
    Shared VLM OCR over rendered images with ground truth in ``text_content``.

    Subclasses implement how OCR and GT strings are scored and aggregated.

    Parameters
    ----------
    *args : Any
        Additional positional arguments (unused; registry compatibility).
    vlm : BaseVLM | None, optional
        Custom VLM instance. If provided, ``vlm_type`` and ``model_name`` are ignored.
    vlm_type : {'litellm', 'transformers'}, optional
        VLM backend. Default is ``'litellm'``.
    model_name : str, optional
        Model name. Default is ``'gpt-4o'``.
    vlm_kwargs : dict, optional
        Extra kwargs for VLM init.
    structured_output : bool, optional
        Use structured generation (litellm pydantic; transformers outlines when applicable).
        Default is True.
    device : str | torch.device | None, optional
        Device for transformers VLM.
    api_key : str | None, optional
        API key for litellm.
    call_type : str, optional
        Call type for the metric.
    **kwargs : Any
        Additional arguments.
    """

    default_call_type: str = "y_gt"
    runs_on: List[str] = ["cuda", "cpu", "mps"]

    def __init__(
        self,
        *args: Any,
        vlm: Optional[BaseVLM] = None,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str = "gpt-4o",
        vlm_kwargs: Optional[dict] = None,
        structured_output: bool = True,
        device: str | torch.device | None = None,
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device)
        self.device = set_to_best_available_device(device)

        self.vlm = get_vlm(
            vlm=vlm,
            vlm_type=vlm_type,
            model_name=model_name,
            device=device,
            api_key=api_key,
            structured_output=structured_output,
            **(vlm_kwargs or {}),
        )
        self.response_format = TextOutput if structured_output else None

        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)

    @abstractmethod
    def _accumulate_sample(self, text_gt: str, ocr_text: str) -> None:
        """Update metric state from one ground-truth / OCR pair."""

    @abstractmethod
    def _compute_result_value(self) -> float:
        """Return the scalar reported as ``MetricResult.result``."""

    def update(self, x: List[Any] | torch.Tensor, gt: List[str], outputs: torch.Tensor) -> None:
        """
        Run OCR on outputs and score against ``text_content`` (or string list) auxiliaries.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            Batch prompts or metadata.
        gt : list of dict or list of str
            Auxiliaries with ``'text_content'`` or plain strings.
        outputs : torch.Tensor
            Rendered images.
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
                    f"{self.metric_name} requires 'text_content' in auxiliaries. "
                    "Use a benchmark that provides it (e.g. LongTextBench, OneIG)."
                )
            self._accumulate_sample(text_gt, ocr_text)

    def compute(self) -> MetricResult:
        """
        Aggregate batched contributions into a single metric value.

        Returns
        -------
        MetricResult
            Named result with ``higher_is_better`` taken from the class.
        """
        value = self._compute_result_value()
        return MetricResult(self.metric_name, self.__dict__, float(value))


@MetricRegistry.register("ocr_levenshtein")
@MetricRegistry.register("text_score")
class TextScoreMetric(_BaseVLMOCRTextMetric):
    """
    OCR then mean Levenshtein distance to ground truth (lower is better).

    Registry: ``ocr_levenshtein`` (descriptive) and ``text_score`` (legacy).

    Uses light normalization only (not the full OneIG preprocess). See
    :class:`OneIGTextScoreMetric` for the OneIG composite ``ocr_text_score``.

    Parameters
    ----------
    *args : Any
        Additional positional arguments (unused; registry compatibility).
    vlm : BaseVLM | None, optional
        Custom VLM instance. If provided, ``vlm_type`` and ``model_name`` are ignored.
    vlm_type : {'litellm', 'transformers'}, optional
        VLM backend. Default is ``'litellm'``.
    model_name : str, optional
        Model name. Default is ``'gpt-4o'``.
    vlm_kwargs : dict, optional
        Extra kwargs for VLM init.
    structured_output : bool, optional
        Use structured generation (litellm pydantic; transformers outlines when applicable).
        Default is True.
    device : str | torch.device | None, optional
        Device for transformers VLM.
    api_key : str | None, optional
        API key for litellm.
    call_type : str, optional
        Call type for the metric.
    **kwargs : Any
        Additional keyword arguments forwarded to :class:`_BaseVLMOCRTextMetric`.
    """

    scores: List[float]
    higher_is_better: bool = False
    metric_name: str = "text_score"

    def __init__(
        self,
        *args: Any,
        vlm: Optional[BaseVLM] = None,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str = "gpt-4o",
        vlm_kwargs: Optional[dict[str, Any]] = None,
        structured_output: bool = True,
        device: str | torch.device | None = None,
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            vlm=vlm,
            vlm_type=vlm_type,
            model_name=model_name,
            vlm_kwargs=vlm_kwargs,
            structured_output=structured_output,
            device=device,
            api_key=api_key,
            call_type=call_type,
            **kwargs,
        )
        self.add_state("scores", [])

    def _accumulate_sample(self, text_gt: str, ocr_text: str) -> None:
        norm_gt = normalize_text_simple(text_gt)
        norm_ocr = normalize_text_simple(ocr_text)
        self.scores.append(levenshtein(norm_ocr, norm_gt))

    def _compute_result_value(self) -> float:
        if not self.scores:
            return 0.0
        return float(np.mean(self.scores))


@MetricRegistry.register("ocr_text_score")
@MetricRegistry.register("oneig_text_score")
class OneIGTextScoreMetric(_BaseVLMOCRTextMetric):
    """
    OCR then OneIG-style composite text score (higher is better).

    Registry: ``ocr_text_score`` (descriptive) and ``oneig_text_score`` (protocol).

    Aggregates edit distance, completion rate, and word/char accuracy like
    ``OneIG-Benchmark/scripts/text/text_score.py``.

    Parameters
    ----------
    *args : Any
        Additional positional arguments (forwarded to :class:`_BaseVLMOCRTextMetric`).
    language_mode : {'EN', 'ZH'}, optional
        Selects ``MAX_EDIT_DISTANCE`` (100 vs 50) for the composite.
    vlm : BaseVLM | None, optional
        Custom VLM instance. If provided, ``vlm_type`` and ``model_name`` are ignored.
    vlm_type : {'litellm', 'transformers'}, optional
        VLM backend. Default is ``'litellm'``.
    model_name : str, optional
        Model name. Default is ``'gpt-4o'``.
    vlm_kwargs : dict, optional
        Extra kwargs for VLM init (e.g. ``model_load_kwargs`` for transformers).
    structured_output : bool, optional
        Use structured generation (litellm pydantic; transformers outlines when applicable).
        Default is True.
    device : str | torch.device | None, optional
        Device for transformers VLM.
    api_key : str | None, optional
        API key for litellm.
    call_type : str, optional
        Call type for the metric.
    **kwargs : Any
        Additional keyword arguments forwarded to :class:`_BaseVLMOCRTextMetric`.
    """

    edit_distances: List[float]
    completion_ratios: List[float]
    match_counts: List[int]
    gt_totals: List[int]

    higher_is_better: bool = True
    metric_name: str = "oneig_text_score"

    def __init__(
        self,
        *args: Any,
        language_mode: Literal["EN", "ZH"] = "EN",
        vlm: Optional[BaseVLM] = None,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str = "gpt-4o",
        vlm_kwargs: Optional[dict[str, Any]] = None,
        structured_output: bool = True,
        device: str | torch.device | None = None,
        api_key: Optional[str] = None,
        call_type: str = SINGLE,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            vlm=vlm,
            vlm_type=vlm_type,
            model_name=model_name,
            vlm_kwargs=vlm_kwargs,
            structured_output=structured_output,
            device=device,
            api_key=api_key,
            call_type=call_type,
            **kwargs,
        )
        self.language_mode = language_mode
        self.add_state("edit_distances", [])
        self.add_state("completion_ratios", [])
        self.add_state("match_counts", [])
        self.add_state("gt_totals", [])

    def _accumulate_sample(self, text_gt: str, ocr_text: str) -> None:
        ed, cr, mcount, gtot = oneig_per_sample_contributions(text_gt, ocr_text)
        self.edit_distances.append(ed)
        self.completion_ratios.append(cr)
        self.match_counts.append(mcount)
        self.gt_totals.append(gtot)

    def _compute_result_value(self) -> float:
        *_, text_score = oneig_mean_text_score(
            self.edit_distances,
            self.completion_ratios,
            self.match_counts,
            self.gt_totals,
            self.language_mode,
        )
        return text_score
