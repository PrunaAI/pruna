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

"""Helpers for text rendering metrics (simple Levenshtein vs OneIG-style composite).

OneIG-style preprocessing and aggregation follow
`OneIG-Benchmark/scripts/text/text_utils.py` and `text_score.py` (Apache-2.0).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

_OCR_HALLUCINATION_KEYWORDS = ("addCriterion", "No text recognized.", "No text recognized")


def normalize_text_simple(s: str) -> str:
    """
    Normalize text for the legacy ``text_score`` metric (light cleanup + spacing).

    Parameters
    ----------
    s : str
        Raw string.

    Returns
    -------
    str
        Normalized string.
    """
    cleaned = re.sub(
        r"[^\u4e00-\u9fa5a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]",
        "",
        s or "",
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def levenshtein(s1: str, s2: str) -> float:
    """
    Symmetric Levenshtein edit distance.

    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.

    Returns
    -------
    float
        Edit distance.
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j] + (c1 != c2), prev[j + 1] + 1, curr[-1] + 1))
        prev = curr
    return float(prev[-1])


def contains_chinese(text: str) -> bool:
    """
    Return True if ``text`` contains CJK unified ideographs.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    bool
        Whether Chinese characters are present.
    """
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def preprocess_string_oneig(s: str) -> str:
    """
    OneIG ``preprocess_string``: charset filter, Chinese vs whitespace normalization.

    Parameters
    ----------
    s : str
        Raw string.

    Returns
    -------
    str
        Preprocessed string (ground truth or OCR).
    """
    raw = s or ""
    cleaned = re.sub(
        r"[^\u4e00-\u9fa5a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]",
        "",
        raw,
    )
    if contains_chinese(cleaned):
        pattern = re.compile(
            r"[\u4e00-\u9fa5a-zA-Z0-9àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]",
        )
        return "".join(pattern.findall(raw)).strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_oneig_ocr_hallucinations(text: str) -> str:
    """
    Remove known OCR boilerplate substrings (OneIG ``clean_and_remove_hallucinations``).

    Parameters
    ----------
    text : str
        Raw OCR output.

    Returns
    -------
    str
        Cleaned OCR text.
    """
    out = text or ""
    for keyword in _OCR_HALLUCINATION_KEYWORDS:
        out = (
            out.replace(keyword, "")
            .replace(f"\n{keyword}", "")
            .replace(f"{keyword}\n", "")
        )
    return out


def calculate_char_match_ratio(
    text_gt: str,
    ocr_str: str,
) -> tuple[int, float, int]:
    """
    OneIG overlap stats: character multiset for ZH, word multiset for EN.

    Parameters
    ----------
    text_gt : str
        Preprocessed ground truth.
    ocr_str : str
        Preprocessed OCR.

    Returns
    -------
    total_match_count : int
        Overlap count used in WAC numerator aggregation.
    ratio : float
        Per-sample ratio (mean of ratios is not used in the official aggregate).
    gt_total : int
        Denominator term: ``sum(gt_counter.values())`` for WAC aggregation.
    """
    if contains_chinese(text_gt):
        gt_counter: Counter[str] = Counter(text_gt)
        ocr_counter: Counter[str] = Counter(ocr_str)
        total_match_count = int(sum((gt_counter & ocr_counter).values()))
        ratio = total_match_count / len(text_gt) if len(text_gt) > 0 else 0.0
        return total_match_count, ratio, int(sum(gt_counter.values()))

    words_gt = text_gt.split()
    words_ocr = ocr_str.split()
    gt_counter = Counter(words_gt)
    ocr_counter = Counter(words_ocr)
    total_match_count = int(sum((gt_counter & ocr_counter).values()))
    total_gt_count = len(words_gt)
    ratio = total_match_count / total_gt_count if total_gt_count > 0 else 0.0
    return total_match_count, ratio, int(sum(gt_counter.values()))


def max_edit_distance_for_language(language_mode: Literal["EN", "ZH"]) -> int:
    """
    OneIG ``MAX_EDIT_DISTANCE`` (100 for English, 50 for Chinese benchmark split).

    Parameters
    ----------
    language_mode : {'EN', 'ZH'}
        Benchmark language mode.

    Returns
    -------
    int
        Cap used in the composite text score.
    """
    return 50 if language_mode == "ZH" else 100


def oneig_per_sample_contributions(text_gt: str, ocr_raw: str) -> tuple[float, float, int, int]:
    """
    Per-sample terms for OneIG aggregation (ED, CR, WAC numerator/denominator parts).

    Parameters
    ----------
    text_gt : str
        Ground-truth text (dataset field).
    ocr_raw : str
        Raw OCR string from the VLM.

    Returns
    -------
    edit_distance : float
        Levenshtein distance after OneIG preprocess.
    completion_ratio : float
        1.0 if distance is zero, else 0.0.
    match_count : int
        Overlap count for WAC.
    gt_total : int
        Ground-truth token count term for WAC denominator.
    """
    ocr_clean = clean_oneig_ocr_hallucinations(ocr_raw)
    gt_pre = preprocess_string_oneig(text_gt)
    ocr_pre = preprocess_string_oneig(ocr_clean)
    ed = levenshtein(ocr_pre, gt_pre)
    cr = 1.0 if ed == 0.0 else 0.0
    match_count, _, gt_total = calculate_char_match_ratio(gt_pre, ocr_pre)
    return ed, cr, match_count, gt_total


def oneig_mean_text_score(
    edit_distances: list[float],
    completion_ratios: list[float],
    match_counts: list[int],
    gt_totals: list[int],
    language_mode: Literal["EN", "ZH"],
) -> tuple[float, float, float, float]:
    """
    Aggregate OneIG ED, CR, WAC and composite text score (higher is better).

    Parameters
    ----------
    edit_distances : list of float
        Per-sample edit distances.
    completion_ratios : list of float
        Per-sample completion indicators.
    match_counts : list of int
        Per-sample WAC numerators.
    gt_totals : list of int
        Per-sample WAC denominator terms.
    language_mode : {'EN', 'ZH'}
        Selects ``MAX_EDIT_DISTANCE``.

    Returns
    -------
    ed_mean : float
        Mean edit distance.
    cr_mean : float
        Mean completion ratio.
    wac : float
        Micro-averaged WAC: ``sum(match_counts) / sum(gt_totals)``.
    text_score : float
        Composite: ``1 - min(MAX_ED, ED) * (1 - CR) * (1 - WAC) / MAX_ED``.
    """
    cap = float(max_edit_distance_for_language(language_mode))
    if not edit_distances:
        return 0.0, 0.0, 0.0, 0.0
    ed_mean = float(sum(edit_distances) / len(edit_distances))
    cr_mean = float(sum(completion_ratios) / len(completion_ratios))
    denom = float(sum(gt_totals))
    wac = float(sum(match_counts) / denom) if denom > 0.0 else 0.0
    text_score = 1.0 - min(cap, ed_mean) * (1.0 - cr_mean) * (1.0 - wac) / cap
    return ed_mean, cr_mean, wac, text_score
