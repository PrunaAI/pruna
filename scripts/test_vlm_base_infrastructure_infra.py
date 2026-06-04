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

"""Tests for VLM base classes and vlm_utils (infrastructure PR only)."""

from unittest.mock import MagicMock, patch

import pytest

from pruna.evaluation.metrics.vlm_base import BaseVLM, LitellmVLM, get_vlm
from pruna.evaluation.metrics.vlm_utils import FloatOutput, get_score_from_response, yes_no_first_token_id_groups


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (FloatOutput(score=8.0), 0.8),
        ({"score": 5.0}, 0.5),
        ('{"score": 7.5}', 0.75),
        ('{"score": 10}', 1.0),
        ("8", 0.8),
        ("Score: 7.5 out of 10", 0.75),
        ("", 0.0),
    ],
)
def test_get_score_from_response(raw: object, expected: float) -> None:
    """``get_score_from_response`` maps pydantic, dict, JSON, and text to ``[0, 1]``."""
    assert get_score_from_response(raw) == pytest.approx(expected)


@pytest.mark.cpu
def test_get_vlm_returns_custom() -> None:
    """get_vlm returns the provided VLM instance unchanged."""
    custom = MagicMock(spec=BaseVLM)
    out = get_vlm(vlm=custom, vlm_type="litellm", model_name="gpt-4o")
    assert out is custom


@pytest.mark.cpu
def test_yes_no_first_token_id_groups_disjoint() -> None:
    """Prefix token ids for Yes vs No should not overlap (avoids double-counting)."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    yes_ids, no_ids = yes_no_first_token_id_groups(tok)
    assert yes_ids and no_ids
    assert not (set(yes_ids) & set(no_ids))


@pytest.mark.cpu
def test_get_vlm_requires_model_name_without_vlm() -> None:
    """get_vlm raises ValueError when no model_name is given and no vlm is provided."""
    with pytest.raises(ValueError, match="model_name"):
        get_vlm(vlm=None, vlm_type="litellm")


@pytest.mark.cpu
def test_litellm_logprob_aggregation_sums_all_yes_tokens() -> None:
    """LitellmVLM logprob scoring must sum all yes-prefix token probs, not return the first."""
    pytest.importorskip("litellm")
    import math

    import numpy as np
    from PIL import Image

    def make_top_logprob(token, logprob):
        t = MagicMock()
        t.token = token
        t.logprob = logprob
        return t

    first_tok = MagicMock()
    first_tok.top_logprobs = [
        make_top_logprob("Yes", math.log(0.10)),
        make_top_logprob(" yes", math.log(0.05)),
        make_top_logprob("No", math.log(0.20)),
        make_top_logprob(" no", math.log(0.10)),
        make_top_logprob("maybe", math.log(0.55)),
    ]

    mock_logprobs = MagicMock()
    mock_logprobs.content = [first_tok]

    mock_choice = MagicMock()
    mock_choice.logprobs = mock_logprobs
    mock_choice.message.content = "Yes"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("litellm.completion", return_value=mock_response):
        vlm = LitellmVLM(model_name="openai/gpt-4o")
        img = Image.fromarray(np.zeros((32, 32, 3), dtype="uint8"))
        score = vlm._score_with_logprobs(img, "Is there a cat?", "Yes")

    assert 0.28 < score < 0.40, f"Expected ~0.333 (sum-normalized), got {score}"


@pytest.mark.cpu
@pytest.mark.slow
def test_yes_no_token_ids_smolvlm_nonempty() -> None:
    """SmolVLM tokenizer yields non-empty yes/no prefix id groups."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    yes_ids, no_ids = yes_no_first_token_id_groups(tok)
    assert yes_ids
    assert no_ids
