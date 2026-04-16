"""Unit tests for vlm_utils score parsing."""

import pytest

from pruna.evaluation.metrics.vlm_utils import FloatOutput, get_score_from_response


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
