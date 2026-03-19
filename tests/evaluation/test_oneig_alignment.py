"""Tests for OneIG alignment dependency masking and metric wiring."""

from unittest.mock import MagicMock

import pytest
import torch

from pruna.evaluation.metrics.metric_oneig_alignment import (
    OneIGAlignmentMetric,
    aggregate_oneig_alignment_per_cell,
    apply_oneig_dependency_mask,
)
from pruna.evaluation.metrics.vlm_base import BaseVLM


def test_apply_oneig_dependency_mask_parent_no_zeros_child() -> None:
    """Parent ``No`` forces dependent question score to zero."""
    raw = {1: 0.0, 2: 1.0}
    deps = {1: [0], 2: [1]}
    out = apply_oneig_dependency_mask(raw, deps)
    assert out[1] == 0.0
    assert out[2] == 0.0
    assert aggregate_oneig_alignment_per_cell(out, [1, 2]) == 0.0


def test_apply_oneig_dependency_mask_parent_yes_keeps_child() -> None:
    """All ``Yes`` yields nonzero child and mean 1.0 over two questions."""
    raw = {1: 1.0, 2: 1.0}
    deps = {1: [0], 2: [1]}
    out = apply_oneig_dependency_mask(raw, deps)
    assert out == {1: 1.0, 2: 1.0}
    assert aggregate_oneig_alignment_per_cell(out, [1, 2]) == 1.0


def test_apply_oneig_dependency_mask_uses_raw_parent_not_filtered_for_chain() -> None:
    r"""Grandchild may stay 1 when parent's **raw** VLM score is Yes even if parent was masked to 0."""
    raw = {1: 0.0, 2: 1.0, 3: 1.0}
    deps = {1: [0], 2: [1], 3: [2]}
    out = apply_oneig_dependency_mask(raw, deps)
    assert out[1] == 0.0
    assert out[2] == 0.0
    assert out[3] == 1.0


@pytest.mark.cpu
def test_oneig_alignment_metric_respects_question_id_order() -> None:
    """Questions are scored in numeric id order; masking uses aligned raw scores."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.score.return_value = [0.0, 1.0]

    metric = OneIGAlignmentMetric(vlm=mock_vlm, vlm_type="litellm", device="cpu")
    images = torch.rand(1, 3, 64, 64)
    aux = {
        "questions": {"2": "second", "1": "first"},
        "dependencies": {"1": [0], "2": [1]},
    }
    metric.update(["p"], [aux], images)
    result = metric.compute()
    assert result.name == "oneig_alignment"
    assert result.result == 0.0
    call = mock_vlm.score.call_args
    assert call[0][1] == ["first", "second"]
