"""Tests for OneIG alignment dependency masking and metric wiring."""

from unittest.mock import MagicMock

import pytest
import torch

from pruna.data.datasets.prompt import _to_oneig_record
from pruna.evaluation.metrics.metric_oneig_alignment import (
    OneIGAlignmentMetric,
    _active_oneig_question_ids,
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


def test_apply_oneig_dependency_mask_grandchild_chain() -> None:
    """3-level chain: grandparent No masks parent; grandchild uses raw parent (stays 1.0)."""
    # q1=grandparent(No), q2=parent(Yes) depends on q1, q3=grandchild(Yes) depends on q2
    raw_scores = {1: 0.0, 2: 1.0, 3: 1.0}
    dependencies = {2: [1], 3: [2]}
    filtered = apply_oneig_dependency_mask(raw_scores, dependencies)
    # q2 masked because q1 is No (raw[1]=0.0)
    assert filtered[2] == 0.0
    # q3 uses raw[2]=1.0, NOT filtered[2]=0.0 → stays 1.0
    assert filtered[3] == 1.0
    # q1 unchanged
    assert filtered[1] == 0.0


def test_active_oneig_question_ids_skips_padding() -> None:
    """Padded ``None`` and blank slots are excluded; numeric order preserved."""
    qmap = {1: "a", 21: None, 3: "  ", 2: "b"}
    assert _active_oneig_question_ids(qmap) == [1, 2]


def test_active_oneig_question_ids_skips_literal_none_string() -> None:
    r"""The literal ``\"None\"`` string is treated as a missing label (legacy / bad rows)."""
    assert _active_oneig_question_ids({1: "None", 2: "ok"}) == [2]


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
    assert result.higher_is_better is True
    assert result.metric_units == "alignment"
    assert result.result == 0.0
    call = mock_vlm.score.call_args
    assert call[0][1] == ["first", "second"]


@pytest.mark.cpu
def test_oneig_alignment_skips_none_question_texts() -> None:
    """HF ``datasets`` schema padding (``None`` question text) is not sent to the VLM."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.score.return_value = [1.0]

    metric = OneIGAlignmentMetric(vlm=mock_vlm, vlm_type="litellm", device="cpu")
    images = torch.rand(1, 3, 64, 64)
    aux = {
        "questions": {"1": "first", "21": None},
        "dependencies": {"1": [0], "21": [0]},
    }
    metric.update(["p"], [aux], images)
    result = metric.compute()
    assert result.name == "oneig_alignment"
    assert result.result == 1.0
    mock_vlm.score.assert_called_once()
    assert mock_vlm.score.call_args[0][1] == ["first"]


@pytest.mark.cpu
def test_oneig_alignment_all_padding_questions_yields_zero_without_vlm() -> None:
    """When every slot is padding, score is 0.0 and the VLM is not called."""
    mock_vlm = MagicMock(spec=BaseVLM)
    metric = OneIGAlignmentMetric(vlm=mock_vlm, vlm_type="litellm", device="cpu")
    aux = {"questions": {"1": None, "2": None}, "dependencies": {}}
    metric.update(["p"], [aux], torch.rand(1, 3, 64, 64))
    assert metric.compute().result == 0.0
    mock_vlm.score.assert_not_called()


def test_to_oneig_record_strips_null_questions_and_dependencies() -> None:
    """Null-valued Q_D entries are filtered out at record construction time."""
    row = {"category": "Anime_Stylization", "id": "001", "class": "None", "prompt_en": "a cat"}
    questions_by_key = {
        "anime_001": {
            "questions": {"1": "Is there a cat?", "21": None},
            "dependencies": {"1": [0], "21": None},
        }
    }
    record = _to_oneig_record(row, questions_by_key, {}, {})
    assert "21" not in record["questions"]
    assert "21" not in record["dependencies"]
    assert record["questions"] == {"1": "Is there a cat?"}
    assert record["dependencies"] == {"1": [0]}
