"""Tests for OneIG alignment (masking, wiring) and OneIG reasoning (LLM2CLIP)."""

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
from pruna.evaluation.metrics.metric_oneig_reasoning import (
    OneIGReasoningMetric,
    _LLM2CLIPScorer,
)
from pruna.evaluation.metrics.registry import MetricRegistry
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
    raw_scores = {1: 0.0, 2: 1.0, 3: 1.0}
    dependencies = {2: [1], 3: [2]}
    filtered = apply_oneig_dependency_mask(raw_scores, dependencies)
    assert filtered[2] == 0.0
    assert filtered[3] == 1.0
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


def _make_mock_scorer(return_value: float = 0.5) -> MagicMock:
    mock = MagicMock(spec=_LLM2CLIPScorer)
    mock.score.return_value = [return_value]
    return mock


@pytest.mark.cpu
def test_oneig_reasoning_uses_gt_answer_from_aux() -> None:
    """Metric reads reasoning_gt_answer from aux."""
    mock_scorer = _make_mock_scorer(0.7)
    metric = OneIGReasoningMetric(scorer=mock_scorer, device="cpu")
    images = torch.rand(1, 3, 64, 64)
    aux = {"reasoning_gt_answer": "A blue circle"}
    metric.update(["p"], [aux], images)
    result = metric.compute()
    assert result.name == "oneig_reasoning"
    assert result.result == 0.7
    mock_scorer.score.assert_called_once()
    call_args = mock_scorer.score.call_args
    assert call_args[0][1] == "A blue circle"


@pytest.mark.cpu
def test_oneig_reasoning_averages_per_sample_scores() -> None:
    """Compute returns mean of per-sample scores."""
    mock_scorer = _make_mock_scorer(0.5)
    metric = OneIGReasoningMetric(scorer=mock_scorer, device="cpu")
    images = torch.rand(2, 3, 64, 64)
    aux_list = [
        {"reasoning_gt_answer": "First answer"},
        {"reasoning_gt_answer": "Second answer"},
    ]
    metric.update(["p1", "p2"], aux_list, images)
    result = metric.compute()
    assert result.result == 0.5
    assert mock_scorer.score.call_count == 2


@pytest.mark.cpu
def test_oneig_reasoning_missing_gt_raises() -> None:
    """Missing GT answer raises ValueError."""
    mock_scorer = _make_mock_scorer(0.8)
    metric = OneIGReasoningMetric(scorer=mock_scorer, device="cpu")
    images = torch.rand(1, 3, 64, 64)
    aux = {}
    with pytest.raises(ValueError, match="reasoning_gt_answer"):
        metric.update(["p"], [aux], images)
    mock_scorer.score.assert_not_called()


@pytest.mark.cpu
def test_oneig_reasoning_scorer_none_raises() -> None:
    """When scorer returns None, metric raises RuntimeError."""
    mock_scorer = _make_mock_scorer()
    mock_scorer.score.return_value = None
    metric = OneIGReasoningMetric(scorer=mock_scorer, device="cpu")
    images = torch.rand(1, 3, 64, 64)
    aux = {"reasoning_gt_answer": "Some answer"}
    with pytest.raises(RuntimeError, match="no scores"):
        metric.update(["p"], [aux], images)


@pytest.mark.cpu
def test_oneig_reasoning_compute_without_update_raises() -> None:
    """Compute with no updates raises RuntimeError."""
    mock_scorer = _make_mock_scorer()
    metric = OneIGReasoningMetric(scorer=mock_scorer, device="cpu")
    with pytest.raises(RuntimeError, match="no samples were scored"):
        metric.compute()


@pytest.mark.cpu
def test_oneig_reasoning_has_metric_registered() -> None:
    """oneig_reasoning is available via MetricRegistry (lazy)."""
    assert MetricRegistry.has_metric("oneig_reasoning")


@pytest.mark.cpu
def test_transformers_major_version_supported_for_oneig_reasoning() -> None:
    """Enforce pyproject ``transformers<5`` expectation for LLM2CLIP loading."""
    import transformers

    major = int(transformers.__version__.split(".", 1)[0])
    assert major < 5, (
        "oneig_reasoning expects transformers 4.x (see pyproject.toml); "
        "5.x from_pretrained buffer initialization can break CLIP/Llama stacks."
    )
