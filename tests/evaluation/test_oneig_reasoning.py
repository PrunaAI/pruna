"""Tests for OneIG reasoning metric (LLM2CLIP text-image similarity)."""

from unittest.mock import MagicMock

import pytest
import torch

from pruna.evaluation.metrics.metric_oneig_reasoning import (
    OneIGReasoningMetric,
    _LLM2CLIPScorer,
)


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
    from pruna.evaluation.metrics.registry import MetricRegistry

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


@pytest.mark.slow
@pytest.mark.skip(reason="Requires HF model download; run manually")
def test_oneig_reasoning_smoke_with_real_scorer() -> None:
    """Optional: full LLM2CLIP scorer on one sample (slow)."""
    from pruna.data.datasets.prompt import setup_oneig_knowledge_reasoning_dataset

    metric = OneIGReasoningMetric(device="cpu")
    _train, _val, test = setup_oneig_knowledge_reasoning_dataset(test_sample_size=1)
    row = test[0]
    aux = {k: v for k, v in row.items() if k != "text"}
    images = torch.rand(1, 3, 224, 224)
    metric.update([row["text"]], [aux], images)
    result = metric.compute()
    assert 0 <= result.result <= 1
