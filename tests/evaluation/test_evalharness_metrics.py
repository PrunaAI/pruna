import pytest
import numpy as np
from pruna.evaluation.metrics.metric_evalharness import LMEvalMetric
from pruna.evaluation.metrics.result import MetricResult


@pytest.mark.cpu
def test_lm_eval_metric_bleu_like():
    """Test BLEU metric (string overlap metric)."""

    refs = ["the cat is on the mat", "a quick brown fox"]
    preds = ["the cat is on mat", "a quick brown fox"]

    metric = LMEvalMetric(metric_name="bleu")
    metric.update(preds, refs)
    result = metric.compute()

    assert isinstance(result, MetricResult)
    assert "num_samples" in result.params
    assert result.params["num_samples"] == 2
    assert isinstance(result.result, float)


@pytest.mark.cpu
def test_lm_eval_metric_empty_pairs():
    """Test that compute() returns 0.0 when no pairs are provided."""

    metric = LMEvalMetric(metric_name="acc")
    result = metric.compute()

    assert isinstance(result, MetricResult)
    assert result.result == 0.0
    assert result.params["num_samples"] == 0


@pytest.mark.cpu
def test_lm_eval_metric_length_mismatch():
    """Test that mismatched preds/refs raises an error."""

    metric = LMEvalMetric(metric_name="acc")

    refs = ["a", "b", "c"]
    preds = ["a", "b"]

    with pytest.raises(ValueError, match="Preds and refs length mismatch"):
        metric.update(preds, refs)
