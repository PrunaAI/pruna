import pytest
from pruna.evaluation.metrics.metric_evalharness import LMEvalMetric
from pruna.evaluation.metrics.result import MetricResult
import pytest
from pruna.evaluation.task import (
    Task,
    _process_single_request,
    _get_lm_eval_task_metrics,
)
from pruna.data.pruna_datamodule import PrunaDataModule


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


@pytest.mark.cpu
def test_lm_eval_task_metric_extraction():
    """Test that _get_lm_eval_task_metrics() returns valid metric list from lm-eval task."""
    metrics = _get_lm_eval_task_metrics("lambada_openai")
    assert isinstance(metrics, list)
    assert all("metric" in m for m in metrics)
    assert any(m["metric"] == "acc" for m in metrics)


@pytest.mark.cpu
def test_process_single_request_lm_eval_task_creates_metrics():
    """Test that lm_eval:task request creates LMEvalMetric objects with correct metric names."""
    metrics = _process_single_request(
        "lm_eval:lambada_openai", stateful_metric_device="cpu", inference_device="cpu"
    )

    assert all(isinstance(m, LMEvalMetric) for m in metrics)
    assert len(metrics) > 0
    # confirm the metric names match lambada_openai config
    metric_names = [m.metric_name for m in metrics]
    expected_names = [d["metric"] for d in _get_lm_eval_task_metrics("lambada_openai")]
    assert all(name in expected_names for name in metric_names)


@pytest.mark.cpu
def test_task_initialization_with_lm_eval_request():
    """Integration test: ensure Task builds proper LMEvalMetric instances from lm_eval:task string."""
    datamodule = PrunaDataModule.from_string("LAION256")
    task = Task(request="lm_eval:lambada_openai", datamodule=datamodule, device="cpu")

    assert all(isinstance(m, LMEvalMetric) for m in task.metrics)
    assert all(
        m.metric_name
        in [d["metric"] for d in _get_lm_eval_task_metrics("lambada_openai")]
        for m in task.metrics
    )
    assert any(m.higher_is_better for m in task.metrics)
    assert (
        len(task.get_single_stateful_metrics()) > 0
    )  # LMEvalMetric is a StatefulMetric subclass


@pytest.mark.cpu
def test_task_lm_eval_invalid_task_raises():
    """Ensure invalid lm_eval task name raises ValueError."""
    datamodule = PrunaDataModule.from_string("LAION256")
    with pytest.raises(KeyError):
        Task(
            request="lm_eval:nonexistent_task_xyz", datamodule=datamodule, device="cpu"
        )
