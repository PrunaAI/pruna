import pytest
import torch

from pruna.evaluation.metrics.metric_evalharness import EvalHarnessMetric
from pruna.evaluation.metrics.result import MetricResult


@pytest.mark.cpu
def test_eval_harness_basic():
    """Test EvalHarnessMetric on a tiny HF model + task."""

    # Use a very small HF model for speed
    model_args = {
        "pretrained": "sshleifer/tiny-gpt2",
        "revision": "main",
    }

    metric = EvalHarnessMetric(
        tasks=["lambada_openai"],  # lightweight benchmark
        model_args=model_args,
        device=torch.device("cpu"),
    )

    # compute ignores Pruna dataloader/model, so pass None
    result: MetricResult = metric.compute(model=None, dataloader=None)

    print(f"Result: {result}")

    # Assertions
    assert isinstance(result, MetricResult)
    assert "task_scores" in result.params
    assert "lambada_openai" in result.params["task_scores"]
    # Result should be a float between 0 and 1 for accuracy-like tasks
    score = result.params["task_scores"]["lambada_openai"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # Final score matches average
    assert result.result == pytest.approx(score, abs=1e-6)


@pytest.mark.cpu
def test_eval_harness_multiple_tasks():
    """Test EvalHarnessMetric with multiple tasks aggregated."""

    model_args = {
        "pretrained": "sshleifer/tiny-gpt2",
    }

    tasks = ["lambada_openai", "hellaswag"]

    metric = EvalHarnessMetric(
        tasks=tasks,
        model_args=model_args,
        device=torch.device("cpu"),
    )

    result = metric.compute(model=None, dataloader=None)

    # It should produce scores for all requested tasks
    for task in tasks:
        assert task in result.params["task_scores"]

    # Final score is average of per-task scores
    scores = list(result.params["task_scores"].values())
    assert result.result == pytest.approx(sum(scores) / len(scores), rel=1e-6)
