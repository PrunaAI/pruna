"""GenEval task wires strict multi-question QA plus CLIP."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pruna.evaluation.task import Task


@pytest.mark.cpu
@pytest.mark.slow
@patch("pruna.evaluation.task.PrunaDataModule.from_string")
def test_geneval_from_benchmark_uses_qa_accuracy_all_or_nothing(mock_from_string: MagicMock) -> None:
    """GenEval uses strict per-image QA aggregation and CLIP."""
    mock_dm = MagicMock()
    mock_dm.test_dataloader.return_value = iter([])
    mock_from_string.return_value = mock_dm
    task = Task.from_benchmark("GenEval", dataloader_args={"batch_size": 1})
    qa = next(m for m in task.metrics if getattr(m, "metric_name", None) == "qa_accuracy")
    assert qa.aggregation == "all_or_nothing"
    assert any(getattr(m, "metric_name", None) == "clip_score" for m in task.metrics)
