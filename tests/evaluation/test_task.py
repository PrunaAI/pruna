import pytest
from unittest.mock import patch
from pruna.evaluation.task import Task
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.data import base_datasets
from pruna.evaluation.metrics.metric_torch import TorchMetrics
from torchmetrics.classification import Accuracy, Precision, Recall
from functools import partial

@pytest.fixture(autouse=True)
def _mock_torch_metrics():
    """Mock TorchMetrics enum values for accuracy, perplexity and recall only for this test."""
    def make_mock_metric(metric_class):
        metric = partial(metric_class, task="binary")
        metric.tm = metric
        metric.update_fn = None
        metric.call_type = "y_gt"
        return metric

    mock_metrics = {
        "accuracy": make_mock_metric(Accuracy),
        "precision": make_mock_metric(Precision),
        "recall": make_mock_metric(Recall)
    }

    with patch.object(TorchMetrics, '_member_map_', {**TorchMetrics._member_map_, **mock_metrics}):
        yield

@pytest.mark.parametrize("metric_name", MetricRegistry()._registry)
def test_metric_initialization_from_metric_name(metric_name):
    datamodule = PrunaDataModule.from_string("LAION256")
    Task(request=[metric_name], datamodule=datamodule)

@pytest.mark.parametrize("metrics, modality", [
    (["cmmd", "ssim", "lpips", "latency"], "image"),
    (["perplexity", "disk_memory"], "text"),
    ([MetricRegistry().get_metric("accuracy", task="binary"), MetricRegistry().get_metric("recall", task="binary")], "general"),
    (["background_consistency", "dynamic_degree"], "video")
])
def test_task_modality(metrics, modality):
    datamodule = type("dm", (), {"test_dataloader": lambda self: []})()
    task = Task(request=metrics, datamodule=datamodule)
    assert task.modality == modality

def test_task_modality_mixed_raises():
    datamodule = type("dm", (), {"test_dataloader": lambda self: []})()
    with pytest.raises(ValueError):
        Task(request=["cmmd", "background_consistency"], datamodule=datamodule)
