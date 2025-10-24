import pytest
from unittest.mock import patch
from pruna.evaluation.task import Task
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.data import base_datasets
from pruna.evaluation.metrics.metric_torch import TorchMetrics
from torchmetrics.classification import Accuracy, Precision, Recall
from functools import partial
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.engine.utils import split_device, device_to_string
from ..common import device_parametrized
from pruna.evaluation.metrics.metric_elapsed_time import LatencyMetric
from pruna.evaluation.metrics.metric_cmmd import CMMD
from pruna.evaluation.metrics.metric_pairwise_clip import PairwiseClipScore
from pruna.evaluation.metrics.metric_torch import TorchMetricWrapper

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


@device_parametrized
def test_device_is_set_correctly_for_metrics(device:str):
    task = Task(request=['latency', 'cmmd', 'pairwise_clip_score'], datamodule=PrunaDataModule.from_string("LAION256"), device = device)
    assert split_device(device_to_string(task.device)) == split_device(device_to_string(device))
    for metric in task.metrics:
        if isinstance(metric, BaseMetric):
            assert split_device(device_to_string(metric.device)) == split_device(device_to_string(device))
        elif isinstance(metric, StatefulMetric):
            if hasattr(metric, 'metric'):
                assert split_device(device_to_string(metric.metric.device)) == split_device(device_to_string(task.stateful_metric_device))
            assert split_device(device_to_string(metric.device)) == split_device(device_to_string(task.stateful_metric_device))


@pytest.mark.cuda
@pytest.mark.parametrize(
    "inference_device, stateful_metric_device, task_device",
    [
        ("accelerate", "cpu", "cpu"),
        ("accelerate", "cpu", "cuda"),
        ("accelerate", "cpu", "accelerate"),
        ("accelerate", "cuda", "cpu"),
        ("accelerate", "cuda", "cuda"),
        ("accelerate", "cuda", "accelerate"),
        ("cpu", "cpu", "cuda"),
        ("cpu", "cpu", "cpu"),
        ("cpu", "cpu", "accelerate"),
        ("cpu", "cuda", "cpu"),
        ("cpu", "cuda", "cuda"),
        ("cpu", "cuda", "accelerate"),
        ("cuda", "cpu", "cuda"),
        ("cuda", "cpu", "cpu"),
        ("cuda", "cpu", "accelerate"),
        ("cuda", "cuda", "cuda"),
        ("cuda", "cuda", "accelerate"),
        ("cuda", "cuda", "cpu"),
    ],
)
def test_metric_device_adapts_to_task_device(inference_device: str, stateful_metric_device: str, task_device: str):
    """ Test that the metrics in the task are moved to the task device if they are on a different device."""
    latency = LatencyMetric(device=inference_device)
    cmmd = CMMD(device=stateful_metric_device)
    pairwise_clip_score = PairwiseClipScore(device=stateful_metric_device)
    psnr = TorchMetricWrapper('psnr', device=stateful_metric_device)

    task = Task(request=[latency, cmmd, pairwise_clip_score, psnr], datamodule=PrunaDataModule.from_string("LAION256"), device = task_device)
    assert split_device(device_to_string(task.device)) == split_device(device_to_string(task_device))
    for metric in task.metrics:
        if isinstance(metric, BaseMetric):
            assert split_device(device_to_string(metric.device)) == split_device(device_to_string(task.device))
        elif isinstance(metric, StatefulMetric):
            if hasattr(metric, "device"):
                assert split_device(device_to_string(metric.device)) == split_device(device_to_string(task.stateful_metric_device))
            if hasattr(metric, "metric") and hasattr(metric.metric, "device"): # Wrapper metric
                assert split_device(device_to_string(metric.metric.device)) == split_device(device_to_string(task.stateful_metric_device))
            if not hasattr(metric, "device") and not hasattr(metric.metric, "device"):
                raise ValueError("Could not find device for metric.")

@pytest.mark.cpu
def test_task_from_string_request():
    request = ["cmmd", "pairwise_clip_score", "psnr"]
    task = Task(request=request, datamodule=PrunaDataModule.from_string("LAION256"), device = "cpu")
    assert isinstance(task.metrics[0], CMMD)
    assert isinstance(task.metrics[1], PairwiseClipScore)
    assert isinstance(task.metrics[2], TorchMetricWrapper)


@pytest.mark.parametrize("metrics, modality", [
    (["cmmd", "ssim", "lpips", "latency"], "image"),
    (["perplexity", "disk_memory"], "text"),
    ([MetricRegistry().get_metric("accuracy", task="binary"), MetricRegistry().get_metric("recall", task="binary")], "general"),
    (["background_consistency", "dynamic_degree"], "video")
])
def test_task_modality(metrics, modality):
    """ Test that the task modality is assigned correctly for image, text, general and video metrics."""
    datamodule = type("dm", (), {"test_dataloader": lambda self: []})()
    task = Task(request=metrics, datamodule=datamodule)
    assert task.modality == modality

def test_task_modality_mixed_raises():
    """ Test that we raise an error if the task modality is mixed."""
    datamodule = type("dm", (), {"test_dataloader": lambda self: []})()
    with pytest.raises(ValueError):
        Task(request=["cmmd", "background_consistency"], datamodule=datamodule)
