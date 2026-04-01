from functools import partial
from unittest.mock import patch

import pytest
from torchmetrics.classification import Accuracy, Precision, Recall
from transformers import AutoTokenizer

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.engine.utils import device_to_string, split_device
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.metric_cmmd import CMMD
from pruna.evaluation.metrics.metric_elapsed_time import LatencyMetric
from pruna.evaluation.metrics.metric_pairwise_clip import PairwiseClipScore
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.metric_torch import TorchMetrics, TorchMetricWrapper
from pruna.evaluation.metrics.metric_vqa import VQAMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.task import Task

from ..common import device_parametrized


def _require(condition: bool, message: str = "test condition failed") -> None:
    if not condition:
        pytest.fail(message)


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
    """All registered metric names should instantiate through Task."""
    datamodule = PrunaDataModule.from_string("LAION256")
    Task(request=[metric_name], datamodule=datamodule, device="cpu")


@patch("pruna.evaluation.task.set_to_best_available_device")
def test_vlm_metrics_fallback_to_cpu_on_auto_device(mock_set_to_best_available_device):
    """VLM metrics should stay on CPU when task auto-selects CUDA."""
    def fake_best_device(device=None, *args, **kwargs):
        if device is None:
            return "cuda"
        return device

    mock_set_to_best_available_device.side_effect = fake_best_device

    task = Task(request=["vqa"], datamodule=PrunaDataModule.from_string("PartiPrompts"))

    _require(split_device(device_to_string(task.device))[0] == "cuda")
    _require(isinstance(task.metrics[0], VQAMetric))
    _require(split_device(device_to_string(task.metrics[0].device))[0] == "cpu")


@device_parametrized
def test_device_is_set_correctly_for_metrics(device: str):
    """Task and metric devices should align with the requested device."""
    task = Task(
        request=["latency", "cmmd", "pairwise_clip_score"],
        datamodule=PrunaDataModule.from_string("LAION256"),
        device=device,
    )
    _require(split_device(device_to_string(task.device)) == split_device(device_to_string(device)))
    for metric in task.metrics:
        if isinstance(metric, BaseMetric):
            _require(split_device(device_to_string(metric.device)) == split_device(device_to_string(device)))
        elif isinstance(metric, StatefulMetric):
            if hasattr(metric, "metric"):
                _require(
                    split_device(device_to_string(metric.metric.device))
                    == split_device(device_to_string(task.stateful_metric_device))
                )
            _require(
                split_device(device_to_string(metric.device))
                == split_device(device_to_string(task.stateful_metric_device))
            )


@pytest.mark.cuda
# We need to mark this test as cuda because it requires modern architectures
@pytest.mark.high
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
    """Test that the metrics in the task are moved to the task device if they are on a different device."""
    latency = LatencyMetric(device=inference_device)
    cmmd = CMMD(device=stateful_metric_device)
    pairwise_clip_score = PairwiseClipScore(device=stateful_metric_device)
    psnr = TorchMetricWrapper("psnr", device=stateful_metric_device)

    task = Task(
        request=[latency, cmmd, pairwise_clip_score, psnr],
        datamodule=PrunaDataModule.from_string("LAION256"),
        device=task_device,
    )
    _require(split_device(device_to_string(task.device)) == split_device(device_to_string(task_device)))
    for metric in task.metrics:
        if isinstance(metric, BaseMetric):
            _require(split_device(device_to_string(metric.device)) == split_device(device_to_string(task.device)))
        elif isinstance(metric, StatefulMetric):
            if hasattr(metric, "device"):
                _require(
                    split_device(device_to_string(metric.device))
                    == split_device(device_to_string(task.stateful_metric_device))
                )
            if hasattr(metric, "metric") and hasattr(metric.metric, "device"):  # Wrapper metric
                _require(
                    split_device(device_to_string(metric.metric.device))
                    == split_device(device_to_string(task.stateful_metric_device))
                )
            if not hasattr(metric, "device") and not hasattr(metric.metric, "device"):
                raise ValueError("Could not find device for metric.")


@pytest.mark.cpu
def test_task_from_string_request():
    """Task should instantiate requested metric wrappers by name."""
    request = ["cmmd", "pairwise_clip_score", "psnr"]
    task = Task(request=request, datamodule=PrunaDataModule.from_string("LAION256"), device="cpu")
    _require(isinstance(task.metrics[0], CMMD))
    _require(isinstance(task.metrics[1], PairwiseClipScore))
    _require(isinstance(task.metrics[2], TorchMetricWrapper))


@pytest.mark.cpu
def test_task_text_generation_quality_request():
    """Test that 'text_generation_quality' named request creates perplexity metric."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    task = Task(
        request="text_generation_quality",
        datamodule=PrunaDataModule.from_string("TinyWikiText", tokenizer=tokenizer),
        device="cpu",
    )
    _require(len(task.metrics) == 1)
    _require(isinstance(task.metrics[0], TorchMetricWrapper))
    _require(task.metrics[0].metric_name == "perplexity")


@pytest.mark.cpu
def test_task_invalid_named_request():
    """Test that an invalid named request raises a ValueError."""
    with pytest.raises(ValueError, match="not found"):
        Task(request="nonexistent_quality", datamodule=PrunaDataModule.from_string("LAION256"), device="cpu")
