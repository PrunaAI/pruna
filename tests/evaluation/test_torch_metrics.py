from typing import Any

import pytest
import torch

from pruna.evaluation.metrics.metric_torch import TorchMetricWrapper, TorchMetrics
from pruna.data.pruna_datamodule import PrunaDataModule

@pytest.mark.parametrize(
    "datamodule_fixture, device",
    [
        pytest.param("WikiText", "cpu", marks=pytest.mark.cpu),
        pytest.param("WikiText", "cuda", marks=pytest.mark.cuda),
    ],
    indirect=["datamodule_fixture"],
)
def test_perplexity(datamodule_fixture: PrunaDataModule, device: str) -> None:
    """Test the perplexity."""
    metric = TorchMetricWrapper("perplexity", device=device)
    dataloader = datamodule_fixture.val_dataloader()

    _, gt = next(iter(dataloader))

    vocab_size = 50257
    logits = torch.zeros(gt.shape[0], gt.shape[1], vocab_size)

    for b in range(gt.shape[0]):
        for s in range(gt.shape[1]):
            logits[b, s, gt[b, s]] = 100.0

    metric.update(gt, gt, logits)
    result = metric.compute()
    assert result.result == 1.0


@pytest.mark.parametrize(
    "datamodule_fixture, device",
    [
        pytest.param("LAION256", "cpu", marks=pytest.mark.cpu),
        pytest.param("LAION256", "cuda", marks=pytest.mark.cuda),
    ],
    indirect=["datamodule_fixture"],
)
def test_fid(datamodule_fixture: PrunaDataModule, device: str) -> None:
    """Test the fid."""
    metric = TorchMetricWrapper("fid", device=device)
    dataloader = datamodule_fixture.val_dataloader()
    dataloader_iter = iter(dataloader)

    _, gt1 = next(dataloader_iter)
    _, gt2 = next(dataloader_iter)
    gt = torch.cat([gt1, gt2], dim=0)
    metric.update(gt, gt, gt)
    assert metric.compute().result == pytest.approx(0.0, abs=1e-2)


@pytest.mark.parametrize(
    "datamodule_fixture, device",
    [
        pytest.param("LAION256", "cpu", marks=pytest.mark.cpu),
        pytest.param("LAION256", "cuda", marks=pytest.mark.cuda),
    ],
    indirect=["datamodule_fixture"],
)
def test_clip_score(datamodule_fixture: PrunaDataModule, device: str) -> None:
    """Test the clip score."""
    metric = TorchMetricWrapper("clip_score", device=device)
    dataloader = datamodule_fixture.val_dataloader()
    dataloader_iter = iter(dataloader)

    x, gt = next(dataloader_iter)
    metric.update(x, gt, gt)
    score = metric.compute()
    assert score.result > 0.0 and score.result < 100.0

@pytest.mark.cpu
@pytest.mark.parametrize("dataloader_fixture", ["LAION256"], indirect=True)
def test_clipiqa(dataloader_fixture: Any) -> None:
    """Test the clipiqa."""
    metric = TorchMetricWrapper("clipiqa")
    x, gt = next(iter(dataloader_fixture))
    metric.update(x, gt, gt)
    score = metric.compute()
    assert score.result > 0.0 and score.result < 1.0


@pytest.mark.parametrize(
    "datamodule_fixture, device, metric",
    [
        pytest.param("ImageNet", "cpu", "accuracy", marks=pytest.mark.cpu),
        pytest.param("ImageNet", "cuda", "accuracy", marks=pytest.mark.cuda),
        pytest.param("ImageNet", "cpu", "recall", marks=pytest.mark.cpu),
        pytest.param("ImageNet", "cuda", "recall", marks=pytest.mark.cuda),
        pytest.param("ImageNet", "cpu", "precision", marks=pytest.mark.cpu),
        pytest.param("ImageNet", "cuda", "precision", marks=pytest.mark.cuda),
    ],
    indirect=["datamodule_fixture"],
)
def test_torch_metrics(datamodule_fixture: PrunaDataModule, device: str, metric: str) -> None:
    """Test the torch metrics accuracy, recall, precision."""
    metric = TorchMetricWrapper(metric, task="multiclass", num_classes=1000, device=device)
    dataloader = datamodule_fixture.val_dataloader()
    dataloader_iter = iter(dataloader)

    x, gt = next(dataloader_iter)
    metric.update(gt, gt, gt)
    assert metric.compute().result == 1.0

@pytest.mark.cpu
@pytest.mark.parametrize("dataloader_fixture", ["LAION256"], indirect=True)
def test_arniqa(dataloader_fixture: Any) -> None:
    """Test arniqa."""
    metric = TorchMetricWrapper("arniqa")
    x, gt = next(iter(dataloader_fixture))
    metric.update(x, gt, gt)

@pytest.mark.cpu
@pytest.mark.parametrize("metric", TorchMetrics.__members__.keys())
@pytest.mark.parametrize("call_type", ["single", "pairwise"])
def test_check_call_type(metric: str, call_type: str):
    """Check the call type of the metric."""
    kwargs = {}
    if metric in ['accuracy', 'recall', 'precision']:
        kwargs = {"task": "multiclass", "num_classes": 1000}
    if metric in ["arniqa", "clipiqa"] and call_type == "pairwise":
        with pytest.raises(Exception):
            TorchMetricWrapper(metric, call_type=call_type, **kwargs)
        return
    metric = TorchMetricWrapper(metric, call_type=call_type, **kwargs)
    if call_type == "pairwise" and metric.metric_name not in ["arniqa", "clipiqa"]:
        assert metric.call_type.startswith("pairwise")
    elif metric.metric_name in ["arniqa", "clipiqa"]:
        assert metric.call_type == "y"
    else:
        assert not metric.call_type.startswith("pairwise")
