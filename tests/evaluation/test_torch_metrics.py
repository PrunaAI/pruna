from typing import Any

import pytest
import torch

from pruna.evaluation.metrics.metric_torch import TorchMetricWrapper


@pytest.mark.cuda
@pytest.mark.parametrize("dataloader_fixture", ["WikiText"], indirect=True)
def test_perplexity(dataloader_fixture: Any) -> None:
    """Test the perplexity."""
    metric = TorchMetricWrapper("perplexity")

    _, gt = next(iter(dataloader_fixture))

    vocab_size = 50257
    logits = torch.zeros(gt.shape[0], gt.shape[1], vocab_size)

    for b in range(gt.shape[0]):
        for s in range(gt.shape[1]):
            logits[b, s, gt[b, s]] = 100.0

    metric.update(gt, gt, logits)
    result = metric.compute()
    assert result.result == 1.0


@pytest.mark.cpu
@pytest.mark.parametrize("dataloader_fixture", ["LAION256"], indirect=True)
def test_fid(dataloader_fixture: Any) -> None:
    """Test the fid."""
    metric = TorchMetricWrapper("fid")

    dataloader_iter = iter(dataloader_fixture)

    _, gt1 = next(dataloader_iter)
    _, gt2 = next(dataloader_iter)
    gt = torch.cat([gt1, gt2], dim=0)
    metric.update(gt, gt, gt)
    assert metric.compute().result == pytest.approx(0.0, abs=1e-2)


@pytest.mark.cpu
@pytest.mark.parametrize("dataloader_fixture", ["LAION256"], indirect=True)
def test_clip_score(dataloader_fixture: Any) -> None:
    """Test the clip score."""
    metric = TorchMetricWrapper("clip_score")
    x, gt = next(iter(dataloader_fixture))
    metric.update(x, gt, gt)
    score = metric.compute()
    assert score.result > 0.0 and score.result < 100.0


@pytest.mark.cpu
@pytest.mark.parametrize("dataloader_fixture", ["ImageNet"], indirect=True)
@pytest.mark.parametrize("metric", ["accuracy", "recall", "precision"])
def test_torch_metrics(dataloader_fixture: Any, metric: str) -> None:
    """Test the torch metrics accuracy, recall, precision."""
    metric = TorchMetricWrapper(metric, task="multiclass", num_classes=1000)
    _, gt = next(iter(dataloader_fixture))
    metric.update(gt, gt, gt)
    assert metric.compute().result == 1.0
