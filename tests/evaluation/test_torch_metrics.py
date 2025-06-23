from typing import Any

import pytest
import torch

from pruna.evaluation.metrics.metric_torch import TorchMetricWrapper
from pruna.data.pruna_datamodule import PrunaDataModule

@pytest.mark.cuda
@pytest.mark.parametrize("datamodule_fixture", ["WikiText"], indirect=True)
def test_perplexity(datamodule_fixture: PrunaDataModule) -> None:
    """Test the perplexity."""
    metric = TorchMetricWrapper("perplexity")
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


@pytest.mark.cpu
@pytest.mark.parametrize("datamodule_fixture", ["LAION256"], indirect=True)
def test_fid(datamodule_fixture: PrunaDataModule) -> None:
    """Test the fid."""
    metric = TorchMetricWrapper("fid")
    dataloader = datamodule_fixture.val_dataloader()
    dataloader_iter = iter(dataloader)

    _, gt1 = next(dataloader_iter)
    _, gt2 = next(dataloader_iter)
    gt = torch.cat([gt1, gt2], dim=0)
    metric.update(gt, gt, gt)
    assert metric.compute().result == pytest.approx(0.0, abs=1e-2)


@pytest.mark.cpu
@pytest.mark.parametrize("datamodule_fixture", ["LAION256"], indirect=True)
def test_clip_score(datamodule_fixture: PrunaDataModule) -> None:
    """Test the clip score."""
    metric = TorchMetricWrapper("clip_score")
    dataloader = datamodule_fixture.val_dataloader()
    dataloader_iter = iter(dataloader)

    x, gt = next(dataloader_iter)
    metric.update(x, gt, gt)
    score = metric.compute()
    assert score.result > 0.0 and score.result < 100.0


@pytest.mark.cpu
@pytest.mark.parametrize("datamodule_fixture", ["ImageNet"], indirect=True)
@pytest.mark.parametrize("metric", ["accuracy", "recall", "precision"])
def test_torch_metrics(datamodule_fixture: PrunaDataModule, metric: str) -> None:
    """Test the torch metrics accuracy, recall, precision."""
    metric = TorchMetricWrapper(metric, task="multiclass", num_classes=1000)
    dataloader = datamodule_fixture.val_dataloader()
    dataloader_iter = iter(dataloader)

    x, gt = next(dataloader_iter)
    metric.update(gt, gt, gt)
    assert metric.compute().result == 1.0
