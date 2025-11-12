from typing import Any

import pytest
import torch
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure
)

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
def test_kid(datamodule_fixture: PrunaDataModule, device: str) -> None:
    """Test the kid."""
    dataloader = datamodule_fixture.val_dataloader()
    dataloader_iter = iter(dataloader)

    # Get multiple batches to ensure enough samples
    batches = []
    for _ in range(4):  # Get 4 batches
        _, batch = next(dataloader_iter)
        batches.append(batch)
    gt = torch.cat(batches, dim=0)
    
    total_samples = gt.shape[0]
    # subset_size must be strictly smaller than number of samples
    # Use a subset_size that's safely smaller (at least 1 less)
    subset_size = min(50, max(2, total_samples - 1))
    
    metric = TorchMetricWrapper("kid", device=device, subset_size=subset_size)
    metric.update(gt, gt, gt)
    result = metric.compute()
    
    # KID should be close to 0 when comparing identical images
    # Use absolute value and a slightly larger tolerance due to numerical precision
    assert not torch.isnan(torch.tensor(result.result)), f"KID returned NaN"
    assert abs(result.result) < 0.25, f"KID should be close to 0 for identical images, got {result.result}"


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
@pytest.mark.parametrize("datamodule_fixture", ["LAION256"], indirect=True)
def test_clipiqa(datamodule_fixture: PrunaDataModule) -> None:
    """Test the clipiqa."""
    metric = TorchMetricWrapper("clipiqa", device="cpu")

    dataloader = datamodule_fixture.val_dataloader()
    dataloader_iter = iter(dataloader)
    x, gt = next(dataloader_iter)
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
@pytest.mark.parametrize("datamodule_fixture", ["LAION256"], indirect=True)
def test_arniqa(datamodule_fixture: PrunaDataModule) -> None:
    """Test arniqa."""
    metric = TorchMetricWrapper("arniqa", device="cpu")
    dataloader = datamodule_fixture.val_dataloader()
    dataloader_iter = iter(dataloader)
    x, gt = next(dataloader_iter)
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

@pytest.mark.cpu
@pytest.mark.parametrize(
    'metric_name,metric_type',
    (
        ('ssim', StructuralSimilarityIndexMeasure),
        ('msssim', MultiScaleStructuralSimilarityIndexMeasure)
    )
)
def test_ssim_generalization_metric_type(metric_name, metric_type):
    wrapper = TorchMetricWrapper(metric_name=metric_name)
    assert isinstance(wrapper.metric, metric_type)

@pytest.mark.cpu
@pytest.mark.parametrize(
    'metric_name,invalid_param_args',
    (
        pytest.param('ssim', {'betas': [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]}),
        pytest.param('ssim', {'normalize': 'relu'}),
        pytest.param('msssim', {'return_full_image': True}),
        pytest.param('msssim', {'return_contrast_sensitivity': True}),
    )
)
def test_ssim_generalization_invalid_param_type(metric_name, invalid_param_args):
    with pytest.raises(ValueError):
        TorchMetricWrapper(metric_name, **invalid_param_args)
