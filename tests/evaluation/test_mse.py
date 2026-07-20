import math
import pytest
import torch

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.metrics import MetricRegistry
from pruna.evaluation.metrics.metric_mse import MSE


@pytest.mark.parametrize(
    "datamodule_fixture, device",
    [
        pytest.param("LAION256", "cpu", marks=pytest.mark.cpu),
        pytest.param("LAION256", "cuda", marks=pytest.mark.cuda),
    ],
    indirect=["datamodule_fixture"],
)
def test_mse_on_pruna_data(datamodule_fixture: PrunaDataModule, device: str) -> None:
    """Test the MSE on real images: zero for identical images and positive for different ones."""
    metric = MSE(device=device)
    dataloader = datamodule_fixture.val_dataloader()
    dataloader_iter = iter(dataloader)

    x, gt1 = next(dataloader_iter)
    _, gt2 = next(dataloader_iter)

    # Identical images overlap perfectly, so their MSE is 0.0
    metric.update(x, gt1, gt1)
    assert metric.compute().result == pytest.approx(0.0)

    # Two different images have a positive pixel-wise error
    metric.reset()
    metric.update(x, gt1, gt2)
    assert metric.compute().result > 0.0


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
def test_mse_identical_tensors(device: str) -> None:
    """Test that the MSE of two identical tensors is zero."""
    image = torch.rand(1, 3, 16, 16)

    metric = MSE(device=device)
    metric.update(image, image, image)

    assert metric.compute().result == pytest.approx(0.0)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
def test_mse_known_value(device: str) -> None:
    """Test the MSE against a simple pre-computed example."""
    gt = torch.zeros(2, 2)
    outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    metric = MSE(device=device)
    metric.update(gt, gt, outputs)

    assert metric.compute().result == pytest.approx(7.5)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
def test_mse_accumulates_and_resets(device: str) -> None:
    """Test that the MSE is pooled across batches and that reset clears the state."""
    gt = torch.zeros(1, 3, 4, 4) # Ground-truth of 0.0, 1*3*4*4 = 48 elements per batch
    outputs_batch_one = torch.ones_like(gt)  # Squared error of 1.0 per element
    outputs_batch_two = torch.full_like(gt, 3.0)  # Squared error of 9.0 per element

    metric = MSE(device=device)
    metric.update(gt, gt, outputs_batch_one)
    metric.update(gt, gt, outputs_batch_two)

    # The pooled mean is (48 * 1.0 + 48 * 9.0) / (48 + 48) = 5.0
    assert metric.compute().result == pytest.approx(5.0)

    metric.reset()
    metric.update(gt, gt, outputs_batch_one)
    assert metric.compute().result == pytest.approx(1.0)


@pytest.mark.cpu
def test_mse_no_update() -> None:
    """Test that computing without any update explicitly returns NaN."""
    metric = MSE(device="cpu")

    assert math.isnan(metric.compute().result)


@pytest.mark.cpu
def test_mse_shape_mismatch_raises() -> None:
    """Test that comparing tensors of different shapes raises a ValueError."""
    gt = torch.zeros(1, 3, 8, 8)
    outputs = torch.zeros(1, 3, 4, 4)

    metric = MSE(device="cpu")
    with pytest.raises(ValueError):
        metric.update(gt, gt, outputs)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "call_type, expected_call_type",
    [
        ("single", "gt_y"),
        ("pairwise", "pairwise_gt_y"),
    ],
)
def test_mse_call_type(call_type: str, expected_call_type: str) -> None:
    """Test that the MSE exposes the expected attributes and resolves its call type."""
    metric = MSE(call_type=call_type, device="cpu")

    assert metric.metric_name == "mse"
    assert metric.higher_is_better is False
    assert metric.call_type == expected_call_type
