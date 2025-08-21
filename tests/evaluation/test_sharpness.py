from typing import Any

import pytest
import torch
import torch.nn.functional as F

from pruna.config.smash_config import SmashConfig
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.engine.pruna_model import PrunaModel
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.evaluation.metrics.metric_sharpness import SharpnessMetric
from pruna.evaluation.task import Task


@pytest.mark.parametrize(
    "model_fixture, device, kernel_size",
    [
        pytest.param("flux_tiny_random", "cpu", 3, marks=pytest.mark.cpu),
        pytest.param("flux_tiny_random", "cuda", 5, marks=pytest.mark.cuda),
    ],
    indirect=["model_fixture"],
)
def test_sharpness(model_fixture: tuple[Any, SmashConfig], device: str, kernel_size: int) -> None:
    """Test SharpnessMetric initialization and scoring."""
    model, smash_config = model_fixture
    smash_config.device = device
    pruna_model = PrunaModel(model, smash_config=smash_config)


    metric = SharpnessMetric(kernel_size=kernel_size, device=device)

    batch = next(iter(smash_config.test_dataloader()))
    x, gt = batch
    outputs = pruna_model.run_inference(batch, device)

    # Calculate sharpness for model outputs
    metric.update(x, gt, outputs)
    comparison_results = metric.compute().result

    metric.reset()

    # Calculate sharpness for ground truth
    metric.update(x, gt, gt)
    gt_results = metric.compute().result

    # Both should be positive sharpness values (higher is better)
    assert comparison_results > 0.0
    assert gt_results > 0.0


def test_sharpness_blurry_vs_sharp():
    """Test that sharpness metric decreases for blurrier images."""
    # Create a sharp test image (checkerboard pattern)
    size = 64
    sharp_image = torch.zeros(1, 3, size, size)

    # Create a checkerboard pattern for sharp edges
    for i in range(size):
        for j in range(size):
            if (i // 8 + j // 8) % 2 == 0:
                sharp_image[0, :, i, j] = 1.0

    # Create a blurry version by applying average pooling to simulate blur
    # Use padding to maintain the same size
    blurry_image = F.avg_pool2d(sharp_image, kernel_size=3, stride=1, padding=1)
    # Apply multiple times to increase blur effect
    blurry_image = F.avg_pool2d(blurry_image, kernel_size=3, stride=1, padding=1)

    # Create metric
    metric = SharpnessMetric(kernel_size=3, device="cpu")

    # Test sharp image
    metric.update([sharp_image], sharp_image, sharp_image)
    sharp_score = metric.compute().result

    metric.reset()

    # Test blurry image
    metric.update([blurry_image], blurry_image, blurry_image)
    blurry_score = metric.compute().result

    # Sharp image should have higher sharpness score
    assert sharp_score > blurry_score, f"Sharp score ({sharp_score}) should be higher than blurry score ({blurry_score})"
    assert sharp_score > 0.0
    assert blurry_score > 0.0


@pytest.mark.parametrize(
    "model_fixture, device, kernel_size",
    [
        pytest.param("sd_tiny_random", "cuda", 3, marks=pytest.mark.cuda),
        pytest.param("flux_tiny_random", "cpu", 5, marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],
)
def test_task_sharpness_from_instance(model_fixture: tuple[Any, SmashConfig], device: str, kernel_size: int):
    """Test EvaluationAgent with SharpnessMetric instance."""
    model, _ = model_fixture
    data_module = PrunaDataModule.from_string("LAION256")
    data_module.limit_datasets(10)

    # Create metric instance
    sharpness_metric = SharpnessMetric(kernel_size=kernel_size, device=device)

    task = Task(
        request=[sharpness_metric],
        datamodule=data_module,
        device=device,
    )
    eval_agent = EvaluationAgent(task=task)

    result = eval_agent.evaluate(model)


    assert result[0].result > 0.0


@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("sd_tiny_random", "cuda", marks=pytest.mark.cuda),
        pytest.param("flux_tiny_random", "cpu", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],
)
def test_task_sharpness_from_string(model_fixture: tuple[Any, SmashConfig], device: str):
    """Test EvaluationAgent with sharpness metric specified as string."""
    model, _ = model_fixture
    data_module = PrunaDataModule.from_string("LAION256")
    data_module.limit_datasets(10)

    eval_agent = EvaluationAgent(
        request=["sharpness"],
        datamodule=data_module,
        device=device,
    )

    result = eval_agent.evaluate(model)

    # Sharpness should be a positive value
    assert result[0].result > 0.0


@pytest.mark.parametrize(
    "kernel_size, call_type",
    [
        (3, "single"),
        (7, "single"),
    ],
)
def test_sharpness_metric_parameters(kernel_size: int, call_type: str):
    """Test SharpnessMetric with different parameters."""
    metric = SharpnessMetric(kernel_size=kernel_size, call_type=call_type, device="cpu")

    assert metric.kernel_size == kernel_size
    assert metric.call_type == "y"
    assert metric.metric_name == "sharpness"
    assert metric.higher_is_better is True
    assert "cpu" in metric.runs_on
    assert "cuda" in metric.runs_on
