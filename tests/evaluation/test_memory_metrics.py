import tempfile
from typing import Any

import pytest

from pruna import PrunaModel, SmashConfig
from pruna.evaluation.metrics.metric_memory import DiskMemoryMetric, InferenceMemoryMetric, TrainingMemoryMetric


@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("noref_resnet_18", "cpu", marks=pytest.mark.cpu),
        pytest.param("stable_diffusion_v1_4", "cuda", marks=pytest.mark.cuda),
    ],
    indirect=["model_fixture"],
)
def test_disk_memory_metric(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the disk memory metric."""
    model, smash_config = model_fixture
    disk_memory_metric = DiskMemoryMetric()
    pruna_model = PrunaModel(model, smash_config=smash_config)
    disk_memory_results = disk_memory_metric.compute(pruna_model, smash_config.test_dataloader())
    assert disk_memory_results.result > 0


@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("noref_resnet_18", "cpu", marks=pytest.mark.cpu),
        pytest.param("stable_diffusion_v1_4", "cuda", marks=pytest.mark.cuda),
    ],
    indirect=["model_fixture"],
)
def test_inference_memory_metric(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the inference memory metric."""
    model, smash_config = model_fixture
    inference_memory_metric = InferenceMemoryMetric()
    pruna_model = PrunaModel(model, smash_config=smash_config)
    inference_memory_results = inference_memory_metric.compute(pruna_model, smash_config.test_dataloader())
    assert inference_memory_results.result > 0


@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("noref_resnet_18", "cpu", marks=pytest.mark.cpu),
        pytest.param("stable_diffusion_v1_4", "cuda", marks=pytest.mark.cuda),
    ],
    indirect=["model_fixture"],
)
def test_training_memory_metric(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the training memory metric."""
    model, smash_config = model_fixture
    training_memory_metric = TrainingMemoryMetric()
    pruna_model = PrunaModel(model, smash_config=smash_config)
    training_memory_results = training_memory_metric.compute(pruna_model, smash_config.test_dataloader())
    assert training_memory_results.result > 0
