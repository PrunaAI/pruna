from typing import Any

import pytest

from pruna import PrunaModel, SmashConfig
from pruna.engine.utils import move_to_device
from pruna.evaluation.metrics.metric_memory import DiskMemoryMetric, InferenceMemoryMetric, TrainingMemoryMetric

@pytest.mark.cuda
@pytest.mark.parametrize(
    "model_fixture",
    [
       "sd_tiny_random",
    ],
    indirect=["model_fixture"],
)
def test_disk_memory_metric(model_fixture: tuple[Any, SmashConfig]) -> None:
    """Test the disk memory metric."""
    model, smash_config = model_fixture
    disk_memory_metric = DiskMemoryMetric()
    pruna_model = PrunaModel(model, smash_config=smash_config)
    move_to_device(pruna_model, "cuda")
    disk_memory_results = disk_memory_metric.compute(pruna_model, smash_config.test_dataloader())
    assert disk_memory_results.result > 0

@pytest.mark.cuda
@pytest.mark.parametrize(
    "model_fixture",
    [
        "sd_tiny_random",
    ],
    indirect=["model_fixture"],
)
def test_inference_memory_metric(model_fixture: tuple[Any, SmashConfig]) -> None:
    """Test the inference memory metric."""
    model, smash_config = model_fixture
    inference_memory_metric = InferenceMemoryMetric()
    pruna_model = PrunaModel(model, smash_config=smash_config)
    move_to_device(pruna_model, "cuda")
    inference_memory_results = inference_memory_metric.compute(pruna_model, smash_config.test_dataloader())
    assert inference_memory_results.result > 0

@pytest.mark.cuda
@pytest.mark.parametrize(
    "model_fixture",
    [
        "sd_tiny_random",
    ],
    indirect=["model_fixture"],
)
def test_training_memory_metric(model_fixture: tuple[Any, SmashConfig]) -> None:
    """Test the training memory metric."""
    model, smash_config = model_fixture
    training_memory_metric = TrainingMemoryMetric()
    pruna_model = PrunaModel(model, smash_config=smash_config)
    move_to_device(pruna_model, "cuda")
    training_memory_results = training_memory_metric.compute(pruna_model, smash_config.test_dataloader())
    assert training_memory_results.result > 0

@pytest.mark.cpu
@pytest.mark.parametrize(
    "model_fixture",
    [
        "sd_tiny_random",
    ],
    indirect=["model_fixture"],
)
def test_memory_metric_raises_when_device_is_not_cuda(model_fixture: tuple[Any, SmashConfig]):
    model, smash_config = model_fixture
    dmm = DiskMemoryMetric()
    pruna_model = PrunaModel(model, smash_config=smash_config)
    move_to_device(pruna_model, "cpu")
    with pytest.raises(ValueError):
        dmm.compute(pruna_model, smash_config.test_dataloader())
    imm = InferenceMemoryMetric()
    with pytest.raises(ValueError):
        imm.compute(pruna_model, smash_config.test_dataloader())
    tmm = TrainingMemoryMetric()
    with pytest.raises(ValueError):
        tmm.compute(pruna_model, smash_config.test_dataloader())
