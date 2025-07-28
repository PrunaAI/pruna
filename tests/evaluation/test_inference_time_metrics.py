from typing import Any

import pytest
import torch

from ..common import construct_device_map_manually
from pruna import SmashConfig
from pruna.engine.pruna_model import PrunaModel
from pruna.engine.utils import move_to_device, get_device, get_device_map
from pruna.evaluation.metrics.metric_elapsed_time import LatencyMetric, ThroughputMetric, TotalTimeMetric
from pruna.evaluation.evaluation_agent import EvaluationAgent

@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("flux_tiny_random", "cuda", marks=pytest.mark.cuda),
        pytest.param("shufflenet", "cpu", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],
)
def test_latency_metric(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the latency metric."""
    model, smash_config = model_fixture
    metric = LatencyMetric(n_iterations=5, n_warmup_iterations=5, device=device)
    pruna_model = PrunaModel(model, smash_config=smash_config)
    move_to_device(pruna_model, device)
    results = metric.compute(pruna_model, smash_config.test_dataloader())
    assert results.result > 0  # Assuming latency should be positive


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs ≥2 GPUs to build a split model")
@pytest.mark.parametrize(
    "model_fixture",
    [
        "sd_tiny_random",
    ],
    indirect=["model_fixture"],
)
def test_latency_metric_distributed(model_fixture: tuple[Any, SmashConfig]):
    """Test the latency metric."""
    model, smash_config = model_fixture
    device_map = construct_device_map_manually(model)
    move_to_device(model, "accelerate", device_map=device_map)

    metric = LatencyMetric(n_iterations=5, n_warmup_iterations=5, device="accelerate")
    pruna_model = PrunaModel(model, smash_config=smash_config)
    results = metric.compute(pruna_model, smash_config.test_dataloader())

    assert get_device(model) == "accelerate"
    assert get_device_map(model) == device_map
    assert results.result > 0  # Assuming latency should be positive

@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs ≥2 GPUs to build a split model")
@pytest.mark.parametrize(
    "model_fixture",
    [
        "sd_tiny_random",
    ],
    indirect=["model_fixture"],
)
def test_latency_metric_distributed_agent(model_fixture: tuple[Any, SmashConfig]):
    """Test the latency metric."""
    model, smash_config = model_fixture
    device_map = construct_device_map_manually(model)
    move_to_device(model, "accelerate", device_map=device_map)

    request = [LATENCY]

    eval_agent = EvaluationAgent(request=request, datamodule=smash_config.data, device="accelerate")
    eval_agent.task.metrics[0].n_iterations = 5
    eval_agent.task.metrics[0].n_warmup_iterations = 5
    results =eval_agent.evaluate(model)

    assert get_device(model) == "accelerate"
    assert get_device_map(model) == device_map
    assert results[0].result > 0

@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("flux_tiny_random", "cuda", marks=pytest.mark.cuda),
        pytest.param("resnet_18", "cpu", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],
)
def test_throughput_metric(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the throughput metric."""
    model, smash_config = model_fixture
    metric = ThroughputMetric(n_iterations=5, n_warmup_iterations=5, device=device)
    pruna_model = PrunaModel(model, smash_config=smash_config)
    move_to_device(pruna_model, device)
    results = metric.compute(pruna_model, smash_config.test_dataloader())
    assert results.result > 0  # Assuming throughput should be positive

@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("flux_tiny_random", "cuda", marks=pytest.mark.cuda),
        pytest.param("resnet_18", "cpu", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],
)
def test_total_time_metric(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the total time metric."""
    model, smash_config = model_fixture
    metric = TotalTimeMetric(n_iterations=5, n_warmup_iterations=5, device=device)
    pruna_model = PrunaModel(model, smash_config=smash_config)
    move_to_device(pruna_model, device)
    results = metric.compute(pruna_model, smash_config.test_dataloader())
    assert results.result > 0  # Assuming total time should be positive
