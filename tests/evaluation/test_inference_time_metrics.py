from typing import Any

import pytest

from pruna import SmashConfig
from pruna.engine.pruna_model import PrunaModel
from pruna.evaluation.metrics.metric_elapsed_time import LatencyMetric, ThroughputMetric, TotalTimeMetric

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
    results = metric.compute(pruna_model, smash_config.test_dataloader())
    assert results.result > 0  # Assuming latency should be positive

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
    results = metric.compute(pruna_model, smash_config.test_dataloader())
    assert results.result > 0  # Assuming total time should be positive
