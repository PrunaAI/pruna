from typing import Any

import pytest

from pruna import PrunaModel, SmashConfig
from pruna.evaluation.metrics.metric_energy import EnergyConsumedMetric, CO2EmissionsMetric

@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("shufflenet", "cuda", marks=pytest.mark.cuda),
        pytest.param("sd_tiny_random", "cpu", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],
)
def test_energy_consumed_metric(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the energy consumed metric."""
    model, smash_config = model_fixture
    metric = EnergyConsumedMetric(n_iterations=5, n_warmup_iterations=5, device=device)
    pruna_model = PrunaModel(model, smash_config=smash_config)
    results = metric.compute(pruna_model, smash_config.test_dataloader())
    assert results.result >= 0  # Assuming energy consumption should be non-negative

@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("flux_tiny_random", "cuda", marks=pytest.mark.cuda),
        pytest.param("resnet_18", "cpu", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],
)
def test_co2_emissions_metric(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the CO2 emissions metric."""
    model, smash_config = model_fixture
    metric = CO2EmissionsMetric(n_iterations=5, n_warmup_iterations=5, device=device)
    pruna_model = PrunaModel(model, smash_config=smash_config)
    results = metric.compute(pruna_model, smash_config.test_dataloader())
    assert results.result >= 0  # Assuming CO2 emissions should be non-negative
