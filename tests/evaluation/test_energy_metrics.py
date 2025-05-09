from typing import Any

import pytest

from pruna import PrunaModel, SmashConfig
from pruna.evaluation.metrics.metric_energy import EnergyMetric


@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("shufflenet", "cuda", marks=pytest.mark.cuda),
        pytest.param("sd_tiny_random", "cpu", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],
)
def test_energy_metrics(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the energy metrics."""
    model, smash_config = model_fixture
    smash_config.device = device
    metric = EnergyMetric(n_iterations=5, n_warmup_iterations=5, device=device)
    pruna_model = PrunaModel(model, smash_config=smash_config)
    results = metric.compute(pruna_model, smash_config.test_dataloader())
    for key, value in results.items():
        assert value > 0
