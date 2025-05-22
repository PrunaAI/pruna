from typing import Any

import pytest

from pruna import SmashConfig
from pruna.engine.pruna_model import PrunaModel
from pruna.evaluation.metrics.metric_model_architecture import TotalMACsMetric, TotalParamsMetric, ModelArchitectureMetric, TOTAL_MACS, TOTAL_PARAMS


@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("shufflenet", "cpu", marks=pytest.mark.cpu),
        pytest.param("sd_tiny_random", "cuda", marks=pytest.mark.cuda),
    ],
    indirect=["model_fixture"],
)
def test_total_macs_metric(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the total MACs metric."""
    model, smash_config = model_fixture
    macs_metric = TotalMACsMetric(device=device)
    pruna_model = PrunaModel(model, smash_config=smash_config)
    macs_results = macs_metric.compute(pruna_model, smash_config.test_dataloader())
    assert macs_results.result > 0


@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("noref_resnet_18", "cpu", marks=pytest.mark.cpu),
        pytest.param("stable_diffusion_v1_4", "cuda", marks=pytest.mark.cuda),
    ],
    indirect=["model_fixture"],
)
def test_total_params_metric(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the total parameters metric."""
    model, smash_config = model_fixture
    params_metric = TotalParamsMetric(device=device)
    pruna_model = PrunaModel(model, smash_config=smash_config)
    params_results = params_metric.compute(pruna_model, smash_config.test_dataloader())
    assert params_results.result > 0

@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("stable_diffusion_v1_4", "cuda", marks=pytest.mark.cuda),
    ],
    indirect=["model_fixture"],
)
def test_deprecated_model_architecture(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the deprecated model architecture metrics."""
    model, smash_config = model_fixture
    params_metric = ModelArchitectureMetric(device=device)
    pruna_model = PrunaModel(model, smash_config=smash_config)
    params_results = params_metric.compute(pruna_model, smash_config.test_dataloader())
    assert params_results[TOTAL_MACS] > 0
    assert params_results[TOTAL_PARAMS] > 0