from typing import Any

import pytest

from pruna import SmashConfig
from pruna.engine.pruna_model import PrunaModel
from pruna.evaluation.metrics.metric_elapsed_time import ElapsedTimeMetric


@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("sd_tiny_random", "cuda", marks=pytest.mark.cuda),
        pytest.param("shufflenet", "cpu", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],
)
def test_inference_time_metrics(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the inference time metrics."""
    model, smash_config = model_fixture
    metric = ElapsedTimeMetric(n_iterations=5, n_warmup_iterations=5, device=device)
    pruna_model = PrunaModel(model, smash_config=smash_config)
    results = metric.compute(pruna_model, smash_config.test_dataloader())
    assert results["inference_elapsed_time_ms_@1"] > 0
    assert results["inference_latency_ms_@1"] > 0
    assert results["inference_throughput_batches_per_ms_@1"] > 0
