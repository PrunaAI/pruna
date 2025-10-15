from typing import Any

import pytest

from pruna import SmashConfig
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from ..common import run_full_integration
from .testers.base_tester import AlgorithmTesterBase


class CombinationsTester(AlgorithmTesterBase):
    """Test the combo tester."""

    def __init__(self, algorithms: list[str], allow_pickle_files: bool, metric:str) -> None:
        super().__init__()
        self.algorithms = algorithms
        self._allow_pickle_files = allow_pickle_files
        self._metrics = [metric]

    @property
    def allow_pickle_files(self) -> bool:
        """Allow pickle files."""
        return self._allow_pickle_files

    def get_metrics(self, device: str) -> list[BaseMetric | StatefulMetric]:
        """Get the metrics."""
        metrics = self._metrics
        return super().get_metric_instances(metrics, device)

    def compatible_devices(self) -> list[str]:
        """Return the compatible devices for the test."""
        return ["cuda"]

    def prepare_smash_config(self, smash_config: SmashConfig, device: str) -> None:
        """Prepare the smash config for the test."""
        smash_config["device"] = device
        smash_config.add(self.algorithms)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "model_fixture, algorithms, allow_pickle_files, metric",
    [
        ("sd_tiny_random", ["deepcache", "stable_fast"], False, 'cmmd'),
        ("mobilenet_v2", ["torch_unstructured", "half"], True, 'latency'),
        ("sd_tiny_random", ["hqq_diffusers", "torch_compile"], False, 'cmmd'),
        ("flux_tiny_random", ["hqq_diffusers", "torch_compile"], False, 'cmmd'),
        ("sd_tiny_random", ["diffusers_int8", "torch_compile"], False, 'cmmd'),
        ("tiny_llama", ["gptq", "torch_compile"], True, 'perplexity'),
        ("llama_3_tiny_random_as_pipeline", ["llm_int8", "torch_compile"], True, 'perplexity'),
        ("flux_tiny_random", ["pab", "hqq_diffusers"], False, 'cmmd'),
        ("flux_tiny_random", ["pab", "diffusers_int8"], False, 'cmmd'),
        ("flux_tiny_random", ["fastercache", "hqq_diffusers"], False, 'cmmd'),
        ("flux_tiny_random", ["fastercache", "diffusers_int8"], False, 'cmmd'),
        ("flux_tiny_random", ["fora", "hqq_diffusers"], False, 'cmmd'),
        ("flux_tiny_random", ["fora", "diffusers_int8"], False, 'cmmd'),
        ("flux_tiny_random", ["fora", "torch_compile"], False, 'cmmd'),
        ("flux_tiny_random", ["fora", "stable_fast"], False, 'cmmd'),
        ("tiny_janus", ["hqq", "torch_compile"], False, 'cmmd'),
        pytest.param("flux_tiny", ["fora", "flash_attn3", "torch_compile"], False, 'cmmd', marks=pytest.mark.high),
    ],
    indirect=["model_fixture"],
)
def test_full_integration_combo(
    algorithms: list[str], allow_pickle_files: bool, model_fixture: tuple[Any, SmashConfig], metric:str
) -> None:
    """Test the full integration of the algorithm."""
    algorithm_tester = CombinationsTester(algorithms, allow_pickle_files, metric)
    run_full_integration(algorithm_tester, device="cuda", model_fixture=model_fixture)
