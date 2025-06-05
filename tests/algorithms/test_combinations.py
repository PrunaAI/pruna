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

    def __init__(self, config_dict: dict[str, Any], allow_pickle_files: bool, metric:str) -> None:
        super().__init__()
        self.config_dict = config_dict
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
        smash_config.load_dict(self.config_dict)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "model_fixture, config_dict, allow_pickle_files, metric",
    [
        ("sd_tiny_random", dict(cacher="deepcache", compiler="stable_fast"), False, 'cmmd'),
        ("mobilenet_v2", dict(pruner="torch_unstructured", quantizer="half"), True, 'latency'),
        ("whisper_tiny_random", dict(batcher="whisper_s2t", compiler="c_whisper"), False, 'latency'),
        ("sd_tiny_random", dict(quantizer="hqq_diffusers", compiler="torch_compile"), False, 'cmmd'),
        ("flux_tiny_random", dict(quantizer="hqq_diffusers", compiler="torch_compile"), False, 'cmmd'),
        ("sd_tiny_random", dict(quantizer="diffusers_int8", compiler="torch_compile", torch_compile_fullgraph=False), False, 'cmmd'),
        ("llama_3_tiny_random", dict(quantizer="gptq", compiler="torch_compile", torch_compile_fullgraph=False), True, 'perplexity'),
        ("flux_tiny_random", dict(cacher="pab", quantizer="hqq_diffusers"), False, 'cmmd'),
        ("flux_tiny_random", dict(cacher="pab", quantizer="diffusers_int8"), False, 'cmmd'),
        ("flux_tiny_random", dict(cacher="fastercache", quantizer="hqq_diffusers"), False, 'cmmd'),
        ("flux_tiny_random", dict(cacher="fastercache", quantizer="diffusers_int8"), False, 'cmmd'),
        ("flux_tiny_random", dict(cacher="fora", quantizer="hqq_diffusers"), False, 'cmmd'),
        ("flux_tiny_random", dict(cacher="fora", quantizer="diffusers_int8"), False, 'cmmd'),
        ("flux_tiny_random", dict(cacher="fora", compiler="torch_compile"), False, 'cmmd'),
        ("flux_tiny_random", dict(cacher="fora", compiler="stable_fast"), False, 'cmmd'),
        ("tiny_janus_pro", dict(quantizer="hqq", compiler="torch_compile"), False, 'cmmd'),
    ],
    indirect=["model_fixture"],
)
def test_full_integration_combo(
    config_dict: dict[str, Any], allow_pickle_files: bool, model_fixture: tuple[Any, SmashConfig], metric:str
) -> None:
    """Test the full integration of the algorithm."""
    algorithm_tester = CombinationsTester(config_dict, allow_pickle_files, metric)
    run_full_integration(algorithm_tester, device="cuda", model_fixture=model_fixture)
