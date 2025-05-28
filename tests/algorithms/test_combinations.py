from typing import Any

import pytest

from pruna import SmashConfig

from ..common import run_full_integration
from .testers.base_tester import AlgorithmTesterBase


class CombinationsTester(AlgorithmTesterBase):
    """Test the combo tester."""

    def __init__(self, config_dict: dict[str, Any], allow_pickle_files: bool) -> None:
        super().__init__()
        self.config_dict = config_dict
        self._allow_pickle_files = allow_pickle_files

    @property
    def allow_pickle_files(self) -> bool:
        """Allow pickle files."""
        return self._allow_pickle_files

    def compatible_devices(self) -> list[str]:
        """Return the compatible devices for the test."""
        return ["cuda"]

    def prepare_smash_config(self, smash_config: SmashConfig, device: str) -> None:
        """Prepare the smash config for the test."""
        smash_config["device"] = device
        smash_config.load_dict(self.config_dict)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "model_fixture, config_dict, allow_pickle_files",
    [
        ("sd_tiny_random", dict(cacher="deepcache", compiler="stable_fast"), False),
        ("mobilenet_v2", dict(pruner="torch_unstructured", quantizer="half"), True),
        ("whisper_tiny_random", dict(batcher="whisper_s2t", compiler="c_whisper"), False),
        ("sd_tiny_random", dict(quantizer="hqq_diffusers", compiler="torch_compile"), False),
        ("flux_tiny_random", dict(quantizer="hqq_diffusers", compiler="torch_compile"), False),
        ("sd_tiny_random", dict(quantizer="diffusers_int8", compiler="torch_compile", torch_compile_fullgraph=False), False),
        ("llama_3_2_1b", dict(quantizer="gptq", compiler="torch_compile"), True),
        ("llama_3_tiny_random", dict(quantizer="llm_int8", compiler="torch_compile", torch_compile_fullgraph=False), True),
        ("flux_tiny_random", dict(cacher="pab", quantizer="hqq_diffusers"), False),
        ("flux_tiny_random", dict(cacher="pab", quantizer="diffusers_int8"), False),
        ("flux_tiny_random", dict(cacher="fastercache", quantizer="hqq_diffusers"), False),
        ("flux_tiny_random", dict(cacher="fastercache", quantizer="diffusers_int8"), False),
        ("flux_tiny_random", dict(cacher="fora", quantizer="hqq_diffusers"), False),
        ("flux_tiny_random", dict(cacher="fora", quantizer="diffusers_int8"), False),
        ("flux_tiny_random", dict(cacher="fora", compiler="torch_compile"), False),
        ("flux_tiny_random", dict(cacher="fora", compiler="stable_fast"), False),
        ("tiny_janus_pro", dict(quantizer="hqq", compiler="torch_compile"), False),
    ],
    indirect=["model_fixture"],
)
def test_full_integration_combo(
    config_dict: dict[str, Any], allow_pickle_files: bool, model_fixture: tuple[Any, SmashConfig]
) -> None:
    """Test the full integration of the algorithm."""
    algorithm_tester = CombinationsTester(config_dict, allow_pickle_files)
    run_full_integration(algorithm_tester, device="cuda", model_fixture=model_fixture)
