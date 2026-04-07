import pytest

from pruna.algorithms.ipex_llm import IPEXLLM

from .base_tester import AlgorithmTesterBase


@pytest.mark.high_cpu
@pytest.mark.requires_intel
class TestIPEXLLM(AlgorithmTesterBase):
    """Test the IPEX LLM algorithm."""

    models = ["opt_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = IPEXLLM
    metrics = ["latency"]
