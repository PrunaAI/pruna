import pytest

from pruna.algorithms.ipex_llm import IPEXLLM

from .base_tester import AlgorithmTesterBase


# this prevents the test from running on GitHub Actions, which does not reliably provide Intel CPUs
@pytest.mark.high_cpu
class TestIPEXLLM(AlgorithmTesterBase):
    """Test the IPEX LLM algorithm."""

    models = ["opt_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = IPEXLLM
    metrics = ["latency"]
