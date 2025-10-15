import pytest

from pruna.algorithms.llm_compressor import LLMCompressor

from .base_tester import AlgorithmTesterBase


@pytest.mark.slow
class TestLLMCompressor(AlgorithmTesterBase):
    """Test the LLM Compressor quantizer."""

    models = ["noref_tiny_llama"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = LLMCompressor
    metrics = ["perplexity"]
