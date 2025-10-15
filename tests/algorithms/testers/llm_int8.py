from pruna.algorithms.huggingface_llm_int8 import LLMInt8

from .base_tester import AlgorithmTesterBase


class TestLLMint8(AlgorithmTesterBase):
    """Test the LLMint8 quantizer."""

    models = ["opt_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = LLMInt8
    metrics = ["perplexity"]
