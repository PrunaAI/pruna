from pruna.algorithms.llama_cpp import LlamaCpp
from .base_tester import AlgorithmTesterBase


class TestLlamaCpp(AlgorithmTesterBase):
    """Test the LlamaCpp quantizer."""

    models = ["llama_3_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = LlamaCpp
    metrics = ["perplexity"]
