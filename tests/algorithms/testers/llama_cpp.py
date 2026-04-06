from pruna.algorithms.llama_cpp import LlamaCpp
from .base_tester import AlgorithmTesterBase


class TestLlamaCpp(AlgorithmTesterBase):
    """Test the LlamaCpp quantizer."""

    __test__ = True

    models = ["llama_3_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = LlamaCpp
    metrics = []

    def pre_smash_hook(self, model):
        import pytest
        pytest.importorskip("llama_cpp")
