from pruna.algorithms.factorizing.qkv_diffusers import QKVDiffusers

from .base_tester import AlgorithmTesterBase


class TestQKVDiffusers(AlgorithmTesterBase):
    """Test the qkv factorizing algorithm."""

    models = ["flux_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = QKVDiffusers
    metrics = ["cmmd"]
