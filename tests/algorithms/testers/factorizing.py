from pruna.algorithms.factorizing.qkv_diffusers import QKVDiffusers

from .base_tester import AlgorithmTesterBase


class TestQKVDiffusers(AlgorithmTesterBase):
    """Test the qkv factorizing algorithm."""

    models = ["stable_diffusion_v1_4"]
    reject_models = ["opt_125m"]
    allow_pickle_files = False
    algorithm_class = QKVDiffusers
