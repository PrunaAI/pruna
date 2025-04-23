from pruna.algorithms.fusing.qkv_fusing import QKVFuser

from .base_tester import AlgorithmTesterBase


class TestQKVFusing(AlgorithmTesterBase):
    """Test the qkv fusing algorithm."""

    models = ["stable_diffusion_v1_4"]
    reject_models = ["opt_125m"]
    allow_pickle_files = False
    algorithm_class = QKVFuser
