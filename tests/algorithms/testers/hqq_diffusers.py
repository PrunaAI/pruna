from pruna.algorithms.hqq_diffusers import HQQDiffusers

from .base_tester import AlgorithmTesterBase


class TestHQQDiffusers(AlgorithmTesterBase):
    """Test the HQQ quantizer."""

    models = ["flux_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = HQQDiffusers
    metrics = ["cmmd"]
