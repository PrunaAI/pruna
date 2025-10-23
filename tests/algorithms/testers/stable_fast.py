from pruna.algorithms.stable_fast import StableFast

from .base_tester import AlgorithmTesterBase


class TestStableFast(AlgorithmTesterBase):
    """Test the stable_fast algorithm."""

    models = ["sd_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = StableFast
    metrics = ["cmmd"]
