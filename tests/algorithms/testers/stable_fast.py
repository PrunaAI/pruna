from pruna.algorithms.stable_fast import StableFast

from .base_tester import AlgorithmTesterBase


class TestStableFast(AlgorithmTesterBase):
    """Test the stable_fast algorithm."""

    models = ["stable_diffusion_v1_4"]  # sd_tiny_random is too small: no backend
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = StableFast
    metrics = ["cmmd"]
