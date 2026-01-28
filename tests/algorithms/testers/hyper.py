from pruna.algorithms.hyper import Hyper

from .base_tester import AlgorithmTesterBase


class TestHyper(AlgorithmTesterBase):
    """Test the Hyper algorithm."""

    models = ["stable_diffusion_v1_4"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = Hyper
    metrics = ["psnr"]
