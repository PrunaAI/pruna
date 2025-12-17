from pruna.algorithms.denoise import Img2ImgDenoise

from .base_tester import AlgorithmTesterBase


class TestDenoise(AlgorithmTesterBase):
    """Test the Denoise algorithm."""

    models = ["sd_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = Img2ImgDenoise
    metrics = ["lpips"]
