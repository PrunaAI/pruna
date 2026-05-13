import pytest

from pruna.algorithms.upscale import RealESRGAN

from .base_tester import AlgorithmTesterBase


@pytest.mark.requires_upscale
class TestUpscale(AlgorithmTesterBase):
    """Test the Upscale algorithm."""

    models = ["sd_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = RealESRGAN
    metrics = ["cmmd"]

    @classmethod
    def compatible_devices(cls) -> list[str]:
        """Exclude CPU (too slow)."""
        return [d for d in super().compatible_devices() if d != "cpu"]
