from pruna import PrunaModel
from pruna.algorithms.caching.deepcache import DeepCacheCacher
from pruna.algorithms.caching.fora import FORACacher

from .base_tester import AlgorithmTesterBase


class TestDeepCache(AlgorithmTesterBase):
    """Test the deepcache algorithm."""

    models = ["stable_diffusion_v1_4"]
    reject_models = ["opt_125m"]
    allow_pickle_files = False
    algorithm_class = DeepCacheCacher

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert hasattr(model, "deepcache_unet_helper")


class TestFORA(AlgorithmTesterBase):
    """Test the fora algorithm."""

    models = ["flux_tiny_random"]
    reject_models = ["opt_125m"]
    allow_pickle_files = False
    algorithm_class = FORACacher

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert hasattr(model, "cache_helper")
