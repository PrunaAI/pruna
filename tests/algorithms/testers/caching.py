from pruna import PrunaModel
from pruna.algorithms.caching.deepcache import DeepCacheCacher
from pruna.algorithms.caching.fastercache import FasterCacheCacher
from pruna.algorithms.caching.fora import FORACacher
from pruna.algorithms.caching.pab import PABCacher

from .base_tester import AlgorithmTesterBase


class TestDeepCache(AlgorithmTesterBase):
    """Test the deepcache algorithm."""

    models = ["sd_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = DeepCacheCacher
    metrics = ["ssim"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert hasattr(model, "deepcache_unet_helper")


class TestFORA(AlgorithmTesterBase):
    """Test the fora algorithm."""

    models = ["flux_tiny_random"]
    reject_models = ["opt_125m"]
    allow_pickle_files = False
    algorithm_class = FORACacher
    metrics = ["lpips", "throughput"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert hasattr(model, "cache_helper")


class TestFasterCache(AlgorithmTesterBase):
    """Test the fastercache algorithm."""

    models = ["flux_tiny_random"]
    reject_models = ["opt_125m"]
    allow_pickle_files = False
    algorithm_class = FasterCacheCacher
    metrics = ["pairwise_clip_score"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert model.transformer.is_cache_enabled


class TestPAB(AlgorithmTesterBase):
    """Test the PAB algorithm."""

    models = ["flux_tiny_random"]
    reject_models = ["opt_125m"]
    allow_pickle_files = False
    algorithm_class = PABCacher
    metrics = ["psnr"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert model.transformer.is_cache_enabled
