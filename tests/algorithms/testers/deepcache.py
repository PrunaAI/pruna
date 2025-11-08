from pruna import PrunaModel
from pruna.algorithms.deepcache import DeepCache

from .base_tester import AlgorithmTesterBase


class TestDeepCache(AlgorithmTesterBase):
    """Test the deepcache algorithm."""

    models = ["sd_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = DeepCache
    metrics = ["ssim"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert hasattr(model, "deepcache_unet_helper")
