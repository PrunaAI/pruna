from pruna import PrunaModel
from pruna.algorithms.fastercache import FasterCache

from .base_tester import AlgorithmTesterBase


class TestFasterCache(AlgorithmTesterBase):
    """Test the fastercache algorithm."""

    models = ["flux_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = FasterCache
    metrics = ["pairwise_clip_score"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert model.transformer.is_cache_enabled
