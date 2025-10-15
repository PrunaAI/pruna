from pruna import PrunaModel
from pruna.algorithms.fora import FORA

from .base_tester import AlgorithmTesterBase


class TestFORA(AlgorithmTesterBase):
    """Test the fora algorithm."""

    models = ["flux_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = FORA
    metrics = ["lpips", "throughput"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert hasattr(model, "cache_helper")
