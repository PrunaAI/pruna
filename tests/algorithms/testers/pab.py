from pruna import PrunaModel
from pruna.algorithms.pab import PAB

from .base_tester import AlgorithmTesterBase


class TestPAB(AlgorithmTesterBase):
    """Test the PAB algorithm."""

    models = ["flux_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = PAB
    metrics = ["psnr"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert model.transformer.is_cache_enabled
