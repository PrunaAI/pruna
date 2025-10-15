from pruna import PrunaModel
from pruna.algorithms.c_translate import CGenerate

from .base_tester import AlgorithmTesterBase


class TestCGenerate(AlgorithmTesterBase):
    """Test the c_generate algorithm."""

    models = ["opt_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = CGenerate
    metrics = ["latency"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert hasattr(model, "generator")
