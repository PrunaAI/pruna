from typing import Any

from pruna import PrunaModel
from pruna.algorithms.torch_unstructured import TorchUnstructured

from .base_tester import AlgorithmTesterBase
from .utils import get_model_sparsity


class TestTorchUnstructured(AlgorithmTesterBase):
    """Test the torch unstructured pruner."""

    models = ["shufflenet"]
    reject_models = []
    hyperparameters = {"torch_unstructured_sparsity": 0.5}
    allow_pickle_files = True
    algorithm_class = TorchUnstructured
    metrics = ["latency"]

    def pre_smash_hook(self, model: Any) -> None:
        """Hook to modify the model before smashing."""
        self.original_sparsity = get_model_sparsity(model)

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        new_sparsity = get_model_sparsity(model)
        assert new_sparsity > self.original_sparsity
