from typing import Any

from pruna import PrunaModel
from pruna.algorithms.pruning.torch_structured import TorchStructuredPruner
from pruna.algorithms.pruning.torch_unstructured import TorchUnstructuredPruner

from .base_tester import AlgorithmTesterBase
from .utils import get_model_sparsity


class TestTorchUnstructured(AlgorithmTesterBase):
    """Test the torch unstructured pruner."""

    models = ["shufflenet"]
    reject_models = []
    hyperparameters = {"torch_unstructured_sparsity": 0.5}
    allow_pickle_files = True
    algorithm_class = TorchUnstructuredPruner
    metrics = ["latency"]

    def pre_smash_hook(self, model: Any) -> None:
        """Hook to modify the model before smashing."""
        self.original_sparsity = get_model_sparsity(model)

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        new_sparsity = get_model_sparsity(model)
        assert new_sparsity > self.original_sparsity


class TestTorchStructured(AlgorithmTesterBase):
    """Test the torch structured pruner."""

    models = ["resnet_18"]
    reject_models = []
    allow_pickle_files = True
    algorithm_class = TorchStructuredPruner
    metrics = ["total_macs", "total_params"]

    def pre_smash_hook(self, model: Any) -> None:
        """Hook to modify the model before smashing."""
        self.original_num_params = sum(p.numel() for p in model.parameters())

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        new_num_params = sum(p.numel() for p in model.parameters())
        assert new_num_params < self.original_num_params
