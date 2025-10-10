from typing import Any

from pruna import PrunaModel
from pruna.algorithms.pruning.torch_structured import TorchStructuredPruner
from pruna.algorithms.pruning.torch_unstructured import TorchUnstructuredPruner
from pruna.algorithms.pruning.token_merging import TokenMergingPruner

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
    metrics = ["total_macs"]

    def pre_smash_hook(self, model: Any) -> None:
        """Hook to modify the model before smashing."""
        self.original_num_params = sum(p.numel() for p in model.parameters())

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        new_num_params = sum(p.numel() for p in model.parameters())
        assert new_num_params < self.original_num_params


class TestTokenMerging(AlgorithmTesterBase):
    """Test the token merging pruner."""

    models = ["vit_small"]
    reject_models = []
    hyperparameters = {"token_merging_reduction_ratio": 0.3}
    allow_pickle_files = True
    algorithm_class = TokenMergingPruner
    metrics = ["latency"]

    def pre_smash_hook(self, model: Any) -> None:
        """Hook to modify the model before smashing."""
        # Store original model info
        self.original_has_tome = hasattr(model, "_tome_r")

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        # Verify that token merging was applied
        assert hasattr(model, "_tome_r"), "Token merging was not applied to the model"
        assert model._tome_r > 0, "Token merging ratio should be greater than 0"
