from typing import Any

from pruna import PrunaModel
from pruna.algorithms.torch_structured import TorchStructured

from .base_tester import AlgorithmTesterBase


class TestTorchStructured(AlgorithmTesterBase):
    """Test the torch structured pruner."""

    models = ["resnet_18"]
    reject_models = []
    allow_pickle_files = True
    algorithm_class = TorchStructured
    metrics = ["total_macs"]

    def pre_smash_hook(self, model: Any) -> None:
        """Hook to modify the model before smashing."""
        self.original_num_params = sum(p.numel() for p in model.parameters())

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        new_num_params = sum(p.numel() for p in model.parameters())
        assert new_num_params < self.original_num_params
