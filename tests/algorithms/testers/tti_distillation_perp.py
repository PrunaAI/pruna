from typing import Any

import pytest
import torch

from pruna import SmashConfig
from pruna.algorithms.distillation_perp import TextToImagePERPDistillation
from pruna.engine.pruna_model import PrunaModel
from pruna.engine.utils import get_nn_modules

from .base_tester import AlgorithmTesterBase
from .utils import replace_datamodule_with_distillation_datamodule, restrict_recovery_time


def assert_no_nan_values(module: Any) -> None:
    """Check for NaN values in the module or its components.

    Parameters
    ----------
    module : Any
        The module to check.
    """
    for nn_module in get_nn_modules(module).values():
        for name, param in nn_module.named_parameters():
            assert not torch.isnan(param).any(), f"NaN values found in {name}"


@pytest.mark.slow
class TestTTIDistillationPerp(AlgorithmTesterBase):
    """Test the TTI Distillation Perp recovery algorithm."""

    models = ["flux_tiny_random", "sd_tiny_random", "sana_tiny_random"]
    reject_models = ["opt_tiny_random"]
    metrics = ["cmmd"]
    allow_pickle_files = True
    algorithm_class = TextToImagePERPDistillation

    def prepare_smash_config(self, smash_config: SmashConfig, device: str) -> None:
        """Prepare the smash config for the test."""
        super().prepare_smash_config(smash_config, device)
        restrict_recovery_time(smash_config, self.algorithm_class.algorithm_name)

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Fast hook to verify algorithm application after smashing."""
        assert_no_nan_values(model)

    def execute_smash(self, model: Any, smash_config: SmashConfig) -> Any:
        """Execute the smash."""
        if any("distillation" in algorithm for algorithm in smash_config.get_active_algorithms()):
            self.replaced_datamodule = smash_config.data
            replace_datamodule_with_distillation_datamodule(smash_config, model)
        smashed_model = super().execute_smash(model, smash_config)
        if any("distillation" in algorithm for algorithm in smash_config.get_active_algorithms()):
            smash_config.add_data(self.replaced_datamodule)
            self.replaced_datamodule = None
        return smashed_model
