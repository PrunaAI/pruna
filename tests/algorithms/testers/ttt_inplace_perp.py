from typing import Any

import pytest
import torch

from pruna import SmashConfig
from pruna.algorithms.perp import TextToTextInPlacePERP
from pruna.engine.pruna_model import PrunaModel
from pruna.engine.utils import get_nn_modules

from .base_tester import AlgorithmTesterBase
from .utils import restrict_recovery_time


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
class TestTTTInPlacePerp(AlgorithmTesterBase):
    """Test the TTT InPlace Perp recovery algorithm."""

    models = ["noref_opt_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = True
    algorithm_class = TextToTextInPlacePERP
    metrics = ["perplexity"]

    def prepare_smash_config(self, smash_config: SmashConfig, device: str) -> None:
        """Prepare the smash config for the test."""
        super().prepare_smash_config(smash_config, device)
        restrict_recovery_time(smash_config, self.algorithm_class.algorithm_name)

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Fast hook to verify algorithm application after smashing."""
        assert_no_nan_values(model)
