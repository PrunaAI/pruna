import pytest

from pruna import PrunaModel
from pruna.algorithms.gptq_model import GPTQ

from .base_tester import AlgorithmTesterBase


@pytest.mark.slow
@pytest.mark.high
class TestGPTQ(AlgorithmTesterBase):
    """Test the GPTQ quantizer."""

    models = ["opt_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = GPTQ
    hyperparameters = {
        "gptq_weight_bits": 4,
        "gptq_group_size": 128,
    }
    metrics = ["perplexity"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert "GPTQ" in model.model.__class__.__name__
