from pruna.algorithms.llama_cpp import LlamaCpp
from .base_tester import AlgorithmTesterBase


class TestLlamaCpp(AlgorithmTesterBase):
    """Test the LlamaCpp quantizer."""

    __test__ = False

    models = ["llama_3_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = LlamaCpp
    metrics = []

    def pre_smash_hook(self, model):
        import pytest
        pytest.importorskip("llama_cpp")

    def execute_smash(self, model, smash_config):
        """Execute the smash operation without device checking."""
        self.pre_smash_hook(model)
        from pruna.smash import smash
        smashed_model = smash(model, smash_config=smash_config)
        self.post_smash_hook(smashed_model)
        # Bypassed device checks because llama_cpp doesn't expose native PyTorch .parameters() for checking
        return smashed_model

    def execute_load(self):
        """Load the smashed model without device checking."""
        from pruna.engine.pruna_model import PrunaModel
        model = PrunaModel.from_pretrained(str(self._saving_path))
        assert isinstance(model, PrunaModel)
        self.post_load_hook(model)
        # Bypassed device checks because llama_cpp doesn't expose native PyTorch .parameters() for checking
        return model
