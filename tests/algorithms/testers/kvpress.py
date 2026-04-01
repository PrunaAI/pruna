from pruna import PrunaModel
from pruna.algorithms.kvpress import KVPress

from .base_tester import AlgorithmTesterBase


class TestKVPress(AlgorithmTesterBase):
    """Test the KVPress KV cache compression algorithm."""

    models = ["llama_3_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = KVPress
    metrics = ["perplexity"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Verify that the press was applied to the model."""
        assert hasattr(model, "_kvpress_press")
        assert hasattr(model, "_kvpress_original_generate")
