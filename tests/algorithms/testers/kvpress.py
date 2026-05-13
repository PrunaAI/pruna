import pytest

from pruna import PrunaModel
from pruna.algorithms.kvpress import KVPress

from .base_tester import AlgorithmTesterBase


@pytest.mark.requires_kvpress
class TestKVPress(AlgorithmTesterBase):
    """Test the KVPress KV cache compression algorithm with default settings."""

    models = ["llama_3_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = KVPress
    metrics = ["perplexity"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Verify that the press was applied to the model."""
        assert hasattr(model, "_kvpress_press")
        assert hasattr(model, "_kvpress_original_generate")


@pytest.mark.requires_kvpress
class TestKVPressSnapKV(AlgorithmTesterBase):
    """Test the KVPress algorithm with SnapKV and custom press_kwargs."""

    models = ["llama_3_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = KVPress
    metrics = ["perplexity"]
    hyperparameters = {
        "kvpress_press_type": "SnapKVPress",
        "kvpress_compression_ratio": 0.3,
        "kvpress_press_kwargs": {"window_size": 32, "kernel_size": 3},
    }

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Verify that SnapKV press was applied with correct parameters."""
        assert hasattr(model, "_kvpress_press")
        press = model._kvpress_press
        assert type(press).__name__ == "SnapKVPress"
        assert press.window_size == 32
        assert press.kernel_size == 3
