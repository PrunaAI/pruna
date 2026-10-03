import inspect

from pruna import PrunaModel
from pruna.algorithms.first_block_cache import FirstBlockCache

from .base_tester import AlgorithmTesterBase

# Tiny Hub checkpoints default to a large canvas (Flux and Qwen use 128 * vae scale, Z-Image uses 1024).
_TINY_INFERENCE_ARGS = {
    "num_inference_steps": 2,
    "height": 64,
    "width": 64,
    "max_sequence_length": 16,
}


class TestFirstBlockCache(AlgorithmTesterBase):
    """Test first-block cache on tiny diffusers checkpoints that emit images."""

    models = ["flux_tiny_random", "qwen_image_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = FirstBlockCache
    metrics = ["psnr"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert model.transformer.is_cache_enabled

    def post_load_hook(self, model: PrunaModel) -> None:
        """Keep CPU evaluation on a small canvas. The cache is reapplied on load."""
        assert model.transformer.is_cache_enabled
        signature = inspect.signature(type(model.model).__call__)
        model.inference_handler.model_args.update(
            {name: value for name, value in _TINY_INFERENCE_ARGS.items() if name in signature.parameters}
        )
