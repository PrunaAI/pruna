from pruna import PrunaModel
from pruna.algorithms.hqq import HQQ

from .base_tester import AlgorithmTesterBase


class TestHQQ(AlgorithmTesterBase):
    """Test the HQQ quantizer."""

    models = ["llama_3_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = HQQ
    metrics = ["perplexity"]


class TestHQQJanus(AlgorithmTesterBase):
    """Test the HQQ quantizer."""

    models = ["tiny_janus"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = HQQ
    metrics = ["cmmd"]
    hyperparameters = {"hqq_compute_dtype": "torch.bfloat16"}

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        model.inference_handler.model_args["generation_mode"] = "image"
        model.inference_handler.model_args["do_sample"] = True
        model.inference_handler.model_args["use_cache"] = True
