from transformers import AutomaticSpeechRecognitionPipeline

from pruna import PrunaModel
from pruna.algorithms.ifw import IFW

from .base_tester import AlgorithmTesterBase


class TestIFW(AlgorithmTesterBase):
    """Test the IFW batcher."""

    models = ["whisper_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = IFW
    metrics = ["latency"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert isinstance(model.model, AutomaticSpeechRecognitionPipeline)
