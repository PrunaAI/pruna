import pytest
from transformers import AutomaticSpeechRecognitionPipeline

from pruna import PrunaModel
from pruna.algorithms.batching.ifw import IFWBatcher
from pruna.algorithms.batching.ws2t import WhisperS2TWrapper, WS2TBatcher

from .base_tester import AlgorithmTesterBase


@pytest.mark.skip(reason="This test / the importing of whisper_s2t is affecting other tests.")
@pytest.mark.slow
class TestWhisperS2T(AlgorithmTesterBase):
    """Test the WhisperS2T batcher."""

    models = ["whisper_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = WS2TBatcher
    metrics = ["latency"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert isinstance(model.model, WhisperS2TWrapper)


class TestIFW(AlgorithmTesterBase):
    """Test the IFW batcher."""

    models = ["whisper_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = IFWBatcher
    metrics = ["latency"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert isinstance(model.model, AutomaticSpeechRecognitionPipeline)
