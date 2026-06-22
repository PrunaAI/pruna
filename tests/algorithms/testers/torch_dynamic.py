from typing import Any

from pruna.algorithms.torch_dynamic import TorchDynamic
from pruna.engine.utils import get_device_type

from .base_tester import AlgorithmTesterBase


class TestTorchDynamic(AlgorithmTesterBase):
    """Test the torch dynamic quantizer."""

    models = ["shufflenet", "smollm_135m"]
    reject_models = []
    allow_pickle_files = False
    algorithm_class = TorchDynamic
    metrics = ["latency"]

    def pre_smash_hook(self, model: Any) -> None:
        """Ensure CPU dynamic quantized linear layers receive float32 activations."""
        if get_device_type(model) == "cpu":
            model.float()
