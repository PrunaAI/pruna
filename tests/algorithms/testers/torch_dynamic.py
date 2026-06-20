from pruna.algorithms.torch_dynamic import TorchDynamic

from .base_tester import AlgorithmTesterBase


class TestTorchDynamic(AlgorithmTesterBase):
    """Test the torch dynamic quantizer."""

    models = ["shufflenet", "smollm_135m"]
    reject_models = []
    allow_pickle_files = False
    algorithm_class = TorchDynamic
    metrics = ["latency"]
