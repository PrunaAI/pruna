from pruna.algorithms.torch_compile import TorchCompile

from .base_tester import AlgorithmTesterBase


class TestTorchCompile(AlgorithmTesterBase):
    """Test the torch_compile algorithm."""

    models = ["shufflenet"]
    reject_models = []
    allow_pickle_files = False
    algorithm_class = TorchCompile
    metrics = ["latency"]
