from pruna.algorithms.torchao import Torchao

from .base_tester import AlgorithmTesterBase


class TestTorchao(AlgorithmTesterBase):
    """Test the torchao quantizer."""

    models = ["flux_tiny_random", "sd_tiny_random"]
    reject_models = ["dummy_lambda"]
    allow_pickle_files = False
    algorithm_class = Torchao
    metrics = ["cmmd"]
