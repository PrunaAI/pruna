from pruna.algorithms.huggingface_diffusers_int8 import DiffusersInt8

from .base_tester import AlgorithmTesterBase


class TestDiffusersInt8(AlgorithmTesterBase):
    """Test the DiffusersInt8 quantizer."""

    models = ["sana_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = DiffusersInt8
    metrics = ["cmmd"]
