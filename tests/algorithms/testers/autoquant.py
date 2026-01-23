import pytest

from pruna.algorithms.torchao_autoquant import Autoquant

from .base_tester import AlgorithmTesterBase


# This is classified as high because autoquant requires INT8 tensor cores which require GPUs with
# Turing architecture and above
@pytest.mark.high
class TestAutoquant(AlgorithmTesterBase):
    """Test the Autoquant quantizer."""

    models = ["flux_tiny_random"]
    reject_models = ["dummy_lambda"]
    allow_pickle_files = False
    algorithm_class = Autoquant
    metrics = ["cmmd"]
