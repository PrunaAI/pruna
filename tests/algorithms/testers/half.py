from pruna.algorithms.half import Half

from .base_tester import AlgorithmTesterBase


class TestHalf(AlgorithmTesterBase):
    """Test the half quantizer."""

    models = ["opt_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = Half
    metrics = ["perplexity"]
