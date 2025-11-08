from pruna.algorithms.quanto import Quanto

from .base_tester import AlgorithmTesterBase


class TestQuanto(AlgorithmTesterBase):
    """Test the Quanto quantizer."""

    models = ["opt_tiny_random"]
    reject_models = ["dummy_lambda"]
    allow_pickle_files = False
    algorithm_class = Quanto
    metrics = ["perplexity"]
