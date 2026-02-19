from pruna.algorithms.x_fast import XFast

from .base_tester import AlgorithmTesterBase


class TestXFast(AlgorithmTesterBase):
    """Test the X-Fast algorithm."""

    models = ["opt_tiny_random"]
    reject_models = ["dummy_lambda"]
    allow_pickle_files = False
    algorithm_class = XFast
    metrics = ["latency"]
