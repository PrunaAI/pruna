from pruna.algorithms.red_noe import RedNOE

from .base_tester import AlgorithmTesterBase


class TestRedNOE(AlgorithmTesterBase):
    """Test the RedNOE algorithm."""

    models = ["qwen3_next_moe_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = RedNOE
    metrics = ["perplexity"]
