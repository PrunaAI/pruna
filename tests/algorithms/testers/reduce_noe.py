from pruna.algorithms.reduce_noe import ReduceNOE

from .base_tester import AlgorithmTesterBase


class TestReduceNOE(AlgorithmTesterBase):
    """Test the ReduceNOE algorithm."""

    models = ["qwen3_next_moe_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = ReduceNOE
    metrics = ["perplexity"]
