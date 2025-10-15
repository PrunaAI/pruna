from pruna.algorithms.hqq import HQQ

from .base_tester import AlgorithmTesterBase


class TestHQQ(AlgorithmTesterBase):
    """Test the HQQ quantizer."""

    models = ["llama_3_tiny_random", "tiny_janus_pro"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = HQQ
    metrics = ["perplexity"]
