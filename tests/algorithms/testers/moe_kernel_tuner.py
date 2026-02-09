from pruna.algorithms.moe_kernel_tuner import MoeKernelTuner

from .base_tester import AlgorithmTesterBase


class TestMoeKernelTuner(AlgorithmTesterBase):
    """Test the MoeKernelTuner."""

    models = ["qwen3_next_moe_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = MoeKernelTuner
    metrics = ["perplexity"]
