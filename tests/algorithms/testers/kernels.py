from pruna.algorithms.kernels.flash_attn3 import FlashAttn3Kernel

from .base_tester import AlgorithmTesterBase


class TestFlashAttn3(AlgorithmTesterBase):
    """Test the flash attention 3 kernel."""

    models = ["flux_tiny", "wan_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = FlashAttn3Kernel
    metrics = ["latency"]
