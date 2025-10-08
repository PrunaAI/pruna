import pytest

from pruna.algorithms.flash_attn3 import FlashAttn3

from .base_tester import AlgorithmTesterBase


@pytest.mark.high
class TestFlashAttn3(AlgorithmTesterBase):
    """Test the flash attention 3 kernel."""

    models = ["flux_tiny", "wan_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = FlashAttn3
    metrics = ["background_consistency"]
