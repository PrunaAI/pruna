import pytest

from pruna.algorithms.sage_attn import SageAttn

from .base_tester import AlgorithmTesterBase


@pytest.mark.high
class TestSageAttn(AlgorithmTesterBase):
    """Test the sage attention kernel."""

    models = ["flux_tiny", "wan_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = SageAttn
    metrics = ["latency"]