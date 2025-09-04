import pytest

from pruna import PrunaModel
from pruna.algorithms.quantization.gptq_model import GPTQQuantizer
from pruna.algorithms.quantization.half import HalfQuantizer
from pruna.algorithms.quantization.hqq import HQQQuantizer
from pruna.algorithms.quantization.hqq_diffusers import HQQDiffusersQuantizer
from pruna.algorithms.quantization.huggingface_diffusers_int8 import (
    DiffusersInt8Quantizer,
)
from pruna.algorithms.quantization.huggingface_llm_int8 import LLMInt8Quantizer
from pruna.algorithms.quantization.llm_compressor import LLMCompressorQuantizer
from pruna.algorithms.quantization.quanto import QuantoQuantizer
from pruna.algorithms.quantization.torch_dynamic import TorchDynamicQuantizer
from pruna.algorithms.quantization.torchao import TorchaoQuantizer

from .base_tester import AlgorithmTesterBase


class TestTorchDynamic(AlgorithmTesterBase):
    """Test the torch dynamic quantizer."""

    models = ["shufflenet"]
    reject_models = []
    allow_pickle_files = False
    algorithm_class = TorchDynamicQuantizer
    metrics = ["latency"]


class TestQuanto(AlgorithmTesterBase):
    """Test the Quanto quantizer."""

    models = ["opt_tiny_random"]
    reject_models = ["dummy_lambda"]
    allow_pickle_files = False
    algorithm_class = QuantoQuantizer
    metrics = ["perplexity"]


class TestLLMint8(AlgorithmTesterBase):
    """Test the LLMint8 quantizer."""

    models = ["opt_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = LLMInt8Quantizer
    metrics = ["perplexity"]


class TestDiffusersInt8(AlgorithmTesterBase):
    """Test the DiffusersInt8 quantizer."""

    models = ["sana_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = DiffusersInt8Quantizer
    metrics = ["cmmd"]


class TestHQQ(AlgorithmTesterBase):
    """Test the HQQ quantizer."""

    models = ["llama_3_tiny_random", "tiny_janus_pro"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = HQQQuantizer
    metrics = ["perplexity"]


class TestHQQDiffusers(AlgorithmTesterBase):
    """Test the HQQ quantizer."""

    models = ["flux_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = HQQDiffusersQuantizer
    metrics = ["cmmd"]


class TestHalf(AlgorithmTesterBase):
    """Test the half quantizer."""

    models = ["opt_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = HalfQuantizer
    metrics = ["perplexity"]


class TestTorchao(AlgorithmTesterBase):
    """Test the torchao quantizer."""

    models = ["flux_tiny_random", "sd_tiny_random"]
    reject_models = ["dummy_lambda"]
    allow_pickle_files = False
    algorithm_class = TorchaoQuantizer
    metrics = ["cmmd"]


@pytest.mark.slow
@pytest.mark.high
class TestGPTQ(AlgorithmTesterBase):
    """Test the GPTQ quantizer."""

    models = ["opt_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = GPTQQuantizer
    hyperparameters = {
        "gptq_weight_bits": 4,
        "gptq_group_size": 128,
    }
    metrics = ["perplexity"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert "GPTQ" in model.model.__class__.__name__


@pytest.mark.high
@pytest.mark.slow
class TestLLMCompressor(AlgorithmTesterBase):
    """Test the LLM Compressor quantizer."""

    models = ["noref_tiny_llama"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = LLMCompressorQuantizer
    metrics = ["perplexity"]
