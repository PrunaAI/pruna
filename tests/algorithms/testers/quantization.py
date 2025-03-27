import pytest

from pruna.algorithms.quantization.gptq_model import GPTQQuantizer
from pruna.algorithms.quantization.half import HalfQuantizer
from pruna.algorithms.quantization.hqq import HQQQuantizer
from pruna.algorithms.quantization.hqq_diffusers import HQQDiffusersQuantizer
from pruna.algorithms.quantization.huggingface_awq import AWQQuantizer
from pruna.algorithms.quantization.huggingface_diffusers_int8 import (
    DiffusersInt8Quantizer,
)
from pruna.algorithms.quantization.huggingface_llm_int8 import LLMInt8Quantizer
from pruna.algorithms.quantization.quanto import QuantoQuantizer
from pruna.algorithms.quantization.torch_dynamic import TorchDynamicQuantizer
from pruna.algorithms.quantization.torch_static import TorchStaticQuantizer

from .base_tester import AlgorithmTesterBase


class TestTorchStatic(AlgorithmTesterBase):
    """Test the torch static quantizer."""

    models = ["noref_mobilenet_v2"]
    reject_models = []
    allow_pickle_files = False
    algorithm_class = TorchStaticQuantizer


class TestTorchDynamic(AlgorithmTesterBase):
    """Test the torch dynamic quantizer."""

    models = ["mobilenet_v2"]
    reject_models = []
    allow_pickle_files = False
    algorithm_class = TorchDynamicQuantizer


class TestQuanto(AlgorithmTesterBase):
    """Test the Quanto quantizer."""

    models = ["opt_125m"]
    reject_models = ["dummy_lambda"]
    allow_pickle_files = False
    algorithm_class = QuantoQuantizer


class TestLLMint8(AlgorithmTesterBase):
    """Test the LLMint8 quantizer."""

    models = ["opt_125m"]
    reject_models = ["stable_diffusion_v1_4"]
    allow_pickle_files = False
    algorithm_class = LLMInt8Quantizer


class TestDiffusersInt8(AlgorithmTesterBase):
    """Test the DiffusersInt8 quantizer."""

    models = ["sana"]
    reject_models = ["opt_125m"]
    allow_pickle_files = False
    algorithm_class = DiffusersInt8Quantizer


class TestHQQ(AlgorithmTesterBase):
    """Test the HQQ quantizer."""

    models = ["llama_3_2_1b"]
    reject_models = ["stable_diffusion_v1_4"]
    allow_pickle_files = False
    algorithm_class = HQQQuantizer


class TestHQQDiffusers(AlgorithmTesterBase):
    """Test the HQQ quantizer."""

    models = ["sana"]
    reject_models = ["opt_125m"]
    allow_pickle_files = False
    algorithm_class = HQQDiffusersQuantizer


class TestHalf(AlgorithmTesterBase):
    """Test the half quantizer."""

    models = ["opt_125m"]
    reject_models = ["stable_diffusion_v1_4"]
    allow_pickle_files = False
    algorithm_class = HalfQuantizer


@pytest.mark.slow
class TestGPTQ(AlgorithmTesterBase):
    """Test the GPTQ quantizer."""

    models = ["opt_125m"]
    reject_models = ["stable_diffusion_v1_4"]
    allow_pickle_files = False
    algorithm_class = GPTQQuantizer


@pytest.mark.slow
class TestAWQ(AlgorithmTesterBase):
    """Test the AWQ quantizer."""

    models = ["opt_125m"]
    reject_models = ["stable_diffusion_v1_4"]
    allow_pickle_files = False
    algorithm_class = AWQQuantizer
