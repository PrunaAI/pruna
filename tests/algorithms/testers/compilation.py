import pytest

from pruna import PrunaModel
from pruna.algorithms.compilation.c_translate import CGenerateCompiler
from pruna.algorithms.compilation.stable_fast import StableFastCompiler
from pruna.algorithms.compilation.torch_compile import TorchCompileCompiler

from .base_tester import AlgorithmTesterBase


class TestTorchCompile(AlgorithmTesterBase):
    """Test the torch_compile algorithm."""

    models = ["mobilenet_v2"]
    reject_models = []
    allow_pickle_files = False
    algorithm_class = TorchCompileCompiler


class TestStableFast(AlgorithmTesterBase):
    """Test the stable_fast algorithm."""

    models = ["stable_diffusion_v1_4"]
    reject_models = ["opt_125m"]
    allow_pickle_files = False
    algorithm_class = StableFastCompiler


class TestCGenerate(AlgorithmTesterBase):
    """Test the c_generate algorithm."""

    models = ["opt_125m"]
    reject_models = ["stable_diffusion_v1_4"]
    allow_pickle_files = False
    algorithm_class = CGenerateCompiler

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert hasattr(model, "generator")
