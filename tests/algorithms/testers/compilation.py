from pruna import PrunaModel
from pruna.algorithms.compilation.c_translate import CGenerateCompiler
from pruna.algorithms.compilation.stable_fast import StableFastCompiler
from pruna.algorithms.compilation.torch_compile import TorchCompileCompiler

from .base_tester import AlgorithmTesterBase


class TestTorchCompile(AlgorithmTesterBase):
    """Test the torch_compile algorithm."""

    models = ["shufflenet"]
    reject_models = []
    allow_pickle_files = False
    algorithm_class = TorchCompileCompiler
    metrics = ["latency"]


class TestStableFast(AlgorithmTesterBase):
    """Test the stable_fast algorithm."""

    models = ["sd_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = StableFastCompiler
    metrics = ["cmmd"]


class TestCGenerate(AlgorithmTesterBase):
    """Test the c_generate algorithm."""

    models = ["opt_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = CGenerateCompiler
    metrics = ["latency"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert hasattr(model, "generator")
