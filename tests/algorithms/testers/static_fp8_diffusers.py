from typing import Any

import pytest

from pruna.algorithms.static_fp8_diffusers import StaticFp8Diffusers
from pruna.algorithms.static_fp8_diffusers.utils import StaticFp8Linear
from pruna.engine.load_artifacts import (
    STATIC_FP8_DIFFUSERS_ARTIFACTS_FILENAME,
    STATIC_FP8_DIFFUSERS_ARTIFACTS_FUNCTION_NAME,
)
from pruna.engine.pruna_model import PrunaModel

from .base_tester import AlgorithmTesterBase


@pytest.mark.high_gpu  # requires GPUs with compute capabilities >= 9.0 (e.g. H100) or >= 8.9 (e.g. L40S)
@pytest.mark.slow
class TestStaticFp8Diffusers(AlgorithmTesterBase):
    """Test static FP8 quantization for diffusers."""

    models = ["flux_tiny_random"]
    reject_models = ["llama_3_tiny_random"]
    allow_pickle_files = False
    algorithm_class = StaticFp8Diffusers
    metrics = ["ssim"]
    artifacts_filename = STATIC_FP8_DIFFUSERS_ARTIFACTS_FILENAME
    artifacts_function_name = STATIC_FP8_DIFFUSERS_ARTIFACTS_FUNCTION_NAME
    hyperparameters = {
        "static_fp8_diffusers_calibration_batches": 1,
    }

    @classmethod
    def compatible_devices(cls) -> list[str]:
        """CPU cannot run torch._scaled_mm fp8 matmul used by this algorithm."""
        return [d for d in super().compatible_devices() if d != "cpu"]

    @staticmethod
    def _fp8_linear_layers(model: PrunaModel, linear_cls: type) -> list[Any]:
        """Collect all submodules of ``linear_cls`` across pipeline roots."""
        return [
            submodule
            for module in model.get_nn_modules().values()
            for submodule in module.modules()
            if isinstance(submodule, linear_cls)
        ]

    def _assert_calibrated_linears(self, model: PrunaModel) -> None:
        """Assert the model contains calibrated StaticFp8Linear layers."""
        layers = self._fp8_linear_layers(model, StaticFp8Linear)
        assert layers, "Expected StaticFp8Linear layers"
        assert all(layer.input_initialized for layer in layers)

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Assert calibrated StaticFp8Linear layers after smash."""
        self._assert_calibrated_linears(model)

    def post_load_hook(self, model: PrunaModel) -> None:
        """Assert calibrated layers and that the artifact loader is registered."""
        super().post_load_hook(model)
        self._assert_calibrated_linears(model)
        assert self.artifacts_function_name in model.smash_config.load_artifacts_fns

    def execute_save(self, smashed_model: PrunaModel) -> None:
        """Save the smashed model and assert the calibration artifact was written."""
        super().execute_save(smashed_model)
        artifact_path = self._saving_path / self.artifacts_filename
        assert artifact_path.exists(), f"Expected artifact at {artifact_path} after save"
        assert artifact_path.stat().st_size > 0
