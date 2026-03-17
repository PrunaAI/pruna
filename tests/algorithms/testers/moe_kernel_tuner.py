from __future__ import annotations

import json
from pathlib import Path

from pruna import PrunaModel
from pruna.algorithms.moe_kernel_tuner import MoeKernelTuner

from .base_tester import AlgorithmTesterBase


class TestMoeKernelTuner(AlgorithmTesterBase):
    """Test the MoeKernelTuner."""

    models = ["qwen_moe_tiny_random"]
    reject_models = ["sd_tiny_random"]
    allow_pickle_files = False
    algorithm_class = MoeKernelTuner
    metrics = ["perplexity"]
    # for faster testing
    hyperparameters = {
        "moe_kernel_tuner_num_iters": 1,
        "moe_kernel_tuner_block_size_m_max": 4,
        "moe_kernel_tuner_block_size_n_max": 5,
        "moe_kernel_tuner_block_size_k_max": 6,
    }

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Assert the tuned config artifact was written to cache_dir."""
        artifact = Path(model.smash_config.cache_dir) / "moe_kernel_tuner.json"
        assert artifact.exists(), f"Expected artifact at {artifact} after smash"

    def _resolve_hf_cache_config_path(self) -> Path:
        """Read the saved artifact and compute the expected HF cache config path."""
        from pruna.algorithms.utils.moe_kernel_tuner import save_configs  # noqa: F401

        tuner = MoeKernelTuner()
        imported_packages = tuner.import_algorithm_packages()

        smash_config_path = self._saving_path / "smash_config.json"
        with open(smash_config_path) as f:
            smash_cfg = json.load(f)

        artifact_path = self._saving_path / "moe_kernel_tuner.json"
        with open(artifact_path) as f:
            artifact = json.load(f)

        dtype_str = imported_packages["_get_config_dtype_str"](
            artifact["dtype"] == "bfloat16" and __import__("torch").bfloat16 or __import__("torch").float16,
            use_int8_w8a16=artifact["use_int8_w8a16"],
            use_fp8_w8a8=artifact["use_fp8_w8a8"],
        )
        filename = imported_packages["FusedMoE"].get_config_file_name(
            artifact["num_experts"],
            artifact["shard_intermediate_size"] // 2,
            dtype_str,
            None,
        )

        hf_cache_base = smash_cfg.get("moe_kernel_tuner_path_to_huggingface_hub_cache", "~")
        return (
            Path(hf_cache_base)
            / ".cache/huggingface/hub/models--RedHatAI--moe/blobs/configs"
            / filename
        )

    def execute_load(self) -> PrunaModel:
        """Load the model after deleting the HF cache config to verify restoration."""
        artifact = self._saving_path / "moe_kernel_tuner.json"
        assert artifact.exists(), f"Expected artifact at {artifact} before load"

        hf_config_path = self._resolve_hf_cache_config_path()
        if hf_config_path.exists():
            hf_config_path.unlink()

        return super().execute_load()

    def post_load_hook(self, model: PrunaModel) -> None:
        """Assert the artifact loader restored the HF cache config file."""
        hf_config_path = self._resolve_hf_cache_config_path()
        assert hf_config_path.exists(), (
            f"Expected HF cache config at {hf_config_path} to be restored by the artifact loader"
        )
