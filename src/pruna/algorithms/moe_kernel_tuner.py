# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any

import torch
from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.hyperparameters import UnconstrainedHyperparameter
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_moe_lm, is_transformers_pipeline_with_moe_lm
from pruna.logging.logger import pruna_logger


class MoeKernelTuner(PrunaAlgorithmBase):
    """
    Tune the MoE (Mixture of Experts) Triton kernel for the model.

    Uses vLLM to tune the MoE kernel. If an existing artifact exists and the
    Triton version matches, tuning is skipped and cached configs are reused.
    """

    algorithm_name: str = "moe_kernel_tuner"
    group_tags: list[str] = [tags.KERNEL]
    save_fn: None = None
    references: dict[str, str] = {
        "GitHub": "https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py",
    }
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda", "accelerate"]
    dataset_required: bool = False
    compatible_before: Iterable[str] = tags.tags_compatible_with_moe_kernel()
    compatible_after: Iterable[str] = tags.tags_compatible_with_moe_kernel()
    required_install = "``uv pip install vllm>=0.11.0``"

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            CategoricalHyperparameter(
                "compute_dtype",
                choices=["bfloat16", "float16"],
                default_value="bfloat16",
                meta=dict(desc="Compute dtype to use."),
            ),
            CategoricalHyperparameter(
                "weight_dtype",
                choices=["fp16", "fp8_w8a8", "int8_w8a16"],
                default_value="fp16",
                meta=dict(desc="Dtype to use for the weights (and activations)."),
            ),
            OrdinalHyperparameter(
                "tensor_parallel_size",
                sequence=[1, 2, 4, 8, 16, 32],
                default_value=1,
                meta=dict(desc="Tensor parallel size to use if the model can not fit on a single GPU."),
            ),
            UnconstrainedHyperparameter(
                "path_to_huggingface_hub_cache",
                default_value="~",
                meta=dict(
                    desc=(
                        "Path to the Hugging Face Hub cache directory "
                        "(that contains `kernels` configs). If not provided, "
                        "the cache will be saved in the current working directory."
                    )
                ),
            ),
            UnconstrainedHyperparameter(
                "path_to_vllm_cache",
                default_value="vllm/model_executor/layers/fused_moe/configs",
                meta=dict(desc="Path to the vLLM MoE configs directory."),
            ),
            OrdinalHyperparameter(
                "num_iters",
                sequence=[1, 20, 50, 100],
                default_value=20,
                meta=dict(desc="Number of iterations to average the kernel times on."),
            ),
            OrdinalHyperparameter(
                "block_size_m_max",
                sequence=[4, 5, 6, 7, 8, 9, 10],
                default_value=8,
                meta=dict(desc="Maximum (log) block size for tiling through input dimension."),
            ),
            OrdinalHyperparameter(
                "block_size_n_max",
                sequence=[5, 6, 7, 8, 9, 10],
                default_value=8,
                meta=dict(desc="Maximum (log) block size for tiling through output dimension."),
            ),
            OrdinalHyperparameter(
                "block_size_k_max",
                sequence=[6, 7, 8, 9, 10],
                default_value=8,
                meta=dict(desc="Maximum (log) block size for tiling through intermediate dimension."),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a MoE model.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a valid model for the algorithm, False otherwise.
        """
        if model.__class__.__name__ == "HunyuanImage3ForCausalMM":
            return True
        return is_moe_lm(model) or is_transformers_pipeline_with_moe_lm(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Tune the MoE Triton kernel for the model.

        If an existing artifact exists in the cache with the same Triton version,
        tuning is skipped and the cached configs are restored to the HF/vLLM caches.

        Parameters
        ----------
        model : Any
            The model to wrap.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the application of the algorithm.

        Returns
        -------
        Any
            The untouched model.
        """
        if is_transformers_pipeline_with_moe_lm(model):
            return self._apply_to_model_within_transformers_pipeline(model, smash_config)

        imported_packages = self.import_algorithm_packages()

        # (i) Get the MoE parameters: exception first (Hunyuan), then general case.
        model_config = getattr(model, "config", None)
        if model_config is None:
            raise ValueError(f"Model {model.__class__.__name__} has no config.")

        tensor_parallel_size = int(smash_config["tensor_parallel_size"])
        if model.__class__.__name__ == "HunyuanImage3ForCausalMM":
            nb_experts, shard_intermediate_size, hidden_size, topk = extract_hunyuan_dimensions(
                model, model_config, tensor_parallel_size
            )
        else:
            nb_experts, shard_intermediate_size, hidden_size, topk = extract_transformers_moe_dimensions(
                model, model_config, tensor_parallel_size
            )

        # (ii) Get the compute parameters
        dtype = smash_config["compute_dtype"]
        dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        use_fp8_w8a8 = smash_config["weight_dtype"] == "fp8_w8a8"
        use_int8_w8a16 = smash_config["weight_dtype"] == "int8_w8a16"

        # (iii) Tune the kernel over a range of batch sizes (single GPU per Ray worker).
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        ray = imported_packages["ray"]
        tune_kernel = imported_packages["tune_kernel"]
        get_configs_compute_bound = imported_packages["get_configs_compute_bound"]
        ensure_benchmark_config = imported_packages["ensure_benchmark_config"]
        save_configs = imported_packages["save_configs"]
        no_valid_config_error = imported_packages["NoValidConfigError"]

        ray.init(ignore_reinit_error=True)
        search_space = get_configs_compute_bound(smash_config)
        pruna_logger.info(f"Start tuning over {len(search_space)} configurations...")

        start = time.time()
        ray_outputs: list[Any] = []
        tune_kernel = ray.remote(num_gpus=1)(tune_kernel)

        for batch_size in batch_sizes:
            out = tune_kernel.remote(
                batch_size,
                nb_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a16,
                search_space,
                None,
                False,
                imported_packages,
                0,  # fixed seed for reproducibility
                smash_config["num_iters"],
            )
            ray_outputs.append(out)

        # Shutdown Ray processes on other devices and log any resulting exception as warnings.
        tuned_config_by_batch_size: dict[int, Any] = {}
        try:
            for batch_size, output_ref in zip(batch_sizes, ray_outputs):
                try:
                    raw_config = ray.get(output_ref)
                    tuned_config_by_batch_size[batch_size] = ensure_benchmark_config(raw_config)
                except no_valid_config_error:
                    pruna_logger.warning(
                        "No valid config for batch_size=%s; skipping (smaller batch sizes may still be used).",
                        batch_size,
                    )
        finally:
            try:
                ray.shutdown()
            except Exception as e:
                pruna_logger.warning("Exception during ray.shutdown(): %s", e)

        if not tuned_config_by_batch_size:
            raise RuntimeError(
                "No valid kernel configuration was found for any batch size. "
                "All configurations failed (e.g., due to OutOfResources). "
                "This can happen on GPUs with limited resources. "
                "Consider reducing your model size or tuning search space."
            )

        end = time.time()
        pruna_logger.info(f"Tuning took {end - start:.2f} seconds")

        save_configs(
            tuned_config_by_batch_size,
            nb_experts,
            shard_intermediate_size,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a16,
            None,
            smash_config["path_to_huggingface_hub_cache"],
            smash_config["path_to_vllm_cache"],
            imported_packages,
        )

        best_configs_and_hyperparameters = {
            "triton_version": imported_packages["triton"].__version__,
            "best_configs_moe_kernel": tuned_config_by_batch_size,
            "num_experts": nb_experts,
            "shard_intermediate_size": shard_intermediate_size,
            "dtype": "bfloat16" if dtype == torch.bfloat16 else "float16",
            "use_fp8_w8a8": use_fp8_w8a8,
            "use_int8_w8a16": use_int8_w8a16,
        }
        with open(smash_config.cache_dir / "moe_kernel_tuner.json", "w") as f:
            json.dump(best_configs_and_hyperparameters, f)

        smash_config.save_artifacts_fns.append(SAVE_ARTIFACTS_FUNCTIONS.moe_kernel_tuner_artifacts.name)
        return model

    def import_algorithm_packages(self) -> dict[str, Any]:
        """
        Import the algorithm packages (vLLM, Triton, Ray, and utils).

        Ray is imported here so it is not a top-level dependency of the package.
        Utils are imported here so they are only loaded when this algorithm is used.

        Returns
        -------
        dict[str, Any]
            The algorithm packages and helpers (tune, save_configs, etc.).
        """
        import ray
        import vllm.envs as envs
        import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe
        import vllm.platforms as vllm_platforms
        from vllm.model_executor.layers.fused_moe import override_config
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEQuantConfig,
            _get_config_dtype_str,
        )
        from vllm.triton_utils import triton
        from pruna.algorithms.utils.moe_kernel_tuner import (
            BenchmarkConfig,
            NoValidConfigError,
            ensure_benchmark_config,
            get_configs_compute_bound,
            save_configs,
            tune_kernel,
        )

        return dict(
            FusedMoEQuantConfig=FusedMoEQuantConfig,
            _get_config_dtype_str=_get_config_dtype_str,
            FusedMoE=fused_moe,
            vllm_platforms=vllm_platforms,
            triton=triton,
            override_config=override_config,
            envs=envs,
            ray=ray,
            tune_kernel=tune_kernel,
            get_configs_compute_bound=get_configs_compute_bound,
            ensure_benchmark_config=ensure_benchmark_config,
            save_configs=save_configs,
            NoValidConfigError=NoValidConfigError,
            BenchmarkConfig=BenchmarkConfig,
        )


def extract_hunyuan_dimensions(
    model: Any,
    model_config: Any,
    tensor_parallel_size: int,
) -> tuple[int, int, int, int]:
    """
    Extract MoE dimensions for HunyuanImage3ForCausalMM.

    Parameters
    ----------
    model : Any
        The model (used only for type context).
    model_config : Any
        The model's config object. Must have num_experts, moe_topk, intermediate_size, hidden_size.
    tensor_parallel_size : int
        Tensor parallel size (must divide intermediate_size).

    Returns
    -------
    tuple[int, int, int, int]
        (nb_experts, shard_intermediate_size, hidden_size, topk).
        - nb_experts: number of experts in the MoE layer.
        - shard_intermediate_size: intermediate dimension per GPU shard (after SiLU-and-mul).
        - hidden_size: model hidden dimension.
        - topk: number of active experts per token.
    """
    if getattr(model_config, "num_experts", None) is None:
        raise ValueError(f"Model config {type(model_config).__name__} is missing attribute 'num_experts'.")
    if getattr(model_config, "moe_topk", None) is None:
        raise ValueError(f"Model config {type(model_config).__name__} is missing attribute 'moe_topk'.")
    if getattr(model_config, "intermediate_size", None) is None:
        raise ValueError(f"Model config {type(model_config).__name__} is missing attribute 'intermediate_size'.")
    if getattr(model_config, "hidden_size", None) is None:
        raise ValueError(f"Model config {type(model_config).__name__} is missing attribute 'hidden_size'.")

    nb_experts = int(model_config.num_experts)
    topk = int(model_config.moe_topk[0])
    intermediate_size = int(model_config.intermediate_size)
    hidden_size = int(model_config.hidden_size)

    if intermediate_size % tensor_parallel_size != 0:
        raise ValueError(
            f"Expected tensor_parallel_size to be a divisor of the model's intermediate size "
            f"(MLP hidden dimension) {intermediate_size}, but got {tensor_parallel_size}."
        )
    shard_intermediate_size = 2 * intermediate_size // tensor_parallel_size
    return nb_experts, shard_intermediate_size, hidden_size, topk


def extract_transformers_moe_dimensions(
    model: Any,
    model_config: Any,
    tensor_parallel_size: int,
) -> tuple[int, int, int, int]:
    """
    Extract MoE dimensions for standard transformers MoE LMs (e.g. Mixtral, Qwen-MoE).

    Parameters
    ----------
    model : Any
        The model (used to decide num_experts_per_tok vs moe_topk).
    model_config : Any
        The model's config. Must have num_experts, intermediate_size (or moe_intermediate_size),
        hidden_size, and either num_experts_per_tok or moe_topk.
    tensor_parallel_size : int
        Tensor parallel size (must divide intermediate_size).

    Returns
    -------
    tuple[int, int, int, int]
        (nb_experts, shard_intermediate_size, hidden_size, topk).
        - nb_experts: number of experts in the MoE layer.
        - shard_intermediate_size: intermediate dimension per GPU shard.
        - hidden_size: model hidden dimension.
        - topk: number of active experts per token.
    """
    if getattr(model_config, "num_experts", None) is None:
        raise ValueError(f"Model config {type(model_config).__name__} is missing attribute 'num_experts'.")
    if getattr(model_config, "hidden_size", None) is None:
        raise ValueError(f"Model config {type(model_config).__name__} is missing attribute 'hidden_size'.")

    nb_experts = int(model_config.num_experts)
    hidden_size = int(model_config.hidden_size)

    if is_moe_lm(model):
        if getattr(model_config, "num_experts_per_tok", None) is None:
            raise ValueError(f"Model config {type(model_config).__name__} is missing attribute 'num_experts_per_tok'.")
        topk = int(model_config.num_experts_per_tok)
    else:
        if getattr(model_config, "moe_topk", None) is None:
            raise ValueError(f"Model config {type(model_config).__name__} is missing attribute 'moe_topk'.")
        topk = int(model_config.moe_topk[0])

    moe_intermediate = getattr(model_config, "moe_intermediate_size", None)
    if moe_intermediate is not None:
        intermediate_size = int(moe_intermediate)
    else:
        if getattr(model_config, "intermediate_size", None) is None:
            raise ValueError(
                f"Model config {type(model_config).__name__} is missing attribute "
                "'intermediate_size' or 'moe_intermediate_size'."
            )
        intermediate_size = int(model_config.intermediate_size)

    if intermediate_size % tensor_parallel_size != 0:
        raise ValueError(
            f"Expected tensor_parallel_size to be a divisor of the model's intermediate size "
            f"(MLP hidden dimension) {intermediate_size}, but got {tensor_parallel_size}."
        )
    shard_intermediate_size = 2 * intermediate_size // tensor_parallel_size
    return nb_experts, shard_intermediate_size, hidden_size, topk
