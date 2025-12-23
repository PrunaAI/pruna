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
import pathlib
import time
from collections.abc import Iterable
from datetime import datetime
from importlib.util import find_spec
from itertools import product
from typing import Any, Dict, TypedDict

import ray
import ray.experimental.tqdm_ray as tqdm_ray
import torch
from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.hyperparameters import UnconstrainedHyperparameter
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.load import LOAD_FUNCTIONS
from pruna.engine.model_checks import is_moe_lm, is_transformers_pipeline_with_moe_lm
from pruna.logging.logger import pruna_logger


class MoeKernelTuner(PrunaAlgorithmBase):
    """
    Tune the MoE Triton kernel for the model.

    Uses vLLM to tune the MoE kernel.
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
    compatible_before: Iterable[str] = [
        tags.KERNEL,
        tags.QUANTIZER,
        tags.PRUNER,
        tags.CACHER,
        tags.FACTORIZER,
        tags.BATCHER,
        tags.COMPILER,
    ]
    compatible_after: Iterable[str] = [
        tags.KERNEL,
        tags.QUANTIZER,
        tags.PRUNER,
        tags.CACHER,
        tags.FACTORIZER,
        tags.BATCHER,
        tags.COMPILER,
    ]
    required_install = "``uv pip install vllm``"

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
        # Hunyuan3-image is a MoE model, but not depending on mixtral
        if model.__class__.__name__ == "HunyuanImage3ForCausalMM":
            return True
        else:
            return is_moe_lm(model) or is_transformers_pipeline_with_moe_lm(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Tune the MoE Triton kernel for the model.

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

        # (i) Get the MoE parameters
        model_config = model.config
        if model_config is None:
            raise ValueError(f"Model {model.__class__.__name__} has no config.")
        nb_experts = model_config.num_experts                # number of experts
        # number of active experts per token
        topk = (
            model_config.num_experts_per_tok
            if is_moe_lm(model)
            else model_config.moe_topk[0]
        )
        # qwen_moe can use different intermediate size compared to mixtral.
        intermediate_size = (
            model_config.moe_intermediate_size
            if model_config.moe_intermediate_size is not None
            else model_config.intermediate_size
        )
        hidden_size = model_config.hidden_size  # model hidden dim
        assert intermediate_size % smash_config["tensor_parallel_size"] == 0, (
            f"intermediate_size {intermediate_size} is not divisible by tp "
            f"{smash_config['tensor_parallel_size']}."
        )
        shard_intermediate_size = 2 * intermediate_size // smash_config["tensor_parallel_size"]

        # (ii) Get the compute parameters
        dtype = smash_config["compute_dtype"]
        dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        use_fp8_w8a8 = smash_config["weight_dtype"] == "fp8_w8a8"
        use_int8_w8a16 = smash_config["weight_dtype"] == "int8_w8a16"

        # (iii) Tune the kernel over a range of batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

        # use ray to parallelize the tuning
        ray.init()

        is_fp16 = not (use_fp8_w8a8 or use_int8_w8a16)
        search_space = get_configs_compute_bound(is_fp16, smash_config)
        pruna_logger.info(f"Start tuning over {len(search_space)} configurations...")

        start = time.time()
        outputs = []
        for batch_size in batch_sizes:
            output = tune.remote(
                batch_size,                 # num_tokens
                nb_experts,                          # num_experts per block
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a16,
                search_space,
                None,                       # we don't suport block quantization for now
                False,                      # not use_deep_gemm
                imported_packages,
                0,                          # random seed
                smash_config["num_iters"],
            )
            outputs.append(output)

        configs = ray.get(outputs)

        # (iv) Sort the configs by batch size and save the best configs
        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        # save configs in caches (for hf and vllm)
        save_configs(
            best_configs,
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
        # stash results in the SmashConfig for later loading (cannot add new hyperparams to ConfigSpace here)
        payload = dict(
            best_configs_moe_kernel=best_configs,
            num_experts=nb_experts,
            shard_intermediate_size=shard_intermediate_size,
            dtype=dtype,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
        )
        # store artifacts in SmashConfig so they persist across save/load
        smash_config.artifacts["moe_kernel_tuner"] = payload
        # attach load function to the smash config for loading
        smash_config.load_fns.append(LOAD_FUNCTIONS.moe_kernel_tuner.name)
        end = time.time()
        pruna_logger.info(f"Tuning took {end - start:.2f} seconds")

        # (v) Return the model (untouched; only the configs are saved)
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        import vllm.envs as envs
        import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe
        import vllm.platforms as vllm_platforms
        from vllm.model_executor.layers.fused_moe import override_config
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEQuantConfig,
            _get_config_dtype_str,
        )
        from vllm.triton_utils import triton

        return dict(
            FusedMoEQuantConfig=FusedMoEQuantConfig,
            _get_config_dtype_str=_get_config_dtype_str,
            FusedMoE=fused_moe,
            vllm_platforms=vllm_platforms,
            triton=triton,
            override_config=override_config,
            envs=envs,
        )


class BenchmarkConfig(TypedDict):
    """The configuration for the matrix multiplication (tiling and warp scheduling)."""

    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


# Converts the function into a Ray actor and requests one GPU per actor instance
@ray.remote(num_gpus=1)
def tune(
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        search_space: list[dict[str, int]],
        block_quant_shape: list[int],
        use_deep_gemm: bool,
        imported_packages: Dict[str, Any],
        seed: int,
        num_iters: int,
    ) -> dict[str, int]:
    """
    Tune a given Triton kernel.

    Parameters
    ----------
    num_tokens: int
        The number of tokens in the batch.
    num_experts: int
        The number of experts.
    shard_intermediate_size: int
        The intermediate size of the model in the shard (if using tensor parallelism).
    hidden_size: int
        The hidden size of the model.
    topk: int
        The number of active experts per token.
    dtype: torch.dtype
        The dtype to use for the weights and activations.
    use_fp8_w8a8: bool
        Whether to use fp8_w8a8.
    use_int8_w8a16: bool
        Whether to use int8_w8a16.
    search_space: list[dict[str, int]]
        The search space for the kernel (tiling and warp scheduling).
    block_quant_shape: list[int]
        The block shape for the kernel (None here).
    use_deep_gemm: bool
        Whether to use deep gemm (False here).
    imported_packages: Dict[str, Any]
        The imported packages (vllm, triton, etc.).
    seed: int
        The random seed.
    num_iters: int
        The number of iterations to average the kernel time on.

    Returns
    -------
    dict[str, int]
        The best config.
    """
    imported_packages["vllm_platforms"].current_platform.seed_everything(seed)
    best_config = None
    best_time = float("inf")

    for config in tqdm_ray.tqdm(search_space):
        try:
            kernel_time = benchmark_config(
                config,
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a16,
                num_iters=num_iters,
                block_quant_shape=block_quant_shape,
                use_deep_gemm=use_deep_gemm,
                imported_packages=imported_packages,
            )
        except imported_packages["triton"].runtime.autotuner.OutOfResources:
            # Some configurations may be invalid and fail to compile.
            continue

        if kernel_time < best_time:
            best_time, best_config = kernel_time, config

    now = datetime.now()
    pruna_logger.info(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
    assert best_config is not None
    return best_config


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    """
    Sort the configuration (tiling and warp scheduling).

    Parameters
    ----------
    config: BenchmarkConfig
        The configuration to sort.

    Returns
    -------
    BenchmarkConfig
        The sorted configuration.
    """
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        **(
            {"waves_per_eu": config.get("waves_per_eu")} if "waves_per_eu" in config else {}
        ),
        **(
            {"matrix_instr_nonkdim": config.get("matrix_instr_nonkdim")}
            if "matrix_instr_nonkdim" in config
            else {}
        ),
        **({"kpack": config.get("kpack")} if "kpack" in config else {}),
    }


def get_configs_compute_bound(use_fp16: bool, smash_config: SmashConfigPrefixWrapper) -> list[dict[str, int]]:
    """
    Get the gridsearch space for the kernel (tiling and warp scheduling).

    Parameters
    ----------
    use_fp16: bool
        Whether to use fp16.
    smash_config: SmashConfigPrefixWrapper
        The Smash configuration.

    Returns
    -------
    list[dict[str, int]]
        The search space for the kernel (tiling and warp scheduling).
    """
    configs: list[BenchmarkConfig] = []

    # Reduced search space for faster tuning.
    block_m_range = [2**i for i in range(4, smash_config["block_size_m_max"] + 1)]
    block_n_range = [2**i for i in range(5, smash_config["block_size_n_max"] + 1)]
    block_k_range = [2**i for i in range(6, smash_config["block_size_k_max"] + 1)]
    num_warps_range = [4, 8]
    group_m_range = [1, 16, 32, 64]
    num_stage_range = [2, 3, 4, 5]

    param_ranges = {
        "BLOCK_SIZE_M": block_m_range,
        "BLOCK_SIZE_N": block_n_range,
        "BLOCK_SIZE_K": block_k_range,
        "GROUP_SIZE_M": group_m_range,
        "num_warps": num_warps_range,
        "num_stages": num_stage_range,
    }

    keys, values = zip(*param_ranges.items())
    for config_values in product(*values):
        config = dict(zip(keys, config_values))
        configs.append(config)

    return configs


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    num_iters: int = 100,
    block_quant_shape: list[int] = None,
    use_deep_gemm: bool = False,
    imported_packages: Dict[str, Any] = None,
) -> float:
    """
    Benchmark a given Triton kernel using CUDAGraph.

    This function is copied from the vllm repository.
    https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py

    Parameters
    ----------
    config: BenchmarkConfig
        The configuration to benchmark.
    num_tokens: int
        The number of tokens in the batch.
    num_experts: int
        The number of experts.
    shard_intermediate_size: int
        The intermediate size of the model in the shard (if using tensor parallelism).
    hidden_size: int
        The hidden size of the model.
    topk: int
        The number of active experts per token.
    dtype: torch.dtype
        The dtype to use for the weights and activations.
    use_fp8_w8a8: bool
        Whether to use fp8_w8a8.
    use_int8_w8a16: bool
        Whether to use int8_w8a16.
    num_iters: int
        The number of iterations to run the benchmark.
    block_quant_shape: list[int]
        The block shape for the kernel (None here).
    use_deep_gemm: bool
        Whether to use deep gemm (False here).
    imported_packages: Dict[str, Any]
        The imported packages (vllm, triton, etc.).

    Returns
    -------
    float
        The average latency of the kernel.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MoeKernelTuner.")
    # Ray sets CUDA_VISIBLE_DEVICES per worker to the GPU it scheduled
    torch.cuda.set_device(0)
    device = torch.device("cuda")

    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    if use_int8_w8a16:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
            device=device,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
            device=device,
        )
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype, device=device
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype, device=device
        )
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32, device=device)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32, device=device
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32, device=device)
    if use_deep_gemm:
        # we use the default block shape for deepgemm
        block_quant_shape = [128, 128]
    if use_fp8_w8a8:
        if block_quant_shape:
            block_n, block_k = block_quant_shape[0], block_quant_shape[1]
            e = num_experts
            n = shard_intermediate_size // 2
            k = hidden_size
            factor_for_scale = 1e-2
            n_tiles_w1 = (2 * n + block_n - 1) // block_n
            n_tiles_w2 = (k + block_n - 1) // block_n
            k_tiles_w1 = (k + block_k - 1) // block_k
            k_tiles_w2 = (n + block_k - 1) // block_k
            w1_scale = (
                torch.rand((e, n_tiles_w1, k_tiles_w1), dtype=torch.float32, device=device)
                * factor_for_scale
            )
            w2_scale = (
                torch.rand((e, n_tiles_w2, k_tiles_w2), dtype=torch.float32, device=device)
                * factor_for_scale
            )
        else:
            w1_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
            w2_scale = torch.randn(num_experts, dtype=torch.float32, device=device)

        a1_scale = torch.randn(1, dtype=torch.float32, device=device)
        a2_scale = torch.randn(1, dtype=torch.float32, device=device)

        w1 = w1.to(device=device, dtype=imported_packages["vllm_platforms"].current_platform.fp8_dtype())
        w2 = w2.to(device=device, dtype=imported_packages["vllm_platforms"].current_platform.fp8_dtype())

    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32, device=device)

    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        if use_fp8_w8a8:
            quant_dtype = torch.float8_e4m3fn
        elif use_int8_w8a16:
            quant_dtype = torch.int8
        else:
            quant_dtype = None

        quant_config = imported_packages["FusedMoEQuantConfig"].make(
            quant_dtype=quant_dtype,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_quant_shape,
        )

        with imported_packages["override_config"](config):
            topk_weights, topk_ids, token_expert_indices = imported_packages["FusedMoE"].fused_topk(
                x, input_gating, topk, renormalize=not use_deep_gemm
            )
            return imported_packages["FusedMoE"].fused_experts(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                inplace=True,
                quant_config=quant_config,
                allow_deep_gemm=use_deep_gemm,
            )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


def save_configs(
    configs: dict[int, BenchmarkConfig],
    num_experts: int,
    shard_intermediate_size: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_quant_shape: list[int],
    path_to_huggingface_hub_cache: str,
    path_to_vllm_cache: str,
    imported_packages: Dict[str, Any],
) -> None:
    """
    Save the best configs to the hf cache and vllm cache.

    Parameters
    ----------
    configs: dict[int, BenchmarkConfig]
        The best configs.
    num_experts: int
        The number of experts.
    shard_intermediate_size: int
        The intermediate size of the model in the shard (if using tensor parallelism).
    hidden_size: int
        The hidden size of the model.
    topk: int
        The number of active experts per token.
    dtype: torch.dtype
        The dtype to use for the weights and activations.
    use_fp8_w8a8: bool
        Whether to use fp8_w8a8.
    use_int8_w8a16: bool
        Whether to use int8_w8a16.
    block_quant_shape: list[int]
        The block shape for the kernel (None here).
    path_to_huggingface_hub_cache: str
        The path to the huggingface hub cache.
    path_to_vllm_cache: str
        The path to the vllm cache.
    imported_packages: Dict[str, Any]
        The imported packages (vllm, triton, etc.).
    """
    dtype_str = imported_packages["_get_config_dtype_str"](
        dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8
    )

    # (i) Get the name of the config file
    # NB from vllm: The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    filename = imported_packages["FusedMoE"].get_config_file_name(
        num_experts, shard_intermediate_size // 2, dtype_str, block_quant_shape
    )

    # (ii) Save the config to the hf cache (where `kernels` lib expects to find it)
    path_to_kernel_configs = (
        pathlib.Path(path_to_huggingface_hub_cache) /
        ".cache/huggingface/hub/models--RedHatAI--moe/blobs/configs"
    )
    pathlib.Path(path_to_kernel_configs).mkdir(exist_ok=True, parents=True)
    filename_hf = path_to_kernel_configs / filename
    if not pathlib.Path(filename_hf).exists():
        pruna_logger.info(f"Writing best config to {filename_hf}...")
        with open(filename_hf, "w") as f:
            json.dump({"triton_version": imported_packages["triton"].__version__, **configs}, f, indent=4)
            f.write("\n")

    # (iii) Save the config to the vllm cache (where `vllm` expects to find it)
    path_to_vllm_configs = imported_packages["envs"].VLLM_TUNED_CONFIG_FOLDER
    if path_to_vllm_configs is None:
        submodule_locations = find_spec("vllm").submodule_search_locations
        if submodule_locations is not None and len(submodule_locations) > 0:
            path_where_vllm_is_installed = submodule_locations[0]
        else:
            raise RuntimeError("Could not determine installation path for vllm.")
        path_to_vllm_configs = pathlib.Path(path_where_vllm_is_installed).parent / path_to_vllm_cache
    pathlib.Path(path_to_vllm_configs).mkdir(exist_ok=True, parents=True)
    filename_vllm = path_to_vllm_configs / filename
    if not pathlib.Path(filename_vllm).exists():
        pruna_logger.info(f"Writing best config to {filename_vllm}...")
        with open(filename_vllm, "w") as f:
            json.dump({"triton_version": imported_packages["triton"].__version__, **configs}, f, indent=4)
            f.write("\n")
