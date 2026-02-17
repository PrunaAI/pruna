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
from importlib.util import find_spec
from itertools import product
from typing import Any, Dict, TypedDict

import torch

from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.logging.logger import pruna_logger


class NoValidConfigError(RuntimeError):
    """
    Raised when no valid kernel configuration was found for a given batch size.

    All configurations failed (e.g., due to OutOfResources). This can happen on
    GPUs with limited resources. The caller may retry with smaller batch sizes.
    """


class BenchmarkConfig(TypedDict):
    """The configuration for the matrix multiplication (tiling and warp scheduling)."""

    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def tune_kernel(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    search_space: list[dict[str, int]],
    block_quant_shape: list[int] | None,
    use_deep_gemm: bool,
    imported_packages: Dict[str, Any],
    seed: int,
    num_iters: int,
) -> BenchmarkConfig:
    """
    Tune a given Triton kernel (run inside a Ray worker; no Ray at module level).

    Parameters
    ----------
    num_tokens : int
        Batch size in tokens (sequence length × batch dimension).
    num_experts : int
        Number of experts in the MoE layer.
    shard_intermediate_size : int
        Intermediate size of the model in the shard (if using tensor parallelism).
    hidden_size : int
        Model hidden dimension (input/output of the expert layer).
    topk : int
        Number of active experts per token.
    dtype : torch.dtype
        Dtype for weights and activations.
    use_fp8_w8a8 : bool
        Whether to use fp8_w8a8.
    use_int8_w8a16 : bool
        Whether to use int8_w8a16.
    search_space : list[dict[str, int]]
        Search space for the kernel (tiling and warp scheduling).
    block_quant_shape : list[int] | None
        Block shape for the kernel (None here).
    use_deep_gemm : bool
        Whether to use deep gemm (False here).
    imported_packages : Dict[str, Any]
        Imported packages (vllm, triton, etc.).
    seed : int
        Random seed for reproducibility.
    num_iters : int
        Number of iterations to average the kernel time on.

    Returns
    -------
    BenchmarkConfig
        The best config found.
    """
    import ray.experimental.tqdm_ray as tqdm_ray
    from datetime import datetime

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
            continue

        if kernel_time < best_time:
            best_time, best_config = kernel_time, config

    now = datetime.now()
    pruna_logger.info(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
    if best_config is None:
        raise NoValidConfigError(
            f"No valid kernel configuration was found for batch_size={num_tokens}. "
            "All configurations failed (e.g., due to OutOfResources). "
            "This can happen on GPUs with limited resources. "
            "Consider reducing your model size, or tuning search space."
        )
    return best_config


def ensure_benchmark_config(config: dict[str, Any]) -> BenchmarkConfig:
    """
    Convert the raw dict returned by tune to a canonical BenchmarkConfig.

    Preserves optional keys (e.g. waves_per_eu, matrix_instr_nonkdim, kpack) when
    present so that the config file format stays compatible with vLLM.

    Parameters
    ----------
    config : dict[str, Any]
        Raw config from tune (same keys as BenchmarkConfig, possibly with extras).

    Returns
    -------
    BenchmarkConfig
        Config with required keys and any optional keys that were present.
    """
    result: dict[str, Any] = {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }
    if "waves_per_eu" in config:
        result["waves_per_eu"] = config.get("waves_per_eu")
    if "matrix_instr_nonkdim" in config:
        result["matrix_instr_nonkdim"] = config.get("matrix_instr_nonkdim")
    if "kpack" in config:
        result["kpack"] = config.get("kpack")
    return result


def get_configs_compute_bound(smash_config: SmashConfigPrefixWrapper) -> list[dict[str, int]]:
    """
    Get the grid-search space for the kernel (tiling and warp scheduling).

    Parameters
    ----------
    smash_config : SmashConfigPrefixWrapper
        The Smash configuration.

    Returns
    -------
    list[dict[str, int]]
        The search space for the kernel (tiling and warp scheduling).
    """
    configs: list[dict[str, int]] = []

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
    block_quant_shape: list[int] | None = None,
    use_deep_gemm: bool = False,
    imported_packages: Dict[str, Any] | None = None,
) -> float:
    """
    Benchmark a given Triton kernel using CUDAGraph.

    This function is copied from the vLLM repository.
    https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py

    Tuning runs on a single GPU per Ray worker; Ray sets CUDA_VISIBLE_DEVICES
    so that set_device(0) in this function selects the assigned GPU.

    Parameters
    ----------
    config : BenchmarkConfig
        The configuration to benchmark.
    num_tokens : int
        Batch size in tokens.
    num_experts : int
        Number of experts.
    shard_intermediate_size : int
        Intermediate size of the model in the shard (if using tensor parallelism).
    hidden_size : int
        Model hidden dimension.
    topk : int
        Number of active experts per token.
    dtype : torch.dtype
        Dtype for weights and activations.
    use_fp8_w8a8 : bool
        Whether to use fp8_w8a8.
    use_int8_w8a16 : bool
        Whether to use int8_w8a16.
    num_iters : int
        Number of iterations to run the benchmark.
    block_quant_shape : list[int] | None
        Block shape for the kernel (None here).
    use_deep_gemm : bool
        Whether to use deep gemm (False here).
    imported_packages : Dict[str, Any] | None
        Imported packages (vllm, triton, etc.).

    Returns
    -------
    float
        Average latency per kernel invocation in microseconds (each CUDA graph
        replay runs 10 kernel invocations; we average over num_iters replays
        then convert to µs).
    """
    if imported_packages is None:
        raise ValueError("imported_packages is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MoeKernelTuner.")
    # Ray sets CUDA_VISIBLE_DEVICES per worker to the GPU it scheduled; use device 0.
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

    run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

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
    # Average latency per kernel invocation in microseconds (each replay runs 10 invocations).
    avg_us = sum(latencies) / (num_iters * 10) * 1000
    graph.reset()
    return avg_us


def write_config_file(
    path: pathlib.Path,
    data: dict[str, Any],
    merge_if_exists: bool,
    write_only_if_missing: bool,
    log_label: str,
) -> None:
    """
    Write config dict to a JSON file, optionally merging with existing content.

    Parameters
    ----------
    path : pathlib.Path
        Target file path.
    data : dict[str, Any]
        Config to write (will be merged into existing if merge_if_exists and file exists).
    merge_if_exists : bool
        If True and path exists, load existing JSON, update with data, then write.
    write_only_if_missing : bool
        If True and path already exists, do not write (preserves existing file).
    log_label : str
        Label for the log message when writing.
    """
    if write_only_if_missing and path.exists():
        return
    if merge_if_exists and path.exists():
        with open(path) as f:
            existing: dict[str, Any] = json.load(f)
        existing.update(data)
        data = existing
    path.parent.mkdir(parents=True, exist_ok=True)
    pruna_logger.info(f"Writing best config to {log_label}...")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def save_configs(
    configs: dict[int, BenchmarkConfig],
    num_experts: int,
    shard_intermediate_size: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_quant_shape: list[int] | None,
    path_to_huggingface_hub_cache: str,
    path_to_vllm_cache: str,
    imported_packages: Dict[str, Any],
) -> None:
    """
    Save the best configs to the HF cache and vLLM cache.

    Writes one file per cache; for the HF path we merge with existing file content
    if present so that other keys are preserved. Uses a shared helper to avoid
    duplicating the write logic.

    Parameters
    ----------
    configs : dict[int, BenchmarkConfig]
        Best config per batch size (keys are batch sizes).
    num_experts : int
        Number of experts.
    shard_intermediate_size : int
        Intermediate size of the model in the shard (if using tensor parallelism).
    dtype : torch.dtype
        Dtype for weights and activations.
    use_fp8_w8a8 : bool
        Whether to use fp8_w8a8.
    use_int8_w8a16 : bool
        Whether to use int8_w8a16.
    block_quant_shape : list[int] | None
        Block shape for the kernel (None here).
    path_to_huggingface_hub_cache : str
        Path to the Hugging Face Hub cache.
    path_to_vllm_cache : str
        Path to the vLLM MoE configs directory (used when VLLM_TUNED_CONFIG_FOLDER is None).
    imported_packages : Dict[str, Any]
        Imported packages (vllm, triton, etc.).
    """
    dtype_str = imported_packages["_get_config_dtype_str"](
        dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8
    )

    filename = imported_packages["FusedMoE"].get_config_file_name(
        num_experts, shard_intermediate_size // 2, dtype_str, block_quant_shape
    )

    triton_version = imported_packages["triton"].__version__
    # JSON keys must be strings; configs keys are int (batch sizes).
    data: dict[str, Any] = {"triton_version": triton_version}
    for k, v in configs.items():
        data[str(k)] = v

    path_to_kernel_configs = (
        pathlib.Path(path_to_huggingface_hub_cache) /
        ".cache/huggingface/hub/models--RedHatAI--moe/blobs/configs"
    )
    path_hf = path_to_kernel_configs / filename
    write_config_file(
        path_hf, data,
        merge_if_exists=True,
        write_only_if_missing=False,
        log_label=str(path_hf),
    )

    path_to_vllm_configs = imported_packages["envs"].VLLM_TUNED_CONFIG_FOLDER
    if path_to_vllm_configs is None:
        submodule_locations = find_spec("vllm").submodule_search_locations
        if submodule_locations is not None and len(submodule_locations) > 0:
            path_where_vllm_is_installed = submodule_locations[0]
        else:
            raise RuntimeError("Could not determine installation path for vllm.")
        path_to_vllm_configs = pathlib.Path(path_where_vllm_is_installed).parent / path_to_vllm_cache
    path_vllm = pathlib.Path(path_to_vllm_configs) / filename
    write_config_file(
        path_vllm, data,
        merge_if_exists=False,
        write_only_if_missing=True,
        log_label=str(path_vllm),
    )
