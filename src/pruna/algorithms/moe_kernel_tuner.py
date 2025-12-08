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

from collections.abc import Iterable
from typing import Any, Dict, TypedDict

import ray.experimental.tqdm_ray as tqdm_ray
import ray
import time
import torch
from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter
from contextlib import nullcontext
from datetime import datetime
import os
import json
from itertools import product

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_moe_lm, is_transformers_pipeline_with_moe_lm
from pruna.logging.logger import pruna_logger


class MoeKernelTuner(PrunaAlgorithmBase):
    """
    Tune the MoE kernel for the model.

    Uses vLLM to tune the MoE kernel of the model.
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
                choices=["fp8_w8a8", "int8_w8a16"],
                default_value="fp8_w8a8",
                meta=dict(desc="Dtype to use for the weights (and activations)."),
            ),
            OrdinalHyperparameter(
                "tensor_parallel_size",
                sequence=[1, 2, 4, 8, 16, 32],
                default_value=1,
                meta=dict(desc="Tensor parallel size to use if the model can not fit on a single GPU."),
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
        Wrap the model to use flash_attn3 where possible.

        Parameters
        ----------
        model : Any
            The model to wrap.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the application of the algorithm.

        Returns
        -------
        Any
            The wrapped model.
        """
        imported_packages = self.import_algorithm_packages()

        @ray.remote(num_gpus=1)
        class BenchmarkWorker:
            def __init__(self, seed: int) -> None:
                torch.set_default_device("cuda")
                imported_packages["vllm_platforms"].current_platform.seed_everything(seed)
                self.seed = seed
                self.device_id = int(ray.get_gpu_ids()[0])

            def tune(
                self,
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
            ) -> dict[str, int]:
                best_config = None
                best_time = float("inf")

                need_device_guard = False

                with torch.cuda.device(self.device_id) if need_device_guard else nullcontext():
                    for config in tqdm_ray(search_space):
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
                                num_iters=20,
                                block_quant_shape=block_quant_shape,
                                use_deep_gemm=use_deep_gemm,
                            )
                        except imported_packages["triton"].runtime.autotuner.OutOfResources:
                            # Some configurations may be invalid and fail to compile.
                            continue

                        if kernel_time < best_time:
                            best_time = kernel_time
                            best_config = config
                now = datetime.now()
                pruna_logger.info(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
                assert best_config is not None
                return best_config

        E = model.num_experts if is_moe_lm(model)or is_transformers_pipeline_with_moe_lm(model) else model.num_experts                # number of experts
        topk = model.num_experts_per_tok                    # number of active experts per token
        intermediate_size = model.intermediate_size # 3072 # FFN intermediate size
        hidden_size = model.hidden_size #4096        # model hidden dim
        assert intermediate_size % smash_config["tensor_parallel_size"] == 0, (
            f"intermediate_size {intermediate_size} is not divisible by tp "
            f"{smash_config['tensor_parallel_size']}."
        )
        shard_intermediate_size = 2 * intermediate_size // smash_config["tensor_parallel_size"]
        dtype = smash_config["compute_dtype"]
        use_fp8_w8a8 = smash_config["weight_dtype"] == "fp8_w8a8"
        use_int8_w8a16 = smash_config["weight_dtype"] == "int8_w8a16"
        FP8_DTYPE = imported_packages["vllm_platforms"].current_platform.fp8_dtype()
        batch_sizes = [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
        ]

        ray.init()
        num_gpus = int(ray.available_resources()["GPU"])
        workers = [BenchmarkWorker.remote(0) for _ in range(num_gpus)]

        is_fp16 = not (use_fp8_w8a8 or use_int8_w8a16)
        search_space = get_configs_compute_bound(is_fp16, None)
        pruna_logger.info(f"Start tuning over {len(search_space)} configurations...")

        start = time.time()
        outputs = []
        worker_idx = 0
        for batch_size in batch_sizes:
            input_args = (
                batch_size,
                E,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a16,
                search_space,
                None,
                False,
            )
            worker = workers[worker_idx]
            worker_method = getattr(worker, "tune")
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        configs = ray.get(outputs)

        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        self.save_configs(
            best_configs,
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a16,
            None,
            args.save_dir,
            imported_packages,
        )
        end = time.time()
        pruna_logger.info(f"Tuning took {end - start:.2f} seconds")

        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe
        import vllm.platforms as vllm_platforms
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEQuantConfig,
            _get_config_dtype_str,
        )
        from vllm.model_executor.layers.fused_moe import override_config
        from vllm.triton_utils import triton

        return dict(
            FusedMoEQuantConfig=FusedMoEQuantConfig,
            _get_config_dtype_str=_get_config_dtype_str,
            FusedMoE=fused_moe,
            vllm_platforms=vllm_platforms,
            triton=triton,
            tqdm_ray=tqdm_ray,
            override_config=override_config,
        )

    def save_configs(
        configs: dict[int, BenchmarkConfig],
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        block_quant_shape: list[int],
        save_dir: str,
        imported_packages: Dict[str, Any],
    ) -> None:
        dtype_str = imported_packages["_get_config_dtype_str"](
            dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8
        )

        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        filename = imported_packages["fused_moe"].get_config_file_name(
            num_experts, shard_intermediate_size // 2, dtype_str, block_quant_shape
        )

        # We want to save at 3 different places:
        # 1. The cache of vllm
        # 2. The cache of kernels hub
        # 3. The smashconfig (to be reused once mode is smashed and saved).


        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, filename)
        pruna_logger.info(f"Writing best config to {filename}...")
        with open(filename, "w") as f:
            json.dump({"triton_version": imported_packages["triton"].__version__, **configs}, f, indent=4)
            f.write("\n")


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int

def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        **(
            {"waves_per_eu": config["waves_per_eu"]} if "waves_per_eu" in config else {}
        ),
        **(
            {"matrix_instr_nonkdim": config["matrix_instr_nonkdim"]}
            if "matrix_instr_nonkdim" in config
            else {}
        ),
        **({"kpack": config["kpack"]} if "kpack" in config else {}),
    }


def get_configs_compute_bound(use_fp16, block_quant_shape) -> list[dict[str, int]]:
    configs: list[BenchmarkConfig] = []

    # Reduced search space for faster tuning.
    block_m_range = [16, 32, 64, 128, 256]
    block_n_range = [32, 64, 128, 256]
    block_k_range = [64, 128, 256]
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

    # Remove configs that are not compatible with fp8 block quantization
    # BLOCK_SIZE_K must be a multiple of block_k
    # BLOCK_SIZE_N must be a multiple of block_n
    if block_quant_shape is not None and not use_fp16:
        block_n, block_k = block_quant_shape[0], block_quant_shape[1]
        for config in configs[:]:
            if (
                config["BLOCK_SIZE_K"] % block_k != 0
                or config["BLOCK_SIZE_N"] % block_n != 0
            ):
                configs.remove(config)
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
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
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
        )
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype
        )
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_deep_gemm:
        # we use the default block shape for deepgemm
        block_quant_shape = [128, 128]
    if use_fp8_w8a8:
        if block_quant_shape:
            block_n, block_k = block_quant_shape[0], block_quant_shape[1]
            E = num_experts
            N = shard_intermediate_size // 2
            K = hidden_size
            factor_for_scale = 1e-2
            n_tiles_w1 = (2 * N + block_n - 1) // block_n
            n_tiles_w2 = (K + block_n - 1) // block_n
            k_tiles_w1 = (K + block_k - 1) // block_k
            k_tiles_w2 = (N + block_k - 1) // block_k
            w1_scale = (
                torch.rand((E, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
                * factor_for_scale
            )
            w2_scale = (
                torch.rand((E, n_tiles_w2, k_tiles_w2), dtype=torch.float32)
                * factor_for_scale
            )
        else:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)

        a1_scale = torch.randn(1, dtype=torch.float32)
        a2_scale = torch.randn(1, dtype=torch.float32)

        w1 = w1.to(imported_packages["vllm_platforms"].current_platform.fp8_dtype())
        w2 = w2.to(imported_packages["vllm_platforms"].current_platform.fp8_dtype())

    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)

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
