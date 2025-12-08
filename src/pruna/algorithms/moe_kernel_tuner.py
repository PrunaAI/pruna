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
from typing import Any, Dict

import ray.experimental.tqdm_ray as tqdm_ray

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_moe_lm, is_transformers_pipeline_with_moe_lm


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

        # TODO: Implement the MoE kernel tuning.
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
        from vllm.transformers_utils.config import get_config
        from vllm.triton_utils import triton
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        return dict(
            FusedMoEQuantConfig=FusedMoEQuantConfig,
            _get_config_dtype_str=_get_config_dtype_str,
            FusedMoE=fused_moe,
            vllm_platforms=vllm_platforms,
            get_config=get_config,
            triton=triton,
            FlexibleArgumentParser=FlexibleArgumentParser,
            tqdm_ray=tqdm_ray,
        )
