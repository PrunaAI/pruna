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

import torch
from ConfigSpace import UniformFloatHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.save import SAVE_FUNCTIONS

# FirstBlockCacheConfig landed in diffusers 0.35.0.
_MIN_DIFFUSERS_VERSION = "0.35.0"


class FirstBlockCache(PrunaAlgorithmBase):
    """
    Cache diffusion-transformer blocks with diffusers' first-block cache.

    First Block Cache compares the residual of the first transformer block with the residual from the previous step.
    When the relative absmean difference is below ``threshold``, the remaining blocks are skipped and the cached tail
    residual is reused. The hook is `FirstBlockCacheConfig` applied through ``transformer.enable_cache``.

    The cacher is not tied to a pipeline family. A model is eligible when its transformer exposes ``enable_cache``,
    has at least two blocks in a diffusers block list (``transformer_blocks``, ``single_transformer_blocks``,
    ``blocks``, ``layers``, and the other names diffusers caches), and every one of those blocks is registered in
    diffusers' ``TransformerBlockRegistry``. On diffusers 0.39 that includes Flux.1, Qwen-Image, LTX-Video (two or
    more layers), Wan, and HunyuanVideo.

    Z-Image's ``ZImageTransformerBlock`` is registered, but ``ZImageTransformer2DModel`` does not inherit
    ``CacheMixin`` and ``ZImagePipeline`` does not enter ``cache_context``. The hooks would raise ``ValueError`` on
    the first forward. The check stays false until the transformer exposes ``enable_cache``.
    """

    algorithm_name: str = "first_block_cache"
    group_tags: list[tags] = [tags.CACHER]
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.reapply
    references: dict[str, str] = {
        "GitHub": "https://github.com/chengzeyi/ParaAttention",
        "HuggingFace": "https://huggingface.co/docs/diffusers/main/en/api/cache",
    }
    tokenizer_required: bool = False
    processor_required: bool = False
    dataset_required: bool = False
    runs_on: list[str] = ["cpu", "cuda", "accelerate"]
    compatible_before: Iterable[str] = [
        "hqq_diffusers",
        "diffusers_int8",
        "sage_attn",
        "hyper",
        "padding_pruning",
        "static_fp8_diffusers",
        "time_aware_fp8_diffusers",
        "moe_kernel_tuner",
    ]
    compatible_after: Iterable[str] = ["img2img_denoise", "realesrgan_upscale", "moe_kernel_tuner"]

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for the algorithm.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            UniformFloatHyperparameter(
                "threshold",
                lower=0.0,
                upper=1.0,
                default_value=0.05,
                meta={
                    "desc": "Residual absmean threshold used to skip the remaining transformer blocks. "
                    "Higher is faster and can reduce quality. 0 recomputes every step."
                },
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model transformer supports diffusers' first-block cache.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the transformer can take a ``FirstBlockCacheConfig``, False otherwise.
        """
        transformer = getattr(model, "transformer", None)
        if transformer is None or not hasattr(transformer, "enable_cache"):
            return False
        try:
            blocks = _cacheable_transformer_blocks(transformer)
        except ImportError:
            return False
        return len(blocks) >= 2

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Apply first-block cache to the model transformer.

        Parameters
        ----------
        model : Any
            The model to apply the algorithm to.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the caching.

        Returns
        -------
        Any
            The smashed model.
        """
        imported_modules = self.import_algorithm_packages()
        cache_config = imported_modules["FirstBlockCacheConfig"](threshold=smash_config["threshold"])
        transformer = model.transformer
        transformer.enable_cache(cache_config)
        # cache_context records child hook registries on first use. A forward that ran before enable_cache
        # leaves that list empty, so the block hooks never receive a context.
        registry = getattr(transformer, "_diffusers_hook", None)
        if registry is not None and hasattr(registry, "_child_registries_cache"):
            registry._child_registries_cache = None
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        try:
            from diffusers import FirstBlockCacheConfig
        except ImportError as error:
            raise ImportError(
                "first_block_cache requires diffusers with FirstBlockCacheConfig "
                f"(diffusers>={_MIN_DIFFUSERS_VERSION})."
            ) from error
        return dict(FirstBlockCacheConfig=FirstBlockCacheConfig)


def _cacheable_transformer_blocks(transformer: torch.nn.Module) -> list[torch.nn.Module]:
    """
    Return transformer blocks that diffusers' first-block cache can hook.

    The block lists and the registry are the same ones ``apply_first_block_cache`` uses. An empty list means the
    installed diffusers cannot cache this architecture, including the case where a block class is not registered.

    Parameters
    ----------
    transformer : torch.nn.Module
        The diffusion transformer to inspect.

    Returns
    -------
    list[torch.nn.Module]
        Blocks in the order first-block cache visits them. Empty when the cache API or a block registration is missing.
    """
    try:
        from diffusers.hooks._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
        from diffusers.hooks._helpers import TransformerBlockRegistry
    except ImportError:
        return []

    blocks: list[torch.nn.Module] = []
    for name, submodule in transformer.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        blocks.extend(submodule)

    for block in blocks:
        try:
            TransformerBlockRegistry.get(type(block))
        except (ValueError, KeyError):
            return []
    return blocks
