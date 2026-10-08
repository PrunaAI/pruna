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

import functools
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

    Flux.2 is cached per block list. ``transformer_blocks`` return ``(encoder_hidden_states, hidden_states)`` and
    ``single_transformer_blocks`` return one concatenated token tensor, and the forward concatenates the two streams
    between those loops. diffusers hooks both lists as one head/tail pair, which subtracts those incompatible tensors.
    Each list with at least two blocks gets its own cache. The Flux.2 block classes are not in diffusers'
    ``TransformerBlockRegistry``; this algorithm registers that metadata before hooking them. ``Flux2Pipeline`` does
    not enter ``cache_context``, so the wrapped forward opens one when the caller has not.
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
        if _is_flux2_transformer(transformer):
            return len(_flux2_block_lists(transformer)) > 0
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
        if _is_flux2_transformer(transformer):
            _apply_flux2_first_block_cache(transformer, cache_config)
            return model
        transformer.enable_cache(cache_config)
        _drop_stale_cache_context_registry(transformer)
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


def _is_flux2_transformer(transformer: torch.nn.Module) -> bool:
    """
    Return whether this module is a Flux.2 diffusion transformer.

    Parameters
    ----------
    transformer : torch.nn.Module
        The module stored on ``pipeline.transformer``.

    Returns
    -------
    bool
        True when the class name starts with ``Flux2``.
    """
    return type(transformer).__name__.startswith("Flux2")


def _flux2_block_lists(transformer: torch.nn.Module) -> list[str]:
    """
    Return Flux.2 block lists that have a head block and a tail block.

    Parameters
    ----------
    transformer : torch.nn.Module
        A Flux.2 transformer.

    Returns
    -------
    list[str]
        ``transformer_blocks`` and ``single_transformer_blocks``, when each contains at least two Flux.2 blocks.
    """
    names: list[str] = []
    for name in ("transformer_blocks", "single_transformer_blocks"):
        blocks = getattr(transformer, name, None)
        if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) < 2:
            continue
        if all(type(block).__name__.startswith("Flux2") for block in blocks):
            names.append(name)
    return names


def _register_flux2_block_metadata() -> None:
    """
    Register Flux.2 blocks with diffusers' cache registry when the installed release has not.

    Double-stream blocks return ``(encoder_hidden_states, hidden_states)``. Single-stream blocks return the
    concatenated token tensor. A release that already registered the class is left unchanged.
    """
    from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2SingleTransformerBlock,
        Flux2TransformerBlock,
    )

    double_metadata = TransformerBlockMetadata(
        return_hidden_states_index=1,
        return_encoder_hidden_states_index=0,
    )
    # Single blocks return one tensor. Leaving the encoder index at its default marks that.
    single_metadata = TransformerBlockMetadata(return_hidden_states_index=0)
    for block_cls, metadata in (
        (Flux2TransformerBlock, double_metadata),
        (Flux2SingleTransformerBlock, single_metadata),
    ):
        try:
            TransformerBlockRegistry.get(block_cls)
        except ValueError:
            TransformerBlockRegistry.register(block_cls, metadata)


def _apply_flux2_first_block_cache(transformer: torch.nn.Module, config: Any) -> None:
    """
    Hook first-block cache on each Flux.2 block list.

    Parameters
    ----------
    transformer : torch.nn.Module
        The Flux.2 transformer.
    config : Any
        A ``FirstBlockCacheConfig``.
    """
    from diffusers.hooks.first_block_cache import apply_first_block_cache

    if getattr(transformer, "is_cache_enabled", False):
        raise ValueError("Caching has already been enabled on this Flux.2 transformer.")

    _register_flux2_block_metadata()
    hooked = False
    for name in _flux2_block_lists(transformer):
        blocks = getattr(transformer, name)
        # apply_first_block_cache only sees direct children. Park the list on a holder so this call
        # cannot pair a double-stream head with a single-stream tail, then put the list back.
        holder = torch.nn.Module()
        holder.add_module(name, blocks)
        apply_first_block_cache(holder, config)
        transformer.add_module(name, blocks)
        hooked = True
    if not hooked:
        raise ValueError("Flux.2 transformer has no block list with at least two blocks.")

    transformer._cache_config = config
    _drop_stale_cache_context_registry(transformer)
    _wrap_forward_with_cache_context(transformer)


def _drop_stale_cache_context_registry(transformer: torch.nn.Module) -> None:
    """
    Forget a child-hook list recorded before these cache hooks existed.

    Parameters
    ----------
    transformer : torch.nn.Module
        The transformer whose ``cache_context`` walks child hooks.
    """
    registry = getattr(transformer, "_diffusers_hook", None)
    if registry is not None and hasattr(registry, "_child_registries_cache"):
        registry._child_registries_cache = None


def _fbc_context_is_active(transformer: torch.nn.Module) -> bool:
    """
    Return whether a caller already opened ``cache_context``.

    Parameters
    ----------
    transformer : torch.nn.Module
        The transformer that owns the cache hooks.

    Returns
    -------
    bool
        True when a hook state manager has a current context name.
    """
    for module in transformer.modules():
        registry = getattr(module, "_diffusers_hook", None)
        if registry is None:
            continue
        for hook in registry.hooks.values():
            manager = getattr(hook, "state_manager", None)
            if manager is not None and manager._current_context is not None:
                return True
    return False


def _wrap_forward_with_cache_context(transformer: torch.nn.Module) -> None:
    """
    Open a cache context for pipelines that call the transformer without one.

    Klein already enters ``cond`` and ``uncond`` contexts. Those calls are left alone. ``Flux2Pipeline`` does not,
    and the hooks raise ``ValueError`` until a context exists.

    Parameters
    ----------
    transformer : torch.nn.Module
        The Flux.2 transformer whose ``forward`` is wrapped.
    """
    if getattr(transformer, "_pruna_fbc_context_wrapped", False):
        return
    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def forward(*args: Any, **kwargs: Any) -> Any:
        if _fbc_context_is_active(transformer):
            return original_forward(*args, **kwargs)
        with transformer.cache_context("default"):
            return original_forward(*args, **kwargs)

    transformer.forward = forward
    transformer._pruna_fbc_context_wrapped = True
