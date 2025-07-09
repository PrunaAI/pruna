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

from typing import Any, Dict, Optional, Tuple

from ConfigSpace import OrdinalHyperparameter

from pruna.algorithms.caching import PrunaCacher
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import (
    is_allegro_pipeline,
    is_cogvideo_pipeline,
    is_flux_pipeline,
    is_hunyuan_pipeline,
    is_latte_pipeline,
    is_mochi_pipeline,
    is_wan_pipeline,
)


class FasterCacheCacher(PrunaCacher):
    """
    Implement FasterCache.

    FasterCache is a method that speeds up inference in diffusion transformers by:
    - Reusing attention states between successive inference steps, due to high similarity between them
    - Skipping unconditional branch prediction used in classifier-free guidance by revealing redundancies between
      unconditional and conditional branch outputs for the same timestep, and therefore approximating the unconditional
      branch output using the conditional branch output
    This implementation reduces the number of tunable parameters by setting pipeline specific parameters according to
    https://github.com/huggingface/diffusers/pull/9562.
    """

    algorithm_name: str = "fastercache"
    references: dict[str, str] = {
        "GitHub": "https://github.com/Vchitect/FasterCache",
        "Paper": "https://arxiv.org/abs/2410.19355",
    }
    tokenizer_required: bool = False
    processor_required: bool = False
    dataset_required: bool = False
    runs_on: list[str] = ["cpu", "cuda", "accelerate"]
    compatible_algorithms: dict[str, list[str]] = dict(quantizer=["hqq_diffusers", "diffusers_int8"])

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for the algorithm.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            OrdinalHyperparameter(
                "interval",
                sequence=[1, 2, 3, 4, 5],
                default_value=2,
                meta=dict(
                    desc="Interval at which to cache spatial attention blocks - 1 disables caching."
                    "Higher is faster but might degrade quality."
                ),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a valid model for the algorithm.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a valid model for the algorithm, False otherwise.
        """
        pipeline_check_fns = [
            is_allegro_pipeline,
            is_cogvideo_pipeline,
            is_flux_pipeline,
            is_hunyuan_pipeline,
            is_mochi_pipeline,
            is_wan_pipeline,
        ]
        return any(is_pipeline(model) for is_pipeline in pipeline_check_fns)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Apply the fastercache algorithm to the model.

        Parameters
        ----------
        model : Any
            The model to apply the fastercache algorithm to.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the caching.

        Returns
        -------
        Any
            The smashed model.
        """
        imported_modules = self.import_algorithm_packages()
        # set default values according to https://huggingface.co/docs/diffusers/en/api/cache
        temporal_attention_block_skip_range: Optional[int] = None
        spatial_attention_timestep_skip_range: Tuple[int, int] = (-1, 681)
        temporal_attention_timestep_skip_range: Optional[Tuple[int, int]] = None
        low_frequency_weight_update_timestep_range: Tuple[int, int] = (99, 901)
        high_frequency_weight_update_timestep_range: Tuple[int, int] = (-1, 301)
        unconditional_batch_skip_range: int = 5
        unconditional_batch_timestep_skip_range: Tuple[int, int] = (-1, 641)
        spatial_attention_block_identifiers: Tuple[str, ...] = (
            "blocks.*attn1",
            "transformer_blocks.*attn1",
            "single_transformer_blocks.*attn1",
        )
        temporal_attention_block_identifiers: Tuple[str, ...] = ("temporal_transformer_blocks.*attn1",)
        attention_weight_callback = lambda _: 0.5  # noqa: E731
        tensor_format: str = "BFCHW"
        is_guidance_distilled: bool = False

        # set configs according to https://github.com/huggingface/diffusers/pull/9562
        if is_allegro_pipeline(model):
            low_frequency_weight_update_timestep_range = (99, 641)
            spatial_attention_block_identifiers = ("transformer_blocks",)
        elif is_cogvideo_pipeline(model):
            low_frequency_weight_update_timestep_range = (99, 641)
            spatial_attention_block_identifiers = ("transformer_blocks",)
            attention_weight_callback = lambda _: 0.3  # noqa: E731
        elif is_flux_pipeline(model):
            spatial_attention_timestep_skip_range = (-1, 961)
            spatial_attention_block_identifiers = (
                "transformer_blocks",
                "single_transformer_blocks",
            )
            tensor_format = "BCHW"
            is_guidance_distilled = True
        elif is_hunyuan_pipeline(model):
            spatial_attention_timestep_skip_range = (99, 941)
            spatial_attention_block_identifiers = (
                "transformer_blocks",
                "single_transformer_blocks",
            )
            tensor_format = "BCFHW"
            is_guidance_distilled = True
        elif is_latte_pipeline(model):
            temporal_attention_block_skip_range = 2
            temporal_attention_timestep_skip_range = (-1, 681)
            low_frequency_weight_update_timestep_range = (99, 641)
            spatial_attention_block_identifiers = ("transformer_blocks.*attn1",)
            temporal_attention_block_identifiers = ("temporal_transformer_blocks",)
        elif is_mochi_pipeline(model):
            spatial_attention_timestep_skip_range = (-1, 981)
            low_frequency_weight_update_timestep_range = (301, 961)
            high_frequency_weight_update_timestep_range = (-1, 851)
            unconditional_batch_skip_range = 4
            unconditional_batch_timestep_skip_range = (-1, 975)
            spatial_attention_block_identifiers = ("transformer_blocks",)
            attention_weight_callback = lambda _: 0.6  # noqa: E731
        elif is_wan_pipeline(model):
            spatial_attention_block_identifiers = ("blocks",)
            tensor_format = "BCFHW"
            is_guidance_distilled = True

        fastercache_config = imported_modules["FasterCacheConfig"](
            spatial_attention_block_skip_range=smash_config["interval"],
            temporal_attention_block_skip_range=temporal_attention_block_skip_range,
            spatial_attention_timestep_skip_range=spatial_attention_timestep_skip_range,
            temporal_attention_timestep_skip_range=temporal_attention_timestep_skip_range,
            low_frequency_weight_update_timestep_range=low_frequency_weight_update_timestep_range,
            high_frequency_weight_update_timestep_range=high_frequency_weight_update_timestep_range,
            alpha_low_frequency=1.1,
            alpha_high_frequency=1.1,
            unconditional_batch_skip_range=unconditional_batch_skip_range,
            unconditional_batch_timestep_skip_range=unconditional_batch_timestep_skip_range,
            spatial_attention_block_identifiers=spatial_attention_block_identifiers,
            temporal_attention_block_identifiers=temporal_attention_block_identifiers,
            attention_weight_callback=attention_weight_callback,
            tensor_format=tensor_format,
            current_timestep_callback=lambda: model.current_timestep,
            is_guidance_distilled=is_guidance_distilled,
        )
        model.transformer.enable_cache(fastercache_config)
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        from diffusers import FasterCacheConfig

        return dict(FasterCacheConfig=FasterCacheConfig)
