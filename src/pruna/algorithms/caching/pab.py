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


class PABCacher(PrunaCacher):
    """
    Implement PAB.

    Pyramid Attention Broadcast (PAB) is a method that speeds up inference in diffusion models by systematically skipping
    attention computations between successive inference steps and reusing cached attention states. This implementation
    reduces the number of tunable parameters by setting pipeline specific parameters according to https://github.com/huggingface/diffusers/pull/9562.
    """

    algorithm_name: str = "pab"
    references: dict[str, str] = {
        "Paper": "https://arxiv.org/abs/2408.12588",
        "HuggingFace": "https://huggingface.co/docs/diffusers/main/api/cache#pyramid-attention-broadcast",
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
            )
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
        Apply the PAB algorithm to the model.

        Parameters
        ----------
        model : Any
            The model to apply the PAB algorithm to.
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
        cross_attention_block_skip_range: Optional[int] = None
        spatial_attention_timestep_skip_range: Tuple[int, int] = (100, 800)
        temporal_attention_timestep_skip_range: Tuple[int, int] = (100, 800)
        cross_attention_timestep_skip_range: Tuple[int, int] = (100, 800)
        spatial_attention_block_identifiers: Tuple[str, ...] = (
            "blocks",
            "transformer_blocks",
        )
        temporal_attention_block_identifiers: Tuple[str, ...] = ("temporal_transformer_blocks",)
        cross_attention_block_identifiers: Tuple[str, ...] = (
            "blocks",
            "transformer_blocks",
        )

        # set configs according to https://github.com/huggingface/diffusers/pull/9562
        if is_allegro_pipeline(model):
            cross_attention_block_skip_range = 6
            spatial_attention_timestep_skip_range = (100, 700)
            cross_attention_block_identifiers = ("transformer_blocks",)
        elif is_cogvideo_pipeline(model):
            spatial_attention_block_identifiers = ("transformer_blocks",)
        elif is_flux_pipeline(model):
            spatial_attention_timestep_skip_range = (100, 950)
            spatial_attention_block_identifiers = (
                "transformer_blocks",
                "single_transformer_blocks",
            )
        elif is_hunyuan_pipeline(model):
            spatial_attention_block_identifiers = (
                "transformer_blocks",
                "single_transformer_blocks",
            )
        elif is_latte_pipeline(model):
            temporal_attention_block_skip_range = None
            cross_attention_block_skip_range = None
            spatial_attention_timestep_skip_range = (100, 700)
            spatial_attention_block_identifiers = ("transformer_blocks",)
            cross_attention_block_identifiers = ("transformer_blocks",)
        elif is_mochi_pipeline(model):
            spatial_attention_timestep_skip_range = (400, 987)
            spatial_attention_block_identifiers = ("transformer_blocks",)
        elif is_wan_pipeline(model):
            spatial_attention_block_identifiers = ("blocks",)

        pab_config = imported_modules["pab_config"](
            spatial_attention_block_skip_range=smash_config["interval"],
            temporal_attention_block_skip_range=temporal_attention_block_skip_range,
            cross_attention_block_skip_range=cross_attention_block_skip_range,
            spatial_attention_timestep_skip_range=spatial_attention_timestep_skip_range,
            temporal_attention_timestep_skip_range=temporal_attention_timestep_skip_range,
            cross_attention_timestep_skip_range=cross_attention_timestep_skip_range,
            spatial_attention_block_identifiers=spatial_attention_block_identifiers,
            temporal_attention_block_identifiers=temporal_attention_block_identifiers,
            cross_attention_block_identifiers=cross_attention_block_identifiers,
            current_timestep_callback=lambda: model.current_timestep,
        )
        model.transformer.enable_cache(pab_config)
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        from diffusers import PyramidAttentionBroadcastConfig

        return dict(pab_config=PyramidAttentionBroadcastConfig)
