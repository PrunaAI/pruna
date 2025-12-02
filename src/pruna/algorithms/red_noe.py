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
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ConfigSpace import UniformIntegerHyperparameter
from transformers import AutoModelForCausalLM

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.target_modules import TargetModules
from pruna.engine.model_checks import is_moe_lm, is_transformers_pipeline_with_moe_lm
from pruna.engine.utils import get_device_map, move_to_device, safe_memory_cleanup


class RedNOE(PrunaAlgorithmBase):
    """
    Implement RedNOE for LMs and diffusers pipelines with MoE blocks.

    RedNOE is a method to Reduce the Number Of Experts per token.
    """

    algorithm_name: str = "red_noe"
    group_tags: list[str] = [tags.PRUNER]
    references: dict[str, str] = {}
    tokenizer_required: bool = False
    processor_required: bool = False
    dataset_required: bool = False
    runs_on: list[str] = ["cuda", "accelerate"]
    save_fn: None = None
    compatible_after: Iterable[str] = ["*"]

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            UniformIntegerHyperparameter(
                name="num_experts_per_token",
                lower=1,
                upper=256,
                default_value=2,
                meta=dict(desc="Number of experts triggered per token."),
            ),
            TargetModules(
                name="target_name",
                default_value={"include": ["num_experts_per_tok"], "exclude": []},
                meta=dict(
                    desc="Name of of the parameter in the config.json file to be modified, "
                    "e.g. 'num_experts_per_tok' for mixtral models. "
                ),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a causal language model or a diffusers pipeline with a MoE block.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a MoE LM or a transformers pipeline with a MoE block, False otherwise.
        """
        # Hunyuan3-image is a MoE model, but not depending on mixtral
        if model.__class__.__name__ == "HunyuanImage3ForCausalMM":
            return True
        else:
            return is_moe_lm(model) or is_transformers_pipeline_with_moe_lm(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Reduce the number of experts per token in the config.

        Parameters
        ----------
        model : Any
            The model to reduce the number of experts per token in.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the reduction of the number of experts per token.

        Returns
        -------
        Any
            The model with the reduced number of experts per token.
        """
        device_map = get_device_map(model)
        # we need to save and reload with the new config, because immutable object.
        with tempfile.TemporaryDirectory() as temp_dir:
            move_to_device(model, "cpu")
            model.save_pretrained(temp_dir)
            config_path = Path(temp_dir) / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
            else:
                with config_path.open("r", encoding="utf-8") as f:
                    config_json = json.load(f)
                config_json[smash_config["target_name"]["include"][0]] = smash_config["num_experts_per_token"]
                with config_path.open("w", encoding="utf-8") as f:
                    json.dump(config_json, f, indent=2)
                safe_memory_cleanup()
                model = AutoModelForCausalLM.from_pretrained(temp_dir, device_map=device_map)
        return model
