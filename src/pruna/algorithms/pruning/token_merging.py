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

from typing import Any, Dict

import torch
from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter

from pruna.algorithms.pruning import PrunaPruner
from pruna.config.smash_config import SmashConfigPrefixWrapper


class TokenMergingPruner(PrunaPruner):
    """
    Implement Token Merging (ToMe) for Vision Transformers and similar models.

    Token Merging reduces the number of tokens processed by transformers through bipartite soft matching,
    progressively merging similar tokens across layers. This results in significant speedups (1.5-2x) with
    minimal accuracy degradation (<1%) for vision transformers, CLIP models, and diffusion models.

    The algorithm works by:
    1. Computing similarity between tokens using cosine similarity
    2. Using bipartite matching to find pairs of similar tokens
    3. Merging matched tokens through weighted averaging
    4. Progressively reducing token count across layers
    """

    algorithm_name: str = "token_merging"
    references: dict[str, str] = {
        "GitHub": "https://github.com/facebookresearch/ToMe",
        "Paper": "https://arxiv.org/abs/2210.09461",
    }
    save_fn = None
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda"]
    dataset_required: bool = False
    compatible_algorithms: dict[str, list[str]] = dict(
        quantizer=["half", "hqq", "quanto"],
        compiler=["torch_compile"],
    )

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            UniformFloatHyperparameter(
                "reduction_ratio",
                lower=0.0,
                upper=0.7,
                log=False,
                default_value=0.3,
                meta=dict(
                    desc="Ratio of tokens to merge. 0.3 means reduce tokens by 30%. Higher values = faster but may affect quality."
                ),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is compatible with token merging.

        Token merging works best with Vision Transformers, CLIP models, and similar architectures
        that process tokens through attention layers.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a transformer-based model, False otherwise.
        """
        # Check if model is a torch module
        if not isinstance(model, torch.nn.Module):
            return False

        # Check for common transformer architectures
        model_class_name = model.__class__.__name__.lower()
        transformer_keywords = [
            "vit",
            "vision_transformer",
            "deit",
            "clip",
            "swin",
            "beit",
            "transformer",
        ]

        return any(keyword in model_class_name for keyword in transformer_keywords)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Apply token merging to the model.

        This injects token merging hooks into the model's attention layers, which will
        progressively merge tokens during forward passes.

        Parameters
        ----------
        model : Any
            The model to apply token merging to
        smash_config : SmashConfigPrefixWrapper
            Configuration object containing token merging hyperparameters

        Returns
        -------
        Any
            The model with token merging applied

        Raises
        ------
        ValueError
            If the reduction ratio is out of valid range
        RuntimeError
            If token merging fails to apply to the model
        """
        from pruna.algorithms.pruning.token_merging_utils import apply_tome_to_vit

        # Get and validate configuration
        reduction_ratio = smash_config["reduction_ratio"]

        if not 0.0 <= reduction_ratio <= 0.7:
            raise ValueError(
                f"reduction_ratio must be between 0.0 and 0.7, got {reduction_ratio}. "
                "Higher values may cause instability."
            )

        # Convert to percentage (e.g., 0.3 -> 30)
        r = int(reduction_ratio * 100)

        # Apply ToMe to the model
        try:
            model = apply_tome_to_vit(model, r=r)
        except Exception as e:
            raise RuntimeError(
                f"Failed to apply token merging to {model.__class__.__name__}. "
                "Ensure the model is a Vision Transformer or compatible architecture."
            ) from e

        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        # No external packages needed - we have our own implementation
        return dict()
