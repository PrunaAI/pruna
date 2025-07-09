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

from pruna.algorithms.factorizing import PrunaFactorizer
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import has_fused_attention_processor, is_diffusers_model
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.utils import ModelContext


class QKVDiffusers(PrunaFactorizer):
    """
    Implement QKV factorizing for diffusers models.

    QKV factorizing fuses the QKV matrices of the denoiser model into a single matrix,
    reducing the number of operations. In the attention layer, we can compute the q, k, v signals
    all at once: the matrix multiplication involve a larger matrix but we compute one operation instead of three.
    """

    algorithm_name: str = "qkv_diffusers"
    references: dict[str, str] = {
        "BFL": "https://github.com/black-forest-labs/flux/",
        "Github": "https://github.com/huggingface/diffusers/pull/9185",
    }
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.save_before_apply
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda", "accelerate"]
    dataset_required: bool = False
    compatible_algorithms: dict[str, list[str]] = dict(
        quantizer=["diffusers_int8", "hqq_diffusers", "quanto", "torchao"],
        cacher=["deepcache", "fora"],
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
        return list()

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a pipeline with unet/transformer denoiser compatible with a fused attention processor.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model has a fused attention processor, False otherwise.
        """
        if is_diffusers_model(model):
            return has_fused_attention_processor(model)
        return False

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Fuse the QKV matrices of the denoiser model.

        Parameters
        ----------
        model : Any
            The model to fuse.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the fusion.

        Returns
        -------
        Any
            The fused model.
        """
        # Use context manager to handle the model vs working_model.
        with ModelContext(model) as (pipeline, working_model, denoiser_type):
            # only this line thanks to https://github.com/huggingface/diffusers/pull/9185
            working_model.fuse_qkv_projections()
            # redefining the working_model breaks links with context manager
            # so we need to re-define the working_model as an attribute of the model.
            pipeline.working_model = working_model

        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide an algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        return dict()
