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

from ConfigSpace import OrdinalHyperparameter

from pruna.algorithms.caching import PrunaCacher
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_unet_pipeline
from pruna.logging.logger import pruna_logger


class DeepCacheCacher(PrunaCacher):
    """
    Implement DeepCache.

    DeepCache accelerates inference by leveraging the U-Net blocks of diffusion pipelines to reuse high-level
    features.
    """

    algorithm_name = "deepcache"
    references = {"GitHub": "https://github.com/horseee/DeepCache", "Paper": "https://arxiv.org/abs/2312.00858"}
    tokenizer_required = False
    processor_required = False
    dataset_required = False
    run_on_cpu = True
    run_on_cuda = True
    compatible_algorithms = dict(
        compiler=["stable_fast", "torch_compile"],
        quantizer=["half", "hqq_diffusers", "diffusers_int8"],
    )

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
                    desc="Interval at which to cache - 1 disables caching. Higher is faster but might affect quality."
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
        return is_unet_pipeline(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Apply the deepcache algorithm to the model.

        Parameters
        ----------
        model : Any
            The model to apply the deepcache algorithm to.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the caching.

        Returns
        -------
        Any
            The smashed model.
        """
        imported_modules = self.import_algorithm_packages()

        model.function_dict = {}
        model.function_dict["pre_deepcache_call"] = model.__class__.__call__

        # redefines the forward pass of Unet and Unet_blocks to skip some diffusion steps
        deepcache_unet_helper = imported_modules["DeepCacheUnetHelper"](pipe=model)

        deepcache_unet_helper.set_params(cache_interval=smash_config["interval"], cache_branch_id=0)

        deepcache_unet_helper.enable()
        model.deepcache_unet_helper = deepcache_unet_helper
        model.deepcache_unet_helper.function_dicts = [model.deepcache_unet_helper.function_dict]

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
            from DeepCache import DeepCacheSDHelper as DeepCacheUnetHelper
        except ModuleNotFoundError:  # DeepCache is not installed for Pruna on CPU
            pruna_logger.error(
                "You are trying to use DeepCache Compiler, but DeepCache is not installed. "
                "This is likely because you did not install the GPU version of Pruna."
            )
            raise

        return dict(DeepCacheUnetHelper=DeepCacheUnetHelper)
