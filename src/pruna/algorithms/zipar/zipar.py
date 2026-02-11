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

from ConfigSpace import UniformIntegerHyperparameter
from packaging.version import Version
from transformers import __version__ as transformers_version

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag
from pruna.algorithms.zipar.utils import JanusZipARGenerator
from pruna.engine.model_checks import is_janus_llamagen_ar
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger


class ZipAR(PrunaAlgorithmBase):
    """
    Implement ZipAR fast decoding algorithm for Janus LlamaGen AR models.

    ZipAR is a parallel decoding algorithm for accelerating visual autoregressive models.
    Instead of producing the visual tokens one after another, it decodes in parallel different rows of the image.
    """

    algorithm_name: str = "zipar"
    group_tags: list[AlgorithmTag] = [AlgorithmTag.COMPILER]
    save_fn = SAVE_FUNCTIONS.reapply
    references: dict[str, str] = {
        "GitHub": "https://github.com/thisisbillhe/zipar",
        "Paper": "https://arxiv.org/abs/2412.04062",
    }
    runs_on: list[str] = ["cpu", "cuda"]
    tokenizer_required: bool = False
    processor_required: bool = False
    dataset_required: bool = False

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is compatible with ZipAR.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a Janus LlamaGen AR model, False otherwise.
        """
        # Warn when using a transformers version other than 4.54.0 (ZipAR is tested with 4.54.0 due to Janus API)
        if Version(transformers_version) != Version("4.54.0"):
            pruna_logger.warning(
                f"ZipAR is tested with transformers==4.54.0; you have {transformers_version}. "
                "Other versions may be unstable with Janus models. Consider pinning: pip install transformers==4.54.0"
            )
        return is_janus_llamagen_ar(model)

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import necessary algorithm packages.

        Returns
        -------
        dict
            An empty dictionary as no packages are imported in this implementation.
        """
        return dict()

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for the ZipAR fast decoding algorithm.

        Returns
        -------
        list
            A list of consisting of the window_size hyperparameter.
        """
        return [
            UniformIntegerHyperparameter(
                "window_size",
                lower=1,
                upper=24,
                default_value=8,
                meta={
                    "desc": "Number of token columns between two decoding heads. Lower is faster but may affect quality."
                },
            )
        ]

    def _apply(self, model: Any, smash_config: Any) -> Any:
        """
        Apply the ZipAR decoder to the model.

        Parameters
        ----------
        model : Any
            The model to apply the algorithm to.
        smash_config : Any
            The configuration for the algorithm.

        Returns
        -------
        Any
            The model with the ZipAR decoder applied.
        """
        if Version(transformers_version) != Version("4.54.0"):
            pruna_logger.warning(
                f"Applying ZipAR with transformers=={transformers_version}. "
                "ZipAR is tested with transformers==4.54.0; if you encounter errors, "
                "consider pinning: pip install transformers==4.54.0"
            )
        model.zipar_helper = JanusZipARGenerator(model, window_size=smash_config["window_size"])
        model.zipar_helper.enable()
        return model
