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
import re
from pathlib import Path
from typing import Any, Dict

import torch
from ConfigSpace import OrdinalHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger


class IPEXLLM(PrunaAlgorithmBase):
    """
    Implement IPEX LLM compilation using the intel library.

    This compiler leverages advanced graph optimizations, quantization, and kernel fusion techniques to accelerate
    PyTorch-based LLM inference on Intel CPUs.

    Note: After compilation, the model supports sequence lengths that are either ≤ 32, or even numbers.
    """

    algorithm_name: str = "ipex_llm"
    group_tags: list[AlgorithmTag] = [AlgorithmTag.COMPILER]
    references: dict[str, str] = {"Github": "https://github.com/intel/intel-extension-for-pytorch"}
    tokenizer_required: bool = False
    processor_required: bool = False
    dataset_required: bool = False
    save_fn = SAVE_FUNCTIONS.save_before_apply
    runs_on: list[str] = ["cpu"]
    compatible_before: list[str] = ["half"]
    required_install = (
        "``pip install pruna[intel]`` "
        "``--extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/cn/``"
    )

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for IPEX LLM compilation.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            OrdinalHyperparameter(
                "weight_bits",
                sequence=[8, 4],
                default_value=8,
                meta=dict(desc="The number of bits to use for weight quantization."),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is compatible with IPEX LLM compilation.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            Whether the model is compatible with IPEX LLM compilation.
        """
        imported_modules = self.import_algorithm_packages()
        # Find the installation path of ipex
        ipex_path = Path(imported_modules["ipex"].__file__).parent
        # Try to find the models.py file
        transformers_path = ipex_path / "transformers"
        # Find the full path of models.py if it exists
        models_path = transformers_path / "models" / "reference" / "models.py"
        if models_path.exists():
            # Read the function names from the file
            with open(models_path, "r") as f:
                content = f.read()
                # Simple regex to find function definitions
                funcs = [f for f in re.findall(r"def\s+([A-Z][a-zA-Z0-9_]*)\s*\(", content) if f.endswith("_forward")]
                compatible_list = [name.replace("_forward", "") for name in funcs]
                return model.__class__.__name__ in compatible_list
        else:
            pruna_logger.warning("IPEX models.py file not found. Please check if IPEX is installed correctly.")
            return False

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Compile the model with IPEX LLM.

        Parameters
        ----------
        model : Any
            The model to compile.
        smash_config : SmashConfigPrefixWrapper
            The configuration to use for compilation.

        Returns
        -------
        Any
            The compiled model.
        """
        imported_modules = self.import_algorithm_packages()
        ipex = imported_modules["ipex"]
        woq_weight_dtype = imported_modules["WoqWeightDtype"]

        weight_dtype = woq_weight_dtype.INT8 if smash_config["weight_bits"] == 8 else woq_weight_dtype.INT4

        lowp_mode = ipex.quantization.WoqLowpMode.INT8

        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(weight_dtype=weight_dtype, lowp_mode=lowp_mode)

        model = ipex.llm.optimize(
            model.eval(),
            dtype=getattr(torch, "float32"),
            quantization_config=qconfig,
            low_precision_checkpoint=None,
            deployment_mode=True,
            inplace=True,
        )

        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        # Import necessary modules here to avoid unnecessary imports and ensure they're available when needed
        import intel_extension_for_pytorch as ipex
        from intel_extension_for_pytorch.quantization import WoqWeightDtype

        return dict(
            ipex=ipex,
            WoqWeightDtype=WoqWeightDtype,
        )
