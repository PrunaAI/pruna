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

from typing import Any, Dict

import torch
from ConfigSpace import Constant, OrdinalHyperparameter

from pruna import SmashConfig
from pruna.algorithms.base.algorithm_tags import Quantizer
from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.target_modules import TARGET_MODULES_TYPE, TargetModules, map_targeted_nn_roots
from pruna.data.utils import wrap_batch_for_model_call
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.utils import get_nn_modules
from pruna.logging.logger import pruna_logger


class Quanto(PrunaAlgorithmBase):
    """
    Implement Quanto using huggingface optimum-quanto.

    With Quanto, models with int8/float8 weights and float8 activations maintain nearly full-precision accuracy.
    Lower bit quantization is also supported.
    When only weights are quantized and optimized kernels are available, inference latency remains comparable,
    and device memory usage is roughly reduced in proportion to the bitwidth ratio.
    """

    algorithm_name: str = "quanto"
    group_tags: list[str] = [Quantizer]
    references: dict[str, str] = {"GitHub": "https://github.com/huggingface/optimum-quanto"}
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.pickled
    tokenizer_required: bool = False
    processor_required: bool = False
    dataset_required: bool = False
    runs_on: list[str] = ["cuda"]
    compatible_before: list[str] = ["qkv_diffusers"]
    compatible_after: list[str] = ["deepcache"]

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            OrdinalHyperparameter(
                "weight_bits",
                sequence=["qint2", "qint4", "qint8", "qfloat8"],
                default_value="qfloat8",
                meta=dict(desc="Tensor type to use for quantization."),
            ),
            Constant("act_bits", value=None),
            Boolean("calibrate", default=True, meta=dict(desc="Whether to calibrate the model.")),
            Constant(name="calibration_samples", value=64),
            TargetModules(
                name="target_modules",
                default_value=None,
                meta=dict(
                    desc="Precise choices of which modules to quantize, "
                    "e.g. {include: ['transformer.*']} to quantize only the transformer in a diffusion pipeline. "
                    f"See the {TargetModules.documentation_name_with_link} documentation for more details."
                ),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is supported.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is supported, False otherwise.
        """
        if isinstance(model, torch.nn.Module):
            return True
        if hasattr(model, "unet") and isinstance(model.unet, torch.nn.Module):
            return True
        return hasattr(model, "transformer") and isinstance(model.transformer, torch.nn.Module)

    def get_model_dependent_hyperparameter_defaults(
        self, model: Any, smash_config: SmashConfig | SmashConfigPrefixWrapper
    ) -> TARGET_MODULES_TYPE:
        """
        Get default values for the target_modules based on the model and configuration.

        Parameters
        ----------
        model : Any
            The model to get the default hyperparameters from.
        smash_config : SmashConfig
            The SmashConfig object.

        Returns
        -------
        TARGET_MODULES_TYPE
            The default target_modules for the algorithm.
        """
        include: list[str]
        if hasattr(model, "unet"):
            include = ["unet*"]
        elif hasattr(model, "transformer"):
            include = ["transformer*"]
        else:
            include = ["*"]
        return {"include": include, "exclude": []}

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Quantize the model with QUANTO.

        Parameters
        ----------
        model : Any
            The model to quantize.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the quantization.

        Returns
        -------
        Any
            The quantized model.
        """
        imported_modules = self.import_algorithm_packages()
        target_modules = smash_config["target_modules"]
        if target_modules is None:
            target_modules = self.get_model_dependent_hyperparameter_defaults(model, smash_config)

        weights = getattr(imported_modules["quanto"], smash_config["weight_bits"])
        activations = (
            getattr(imported_modules["quanto"], smash_config["act_bits"])
            if smash_config["act_bits"] is not None
            else None
        )

        def quantize_nn(attr_name: str | None, module: torch.nn.Module, subpaths: list[str]) -> Any:
            """
            Apply Quanto quantization to a nn.Module.

            Parameters
            ----------
            attr_name : str
                The name of the attribute in the model pointing to the nn.Module to quantize.
            module : torch.nn.Module
                The nn.Module to quantize.
            subpaths : list[str]
                The subpaths of the module to quantize.
            """
            try:
                imported_modules["quantize"](
                    module,
                    weights=weights,
                    activations=activations,
                    include=subpaths,
                )
            except Exception as e:
                pruna_logger.error("Error during quantization: %s", e)
                raise
            return module

        model = map_targeted_nn_roots(quantize_nn, model, target_modules)

        if smash_config["calibrate"]:
            if smash_config.tokenizer is not None and smash_config.data is not None:
                try:
                    with imported_modules["Calibration"](streamline=True, debug=False):
                        calibrate(
                            model,
                            smash_config.val_dataloader(),
                            model.device,  # only e.g. CUDA here is not enough, we need also the correct device index
                            batch_size=smash_config.batch_size,
                            samples=smash_config["calibration_samples"],
                        )
                except Exception as e:
                    pruna_logger.error("Error during calibration: %s", e)
                    raise
            else:
                pruna_logger.error("Calibration requires a tokenizer and dataloader. Skipping calibration.")

        for module in get_nn_modules(model).values():
            try:
                # optimum.quanto.freeze checks whether the module has been quantized by quanto
                # so we can call it on all nn.Module without filtering
                imported_modules["freeze"](module)
            except Exception as e:
                pruna_logger.error("Error while freezing the module: %s", e)
                raise
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        import optimum.quanto as quanto
        from optimum.quanto import Calibration, freeze, quantize

        return dict(Calibration=Calibration, freeze=freeze, quantize=quantize, quanto=quanto)


@torch.no_grad()
def calibrate(
    model: Any,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    batch_size: int,
    samples: int,
) -> None:
    """
    Calibrate the model on a given dataset.

    Parameters
    ----------
    model : Any
        The model to be calibrated, typically a transformer model.
    dataloader : torch.utils.data.DataLoader
        The dataset to iterate over, where each item contains a "text" field.
    device : torch.device
        The device (CPU or GPU) to run the model on.
    batch_size : int
        The number of samples per batch.
    samples : int
        Limits the total number of samples to process.
    """
    model.eval()
    total = 0
    for batch in dataloader:
        wrap_batch_for_model_call(batch, model, device)
        total += batch_size
        if total >= samples:
            break
