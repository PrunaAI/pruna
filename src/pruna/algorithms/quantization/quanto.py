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

from typing import Any, Dict, List, Tuple

import torch
from ConfigSpace import Constant, OrdinalHyperparameter

from pruna import SmashConfig
from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.hyperparameters import TARGET_MODULES_TYPE, Boolean, TargetModules
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.data.utils import wrap_batch_for_model_call
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.utils import get_nn_modules
from pruna.logging.logger import pruna_logger


class QuantoQuantizer(PrunaQuantizer):
    """
    Implement Quanto using huggingface optimum-quanto.

    With Quanto, models with int8/float8 weights and float8 activations maintain nearly full-precision accuracy.
    Lower bit quantization is also supported.
    When only weights are quantized and optimized kernels are available, inference latency remains comparable,
    and device memory usage is roughly reduced in proportion to the bitwidth ratio.
    """

    algorithm_name: str = "quanto"
    references: dict[str, str] = {"GitHub": "https://github.com/huggingface/optimum-quanto"}
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.pickled
    tokenizer_required: bool = False
    processor_required: bool = False
    dataset_required: bool = False
    runs_on: list[str] = ["cuda"]
    compatible_algorithms: dict[str, list[str]] = dict(factorizer=["qkv_diffusers"], cacher=["deepcache"])

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
                    desc=f"Precise choices of which modules to quantize. "
                    f"See {TargetModules.path_to_target_modules_documentation} for more details."
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

    def get_default_hyperparameters(self, model: Any, smash_config: SmashConfig) -> Tuple[TARGET_MODULES_TYPE]:
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
        Tuple[TARGET_MODULES_TYPE]
            The default target_modules for the algorithm.
        """
        include: list[str]
        if hasattr(model, "unet"):
            include = ["unet*"]
        elif hasattr(model, "transformer"):
            include = ["transformer*"]
        else:
            include = ["*"]
        return ({"include": include},)

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
        target_modules = smash_config["target_modules"] or self.get_default_hyperparameters(model, smash_config)[0]

        weights = getattr(imported_modules["optimum"].quanto, smash_config["weight_bits"])
        if smash_config["act_bits"] is not None:
            activations = getattr(imported_modules["optimum"].quanto, smash_config["act_bits"])
        else:
            activations = None

        modules_with_subpaths = self._get_modules_with_subpaths(model, target_modules)
        for module, subpaths in modules_with_subpaths:
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

        if smash_config["calibrate"]:
            if smash_config.tokenizer is not None and smash_config.data is not None:
                if hasattr(model, "unet"):
                    working_model = model.unet
                elif hasattr(model, "transformer"):
                    working_model = model.transformer
                else:
                    working_model = model

                try:
                    with imported_modules["Calibration"](streamline=True, debug=False):
                        calibrate(
                            working_model,
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

        for module, _ in modules_with_subpaths:
            try:
                imported_modules["freeze"](module)
            except Exception as e:
                pruna_logger.error("Error while freezing the module: %s", e)
                raise
        return model

    def _get_modules_with_subpaths(
        self, model: Any, target_modules: TARGET_MODULES_TYPE
    ) -> List[Tuple[torch.nn.Module, List[str]]]:
        """Get torch modules within the model and their associated subpaths."""
        target_modules_paths = TargetModules.to_list_of_modules_paths(target_modules, model)
        modules_with_subpaths: List[Tuple[torch.nn.Module, List[str]]] = []
        for root_name, module in get_nn_modules(model).items():
            targeted_submodules = [path for path in target_modules_paths if path.startswith(f"{root_name}.")]
            if root_name:
                targeted_submodules = [path.removeprefix(f"{root_name}.") for path in targeted_submodules]
            if targeted_submodules:
                modules_with_subpaths.append((module, targeted_submodules))
        return modules_with_subpaths

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        import optimum
        from optimum.quanto import Calibration, freeze, quantize

        return dict(Calibration=Calibration, freeze=freeze, quantize=quantize, optimum=optimum)


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
