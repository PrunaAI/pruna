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
from ConfigSpace import Constant, OrdinalHyperparameter

from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.smash_space import Boolean
from pruna.data.utils import wrap_batch_for_model_call
from pruna.engine.save import SAVE_FUNCTIONS
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

        if hasattr(model, "unet"):
            working_model = model.unet
        elif hasattr(model, "transformer"):
            working_model = model.transformer
        else:
            working_model = model

        weights = getattr(imported_modules["optimum"].quanto, smash_config["weight_bits"])
        if smash_config["act_bits"] is not None:
            activations = getattr(imported_modules["optimum"].quanto, smash_config["act_bits"])
        else:
            activations = None

        try:
            imported_modules["quantize"](working_model, weights=weights, activations=activations)
        except Exception as e:
            pruna_logger.error("Error during quantization: %s", e)
            raise

        if smash_config["calibrate"]:
            if smash_config.tokenizer is not None and smash_config.data is not None:
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

        try:
            imported_modules["freeze"](working_model)
        except Exception as e:
            pruna_logger.error("Error while freezing the model: %s", e)
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
