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
from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter

from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.data.utils import wrap_batch_for_model_call
from pruna.engine.save import SAVE_FUNCTIONS


class TorchStaticQuantizer(PrunaQuantizer):
    """
    Implement static quantization using torch.

    In static quantization, both weights and activations are pre-converted to lower precision (e.g., int8)
    using a calibration process on representative data, which typically yields greater efficiency gains but
    requires additional steps during model preparation.
    """

    algorithm_name = "torch_static"
    references = {"GitHub": "https://github.com/pytorch/pytorch"}
    save_fn = SAVE_FUNCTIONS.pickled
    tokenizer_required = False
    processor_required = False
    run_on_cpu = True
    run_on_cuda = True
    dataset_required = True
    compatible_algorithms = dict()

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
                sequence=["quint8", "qint8"],
                default_value="qint8",
                meta=dict(desc="Tensor type to use for weight quantization."),
            ),
            OrdinalHyperparameter(
                "act_bits",
                sequence=["quint8", "qint8"],
                default_value="qint8",
                meta=dict(desc="Tensor type to use for activation quantization."),
            ),
            CategoricalHyperparameter(
                "qscheme",
                choices=["per_tensor_symmetric", "per_tensor_affine"],
                default_value="per_tensor_affine",
                meta=dict(desc="Quantization scheme to use."),
            ),
            CategoricalHyperparameter(
                "qobserver",
                choices=[
                    "MinMaxObserver",
                    "MovingAverageMinMaxObserver",
                    "PerChannelMinMaxObserver",
                    "HistogramObserver",
                ],
                default_value="MinMaxObserver",
                meta=dict(desc="Observer to use for quantization."),
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
        return isinstance(model, torch.nn.Module)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Quantize the model with torch static quantization.

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
        qtype_weight = getattr(torch, smash_config["weight_bits"])
        qtype_activation = getattr(torch, smash_config["act_bits"])
        qscheme = getattr(torch, smash_config["qscheme"])
        qobserver = getattr(torch.quantization, smash_config["qobserver"])
        model.eval().to("cpu")
        qconfig = torch.quantization.QConfig(
            activation=qobserver.with_args(dtype=qtype_activation, qscheme=qscheme),
            weight=qobserver.with_args(dtype=qtype_weight, qscheme=qscheme),
        )
        quantized_model = QuantWrapper(model, qconfig)
        torch.ao.quantization.prepare(quantized_model, inplace=True)

        # dataloader has been ensured to be set in the config
        for i, batch in enumerate(smash_config.train_dataloader()):  # type: ignore[arg-type]
            if i >= smash_config.calibration_samples:
                break
            wrap_batch_for_model_call(batch, quantized_model, device="cpu")

        torch.ao.quantization.convert(quantized_model, inplace=True)

        return quantized_model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        return dict()


class QuantWrapper(torch.nn.Module):
    """
    A wrapper for quantizing a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to quantize.
    qconfig : torch.quantization.QConfig
        The quantization configuration.
    """

    def __init__(self, model: torch.nn.Module, qconfig: torch.quantization.QConfig) -> None:
        super().__init__()
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.qconfig = qconfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the quantization wrapper.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be quantized and passed through the model.

        Returns
        -------
        torch.Tensor
            Dequantized output tensor from the model.
        """
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
