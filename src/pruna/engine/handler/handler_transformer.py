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

from typing import Any, Dict, List, Optional, Tuple

import torch

from pruna.engine.handler.handler_inference import InferenceHandler
from pruna.logging.logger import pruna_logger


class TransformerHandler(InferenceHandler):
    """
    Handle inference arguments, inputs and outputs for transformer models.

    The first element of the batch is passed as input to the model.
    The generated outputs are expected to have .logits attribute.

    Parameters
    ----------
    model_args : Dict[str, Any]
        The arguments to pass to the model.
    """

    def __init__(self, model_args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args = model_args if model_args else {}

    def prepare_inputs(
        self, batch: List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor | dict[str, Any], ...] | dict[str, Any]
    ) -> Any:
        """
        Prepare the inputs for the model.

        Parameters
        ----------
        batch : List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor | dict[str, Any], ...] | dict[str, Any]
            The batch to prepare the inputs for.

        Returns
        -------
        Any
            The prepared inputs.
        """
        x, _ = batch
        return x

    def process_output(self, output: Any) -> Any:
        """
        Handle the output of the model.

        Parameters
        ----------
        output : Any
            The output to process.

        Returns
        -------
        Any
            The processed output.
        """
        return output.logits.float()

    def log_model_info(self) -> None:
        """Log information about the inference handler."""
        pruna_logger.info(
            "Detected transformers model. Using TransformerHandler.\n"
            "- The first element of the batch is passed as input.\n"
            "- The generated outputs are expected to have .logits attribute."
        )
