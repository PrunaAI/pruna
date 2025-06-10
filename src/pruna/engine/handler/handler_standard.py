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


class StandardHandler(InferenceHandler):
    """
    Handle inference arguments, inputs and outputs for unhandled model types.

    Standard handler expectations:
    - The model should accept 'x' as input, where 'x' is the first element of a two-element data batch.
    - Invoke the model using `model(x)` without additional parameters.
    - Outputs should be directly processable without further modification.

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
        return output

    def log_model_info(self) -> None:
        """Log information about the inference handler."""
        pruna_logger.warning("No handler found for model. Using standard handler.")
        pruna_logger.warning(
            "Standard handler expectations:\n"
            "- The model should accept 'x' as input, where 'x' is the first element of a two-element data batch.\n"
            "- Invoke the model using `model(x)` without additional parameters.\n"
            "- Outputs should be directly processable without further modification.\n\n"
            "Please ensure your model adheres to these conventions to maintain compatibility with the standard handler."
        )
