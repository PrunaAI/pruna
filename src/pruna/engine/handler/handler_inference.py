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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch

from pruna.data.utils import move_batch_to_device


class InferenceHandler(ABC):
    """
    Abstract base class for inference handlers.

    The inference handler is responsible for handling the inference arguments, inputs and outputs for a given model.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the handler."""
        self.model_args: Dict[str, Any] = {}

    @abstractmethod
    def prepare_inputs(self, batch: Any) -> Any:
        """
        Prepare the inputs for the model.

        Parameters
        ----------
        batch : Any
            The batch to prepare the inputs for.

        Returns
        -------
        Any
            The prepared inputs.
        """
        pass

    @abstractmethod
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
        pass

    def move_inputs_to_device(
        self,
        inputs: List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor, ...] | Dict[str, Any],
        device: torch.device | str | dict[str, str],
    ) -> List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor | Dict[str, Any], ...] | Dict[str, Any]:
        """
        Recursively move inputs to device.

        Parameters
        ----------
        inputs : List[str] | torch.Tensor
            The inputs to prepare.
        device : torch.device | str
            The device to move the inputs to.
        device_map : dict[str, str] | None
            The device map to use.

        Returns
        -------
        List[str] | torch.Tensor
            The prepared inputs.
        """
        if device == "accelerate":
            if device_map is None:
                raise ValueError("Device map is required for accelerate device.")
            device = min(device_map.values())  # If we have a device map, we should move to the first device
        # Using the utility function from the data module
        return move_batch_to_device(inputs, device)
