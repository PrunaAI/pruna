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

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import torch

from pruna.data.utils import move_batch_to_device
from pruna.engine.utils import set_to_best_available_device


class InferenceHandler(ABC):
    """
    Abstract base class for inference handlers.

    The inference handler is responsible for handling the inference arguments, inputs and outputs for a given model.
    """

    inference_function_name: str = "__call__"  # Name of the function to call for inference

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
        inputs: List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor, ...],
        device: torch.device | str,
    ) -> List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor, ...]:
        """
        Recursively move inputs to device.

        Parameters
        ----------
        inputs : List[str] | torch.Tensor
            The inputs to prepare.
        device : torch.device | str
            The device to move the inputs to.

        Returns
        -------
        List[str] | torch.Tensor
            The prepared inputs.
        """
        if device == "accelerate":
            device = set_to_best_available_device(None)
        # Using the utility function from the data module
        try:
            return move_batch_to_device(inputs, device)
        except torch.cuda.OutOfMemoryError as e:
            raise e

    def configure_seed(self, seed_strategy: Literal["per_sample", "no_seed"], global_seed: int | None) -> None:
        """
        Set the random seed according to the chosen strategy.

        - If `seed_strategy="per_sample"`,the `global_seed` is used as a base to derive a different seed for each
        sample. This ensures reproducibility while still producing variation across samples,
        making it the preferred option for benchmarking.
        - If `seed_strategy="no_seed"`, no seed is set internally.
        The user is responsible for managing seeds if reproducibility is required.

        Parameters
        ----------
        seed_strategy : Literal["per_sample", "no_seed"]
            The seeding strategy to apply.
        global_seed : int | None
            The base seed value to use (if applicable).
        """
        self.seed_strategy = seed_strategy
        validate_seed_strategy(seed_strategy, global_seed)
        if global_seed is not None:
            self.global_seed = global_seed
            set_seed(global_seed)
        else:
            remove_seed()


def set_seed(seed: int) -> None:
    """
    Set the random seed for the current process.

    Parameters
    ----------
    seed : int
        The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_seed() -> None:
    """Remove the seed from the current process."""
    random.seed(None)
    np.random.seed(None)
    torch.manual_seed(torch.seed())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch.seed())


def validate_seed_strategy(seed_strategy: Literal["per_sample", "no_seed"], global_seed: int | None) -> None:
    """
    Check the consistency of the seed strategy and the global seed.

    If the seed strategy is "no_seed", the global seed must be None.
    If the seed strategy is or "per_sample", the user must provide a global seed.

    Parameters
    ----------
    seed_strategy : Literal["per_sample", "no_seed"]
        The seeding strategy to apply.
    global_seed : int | None
        The base seed value to use (if applicable).
    """
    if seed_strategy != "no_seed" and global_seed is None:
        raise ValueError("Global seed must be provided if seed strategy is not 'no_seed'.")
    elif global_seed is not None and seed_strategy == "no_seed":
        raise ValueError("Seed strategy cannot be 'no_seed' if global seed is provided.")
