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
        inputs: List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor, ...] | Dict[str, Any],
        device: torch.device | str = "cuda",
    ) -> List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor | Dict[str, Any], ...] | Dict[str, Any]:
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
        if isinstance(inputs, dict):
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(device)
        if isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        elif isinstance(inputs, tuple):
            return tuple(self.move_inputs_to_device(item, device) for item in inputs)  # type: ignore
        else:
            return inputs

    def set_correct_dtype(
        self, batch: List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor, ...] | dict, dtype: torch.dtype
    ) -> List[str] | torch.Tensor | Tuple[List[str] | torch.Tensor, ...] | dict:
        """Set the correct dtype for the batch."""
        if isinstance(batch, torch.Tensor):
            return batch.to(dtype)

        if isinstance(batch, list) and all(isinstance(x, str) for x in batch):
            return batch  # don't touch list of strings

        if isinstance(batch, tuple):
            return tuple(self.set_correct_dtype(item, dtype) for item in batch)  # type: ignore

        if isinstance(batch, dict):
            return {
                k: self.set_correct_dtype(v, dtype) if isinstance(v, (torch.Tensor, list, tuple, dict)) else v
                for k, v in batch.items()
            }
        return batch

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
