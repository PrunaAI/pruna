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

from functools import wraps
from typing import Any, Dict, List, cast

import torch

from pruna.engine.pruna_model import PrunaModel


class CallSequenceTracker:
    """
    A utility class that tracks the execution sequence and input specs of PyTorch modules within a non-PyTorch model.

    This class wraps the forward methods of PyTorch modules to collect detailed information about:
    - The order in which modules are called during model execution
    - Input tensor characteristics for each module call, including:
        - Shapes
        - Data types (dtypes)
        - Device placement (CPU/GPU)

    The tracker works with both positional arguments and keyword arguments passed to the modules.
    It can be used for debugging, optimization, or analysis of model execution patterns.

    Examples
    --------
    >>> tracker = CallSequenceTracker()
    >>> tracker.register_wrappers(model)
    >>> output = model(input_tensor)
    >>> call_sequence = tracker.get_call_sequence()
    >>> tracker.clean_call_sequence()  # Reset tracker state

    Attributes
    ----------
    call_sequence : List[Dict[str, Any]]
        A list containing dictionaries with information about each module call:
        - module: The PyTorch module instance
        - module_name: Name of the module class
        - inputs: List of dictionaries containing shape, dtype, and device info for positional args
        - kwargs: Dictionary containing shape, dtype, and device info for keyword args
    """

    def __init__(self) -> None:
        self.call_sequence: List[Dict[str, Any]] = []

    def wrap_forward(self, module: torch.nn.Module) -> None:
        """
        Wrap the forward method of the module, gathering information about the call sequence and inputs.

        Parameters
        ----------
        module : torch.nn.Module
            The module to wrap.
        """
        original_forward = module.forward

        @wraps(original_forward)
        def wrapped_forward(*args, **kwargs) -> Any:
            module_name = module.__class__.__name__
            input_info = [
                {
                    "shape": arg.shape if isinstance(arg, torch.Tensor) else None,
                    "dtype": arg.dtype if isinstance(arg, torch.Tensor) else None,
                    "device": arg.device if isinstance(arg, torch.Tensor) else None,
                }
                for arg in args
            ]

            kwargs_info = {
                k: {
                    "shape": v.shape if isinstance(v, torch.Tensor) else None,
                    "dtype": v.dtype if isinstance(v, torch.Tensor) else None,
                    "device": v.device if isinstance(v, torch.Tensor) else None,
                }
                for k, v in kwargs.items()
            }

            self.call_sequence.append(
                {"module": module, "module_name": module_name, "inputs": input_info, "kwargs": kwargs_info}
            )

            return original_forward(*args, **kwargs)

        module.forward = wrapped_forward

    def wrap(self, model: PrunaModel) -> None:
        """
        Wrap the nn modules of the model.

        Parameters
        ----------
        model : PrunaModel
            The model that will be wrapped.
        """
        for module in model.get_nn_modules().values():
            self.wrap_forward(module)

    def unwrap(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Unwrap the module.

        Parameters
        ----------
        module : torch.nn.Module
            The module to unwrap.

        Returns
        -------
        torch.nn.Module
            The unwrapped module.
        """
        if hasattr(module.forward, "__wrapped__"):
            module.forward = cast(Any, getattr(module.forward, "__wrapped__"))
        return module

    def get_call_sequence(self) -> List[Dict[str, Any]]:
        """
        Get the call sequence.

        Returns
        -------
        List[Dict[str, Any]]
            The call sequence.
        """
        return self.call_sequence

    def clean_call_sequence(self) -> None:
        """Clean the call sequence."""
        self.call_sequence.clear()
