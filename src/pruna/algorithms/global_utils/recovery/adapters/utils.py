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

from typing import Any

import torch
from torch import nn

from pruna.engine.utils import get_nn_modules


def freeze_parameters(module: Any) -> None:
    """
    Disable training for all parameters in the given module.

    Parameters
    ----------
    module : torch.nn.Module
        The module to freeze.
    """
    for nn_module in get_nn_modules(module).values():
        for param in nn_module.parameters():
            param.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    """
    Unfreeze all parameters of the given module.

    Parameters
    ----------
    module : torch.nn.Module
        The module to unfreeze.
    """
    for param in module.parameters():
        param.requires_grad = True


def is_trainable(param: torch.nn.Parameter) -> bool:
    """
    Check whether the parameter has been quantized, making it untrainable.

    Parameters
    ----------
    param : torch.nn.Parameter
        The parameter to check for quantization.

    Returns
    -------
    bool
        Whether the parameter has been quantized.
    """
    # Note that quantized weights can be trained by accumulating gradients in a full-precision copy of the model.
    # This function will be updated in the future when adding support for this mechanism.
    return all(
        [
            "float" in str(param.dtype),
            not hasattr(param, "qtype"),  # check quantization with quanto
        ]
    )


def unfreeze_parameters_by_name(module: torch.nn.Module, target_modules: tuple[str]) -> tuple[int, int]:
    """
    Unfreeze the parameters of the given module when their name contains any of the target_modules.

    Parameters
    ----------
    module : torch.nn.Module
        The module containing the parameters to unfreeze.
    target_modules : tuple[str]
        The names of the parameters, or modules containing the parameters to unfreeze.

    Returns
    -------
    int
        The number of parameters that were activated.
    int
        The number of parameters found that match the given name but were not trainable.
    """
    if len(target_modules) == 0:
        return

    activated_parameters, skipped_parameters = 0, 0
    for name, parameter in module.named_parameters():
        matches_name = any(substr in name for substr in target_modules)
        if matches_name and is_trainable(parameter):
            parameter.requires_grad = True
            activated_parameters += int(parameter.numel())
        elif matches_name:  # parameter has been quantized
            skipped_parameters += int(parameter.numel())

    return activated_parameters, skipped_parameters


def unfreeze_submodules_by_type(
    module: torch.nn.Module,
    target_types: tuple[type[nn.Module], ...],
) -> tuple[int, int]:
    """
    Unfreeze the submodules of the given module when they are of the target type.

    Parameters
    ----------
    module : torch.nn.Module
        The module containing the submodules to unfreeze.
    target_types : tuple[type]
        The types identifying which submodules to unfreeze.

    Returns
    -------
    int
        The number of parameters that were activated.
    int
        The number of parameters found that match the given type but were not trainable.
    """
    if len(target_types) == 0:
        return 0, 0

    activated_parameters, skipped_parameters = 0, 0
    for submodule in module.modules():
        if isinstance(submodule, target_types):
            for param in submodule.parameters():
                if is_trainable(param):
                    param.requires_grad = True
                    activated_parameters += int(param.numel())
                else:  # parameter has been quantized
                    skipped_parameters += int(param.numel())

    return activated_parameters, skipped_parameters


def unfreeze_submodules_by_class_name(
    module: torch.nn.Module,
    target_classes: tuple[str, ...],
) -> tuple[int, int]:
    """
    Unfreeze the submodules of the given module when their class name matches one of the target classes.

    Parameters
    ----------
    module : torch.nn.Module
        The module containing the submodules to unfreeze.
    target_classes : tuple[str]
        The class names identifying which submodules to unfreeze.

    Returns
    -------
    int
        The number of parameters that were activated.
    int
        The number of parameters found that match the given type but were not trainable.
    """
    if len(target_classes) == 0:
        return 0, 0

    activated_parameters, skipped_parameters = 0, 0
    for submodule in module.modules():
        if submodule.__class__.__name__ in target_classes:
            for param in submodule.parameters():
                if is_trainable(param):
                    param.requires_grad = True
                    activated_parameters += int(param.numel())
                else:  # parameter has been quantized
                    skipped_parameters += int(param.numel())

    return activated_parameters, skipped_parameters
