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

import fnmatch
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
from typing_extensions import override

from pruna.config.hyperparameters import UnconstrainedHyperparameter
from pruna.engine.utils import get_nn_modules

TARGET_MODULES_TYPE = Dict[Literal["include", "exclude"], List[str]]


class TargetModules(UnconstrainedHyperparameter):
    """
    Represents a target modules hyperparameter, used to select modules based on include and exclude patterns.

    Parameters
    ----------
    name : str
        The name of the hyperparameter.
    default_value : Optional[TARGET_MODULES_TYPE]
        The default value of the hyperparameter.
    meta : Any
        Meta data describing the hyperparameter.
    """

    documentation_name_with_link = ":ref:`Target Modules <target_modules>`"

    def __init__(self, name: str, default_value: Optional[TARGET_MODULES_TYPE] = None, meta: Any = None) -> None:
        super().__init__(name, default_value, meta=meta)

    @override
    def legal_value(self, value: TARGET_MODULES_TYPE | None):  # type: ignore[override]  # numpydoc ignore=GL08
        """
        Check if a value is a valid target modules of type TARGET_MODULES_TYPE.

        Parameters
        ----------
        value : Any
            The value to check.

        Returns
        -------
        bool or numpy.ndarray
            `True` if the value is of type TARGET_MODULES_TYPE, `False` otherwise.
        """
        # ensure the value is a TARGET_MODULES_TYPE to make errors more explicit for the user
        if value is None:
            pass
        elif not isinstance(value, dict):
            raise TypeError(f"Target modules must be a dictionary with keys 'include' and/or 'exclude'. Got: {value}")
        elif any(key not in ["include", "exclude"] for key in value):
            raise ValueError(f"Target modules must only use keys 'include' and/or 'exclude'. Got: {list(value.keys())}")
        elif any(not isinstance(patterns, list) for patterns in value.values()):
            raise TypeError(
                f"Target modules must be a dictionary with lists of fnmatch patterns as values. Got: {value}"
            )
        else:
            include_patterns = value.get("include", [])
            exclude_patterns = value.get("exclude", [])
            all_patterns = include_patterns + exclude_patterns
            unrecognized_patterns = [pattern for pattern in all_patterns if not isinstance(pattern, str)]
            if unrecognized_patterns:
                raise TypeError(
                    "Target modules must be a dictionary with lists of "
                    "Unix shell-style wildcards (fnmatch-style) patterns as values. "
                    f"Could not recognize the following as fnmatch patterns: {unrecognized_patterns}."
                )

        # handle default value: modify the dict in place to have a match between the value and default value
        if value is None:
            pass  # chosing a default value is left to the algorithm based on the model
        elif "include" not in value:
            value["include"] = ["*"]
        elif "exclude" not in value:
            value["exclude"] = []  # for consistency

        return super().legal_value(value)


def is_leaf_module(module: torch.nn.Module) -> bool:
    """
    Check if a module is a leaf module.

    Parameters
    ----------
    module : torch.nn.Module
        The module to check.

    Returns
    -------
    bool
        True if the module is a leaf module, False otherwise.
    """
    return len(list(module.children())) == 0


def expand_list_of_targeted_paths(
    target_modules: TARGET_MODULES_TYPE, model: Any, leaf_modules: bool = False
) -> List[str]:
    """
    Convert the target modules to a list of module paths.

    Parameters
    ----------
    model : Any
        The model to get the module paths from.
    target_modules : TARGET_MODULES_TYPE
        The target modules to convert to a list of module paths.
    leaf_modules : bool
        If True, only include modules which do not contain other modules themselves. Default is False.

    Returns
    -------
    List[str]
        The list of module paths.

    Raises
    ------
    ValueError
        If no targeted subpath is found within the model.
    """
    include = target_modules.get("include", ["*"])
    exclude = target_modules.get("exclude", [])
    modules_paths = []
    for root_name, module in get_nn_modules(model).items():
        module_paths = [
            f"{root_name}{'.' + path if path else ''}" if root_name else path
            for path, submodule in module.named_modules()
            if (not leaf_modules) or is_leaf_module(submodule)  # only check when leaf_modules is True
        ]
        matching_modules = [
            path
            for path in module_paths
            if any(fnmatch.fnmatch(path, _include) for _include in include)
            and not any(fnmatch.fnmatch(path, _exclude) for _exclude in exclude)
        ]
        modules_paths.extend(matching_modules)

    if not modules_paths:
        raise ValueError(f"No targeted subpath found within the model from target_modules {target_modules}")
    return modules_paths


def expand_dict_of_roots_and_subpaths(
    target_modules: TARGET_MODULES_TYPE, model: Any, leaf_modules: bool = False
) -> Dict[str | None, Tuple[torch.nn.Module, List[str]]]:
    """
    Get the torch modules within the model and their associated targeted subpaths.

    Parameters
    ----------
    target_modules : TARGET_MODULES_TYPE
        The target modules to convert to a list of module paths.
    model : Any
        The model to get the module paths from.
    leaf_modules : bool
        If True, only include modules which do not contain other modules themselves. Default is False.

    Returns
    -------
    Dict[str | None, Tuple[torch.nn.Module, List[str]]]
        The dictionary of modules attributes in the model with their associated targeted subpaths.
        A module attribute which doesn't contain any targeted subpath won't be included in the dictionary.
        Each module-subpaths pair is indexed by the module attribute name within the model.
        Following the convention of get_nn_modules, if the model itself is a torch.nn.Module, the dictionary
        will contain a single item with key None, pointing to the model itself and the targeted paths.
    """
    target_modules_paths = expand_list_of_targeted_paths(target_modules, model)

    modules_with_subpaths: Dict[str | None, Tuple[torch.nn.Module, List[str]]] = {}
    for root_name, module in get_nn_modules(model).items():
        prefix = f"{root_name}." if root_name else ""

        targeted_submodules = [path for path in target_modules_paths if path.startswith(prefix)]
        targeted_submodules = [path.removeprefix(prefix) for path in targeted_submodules]

        # only register the module if it contains at least one targeted submodule
        if targeted_submodules:
            modules_with_subpaths[root_name] = (module, targeted_submodules)

    return modules_with_subpaths


def get_skipped_submodules(module: torch.nn.Module, subpaths: List[str], leaf_modules: bool = False) -> List[str]:
    """
    Get the skipped submodules.

    Parameters
    ----------
    module : torch.nn.Module
        The module to get the skipped submodules from.
    subpaths : List[str]
        The subpaths to get the skipped submodules from.
    leaf_modules : bool
        If True, only include modules which do not contain other modules themselves. Default is False.

    Returns
    -------
    List[str]
        The list of submodules not listed in subpaths.
    """
    subpaths_set = set(subpaths)  # quicker for lookups if the model is very large
    return [
        path
        for path, submodule in module.named_modules()
        if path not in subpaths_set
        and (not leaf_modules or is_leaf_module(submodule))  # only check when leaf_modules is True
    ]


def map_targeted_nn_roots(
    apply_single_root_fn: Callable[[str | None, torch.nn.Module, List[str]], Any],
    model: Any,
    target_modules: TARGET_MODULES_TYPE,
    leaf_modules: bool = False,
) -> Any:
    """
    Apply a function to the model, or to each of its targeted nn.Modules in the case of a Pipeline.

    Parameters
    ----------
    apply_single_root_fn : Callable[[str | None, torch.nn.Module, List[str]], Any]
        The function to apply to each root in the model.
        It must take as input the attribute name of the root in the model, the nn.Module itself, and a list of
        paths within the root, each pointing to a targeted submodule. It must return the modified root.
        The roots are the model itself if it is a torch.nn.Module (attribute name is None in this case),
        or its nn.Module attributes otherwise.
    model : Any
        The model to apply the function to.
    target_modules : TARGET_MODULES_TYPE
        The target modules to apply the function to.
    leaf_modules : bool
        If True, only target modules which do not contain other modules themselves. Default is False.

    Returns
    -------
    Any
        The model after the function has been applied.
    """
    nn_roots_with_subpaths = expand_dict_of_roots_and_subpaths(target_modules, model, leaf_modules=leaf_modules)
    for attr_name, (nn_root, subpaths) in nn_roots_with_subpaths.items():
        # modify the root with the provided function
        applied_root = apply_single_root_fn(attr_name, nn_root, subpaths)
        if applied_root is None:
            raise ValueError("The 'apply_single_root_fn' function must return the modified root.")

        # replace the root with the modified one
        if attr_name is None:
            # by convention, this means the model itself is a torch.nn.Module, which we got as module
            model = applied_root
        else:
            setattr(model, attr_name, applied_root)
    return model
