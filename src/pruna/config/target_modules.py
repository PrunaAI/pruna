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


def is_targeted(path: str, target_modules: TARGET_MODULES_TYPE) -> bool:
    """
    Check if a path is targeted.

    Parameters
    ----------
    path : str
        The path to check.
    target_modules : TARGET_MODULES_TYPE
        The target modules specifying which modules are targeted.
    """
    include = target_modules.get("include", ["*"])
    exclude = target_modules.get("exclude", [])
    is_included = any(fnmatch.fnmatch(path, _include) for _include in include)
    is_excluded = any(fnmatch.fnmatch(path, _exclude) for _exclude in exclude)
    return is_included and not is_excluded


def expand_list_of_targeted_paths(target_modules: TARGET_MODULES_TYPE, model: Any) -> List[str]:
    """
    Convert the target modules to a list of module paths.

    Parameters
    ----------
    model : Any
        The model to get the module paths from.
    target_modules : TARGET_MODULES_TYPE
        The target modules to convert to a list of module paths.

    Returns
    -------
    List[str]
        The list of module paths.

    Raises
    ------
    ValueError
        If no targeted subpath is found within the model.
    """
    modules_paths = []
    for root_name, module in get_nn_modules(model).items():
        module_paths = [
            f"{root_name}{'.' + path if path else ''}" if root_name else path
            for path, submodule in module.named_modules()
        ]
        module_paths = [path for path in module_paths if is_targeted(path, target_modules)]
        modules_paths.extend(module_paths)

    if not modules_paths:
        raise ValueError(f"No targeted subpath found within the model from target_modules {target_modules}")
    return modules_paths


def expand_dict_of_roots_and_subpaths(
    target_modules: TARGET_MODULES_TYPE, model: Any
) -> Dict[str | None, Tuple[torch.nn.Module, List[str]]]:
    """
    Get the torch modules within the model and their associated targeted subpaths.

    Parameters
    ----------
    target_modules : TARGET_MODULES_TYPE
        The target modules to convert to a list of module paths.
    model : Any
        The model to get the module paths from.

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


def filter_targeted_modules(
    keep_targeted_fn: Callable[[torch.nn.Module, str | None], bool],
    model: Any,
    target_modules: TARGET_MODULES_TYPE,
) -> TARGET_MODULES_TYPE:
    """
    Expand the target modules to exclude modules incompatible with FLUTE kernels.

    Parameters
    ----------
    keep_targeted_fn : Callable[[torch.nn.Module, str | None], bool]
        The function to check if a targeted module should be kept.
    model : Any
        The model to get the default target modules for.
    target_modules : TARGET_MODULES_TYPE
        The target modules to expand.

    Returns
    -------
    TARGET_MODULES_TYPE
        The expanded target modules.
    """
    additional_exclude: List[str] = []

    for root_path, (root, subpaths) in expand_dict_of_roots_and_subpaths(target_modules, model).items():
        subpaths_set = set(subpaths)  # faster when there are many subpaths
        for subpath, module in root.named_modules():
            if subpath not in subpaths_set:
                continue  # no need to check if it should be kept if it isn't included in the first place
            full_path = f"{root_path}.{subpath}" if root_path is not None else subpath
            if not keep_targeted_fn(module, full_path):
                additional_exclude.append(full_path)
    return {"include": target_modules["include"], "exclude": target_modules["exclude"] + additional_exclude}


def get_skipped_submodules(
    module: torch.nn.Module,
    subpaths: List[str],
    filter_fn: Callable[[torch.nn.Module, str | None], bool] | None = None,
) -> List[str]:
    """
    Get the skipped submodules.

    Parameters
    ----------
    module : torch.nn.Module
        The module to get the skipped submodules from.
    subpaths : List[str]
        The subpaths to get the skipped submodules from.
    filter_fn : Callable[[torch.nn.Module, str | None], bool] | None
        The function to check if a skipped module should be returned or not. By default, all skipped modules
        are returned. This could be used to only return paths to leaf modules for example.

    Returns
    -------
    List[str]
        The list of submodules not listed in subpaths.
    """
    if filter_fn is None:
        # default behavior: keep all skipped modules
        def filter_fn(_module: torch.nn.Module, _path: str | None) -> bool:
            return True

    subpaths_set = set(subpaths)  # quicker for lookups if the model is very large
    return [
        path for path, submodule in module.named_modules() if path not in subpaths_set and filter_fn(submodule, path)
    ]


def map_targeted_nn_roots(
    apply_single_root_fn: Callable[[str | None, torch.nn.Module, List[str]], Any],
    model: Any,
    target_modules: TARGET_MODULES_TYPE,
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

    Returns
    -------
    Any
        The model after the function has been applied.
    """
    nn_roots_with_subpaths = expand_dict_of_roots_and_subpaths(target_modules, model)
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
