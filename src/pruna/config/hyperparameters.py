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
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from ConfigSpace import CategoricalHyperparameter, Constant
from typing_extensions import override

from pruna.engine.utils import get_nn_modules

TARGET_MODULES_TYPE = Dict[Literal["include", "exclude"], List[str]]


class Boolean(CategoricalHyperparameter):
    """
    Represents a boolean hyperparameter with choices True and False.

    Parameters
    ----------
    name : str
        The name of the hyperparameter.
    default : bool
        The default value of the hyperparameter.
    meta : Any
        The metadata for the hyperparameter.
    """

    def __init__(self, name: str, default: bool = False, meta: Any = dict()) -> None:
        super().__init__(name, choices=[True, False], default_value=default, meta=meta)

    def __new__(cls, name: str, default: bool = False, meta: Any = None) -> CategoricalHyperparameter:  # type: ignore
        """Create a new boolean hyperparameter."""
        return CategoricalHyperparameter(name, choices=[True, False], default_value=default, meta=meta)


class UnconstrainedHyperparameter(Constant):
    """
    Represents a hyperparameter that is unconstrained and can be set to any value by the user.

    Parameters
    ----------
    name : str
        The name of the hyperparameter.
    default_value : Any
        The default value of the hyperparameter.
    meta : Any
        The metadata for the hyperparameter.
    """

    def __init__(
        self,
        name: str,
        default_value: Any = None,
        meta: Any = None,
    ) -> None:
        super().__init__(name, default_value, meta)

    @override
    def legal_value(self, value):  # numpydoc ignore=GL08
        """
        Check if a value is legal for this hyperparameter.

        This hyperparameter is unconstrained and can be set to any value by the user.
        Therefore, this method always returns `True` as long as the format is accepted
        by ConfigSpace.

        Parameters
        ----------
        value : Any
            The value to check.

        Returns
        -------
        bool or numpy.ndarray
            `True` if the value is legal, `False` otherwise. If `value` is an array,
            a boolean mask of legal values is returned.
        """
        # edit the internal state of the Constant to allow for the new value
        self._contains_sequence_as_value = isinstance(value, (list, tuple))
        self._transformer.value = value
        # we still run the super method which should return True, to make sure internal values
        # are correctly updated
        return super().legal_value(value)


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
        elif "include" not in value or len(value["include"]) == 0:
            raise ValueError("Target modules must have at least one 'include' pattern.")
        else:
            all_patterns = [pattern for patterns in value.values() for pattern in patterns]
            unrecognized_patterns = [pattern for pattern in all_patterns if not isinstance(pattern, str)]
            if unrecognized_patterns:
                raise TypeError(
                    "Target modules must be a dictionary with lists of "
                    "Unix shell-style wildcards (fnmatch-style) patterns as values. "
                    f"Could not recognize the following as fnmatch patterns: {unrecognized_patterns}."
                )

        # handle default value
        value = deepcopy(value)
        if value is None:
            pass  # chosing a default value is left to the algorithm based on the model
        elif "include" not in value:
            value["include"] = ["*"]
        elif "exclude" not in value:
            value["exclude"] = []  # for consistency

        return super().legal_value(value)

    @staticmethod
    def to_list_of_modules_paths(target_modules: TARGET_MODULES_TYPE, model: Any) -> List[str]:
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
        include = target_modules.get("include", ["*"])
        exclude = target_modules.get("exclude", [])
        modules_paths = []
        for root_name, module in get_nn_modules(model).items():
            module_paths = [
                f"{root_name}{'.' + path if path else ''}" if root_name else path for path, _ in module.named_modules()
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

    @staticmethod
    def to_list_of_roots_and_subpaths(
        model: Any, target_modules: TARGET_MODULES_TYPE
    ) -> List[Tuple[torch.nn.Module, List[str]]]:
        """
        Get torch modules within the model and their associated subpaths.

        Parameters
        ----------
        model : Any
            The model to get the module paths from.
        target_modules : TARGET_MODULES_TYPE
            The target modules to convert to a list of module paths.

        Returns
        -------
        List[Tuple[torch.nn.Module, List[str]]]
            The list of modules attributes in the model with their associated targeted subpaths.
            A module attribute which doesn't contain any targeted subpath won't be included in the list.
        """
        target_modules_paths = TargetModules.to_list_of_modules_paths(target_modules, model)
        modules_with_subpaths: List[Tuple[torch.nn.Module, List[str]]] = []
        for root_name, module in get_nn_modules(model).items():
            prefix = f"{root_name}." if root_name else ""
            targeted_submodules = [path for path in target_modules_paths if path.startswith(prefix)]
            targeted_submodules = [path.removeprefix(prefix) for path in targeted_submodules]
            if targeted_submodules:
                modules_with_subpaths.append((module, targeted_submodules))
        return modules_with_subpaths
