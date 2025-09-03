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

from typing import Any

from ConfigSpace import CategoricalHyperparameter, Constant
from typing_extensions import override


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
