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
import importlib
import inspect
import logging
import pkgutil
from typing import Any, Callable, Dict

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag


class AlgorithmRegistry:
    """
    Registry for metrics.

    The registry is a dictionary that maps metric names to metric classes.
    """

    _registry: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def discover_algorithms(cls, algorithms_pkg: Any) -> None:
        """
        Discover every package/module under `algorithms_pkg` by walking the package.

        Parameters
        ----------
        algorithms_pkg : Any
            The package to discover algorithms in.

        Returns
        -------
        None
            This function does not return anything.
        """
        prefix = algorithms_pkg.__name__ + "."
        for _finder, modname, _ispkg in pkgutil.walk_packages(algorithms_pkg.__path__, prefix):
            try:
                module = importlib.import_module(modname)
            except Exception as e:
                logging.warning("Skipping %s (import error): %s", modname, e)
                continue

            # Inspect classes defined in this module (avoid classes only re-exported here)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ != module.__name__:
                    continue

                # Must be a subclass (but not the base itself)
                if not issubclass(obj, PrunaAlgorithmBase) or obj is PrunaAlgorithmBase:
                    continue
                # Must be a **direct** child (first-grade)
                if PrunaAlgorithmBase not in obj.__bases__:
                    continue

                # Skip abstract bases
                if inspect.isabstract(obj):
                    continue

                # Instantiate & register with the Smash Configuration Space
                try:
                    instance = obj()
                    cls._registry[instance.algorithm_name] = instance
                except Exception as e:
                    logging.warning("Failed to instantiate %s from %s: %s", obj.__name__, modname, e)

    @classmethod
    def __class_getitem__(cls, algorithm_name: str) -> PrunaAlgorithmBase:  # noqa: D105
        return cls._registry[algorithm_name]

    @classmethod
    def get_algorithms_by_tag(cls, tag: AlgorithmTag) -> list[PrunaAlgorithmBase]:
        """
        Get all algorithms that have the given tag.

        Parameters
        ----------
        tag : AlgorithmTag
            The tag to get algorithms for.

        Returns
        -------
        list[PrunaAlgorithmBase]
            The algorithms that have the given tag.
        """
        return [alg.algorithm_name for alg in cls._registry.values() if tag in alg.group_tags]
