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

"""
Configuration validation that runs without loading a model.

``pruna.smash`` performs its checks as part of the optimization run, which means a bad
configuration is only discovered after a model has been loaded onto a device. The helpers
here reuse the same routines from :mod:`pruna.config.pre_smash_routines` to surface those
problems earlier, plus a static compatibility check that ``smash`` can only perform once a
model is in hand.
"""

from __future__ import annotations

from typing import Any

from .introspection import _to_jsonable


def build_config(
    algorithms: list[str],
    hyperparameters: dict[str, Any] | None = None,
    device: str | None = None,
    batch_size: int = 1,
) -> Any:
    """
    Build a ``SmashConfig`` from the flat arguments the MCP tools accept.

    Parameters
    ----------
    algorithms : list[str]
        Names of the algorithms to activate.
    hyperparameters : dict[str, Any] | None
        Hyperparameter values keyed by their fully qualified name, for example
        ``{"hqq_weight_bits": 4}``. If None, every hyperparameter keeps its default.
    device : str | None
        Target device, for example ``"cpu"`` or ``"cuda"``. If None, the best available
        device is chosen by pruna.
    batch_size : int
        Inference batch size recorded on the configuration.

    Returns
    -------
    Any
        The constructed ``SmashConfig``.
    """
    from pruna import SmashConfig

    payload: dict[str, Any] = dict.fromkeys(algorithms, True)
    payload.update(hyperparameters or {})
    config = SmashConfig(payload)
    if device is not None:
        config.device = device
    config.batch_size = batch_size
    return config


def collect_requirements(algorithms: list[str]) -> dict[str, Any]:
    """
    Report what a set of algorithms will need before it can run.

    Parameters
    ----------
    algorithms : list[str]
        Names of the activated algorithms.

    Returns
    -------
    dict[str, Any]
        Lists of algorithms requiring a tokenizer, a processor or a dataset, together with the
        union of the devices they support.
    """
    from pruna import AlgorithmRegistry

    registry = AlgorithmRegistry
    tokenizer: list[str] = []
    processor: list[str] = []
    dataset: list[str] = []
    devices: list[str] = []
    pip_install: list[str] = []

    for name in algorithms:
        algorithm = registry[name]
        if getattr(algorithm, "tokenizer_required", False):
            tokenizer.append(name)
        if getattr(algorithm, "processor_required", False):
            processor.append(name)
        if getattr(algorithm, "dataset_required", False):
            dataset.append(name)
        for device in getattr(algorithm, "runs_on", []):
            if device not in devices:
                devices.append(device)
        required_install = getattr(algorithm, "required_install", None)
        if required_install and required_install not in pip_install:
            pip_install.append(required_install)

    return {
        "tokenizer": tokenizer,
        "processor": processor,
        "dataset": dataset,
        "supported_devices": devices,
        "pip_install": pip_install,
    }


def find_incompatible_pairs(algorithms: list[str]) -> list[tuple[str, str]]:
    """
    Statically detect algorithm pairs that cannot be combined.

    This mirrors the pairwise branch of ``check_algorithm_cross_compatibility`` without
    needing a model. It therefore cannot detect overlapping ``target_modules`` between
    algorithms that are only disjointly compatible; that check still needs a loaded model.

    Parameters
    ----------
    algorithms : list[str]
        Names of the activated algorithms.

    Returns
    -------
    list[tuple[str, str]]
        The incompatible pairs, each pair ordered deterministically.
    """
    from pruna import AlgorithmRegistry

    incompatible: list[tuple[str, str]] = []
    for index, first in enumerate(algorithms):
        for second in algorithms[index + 1 :]:
            other = AlgorithmRegistry[second]
            if first in other.get_compatible_algorithms():
                continue
            if first in other.get_disjointly_compatible_algorithms():
                continue
            incompatible.append((first, second))
    return incompatible


def determine_execution_order(config: Any) -> list[str] | None:
    """
    Report the order the active algorithms will actually be applied in.

    This delegates to pruna's own ordering routine so the MCP server does not reimplement the
    topological sort. A ``None`` result means the order could not be determined without a model,
    which happens when an algorithm is only disjointly compatible with another and the decision
    depends on their target modules.

    Parameters
    ----------
    config : Any
        The ``SmashConfig`` to order.

    Returns
    -------
    list[str] | None
        The algorithms in the order they will run, or None if it cannot be determined
        without a model.
    """
    from pruna.config.pre_smash_routines import determine_algorithm_order

    try:
        return determine_algorithm_order(None, config)
    except Exception:
        return None


def resolve_hyperparameter(config: Any, algorithm_name: str, hyperparameter: Any) -> Any:
    """
    Read the effective value of one hyperparameter from a configuration.

    ConfigSpace registers hyperparameters under their prefixed name, but the prefixing
    happens as a side effect of algorithm discovery. Both the prefixed and the bare name are
    therefore tried, so an explicitly configured value is never silently replaced by the
    default.

    Parameters
    ----------
    config : Any
        The ``SmashConfig`` to read from.
    algorithm_name : str
        Name of the algorithm the hyperparameter belongs to.
    hyperparameter : Any
        The ConfigSpace hyperparameter object.

    Returns
    -------
    Any
        The configured value, or the hyperparameter's default when it was never set.
    """
    names = [str(hyperparameter.name)]
    prefixed = f"{algorithm_name}_{names[0]}"
    if prefixed != names[0]:
        names.append(prefixed)

    for candidate in names:
        value = config.get(candidate)
        if value is not None:
            return value
    return getattr(hyperparameter, "default_value", None)


def _requirements_or_empty(algorithms: list[str]) -> dict[str, Any]:
    """
    Collect requirements, tolerating names that the registry does not know.

    Parameters
    ----------
    algorithms : list[str]
        Names of the algorithms the caller asked for, which may include unknown ones.

    Returns
    -------
    dict[str, Any]
        The requirements of every recognised algorithm, or an empty block if none is known.
    """
    from pruna import AlgorithmRegistry

    known: list[str] = []
    for name in algorithms:
        try:
            AlgorithmRegistry[name]
        except KeyError:
            continue
        known.append(name)
    return collect_requirements(known) if known else {}


def validate_config(
    algorithms: list[str],
    hyperparameters: dict[str, Any] | None = None,
    device: str | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """
    Validate a proposed optimization configuration without loading a model.

    Runs the same argument and dependency checks that ``pruna.smash`` runs, plus a static
    pairwise compatibility check, so an invalid configuration is caught before any device
    is touched.

    Parameters
    ----------
    algorithms : list[str]
        Names of the algorithms to activate, as returned by ``list_algorithms``.
    hyperparameters : dict[str, Any] | None
        Hyperparameter values keyed by their fully qualified name, for example
        ``{"hqq_weight_bits": 4}``. If None, every hyperparameter keeps its default.
    device : str | None
        Target device, for example ``"cpu"`` or ``"cuda"``. If None, the best available
        device is chosen by pruna.
    batch_size : int
        Inference batch size to record on the configuration.

    Returns
    -------
    dict[str, Any]
        A report with the ``valid`` flag, a list of ``errors``, the resolved configuration,
        the requirements the algorithms impose, the ``execution_order`` they will be applied
        in, and a list of ``notes`` describing what could not be checked without a model.
    """
    from pruna.config.pre_smash_routines import (
        check_algorithm_packages_availability,
        check_argument_compatibility,
    )

    errors: list[dict[str, str]] = []
    notes: list[str] = []

    try:
        config = build_config(algorithms, hyperparameters, device, batch_size)
    except Exception as error:
        # SmashConfig raises on an unknown algorithm or a rejected hyperparameter value. The
        # report has to keep the same shape as the success path, so a client reading a fixed
        # set of keys does not break on exactly the configurations it most needs to see.
        return {
            "valid": False,
            "errors": [
                {
                    "check": "build_config",
                    "message": str(error),
                    "hint": (
                        "Check the algorithm names against list_algorithms and the hyperparameter "
                        "values against the allowed values describe_algorithm reports."
                    ),
                }
            ],
            "active_algorithms": [],
            "resolved_hyperparameters": {},
            "requirements": _requirements_or_empty(algorithms),
            "execution_order": None,
            "notes": ["No configuration could be built, so nothing was checked against the registry."],
        }

    active = config.get_active_algorithms()

    try:
        check_argument_compatibility(config)
    except Exception as error:
        errors.append(
            {
                "check": "check_argument_compatibility",
                "message": str(error),
                "hint": "Attach the missing object to the SmashConfig before smashing.",
            }
        )

    try:
        check_algorithm_packages_availability(config)
    except ImportError as error:
        errors.append(
            {
                "check": "check_algorithm_packages_availability",
                "message": str(error),
                "hint": "Install the missing pip extra named in the message.",
            }
        )
    except Exception as error:
        errors.append(
            {
                "check": "check_algorithm_packages_availability",
                "message": str(error),
                "hint": "",
            }
        )

    for first, second in find_incompatible_pairs(active):
        errors.append(
            {
                "check": "cross_compatibility",
                "message": f"Algorithm '{first}' is incompatible with '{second}'.",
                "hint": (
                    "Call describe_algorithm on both and keep only pairs where one appears in the "
                    "other's 'compatible_with' list."
                ),
            }
        )

    notes.append("Model compatibility and target-module overlap are not checked here; they require a loaded model.")

    resolved: dict[str, Any] = {}
    from pruna import AlgorithmRegistry

    for name in active:
        algorithm = AlgorithmRegistry[name]
        resolved[name] = {
            hyperparameter.name: _to_jsonable(resolve_hyperparameter(config, name, hyperparameter))
            for hyperparameter in algorithm.hyperparameters
        }

    return {
        "valid": not errors,
        "errors": errors,
        "active_algorithms": active,
        "resolved_hyperparameters": resolved,
        "requirements": collect_requirements(active),
        "execution_order": determine_execution_order(config),
        "notes": notes,
    }
