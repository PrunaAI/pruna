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
Introspection helpers that expose pruna's algorithm catalogue as plain, JSON-serializable data.

Every function in this module imports pruna lazily so that merely importing
``pruna.mcp_server`` does not pull in torch and the rest of the heavy runtime.
"""

from __future__ import annotations

from typing import Any

# Attributes of a ConfigSpace hyperparameter that carry its allowed value range.
_RANGE_ATTRIBUTES = ("choices", "sequence", "lower", "upper", "q", "log")


def _registry() -> Any:
    """
    Import and return the algorithm registry.

    Returns
    -------
    Any
        The ``AlgorithmRegistry`` class from pruna.
    """
    from pruna import AlgorithmRegistry

    return AlgorithmRegistry


def _to_jsonable(value: Any) -> Any:
    """
    Convert an arbitrary ConfigSpace value into something JSON-serializable.

    Parameters
    ----------
    value : Any
        The value to convert.

    Returns
    -------
    Any
        A JSON-serializable representation of the value.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def _serialize_hyperparameter(hyperparameter: Any, algorithm_name: str) -> dict[str, Any]:
    """
    Serialize a single ConfigSpace hyperparameter of an algorithm.

    Hyperparameter names are reported both unprefixed (as the algorithm class declares them)
    and fully qualified with the algorithm prefix, because the prefix is what appears in a
    ``SmashConfig``.

    Parameters
    ----------
    hyperparameter : Any
        A ConfigSpace hyperparameter object.
    algorithm_name : str
        Name of the algorithm the hyperparameter belongs to.

    Returns
    -------
    dict[str, Any]
        JSON-serializable description of the hyperparameter.
    """
    raw_name = str(hyperparameter.name)
    prefix = f"{algorithm_name}_"
    short_name = raw_name[len(prefix) :] if raw_name.startswith(prefix) else raw_name
    meta = getattr(hyperparameter, "meta", None) or {}

    described: dict[str, Any] = {
        "name": short_name,
        "qualified_name": f"{algorithm_name}_{short_name}",
        "type": type(hyperparameter).__name__,
        "default": _to_jsonable(getattr(hyperparameter, "default_value", None)),
    }
    if meta.get("desc"):
        described["description"] = str(meta["desc"])

    for attribute in _RANGE_ATTRIBUTES:
        value = getattr(hyperparameter, attribute, None)
        if value is not None:
            described[attribute] = _to_jsonable(value)

    described["allowed"] = _allowed_values(hyperparameter, described["default"])
    return described


def _allowed_values(hyperparameter: Any, default: Any) -> list[Any]:
    """
    Enumerate the values a hyperparameter will accept.

    The ConfigSpace hyperparameters spell their domain differently: categoricals expose
    ``choices``, ordinals expose ``sequence`` and a constant exposes nothing at all. A caller
    should not have to know which spelling applies, so the enumerable domains are collected
    under one key. A continuous domain cannot be enumerated and is reported as an empty list,
    with ``lower`` and ``upper`` left in place.

    Parameters
    ----------
    hyperparameter : Any
        A ConfigSpace hyperparameter object.
    default : Any
        The already-serialized default value, reused for a constant.

    Returns
    -------
    list[Any]
        Every accepted value, or an empty list when the domain is continuous.
    """
    if type(hyperparameter).__name__ == "Constant":
        return [default]

    for attribute in ("choices", "sequence", "q"):
        value = getattr(hyperparameter, attribute, None)
        if value is not None:
            return [_to_jsonable(item) for item in value]

    return []


def _requirements(algorithm: Any) -> dict[str, Any]:
    """
    Summarize what an algorithm needs before it can run.

    Parameters
    ----------
    algorithm : Any
        A registered algorithm instance.

    Returns
    -------
    dict[str, Any]
        JSON-serializable requirements of the algorithm.
    """
    return {
        "tokenizer": bool(getattr(algorithm, "tokenizer_required", False)),
        "processor": bool(getattr(algorithm, "processor_required", False)),
        "dataset": bool(getattr(algorithm, "dataset_required", False)),
        "pip_install": getattr(algorithm, "required_install", None),
    }


def _summary(algorithm: Any) -> dict[str, Any]:
    """
    Build the compact algorithm summary used by ``list_algorithms``.

    Parameters
    ----------
    algorithm : Any
        A registered algorithm instance.

    Returns
    -------
    dict[str, Any]
        JSON-serializable summary of the algorithm.
    """
    save_fn = getattr(algorithm, "save_fn", None)
    return {
        "name": algorithm.algorithm_name,
        "tags": [str(tag) for tag in getattr(algorithm, "group_tags", [])],
        "runs_on": list(getattr(algorithm, "runs_on", [])),
        "requires": _requirements(algorithm),
        "save_fn": getattr(save_fn, "name", None),
        "references": _to_jsonable(getattr(algorithm, "references", None)),
    }


def list_tags() -> list[dict[str, str]]:
    """
    List every algorithm category with its human-readable description.

    Returns
    -------
    list[dict[str, str]]
        One entry per algorithm tag, each with a ``name`` and a ``description``.
    """
    from pruna.algorithms.base.tags import AlgorithmTag

    return [tag.get_documentation() for tag in AlgorithmTag]


def list_algorithms(tag: str | None = None) -> list[dict[str, Any]]:
    """
    List all registered optimization algorithms, optionally filtered by category.

    Parameters
    ----------
    tag : str | None
        Restrict the result to algorithms in this category, for example ``"quantizer"`` or
        ``"cacher"``. Use ``list_tags`` for the full set of valid categories. If None, every
        algorithm is returned.

    Returns
    -------
    list[dict[str, Any]]
        One summary entry per algorithm.

    Raises
    ------
    ValueError
        If ``tag`` is not a known algorithm category.
    """
    from pruna.algorithms.base.tags import AlgorithmTag

    registry = _registry()
    if tag is None:
        return [_summary(algorithm) for algorithm in registry.get_all_algorithms()]

    try:
        algorithm_tag = AlgorithmTag[tag.upper()]
    except KeyError:
        valid = ", ".join(str(member) for member in AlgorithmTag)
        raise ValueError(f"Unknown algorithm tag '{tag}'. Valid tags are: {valid}.") from None

    return [_summary(registry[name]) for name in registry.get_algorithms_by_tag(algorithm_tag)]


def describe_algorithm(name: str) -> dict[str, Any]:
    """
    Describe a single algorithm in full detail.

    This is the tool to call before composing a configuration: it returns every
    hyperparameter with its allowed values and default, the order constraints, and the
    requirements the algorithm imposes.

    Parameters
    ----------
    name : str
        Name of the algorithm, as returned by ``list_algorithms``.

    Returns
    -------
    dict[str, Any]
        Full description of the algorithm.

    Raises
    ------
    KeyError
        If no algorithm with that name is registered.
    """
    registry = _registry()
    if name not in {algorithm.algorithm_name for algorithm in registry.get_all_algorithms()}:
        raise KeyError(f"Unknown algorithm '{name}'. Use list_algorithms to see valid names.")

    algorithm = registry[name]
    described = _summary(algorithm)
    described.update(
        {
            "compatible_before": _to_jsonable(algorithm.get_algorithms_to_run_before()),
            "compatible_after": _to_jsonable(algorithm.get_algorithms_to_run_after()),
            "compatible_with": _to_jsonable(algorithm.get_compatible_algorithms()),
            "disjointly_compatible_with": _to_jsonable(algorithm.get_disjointly_compatible_algorithms()),
            "hyperparameters": [
                _serialize_hyperparameter(hyperparameter, name) for hyperparameter in algorithm.hyperparameters
            ],
        }
    )
    return described


def list_metrics() -> list[str]:
    """
    List every evaluation metric name that can be requested from an evaluation task.

    Returns
    -------
    list[str]
        Sorted metric names available through ``EvaluationAgent``.
    """
    from pruna.evaluation.metrics import MetricRegistry

    # MetricRegistry exposes registration and lookup but no public enumeration helper,
    # so the underlying registry mapping is read directly.
    registered = getattr(MetricRegistry, "_registry", {})
    return sorted(str(key) for key in registered)


def list_datasets() -> list[str]:
    """
    List the named datasets that an evaluation task can be run on.

    These are the exact names ``PrunaDataModule.from_string`` accepts, so a caller never has to
    guess the spelling of a dataset.

    Returns
    -------
    list[str]
        The registered dataset names.
    """
    from pruna.data import base_datasets

    return sorted(base_datasets.keys())


def list_benchmarks(task_type: str | None = None) -> list[dict[str, Any]]:
    """
    List the built-in evaluation benchmarks, optionally filtered by task type.

    Parameters
    ----------
    task_type : str | None
        Restrict the result to benchmarks for this task type, for example
        ``"text_to_image"``. If None, every benchmark is returned.

    Returns
    -------
    list[dict[str, Any]]
        One entry per benchmark, including the metrics it evaluates.
    """
    from pruna.evaluation.benchmarks import BenchmarkRegistry

    described = []
    for key in BenchmarkRegistry.list(task_type=task_type):
        benchmark = BenchmarkRegistry.get(key)
        described.append(
            {
                "name": benchmark.name,
                "lookup_key": benchmark.lookup_key,
                "task_type": benchmark.task_type,
                "description": benchmark.description,
                "metrics": list(benchmark.metrics),
                "reference": benchmark.reference,
            }
        )
    return described
