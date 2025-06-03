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

from functools import partial
from typing import Any, Callable, Dict, Iterable, List

from pruna.engine.load import filter_load_kwargs
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.logging.logger import pruna_logger


class MetricRegistry:
    """
    Registry for metrics.

    The registry is a dictionary that maps metric names to metric classes.
    """

    _registry: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to register a custom metric.

        The metric then can be accessed via MetricRegistry.get_metric(name).

        Parameters
        ----------
            name : str
                The name of the metric.

        Examples
        --------
            @MetricRegistry.register("name_for_MetricA")
            class MetricA: ...
        """

        def decorator(metric_cls: Callable[..., Any]) -> Callable[..., Any]:
            if name not in cls._registry:
                cls._registry[name] = metric_cls
            else:
                pruna_logger.error(f"Metric '{name}' is already registered.")
            return metric_cls

        return decorator

    @classmethod
    def register_wrapper(cls, available_metrics: Iterable[str]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator for a wrapper metric class.

        For each allowed metric name, registers a factory that instantiates the wrapper
        by calling: wrapper_class(metric_name=<name>)

        Parameters
        ----------
            available_metrics : Iterable[str]
                An iterable of available metric names.

        Examples
        --------
            @MetricRegistry.register_wrapper(available_metrics=("fid", "accuracy", "clip_score", ...))
            class TorchMetricsWrapper(...): ...
        """

        def decorator(wrapper_class: Callable[..., Any]) -> Callable[..., Any]:
            for name in available_metrics:
                if name not in cls._registry:
                    cls._registry[name] = partial(wrapper_class, metric_name=name)
                else:
                    pruna_logger.error(f"Metric '{name}' is already registered.")
            return wrapper_class

        return decorator

    @classmethod
    def get_metric(cls, name: str, **kwargs) -> BaseMetric | StatefulMetric:
        """
        Get a metric from the registry.

        Parameters
        ----------
            name : str
                The name of the metric.
            **kwargs :
                Additional keyword arguments for the metric.

        Returns
        -------
            The metric instance.
        """
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' is not registered.")

        metric_cls = cls._registry[name]
        if isinstance(metric_cls, partial) and "TorchMetricWrapper" in metric_cls.func.__name__:
            return metric_cls(**kwargs)
        else:
            return metric_cls(**filter_load_kwargs(metric_cls, kwargs))

    @classmethod
    def get_metrics(cls, names: List[str], **kwargs) -> List[BaseMetric | StatefulMetric]:
        """
        Get requested metrics from the registry.

        Parameters
        ----------
            names : List[str]
                The names of the metrics.
            **kwargs :
                Additional keyword arguments for the metrics.

        Returns
        -------
            A list of metric instances.
        """
        return [cls.get_metric(name, **kwargs) for name in names]
