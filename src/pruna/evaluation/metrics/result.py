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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class MetricResultProtocol(Protocol):
    """
    Protocol defining the shared interface for all metric results.

    Any metric result class should implement these attributes and methods
    to be compatible with the evaluation pipeline.

    # Have to include this to prevent ty errors.

    Parameters
    ----------
    *args :
        Additional arguments passed to the MetricResultProtocol.
    **kwargs :
        Additional keyword arguments passed to the MetricResultProtocol.

    Attributes
    ----------
    name : str
        The name of the metric.
    params : Dict[str, Any]
        The parameters of the metric.
    higher_is_better : Optional[bool]
        Whether larger values mean better performance.
    metric_units : Optional[str]
        The units of the metric.
    """

    name: str
    params: Dict[str, Any]
    higher_is_better: Optional[bool]
    metric_units: Optional[str]

    def __str__(self) -> str:
        """Return a human-readable representation of the metric result."""
        ...


@dataclass
class MetricResult:
    """
    A class to store the result of a single-value metric.

    Parameters
    ----------
    name : str
        The name of the metric.
    params : Dict[str, Any]
        The parameters of the metric.
    result : float | int
        The result of the metric.
    higher_is_better : Optional[bool]
        Whether larger values mean better performance.
    metric_units : Optional[str]
        The units of the metric.
    """

    name: str
    params: Dict[str, Any]
    result: float | int
    higher_is_better: Optional[bool] = None
    metric_units: Optional[str] = None

    def __post_init__(self) -> None:
        """Checker that metric_units and higher_is_better are consistent with the result."""
        if self.metric_units is None:
            object.__setattr__(self, "metric_units", self.params.get("metric_units"))
        if self.higher_is_better is None:
            object.__setattr__(self, "higher_is_better", self.params.get("higher_is_better"))

    def __str__(self) -> str:
        """
        Return a string representation of the MetricResult, including the name and the result.

        Returns
        -------
        str
            A string representation of the MetricResult.
        """
        units = f" {self.metric_units}" if self.metric_units else ""
        return f"{self.name}: {self.result}{units}"

    @classmethod
    def from_results_dict(
        cls,
        metric_name: str,
        metric_params: Dict[str, Any],
        results_dict: Dict[str, Any],
    ) -> MetricResultProtocol:
        """
        Create a MetricResult from a raw results dictionary.

        Parameters
        ----------
        metric_name : str
            The name of the metric.
        metric_params : Dict[str, Any]
            The parameters of the metric.
        results_dict : Dict[str, Any]
            The raw results dictionary.

        Returns
        -------
        MetricResult
            The MetricResult object.
        """
        assert metric_name in results_dict, f"Metric name {metric_name} not found in raw results"
        result = results_dict[metric_name]
        assert isinstance(result, (float, int)), f"Result for metric {metric_name} is not a float or int"
        return cls(metric_name, metric_params, result)


@dataclass
class CompositeMetricResult:
    """
    A class to store the result of a metric that returns multiple labeled scores.

    This is used for metrics where a single evaluation request produces
    scores for multiple entries, such as asynchronous metrics that
    return labeled scores for different settings / models.

    Parameters
    ----------
    name : str
        The name of the metric.
    params : Dict[str, Any]
        The parameters of the metric.
    result : Dict[str, float | int]
        A mapping of labels to scores.
    higher_is_better : Optional[bool]
        Whether larger values mean better performance.
    metric_units : Optional[str]
        The units of the metric.
    """

    name: str
    params: Dict[str, Any]
    result: Dict[str, float | int]
    higher_is_better: Optional[bool] = None
    metric_units: Optional[str] = None

    def __post_init__(self) -> None:
        """Resolve metric_units and higher_is_better from params if not explicitly provided."""
        if self.metric_units is None:
            object.__setattr__(self, "metric_units", self.params.get("metric_units"))
        if self.higher_is_better is None:
            object.__setattr__(self, "higher_is_better", self.params.get("higher_is_better"))

    def __str__(self) -> str:
        """
        Return a string representation of the CompositeMetricResult.

        Each labeled score is displayed on its own line.

        Returns
        -------
        str
            A string representation of the CompositeMetricResult.
        """
        lines = [f"{self.name}:"]
        for key, score in self.result.items():
            units = f" {self.metric_units}" if self.metric_units else ""
            lines.append(f"  {key}: {score}{units}")
        return "\n".join(lines)
