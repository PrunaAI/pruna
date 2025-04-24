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
from typing import Any, Dict


@dataclass
class MetricResult:
    """
    A class to store the results of a metric.

    Parameters
    ----------
    name : str
        The name of the metric.
    params : Dict[str, Any]
        The parameters of the metric.
    result : float | int
        The result of the metric.
    """

    name: str
    params: Dict[str, Any]
    result: float | int

    @classmethod
    def from_results_dict(
        cls,
        metric_name: str,
        metric_params: Dict[str, Any],
        results_dict: Dict[str, Any],
        benchmark_metric: bool = False,
    ) -> "MetricResult":
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
        benchmark_metric : bool
            Whether the metric is a benchmark metric.

        Returns
        -------
        MetricResult
            The MetricResult object.
        """
        assert metric_name in results_dict, f"Metric name {metric_name} not found in raw results"
        result = results_dict[metric_name]
        assert isinstance(result, (float, int)), f"Result for metric {metric_name} is not a float or int"
        metric_params["benchmark_metric"] = benchmark_metric
        return cls(metric_name, metric_params, result)
