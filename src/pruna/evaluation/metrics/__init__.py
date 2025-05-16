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

from pruna.evaluation.metrics.registry import MetricRegistry  # isort:skip

from pruna.evaluation.metrics.metric_cmmd import CMMD
from pruna.evaluation.metrics.metric_elapsed_time import LatencyMetric, ThroughputMetric, TotalTimeMetric
from pruna.evaluation.metrics.metric_energy import CO2EmissionsMetric, EnergyConsumedMetric
from pruna.evaluation.metrics.metric_memory import DiskMemoryMetric, InferenceMemoryMetric, TrainingMemoryMetric
from pruna.evaluation.metrics.metric_model_architecture import TotalMACsMetric, TotalParamsMetric
from pruna.evaluation.metrics.metric_pairwise_clip import PairwiseClipScore
from pruna.evaluation.metrics.metric_torch import TorchMetricWrapper

__all__ = [
    "MetricRegistry",
    "TorchMetricWrapper",
    "TotalTimeMetric",
    "LatencyMetric",
    "ThroughputMetric",
    "EnergyConsumedMetric",
    "CO2EmissionsMetric",
    "DiskMemoryMetric",
    "TrainingMemoryMetric",
    "InferenceMemoryMetric",
    "TotalParamsMetric",
    "TotalMACsMetric",
    "PairwiseClipScore",
    "CMMD",
]
