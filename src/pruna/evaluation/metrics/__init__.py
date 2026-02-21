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

_EVALUATION_DEPS_ERROR = (
    "Evaluation dependencies not installed. "
    "Install with: pip install pruna[evaluation]"
)

try:
    from pruna.evaluation.metrics.registry import MetricRegistry  # isort:skip

    from pruna.evaluation.metrics.aesthetic_laion import AestheticLAION
    from pruna.evaluation.metrics.metric_clip_iqa import CLIPIQAMetric
    from pruna.evaluation.metrics.metric_clip_score import CLIPScoreMetric
    from pruna.evaluation.metrics.metric_cmmd import CMMD
    from pruna.evaluation.metrics.metric_dino_score import DinoScore
    from pruna.evaluation.metrics.metric_elapsed_time import LatencyMetric, ThroughputMetric, TotalTimeMetric
    from pruna.evaluation.metrics.metric_energy import CO2EmissionsMetric, EnergyConsumedMetric
    from pruna.evaluation.metrics.metric_fid import FIDMetric
    from pruna.evaluation.metrics.metric_memory import DiskMemoryMetric, InferenceMemoryMetric, TrainingMemoryMetric
    from pruna.evaluation.metrics.metric_model_architecture import TotalMACsMetric, TotalParamsMetric
    from pruna.evaluation.metrics.metric_pairwise_clip import PairwiseClipScore
    from pruna.evaluation.metrics.metric_sharpness import SharpnessMetric
    from pruna.evaluation.metrics.metric_torch import TorchMetricWrapper
except ImportError as e:
    raise ImportError(_EVALUATION_DEPS_ERROR) from e

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
    "DinoScore",
    "SharpnessMetric",
    "AestheticLAION",
    "CLIPScoreMetric",
    "FIDMetric",
    "CLIPIQAMetric",
]
