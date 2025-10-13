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

from typing import Any, List

import clip
import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms.functional import convert_image_dtype
from vbench.utils import clip_transform, init_submodules

from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import PAIRWISE, SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.evaluation.metrics.vbench_utils import VBenchMixin
from pruna.logging.logger import pruna_logger

METRIC_VBENCH_BACKGROUND_CONSISTENCY = "background_consistency"


@MetricRegistry.register(METRIC_VBENCH_BACKGROUND_CONSISTENCY)
class VBenchBackgroundConsistency(StatefulMetric, VBenchMixin):
    """
    Background Consistency metric for VBench.

    Parameters
    ----------
    *args : Any
        The arguments to pass to the metric.
    device : str | None
        The device to run the metric on.
    call_type : str
        The call type to use for the metric.
    **kwargs : Any
        The keyword arguments to pass to the metric.
    """

    metric_name: str = METRIC_VBENCH_BACKGROUND_CONSISTENCY
    default_call_type: str = "y"  # We just need the outputs
    higher_is_better: bool = True
    # https://github.com/Vchitect/VBench/blob/dc62783c0fb4fd333249c0b669027fe102696682/evaluate.py#L111
    # explicitly sets the device to cuda. We respect this here.
    runs_on: List[str] = ["cuda"]
    modality: List[str] = ["video"]
    # state
    similarity_scores: torch.Tensor
    n_samples: torch.Tensor

    def __init__(
        self,
        *args: Any,
        device: str | None = None,
        call_type: str = SINGLE,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if device is not None and str(device).split(":")[0] not in self.runs_on:
            pruna_logger.error(f"Unsupported device {device}; supported: {self.runs_on}")
            raise ValueError()

        if call_type == PAIRWISE:
            # VBench itself does not support pairwise.
            # We can work on this in the future.
            pruna_logger.error("VBench does not support pairwise metrics. Please use single mode.")
            raise ValueError()

        submodules_dict = init_submodules([METRIC_VBENCH_BACKGROUND_CONSISTENCY])
        model_path = submodules_dict[METRIC_VBENCH_BACKGROUND_CONSISTENCY][0]

        self.device = set_to_best_available_device(device)
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)

        self.clip_model, self.preprocessor = clip.load(model_path, device=self.device)
        self.video_transform = clip_transform(224)

        self.add_state("similarity_scores", torch.tensor(0.0))
        self.add_state("n_samples", torch.tensor(0))

    def update(self, x: List[str], gt: Any, outputs: Any) -> None:
        """
        Update the similarity scores for the batch.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data.
        gt : torch.Tensor
            The ground truth / cached images.
        outputs : torch.Tensor
            The output images.
        """
        outputs = metric_data_processor(x, gt, outputs, self.call_type, device=self.device)
        # Background consistency metric only supports a batch size of 1.
        # To support larger batch sizes, we stack the outputs.
        outputs = super().validate_batch(outputs[0])
        # This metric depends on the outputs being uint8.
        outputs = torch.stack([convert_image_dtype(output, dtype=torch.uint8) for output in outputs])
        outputs = torch.stack([self.video_transform(output) for output in outputs])
        features = torch.stack([self.clip_model.encode_image(output) for output in outputs])
        features = torch.stack([F.normalize(feature, dim=-1, p=2) for feature in features])

        # We vectorize the calculation to avoid for loops.
        first_feature = features[:, 0, ...].unsqueeze(1).repeat(1, features.shape[1] - 1, 1)

        similarity_to_first = F.cosine_similarity(first_feature, features[:, 1:, ...], dim=-1).clamp(min=0.0)
        similarity_to_prev = F.cosine_similarity(features[:, :-1, ...], features[:, 1:, ...], dim=-1).clamp(min=0.0)

        similarities = (similarity_to_first + similarity_to_prev) / 2

        # Update stats
        self.similarity_scores += similarities.sum().item()
        self.n_samples += similarities.numel()

    def compute(self) -> MetricResult:
        """
        Aggregate the final score.

        Returns
        -------
        MetricResult
            The final score.
        """
        score = self.similarity_scores / self.n_samples
        return MetricResult(self.metric_name, self.__dict__, score)

    def reset(self) -> None:
        """Reset the metric states."""
        self.similarity_scores = torch.tensor(0.0)
        self.n_samples = torch.tensor(0)
