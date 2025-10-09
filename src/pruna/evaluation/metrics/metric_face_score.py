
# Copyright (c) 2025 PrunaAI. All rights reserved.
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


import torch
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.evaluation.metrics.registry import MetricRegistry
from .FaceScore import FaceScore



@MetricRegistry.register("face_score")
class FaceScoreMetric(StatefulMetric):
    """
    FaceScoreMetric evaluates the quality of generated human faces using the FaceScore reward model.
    Relies on batch-face and image-reward for scoring.
    """
    metric_name = "face_score"
    higher_is_better = True
    default_call_type = "y"  # Only predictions are needed

    def __init__(self, call_type: str = SINGLE):
        super().__init__()
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.face_score = FaceScore("FaceScore")
        self.add_state("total_score", torch.tensor(0.0))
        self.add_state("count", torch.tensor(0))

    def update(self, inputs, ground_truths, predictions):
        """Update the metric state with a batch of predictions."""
        metric_data = metric_data_processor(inputs, ground_truths, predictions, self.call_type)
        images = metric_data[0]  # Should be a batch of image file paths or PIL images
        batch_score = 0.0
        batch_count = 0
        if not isinstance(images, (list, tuple)):
            images = [images]
        for img in images:
            # If img is a PIL image, save to temp file
            if hasattr(img, "save"):
                import tempfile
                import os
                tmp_path = None
                fd = None
                try:
                    fd, tmp_path = tempfile.mkstemp(suffix=".png")
                    os.close(fd)  # Close the file handle immediately
                    img.save(tmp_path)
                    scores, _, _ = self.face_score.get_reward(tmp_path)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
            elif isinstance(img, str):
                scores, _, _ = self.face_score.get_reward(img)
            else:
                continue
            # Accumulate scores for both scalars and iterables
            if isinstance(scores, (list, tuple)):
                batch_score += sum(scores)
                batch_count += len(scores)
            elif isinstance(scores, (int, float)):
                batch_score += scores
                batch_count += 1
        self.total_score += batch_score
        self.count += batch_count

    def compute(self):
        """Compute the final FaceScore metric value."""
        value = 0.0 if self.count == 0 else self.total_score.item() / self.count.item()
        params = self.__dict__.copy()
        return MetricResult(self.metric_name, params, value)
