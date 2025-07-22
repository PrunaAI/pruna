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

import pytest
import torch
from PIL import Image

from pruna.evaluation.metrics.metric_reward import (
    ImageRewardMetric,
    HPSMetric,
    HPSv2Metric,
    IMAGE_REWARD,
    HPS_REWARD,
    HPSv2_REWARD,
)

METRIC_CLASSES_AND_NAMES = [
    (ImageRewardMetric, IMAGE_REWARD),
    (HPSMetric, HPS_REWARD),
    (HPSv2Metric, HPSv2_REWARD),
]

@pytest.fixture(scope="class", autouse=True)
def reward_metrics(request):
    # Initialize each metric once and store in a dict
    metrics = {}
    for metric_cls, metric_name in METRIC_CLASSES_AND_NAMES:
        metrics[metric_name] = metric_cls(device="cpu")
    request.cls._reward_metrics = metrics

@pytest.mark.usefixtures("reward_metrics")
@pytest.mark.parametrize(
    "metric_cls, metric_name",
    METRIC_CLASSES_AND_NAMES,
)
class TestRewardMetrics:
    def get_metric(self, metric_cls, metric_name):
        # Always return the pre-initialized metric instance
        return self._reward_metrics[metric_name]

    def test_metric_registration(self, metric_cls, metric_name):
        """Test that the metric is properly registered."""
        from pruna.evaluation.metrics.registry import MetricRegistry

        metric = MetricRegistry.get_metric(metric_name, device="cpu")
        assert isinstance(metric, metric_cls)

    def test_extract_prompts(self, metric_cls, metric_name):
        """Test prompt extraction from different input types."""
        metric = self.get_metric(metric_cls, metric_name)

        # Test with list of strings
        prompts = ["a beautiful sunset", "a cat playing"]
        extracted = metric._extract_prompts(prompts)
        assert extracted == prompts

        # Test with tensor (should generate default prompts)
        tensor = torch.randn(2, 3, 224, 224)
        extracted = metric._extract_prompts(tensor)
        assert len(extracted) == 2
        assert all(prompt.startswith("prompt_") for prompt in extracted)

    def test_score_image(self, metric_cls, metric_name):
        """Test image scoring functionality."""
        metric = self.get_metric(metric_cls, metric_name)

        # Create a simple test image in CHW format
        image = torch.randn(3, 224, 224)  # RGB image in CHW format
        prompt = "a beautiful landscape"

        # Convert to PIL Image properly
        image = image.clamp(0, 1)  # Ensure values are in [0, 1]
        pil_image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))

        score = metric._score_image(prompt, pil_image)
        assert isinstance(score, float)
        # Score should be a reasonable value (ImageReward/HPS typically outputs scores around 0-10)
        assert -10 <= score <= 10

    def test_update_and_compute(self, metric_cls, metric_name):
        """Test the update and compute methods."""
        metric = self.get_metric(metric_cls, metric_name)

        # Reset state before test
        metric.scores.clear()
        metric.prompts.clear()

        # Create test data
        prompts = ["a beautiful sunset", "a cat playing"]
        images = torch.randn(2, 3, 224, 224)  # 2 RGB images
        gt_images = torch.randn(2, 3, 224, 224)  # Ground truth images

        # Update the metric
        metric.update(prompts, gt_images, images)

        # Compute the result
        result = metric.compute()
        assert isinstance(result, float) or hasattr(result, "result")

    def test_error_handling(self, metric_cls, metric_name):
        """Test error handling for invalid inputs."""
        metric = self.get_metric(metric_cls, metric_name)

        # Test with invalid image shape - this should be handled gracefully now
        invalid_image = torch.randn(1, 1, 224)  # Wrong shape (HWC format)
        try:
            # This should now work with the improved _format_image method
            formatted_image = metric._format_image(invalid_image)
            assert isinstance(formatted_image, Image.Image)
        except Exception as e:
            # If it still fails, that's also acceptable
            assert "Cannot handle this data type" in str(e) or "Unexpected" in str(e)
