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

from pruna.evaluation.metrics.metric_imagereward import ImageRewardMetric, IMAGE_REWARD


def test_metric_registration():
    """Test that the metric is properly registered."""
    from pruna.evaluation.metrics.registry import MetricRegistry

    metric = MetricRegistry.get_metric(IMAGE_REWARD, device="cpu")
    assert isinstance(metric, ImageRewardMetric)

def test_extract_prompts():
    """Test prompt extraction from different input types."""
    metric = ImageRewardMetric(device="cpu")

    # Test with list of strings
    prompts = ["a beautiful sunset", "a cat playing"]
    extracted = metric._extract_prompts(prompts)
    assert extracted == prompts

    # Test with tensor (should generate default prompts)
    tensor = torch.randn(2, 3, 224, 224)
    extracted = metric._extract_prompts(tensor)
    assert len(extracted) == 2
    assert all(prompt.startswith("prompt_") for prompt in extracted)


def test_score_image():
    """Test image scoring functionality."""
    metric = ImageRewardMetric(device="cpu")

    # Create a simple test image
    image = torch.randn(3, 224, 224)  # RGB image
    prompt = "a beautiful landscape"

    score = metric._score_image(prompt, image)
    assert isinstance(score, float)
    # Score should be a reasonable value (ImageReward typically outputs scores around 0-10)
    assert -10 <= score <= 10


def test_update_and_compute():
    """Test the update and compute methods."""
    metric = ImageRewardMetric(device="cpu")

    # Create test data
    prompts = ["a beautiful sunset", "a cat playing"]
    images = torch.randn(2, 3, 224, 224)  # 2 RGB images
    gt_images = torch.randn(2, 3, 224, 224)  # Ground truth images

    # Update the metric
    metric.update(prompts, gt_images, images)

    # Compute the result
    result = metric.compute()
    import pdb; pdb.set_trace()

def test_error_handling():
    """Test error handling for invalid inputs."""
    metric = ImageRewardMetric(device="cpu")

    # Test with invalid image shape
    invalid_image = torch.randn(1, 1, 224)  # Wrong shape
    score = metric._score_image("test prompt", invalid_image)
    assert score == 0.0  # Should return 0 for invalid inputs
