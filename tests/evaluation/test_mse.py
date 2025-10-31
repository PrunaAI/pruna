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

import math

import pytest
import torch

from pruna.evaluation.metrics.metric_mse import MSEMetric


class TestMSEMetric:
    """Test suite for MSE metric."""

    def test_mse_perfect_match(self):
        """Test MSE when predictions match ground truth exactly."""
        metric = MSEMetric()

        gt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        outputs = torch.tensor([1.0, 2.0, 3.0, 4.0])

        metric.update(None, gt, outputs)
        result = metric.compute()

        assert result.result == 0.0, "MSE should be 0 for perfect match"
        assert result.name == "mse"
        assert not result.higher_is_better

    def test_mse_known_value(self):
        """Test MSE with known expected value."""
        metric = MSEMetric()

        gt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        outputs = torch.tensor([2.0, 3.0, 4.0, 5.0])  # All off by 1

        metric.update(None, gt, outputs)
        result = metric.compute()

        expected_mse = 1.0  # (1^2 + 1^2 + 1^2 + 1^2) / 4 = 1
        assert abs(result.result - expected_mse) < 1e-6, f"Expected MSE {expected_mse}, got {result.result}"

    def test_mse_multiple_batches(self):
        """Test MSE accumulation across multiple batches."""
        metric = MSEMetric()

        # First batch
        gt1 = torch.tensor([1.0, 2.0])
        outputs1 = torch.tensor([2.0, 3.0])
        metric.update(None, gt1, outputs1)

        # Second batch
        gt2 = torch.tensor([3.0, 4.0])
        outputs2 = torch.tensor([4.0, 5.0])
        metric.update(None, gt2, outputs2)

        result = metric.compute()

        # All differences are 1, so MSE = 1
        expected_mse = 1.0
        assert abs(result.result - expected_mse) < 1e-6

    def test_mse_empty_state(self):
        """Test MSE when no data is provided."""
        metric = MSEMetric()
        result = metric.compute()

        assert math.isnan(result.result), "MSE should be NaN when no data provided"

    def test_mse_multidimensional(self):
        """Test MSE with multidimensional tensors."""
        metric = MSEMetric()

        gt = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        outputs = torch.tensor([[1.5, 2.5], [3.5, 4.5]])

        metric.update(None, gt, outputs)
        result = metric.compute()

        # Each element differs by 0.5, so squared error = 0.25
        # Total: 4 * 0.25 / 4 = 0.25
        expected_mse = 0.25
        assert abs(result.result - expected_mse) < 1e-6

    def test_mse_3d_tensors(self):
        """Test MSE with 3D tensors (like images)."""
        metric = MSEMetric()

        # Simulate a batch of 2 small images (2, 3, 4, 4) - batch, channels, height, width
        gt = torch.randn(2, 3, 4, 4)
        outputs = gt + 0.1  # Add small noise

        metric.update(None, gt, outputs)
        result = metric.compute()

        # All differences are 0.1, so squared error = 0.01
        expected_mse = 0.01
        assert abs(result.result - expected_mse) < 1e-6

    def test_mse_reset(self):
        """Test that reset clears the metric state."""
        metric = MSEMetric()

        # First calculation
        gt = torch.tensor([1.0, 2.0, 3.0])
        outputs = torch.tensor([2.0, 3.0, 4.0])
        metric.update(None, gt, outputs)
        result1 = metric.compute()

        # Reset and calculate again
        metric.reset()
        gt = torch.tensor([0.0, 0.0])
        outputs = torch.tensor([0.0, 0.0])
        metric.update(None, gt, outputs)
        result2 = metric.compute()

        assert result1.result == 1.0
        assert result2.result == 0.0

    def test_mse_mixed_values(self):
        """Test MSE with mixed positive and negative errors."""
        metric = MSEMetric()

        gt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        outputs = torch.tensor([2.0, 1.0, 4.0, 3.0])  # +1, -1, +1, -1

        metric.update(None, gt, outputs)
        result = metric.compute()

        # All squared errors are 1
        expected_mse = 1.0
        assert abs(result.result - expected_mse) < 1e-6

    def test_mse_large_errors(self):
        """Test MSE with large errors."""
        metric = MSEMetric()

        gt = torch.tensor([0.0, 0.0, 0.0])
        outputs = torch.tensor([10.0, 10.0, 10.0])

        metric.update(None, gt, outputs)
        result = metric.compute()

        expected_mse = 100.0  # (10^2 + 10^2 + 10^2) / 3 = 100
        assert abs(result.result - expected_mse) < 1e-4

    def test_mse_none_handling(self):
        """Test that metric handles None inputs gracefully."""
        metric = MSEMetric()

        # This should not crash, just skip the update
        metric.update(None, None, None)
        result = metric.compute()

        assert math.isnan(result.result)

    @pytest.mark.cuda
    def test_mse_cuda(self):
        """Test MSE on CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = MSEMetric()

        gt = torch.tensor([1.0, 2.0, 3.0, 4.0]).cuda()
        outputs = torch.tensor([2.0, 3.0, 4.0, 5.0]).cuda()

        metric.update(None, gt, outputs)
        result = metric.compute()

        expected_mse = 1.0
        assert abs(result.result - expected_mse) < 1e-6

    def test_mse_device_mismatch(self):
        """Test MSE handles device mismatch between gt and outputs."""
        metric = MSEMetric()

        gt = torch.tensor([1.0, 2.0, 3.0, 4.0])  # CPU
        outputs = torch.tensor([2.0, 3.0, 4.0, 5.0])  # CPU

        # Should not crash - metric handles device movement
        metric.update(None, gt, outputs)
        result = metric.compute()

        expected_mse = 1.0
        assert abs(result.result - expected_mse) < 1e-6

    def test_mse_single_value(self):
        """Test MSE with single value."""
        metric = MSEMetric()

        gt = torch.tensor([5.0])
        outputs = torch.tensor([3.0])

        metric.update(None, gt, outputs)
        result = metric.compute()

        expected_mse = 4.0  # (5-3)^2 = 4
        assert abs(result.result - expected_mse) < 1e-6

    def test_mse_fractional_values(self):
        """Test MSE with fractional values."""
        metric = MSEMetric()

        gt = torch.tensor([0.1, 0.2, 0.3])
        outputs = torch.tensor([0.15, 0.25, 0.35])

        metric.update(None, gt, outputs)
        result = metric.compute()

        # All differences are 0.05, squared = 0.0025
        expected_mse = 0.0025
        assert abs(result.result - expected_mse) < 1e-8

    def test_mse_batch_independence(self):
        """Test that batches are processed independently."""
        metric1 = MSEMetric()
        metric2 = MSEMetric()

        # Process as one batch
        gt_full = torch.tensor([1.0, 2.0, 3.0, 4.0])
        outputs_full = torch.tensor([2.0, 3.0, 4.0, 5.0])
        metric1.update(None, gt_full, outputs_full)
        result1 = metric1.compute()

        # Process as two batches
        gt1 = torch.tensor([1.0, 2.0])
        outputs1 = torch.tensor([2.0, 3.0])
        metric2.update(None, gt1, outputs1)

        gt2 = torch.tensor([3.0, 4.0])
        outputs2 = torch.tensor([4.0, 5.0])
        metric2.update(None, gt2, outputs2)
        result2 = metric2.compute()

        # Results should be identical
        assert abs(result1.result - result2.result) < 1e-6
