import pytest
import torch

from pruna.evaluation.metrics.metric_mse import MSEMetric


def test_mse_basic():
    """MSE should be 0.25 for simple case."""
    metric = MSEMetric(device="cpu")
    pred = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    target = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
    
    metric.update(None, target, pred)
    result = metric.compute()
    
    assert abs(result.result - 0.25) < 1e-6


def test_mse_perfect():
    """Perfect match -> MSE = 0."""
    metric = MSEMetric(device="cpu")
    data = torch.randn(3, 4, 8, 8)
    
    metric.update(None, data, data)
    result = metric.compute()
    
    assert result.result < 1e-6


def test_mse_accumulation():
    """Accumulate multiple batches."""
    metric = MSEMetric(device="cpu")
    
    # Batch 1: errors of 1.0
    metric.update(None, torch.ones(2, 3), torch.zeros(2, 3))
    # Batch 2: errors of 2.0
    metric.update(None, torch.ones(2, 3) * 2, torch.zeros(2, 3))
    
    result = metric.compute()
    # MSE = (6*1 + 6*4) / 12 = 2.5
    assert abs(result.result - 2.5) < 1e-6


def test_mse_reset():
    """Reset clears state."""
    metric = MSEMetric()
    metric.update(None, torch.ones(2), torch.zeros(2))
    
    assert metric.compute().result > 0
    
    metric.reset()
    assert metric.n_elements.item() == 0


def test_mse_empty():
    """Empty state returns 0."""
    metric = MSEMetric()
    result = metric.compute()
    
    assert result.result == 0.0


def test_mse_properties():
    """Check metric config."""
    metric = MSEMetric()
    assert metric.metric_name == "mse"
    assert metric.higher_is_better is False
    assert "cpu" in metric.runs_on


def test_mse_3d():
    """Works with 3D tensors."""
    metric = MSEMetric(device="cpu")
    pred = torch.randn(2, 4, 4)
    target = pred + 0.1
    
    metric.update(None, target, pred)
    result = metric.compute()
    
    assert abs(result.result - 0.01) < 1e-5  # 0.1^2 = 0.01


def test_mse_4d():
    """Works with 4D tensors (images)."""
    metric = MSEMetric(device="cpu")
    pred = torch.randn(2, 3, 8, 8)
    target = pred + 0.2
    
    metric.update(None, target, pred)
    result = metric.compute()
    
    assert abs(result.result - 0.04) < 1e-5  # 0.2^2 = 0.04
