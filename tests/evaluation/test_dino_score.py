import torch
import pytest
from pruna.evaluation.metrics.metric_dino_score import DinoScore

def test_dino_score():
    """Test the DinoScore metric."""
    # Use CPU for testing
    metric = DinoScore(device="cpu")

    # Create dummy images (batch of 2 images, 3x224x224)
    x = torch.rand(2, 3, 224, 224)
    y = torch.rand(2,3, 224, 224)

    # Update metric
    metric.update(x, y, y)

    # Compute result
    result = metric.compute()

    assert result.name == "dino_score"
    assert isinstance(result.result, float)
    # Cosine similarity should be between -1 and 1
    assert -1.0 <= result.result <= 1.0
