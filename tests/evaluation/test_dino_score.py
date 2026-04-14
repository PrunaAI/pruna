import torch
import pytest
from pruna.evaluation.metrics.metric_dino_score import DinoScore

DINO_MODELS = [
    "dino",
    pytest.param("dinov2_vits14", marks=pytest.mark.slow),
    pytest.param("dinov2_vitb14", marks=pytest.mark.slow),
    pytest.param("dinov2_vitl14", marks=pytest.mark.slow),
    pytest.param(
        "dinov3_vits16",
        marks=[
            pytest.mark.slow,
            pytest.mark.skip(reason="DINOv3 HF models are gated; requires access approval"),
        ],
    ),
    pytest.param(
        "dinov3_convnext_tiny",
        marks=[
            pytest.mark.slow,
            pytest.mark.skip(reason="DINOv3 HF models are gated; requires access approval"),
        ],
    ),
]


@pytest.mark.cpu
@pytest.mark.parametrize("model", DINO_MODELS)
def test_dino_score_models(model: str):
    """Test DinoScore with each supported backbone (dino, dinov2, dinov3)."""
    metric = DinoScore(device="cpu", model=model)
    x = torch.rand(2, 3, 224, 224)
    y = torch.rand(2, 3, 224, 224)
    metric.update(x, y, y)
    result = metric.compute()
    assert result.name == "dino_score"
    assert isinstance(result.result, float)
    assert -1.0 - 1e-5 <= result.result <= 1.0 + 1e-5


def test_dino_score_invalid_model():
    """Test that an unrecognised model key raises a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown DinoScore model"):
        DinoScore(device="cpu", model="facebook/dinov3-irrelevant-wrong-model")


def test_dino_score():
    """Test the DinoScore metric with default model (backward compatibility)."""
    metric = DinoScore(device="cpu")
    x = torch.rand(2, 3, 224, 224)
    y = torch.rand(2, 3, 224, 224)
    metric.update(x, y, y)
    result = metric.compute()
    assert result.name == "dino_score"
    assert isinstance(result.result, float)
    assert -1.0 - 1e-5 <= result.result <= 1.0 + 1e-5
