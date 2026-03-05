import torch
import pytest
from pruna.evaluation.metrics.metric_dino_score import DinoScore

DINO_MODELS = [
    pytest.param("dino", id="dino_v1"),
    pytest.param("dinov2_vits14", id="dinov2_vits14", marks=pytest.mark.slow),
    pytest.param("dinov2_vitb14", id="dinov2_vitb14", marks=pytest.mark.slow),
    pytest.param("dinov2_vitl14", id="dinov2_vitl14", marks=pytest.mark.slow),
    pytest.param(
        "dinov3_vits16",
        id="dinov3_vits16",
        marks=pytest.mark.skip(reason="requires timm>=1.0.20 and Meta weights"),
    ),
    pytest.param(
        "dinov3_vitb16",
        id="dinov3_vitb16",
        marks=pytest.mark.skip(reason="requires timm>=1.0.20 and Meta weights"),
    ),
    pytest.param(
        "dinov3_vitl16",
        id="dinov3_vitl16",
        marks=pytest.mark.skip(reason="requires timm>=1.0.20 and Meta weights"),
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
    assert -1.0 <= result.result <= 1.0


def test_dino_score():
    """Test the DinoScore metric with default model (backward compatibility)."""
    metric = DinoScore(device="cpu")
    x = torch.rand(2, 3, 224, 224)
    y = torch.rand(2, 3, 224, 224)
    metric.update(x, y, y)
    result = metric.compute()
    assert result.name == "dino_score"
    assert isinstance(result.result, float)
    assert -1.0 <= result.result <= 1.0
