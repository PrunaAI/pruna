from typing import Any

import pytest

from pruna.config.smash_config import SmashConfig
from pruna.engine.pruna_model import PrunaModel
from pruna.evaluation.metrics.metric_cmmd import CMMD


@pytest.mark.parametrize(
    "model_fixture, device, clip_model",
    [
        pytest.param("stable_diffusion_v1_4", "cuda", "openai/clip-vit-large-patch14-336", marks=pytest.mark.cuda),
        pytest.param("ddpm-cifar10", "cpu", "openai/clip-vit-large-patch14-336", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],
)
def test_cmmd(model_fixture: tuple[Any, SmashConfig], device: str, clip_model: str) -> None:
    """Test CMMD."""
    model, smash_config = model_fixture
    smash_config.device = device
    pruna_model = PrunaModel(model, smash_config=smash_config)

    metric = CMMD(clip_model_name=clip_model, device=device)

    batch = next(iter(smash_config.test_dataloader()))
    x, gt = batch
    outputs = pruna_model.run_inference(batch, device)

    # Calculate CMMD between model outputs and ground truth
    metric.update(x, gt, outputs)
    comparison_results = metric.compute().detach().cpu().numpy()

    metric.reset()

    # Calculate CMMD between ground truth and itself
    metric.update(x, gt, gt)
    self_comparison_results = metric.compute().detach().cpu().numpy()

    assert self_comparison_results == pytest.approx(0.0, abs=1e-2)
    assert comparison_results > self_comparison_results
