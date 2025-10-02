import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np

import pytest

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.metrics.aesthetic_laion import AestheticLAION, CLIPVariantAesthetics


@pytest.mark.parametrize(
    "device, clip_model",
    [
        pytest.param("cpu", CLIPVariantAesthetics.vit_l_14, marks=pytest.mark.cpu),
        pytest.param("cpu", CLIPVariantAesthetics.vit_b_32, marks=pytest.mark.cpu),
        pytest.param("cpu", CLIPVariantAesthetics.vit_b_16, marks=pytest.mark.cpu),
        pytest.param("cuda", CLIPVariantAesthetics.vit_l_14, marks=pytest.mark.cuda),
    ],
)
def test_aesthetic_laion(device: str, clip_model: str) -> None:
    """Test the AestheticLAION metric."""
    data_module = PrunaDataModule.from_string("LAION256")
    data_module.limit_datasets(10)

    metric = AestheticLAION(clip_model_name=clip_model, device=device)
    for x, gt in data_module.test_dataloader():
        metric.update(x, gt, gt)

    score = metric.compute()
    assert score.result > 1.0 and score.result < 10.0


@pytest.mark.cpu
def test_metric_aesthetic_laion_ipynb_sample() -> None:
    """
    Test the aesthetic_laion metric with an image taken from
    https://github.com/LAION-AI/aesthetic-predictor/blob/main/asthetics_predictor.ipynb
    The result in the original notebook is 4.0330; if you rerun it, you will get 4.4425.
    The Hugging Face model, however, gives 5.049.
    """
    metric = AestheticLAION(device="cpu")
    response = requests.get("https://thumbs.dreamstime.com/b/lovely-cat-as-domestic-animal-view-pictures-182393057.jpg")
    img = np.array(Image.open(BytesIO(response.content)).convert("RGB"))
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous().unsqueeze(0)
    metric.update("lovely cat as domestic animal view pictures", img, img)
    score = metric.compute()
    assert abs(score.result - 5.05) < 1e-2
