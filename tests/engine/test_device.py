import pytest
import torch
from pruna.config.smash_config import SmashConfig
from pruna.engine.utils import move_to_device, get_device
from diffusers import StableDiffusionPipeline
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

@pytest.mark.cuda
def test_device_default() -> None:
    """Test that the default device is 'cuda' when CUDA is available."""
    smash_config = SmashConfig()
    assert smash_config.device == "mps" or smash_config.device == "cuda"


@pytest.mark.cpu
@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_device_cpu(device: str | torch.device) -> None:
    """Test that setting device to 'cpu' works."""
    smash_config = SmashConfig(device=device)
    assert smash_config.device == "cpu"


@pytest.mark.cpu
def test_device_none() -> None:
    """Test that setting device to None defaults to best available device."""
    smash_config = SmashConfig(device=None)
    assert smash_config.device == DEVICE


@pytest.mark.cuda
@pytest.mark.parametrize(
    "device,expected",
    [
        ("mps", "cuda") if torch.cuda.is_available() else ("cuda", "mps"),
        (torch.device("mps"), "cuda") if torch.cuda.is_available() else (torch.device("cuda"), "mps"),
    ],
)
def test_device_available(device: str | torch.device, expected: str) -> None:
    """Test that setting device to an unavailable device falls back to CPU."""
    smash_config = SmashConfig(device=device)
    assert smash_config.device == expected


@pytest.mark.cuda
@pytest.mark.parametrize(
    "input_device,target_device",
    [
        ("cpu", "cuda"),
        ("cuda", "cpu"),
    ],
)
def test_device_casting(input_device: str | torch.device, target_device: str | torch.device) -> None:
    """Test that the device can be cast to the target device."""
    model = torch.nn.Linear(10, 10)
    model.to(input_device)
    move_to_device(model, target_device)
    assert get_device(model) == target_device


@pytest.mark.parametrize(
    "target_device",
    [
        "cuda",
        "cpu",
    ],
)
def test_accelerate_to_single_casting(target_device: str | torch.device) -> None:
    """Test that the device can be cast to the target device."""
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", device_map="balanced")
    device_map = model.hf_device_map
    move_to_device(model, target_device)
    assert get_device(model) == target_device 
    move_to_device(model, "accelerate", device_map=device_map)
    assert get_device(model) == "accelerate" 
