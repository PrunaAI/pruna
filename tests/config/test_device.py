import pytest
import torch
from pruna.config.smash_config import SmashConfig

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

@pytest.mark.cuda
def test_device_default() -> None:
    """Test that the default device is 'cuda' when CUDA is available."""
    smash_config = SmashConfig()
    assert smash_config.device == "mps" or smash_config.device == "cuda"


@pytest.mark.cpu
def test_device_cpu() -> None:
    """Test that setting device to 'cpu' works."""
    smash_config = SmashConfig(device="cpu")
    assert smash_config.device == "cpu"


@pytest.mark.cpu
def test_device_none() -> None:
    """Test that setting device to None defaults to 'cuda'."""
    smash_config = SmashConfig(device=None)
    assert smash_config.device == DEVICE

@pytest.mark.cuda
@pytest.mark.parametrize(
    "device,expected",
    [
        ("mps", "cpu") if torch.cuda.is_available() else ("cuda", "cpu"),
    ],
)
def test_device_available(device: str, expected: str) -> None:
    """Test that setting device to an unavailable device falls back to CPU."""
    smash_config = SmashConfig(device=device)
    assert smash_config.device == expected

@pytest.mark.cpu
def test_device_invalid() -> None:
    """Test that setting an invalid device raises a ValueError."""
    smash_config = SmashConfig(device="invalid_device")
    assert smash_config.device == "cpu"

