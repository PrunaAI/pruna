from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from pruna.config.smash_config import SmashConfig
from pruna.engine.load_artifacts import iter_typed_linears, module_path_prefix
from pruna.engine.save_artifacts import save_module_attr_artifacts


class DummyFp8Linear(torch.nn.Module):
    """Minimal stand-in for an FP8 linear module used in artifact unit tests."""

    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.register_buffer("calibration_value", torch.tensor(1.5))


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("module_name", "submodule_name", "expected"),
    [
        (None, None, ""),
        (None, "child", "child."),
        ("transformer", "", "transformer."),
        ("transformer", "blocks.0", "transformer.blocks.0."),
    ],
)
def test_module_path_prefix(module_name: str | None, submodule_name: str | None, expected: str) -> None:
    """
    Test dotted key prefixes used symmetrically by artifact save and load.

    Parameters
    ----------
    module_name : str | None
        The top-level module name, or None for a bare module tree.
    submodule_name : str | None
        The submodule name within the root module.
    expected : str
        The expected prefix ending with a trailing dot.
    """
    assert module_path_prefix(module_name, submodule_name) == expected


@pytest.mark.cpu
def test_iter_typed_linears_bare_module() -> None:
    """Test ``iter_typed_linears`` on a bare ``nn.Module`` tree without pipeline roots."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), DummyFp8Linear())
    layers = list(iter_typed_linears(model, DummyFp8Linear))
    assert len(layers) == 1
    prefix, layer = layers[0]
    assert prefix == "1."
    assert isinstance(layer, DummyFp8Linear)


@pytest.mark.cpu
def test_iter_typed_linears_pipeline_model() -> None:
    """Test ``iter_typed_linears`` on a model that exposes ``get_nn_modules``."""
    transformer = torch.nn.Sequential(DummyFp8Linear(), torch.nn.Linear(2, 2))
    unet = torch.nn.Sequential(torch.nn.Linear(2, 2), DummyFp8Linear())
    model = SimpleNamespace(
        get_nn_modules=lambda: {"transformer": transformer, "unet": unet},
    )
    layers = list(iter_typed_linears(model, DummyFp8Linear))
    assert len(layers) == 2
    prefixes = {prefix for prefix, _ in layers}
    assert prefixes == {"transformer.0.", "unet.1."}


@pytest.mark.cpu
def test_save_module_attr_artifacts(tmp_path: Path) -> None:
    """
    Test saving module attributes to safetensors and registering the load function.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest for writing artifact files.
    """
    transformer = torch.nn.Sequential(DummyFp8Linear(), torch.nn.Linear(2, 2))
    unet = torch.nn.Sequential(torch.nn.Linear(2, 2), DummyFp8Linear())
    model = SimpleNamespace(
        get_nn_modules=lambda: {"transformer": transformer, "unet": unet},
    )
    smash_config = SmashConfig()
    load_fn_name = "dummy_fp8_artifacts"

    save_module_attr_artifacts(
        model,
        tmp_path,
        smash_config,
        linear_cls=DummyFp8Linear,
        attrs=("calibration_value",),
        filename="dummy_fp8_artifacts.safetensors",
        load_fn_name=load_fn_name,
    )

    artifact_path = tmp_path / "dummy_fp8_artifacts.safetensors"
    assert artifact_path.exists()
    state_dict = load_file(str(artifact_path))
    assert set(state_dict) == {"transformer.0.calibration_value", "unet.1.calibration_value"}
    assert load_fn_name in smash_config.load_artifacts_fns
