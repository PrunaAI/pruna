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
import shutil
from enum import Enum

try:
    from enum import member
except ImportError:
    # Python 3.10 compat: partial prevents Enum from treating functions as methods
    from functools import partial as member
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from pruna.config.smash_config import SmashConfig
from pruna.engine.load_artifacts import (
    LOAD_ARTIFACTS_FUNCTIONS,
    STATIC_FP8_DIFFUSERS_ARTIFACT_ATTRS,
    STATIC_FP8_DIFFUSERS_ARTIFACTS_FILENAME,
    STATIC_FP8_DIFFUSERS_ARTIFACTS_FUNCTION_NAME,
    iter_typed_linears,
)
from pruna.logging.logger import pruna_logger


def save_artifacts(model: Any, model_path: str | Path, smash_config: SmashConfig) -> None:
    """
    Save all configured artifacts for a model.

    This function is intended to be called *after* the main model save function
    (e.g. `save_pruna_model`). It iterates over
    `smash_config.save_artifacts_fns` and invokes each corresponding
    `SAVE_ARTIFACTS_FUNCTIONS` member. Each artifact saver is independent
    and is responsible for appending its own load function(s) to
    `smash_config.load_fns` as needed.

    Parameters
    ----------
    model : Any
        The model to save artifacts for.
    model_path : str | Path
        The directory where the model and its artifacts are saved.
    smash_config : SmashConfig
        The SmashConfig object containing the artifact save function names in
        `save_artifacts_fns`.
    """
    smash_config.load_artifacts_fns.clear()  # accumulate as we run the save artifact functions

    artifact_fns = getattr(smash_config, "save_artifacts_fns", [])
    for fn_name in artifact_fns:
        try:
            SAVE_ARTIFACTS_FUNCTIONS[fn_name](model, model_path, smash_config)
        except KeyError:
            pruna_logger.error(
                "Unknown artifact save function '%s' in smash_config.save_artifacts_fns; skipping.", fn_name
            )


def save_torch_artifacts(model: Any, model_path: str | Path, smash_config: SmashConfig) -> None:
    """
    Save the model by saving the torch artifacts.

    Parameters
    ----------
    model : Any
        The model to save.
    model_path : str | Path
        The directory to save the model to.
    smash_config : SmashConfig
        The SmashConfig object containing the save and load functions.
    """
    artifacts = torch.compiler.save_cache_artifacts()

    assert artifacts is not None
    artifact_bytes, _ = artifacts

    # check if the bytes are empty
    if artifact_bytes == b"\x00\x00\x00\x00\x00\x00\x00\x01":
        pruna_logger.error(
            "Model has not been run before. Please run the model before saving to construct the compilation graph."
        )

    artifact_path = Path(model_path) / "artifact_bytes.bin"
    artifact_path.write_bytes(artifact_bytes)

    smash_config.load_artifacts_fns.append(LOAD_ARTIFACTS_FUNCTIONS.torch_artifacts.name)


def save_moe_kernel_tuner_artifacts(model: Any, model_path: str | Path, smash_config: SmashConfig) -> None:
    """
    Move the tuned config from pruna cache into the model directory.

    Parameters
    ----------
    model : Any
        The model to save artifacts for.
    model_path : str | Path
        The directory where the model and its artifacts will be saved.
    smash_config : SmashConfig
        The SmashConfig object.

    Returns
    -------
    None
        This function does not return anything.
    """
    src_file = Path(smash_config.cache_dir) / "moe_kernel_tuner.json"
    dest_file = Path(model_path) / "moe_kernel_tuner.json"
    shutil.move(src_file, dest_file)

    # define here the load artifacts function
    smash_config.load_artifacts_fns.append(LOAD_ARTIFACTS_FUNCTIONS.moe_kernel_tuner_artifacts.name)


def save_module_attr_artifacts(
    model: Any,
    model_path: str | Path,
    smash_config: SmashConfig,
    *,
    linear_cls: type,
    attrs: tuple[str, ...],
    filename: str,
    load_fn_name: str,
) -> None:
    """
    Save named module attributes for all ``linear_cls`` modules to a safetensors file.

    Parameters
    ----------
    model : Any
        The model whose matching linear modules should be exported.
    model_path : str | Path
        Directory where the artifacts safetensors file will be written.
    smash_config : SmashConfig
        The SmashConfig whose ``load_artifacts_fns`` list will be updated with
        ``load_fn_name`` so the artifacts are restored on load.
    linear_cls : type
        The linear module class to match (e.g. ``StaticFp8Linear``).
    attrs : tuple[str, ...]
        Buffer or tensor attribute names to persist for each matched module.
    filename : str
        Basename of the safetensors file written under ``model_path``.
    load_fn_name : str
        Name of the corresponding load-artifacts function to append to
        ``smash_config.load_artifacts_fns``.
    """
    state_dict = {}
    for prefix, layer in iter_typed_linears(model, linear_cls):
        for attr in attrs:
            state_dict[f"{prefix}{attr}"] = getattr(layer, attr)

    save_file(state_dict, str(Path(model_path) / filename))
    smash_config.load_artifacts_fns.append(load_fn_name)


def save_static_fp8_diffusers_artifacts(model: Any, model_path: str | Path, smash_config: SmashConfig) -> None:
    """
    Save calibrated activation scales for all ``StaticFp8Linear`` modules.

    Parameters
    ----------
    model : Any
        The quantized model whose activation calibration state should be exported.
    model_path : str | Path
        Directory where the artifacts file is written.
    smash_config : SmashConfig
        The SmashConfig whose ``load_artifacts_fns`` is updated such that the artifacts are loaded.
    """
    from pruna.algorithms.static_fp8_diffusers.utils import StaticFp8Linear

    save_module_attr_artifacts(
        model,
        model_path,
        smash_config,
        linear_cls=StaticFp8Linear,
        attrs=STATIC_FP8_DIFFUSERS_ARTIFACT_ATTRS,
        filename=STATIC_FP8_DIFFUSERS_ARTIFACTS_FILENAME,
        load_fn_name=STATIC_FP8_DIFFUSERS_ARTIFACTS_FUNCTION_NAME,
    )


class SAVE_ARTIFACTS_FUNCTIONS(Enum):  # noqa: N801
    """
    Enumeration of *artifact* save functions.

    Artifact savers are called after the main model save function has run.
    They produce additional artifacts (e.g. compilation caches) to speed up
    warmup or make the inference before and after loading consistent.

    This enum provides callable functions for saving such artifacts.

    Parameters
    ----------
    value : callable
        The artifact save function to be called.
    names : str
        The name of the enum member.
    module : str
        The module where the enum is defined.
    qualname : str
        The qualified name of the enum.
    type : type
        The type of the enum.
    start : int
        The start index for auto-numbering enum values.
    boundary : enum.FlagBoundary or None
        Boundary handling mode used by the Enum functional API for Flag and
        IntFlag enums.

    Examples
    --------
    >>> SAVE_ARTIFACTS_FUNCTIONS.torch_artifacts(model, save_path, smash_config)
    # Torch artifacts saved alongside the main model
    """

    torch_artifacts = member(save_torch_artifacts)
    moe_kernel_tuner_artifacts = member(save_moe_kernel_tuner_artifacts)
    static_fp8_diffusers_artifacts = member(save_static_fp8_diffusers_artifacts)

    def __call__(self, *args, **kwargs) -> None:
        """
        Call the underlying save function.

        Parameters
        ----------
        args : Any
            The arguments to pass to the save function.
        kwargs : Any
            The keyword arguments to pass to the save function.
        """
        if self.value is not None:
            self.value(*args, **kwargs)
