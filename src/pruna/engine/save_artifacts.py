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
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

import torch

from pruna.config.smash_config import SmashConfig
from pruna.engine.load_artifacts import LOAD_ARTIFACTS_FUNCTIONS
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

    torch_artifacts = partial(save_torch_artifacts)

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
