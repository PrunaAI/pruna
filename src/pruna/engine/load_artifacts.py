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
from __future__ import annotations

from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

import torch

from pruna.config.smash_config import SmashConfig
from pruna.logging.logger import pruna_logger


def load_artifacts(model: Any, model_path: str | Path, smash_config: SmashConfig) -> None:
    """
    Load available artifacts.

    This function is intended to be called after the main model load function.
    It loads artifacts specific to different algorithms into the pre-loaded model.

    Parameters
    ----------
    model : Any
        The model to load the artifacts for.
    model_path : str | Path
        The directory to load the artifacts from.
    smash_config : SmashConfig
        The SmashConfig object containing the load and save functions.

    Returns
    -------
    None
        The function does not return anything.
    """
    artifact_fns = getattr(smash_config, "load_artifacts_fns", [])
    if not artifact_fns:
        return

    for fn_name in artifact_fns:
        # Only handle artifact loaders we explicitly know about here.
        if fn_name not in LOAD_ARTIFACTS_FUNCTIONS.__members__:
            continue

        LOAD_ARTIFACTS_FUNCTIONS[fn_name](model, model_path, smash_config)


def load_torch_artifacts(model: Any, model_path: str | Path, smash_config: SmashConfig) -> None:
    """
    Load a torch artifacts from the given model path.

    Parameters
    ----------
    model : Any
        The model to load the artifacts for.
    model_path : str | Path
        The directory to load the artifacts from.
    smash_config : SmashConfig
        The SmashConfig object containing the load and save functions.
    """
    artifact_path = Path(model_path) / "artifact_bytes.bin"
    if not artifact_path.exists():
        pruna_logger.error(f"No torch artifacts found at {artifact_path}; skipping torch artifact loading.")
        return

    pruna_logger.info(f"Loading torch artifacts from {artifact_path}")
    artifact_bytes = artifact_path.read_bytes()

    torch.compiler.load_cache_artifacts(artifact_bytes)


class LOAD_ARTIFACTS_FUNCTIONS(Enum):  # noqa: N801
    """
    Enumeration of *artifact* load functions.

    Artifact loaders are functions that are called *after* the main model load
    has completed. They attach additional runtime state to the already-loaded
    model (e.g. FP8 scales) or restore global caches.
    """

    torch_artifacts = partial(load_torch_artifacts)

    def __call__(self, *args, **kwargs) -> None:
        """Call the underlying load function."""
        if self.value is not None:
            self.value(*args, **kwargs)
