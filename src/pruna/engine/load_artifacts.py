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

import json
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
    Load torch artifacts from the given model path.

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


def load_moe_kernel_tuner_artifacts(path: str | Path, smash_config: SmashConfig, **kwargs) -> Any:
    """
    Load a tuned kernel config inside the hf/vllm caches, then load the model.

    Parameters
    ----------
    path : str | Path
        The path to the model directory.
    smash_config : SmashConfig
        The SmashConfig object containing the best configs for the MoE kernel tuner.
    **kwargs : Any
        Additional keyword arguments to pass to the model loading function.

    Returns
    -------
    Any
        The loaded MoE model.
    """
    from pruna.algorithms.moe_kernel_tuner import MoeKernelTuner, save_configs

    imported_packages = MoeKernelTuner().import_algorithm_packages()
    save_dir = Path(path) / "moe_kernel_tuned_configs"
    with open(save_dir / "moe_kernel_tuner.json") as f:
        payload = json.load(f)
    if not payload:
        raise ValueError(f"MoE kernel tuner artifacts not found in {save_dir}")
    else:
        best_configs = payload["best_configs_moe_kernel"]
        num_experts = payload["num_experts"]
        shard_intermediate_size = payload["shard_intermediate_size"]
        dtype = payload["dtype"]
        # Convert dtype string back to torch.dtype if needed
        dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        use_fp8_w8a8 = payload["use_fp8_w8a8"]
        use_int8_w8a16 = payload["use_int8_w8a16"]

        # save the config attached to smash_config, inside the hf and vllm caches.
        save_configs(
            best_configs,
            num_experts,
            shard_intermediate_size,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a16,
            None,
            smash_config["moe_kernel_tuner_path_to_huggingface_hub_cache"],
            smash_config["moe_kernel_tuner_path_to_vllm_cache"],
            imported_packages,
        )


class LOAD_ARTIFACTS_FUNCTIONS(Enum):  # noqa: N801
    """
    Enumeration of *artifact* load functions.

    Artifact loaders are functions that are called after the main model load
    has completed. They attach additional runtime state to the already-loaded
    model (e.g. compilation cache).

    This enum provides callable functions for loading such artifacts.

    Parameters
    ----------
    value : callable
        The artifact load function to be called.
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
    >>> LOAD_ARTIFACTS_FUNCTIONS.torch_artifacts(model, model_path, smash_config)
    # Torch artifacts loaded into the current runtime
    """

    torch_artifacts = partial(load_torch_artifacts)
    moe_kernel_tuner_artifacts = partial(load_moe_kernel_tuner_artifacts)

    def __call__(self, *args, **kwargs) -> None:
        """Call the underlying load function."""
        if self.value is not None:
            self.value(*args, **kwargs)
