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
from collections.abc import Iterator
from enum import Enum

try:
    from enum import member
except ImportError:
    # Python 3.10 compat: partial prevents Enum from treating functions as methods
    from functools import partial as member
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from pruna.config.smash_config import SmashConfig
from pruna.engine.utils import get_nn_modules
from pruna.logging.logger import pruna_logger

STATIC_FP8_DIFFUSERS_ARTIFACTS_FUNCTION_NAME = "static_fp8_diffusers_artifacts"
STATIC_FP8_DIFFUSERS_ARTIFACTS_FILENAME = "static_fp8_diffusers_artifacts.safetensors"
# Calibrated activation state to persist. Weight-side buffers are re-derived deterministically
# from the float weights during resmash, so they are intentionally not saved here.
STATIC_FP8_DIFFUSERS_ARTIFACT_ATTRS = ("input_running_amax",)


def module_path_prefix(module_name: str | None, submodule_name: str | None) -> str:
    """
    Get the dotted key prefix for a (root, submodule) pair, with a trailing dot.

    Used symmetrically by save and load so that state-dict keys line up. The root
    name is ``None`` for a bare ``nn.Module`` and the component attribute name for a
    pipeline. The submodule name is ``""`` for the root module itself.

    Parameters
    ----------
    module_name : str | None
        The name of the top-level module.
    submodule_name : str | None
        The name of the submodule.

    Returns
    -------
    str
        The path prefix ending with a trailing dot, or an empty string if both names are None.
    """
    if module_name is None and submodule_name is None:
        return ""
    elif module_name is None:
        return f"{submodule_name}."
    elif not submodule_name:
        return f"{module_name}."
    else:
        return f"{module_name}.{submodule_name}."


def iter_typed_linears(model: Any, linear_cls: type) -> Iterator[tuple[str, Any]]:
    """
    Yield ``(key_prefix, module)`` for every module of ``linear_cls`` in the model (pipeline-aware).

    Parameters
    ----------
    model : Any
        The model (bare nn.Module or diffusers pipeline) to walk.
    linear_cls : type
        The linear module class to match (e.g. ``StaticFp8Linear``).

    Yields
    ------
    tuple[str, Any]
        The dotted key prefix and the matching linear module found at that path.
    """
    nn_modules: dict[str | None, torch.nn.Module]
    try:
        nn_modules = model.get_nn_modules()
    except AttributeError:
        nn_modules = get_nn_modules(model)

    for root_name, root in nn_modules.items():
        for submodule_name, submodule in root.named_modules():
            if isinstance(submodule, linear_cls):
                yield module_path_prefix(root_name, submodule_name), submodule


def iter_static_fp8_linears(model: Any) -> Iterator[tuple[str, Any]]:
    """
    Yield ``(key_prefix, module)`` for every ``StaticFp8Linear`` in the model (pipeline-aware).

    Parameters
    ----------
    model : Any
        The model (bare nn.Module or diffusers pipeline) to walk.

    Yields
    ------
    tuple[str, Any]
        The dotted key prefix and the ``StaticFp8Linear`` module found at that path.
    """
    # Local import to avoid importing the algorithm package unless artifacts are used.
    from pruna.algorithms.static_fp8_diffusers.utils import StaticFp8Linear

    yield from iter_typed_linears(model, StaticFp8Linear)


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


def load_moe_kernel_tuner_artifacts(model: Any, model_path: str | Path, smash_config: SmashConfig, **kwargs) -> Any:
    """
    Load a tuned kernel config inside the hf/vllm caches.

    Parameters
    ----------
    model : Any
        The model to load the artifacts for.
    model_path : str | Path
        The path to the model directory.
    smash_config : SmashConfig
        The SmashConfig object.
    **kwargs : Any
        Additional keyword arguments to pass to the function.

    Returns
    -------
    Any
        The loaded MoE model.
    """
    from pruna.algorithms.moe_kernel_tuner import MoeKernelTuner
    from pruna.algorithms.utils.moe_kernel_tuner import save_configs

    imported_packages = MoeKernelTuner().import_algorithm_packages()
    save_dir = Path(model_path)
    with open(save_dir / "moe_kernel_tuner.json") as f:
        best_configs_and_hyperparameters = json.load(f)
    if not best_configs_and_hyperparameters:
        raise ValueError(f"MoE kernel tuner artifacts not found in {save_dir}")
    else:
        # check if the triton version is the same as the one used to tune the kernel
        triton_version = best_configs_and_hyperparameters["triton_version"]
        if triton_version != imported_packages["triton"].__version__:
            msg = (
                f"Triton version mismatch: {triton_version} != "
                f"{imported_packages['triton'].__version__}. "
                "Performance may be degraded or config may be invalid. "
                "We recommend re-tuning the kernel in your environment."
            )
            pruna_logger.info(msg)

        best_configs = best_configs_and_hyperparameters["best_configs_moe_kernel"]
        num_experts = best_configs_and_hyperparameters["num_experts"]
        shard_intermediate_size = best_configs_and_hyperparameters["shard_intermediate_size"]
        dtype = best_configs_and_hyperparameters["dtype"]
        # Convert dtype string back to torch.dtype if needed
        dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        use_fp8_w8a8 = best_configs_and_hyperparameters["use_fp8_w8a8"]
        use_int8_w8a16 = best_configs_and_hyperparameters["use_int8_w8a16"]

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


def load_static_fp8_diffusers_artifacts(model: Any, model_path: str | Path, smash_config: SmashConfig) -> None:
    """
    Restore calibrated activation scales saved by ``save_static_fp8_diffusers_artifacts``.

    Parameters
    ----------
    model : Any
        The freshly re-quantized model whose activation scales should be restored.
    model_path : str | Path
        Directory the artifacts file is read from.
    smash_config : SmashConfig
        The SmashConfig (unused, kept for the artifact-loader signature).

    Returns
    -------
    None
        The function restores the activation scales in-place and does not return anything.

    Raises
    ------
    FileNotFoundError
        If the artifacts file is missing (calibration was skipped during load).
    ValueError
        If the artifacts file is incomplete for one or more quantized layers.
    """
    artifacts_path = Path(model_path) / STATIC_FP8_DIFFUSERS_ARTIFACTS_FILENAME
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"static_fp8_diffusers artifacts expected at '{artifacts_path}' but not found."
        )

    state_dict = load_file(str(artifacts_path))
    layers = list(iter_static_fp8_linears(model))
    if not layers:
        raise ValueError("No static_fp8_diffusers quantized layers found in the model.")

    missing_layers: list[str] = []
    for prefix, layer in layers:
        if any(f"{prefix}{attr}" not in state_dict for attr in STATIC_FP8_DIFFUSERS_ARTIFACT_ATTRS):
            missing_layers.append(prefix or "<root>")
            continue

        for attr in STATIC_FP8_DIFFUSERS_ARTIFACT_ATTRS:
            buffer = getattr(layer, attr)
            setattr(layer, attr, state_dict[f"{prefix}{attr}"].to(device=buffer.device, dtype=buffer.dtype))

        layer.freeze_input_scale()

    if missing_layers:
        raise ValueError(
            "static_fp8_diffusers artifacts are incomplete."
            "To use this artifact, exclude the following modules: " + ", ".join(missing_layers)
        )

    pruna_logger.info(f"Loaded static_fp8_diffusers artifacts from '{artifacts_path}'")


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

    torch_artifacts = member(load_torch_artifacts)
    moe_kernel_tuner_artifacts = member(load_moe_kernel_tuner_artifacts)
    static_fp8_diffusers_artifacts = member(load_static_fp8_diffusers_artifacts)

    def __call__(self, *args, **kwargs) -> None:
        """Call the underlying load function."""
        if self.value is not None:
            self.value(*args, **kwargs)
