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

import inspect
import json
import sys
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import diffusers
import torch
import transformers
from huggingface_hub import constants, snapshot_download
from tqdm.auto import tqdm as base_tqdm
from transformers import pipeline

from pruna import SmashConfig
from pruna.engine.load_artifacts import load_artifacts
from pruna.engine.utils import load_json_config, move_to_device, set_to_best_available_device
from pruna.logging.logger import pruna_logger

PICKLED_FILE_NAME = "optimized_model.pt"
SAVE_BEFORE_SMASH_CACHE_DIR = "save_before_smash"
PIPELINE_INFO_FILE_NAME = "pipeline_info.json"


def load_pruna_model(model_path: str | Path, **kwargs) -> tuple[Any, SmashConfig]:
    """
    Load a Pruna model from the given model path.

    Parameters
    ----------
    model_path : str | Path
        The path to the model directory.
    **kwargs : Any
        Additional keyword arguments to pass to the model loading function.

    Returns
    -------
    Any, SmashConfig
        The loaded model and its SmashConfig.
    """
    smash_config = SmashConfig()
    smash_config.load_from_json(model_path)
    if "torch_artifacts" in smash_config.load_fns:
        _convert_to_artifact(smash_config, model_path)
    # since the model was just loaded from a file, we do not need to prepare saving anymore
    smash_config._prepare_saving = False

    resmash_fn = kwargs.pop("resmash_fn", resmash)

    if len(smash_config.load_fns) == 0:
        raise ValueError("Load function has not been set.")

    if len(smash_config.load_fns) > 1:
        pruna_logger.error(f"Load functions not used: {smash_config.load_fns[1:]}")

    model = LOAD_FUNCTIONS[smash_config.load_fns[0]](model_path, smash_config, **kwargs)

    # check if there are any algorithms to reapply
    if any(algorithm is not None for algorithm in smash_config.reapply_after_load.values()):
        model = resmash_fn(model, smash_config)

    # load artifacts (e.g. speed up the warmup or make it more consistent)
    load_artifacts(model, model_path, smash_config)

    return model, smash_config


def _convert_to_artifact(smash_config: SmashConfig, model_path: str | Path) -> None:
    """Convert legacy 'torch_artifacts' entries to the new artifact-based fields.

    This handles older configs that still store 'torch_artifacts' under
    'save_fns' or 'load_fns' instead of the corresponding '*_artifacts_fns'
    fields.
    """
    updated = False

    if "torch_artifacts" in smash_config.save_fns:
        smash_config.save_fns.remove("torch_artifacts")
        smash_config.save_artifacts_fns.append("torch_artifacts")
        updated = True

    if "torch_artifacts" in smash_config.load_fns:
        smash_config.load_fns.remove("torch_artifacts")
        smash_config.load_artifacts_fns.append("torch_artifacts")
        updated = True

    if updated:
        pruna_logger.warning(
            "The legacy 'torch_artifacts' save/load function entry in your SmashConfig is deprecated; "
            "your config file has been updated automatically. If you downloaded this smashed model, "
            "please ask the provider to update their model by loading it once with a recent version "
            "of Pruna and re-uploading it."
        )
        smash_config.save_to_json(model_path)


def load_pruna_model_from_pretrained(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Union[str, Path, None] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Optional[Union[Dict, str]] = None,
    proxies: Optional[Dict] = None,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    force_download: bool = False,
    token: Optional[Union[bool, str]] = None,
    local_files_only: bool = False,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
    max_workers: int = 8,
    tqdm_class: Optional[base_tqdm] = None,
    headers: Optional[Dict[str, str]] = None,
    endpoint: Optional[str] = None,
    # Deprecated args
    local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
    resume_download: Optional[bool] = None,
    **kwargs,
) -> tuple[Any, SmashConfig]:
    """
    Load a Pruna model from the Hugging Face Hub.

    Parameters
    ----------
    repo_id : str
        The repository ID of the model.
    revision : str | None, optional
        The revision of the model.
    cache_dir : str | Path | None, optional
        The cache directory.
    local_dir : str | Path | None, optional
        The local directory.
    library_name : str | None, optional
        The library name.
    library_version : str | None, optional
        The library version.
    user_agent : str | Dict | None, optional
        The user agent.
    proxies : Dict | None, optional
        The proxies.
    etag_timeout : float, optional
        The etag timeout.
    force_download : bool, optional
        The force download.
    token : str | bool | None, optional
        The Hugging Face token.
    local_files_only : bool, optional
        The local files only.
    allow_patterns : List[str] | str | None, optional
        The allow patterns.
    ignore_patterns : List[str] | str | None, optional
        The ignore patterns.
    max_workers : int, optional
        The max workers.
    tqdm_class : tqdm | None, optional
        The tqdm class.
    headers : Dict[str, str] | None, optional
        The headers.
    endpoint : str | None, optional
        The endpoint.
    local_dir_use_symlinks : bool | Literal["auto"], optional
        The local dir use symlinks.
    resume_download : bool | None, optional
        The resume download.
    **kwargs : Any
        Additional keyword arguments to pass to the model loading function of Pruna.

    Returns
    -------
    tuple[Any, SmashConfig]
        The loaded model and its SmashConfig.
    """
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
        library_name=library_name,
        library_version=library_version,
        user_agent=user_agent,
        proxies=proxies,
        etag_timeout=etag_timeout,
        force_download=force_download,
        local_files_only=local_files_only,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_workers=max_workers,
        tqdm_class=tqdm_class,
        headers=headers,
        endpoint=endpoint,
        local_dir_use_symlinks=local_dir_use_symlinks,
        resume_download=resume_download,
    )
    return load_pruna_model(model_path=path, **kwargs)


def resmash(model: Any, smash_config: SmashConfig) -> Any:
    """
    Resmash a model after loading it.

    Parameters
    ----------
    model : Any
        The model to resmash.
    smash_config : SmashConfig
        The SmashConfig object containing the reapply_after_load algorithms.

    Returns
    -------
    Any
        The resmashed model.
    """
    # determine algorithms to reapply
    smash_config_subset = deepcopy(smash_config)
    for algorithm_name in smash_config.reapply_after_load:
        # hyperparameters for algorithms were copied or discarded upon setting to None
        smash_config_subset[algorithm_name] = True

    # if it isn't already imported, import smash
    if "pruna.smash" not in sys.modules:
        from pruna.smash import smash
    else:
        smash = sys.modules["pruna.smash"].smash

    return smash(model=model, smash_config=smash_config_subset)


def load_transformers_model(path: str | Path, smash_config: SmashConfig | None = None, **kwargs) -> Any:
    """
    Load a transformers model or pipeline from the given model path.

    Parameters
    ----------
    path : str | Path
        The path to the model directory.
    smash_config : SmashConfig
        The SmashConfig object containing the device and device_map. If a SmashConfig is not provided,
        it will default to "auto" for the device_map.
    **kwargs : Any
        Additional keyword arguments to pass to the model loading function.

    Returns
    -------
    AutoModel | pipeline
        The loaded model or pipeline.
    """
    if "torch_dtype" not in kwargs:
        # unless specified by the user, load the model in the same dtype as the base model
        kwargs["torch_dtype"] = "auto"

    path = Path(path)
    pipeline_info_path = path / PIPELINE_INFO_FILE_NAME
    if pipeline_info_path.exists():
        with pipeline_info_path.open("r") as f:
            pipeline_info = json.load(f)
        # transformers discards kwargs automatically, no need for filtering
        return pipeline(pipeline_info["task"], str(path), **kwargs)
    else:
        config_path = path / "config.json"
        with config_path.open("r") as f:
            config = json.load(f)
        architecture = config["architectures"][0]
        cls = getattr(transformers, architecture)
        # transformers discards kwargs automatically, no need for filtering
        device_map: str | None
        if smash_config is None:
            device_map = "auto"
        else:
            device = smash_config.device if smash_config.device != "cuda" else "cuda:0"
            device_map = smash_config.device_map if smash_config.device == "accelerate" else device
        return cls.from_pretrained(path, device_map=device_map, **kwargs)


def load_diffusers_model(path: str | Path, smash_config: SmashConfig | None = None, **kwargs) -> Any:
    """
    Load a diffusers model from the given model path.

    Parameters
    ----------
    path : str | Path
        The path to the model directory.
    smash_config : SmashConfig
        The SmashConfig object containing the device and device_map. If a SmashConfig is not provided,
        it will default to "auto" for the device_map.
    **kwargs : Any
        Additional keyword arguments to pass to the model loading function.

    Returns
    -------
    Any
        The loaded diffusers model.
    """
    path = Path(path)
    model_index_path = path / "model_index.json"

    if model_index_path.exists():
        # if it is a diffusers pipeline, it saves the model_index.json file
        with model_index_path.open("r") as f:
            model_index = json.load(f)
    else:
        # individual components like the unet or the vae are saved with a config.json file
        with (path / "config.json").open("r") as f:
            model_index = json.load(f)
    # make loading of model backward compatible with older versions
    # in newer versions the dtype is always saved in the model_config.json file
    try:
        dtype_info = load_json_config(path, "dtype_info.json")
        dtype = dtype_info["dtype"]
        dtype = getattr(torch, dtype)
    except (KeyError, FileNotFoundError):
        dtype = torch.float32

    # do not override user specified dtype
    if "torch_dtype" not in kwargs:
        kwargs["torch_dtype"] = dtype

    cls = getattr(diffusers, model_index["_class_name"])
    # diffusers discards kwargs automatically, no need for filtering
    # diffusers does not support device_maps as dicts at the moment, we load on cpu and cast it ourselves
    model = cls.from_pretrained(path, device_map=None, **kwargs)
    if smash_config is None:
        device = set_to_best_available_device(None)
        device_map = None
    else:
        device = smash_config.device
        device_map = smash_config.device_map
    move_to_device(model, device, device_map=device_map)
    return model


def load_pickled(path: str | Path, smash_config: SmashConfig, **kwargs) -> Any:
    """
    Load a pickled model from the given model path.

    Parameters
    ----------
    path : str | Path
        The path to the model directory.
    smash_config : SmashConfig
        The SmashConfig object containing the device and device_map.
    **kwargs : Any
        Additional keyword arguments to pass to the model loading function.

    Returns
    -------
    Any
        The loaded pickled model.
    """
    # torch load has a target device but no interface to reproduce an accelerate-distributed model, we first map to cpu
    target_device = "cpu" if smash_config.device == "accelerate" else smash_config.device
    model = torch.load(
        Path(path) / PICKLED_FILE_NAME,
        weights_only=False,
        map_location="cpu",
        **filter_load_kwargs(torch.load, kwargs),
    )
    # move to target device, for accelerate it will now distribute / cast
    move_to_device(model, target_device, device_map=smash_config.device_map)
    return model


def load_hqq(model_path: str | Path, smash_config: SmashConfig, **kwargs) -> Any:
    """
    Load a model quantized with HQQ from the given model path.

    Parameters
    ----------
    model_path : str | Path
        The path to the model directory.
    smash_config : SmashConfig
        The SmashConfig object containing the device and device_map.
    **kwargs : Any
        Additional keyword arguments to pass to the model loading function.

    Returns
    -------
    Any
        The loaded model.
    """
    from pruna.algorithms.hqq import HQQ

    algorithm_packages = HQQ().import_algorithm_packages()
    model_path = Path(model_path)

    if "compute_dtype" in kwargs:
        compute_dtype = kwargs.pop("compute_dtype")
    else:
        saved_smash_config = SmashConfig()
        saved_smash_config.load_from_json(model_path)
        compute_dtype = torch.float16 if saved_smash_config["hqq_compute_dtype"] == "torch.float16" else torch.bfloat16

    has_config = (model_path / "config.json").exists()
    is_janus_model = (
        has_config and load_json_config(model_path, "config.json")["architectures"][0] == "JanusForConditionalGeneration"
    )

    def load_quantized_model(quantized_path: str | Path) -> Any:
        try:  # Try to use pipeline for HF specific HQQ quantization
            quantized_model = algorithm_packages["HQQModelForCausalLM"].from_quantized(
                str(quantized_path),
                device=smash_config.device,
                **filter_load_kwargs(algorithm_packages["HQQModelForCausalLM"].from_quantized, kwargs),
            )
        except Exception:  # Default to generic HQQ pipeline if it fails
            pruna_logger.info("Could not load HQQ model using HQQModelForCausalLM, trying generic AutoHQQHFModel...")
            quantized_model = algorithm_packages["AutoHQQHFModel"].from_quantized(
                str(quantized_path),
                device=smash_config.device,
                compute_dtype=compute_dtype,
                **filter_load_kwargs(algorithm_packages["AutoHQQHFModel"].from_quantized, kwargs),
            )
        return quantized_model

    if (model_path / PIPELINE_INFO_FILE_NAME).exists():  # pipeline
        # load the pipeline with a fake model on meta device
        with (model_path / PIPELINE_INFO_FILE_NAME).open("r") as f:
            task = json.load(f)["task"]
        pipe = pipeline(task=task, model=model_path, model_kwargs={"device_map": "meta"})
        # load the quantized model
        pipe.model = load_quantized_model(model_path)
        move_to_device(pipe, smash_config.device)
        return pipe
    elif (model_path / "qmodel.pt").exists():  # the model itself was quantized
        return load_quantized_model(model_path)
    elif is_janus_model:
        model = transformers.JanusForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=compute_dtype, **kwargs
        )
        hqq_model_dir = model_path / "hqq_language_model"

        # Janus language model must be patched to match HQQ's causal LM assumption
        quantized_path = str(hqq_model_dir / "qmodel.pt")
        weights = torch.load(quantized_path, map_location="cpu", weights_only=True)
        is_already_patched = all(k == "lm_head" or k.startswith("model.") for k in weights)
        if not is_already_patched:
            # load the weight on cpu to rename attr -> model.attr,
            weights = {f"model.{k}": v for k, v in weights.items()}
            # artifically add a random lm_head to the weights.
            weights["lm_head"] = torch.nn.Linear(1024, 1024).state_dict()
            torch.save(weights, quantized_path)  # patch weights

        quantized_causal_lm = load_quantized_model(hqq_model_dir)
        model.model.language_model = quantized_causal_lm.model  # drop the lm_head and causal_lm wrapper
        # some weights of the language_model are not on the correct device, so we move it afterwards.
        move_to_device(model, smash_config.device)
        return model
    else:
        raise ValueError(
            f"Could not load HQQ model from {model_path}, only quantized models or pipelines are supported."
        )


def load_hqq_diffusers(path: str | Path, smash_config: SmashConfig, **kwargs) -> Any:
    """
    Load a diffusers model from the given model path.

    Parameters
    ----------
    path : str | Path
        The path to the model directory.
    smash_config : SmashConfig
        The SmashConfig object containing the device and device_map.
    **kwargs : Any
        Additional keyword arguments to pass to the model loading function.

    Returns
    -------
    Any
        The loaded diffusers model.
    """
    from pruna.algorithms.hqq_diffusers import (
        HQQDiffusers,
        construct_base_class,
    )

    pruna_logger.warning(
        "Currently HQQ can only load linear layers. So model (e.g. Sana) with separate torch.nn.Parameters or "
        "buffers will not be loaded correctly."
    )

    hf_quantizer = HQQDiffusers()
    auto_hqq_hf_diffusers_model = construct_base_class(hf_quantizer.import_algorithm_packages(), [])
    quantized_load_kwargs = filter_load_kwargs(auto_hqq_hf_diffusers_model.from_quantized, kwargs)

    path = Path(path)
    if "compute_dtype" not in kwargs and (path / "dtype_info.json").exists():
        dtype = getattr(torch, load_json_config(path, "dtype_info.json")["dtype"])
        kwargs["compute_dtype"] = dtype
        kwargs.setdefault("torch_dtype", dtype)

    qmodel_path = path / "qmodel.pt"
    if qmodel_path.exists():  # the whole model was quantized, not a pipeline, load it directly
        model = auto_hqq_hf_diffusers_model.from_quantized(path, **kwargs)
        # force dtype if specified, from_quantized does not set it properly
        if "torch_dtype" in kwargs:
            model.to(kwargs["torch_dtype"])
    else:  # save directory is for a pipeline, of which some components were quantized
        model_index = load_json_config(path, "model_index.json")

        # first we load each quantized component separately, models like wan-i2v can have multiple components
        # that can be quantized and saved separately.
        # by convention, each component has been saved in a directory f"{attr_name}_quantized".
        quantized_components: dict[str, Any] = {}
        for quantized_path in [qpath for qpath in path.iterdir() if qpath.name.endswith("_quantized")]:
            attr_name = quantized_path.name.replace("_quantized", "")

            # legacy behavior: backbone_quantized -> target the transformer or unet attribute
            if attr_name == "backbone":
                if "transformer" in model_index:
                    attr_name = "transformer"
                elif "unet" in model_index:
                    attr_name = "unet"

            quantized_component = auto_hqq_hf_diffusers_model.from_quantized(
                str(quantized_path), **quantized_load_kwargs
            )
            # force dtype if specified, from_quantized does not set it properly
            if "torch_dtype" in kwargs:
                quantized_component.to(kwargs["torch_dtype"])
            quantized_components[attr_name] = quantized_component

        # then we load the rest of the pipeline
        cls = getattr(diffusers, model_index["_class_name"])
        # from_pretrained accepts arbitrary kwargs, no need for filtering
        model = cls.from_pretrained(path, **quantized_components, **kwargs)
        # check if the components were correctly loaded or if some kwargs were ignored
        for attr_name, quantized_component in quantized_components.items():
            if getattr(model, attr_name) != quantized_component:
                # fallback: manually replace the loaded component with the quantized one
                pruna_logger.warning(
                    f"Loading pipeline with component {attr_name} failed, attaching the model by hand which may lead to"
                    "memory overhead, consider using pruna.engine.utils.safe_memory_cleanup() to free memory."
                )
                setattr(model, attr_name, quantized_component)
    # HQQ does not support direct loading on the correct device, so we move it afterwards
    move_to_device(model, smash_config.device, device_map=smash_config.device_map)
    return model


def load_moe_kernel_tuner(path: str | Path, smash_config: SmashConfig, **kwargs) -> Any:
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
    payload = getattr(smash_config, "artifacts", {}).get("moe_kernel_tuner")
    if not payload:
        pruna_logger.error(
            "MoE kernel tuner artifacts not found in SmashConfig. "
            "Ensure the tuner ran successfully before saving/loading."
        )
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
        smash_config.load_fns.remove(LOAD_FUNCTIONS.moe_kernel_tuner.name)
        return load_transformers_model(path, smash_config, **kwargs)


class LOAD_FUNCTIONS(Enum):  # noqa: N801
    """
    Enumeration of load functions for different model types.

    This enum provides callable functions for loading different types of models,
    including transformers, diffusers, pickled models, IPEX LLM models, and HQQ models.

    Parameters
    ----------
    value : callable
        The load function to be called.
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
    >>> LOAD_FUNCTIONS.transformers(model_path, smash_config)
    <Loaded transformer model>
    """

    transformers = partial(load_transformers_model)
    diffusers = partial(load_diffusers_model)
    pickled = partial(load_pickled)
    hqq = partial(load_hqq)
    hqq_diffusers = partial(load_hqq_diffusers)
    moe_kernel_tuner = partial(load_moe_kernel_tuner)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Call the load function.

        Parameters
        ----------
        args : Any
            The arguments to pass to the load function.
        kwargs : Any
            The keyword arguments to pass to the load function.

        Returns
        -------
        Any
            The result of the load function.
        """
        if self.value is not None:
            return self.value(*args, **kwargs)
        return None


def filter_load_kwargs(func: Callable, kwargs: dict) -> dict:
    """
    Filter out keyword arguments that cannot be passed to the given function.

    Only filters if the function does not accept arbitrary keyword arguments.

    Parameters
    ----------
    func : Callable
        The function to check the keyword arguments for.
    kwargs : dict
        The keyword arguments to filter.

    Returns
    -------
    dict
        The filtered keyword arguments.
    """
    # Get the function's signature
    signature = inspect.signature(func)

    # Check if function accepts arbitrary kwargs
    has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())

    if has_kwargs:
        return kwargs

    # Only filter if function doesn't accept arbitrary kwargs
    valid_params = set(signature.parameters.keys())
    valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    invalid_kwargs = {k: v for k, v in kwargs.items() if k not in valid_params}

    # Log the discarded kwargs
    if invalid_kwargs:
        pruna_logger.info(f"Discarded unused loading kwargs: {list(invalid_kwargs.keys())}")

    return valid_kwargs
