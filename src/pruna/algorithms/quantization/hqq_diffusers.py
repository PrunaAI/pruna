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

import os
from typing import Any, Dict, Type

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from ConfigSpace import OrdinalHyperparameter

from pruna.algorithms.quantization import PrunaQuantizer
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import (
    get_diffusers_transformer_models,
    get_diffusers_unet_models,
)
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.utils import load_json_config
from pruna.logging.filter import SuppressOutput
from pruna.logging.logger import pruna_logger


class HQQDiffusersQuantizer(PrunaQuantizer):
    """
    Implement HQQ for Image-Gen models.

    Half-Quadratic Quantization (HQQ) leverages fast, robust optimization techniques for on-the-fly quantization,
    eliminating the need for calibration data and making it applicable to any model. This algorithm is specifically
    adapted for diffusers models.
    """

    algorithm_name: str = "hqq_diffusers"
    references: dict[str, str] = {
        "GitHub": "https://github.com/mobiusml/hqq",
        "Article": "https://mobiusml.github.io/hqq_blog/",
    }
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.hqq_diffusers
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda"]
    dataset_required: bool = False
    compatible_algorithms: dict[str, list[str]] = dict(
        factorizer=["qkv_diffusers"], cacher=["deepcache", "fastercache", "fora", "pab"], compiler=["torch_compile"]
    )

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            OrdinalHyperparameter(
                "weight_bits",
                sequence=[2, 4, 8],
                default_value=8,
                meta=dict(desc="Number of bits to use for quantization."),
            ),
            OrdinalHyperparameter(
                "group_size",
                sequence=[8, 16, 32, 64, 128],
                default_value=64,
                meta=dict(desc="Group size for quantization."),
            ),
            OrdinalHyperparameter(
                "backend",
                sequence=["gemlite", "bitblas", "torchao_int4", "marlin"],
                default_value="torchao_int4",
                meta=dict(desc="Backend to use for quantization."),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a unet-based or transformer-based diffusion model.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a diffusion model, False otherwise.
        """
        transformer_and_unet_models = get_diffusers_transformer_models() + get_diffusers_unet_models()

        if isinstance(model, tuple(transformer_and_unet_models)):
            return True

        if hasattr(model, "transformer") and isinstance(model.transformer, tuple(transformer_and_unet_models)):
            return True

        return hasattr(model, "unet") and isinstance(model.unet, tuple(transformer_and_unet_models))

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Quantize the model.

        Parameters
        ----------
        model : Any
            The model to quantize.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the quantization.

        Returns
        -------
        Any
            The quantized model.
        """
        imported_modules = self.import_algorithm_packages()

        if hasattr(model, "transformer"):
            # Collect all linear layers recursively
            linear_layers = find_module_layers_type(model.transformer, nn.Linear)
            # put them in the transformer.layers for HQQ
            model.transformer.layers = linear_layers
            working_model = model.transformer
        elif hasattr(model, "unet"):
            linear_layers = find_module_layers_type(model.unet, nn.Linear)
            model.unet.layers = linear_layers
            working_model = model.unet
        else:
            linear_layers = find_module_layers_type(model, nn.Linear)
            model.layers = linear_layers
            working_model = model

        if working_model._class_name == "SD3Transformer2DModel":
            pruna_logger.info(
                "Your are using SD3Transformer2DModel, please be aware that this transformer is not savable for now."
            )

        config = imported_modules["HqqConfig"](nbits=smash_config["weight_bits"], group_size=smash_config["group_size"])

        auto_hqq_hf_diffusers_model = construct_base_class(imported_modules)

        auto_hqq_hf_diffusers_model.quantize_model(
            working_model,
            quant_config=config,
            compute_dtype=next(iter(working_model.parameters())).dtype,
            device=smash_config["device"],
        )

        # Prepare the model for fast inference based on the backend, we use the conditions from the hqq documentation
        if (
            smash_config["backend"] == "torchao_int4"
            and smash_config["weight_bits"] == 4
            and next(iter(working_model.parameters())).dtype == torch.bfloat16
        ):
            imported_modules["prepare_for_inference"](working_model, backend="torchao_int4")
        elif (
            smash_config["backend"] == "gemlite"
            and smash_config["weight_bits"] in [4, 2, 1]
            and next(iter(working_model.parameters())).dtype == torch.float16
        ):
            imported_modules["prepare_for_inference"](working_model, backend="gemlite")
        elif (
            smash_config["backend"] == "bitblas"
            and smash_config["weight_bits"] in [4, 2]
            and next(iter(working_model.parameters())).dtype == torch.float16
        ):
            imported_modules["prepare_for_inference"](working_model, backend="bitblas")
        else:
            # We default to the torch backend if the input backend is not applicable
            imported_modules["prepare_for_inference"](working_model)

        if hasattr(model, "transformer"):
            model.transformer = working_model
        elif hasattr(model, "unet"):
            model.unet = working_model
            for layer in model.unet.up_blocks:
                if layer.upsamplers is not None:
                    layer.upsamplers[0].name = "conv"
        else:
            model = working_model
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a method packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        with SuppressOutput():
            from hqq.core.quantize import BaseQuantizeConfig
            from hqq.models.base import BaseHQQModel, BasePatch
            from hqq.utils.patching import prepare_for_inference

        import diffusers

        return dict(
            prepare_for_inference=prepare_for_inference,
            HqqConfig=BaseQuantizeConfig,
            BaseHQQModel=BaseHQQModel,
            BasePatch=BasePatch,
            diffusers=diffusers,
        )


def construct_base_class(imported_modules: Dict[str, Any]) -> Type[Any]:
    """
    Construct and return the AutoHQQHFDiffusersModel base class.

    Parameters
    ----------
    imported_modules : Dict[str, Any]
        Dictionary containing imported modules needed for the base class construction.

    Returns
    -------
    Type[AutoHQQHFDiffusersModel]
        The constructed AutoHQQHFDiffusersModel class.
    """

    class AutoHQQHFDiffusersModel(imported_modules["BaseHQQModel"], imported_modules["BasePatch"]):  # type: ignore
        """Base class for HQQ Hugging Face Diffusers models."""

        # Save model architecture
        @classmethod
        def cache_model(cls, model: Any, save_dir: str) -> None:
            """
            Cache the model configuration by saving it to disk.

            Parameters
            ----------
            model : Any
                The model whose configuration should be cached.
            save_dir : str
                Directory path where the model configuration will be saved.
            """
            model.save_config(save_dir)

        @classmethod
        def create_model(cls, save_dir: str, kwargs: dict) -> Any:
            """
            Create an empty model from the cached configuration.

            Parameters
            ----------
            save_dir : str
                Directory path where the model configuration is cached.
            kwargs : dict
                Additional keyword arguments for the model creation.

            Returns
            -------
            Any
                The created model.
            """
            model_kwargs: Dict[str, Any] = {}

            with init_empty_weights():
                # recover class from save_dir
                config = load_json_config(save_dir, "config.json")
                model_class = getattr(imported_modules["diffusers"], config["_class_name"])
                model = model_class.from_config(save_dir, **model_kwargs)

            return model

        @classmethod
        def save_quantized(cls, model: Any, save_dir: str) -> None:
            """
            Save the quantized model with additional handling for missing parameters.

            Parameters
            ----------
            model : Any
                The model to save.
            save_dir : str
                Directory path where the model will be saved.
            """
            # FIRST: Capture original parameter values BEFORE calling parent save
            # This ensures we save actual data, not meta tensors
            pruna_logger.debug("Capturing original parameter values before HQQ save...")
            original_params = {}
            original_buffers = {}
            for name, param in model.named_parameters():
                if str(param.device) != "meta":
                    # Save real parameter data
                    original_params[name] = param.detach().cpu().clone()
                    pruna_logger.debug(f"Captured original parameter: {name}")
                else:
                    # For meta tensors, we can't save the data, but we save the placeholder
                    original_params[name] = param
                    pruna_logger.debug(f"Captured meta tensor parameter: {name}")
            for name, buffer in model.named_buffers():
                if str(buffer.device) != "meta":
                    # Save real buffer data
                    original_buffers[name] = buffer.detach().cpu().clone()
                    pruna_logger.debug(f"Captured original buffer: {name}")
                else:
                    # For meta tensors, we can't save the data, but we save the placeholder
                    original_buffers[name] = buffer
                    pruna_logger.debug(f"Captured meta tensor buffer: {name}")

            # Call the parent save_quantized method
            super(AutoHQQHFDiffusersModel, cls).save_quantized(model, save_dir)

            # Preserve all non-linear layer parameters that HQQ might have missed
            qmodel_path = os.path.join(save_dir, "qmodel.pt")
            if os.path.exists(qmodel_path):
                # Load what HQQ saved
                hqq_saved_state = torch.load(qmodel_path, map_location="cpu")
                # Find missing parameters and buffers using our captured original values
                missing_params = {}
                missing_buffers = {}
                # Track which items are parameters vs buffers
                param_names = set()
                buffer_names = set()
                for name, param in original_params.items():
                    if name not in hqq_saved_state:
                        # Use original captured values (real data when available)
                        missing_params[name] = param
                        param_names.add(name)
                        if str(param.device) == "meta":
                            pruna_logger.debug(f"Adding meta tensor parameter to qmodel.pt: {name}")
                        else:
                            pruna_logger.debug(f"Adding missing parameter to qmodel.pt: {name}")
                for name, buffer in original_buffers.items():
                    if name not in hqq_saved_state:
                        # Use original captured values (real data when available)
                        missing_buffers[name] = buffer
                        buffer_names.add(name)
                        if str(buffer.device) == "meta":
                            pruna_logger.debug(f"Adding meta tensor buffer to qmodel.pt: {name}")
                        else:
                            pruna_logger.debug(f"Adding missing buffer to qmodel.pt: {name}")
                # Merge missing parameters/buffers back into the saved state
                if missing_params or missing_buffers:
                    combined_state = {**hqq_saved_state, **missing_params, **missing_buffers}
                    # Save metadata about what are parameters vs buffers
                    combined_state["_pruna_param_names"] = list(param_names)
                    combined_state["_pruna_buffer_names"] = list(buffer_names)
                    # ALSO save ALL original parameters in case HQQ loads them as meta tensors
                    # Use a separate key to avoid conflicts
                    combined_state["_pruna_all_original_params"] = original_params
                    combined_state["_pruna_all_original_buffers"] = original_buffers
                    torch.save(combined_state, qmodel_path)
                    pruna_logger.info(
                        f"Added {len(missing_params)} missing parameters and "
                        f"{len(missing_buffers)} missing buffers to HQQ saved model"
                    )

        @classmethod
        def from_quantized(cls, save_dir: str, **kwargs) -> Any:
            """
            Load the quantized model with additional handling for missing parameters.

            Parameters
            ----------
            save_dir : str
                Directory path where the model is saved.
            **kwargs : Any
                Additional keyword arguments.

            Returns
            -------
            Any
                The loaded model with restored missing parameters.
            """
            # Call the parent from_quantized method
            model = super(AutoHQQHFDiffusersModel, cls).from_quantized(save_dir, **kwargs)

            # Load and restore additional parameters that HQQ didn't handle
            qmodel_path = os.path.join(save_dir, "qmodel.pt")
            if os.path.exists(qmodel_path):
                pruna_logger.info("Loading additional parameters from qmodel.pt...")
                full_saved_state = torch.load(qmodel_path, map_location="cpu", weights_only=True)
                # Get metadata about what was saved
                saved_param_names = full_saved_state.get("_pruna_param_names", set())
                saved_buffer_names = full_saved_state.get("_pruna_buffer_names", set())
                # Get backup of ALL original parameters (for meta tensor cleanup)
                all_original_params = full_saved_state.get("_pruna_all_original_params", {})
                all_original_buffers = full_saved_state.get("_pruna_all_original_buffers", {})
                # Find parameters and buffers that were saved but not loaded by HQQ
                missing_params = {}
                missing_buffers = {}
                # Process parameters from metadata first
                for name in saved_param_names:
                    if name in full_saved_state and isinstance(full_saved_state[name], torch.Tensor):
                        # Skip HQQ quantization-related parameters - let HQQ handle these
                        if any(hqq_key in name for hqq_key in ["W_q", "scale", "zero", "meta", "bias_q"]):
                            continue
                        missing_params[name] = full_saved_state[name]
                # Process buffers from metadata
                for name in saved_buffer_names:
                    if name in full_saved_state and isinstance(full_saved_state[name], torch.Tensor):
                        # Skip HQQ quantization-related buffers
                        if any(hqq_key in name for hqq_key in ["W_q", "scale", "zero", "meta", "bias_q"]):
                            continue
                        missing_buffers[name] = full_saved_state[name]
                # Fallback: if no metadata, use old logic (for backward compatibility)
                if not saved_param_names and not saved_buffer_names:
                    hqq_loaded_state = {name for name, _ in model.named_parameters()}
                    hqq_loaded_buffers = {name for name, _ in model.named_buffers()}
                    for name, tensor in full_saved_state.items():
                        if name.startswith("_pruna_"):
                            continue
                        if (
                            isinstance(tensor, torch.Tensor)
                            and name not in hqq_loaded_state
                            and name not in hqq_loaded_buffers
                        ):
                            if tensor.requires_grad:
                                missing_params[name] = tensor
                            else:
                                missing_buffers[name] = tensor

                # Apply missing parameters and buffers to the loaded model
                if missing_params or missing_buffers:
                    pruna_logger.info(
                        f"Loading {len(missing_params)} missing parameters and "
                        f"{len(missing_buffers)} missing buffers that HQQ didn't handle"
                    )

                    # Convert to proper device and dtype
                    target_device = next(model.parameters()).device
                    target_dtype = next(model.parameters()).dtype
                    restored_params = 0
                    restored_buffers = 0

                    # Restore missing parameters
                    for name, param in missing_params.items():
                        try:
                            # Navigate to the correct module and set the parameter
                            module = model
                            parts = name.split(".")
                            navigation_successful = True
                            for part in parts[:-1]:
                                if hasattr(module, part):
                                    module = getattr(module, part)
                                else:
                                    pruna_logger.debug(f"Could not find module path for parameter: {name}")
                                    navigation_successful = False
                                    break
                            if navigation_successful:
                                param_name = parts[-1]
                                # Handle both meta tensors and real tensors
                                if str(param.device) == "meta":
                                    pruna_logger.debug(f"Creating empty tensor for meta parameter: {name}")
                                    # For meta tensors, create empty tensor with same shape
                                    param_tensor = torch.empty(param.shape, device=target_device, dtype=param.dtype)
                                else:
                                    pruna_logger.debug(f"Using real data for parameter: {name}")
                                    # For real tensors, use the actual data
                                    param_tensor = param.to(device=target_device, dtype=target_dtype)
                                if not isinstance(param_tensor, torch.nn.Parameter):
                                    param_tensor = torch.nn.Parameter(param_tensor, requires_grad=False)
                                setattr(module, param_name, param_tensor)
                                restored_params += 1
                                pruna_logger.debug(f"Restored parameter: {name}")
                        except Exception as e:
                            pruna_logger.debug(f"Failed to restore parameter {name}: {e}")
                            continue

                    # Restore missing buffers
                    for name, buffer in missing_buffers.items():
                        try:
                            # Navigate to the correct module and set the buffer
                            module = model
                            parts = name.split(".")
                            navigation_successful = True
                            for part in parts[:-1]:
                                if hasattr(module, part):
                                    module = getattr(module, part)
                                else:
                                    pruna_logger.debug(f"Could not find module path for buffer: {name}")
                                    navigation_successful = False
                                    break
                            if navigation_successful:
                                buffer_name = parts[-1]
                                # Handle both meta tensors and real tensors
                                if str(buffer.device) == "meta":
                                    pruna_logger.debug(f"Creating empty tensor for meta buffer: {name}")
                                    # For meta tensors, create empty tensor with same shape
                                    buffer_tensor = torch.empty(buffer.shape, device=target_device, dtype=buffer.dtype)
                                else:
                                    pruna_logger.debug(f"Using real data for buffer: {name}")
                                    # For real tensors, use the actual data
                                    buffer_tensor = buffer.to(device=target_device, dtype=target_dtype)
                                module.register_buffer(buffer_name, buffer_tensor)
                                restored_buffers += 1
                                pruna_logger.debug(f"Restored buffer: {name}")
                        except Exception as e:
                            pruna_logger.debug(f"Failed to restore buffer {name}: {e}")
                            continue
            # Comprehensive meta tensor cleanup with actual parameter values
            cls._cleanup_meta_tensors(model, missing_params, missing_buffers, all_original_params, all_original_buffers)

            return model

        @classmethod
        def _cleanup_meta_tensors(
            cls,
            model: Any,
            missing_params: Dict[str, Any],
            missing_buffers: Dict[str, Any],
            all_original_params: Dict[str, Any],
            all_original_buffers: Dict[str, Any],
        ) -> None:
            """
            Clean up any remaining meta tensors in the model.

            Parameters
            ----------
            model : Any
                The model to clean up.
            missing_params : Dict[str, Any]
                Dictionary of parameters that were missing and restored.
            missing_buffers : Dict[str, Any]
                Dictionary of buffers that were missing and restored.
            all_original_params : Dict[str, Any]
                Dictionary of ALL original parameters (backup for meta tensor cleanup).
            all_original_buffers : Dict[str, Any]
                Dictionary of ALL original buffers (backup for meta tensor cleanup).
            """
            try:
                # Find a non-meta device from the model
                target_device = None
                for param in model.parameters():
                    if str(param.device) != "meta":
                        target_device = param.device
                        break
                # If all parameters are on meta, default to cuda:0
                if target_device is None:
                    target_device = torch.device("cuda:0")

                meta_tensors_found = 0
                replacements_made = 0

                def is_tensor_on_meta(obj):
                    """Check if an object is a tensor on meta device."""
                    return isinstance(obj, torch.Tensor) and str(obj.device) == "meta"

                def create_real_tensor_from_saved(meta_tensor, name=""):
                    """Create a real tensor from saved values or fallback to proper empty tensor."""
                    saved_value = None
                    if name in missing_params:
                        saved_value = missing_params[name]
                        pruna_logger.debug(f"Using saved parameter value for {name} (exact match in missing_params)")
                    elif name in missing_buffers:
                        saved_value = missing_buffers[name]
                        pruna_logger.debug(f"Using saved buffer value for {name} (exact match in missing_buffers)")
                    else:
                        # Try without the "model." prefix in missing_params
                        name_without_model = name.replace("model.", "", 1) if name.startswith("model.") else name
                        if name_without_model in missing_params:
                            saved_value = missing_params[name_without_model]
                            pruna_logger.debug(
                                f"Using saved parameter value for {name} -> "
                                f"{name_without_model} (prefix match in missing_params)"
                            )
                        elif name_without_model in missing_buffers:
                            saved_value = missing_buffers[name_without_model]
                            pruna_logger.debug(
                                f"Using saved buffer value for {name} -> "
                                f"{name_without_model} (prefix match in missing_buffers)"
                            )
                        else:
                            # Try adding "model." prefix in missing_params
                            name_with_model = f"model.{name}" if not name.startswith("model.") else name
                            if name_with_model in missing_params:
                                saved_value = missing_params[name_with_model]
                                pruna_logger.debug(
                                    f"Using saved parameter value for {name} -> "
                                    f"{name_with_model} (prefix match in missing_params)"
                                )
                            elif name_with_model in missing_buffers:
                                saved_value = missing_buffers[name_with_model]
                                pruna_logger.debug(
                                    f"Using saved buffer value for {name} -> "
                                    f"{name_with_model} (prefix match in missing_buffers)"
                                )

                    # If not found in missing_params, try the backup all_original data
                    if saved_value is None:
                        # Try exact name match in backup data
                        if name in all_original_params:
                            saved_value = all_original_params[name]
                            pruna_logger.debug(f"Using backup parameter value for {name} (exact match in backup)")
                        elif name in all_original_buffers:
                            saved_value = all_original_buffers[name]
                            pruna_logger.debug(f"Using backup buffer value for {name} (exact match in backup)")
                        else:
                            # Try without the "model." prefix in backup data
                            name_without_model = name.replace("model.", "", 1) if name.startswith("model.") else name
                            if name_without_model in all_original_params:
                                saved_value = all_original_params[name_without_model]
                                pruna_logger.debug(
                                    f"Using backup parameter value for {name} -> "
                                    f"{name_without_model} (prefix match in backup)"
                                )
                            elif name_without_model in all_original_buffers:
                                saved_value = all_original_buffers[name_without_model]
                                pruna_logger.debug(
                                    f"Using backup buffer value for {name} -> "
                                    f"{name_without_model} (prefix match in backup)"
                                )
                            else:
                                # Try adding "model." prefix in backup data
                                name_with_model = f"model.{name}" if not name.startswith("model.") else name
                                if name_with_model in all_original_params:
                                    saved_value = all_original_params[name_with_model]
                                    pruna_logger.debug(
                                        f"Using backup parameter value for {name} -> "
                                        f"{name_with_model} (prefix match in backup)"
                                    )
                                elif name_with_model in all_original_buffers:
                                    saved_value = all_original_buffers[name_with_model]
                                    pruna_logger.debug(
                                        f"Using backup buffer value for {name} -> "
                                        f"{name_with_model} (prefix match in backup)"
                                    )
                    if saved_value is not None and str(saved_value.device) != "meta":
                        # Use the actual saved value
                        if isinstance(meta_tensor, torch.nn.Parameter):
                            real_tensor = saved_value.to(device=target_device, dtype=meta_tensor.dtype)
                            return torch.nn.Parameter(real_tensor, requires_grad=meta_tensor.requires_grad)
                        else:
                            return saved_value.to(device=target_device, dtype=meta_tensor.dtype)
                    else:
                        # Fallback to empty tensor if no saved value found
                        if isinstance(meta_tensor, torch.nn.Parameter):
                            real_tensor = meta_tensor.new_empty(
                                meta_tensor.shape, device=target_device, dtype=meta_tensor.dtype
                            )
                            return torch.nn.Parameter(real_tensor, requires_grad=meta_tensor.requires_grad)
                        else:
                            return meta_tensor.new_empty(
                                meta_tensor.shape, device=target_device, dtype=meta_tensor.dtype
                            )

                def scan_and_fix_module(module, path=""):
                    """Scan and fix meta tensors in a module."""
                    nonlocal meta_tensors_found, replacements_made
                    # Check and fix all named parameters
                    params_to_replace = []
                    for name, param in module.named_parameters(recurse=False):
                        if is_tensor_on_meta(param):
                            meta_tensors_found += 1
                            full_name = f"{path}.{name}" if path else name
                            params_to_replace.append((name, param, full_name))
                    for name, param, full_name in params_to_replace:
                        replacement = create_real_tensor_from_saved(param, full_name)
                        setattr(module, name, replacement)
                        replacements_made += 1

                    # Check and fix all named buffers
                    buffers_to_replace = []
                    for name, buffer in module.named_buffers(recurse=False):
                        if is_tensor_on_meta(buffer):
                            meta_tensors_found += 1
                            full_name = f"{path}.{name}" if path else name
                            buffers_to_replace.append((name, buffer, full_name))
                    for name, buffer, full_name in buffers_to_replace:
                        replacement = create_real_tensor_from_saved(buffer, full_name)
                        module.register_buffer(name, replacement)
                        replacements_made += 1

                    # Check other tensor attributes carefully
                    attrs_to_replace = []
                    for attr_name in list(dir(module)):
                        if attr_name.startswith("__") or attr_name in ["_parameters", "_buffers", "_modules"]:
                            continue
                        try:
                            attr = getattr(module, attr_name)
                            if is_tensor_on_meta(attr) and not isinstance(attr, torch.nn.Parameter):
                                meta_tensors_found += 1
                                full_name = f"{path}.{attr_name}" if path else attr_name
                                attrs_to_replace.append((attr_name, attr, full_name))
                        except (AttributeError, RuntimeError, TypeError):
                            continue
                    for attr_name, attr, full_name in attrs_to_replace:
                        replacement = create_real_tensor_from_saved(attr, full_name)
                        setattr(module, attr_name, replacement)
                        replacements_made += 1
                    # Recursively process child modules
                    for child_name, child_module in module.named_children():
                        child_path = f"{path}.{child_name}" if path else child_name
                        scan_and_fix_module(child_module, child_path)

                # Start the scan with the root model
                scan_and_fix_module(model, "model")
                # Final verification - check if any meta tensors remain in state dict
                remaining_meta = 0
                state_dict = model.state_dict()
                for name, tensor in state_dict.items():
                    if is_tensor_on_meta(tensor):
                        remaining_meta += 1
                if remaining_meta > 0:
                    pruna_logger.warning(f"CRITICAL: {remaining_meta} meta tensors still remain in the model!")
            except Exception as e:
                pruna_logger.warning(f"Error during meta tensor cleanup: {e}")
                import traceback

                pruna_logger.warning(traceback.format_exc())

    return AutoHQQHFDiffusersModel


def find_module_layers_type(model: Any, layer_type: type, exclude_module_names: list[str] = []) -> list:
    """
    Find all layers of a specific type in a model.

    Parameters
    ----------
    model : Any
        The model to search through.
    layer_type : type
        The type of layer to find.
    exclude_module_names : list[str], optional
        The names of the modules to exclude from the search.

    Returns
    -------
    list
        List of found layers matching the specified type.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, layer_type) and name not in exclude_module_names:
            layers.append(module)
    return layers
