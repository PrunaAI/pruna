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

import dataclasses
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from ConfigSpace import Constant
from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger

class XFast(PrunaAlgorithmBase):
    """
    Implement X-Fast compilation using the sfast library.

    Based on stable_fast, this compiler speeds up inference latency for any model using a combination of xformers,
    triton, cudnn, and torch tracing.
    """

    algorithm_name: str = "x_fast"
    group_tags: list[AlgorithmTag] = [AlgorithmTag.COMPILER]
    save_fn = SAVE_FUNCTIONS.save_before_apply
    references: dict[str, str] = {}
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda"]
    dataset_required: bool = False
    compatible_before: list[str | AlgorithmTag] = [
        "quanto",
        "half",
        "text_to_text_lora",
        "text_to_image_lora",
        "text_to_image_perp",
        "text_to_image_inplace_perp",
        "text_to_text_perp",
        "text_to_text_inplace_perp",
        "text_to_image_distillation_inplace_perp",
        "text_to_image_distillation_lora",
        "text_to_image_distillation_perp",
    ]
    required_install: str = "``pip install pruna[stable-fast]``"

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for the X-Fast compiler.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            Constant("fn_to_compile", value="forward"),
            Boolean("xformers", default=True, meta=dict(desc="Whether to use xformers for faster inference.")),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a valid model for the X-Fast compiler.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is valid, False otherwise.
        """
        if isinstance(model, torch.nn.Module):
            return True
        return any(isinstance(attr_value, torch.nn.Module) for _, attr_value in inspect.getmembers(model))

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Compile the model using the X-Fast compiler.

        Parameters
        ----------
        model : Any
            The model to compile.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the compilation process.

        Returns
        -------
        Any
            The compiled model.
        """
        imported_modules = self.import_algorithm_packages()

        model.x_fast_compiler = XFastHelper(imported_modules)

        if smash_config["torch_dynamic"]:
            model = torch.quantization.quantize_dynamic(
                model,  # Input model
                {torch.nn.Linear},
                dtype=torch.qint8,
                inplace=True,
            )

        # Second we need to compile the model and return it
        smashed_model = model.x_fast_compiler.compile(model, smash_config)
        return smashed_model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the necessary packages for the X-Fast compiler.

        Returns
        -------
        Dict[str, Any]
            The imported modules.
        """
        import sfast
        from sfast.compilers.diffusion_pipeline_compiler import (
            CompilationConfig,
            _build_lazy_trace,
            _enable_xformers,
        )
        from sfast.cuda.graphs import (
            make_dynamic_graphed_callable,  # apply_auto_graph_compiler,
        )
        from sfast.utils.memory_format import apply_memory_format

        sfast.cuda.graphs.get_cuda_device_from_tensors = get_cuda_device_from_tensors

        return dict(
            CompilationConfig=CompilationConfig,
            _build_lazy_trace=_build_lazy_trace,
            _enable_xformers=_enable_xformers,
            make_dynamic_graphed_callable=make_dynamic_graphed_callable,
            apply_memory_format=apply_memory_format,
        )


def get_cuda_device_from_tensors(
    x: Union[torch.Tensor, List[Any], Tuple[Any], Dict[Any, Any], Any],
) -> Optional[torch.device | int]:
    """
    Recursively searches for a CUDA device index in a tensor or a nested structure of tensors.

    Parameters
    ----------
    x : Union[torch.Tensor, list, tuple, dict, Any]
        A tensor or a nested structure (list, tuple, dictionary, or dataclass) containing tensors.

    Returns
    -------
    Optional[torch.device | int]
        The index of the CUDA device if a tensor is found on a CUDA device, otherwise None.
    """
    device: Optional[torch.device | int] = None
    if isinstance(x, torch.Tensor):
        device = x.device
        if device.type == "cuda":
            return device.index
        return None
    elif isinstance(x, (list, tuple)):
        for y in x:
            device = get_cuda_device_from_tensors(y)
            if device is not None:
                return device
        return None
    elif dataclasses.is_dataclass(x):
        for k in dataclasses.fields(x):
            device = get_cuda_device_from_tensors(getattr(x, k.name))
            if device is not None:
                return device
        return None
    elif isinstance(x, dict):
        for v in x.values():
            device = get_cuda_device_from_tensors(v)
            if device is not None:
                return device
        return None
    else:
        return None


def get_and_compile_nested_attribute(obj: Any, attr_path: str) -> Any:
    """
    Get and compile a nested attribute of an object.

    Parameters
    ----------
    obj : Any
        The object to retrieve the nested attribute from.
    attr_path : str
        The path to the nested attribute, using dot notation.

    Returns
    -------
    Any
        The compiled nested attribute.
    """
    current_attr = obj
    attr_chain = attr_path.split(".")

    for attr in attr_chain[:-1]:
        current_attr = getattr(current_attr, attr)

    # Get the final attribute (method) in the chain
    final_attr = getattr(current_attr, attr_chain[-1])
    return final_attr


def apply_lazy_tracing_and_dynamic_graphing(
    model: Any,
    config: Any,
    smash_config: SmashConfigPrefixWrapper,
    enable_cuda_graph: bool,
    imported_modules: Dict[str, Any],
) -> None:
    """
    Apply lazy tracing and dynamic graphing to the given model.

    Parameters
    ----------
    model : Any
        The model to apply lazy tracing and dynamic graphing to.
    config : Any
        The configuration for lazy tracing.
    smash_config : SmashConfigPrefixWrapper
        The configuration for smashing (e.g., which functions to compile).
    enable_cuda_graph : bool
        Flag indicating whether to enable CUDA graph.
    imported_modules : Dict[str, Any]
        Dictionary containing the imported modules.
    """
    config.enable_cnn_optimization = False
    lazy_trace_ = imported_modules["_build_lazy_trace"](
        config,
        enable_triton_reshape=enable_cuda_graph,
        enable_triton_layer_norm=enable_cuda_graph,
    )

    current_attribute = get_and_compile_nested_attribute(model, smash_config["fn_to_compile"])
    modified_attribute = lazy_trace_(current_attribute)
    if enable_cuda_graph:
        modified_attribute = imported_modules["make_dynamic_graphed_callable"](modified_attribute)

    attr_chain = smash_config["fn_to_compile"].split(".")
    parent_attr = model
    for attr in attr_chain[:-1]:
        parent_attr = getattr(parent_attr, attr)
    setattr(parent_attr, attr_chain[-1], modified_attribute)


def process_model(
    model: Any,
    config: Any,
    smash_config: SmashConfigPrefixWrapper,
    enable_cuda_graph: bool,
    imported_modules: Dict[str, Any],
) -> None:
    """
    Update the given model by applying various optimizations and transformations.

    This function applies lazy tracing, dynamic graphing, xformers, and memory format optimizations
    to the model based on the provided configuration. It can handle both callable functions and
    torch.nn.Module instances, and recursively processes nested model attributes.

    Parameters
    ----------
    model : Any
        The model or function to be processed. Can be a callable or a torch.nn.Module instance.
    config : Any
        Configuration object containing settings for various optimizations.
    smash_config : SmashConfigPrefixWrapper
        Configuration dictionary for model smashing, including the function to compile.
    enable_cuda_graph : bool
        Flag indicating whether to enable CUDA graph.
    imported_modules : Dict[str, Any]
        Dictionary containing the imported modules.

    Returns
    -------
    None
        The function modifies the model in-place and doesn't return a value.

    Notes
    -----
    - For callable models that are not torch.nn.Module instances, it applies lazy tracing
      and potentially dynamic graphing.
    - For torch.nn.Module instances with the specified compile function, it applies
      various optimizations including xformers and memory format changes.
    - Recursively processes nested model attributes if the model doesn't match the above criteria.
    - Silently returns if an exception occurs during recursive processing.
    """
    if hasattr(model, smash_config["fn_to_compile"]) and isinstance(model, torch.nn.Module):
        if hasattr(model, "eval"):
            model.eval()

        if config.enable_xformers:
            imported_modules["_enable_xformers"](model)
        if config.memory_format is not None:
            imported_modules["apply_memory_format"](model, memory_format=config.memory_format)
        apply_lazy_tracing_and_dynamic_graphing(model, config, smash_config, enable_cuda_graph, imported_modules)
    else:
        # Recursively process model attributes
        try:
            for attribute_name, attribute_value in vars(model).items():
                process_model(
                    attribute_value,
                    config,
                    smash_config,
                    enable_cuda_graph,
                    imported_modules,
                )
        except Exception:
            # Model is not an object, cannot recurse
            return


class XFastHelper:
    """
    A compiler class to process models using various optimizations such as xformers, Triton, and CUDA graphing.

    Parameters
    ----------
    imported_modules : Dict[str, Any]
        Dictionary containing the imported modules.
    """

    def __init__(self, imported_modules: Dict[str, Any]) -> None:
        self.imported_modules = imported_modules

    def compile(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Compile the model by applying optimizations based on the smash configuration.

        Parameters
        ----------
        model : Any
            The model or function to compile.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the compilation process.

        Returns
        -------
        Any
            The compiled model.
        """
        config = self.imported_modules["CompilationConfig"].Default()

        try:
            import xformers  # noqa: F401

            if smash_config["xformers"]:
                config.enable_xformers = True
        except ImportError:
            pruna_logger.info("xformers not installed, skip")

        try:
            import triton  # noqa: F401

            if "CausalLM" in type(model).__name__:
                config.enable_triton = False
            else:
                config.enable_triton = True
        except ImportError:
            pruna_logger.info("Triton not installed, skip")

        config.enable_cuda_graph = True

        device = (
            model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if hasattr(model, "to"):
            model = model.to(device)

        enable_cuda_graph = config.enable_cuda_graph

        process_model(model, config, smash_config, enable_cuda_graph, self.imported_modules)

        return model

    @staticmethod
    def process_function(function: Callable[..., Any]) -> Callable[..., Any] | None:
        """
        For internal use only. Process the given function by applying various optimizations and transformations.

        Parameters
        ----------
        function : Callable
            The function to be processed.

        Returns
        -------
        Callable
            The processed function.
        """
        try:
            import sfast
            from sfast.compilers.diffusion_pipeline_compiler import (
                CompilationConfig,
                _build_lazy_trace,
            )
            from sfast.cuda.graphs import (
                make_dynamic_graphed_callable,  # apply_auto_graph_compiler,
            )

            sfast.cuda.graphs.get_cuda_device_from_tensors = get_cuda_device_from_tensors
        except ImportError:
            pruna_logger.error(
                "You are trying to use XFast compiler, but sfast is not installed. "
                "This is likely because you did not install the GPU version of Pruna."
            )
            raise

        config = CompilationConfig.Default()

        try:
            import xformers  # noqa: F401

            config.enable_xformers = True
        except ImportError:
            pruna_logger.info("xformers not installed, skip")

        try:
            import triton  # noqa: F401

            if "CausalLM" in type(function).__name__:
                config.enable_triton = False
            else:
                config.enable_triton = True
        except ImportError:
            pruna_logger.info("Triton not installed, skip")

        config.enable_cuda_graph = True

        if callable(function) and not isinstance(function, torch.nn.Module):
            lazy_trace_ = _build_lazy_trace(
                config,
                enable_triton_reshape=config.enable_cuda_graph,
                enable_triton_layer_norm=config.enable_cuda_graph,
            )
            function = lazy_trace_(function)
            if config.enable_cuda_graph:
                function = make_dynamic_graphed_callable(function)
            return function
        return None
