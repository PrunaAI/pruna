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

from typing import Any, Dict

from pruna.algorithms.compilation import PrunaCompiler
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_comfy_model, is_diffusers_pipeline, is_flux_pipeline
from pruna.logging.logger import pruna_logger


class StableFastCompiler(PrunaCompiler):
    """
    Implement stable_fast compilation using the sfast library.

    Stable-fast is an optimization framework for Image-Gen models. It accelerates inference by fusing key operations
    into optimized kernels and converting diffusion pipelines into efficient TorchScript graphs.
    """

    algorithm_name: str = "stable_fast"
    references: dict[str, str] = {"GitHub": "https://github.com/chengzeyi/stable-fast"}
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cuda"]
    dataset_required: bool = False
    compatible_algorithms: dict[str, list[str]] = dict(cacher=["deepcache", "fora"], quantizer=["half"])
    required_install: str = "``pip install pruna[stable-fast]``"

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters.
        """
        return []

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a valid model for the algorithm.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a valid model for the algorithm, False otherwise.
        """
        return is_diffusers_pipeline(model, include_video=True) or is_flux_pipeline(model) or is_comfy_model(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Compile the model.

        Parameters
        ----------
        model : Any
            The model to compile.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the compilation.

        Returns
        -------
        Any
            The compiled model.
        """
        imported_modules = self.import_algorithm_packages()
        config = create_config(model, imported_modules)
        cacher_type = smash_config["cacher"]
        if cacher_type in compilation_map:
            return compilation_map[cacher_type](model, imported_modules, config, smash_config)
        return compile_stable_fast(model, smash_config, imported_modules)

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        from sfast.compilers.diffusion_pipeline_compiler import (
            CompilationConfig,
            _build_lazy_trace,
            _build_ts_compiler,
        )
        from sfast.compilers.diffusion_pipeline_compiler import compile as compile_stable_fast
        from sfast.cuda.graphs import make_dynamic_graphed_callable
        from sfast.jit.trace_helper import apply_auto_trace_compiler, lazy_trace

        return dict(
            CompilationConfig=CompilationConfig,
            _build_lazy_trace=_build_lazy_trace,
            compile=compile_stable_fast,
            make_dynamic_graphed_callable=make_dynamic_graphed_callable,
            apply_auto_trace_compiler=apply_auto_trace_compiler,
            lazy_trace=lazy_trace,
            _build_ts_compiler=_build_ts_compiler,
        )


def create_config(model: Any, imported_modules: Dict[str, Any]) -> Any:
    """
    Create a configuration for the compilation process.

    The configuration includes options such as enabling xformers, Triton, and CUDA graph based
    on model attributes and installed libraries.

    Parameters
    ----------
    model : Any
        The model to compile.
    imported_modules : Dict[str, Any]
        The imported modules.

    Returns
    -------
    Any
        A configuration object with the appropriate settings for compilation.
    """
    config = imported_modules["CompilationConfig"].Default()

    try:
        import xformers  # noqa: F401

        config.enable_xformers = True

    except ImportError:
        pruna_logger.warning("Xformers is not installed, skipping this import.")

    try:
        import triton  # noqa: F401

        config.enable_triton = True
    except ImportError:
        pruna_logger.warning("Triton is not installed, skipping this import.")

    config.enable_cuda_graph = True
    return config


def compile_stable_fast(model: Any, smash_config: SmashConfigPrefixWrapper, imported_modules: Dict[str, Any]) -> Any:
    """
    Compile the model with stable_fast.

    Parameters
    ----------
    model : Any
        The model to compile.
    smash_config : SmashConfigPrefixWrapper
        The configuration for the compilation.
    imported_modules : Dict[str, Any]
        The imported modules.

    Returns
    -------
    Any
        The compiled model.
    """
    # Freeze UNET parameters during compilation
    if hasattr(model, "unet"):
        for param in model.unet.parameters():
            param.requires_grad = False

    config = create_config(model, imported_modules)

    # Compile the model with the given configuration
    model = imported_modules["compile"](model, config)
    return model


def deepcache_logic(
    model: Any, imported_modules: Dict[str, Any], config: Any, smash_config: SmashConfigPrefixWrapper
) -> Any:
    """
    Compile the model when DeepCache has previously been applied.

    Parameters
    ----------
    model : Any
        The model to compile.
    imported_modules : Dict[str, Any]
        The imported modules.
    config : Any
        The configuration.
    smash_config : SmashConfigPrefixWrapper
        The configuration for the compilation.

    Returns
    -------
    Any
        The compiled model.
    """
    model = compile_stable_fast(model, smash_config, imported_modules)
    if hasattr(model, "deepcache_unet_helper"):
        model.deepcache_unet_helper.disable()
        model.deepcache_unet_helper.enable()

    # iterate through all the functions that deepcache modified and compile them
    for key, value in model.deepcache_unet_helper.function_dict.items():
        if key == "unet_forward":
            continue
        elif key[1] != "block":
            lazy_trace_ = imported_modules["_build_lazy_trace"](
                config,
                enable_triton_reshape=config.enable_cuda_graph,
                enable_triton_layer_norm=config.enable_cuda_graph,
            )
            value = lazy_trace_(value)
            if config.enable_cuda_graph:
                model.deepcache_unet_helper.function_dict[key] = imported_modules["make_dynamic_graphed_callable"](value)
    return model


def fora_logic(
    model: Any,
    imported_modules: Dict[str, Any],
    config: Any,
    smash_config: SmashConfigPrefixWrapper,
) -> Any:
    """
    Apply compilation to models that use FORA Caching.

    Parameters
    ----------
    model : Any
        The model to compile.
    imported_modules : Dict[str, Any]
        The imported modules.
    config : Any
        The configuration.
    smash_config : SmashConfigPrefixWrapper
        The configuration for the compilation.

    Returns
    -------
    Any
        The compiled model.
    """
    lazy_trace_ = imported_modules["_build_lazy_trace"](
        config,
        enable_triton_reshape=config.enable_cuda_graph,
        enable_triton_layer_norm=config.enable_cuda_graph,
    )

    # compile transformer blocks
    for layer, block_forward in model.cache_helper.double_stream_blocks_forward.items():
        block_forward = lazy_trace_(block_forward)
        if config.enable_cuda_graph:
            model.cache_helper.double_stream_blocks_forward[layer] = imported_modules["make_dynamic_graphed_callable"](
                block_forward
            )
    for layer, block_forward in model.cache_helper.single_stream_blocks_forward.items():
        block_forward = lazy_trace_(block_forward)
        if config.enable_cuda_graph:
            model.cache_helper.single_stream_blocks_forward[layer] = imported_modules["make_dynamic_graphed_callable"](
                block_forward
            )

    # compile vae
    ts_compiler = imported_modules["_build_ts_compiler"](
        config,
        enable_triton_reshape=config.enable_cuda_graph,
        enable_triton_layer_norm=config.enable_cuda_graph,
    )
    model.vae = imported_modules["apply_auto_trace_compiler"](model.vae, ts_compiler=ts_compiler)
    return model


compilation_map = {
    "deepcache": deepcache_logic,
    "fora": fora_logic,
}
