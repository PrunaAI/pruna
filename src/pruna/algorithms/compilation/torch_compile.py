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

from typing import Any, Callable, Dict

import torch
from ConfigSpace import CategoricalHyperparameter

from pruna.algorithms.compilation import PrunaCompiler
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.smash_space import Boolean
from pruna.engine.model_checks import (
    get_diffusers_transformer_models,
    get_diffusers_unet_models,
)
from pruna.logging.logger import pruna_logger

# This allows for torch compile to use more cache memory to compile the model
torch._dynamo.config.cache_size_limit = 128


class TorchCompileCompiler(PrunaCompiler):
    """
    Implement Torch Compile compilation using torch.compile.

    Optimizes given model or function using various backends and is compatible with any model containing PyTorch modules.
    """

    algorithm_name = "torch_compile"
    references = {"GitHub": "https://github.com/pytorch/pytorch"}
    tokenizer_required = False
    processor_required = False
    run_on_cpu = True
    run_on_cuda = True
    dataset_required = False
    compatible_algorithms = dict(
        quantizer=["half", "hqq_diffusers"],
        cacher=["deepcache"],
    )

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for the algorithm.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [
            CategoricalHyperparameter(
                "mode",
                choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                default_value="default",
                meta=dict(desc="Compilation mode."),
            ),
            CategoricalHyperparameter(
                "backend",
                choices=["inductor", "cudagraphs", "onnxrt", "tvm", "openvino", "openxla"],
                default_value="inductor",
                meta=dict(desc="Compilation backend."),
            ),
            Boolean(
                "fullgraph",
                default=True,
                meta=dict(desc="Whether to discover compilable subgraphs or compile the full input graph."),
            ),
            CategoricalHyperparameter(
                "dynamic",
                choices=[None, True, False],
                default_value=None,
                meta=dict(desc="Whether to use dynamic shape tracing or not."),
            ),
        ]

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
        return callable(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Compile the model.

        Parameters
        ----------
        model : Any
            The model to compile or a list of functions to compile.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the compilation.

        Returns
        -------
        Any
            The compiled model.
        """
        cacher_type = smash_config["cacher"]
        if cacher_type in compilation_map:
            return compilation_map[cacher_type](model, smash_config)

        if (
            hasattr(model, "transformer")
            and isinstance(model.transformer, tuple(get_diffusers_transformer_models()))
            or (hasattr(model, "unet") and isinstance(model.unet, tuple(get_diffusers_unet_models())))
        ):
            return unet_transformer_pipeline_logic(model, smash_config)
        return compile_callable(model, smash_config)

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        return dict()


def get_model_device(model: Callable[..., Any]) -> torch.device:
    """
    Get the device (CPU/GPU) that the model parameters are stored on.

    Parameters
    ----------
    model : Callable[..., Any]
        The PyTorch model to check the device for.

    Returns
    -------
    torch.device
        The device that the model parameters are stored on.
    """
    if hasattr(model, "parameters"):
        return next(model.parameters()).device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compile_callable(model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
    """
    Compile a callable model using torch.compile.

    Parameters
    ----------
    model : Any
        The model to compile.
    smash_config : SmashConfigPrefixWrapper
        Configuration settings for compilation.

    Returns
    -------
    Any
        The compiled model.
    """
    backend = smash_config["backend"]
    if smash_config["device"] == "cpu" or str(get_model_device(model)) == "cpu":
        pruna_logger.info("Compiling for CPU")
        backend = "openvino"
    return torch.compile(
        model,
        dynamic=smash_config["dynamic"],
        fullgraph=smash_config["fullgraph"],
        mode=smash_config["mode"],
        backend=backend,
    )


def deepcache_logic(model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
    """
    Apply compilation logic for DeepCache models.

    Parameters
    ----------
    model : Any
        The model to compile.
    smash_config : SmashConfigPrefixWrapper
        Configuration settings for compilation.

    Returns
    -------
    Any
        The compiled model.
    """
    for function_name, function in model.deepcache_unet_helper.function_dict.items():
        if function_name == "unet_forward":
            continue
        elif function_name[1] != "block":
            model.deepcache_unet_helper.function_dict[function_name] = compile_callable(function, smash_config)
    model.text_encoder = compile_callable(model.text_encoder, smash_config)
    model.vae = compile_callable(model.vae, smash_config)
    return model


def unet_transformer_pipeline_logic(model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
    """
    Apply compilation logic for unet and transformer based diffusers pipelines.

    Parameters
    ----------
    model : Any
        The model to compile.
    smash_config : SmashConfigPrefixWrapper
        Configuration settings for compilation.

    Returns
    -------
    Any
        The compiled model.
    """
    if hasattr(model, "transformer"):
        model.transformer.forward = compile_callable(model.transformer.forward, smash_config)
    elif hasattr(model, "unet"):
        model.unet.forward = compile_callable(model.unet.forward, smash_config)
    else:
        model.forward = compile_callable(model.forward, smash_config)
    return model


compilation_map = {
    "deepcache": deepcache_logic,
}
