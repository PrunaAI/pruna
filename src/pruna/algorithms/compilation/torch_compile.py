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

import contextlib
import os
from typing import Any, Callable, Dict

import torch
from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter

from pruna.algorithms.compilation import PrunaCompiler
from pruna.algorithms.compilation.utils import CausalLMGenerator, JanusGenerator
from pruna.config.smash_config import SmashConfig, SmashConfigPrefixWrapper
from pruna.config.smash_space import Boolean
from pruna.engine.model_checks import (
    get_diffusers_transformer_models,
    get_diffusers_unet_models,
    is_causal_lm,
    is_janus_llamagen_ar,
    is_opt_model,
)
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.logging.logger import pruna_logger

# This allows for torch compile to use more cache memory to compile the model
torch._dynamo.config.cache_size_limit = 128


class TorchCompileCompiler(PrunaCompiler):
    """
    Implement Torch Compile compilation using torch.compile.

    Optimizes given model or function using various backends and is compatible with any model containing PyTorch modules.
    """

    algorithm_name: str = "torch_compile"
    references: dict[str, str] = {"GitHub": "https://github.com/pytorch/pytorch"}
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda"]
    dataset_required: bool = False
    compatible_algorithms: dict[str, list[str]] = dict(
        factorizer=["qkv_diffusers"],
        quantizer=["half", "hqq_diffusers", "diffusers_int8", "gptq", "llm_int8", "hqq", "torchao"],
        cacher=["deepcache", "fora"],
        pruner=["torch_structured"],
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
            OrdinalHyperparameter(
                "max_kv_cache_size",
                sequence=[100, 200, 400, 512, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400],
                default_value=400,
                meta=dict(desc="The maximum number of new tokens to generate, for LLMs."),
            ),
            OrdinalHyperparameter(
                "seqlen_manual_cuda_graph",
                sequence=[100, 200, 400, 512, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400],
                default_value=100,
                meta=dict(
                    desc="The sequence length to use for manual CUDA graph capture, for LLMs. "
                    "We recommend to use a smaller value than max_kv_cache_size to avoid "
                    "CUDA graph capture overhead."
                ),
            ),
            Boolean(
                "make_portable",
                meta=dict(
                    desc=(
                        "Whether to make the model compiled model portable or not, "
                        "and significantly reduce the warmup time of the model on a different machine."
                    ),
                ),
            ),
            OrdinalHyperparameter(
                "target",
                default_value="model",
                sequence=["model", "module_list"],
                meta=dict(
                    desc=(
                        "Whether to compile the model itself or the module list. "
                        "Compiling the model itself has a longer warmup and could fail "
                        "in case of graphbreaks but could lead to slightly faster compilation. "
                        "Whereas compiling the module list has a shorter warmup and is more "
                        "robust to graphbreaks but could be slightly slower."
                    )
                ),
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
        # opt models have no cache_position, so will raise error like
        # TypeError: OPTForCausalLM.forward() got an unexpected keyword argument 'cache_position'
        return callable(model) and not is_opt_model(model)

    def apply(self, model: Any, smash_config: SmashConfig) -> Any:
        """
        Apply the compilation algorithm to the model.

        Parameters
        ----------
        model : Any
            The model to compile.
        smash_config : SmashConfig
            The configuration for the compilation.

        Returns
        -------
        Any
            The compiled model.
        """
        if smash_config["torch_compile_make_portable"]:
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

        output = super().apply(model, smash_config)

        # importantly, the torch artifacts saving need to be done *after* the before-compile-save
        if smash_config["torch_compile_make_portable"]:
            smash_config.save_fns.append(SAVE_FUNCTIONS.torch_artifacts.name)
        return output

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
        with contextlib.suppress(KeyError):
            distributer_type = smash_config["distributer"]
            if distributer_type in compilation_map:
                return compilation_map[distributer_type](model, smash_config)

        cacher_type = smash_config["cacher"]
        if cacher_type in compilation_map:
            return compilation_map[cacher_type](model, smash_config)

        if (
            hasattr(model, "transformer")
            and isinstance(model.transformer, tuple(get_diffusers_transformer_models()))
            or (hasattr(model, "unet") and isinstance(model.unet, tuple(get_diffusers_unet_models())))
        ):
            return unet_transformer_pipeline_logic(model, smash_config)

        if is_causal_lm(model) or is_janus_llamagen_ar(model):
            return causal_lm_or_janus_logic(model, smash_config)

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
    if smash_config["target"] == "module_list":
        found_module_list = False
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList):
                found_module_list = True
                for i, submodule in enumerate(module):
                    if isinstance(submodule, torch.nn.Module):
                        submodule = torch.compile(
                            submodule,
                            dynamic=smash_config["dynamic"],
                            fullgraph=smash_config["fullgraph"],
                            mode=smash_config["mode"],
                            backend=backend,
                        )
                    module[i] = submodule

        if not found_module_list:
            pruna_logger.warning(
                "No ModuleList found in the model for compilation. "
                "Torch compile will not have any effect, please switch to target=model."
            )

        return model
    elif smash_config["target"] == "model":
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


def fora_logic(model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
    """
    Apply compilation logic for FORA models.

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
    for idx, function in model.cache_helper.double_stream_blocks_forward.items():
        model.cache_helper.double_stream_blocks_forward[idx] = compile_callable(function, smash_config)
    for idx, function in model.cache_helper.single_stream_blocks_forward.items():
        model.cache_helper.single_stream_blocks_forward[idx] = compile_callable(function, smash_config)
    model.text_encoder = compile_callable(model.text_encoder, smash_config)
    model.text_encoder_2 = compile_callable(model.text_encoder_2, smash_config)
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
        if smash_config["target"] == "module_list":
            model.transformer = compile_callable(model.transformer, smash_config)
        elif smash_config["target"] == "model":
            model.transformer.forward = compile_callable(model.transformer.forward, smash_config)
    elif hasattr(model, "unet"):
        if smash_config["target"] == "module_list":
            model.unet = compile_callable(model.unet, smash_config)
        elif smash_config["target"] == "model":
            model.unet.forward = compile_callable(model.unet.forward, smash_config)
    else:
        if smash_config["target"] == "module_list":
            model = compile_callable(model, smash_config)
        elif smash_config["target"] == "model":
            model.forward = compile_callable(model.forward, smash_config)
    return model


def causal_lm_or_janus_logic(model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
    """
    Apply compilation logic for causal language models or Janus LlamaGen AR models.

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
    if hasattr(model, "generation_config") and model.generation_config is not None:
        top_k = model.generation_config.top_k if hasattr(model.generation_config, "top_k") else 50
        temperature = model.generation_config.temperature if hasattr(model.generation_config, "temperature") else 1.0
    else:
        pruna_logger.warning("No generation config found, using default values for top_k and temperature.")
        # https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig.top_k
        top_k = 50
        # https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig.temperature
        temperature = 1.0

    if is_causal_lm(model):
        # We use a generator as in https://github.com/mobiusml/hqq/blob/1f052eb5a0aab0572d380d48b708ae1c74936d23/hqq/utils/generation_hf.py
        gen = CausalLMGenerator(
            model,
            max_kv_cache_size=smash_config["max_kv_cache_size"],
            temperature=temperature,
            top_k=top_k,
            compile_mode=smash_config["mode"],
            compile_fullgraph=smash_config["fullgraph"],
            batch_size=smash_config.batch_size,
            device=smash_config.device,
        )
    elif is_janus_llamagen_ar(model):
        gen = JanusGenerator(  # type: ignore
            model,
            temperature=temperature,
            top_k=top_k,
            compile_mode=smash_config["mode"],
            compile_fullgraph=smash_config["fullgraph"],
        )
    else:
        raise ValueError(f"Model {model} is not a causal language model or Janus LlamaGen AR model.")

    # If we are using max-autotune-no-cudagraphs, we need to handle the cudagraphs manually.
    if smash_config["mode"] == "max-autotune-no-cudagraphs":
        pruna_logger.error("max-autotune-no-cudagraphs is not supported for causal language models.")
    model.generate = gen.generate
    return model


compilation_map = {
    "deepcache": deepcache_logic,
    "fora": fora_logic,
}
