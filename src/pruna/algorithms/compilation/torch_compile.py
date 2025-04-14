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
from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache

from pruna.algorithms.compilation import PrunaCompiler
from pruna.algorithms.compilation.utils import create_generate_fn, create_generate_fn_hqq, decode_one_token
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.config.smash_space import Boolean
from pruna.engine.model_checks import (
    get_diffusers_transformer_models,
    get_diffusers_unet_models,
    is_causal_lm,
    is_opt_model,
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
        quantizer=["half", "hqq_diffusers", "diffusers_int8", "gptq", "llm_int8", "hqq"],
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
            OrdinalHyperparameter(
                "batch_size",
                sequence=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                default_value=1,
                meta=dict(desc="The batch size to use for compilation, for LLMs."),
            ),
            OrdinalHyperparameter(
                "max_kv_cache_size",
                sequence=[100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400],
                default_value=100,
                meta=dict(desc="The maximum number of new tokens to generate, for LLMs."),
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
        imported_algorithms = self.import_algorithm_packages()
        cacher_type = smash_config["cacher"]
        if cacher_type in compilation_map:
            return compilation_map[cacher_type](model, smash_config)

        if (
            hasattr(model, "transformer")
            and isinstance(model.transformer, tuple(get_diffusers_transformer_models()))
            or (hasattr(model, "unet") and isinstance(model.unet, tuple(get_diffusers_unet_models())))
        ):
            return unet_transformer_pipeline_logic(model, smash_config)

        if is_causal_lm(model):
            return causal_lm_logic(model, smash_config, imported_algorithms)
        return compile_callable(model, smash_config)

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        from hqq.utils.generation_hf import HFGenerator
        return dict(HFGenerator=HFGenerator)


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


def causal_lm_logic(model: Any, smash_config: SmashConfigPrefixWrapper, imported_algorithms: Dict[str, Any]) -> Any:
    """
    Apply compilation logic for causal language models.

    Parameters
    ----------
    model : Any
        The model to compile.
    smash_config : SmashConfigPrefixWrapper
        Configuration settings for compilation.
    imported_algorithms : Dict[str, Any]
        The imported algorithms (specific for HQQ).

    Returns
    -------
    Any
        The compiled model.
    """
    if smash_config["quantizer"] == "hqq":
        if hasattr(smash_config, "tokenizer"):
            tokenizer = smash_config.tokenizer
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            except Exception:
                raise Exception(
                    "Tokenizer not found, please provide a tokenizer with "
                    "`smash_config.add_tokenizer(model_id)`."
                )
        gen = imported_algorithms["HFGenerator"](
                model,
                tokenizer,
                max_new_tokens=smash_config["max_kv_cache_size"],
                do_sample=True,
                compile="partial",
                compile_options={"mode": smash_config["mode"], "fullgraph": smash_config["fullgraph"]},
            ).enable_cuda_graph()
        generate = create_generate_fn_hqq(gen, tokenizer)
    else:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            top_k = model.generation_config.top_k if hasattr(model.generation_config, "top_k") else 50
            temperature = model.generation_config.temperature if hasattr(model.generation_config, "temperature") else 1.0
        else:
            pruna_logger.warning("No generation config found, using default values for top_k and temperature.")
            # https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig.top_k
            top_k = 50
            # https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig.temperature
            temperature = 1.0
        past_kv = StaticCache(
            model.config,
            smash_config["batch_size"],
            smash_config["max_kv_cache_size"],
            device=smash_config["device"],
            dtype=model.dtype,
        )
        compiled_decoding = compile_callable(decode_one_token, smash_config)
        generate = create_generate_fn(model, top_k, temperature, past_kv, compiled_decoding)
    model.generate = generate
    return model


compilation_map = {
    "deepcache": deepcache_logic,
}
