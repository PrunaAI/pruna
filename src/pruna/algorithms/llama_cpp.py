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

import os
import tempfile
import subprocess
from typing import Any, Dict

from ConfigSpace import Constant, OrdinalHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.model_checks import is_causal_lm, is_transformers_pipeline_with_causal_lm
from pruna.logging.logger import pruna_logger


class LlamaCpp(PrunaAlgorithmBase):
    """
    Implement Llama.cpp as a quantizer.

    Converts Hugging Face models to GGUF format and quantizes them using the llama.cpp tools.
    """

    algorithm_name: str = "llama_cpp"
    group_tags: list[tags] = [tags.QUANTIZER]
    references: dict[str, str] = {
        "GitHub": "https://github.com/ggml-org/llama.cpp",
        "Python Bindings": "https://github.com/abetlen/llama-cpp-python",
    }
    save_fn: SAVE_FUNCTIONS = SAVE_FUNCTIONS.llama_cpp
    tokenizer_required: bool = False
    processor_required: bool = False
    dataset_required: bool = False
    runs_on: list[str] = ["cpu", "cuda", "mps"]
    compatible_before: list[str] = []
    compatible_after: list[str] = []

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
                "quantization_method",
                sequence=[
                    "q4_k_m",
                    "q4_k_s",
                    "q5_k_m",
                    "q8_0",
                    "f16"
                ],
                default_value="q4_k_m",
                meta={"desc": "Quantization method for llama.cpp. Examples: q4_k_m, q8_0, f16."},
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is supported.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is supported, False otherwise.
        """
        return is_causal_lm(model) or is_transformers_pipeline_with_causal_lm(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Quantize the model with Llama.cpp by converting to GGUF.

        Parameters
        ----------
        model : Any
            The model to quantize.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the quantization.

        Returns
        -------
        Any
            The quantized Llama object.
        """
        imported_modules = self.import_algorithm_packages()
        llama_cpp = imported_modules["llama_cpp"]

        quantization_method = smash_config["quantization_method"]

        pruna_logger.info(f"Quantizing model with llama.cpp using method {quantization_method}")

        # Ensure we have the causal lm if it's a pipeline
        if is_transformers_pipeline_with_causal_lm(model):
            model_to_export = model.model
        else:
            model_to_export = model

        # Create a temp directory to hold HF model, f16 GGUF, and optimized GGUF
        temp_dir = tempfile.mkdtemp()
        hf_model_dir = os.path.join(temp_dir, "hf_model")
        f16_gguf_path = os.path.join(temp_dir, "model-f16.gguf")
        quant_gguf_path = os.path.join(temp_dir, f"model-{quantization_method}.gguf")

        try:
            # save HF model
            model_to_export.save_pretrained(hf_model_dir)
            if hasattr(smash_config, "tokenizer") and smash_config.tokenizer:
                smash_config.tokenizer.save_pretrained(hf_model_dir)

            # convert to f16 GGUF using gguf-convert-hf-to-gguf
            pruna_logger.info("Converting Hugging Face model to GGUF format...")
            convert_cmd = [
                "python", "-m", "gguf-convert-hf-to-gguf",
                hf_model_dir,
                "--outfile", f16_gguf_path,
                "--outtype", "f16"
            ]
            subprocess.run(convert_cmd, check=True)

            # quantize the GGUF model
            if quantization_method != "f16":
                pruna_logger.info(f"Quantizing GGUF model to {quantization_method}...")
                
                # Retrieve quantize CLI from llama.cpp
                if hasattr(llama_cpp, "llama_model_quantize"):
                    # Using API
                    params = llama_cpp.llama_model_quantize_default_params()
                    
                    # Convert string to enum, e.g. "q4_k_m" -> llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M
                    ftype_name = f"LLAMA_FTYPE_MOSTLY_{quantization_method.upper()}"
                    if hasattr(llama_cpp, ftype_name):
                        params.ftype = getattr(llama_cpp, ftype_name)
                    else:
                        raise ValueError(f"Unknown quantization method: {quantization_method}")
                        
                    llama_cpp.llama_model_quantize(
                        f16_gguf_path.encode('utf-8'),
                        quant_gguf_path.encode('utf-8'),
                        params
                    )
                else:
                    raise RuntimeError("llama-cpp-python does not have llama_model_quantize available")
            else:
                quant_gguf_path = f16_gguf_path

            # Load the quantized model
            pruna_logger.info(f"Loading quantized model from {quant_gguf_path}")
            quantized_model = llama_cpp.Llama(model_path=quant_gguf_path)

            # Keep a reference to the temp file path so the save function can move it
            quantized_model.model_path = quant_gguf_path
            
            if quantization_method != "f16":
                os.remove(f16_gguf_path)
                
            return quantized_model

        except Exception as e:
            pruna_logger.error(f"Error during llama.cpp quantization: {e}")
            raise

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        try:
            import llama_cpp
            return dict(llama_cpp=llama_cpp)
        except ImportError:
            raise ImportError(
                "Could not import llama_cpp. Please install it with `pip install llama-cpp-python`."
            )

