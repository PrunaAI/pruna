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

import shutil
import subprocess  # nosec B404
import sys
import tempfile
import urllib.request
import weakref
from pathlib import Path
from typing import Any, Dict

from ConfigSpace import OrdinalHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag as tags
from pruna.config.hyperparameters import Int
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import (
    is_causal_lm,
    is_transformers_pipeline_with_causal_lm,
)
from pruna.engine.save import SAVE_FUNCTIONS
from pruna.engine.utils import verify_sha256
from pruna.logging.logger import pruna_logger

# SHA256 hash for the pinned version (b3600) of convert_hf_to_gguf.py
LLAMA_CPP_CONVERSION_SCRIPT_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/b3600/convert_hf_to_gguf.py"
LLAMA_CPP_CONVERSION_SCRIPT_SHA256 = "f62ab712618231b3e76050f94e45dcf94567312c209b4b99bfc142229360b018"
LLAMA_CPP_CACHE_DIR = Path.home() / ".cache" / "pruna" / "scripts" / "llama_cpp"


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
            OrdinalHyperparameter(
                "n_gpu_layers",
                sequence=[0, 1, 4, 8, 16, 32, 999],
                default_value=0,
                meta={"desc": "Number of layers to offload to GPU. Use 999 for all layers."},
            ),
            Int(
                "main_gpu",
                default=0,
                meta={"desc": "The GPU to use for the main model tensors."},
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

        # Ensure we have the causal lm if it's a pipeline
        model_to_export = model.model if is_transformers_pipeline_with_causal_lm(model) else model

        quantization_method = self._get_quantization_method(model_to_export, smash_config["quantization_method"])
        pruna_logger.info(f"Quantizing model with llama.cpp using method {quantization_method}")

        _, f16_gguf_path, quant_gguf_path = self._get_cache_paths(
            model_to_export, smash_config, quantization_method
        )

        # Create a temp directory to hold HF model if needed
        temp_dir = Path(tempfile.mkdtemp())
        # Ensure cleanup even if save() is not called
        weakref.finalize(self, shutil.rmtree, str(temp_dir), ignore_errors=True)

        try:
            # Convert to F16 GGUF if needed
            if not f16_gguf_path.exists():
                self._convert_to_gguf(model_to_export, f16_gguf_path, temp_dir, smash_config)
            else:
                pruna_logger.info(f"Using cached F16 GGUF model at {f16_gguf_path}")

            # Quantize GGUF if needed
            if quantization_method != "f16":
                if not quant_gguf_path.exists():
                    self._quantize_gguf(llama_cpp, f16_gguf_path, quant_gguf_path, quantization_method)
                else:
                    pruna_logger.info(f"Using cached quantized model at {quant_gguf_path}")
            else:
                quant_gguf_path = f16_gguf_path

            return self._load_quantized_model(llama_cpp, quant_gguf_path, smash_config, temp_dir)

        except Exception as e:
            pruna_logger.error(f"Error during llama.cpp quantization: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _get_quantization_method(self, model: Any, default_method: str) -> str:
        """Get the quantization method, defaulting to f16 for tiny models."""
        if (
            hasattr(model, "config")
            and hasattr(model.config, "hidden_size")
            and model.config.hidden_size < 32
        ):
            pruna_logger.info("Tiny model detected. Bypassing quantized block sizes and using f16.")
            return "f16"
        return default_method

    def _load_quantized_model(self, llama_cpp: Any, quant_gguf_path: Path, smash_config: Any, temp_dir: Path) -> Any:
        pruna_logger.info(f"Loading quantized model from {quant_gguf_path}")
        n_gpu_layers = smash_config["n_gpu_layers"]
        if n_gpu_layers == 999:
            n_gpu_layers = -1  # llama-cpp-python uses -1 for all layers
        quantized_model = llama_cpp.Llama(
            model_path=str(quant_gguf_path),
            n_gpu_layers=n_gpu_layers,
            main_gpu=smash_config["main_gpu"],
        )
        quantized_model._pruna_temp_dir = str(temp_dir)
        quantized_model.model_path = str(quant_gguf_path)
        quantized_model._pruna_device = smash_config["device"]
        return quantized_model


    def _get_cache_paths(
        self, model: Any, smash_config: SmashConfigPrefixWrapper, q_method: str
    ) -> tuple[Path, Path, Path]:
        """Generate cache paths for the models."""
        llama_cpp_cache = Path(smash_config.cache_dir) / "llama_cpp"
        llama_cpp_cache.mkdir(parents=True, exist_ok=True)

        model_id = "model"
        if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_id = Path(model.config._name_or_path).name

        f16_gguf_path = llama_cpp_cache / f"{model_id}-f16.gguf"
        quant_gguf_path = llama_cpp_cache / f"{model_id}-{q_method}.gguf"
        return llama_cpp_cache, f16_gguf_path, quant_gguf_path

    def _convert_to_gguf(
        self,
        model: Any,
        outfile: Path,
        temp_dir: Path,
        smash_config: SmashConfigPrefixWrapper
    ) -> None:
        """Save HF model and convert it to GGUF format."""
        with tempfile.TemporaryDirectory(dir=str(temp_dir)) as hf_model_dir:
            model.save_pretrained(hf_model_dir)
            if hasattr(smash_config, "tokenizer") and smash_config.tokenizer:
                smash_config.tokenizer.save_pretrained(hf_model_dir)

            script_path = self._get_conversion_script()
            pruna_logger.info(f"Converting Hugging Face model to GGUF format at {outfile}...")

            # Ensure inputs are properly sanitized and validated to prevent arg injection.
            for param in (script_path, hf_model_dir, outfile):
                param_str = str(param)
                if any(c in param_str for c in ("\0", "\n", "\r", ";", "&", "|", "`", "$")):
                    raise ValueError(f"Unsafe characters detected in subprocess argument: {param_str}")

            convert_cmd = [
                sys.executable, str(script_path),
                hf_model_dir,
                "--outfile", str(outfile),
                "--outtype", "f16"
            ]
            try:
                # subprocess needed because convert_hf_to_gguf.py is a standalone CLI script
                subprocess.run(convert_cmd, check=True, capture_output=True, text=True)  # nosec B603
            except subprocess.CalledProcessError as e:
                pruna_logger.error(f"Conversion script failed with error: {e.stderr}")
                raise

    def _quantize_gguf(
        self,
        llama_cpp: Any,
        infile: Path,
        outfile: Path,
        method: str
    ) -> None:
        """Quantize a GGUF file using llama-cpp-python API."""
        pruna_logger.info(f"Quantizing GGUF model to {method} at {outfile}...")

        if not hasattr(llama_cpp, "llama_model_quantize"):
            raise RuntimeError("llama_model_quantize API not available in llama-cpp-python.")

        params = llama_cpp.llama_model_quantize_default_params()
        ftype_name = f"LLAMA_FTYPE_MOSTLY_{method.upper()}"

        if hasattr(llama_cpp, ftype_name):
            params.ftype = getattr(llama_cpp, ftype_name)
        else:
            raise ValueError(f"Unknown quantization method: {method}")

        llama_cpp.llama_model_quantize(
            str(infile).encode("utf-8"),
            str(outfile).encode("utf-8"),
            params,
        )

    def _get_conversion_script(self) -> Path:
        """
        Get the conversion script from cache or download it.

        Returns
        -------
        Path
            The path to the conversion script.
        """
        LLAMA_CPP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        script_path = LLAMA_CPP_CACHE_DIR / "convert_hf_to_gguf.py"

        # Validate URL scheme for security
        if not LLAMA_CPP_CONVERSION_SCRIPT_URL.startswith("https://"):
            raise ValueError(f"Insecure conversion script URL: {LLAMA_CPP_CONVERSION_SCRIPT_URL}")

        if not script_path.exists() or not verify_sha256(script_path, LLAMA_CPP_CONVERSION_SCRIPT_SHA256):
            pruna_logger.info(f"Downloading conversion script from {LLAMA_CPP_CONVERSION_SCRIPT_URL}")
            urllib.request.urlretrieve(LLAMA_CPP_CONVERSION_SCRIPT_URL, script_path)  # nosec B310

            if not verify_sha256(script_path, LLAMA_CPP_CONVERSION_SCRIPT_SHA256):
                script_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Integrity verification failed for {LLAMA_CPP_CONVERSION_SCRIPT_URL}. "
                    "The downloaded script may have been tampered with or the pinned version has changed."
                )

        return script_path

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
