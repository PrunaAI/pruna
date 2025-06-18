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
import sys
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Dict, List

import torch
import transformers
from ConfigSpace import OrdinalHyperparameter
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperConfig,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from pruna.algorithms.compilation import PrunaCompiler
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_causal_lm, is_translation_model
from pruna.logging.logger import pruna_logger

if TYPE_CHECKING:
    from ctranslate2.models import Whisper


class CTranslateCompiler(PrunaCompiler):
    """
    Implement CTranslate compilation using ctranslate2.

    CTranslate employs a custom runtime that leverages optimizations like weight quantization, layer fusion, and batch
    reordering to boost performance and reduce memory usage on both CPUs and GPUs for Causal LM models used for
    Translation.

    Parameters
    ----------
    task_name : str
        The task name.
    """

    algorithm_name: str = "c_translate"
    references = {"GitHub": "https://github.com/OpenNMT/CTranslate2"}
    tokenizer_required: bool = True
    processor_required: bool = False
    runs_on: list[str] = ["cuda"]
    dataset_required: bool = False
    compatible_algorithms = dict(batcher=["whisper_s2t"], quantizer=["half"])

    def __init__(self, task_name: str = "translate") -> None:
        self.task_name = task_name
        if task_name == "generate":
            self.algorithm_name = "c_generate"
        elif task_name == "whisper":
            self.algorithm_name = "c_whisper"
            self.tokenizer_required = False
            self.processor_required = True
        super().__init__()

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
                sequence=[8, 16],
                default_value=16,
                meta=dict(desc="Sets the number of bits to use for weight quantization."),
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
        imported_modules = self.import_algorithm_packages()

        if isinstance(model, TranslatorWrapper):
            return True
        if isinstance(model, GeneratorWrapper):
            return True
        if isinstance(model, WhisperWrapper):
            return True
        if isinstance(model, AutomaticSpeechRecognitionPipeline):
            return True

        # check that c_translate2 performs before transformers conversion
        if hasattr(model, "config") and model.config.__class__.__name__ not in imported_modules["_MODEL_LOADERS"]:
            return False

        return (
            (isinstance(model.config, WhisperConfig) and self.algorithm_name == "c_whisper")
            or (is_translation_model(model) and self.algorithm_name == "c_translate")
            or (is_causal_lm(model) and self.algorithm_name == "c_generate")
        )

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

        # Extract the model itself
        if isinstance(model, TranslatorWrapper):
            model = model.translator
        elif isinstance(model, GeneratorWrapper):
            model = model.generator
        elif isinstance(model, WhisperWrapper):
            model = model.whisper
        elif isinstance(model, AutomaticSpeechRecognitionPipeline):
            model = model.model

        # Create outfile directory
        out_dir = os.path.join(smash_config["cache_dir"], "outputs")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        model.save_pretrained(out_dir)
        # we can ignore mypy warnings here because we ensure beforehand that processor and tokenizer are not None
        if self.processor_required:
            smash_config.processor.save_pretrained(out_dir)  # type: ignore[attr-defined]
        elif self.tokenizer_required:
            smash_config.tokenizer.save_pretrained(out_dir)  # type: ignore[attr-defined]

        temp_dir = os.path.join(out_dir, "temp")
        os.mkdir(temp_dir)
        args = Namespace(
            output_dir=temp_dir,
            vocab_mapping=None,
            quantization="int8" if smash_config["weight_bits"] == 8 else "float16",
            force=True,
        )

        # For BART models due to weird hardcoded function in c_translate2
        def load_model(self: Any, model_class: Any, model_name_or_path: str, **kwargs: Any) -> None:
            model = model_class.from_pretrained(model_name_or_path, **kwargs)
            if not hasattr(model.config, "normalize_before"):
                model.config.normalize_before = False
            return model

        setattr(imported_modules["TransformersConverter"], "load_model", load_model)

        # Convert the model to the c_translate2 format
        converter = imported_modules["TransformersConverter"](
            out_dir,
            load_as_float16=True,
        )

        converter.convert_from_args(args)
        if self.task_name == "translate":
            optimized_model = imported_modules["Translator"](temp_dir, device=smash_config["device"])
            optimized_model = TranslatorWrapper(optimized_model, temp_dir, smash_config.tokenizer)
        elif self.task_name == "generate":
            optimized_model = imported_modules["Generator"](temp_dir, device=smash_config["device"])
            optimized_model = GeneratorWrapper(optimized_model, temp_dir, smash_config.tokenizer)
        elif self.task_name == "whisper":
            optimized_model = imported_modules["Whisper"](temp_dir, device=smash_config["device"])
            optimized_model = WhisperWrapper(optimized_model, temp_dir, smash_config.processor)
        else:
            raise ValueError("Task not supported")
        optimized_model.config = model.config
        return optimized_model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        cudnn_path = os.path.join(
            os.path.dirname(sys.executable).strip("bin"),
            f"lib/python{sys.version_info[0]}.{sys.version_info[1]}/site-packages/nvidia/cudnn/lib",
        )
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

        # Update LD_LIBRARY_PATH for the current process
        os.environ["LD_LIBRARY_PATH"] = f"{cudnn_path}:{ld_library_path}"

        if not os.path.exists(cudnn_path):
            # Try to find cudnn in alternative locations
            possible_paths = [
                "/usr/local/cuda/lib64",
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib/cuda/lib64",
                os.path.join(os.path.dirname(sys.executable), "lib"),
            ]

            for path in possible_paths:
                if os.path.exists(path) and any(f.startswith("libcudnn") for f in os.listdir(path)):
                    os.environ["LD_LIBRARY_PATH"] = f"{path}:{os.environ['LD_LIBRARY_PATH']}"
                    break

        # Ensure the library path is actually used by updating the dynamic loader
        try:
            import ctypes

            # Try to load the specific cuDNN library file
            cudnn_libs = [
                "libcudnn_ops.so.9.1.0",
                "libcudnn_ops.so.9.1",
                "libcudnn_ops.so.9",
                "libcudnn_ops.so",
                "libcudnn.so",
            ]
            loaded = False

            for lib in cudnn_libs:
                try:
                    ctypes.CDLL(lib)
                    loaded = True
                    break
                except Exception:
                    continue

            if not loaded:
                pruna_logger.warning("Could not load cuDNN library. CTranslate2 may not work correctly with CUDA.")
        except Exception as e:
            pruna_logger.warning(f"Error loading cuDNN: {e}")

        from ctranslate2 import Generator, Translator
        from ctranslate2.converters.transformers import (
            _MODEL_LOADERS,
            TransformersConverter,
        )
        from ctranslate2.models import Whisper

        return dict(
            _MODEL_LOADERS=_MODEL_LOADERS,
            Generator=Generator,
            Translator=Translator,
            Whisper=Whisper,
            TransformersConverter=TransformersConverter,
            AutomaticSpeechRecognitionPipeline=AutomaticSpeechRecognitionPipeline,
        )


class CGenerateCompiler(CTranslateCompiler):
    """
    Implement CGenerate compilation using ctranslate2.

    CGenerate employs a custom runtime that leverages optimizations like weight quantization, layer fusion, and
    batch reordering to boost performance and reduce memory usage on both CPUs and GPUs for Causal LM models.
    """

    algorithm_name: str = "c_generate"

    def __init__(self) -> None:
        super().__init__(task_name="generate")


class CWhisperCompiler(CTranslateCompiler):
    """
    Implement CWhisper compilation using ctranslate2.

    CWhisper employs a custom runtime that leverages optimizations like weight quantization, layer fusion, and batch
    reordering to boost performance and reduce memory usage on both CPUs and GPUs for Whisper models.
    """

    algorithm_name: str = "c_whisper"

    def __init__(self) -> None:
        super().__init__(task_name="whisper")


class GeneratorWrapper:
    """
    A wrapper for Hugging Face's Generator models.

    Parameters
    ----------
    generator : PreTrainedModel
        The underlying Hugging Face Generator model.
    output_dir : str
        The output directory for the model.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer for the model.
    """

    def __init__(self, generator: PreTrainedModel, output_dir: str, tokenizer: PreTrainedTokenizerBase) -> None:
        self.generator = generator
        self.output_dir = output_dir
        self.task = "generation"
        self.tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        """
        Forward attribute access to the wrapped generator object.

        Parameters
        ----------
        name : str
            The name of the attribute to access.

        Returns
        -------
        Any
            The attribute value.
        """
        return getattr(self.generator, name)

    def __call__(
        self,
        x: torch.Tensor | dict[str, Any] | transformers.tokenization_utils_base.BatchEncoding,
        max_length: int = 1,
        min_length: int = 1,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate a sequence from the input tensor using the generator model.

        Parameters
        ----------
        x : torch.Tensor | dict[str, Any] | transformers.tokenization_utils_base.BatchEncoding
            The input tensor.
        max_length : int
            The maximum length of the generated sequence.
        min_length : int
            The minimum length of the generated sequence.
        *args : Additional arguments for the model's `generate` algorithm.
        **kwargs : Additional keyword arguments for the model's `generate` algorithm.

        Returns
        -------
        torch.Tensor
            The generated sequence.
        """
        if type(x) is dict or isinstance(x, transformers.tokenization_utils_base.BatchEncoding):
            x_tensor = x["input_ids"]
        else:
            x_tensor = x
        token_list = [self.tokenizer.convert_ids_to_tokens(x_tensor[i]) for i in range(len(x_tensor))]
        return self.generator.generate_batch(token_list, min_length=min_length, max_length=max_length, *args, **kwargs)  # type: ignore[operator]


class TranslatorWrapper:
    """
    A wrapper for Hugging Face's Translator models.

    Parameters
    ----------
    translator : PreTrainedModel
        The underlying Hugging Face Translator model.
    output_dir : str
        The output directory for the model.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer for the model.
    """

    def __init__(self, translator: PreTrainedModel, output_dir: str, tokenizer: PreTrainedTokenizerBase) -> None:
        self.translator = translator
        self.output_dir = output_dir
        self.task = "translation"
        self.tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        """
        Forward attribute access to the wrapped generator object.

        Parameters
        ----------
        name : str
            The name of the attribute to access.

        Returns
        -------
        Any
            The attribute value.
        """
        return getattr(self.translator, name)

    def __call__(
        self,
        x: torch.Tensor | dict[str, Any] | transformers.tokenization_utils_base.BatchEncoding,
        min_decoding_length: int = 20,
        max_decoding_length: int = 20,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Translate the input tensor using the translator model.

        Parameters
        ----------
        x : torch.Tensor | dict[str, Any] | transformers.tokenization_utils_base.BatchEncoding
            The input tensor.
        min_decoding_length : int
            The minimum length of the generated sequence.
        max_decoding_length : int
            The maximum length of the generated sequence.
        *args : Additional arguments for the model's `generate` algorithm.
        **kwargs : Additional keyword arguments for the model's `generate` algorithm.

        Returns
        -------
        torch.Tensor
            The generated sequence.
        """
        if "min_length" in kwargs:
            min_decoding_length = kwargs["min_length"]
        if "max_length" in kwargs:
            max_decoding_length = kwargs["max_length"]
        if type(x) is dict or isinstance(x, transformers.tokenization_utils_base.BatchEncoding):
            x_tensor = x["input_ids"]
        else:
            x_tensor = x
        token_list = [self.tokenizer.convert_ids_to_tokens(x_tensor[i]) for i in range(len(x_tensor))]
        return self.translator.translate_batch(  # type: ignore[operator]
            token_list,
            min_decoding_length=min_decoding_length,
            max_decoding_length=max_decoding_length,
            *args,
            **kwargs,
        )


class WhisperWrapper:
    """
    A wrapper for Hugging Face's Whisper models.

    Parameters
    ----------
    whisper : PreTrainedModel
        The underlying Hugging Face Whisper model.
    output_dir : str
        The output directory for the model.
    processor : ProcessorMixin
        The tokenizer for the model.
    """

    def __init__(self, whisper: Whisper, output_dir: str, processor: ProcessorMixin) -> None:
        self.whisper = whisper
        self.output_dir = output_dir
        self.task = "whisper"
        self.processor = processor
        self.language = None
        self.prompt = None

    def __getattr__(self, name: str) -> Any:
        """
        Forward attribute access to the wrapped generator object.

        Parameters
        ----------
        name : str
            The name of the attribute to access.

        Returns
        -------
        Any
            The attribute value.
        """
        return getattr(self.whisper, name)

    def __call__(self, features: torch.Tensor, prompt: List[str] = [], *args, **kwargs) -> torch.Tensor:
        """
        Transcribe audio features using the Whisper model.

        This algorithm detects the language of the audio if not provided, sets up the prompt,
        and generates the transcription.

        Parameters:
        ----------
        features : torch.Tensor
            The input audio features to transcribe.
        prompt : List[str], optional
            A list of prompt tokens to guide the transcription. Defaults to an empty list.
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments. Can include 'language' to specify the audio language.

        Returns:
        -------
        torch.Tensor
            The generated sequence IDs representing the transcription.
        """
        from ctranslate2 import StorageView

        features = StorageView.from_array(features.cpu().numpy())
        # Detect the language.
        if kwargs.get("language"):
            language = kwargs["language"]
        elif self.language is None:
            results = self.whisper.detect_language(features)
            language, _ = results[0][0]
            self.language = language
        if kwargs.get("prompt"):
            self.prompt = kwargs["prompt"]
        elif self.prompt is None:
            self.prompt = self.processor.tokenizer.convert_tokens_to_ids(  # type: ignore[attr-defined]
                [
                    "<|startoftranscript|>",
                    language,
                    "<|transcribe|>",
                    "<|notimestamps|>",  # Remove this token to generate timestamps.
                ]
            )

        return self.whisper.generate(features, [self.prompt], *args, **kwargs)[0].sequences_ids[0]
