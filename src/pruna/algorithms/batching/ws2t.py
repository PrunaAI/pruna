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
from pathlib import Path
from typing import Any, Dict, List, Union

from tokenizers import Tokenizer
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    AutoTokenizer,
    WhisperForConditionalGeneration,
)


from pruna.algorithms.batching import PrunaBatcher
from pruna.algorithms.compilation.c_translate import WhisperWrapper
from pruna.config.hyperparameters import Boolean
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_speech_seq2seq_model, is_transformers_pipeline_with_speech_recognition
from pruna.logging.filter import SuppressOutput
from pruna.logging.logger import pruna_logger


class WS2TBatcher(PrunaBatcher):
    """
    Implement whisper processing using the Faster-Whisper library (systran/faster-whisper).

    Faster-Whisper is an optimized speech-to-text pipeline built for Whisper models.
    Note: WS2T prepares the model for inference with the batch size specified in the smash config. Make sure to set the
    batch size to a value that corresponds to your inference requirements.
    """

    algorithm_name: str = "faster_whisper"
    references: dict[str, str] = {"GitHub": "https://github.com/SYSTRAN/faster-whisper"}
    tokenizer_required: bool = True
    processor_required: bool = True
    dataset_required: bool = False
    runs_on: list[str] = ["cuda", "cpu"]
    compatible_algorithms: dict[str, list[str]] = dict(
        compiler=["c_translate", "c_generate", "c_whisper"], quantizer=["half"]
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
            Boolean("int8", meta=dict(desc="Whether to quantize to int8 for inference.")),
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
        if isinstance(model, WhisperWrapper):
            return True
        return is_speech_seq2seq_model(model) or is_transformers_pipeline_with_speech_recognition(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Apply the WS2T batcher to the model.

        Parameters
        ----------
        model : Any
            The model to apply the WS2T batcher to.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the batching.

        Returns
        -------
        Any
            The smashed model.
        """
        imported_modules = self.import_algorithm_packages()
        # Infer task from model type
        if isinstance(model, WhisperWrapper):
            # This means it has already been optimized using c_translate2
            task = "whisper_ct"
        elif isinstance(model, AutomaticSpeechRecognitionPipeline):
            model = model.model
            task = "audio_text_transcription"
        elif isinstance(model, WhisperForConditionalGeneration):
            task = "whisper_ct"
        else:
            pruna_logger.error("Model type not supported.")

        output_dir = None if not hasattr(model, "output_dir") else model.output_dir

        # Requirements from WhisperS2T
        if model.config.num_mel_bins == 128:
            n_mels = 128
        elif "distil" in model.config._name_or_path:
            max_speech_len = 15.0
            max_text_token_len = 128

        temp_directory_name = "whisper"

        # ignore mypy warnings here because we ensure beforehand that processor is not None
        processor: Any = smash_config.processor  # type: ignore[attr-defined]

        if task == "audio_text_transcription":
            model.save_pretrained(f"openai/whisper-{temp_directory_name}")
            processor.save_pretrained(f"openai/whisper-{temp_directory_name}")
        elif task == "whisper_ct" and hasattr(processor.tokenizer, "name_or_path"):
            # this requires a little trick, a transformers tokenizer can not be directly converted
            # we can either go via a download or via a file and then parsing out the Tokenizer
            if Path(processor.tokenizer.name_or_path).exists():
                processor = AutoTokenizer.from_pretrained(processor.tokenizer.name_or_path, use_fast=True)
                processor = processor.backend_tokenizer
            else:
                processor = Tokenizer.from_pretrained(processor.tokenizer.name_or_path)
            processor.save(str(Path(model.output_dir) / "tokenizer.json"))
        else:
            pruna_logger.error("Please pass a Huggingface Whisper Processor.")

        model_kwargs: Dict[str, Any] = {}

        # Set device based on runs_on configuration
        device = "cuda" if "cuda" in self.runs_on else "cpu"

        # Map compute_type correctly for faster-whisper
        compute_type = "int8" if smash_config["int8"] else "float16"  # default for GPU, use "int8" for CPU if needed

        if "n_mels" in locals():
            model_kwargs["num_mel_bins"] = n_mels  # parameter name
        if "max_speech_len" in locals():
            model_kwargs["max_speech_len"] = max_speech_len
        if "max_text_token_len" in locals():
            model_kwargs["max_text_token_len"] = max_text_token_len

        with SuppressOutput():
            if task == "whisper_ct":
                # CHANGED: Use WhisperModel constructor instead of load_model function
                model = imported_modules["WhisperModel"](
                    model_size_or_path=model.output_dir, device=device, compute_type=compute_type, **model_kwargs
                )
            else:
                # CHANGED: For HF models, use model path directly
                model = imported_modules["WhisperModel"](
                    model_size_or_path=f"openai/whisper-{temp_directory_name}",
                    device=device,
                    compute_type=compute_type,
                    **model_kwargs,
                )
                shutil.rmtree(f"openai/whisper-{temp_directory_name}")

        model.output_dir = output_dir
        if "n_mels" in locals():
            model.n_mels = n_mels
        if "max_speech_len" in locals():
            model.max_speech_len = max_speech_len
        if "max_text_token_len" in locals():
            model.max_text_token_len = max_text_token_len
        pruna_logger.info(f"Preparing model for inference with batch size {smash_config.batch_size}...")
        smash_config.lock_batch_size()
        return WhisperS2TWrapper(model, smash_config.batch_size)

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        # Import WhisperModel class instead of load_model function
        from faster_whisper import WhisperModel

        return dict(WhisperModel=WhisperModel)


class WhisperS2TWrapper:
    """
    A wrapper for the Faster-Whisper model.

    Parameters
    ----------
    whisper : WhisperModel
        The underlying Faster-Whisper model.
    batch_size : int | None
        The batch size for the model.
    """

    def __init__(self, whisper: WhisperModel, batch_size: int | None = None) -> None:
        self.whisper = whisper
        self.batch_size = batch_size
        self.output_dir = getattr(whisper, "output_dir", None)

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

    def __call__(self, files: Union[str, List[str]], *args, **kwargs) -> str:
        """
        Call the model to transcribe audio files.

        Parameters
        ----------
        files : Union[str, List[str]]
            The audio files to transcribe.
        *args : Additional arguments for the model's `transcribe` method.
        **kwargs : Additional keyword arguments for the model's `transcribe` method.
            Common kwargs:
            - language: str = None (e.g., "en")
            - task: str = "transcribe" (or "translate")
            - vad_filter: bool = False (enable VAD filtering)
            - beam_size: int = 5
            - word_timestamps: bool = False

        Returns
        -------
        str
            The transcribed text.
        """
        if isinstance(files, str):
            files = [files]

        # Use transcribe() method instead of transcribe_with_vad()
        # Process single file (for compatibility with original single-file design)
        if len(files) == 1:
            segments, info = self.whisper.transcribe(files[0], batch_size=self.batch_size, *args, **kwargs)
            # Collect text from segment generator
            text = " ".join([segment.text for segment in segments])
            return text
        else:
            # For multiple files, process each separately
            all_texts = []
            for audio_file in files:
                segments, info = self.whisper.transcribe(audio_file, batch_size=self.batch_size, *args, **kwargs)
                text = " ".join([segment.text for segment in segments])
                all_texts.append(text)
            return " ".join(all_texts)
