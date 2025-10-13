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
from typing import Any, Dict, List, Union

from transformers import (
    AutomaticSpeechRecognitionPipeline,
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
        original_model = model
        if isinstance(model, WhisperWrapper):
            task = "whisper_ct"
        elif isinstance(model, AutomaticSpeechRecognitionPipeline):
            model = model.model
            task = "audio_text_transcription"
        elif isinstance(model, WhisperForConditionalGeneration):
            task = "whisper_ct"
        else:
            pruna_logger.error("Model type not supported.")
            raise ValueError("Model type not supported.")

        import tempfile

        # Determine model path
        if task == "whisper_ct" and hasattr(original_model, "output_dir") and original_model.output_dir is not None:
            # WhisperWrapper already has CTranslate2 format
            model_path = original_model.output_dir
            output_dir = original_model.output_dir
            pruna_logger.info(f"Using pre-converted CTranslate2 model from {model_path}")
        else:
            # Need to convert HuggingFace model to CTranslate2 format
            temp_dir = tempfile.mkdtemp()
            pruna_logger.info(f"Converting model to CTranslate2 format in {temp_dir}")

            # Save HuggingFace model temporarily
            hf_model_dir = os.path.join(temp_dir, "hf_model")
            model.save_pretrained(hf_model_dir)
            if hasattr(smash_config, "processor") and smash_config.processor is not None:
                smash_config.processor.save_pretrained(hf_model_dir)

            # Convert to CTranslate2 format
            ct2_model_dir = os.path.join(temp_dir, "ct2_model")
            try:
                import ctranslate2

                converter = ctranslate2.converters.TransformersConverter(hf_model_dir)
                converter.convert(ct2_model_dir, force=True)
                model_path = ct2_model_dir
                output_dir = ct2_model_dir
                pruna_logger.info("Successfully converted model to CTranslate2 format")
            except Exception as e:
                pruna_logger.error(f"Failed to convert model to CTranslate2 format: {e}")
                raise

        # Whisper model defaults
        n_mels = getattr(model.config, "num_mel_bins", 80)
        max_speech_len = 30.0
        max_text_token_len = 448

        # Initialize WhisperModel with only supported parameters
        device = "cuda" if "cuda" in self.runs_on else "cpu"
        compute_type = "int8" if smash_config["int8"] else "float16"

        pruna_logger.info(f"Initializing Faster-Whisper model with device={device}, compute_type={compute_type}")

        with SuppressOutput():
            whisper_model = imported_modules["WhisperModel"](
                model_size_or_path=model_path, device=device, compute_type=compute_type
            )

        # Store metadata on the model object after initialization
        whisper_model.output_dir = output_dir
        whisper_model.n_mels = n_mels
        whisper_model.max_speech_len = max_speech_len
        whisper_model.max_text_token_len = max_text_token_len

        pruna_logger.info(f"Preparing model for inference with batch size {smash_config.batch_size}...")
        smash_config.lock_batch_size()
        return WhisperS2TWrapper(whisper_model, smash_config.batch_size)

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
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

    def __init__(self, whisper: Any, batch_size: int | None = None) -> None:
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
            - processor: WhisperProcessor for HF models
            - language: str = None (e.g., "en")
            - task: str = "transcribe" or "translate"
            - vad_filter: bool = False
            - beam_size: int = 5
            - word_timestamps: bool = False

        Returns
        -------
        str
            The transcribed text.
        """
        if isinstance(files, str):
            files = [files]

        all_texts: List[str] = []

        # Determine if Faster-Whisper model
        is_faster_whisper = hasattr(self.whisper, "transcribe") and callable(self.whisper.transcribe)

        for audio_file in files:
            if is_faster_whisper:
                # Faster-Whisper inference
                segments, _info = self.whisper.transcribe(audio_file, batch_size=self.batch_size, *args, **kwargs)
                # Convert generator to list to avoid iteration issues
                segments_list = list(segments)
                text = " ".join([segment.text for segment in segments_list])
            else:
                # HuggingFace Whisper inference
                import torch
                import torchaudio
                from transformers import WhisperProcessor

                # Memory-safe audio loading
                waveform, sr = torchaudio.load(audio_file)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
                waveform = waveform.mean(dim=0)  # Convert to mono

                # Use provided processor or load default HF processor
                processor: WhisperProcessor = kwargs.get(
                    "processor", WhisperProcessor.from_pretrained("openai/whisper-large-v3", use_fast=True)
                )

                inputs = processor(waveform, sampling_rate=16000, return_tensors="pt").to(self.whisper.device)
                with torch.no_grad():
                    generated_ids = self.whisper.generate(**inputs)
                    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            all_texts.append(text)

        return " ".join(all_texts)
