# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Vendored from OneIG-Benchmark (commit 41b49831e79e6dde5323618c164da1c4cf0f699d).
# See NOTICE.oneig_llm2vec in parent directory.

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from peft import PeftModel
from torch import Tensor, device, nn
from tqdm import trange
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    LlamaConfig,
    PretrainedConfig,
)

from pruna.evaluation.metrics.vendor.oneig_llm2vec.models.bidirectional_llama import LlamaBiModel

logger = logging.getLogger(__name__)


def batch_to_device(batch, target_device: device | str):
    """
    Move tensor values in a batch dict to ``target_device``.

    Parameters
    ----------
    batch : dict[str, Any]
        Mapping of feature names to tensors or other values; only ``torch.Tensor``
        values are moved.
    target_device : torch.device or str
        Device to move tensors to.

    Returns
    -------
    dict[str, Any]
        The same ``batch`` object with tensors updated in place.
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class LLM2Vec(nn.Module):
    """
    Bidirectional LLM wrapper with configurable pooling for dense embeddings.

    Parameters
    ----------
    model : transformers.AutoModel
        Encoder model used for hidden states.
    tokenizer : transformers.AutoTokenizer
        Tokenizer aligned with ``model``.
    pooling_mode : str, optional
        How to pool token hidden states (e.g. ``mean``, ``eos_token``).
    max_length : int, optional
        Maximum sequence length for tokenization.
    doc_max_length : int, optional
        Soft cap used when shortening document segments during encoding.
    skip_instruction : bool, optional
        If True, restrict attention to embed regions when pooling.
    """

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        pooling_mode: str = "mean",
        max_length: int = 512,
        doc_max_length: int = 512,
        skip_instruction: bool = True,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.pooling_mode = pooling_mode
        self.skip_instruction = skip_instruction
        self.max_length = max_length
        self.doc_max_length = doc_max_length
        self.config = model.config

    @classmethod
    def _get_model_class(cls, config_class_name, enable_bidirectional):
        if not enable_bidirectional:
            return AutoModel
        elif config_class_name == "LlamaConfig":
            return LlamaBiModel
        else:
            raise ValueError(f"{config_class_name} is not supported yet with bidirectional models.")

    @classmethod
    def from_pretrained(
        cls,
        base_model_name_or_path,
        peft_model_name_or_path=None,
        merge_peft=False,
        enable_bidirectional=True,
        extra_model_name_or_path=None,
        **kwargs,
    ):
        """
        Load tokenizer and encoder weights and return an ``LLM2Vec`` instance.

        Optional PEFT adapters, bidirectional Llama, and extra adapter paths are
        supported; keyword arguments are forwarded to Hugging Face
        ``from_pretrained`` calls.

        Parameters
        ----------
        base_model_name_or_path : str or pathlib.Path
            Hub id or local directory for the base model.
        peft_model_name_or_path : str or pathlib.Path, optional
            Optional PEFT adapter to load on top of the base model.
        merge_peft : bool, optional
            If True, merge PEFT weights into the base weights after loading.
        enable_bidirectional : bool, optional
            If True, use bidirectional Llama when the config is ``LlamaConfig``.
        extra_model_name_or_path : str, list of str, or None, optional
            Additional PEFT checkpoint(s) applied sequentially when set.
        **kwargs
            Forwarded to Hugging Face ``from_pretrained`` (and related) calls.

        Returns
        -------
        LLM2Vec
            Configured wrapper around the loaded encoder and tokenizer.
        """
        keys = ["pooling_mode", "max_length", "doc_max_length", "skip_instruction"]
        encoder_args = {key: kwargs.pop(key, None) for key in keys if kwargs.get(key) is not None}

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(base_model_name_or_path)
        config_class_name = config.__class__.__name__

        model_class = cls._get_model_class(config_class_name, enable_bidirectional=enable_bidirectional)
        model = model_class.from_pretrained(base_model_name_or_path, **kwargs)

        base_path = pathlib.Path(base_model_name_or_path)
        config_json = base_path / "config.json"
        if base_path.is_dir() and config_json.exists():
            with open(config_json, encoding="utf-8") as config_file:
                config_dict = json.load(config_file)
            config = PretrainedConfig.from_dict(config_dict)
            model.config._name_or_path = config._name_or_path

        if hasattr(model, "peft_config"):
            model = PeftModel.from_pretrained(
                model,
                base_model_name_or_path,
            )
            model = model.merge_and_unload()

        if peft_model_name_or_path is not None:
            model = PeftModel.from_pretrained(
                model,
                peft_model_name_or_path,
            )
            if merge_peft:
                model = model.merge_and_unload()
        if extra_model_name_or_path is not None:
            logger.info(f"Loading extra model from {extra_model_name_or_path}")
            if not merge_peft:
                model = model.merge_and_unload()
            if isinstance(extra_model_name_or_path, str):
                model = PeftModel.from_pretrained(
                    model,
                    extra_model_name_or_path,
                )
                peft_model_name_or_path = extra_model_name_or_path
                model = model.merge_and_unload()
            elif isinstance(extra_model_name_or_path, list):
                for extra_model in extra_model_name_or_path:
                    model = PeftModel.from_pretrained(
                        model,
                        extra_model,
                    )
                    peft_model_name_or_path = extra_model
                    model = model.merge_and_unload()
            else:
                raise ValueError("extra_model_name_or_path should be a string or a list of strings.")
        config = {}
        config_addr = peft_model_name_or_path if peft_model_name_or_path is not None else base_model_name_or_path
        llm2vec_config_path = pathlib.Path(config_addr) / "llm2vec_config.json"
        if llm2vec_config_path.exists():
            with open(llm2vec_config_path, encoding="utf-8") as config_file:
                llm2vec_config = json.load(config_file)
            config.update(llm2vec_config)
            logger.info(f"LLM2Vec config: {config}")
        for key, value in encoder_args.items():
            config[key] = value

        return cls(model=model, tokenizer=tokenizer, **config)

    def prepare_for_tokenization(self, text):
        """
        Apply model-specific chat or EOS wrappers so tokenization matches training.

        Parameters
        ----------
        text : str
            Raw input text before tokenization.

        Returns
        -------
        str
            Text with any required special tokens or chat template prefixes or suffixes.
        """
        if "Llama-3" in self.model.config._name_or_path and "Instruct" in self.model.config._name_or_path:
            text = "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
            return text
        if self.model.config._name_or_path == "microsoft/Phi-3.5-mini-instruct":
            text = "<|user|>\n" + text.strip() + "<|end|>\n"
            return text
        if self.pooling_mode == "eos_token":
            if self.model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
                text = text.strip() + "<|end_of_text|>"
            elif isinstance(self.model.config, LlamaConfig):
                text = text.strip() + " "
        return text

    def tokenize(self, texts):
        """
        Tokenize texts with optional embed-region markers for instruction/document split.

        Parameters
        ----------
        texts : list of str
            Strings that may contain the ``!@#$%^&*()`` delimiter between instruction and document.

        Returns
        -------
        dict[str, torch.Tensor]
            Tokenizer outputs including ``embed_mask`` when the delimiter is present.
        """
        texts_2 = []
        original_texts = []
        for text in texts:
            t = text.split("!@#$%^&*()")
            texts_2.append(t[1] if len(t) > 1 else "")
            original_texts.append("".join(t))

        original = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            if embed_mask is None:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(len(ids["input_ids"][0]))
                embed_mask = e_m.unsqueeze(0)
            else:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(len(ids["input_ids"][0]))
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        original["embed_mask"] = embed_mask
        return original

    def _skip_instruction(self, sentence_feature):
        assert sentence_feature["attention_mask"].shape == sentence_feature["embed_mask"].shape
        sentence_feature["attention_mask"] = sentence_feature["embed_mask"]

    def forward(self, sentence_feature: Dict[str, Tensor]):
        """
        Run the encoder and return pooled sentence embeddings.

        Parameters
        ----------
        sentence_feature : dict[str, torch.Tensor]
            Batch of tokenizer outputs; may include ``embed_mask`` for instruction masking.

        Returns
        -------
        torch.Tensor
            Pooled embeddings with shape ``(batch_size, hidden_size)``.
        """
        embed_mask = None
        if "embed_mask" in sentence_feature:
            embed_mask = sentence_feature.pop("embed_mask")
        reps = self.model(**sentence_feature)
        if embed_mask is not None:
            sentence_feature["embed_mask"] = embed_mask

        return self.get_pooling(sentence_feature, reps.last_hidden_state)

    def get_pooling(self, features, last_hidden_states):
        """
        Pool token hidden states according to ``pooling_mode``.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            Tokenizer batch (attention mask, optional ``embed_mask``, etc.).
        last_hidden_states : torch.Tensor
            Sequence hidden states from the encoder, shape ``(batch, seq, hidden)``.

        Returns
        -------
        torch.Tensor
            Pooled embeddings, shape ``(batch, hidden)``.
        """
        assert self.tokenizer.padding_side == "left", "Pooling modes are implemented for padding from left."
        if self.skip_instruction:
            self._skip_instruction(features)
        seq_lengths = features["attention_mask"].sum(dim=-1)
        if self.pooling_mode == "mean":
            return torch.stack(
                [last_hidden_states[i, -length:, :].mean(dim=0) for i, length in enumerate(seq_lengths)],
                dim=0,
            )
        elif self.pooling_mode == "weighted_mean":
            bs, seq_len, _ = last_hidden_states.shape
            complete_weights = torch.zeros(bs, seq_len, device=last_hidden_states.device)
            for i, seq_l in enumerate(seq_lengths):
                if seq_l > 0:
                    complete_weights[i, -seq_l:] = torch.arange(seq_l) + 1
                    complete_weights[i] /= torch.clamp(complete_weights[i].sum(), min=1e-9)
            return torch.sum(last_hidden_states * complete_weights.unsqueeze(-1), dim=1)
        elif self.pooling_mode == "eos_token" or self.pooling_mode == "last_token":
            return last_hidden_states[:, -1]
        elif self.pooling_mode == "bos_token":
            return last_hidden_states[features["input_ids"] == self.tokenizer.bos_token_id]
        else:
            raise ValueError(f"{self.pooling_mode} is not implemented yet.")

    def _convert_to_str(self, instruction, text):
        tokenized_q = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        tokenized_q_length = len(tokenized_q["input_ids"][0])

        while tokenized_q_length > self.doc_max_length:
            reduction_ratio = self.doc_max_length / tokenized_q_length
            reduced_length = int(len(text.split()) * reduction_ratio)
            text = " ".join(text.split()[:reduced_length])
            tokenized_q = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            tokenized_q_length = len(tokenized_q["input_ids"][0])

        return f"{instruction.strip()} !@#$%^&*(){text}" if instruction else f"!@#$%^&*(){text}"

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = True,
        device: Optional[str] = None,
    ):
        """
        Encode sentences (optionally instruction + document) to embedding tensors.

        Parameters
        ----------
        sentences : str, list of str, or nested list
            Plain strings, or ``[instruction, document]`` pairs, or batches thereof.
        batch_size : int, optional
            Micro-batch size during encoding.
        show_progress_bar : bool, optional
            Ignored; progress is disabled in the implementation.
        convert_to_numpy : bool, optional
            If True, return a NumPy array instead of a tensor (mutually exclusive with ``convert_to_tensor``).
        convert_to_tensor : bool, optional
            If True (default), return a ``torch.Tensor`` of dtype float32.
        device : str, optional
            Device name; defaults to CUDA when available else CPU.

        Returns
        -------
        torch.Tensor or numpy.ndarray
            Stacked embeddings for all inputs, reordered to the original sentence order.
        """
        seq: Any = sentences
        if isinstance(seq[0], str) and isinstance(seq[-1], int):
            seq = [seq]
        if isinstance(seq[0], str):
            seq = [[""] + [sentence] for sentence in seq]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        concatenated_input_texts = []
        for sentence in seq:
            assert isinstance(sentence[0], str)
            assert isinstance(sentence[1], str)
            concatenated_input_texts.append(self._convert_to_str(sentence[0], sentence[1]))
        sentences = concatenated_input_texts

        self.train(mode=False)

        if convert_to_tensor:
            convert_to_numpy = False

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        all_embeddings = []

        self.to(device)
        for start_index in trange(
            0,
            len(sentences),
            batch_size,
            desc="Batches",
            disable=True,
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            embeddings = self._encode(sentences_batch, device=device, convert_to_numpy=convert_to_numpy)
            all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]
        all_embeddings = all_embeddings.to(torch.float32)
        if convert_to_numpy:
            return all_embeddings.cpu().numpy()
        return all_embeddings

    def save(self, output_path, merge_before_save=False, save_config=True):
        """
        Persist model, tokenizer, and optional ``llm2vec_config.json`` to ``output_path``.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Directory to write weights and tokenizer files into.
        merge_before_save : bool, optional
            If True and the inner model is a ``PeftModel``, merge adapters before saving.
        save_config : bool, optional
            If True, write ``llm2vec_config.json`` with pooling and length settings.
        """
        if merge_before_save and isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
        if hasattr(self.model, "_hf_peft_config_loaded"):
            setattr(self.model, "_hf_peft_config_loaded", False)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        llm2vec_config = {
            "pooling_mode": self.pooling_mode,
            "max_length": self.max_length,
            "doc_max_length": self.doc_max_length,
            "skip_instruction": self.skip_instruction,
        }

        if save_config:
            pathlib.Path(output_path).mkdir(exist_ok=True, parents=True)
            config_out = pathlib.Path(output_path) / "llm2vec_config.json"
            with open(config_out, "w", encoding="utf-8") as config_file:
                json.dump(llm2vec_config, config_file, indent=4)

    def _encode(
        self,
        sentences_batch,
        device: Optional[str] = None,
        convert_to_numpy: bool = False,
        multiprocessing=False,
    ):
        if multiprocessing:
            rank = mp.current_process()._identity[0]
            if device is None and torch.cuda.is_available():
                device = f"cuda:{rank % torch.cuda.device_count()}"

        use_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(use_device)
        features = self.tokenize([self.prepare_for_tokenization(sentence) for sentence in sentences_batch])
        features = batch_to_device(features, use_device)

        with torch.no_grad():
            embeddings = self.forward(features)
        return embeddings

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        if isinstance(text, str) or (isinstance(text, list) and isinstance(text[0], int)) or len(text) == 0:
            return len(text)
        if isinstance(text, dict):
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):
            return 1
        else:
            return sum(len(t) if not isinstance(t, int) else 1 for t in text)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        """
        Resize the underlying model token embedding matrix.

        Parameters
        ----------
        new_num_tokens : int, optional
            New vocabulary size for the embedding table.
        pad_to_multiple_of : int, optional
            Pad vocabulary size to a multiple of this value when resizing.

        Returns
        -------
        torch.nn.Embedding
            The resized embedding module from the wrapped model.
        """
        return self.model.resize_token_embeddings(new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing on the wrapped model.

        Parameters
        ----------
        gradient_checkpointing_kwargs : dict, optional
            Keyword arguments forwarded to the underlying ``gradient_checkpointing_enable`` call.
        """
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
