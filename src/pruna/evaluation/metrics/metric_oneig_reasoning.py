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

"""OneIG reasoning score via LLM2CLIP text-image similarity.

Llama-derived checkpoints may require ``HF_TOKEN`` and ``huggingface-cli login``.

Hugging Face download tuning (optional):

- ``PRUNA_ONEIG_HF_VERBOSE=1`` or ``HF_DEBUG=1`` — hub **debug** logging and tqdm
  progress bars (helps when stderr is piped; pair with ``python -u`` or
  ``PYTHONUNBUFFERED=1`` for line-buffered output).
- ``PRUNA_ONEIG_HF_FAST_DOWNLOAD=1`` — enable **hf_transfer** multi-part downloads
  (requires ``pruna[evaluation]``, which lists ``hf_transfer``). Alternatively, set
  ``HF_HUB_ENABLE_HF_TRANSFER=1`` **before** starting Python so the hub picks it up at
  import time.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.vlm_utils import _process_images
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import (
    SINGLE,
    get_call_type_for_single_metric,
    metric_data_processor,
)
from pruna.logging.logger import pruna_logger


def _env_truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().upper() in {"1", "ON", "YES", "TRUE"}


def _prepare_huggingface_hub_for_oneig_downloads() -> None:
    """
    Apply Hugging Face Hub verbosity and optional fast downloads before checkpoints load.

    ``HF_HUB_ENABLE_HF_TRANSFER`` is read when ``huggingface_hub`` loads; if it was
    false, we flip the in-module flag after importing ``hf_transfer`` when
    ``PRUNA_ONEIG_HF_FAST_DOWNLOAD=1``.
    """
    if _env_truthy(os.environ.get("PRUNA_ONEIG_HF_VERBOSE")) or _env_truthy(os.environ.get("HF_DEBUG")):
        from huggingface_hub.utils import enable_progress_bars
        from huggingface_hub.utils.logging import set_verbosity_debug

        set_verbosity_debug()
        enable_progress_bars()

    if not _env_truthy(os.environ.get("PRUNA_ONEIG_HF_FAST_DOWNLOAD")):
        return

    import hf_transfer  # noqa: F401  # type: ignore[import-not-found]

    import huggingface_hub.constants as hf_constants

    hf_constants.HF_HUB_ENABLE_HF_TRANSFER = True
    pruna_logger.info(
        "oneig_reasoning: enabled hf_transfer downloads (PRUNA_ONEIG_HF_FAST_DOWNLOAD=1)."
    )


def _to_pil_list(images: list) -> list:
    """Convert images to list of PIL.Image (RGB)."""
    from PIL import Image

    import numpy as np

    out: list = []
    for img in images:
        if isinstance(img, Image.Image):
            out.append(img.convert("RGB"))
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:
                img = img[0]
            if img.max() > 1:
                img = img / 255.0
            np_img = (img.cpu().numpy() * 255).astype("uint8")
            if np_img.shape[0] == 3:
                np_img = np_img.transpose(1, 2, 0)
            out.append(Image.fromarray(np_img))
        elif hasattr(img, "__array__"):
            out.append(Image.fromarray(np.asarray(img)).convert("RGB"))
        else:
            out.append(img)
    return out


class _LLM2CLIPScorer:
    """
    Thin wrapper around LLM2CLIP text-image similarity.

    Accepts PIL images and a single answer string; returns per-image scores.
    Best-effort alignment with OneIG-Benchmark scripts (CUDA + bfloat16).
    """

    def __init__(
        self,
        processor_model: str = "openai/clip-vit-large-patch14-336",
        model_name: str = "microsoft/LLM2CLIP-Openai-L-14-336",
        llm_model_name: str = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned",
        device: str = "cuda",
    ) -> None:
        self.processor_model = processor_model
        self.model_name = model_name
        self.llm_model_name = llm_model_name
        self.device = device
        self._processor = None
        self._clip_model = None
        self._l2v = None

    def _load_models(self) -> None:
        if self._clip_model is not None:
            return
        _prepare_huggingface_hub_for_oneig_downloads()
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers import CLIPImageProcessor

        from pruna.evaluation.metrics.vendor.oneig_llm2vec import LLM2Vec
        from pruna.evaluation.metrics.vendor.oneig_llm2vec.modeling_llama_encoder import LlamaEncoderModel

        pruna_logger.info(
            "oneig_reasoning: downloading or loading LLM2CLIP checkpoints "
            "(%s, %s). First run can take many minutes and several gigabytes; "
            "Hugging Face download progress may look idle when logs are piped.",
            self.model_name,
            self.llm_model_name,
        )
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._processor = CLIPImageProcessor.from_pretrained(self.processor_model)
        self._clip_model = AutoModel.from_pretrained(
            self.model_name,
            dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self._clip_model.train(mode=False)

        config = AutoConfig.from_pretrained(self.llm_model_name, trust_remote_code=True)
        dev_str = str(self.device)
        attn_impl = "sdpa" if dev_str == "cuda" or dev_str.startswith("cuda:") else "eager"
        config.attn_implementation = attn_impl
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = attn_impl
        llm_model = LlamaEncoderModel.from_pretrained(
            self.llm_model_name,
            dtype=dtype,
            config=config,
            trust_remote_code=True,
        )
        llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self._l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

    def score(self, images: list, text_prompt: str) -> list[float] | None:
        """
        Compute similarity scores between images and text.

        Parameters
        ----------
        images : list
            List of PIL.Image.Image.
        text_prompt : str
            Reference text (e.g. ground-truth answer).

        Returns
        -------
        list[float] | None
            Per-image scores, or None on failure.
        """
        self._load_models()
        pil_images = _to_pil_list(images)
        if not pil_images:
            return None
        input_pixels = self._processor(images=pil_images, return_tensors="pt").pixel_values.to(self.device)
        captions = [text_prompt]
        text_features = self._l2v.encode(captions, convert_to_tensor=True, device=self.device).to(self.device)
        text_features = self._clip_model.get_text_features(text_features)

        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    image_features = self._clip_model.get_image_features(input_pixels)
            else:
                image_features = self._clip_model.get_image_features(input_pixels.float())

        image_features = image_features.float()
        text_features = text_features.float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (image_features @ text_features.T).cpu().tolist()
        return [p[0] for p in text_probs]


@MetricRegistry.register("oneig_reasoning")
class OneIGReasoningMetric(StatefulMetric):
    """
    OneIG reasoning score: LLM2CLIP similarity between GT answer text and generated image.

    Uses ``reasoning_gt_answer`` from aux (populated by OneIG Knowledge_Reasoning loader;
    language is chosen at dataset load via ``reasoning_language``). MVP: 1×1 grid (whole
    image as single cell). Llama-derived checkpoints may require
    ``HF_TOKEN`` and ``huggingface-cli login``.

    Parameters
    ----------
    processor_model : str, optional
        CLIP processor model ID.
    model_name : str, optional
        LLM2CLIP model ID.
    llm_model_name : str, optional
        LLM2Vec model ID.
    device : str | torch.device | None, optional
        Device for inference.
    scorer : _LLM2CLIPScorer | None, optional
        Optional scorer instance for testing (injected mock).
    call_type : str, optional
        Call type for the metric.
    **kwargs : Any
        Additional keyword arguments for :class:`StatefulMetric`.

    Notes
    -----
    Prompt benchmarks yield ``(prompts, aux_list)``. With default ``call_type``
    ``y_gt``, ``aux_list`` is the list (or tensor coerced to a list) of per-sample
    dicts parallel to generated images. Each dict must include a non-empty
    ``reasoning_gt_answer`` for Knowledge/Reasoning samples. Missing GT, scorer
    failures, or :meth:`compute` with no scored samples raise ``ValueError`` or
    ``RuntimeError`` instead of returning a placeholder score.
    """

    metric_name: str = "oneig_reasoning"
    default_call_type: str = "y_gt"
    higher_is_better: bool = True
    runs_on: list[str] = ["cuda", "cpu"]

    def __init__(
        self,
        processor_model: str = "openai/clip-vit-large-patch14-336",
        model_name: str = "microsoft/LLM2CLIP-Openai-L-14-336",
        llm_model_name: str = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned",
        device: str | torch.device | None = None,
        scorer: _LLM2CLIPScorer | None = None,
        call_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, **kwargs)
        self.call_type = get_call_type_for_single_metric(
            call_type if call_type is not None else SINGLE, self.default_call_type
        )
        self.processor_model = processor_model
        self.model_name = model_name
        self.llm_model_name = llm_model_name
        self._scorer = scorer
        self.add_state("scores", default=[])

    def _get_scorer(self) -> _LLM2CLIPScorer:
        if self._scorer is not None:
            return self._scorer
        return _LLM2CLIPScorer(
            processor_model=self.processor_model,
            model_name=self.model_name,
            llm_model_name=self.llm_model_name,
            device=self.device,
        )

    def _get_gt_text(self, aux: dict) -> str:
        val = aux.get("reasoning_gt_answer")
        if val is None or (isinstance(val, str) and not val.strip()):
            raise ValueError(
                "oneig_reasoning requires 'reasoning_gt_answer' in aux for Knowledge_Reasoning rows. "
                f"Got keys: {list(aux.keys())}."
            )
        return str(val).strip()

    def update(self, x: list[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Score each image against its GT answer text via LLM2CLIP similarity.

        Parameters
        ----------
        x : list[Any] | torch.Tensor
            Unused batch metadata.
        gt : torch.Tensor
            Ground-truth slot with per-sample aux dicts containing ``reasoning_gt_answer``.
        outputs : torch.Tensor
            Model outputs (generated images).

        Raises
        ------
        ValueError
            If a per-sample aux entry is not a dict or lacks a non-empty
            ``reasoning_gt_answer``.
        RuntimeError
            If the LLM2CLIP scorer returns no scores for a sample.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        aux_list = inputs[1] if len(inputs) > 1 else []
        if isinstance(aux_list, torch.Tensor):
            aux_list = aux_list.tolist()

        scorer = self._get_scorer()

        for i, image in enumerate(images):
            aux = aux_list[i] if i < len(aux_list) else {}
            if not isinstance(aux, dict):
                raise ValueError(
                    f"oneig_reasoning requires aux[{i}] to be a dict. Got: {type(aux)}."
                )
            text = self._get_gt_text(aux)
            result = scorer.score([image], text)
            if result is None or len(result) == 0:
                raise RuntimeError(
                    f"oneig_reasoning: LLM2CLIP scorer returned no scores for sample {i}."
                )
            self.scores.append(float(sum(result) / len(result)))

    def compute(self) -> MetricResult:
        """
        Compute the mean reasoning score across all samples.

        Returns
        -------
        MetricResult
            Mean LLM2CLIP similarity.

        Raises
        ------
        RuntimeError
            If :meth:`update` was not called or scored no samples.
        """
        if not self.scores:
            raise RuntimeError(
                "oneig_reasoning: no samples were scored; call update() with valid "
                "batches and non-empty reasoning_gt_answer before compute()."
            )
        mean_score = sum(self.scores) / len(self.scores)
        return MetricResult(self.metric_name, self.__dict__, float(mean_score))
