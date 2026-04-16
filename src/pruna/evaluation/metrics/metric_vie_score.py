# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
VIEScore metric for conditional image synthesis (semantic + quality).

Reference: VIEScore (ACL 2024) — https://arxiv.org/abs/2312.14867
Both task modes follow `TIGER-AI-Lab/VIEScore`:

- ``t2i`` (text-to-image, single image): SC uses two sub-scores (semantic consistency +
  detail correspondence), PQ uses two sub-scores (naturalness + artifacts). Overall is
  ``sqrt(min(SC) * min(PQ)) / 10``.
- ``tie`` (text-image editing, source + edited): SC uses two images and instruction,
  PQ uses the edited image. Same aggregation formula.

GEdit-Bench evaluation: https://arxiv.org/abs/2504.17761
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from PIL import Image

from pruna.evaluation.metrics.metric_vlm_base import (
    StatefulVLMMeanScoresMetric,
    auxiliary_dicts_from_gt,
)
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import (
    SINGLE,
    metric_data_processor,
)
from pruna.evaluation.metrics.viescore_prompts import (
    build_viescore_pq_prompt,
    build_viescore_t2i_sc_prompt,
    build_viescore_tie_sc_prompt,
)
from pruna.evaluation.metrics.vlm_base import BaseVLM
from pruna.evaluation.metrics.vlm_utils import (
    VIEScoreJsonOutput,
    _process_images,
    pil_rgb_from_aux_image_bytes,
    viescore_min_scores_0_10,
    viescore_tie_overall_unit,
)


@MetricRegistry.register("vie_score")
class VieScoreMetric(StatefulVLMMeanScoresMetric):
    """
    VIEScore: semantic + perceptual quality with geometric-mean overall.

    **Text-to-image (one generated image):** uses the VIEScore ``t2i`` SC prompt (semantic
    consistency + detail correspondence, 0--10 each) and the shared PQ prompt (naturalness +
    artifacts, 0--10 each). Overall is ``sqrt(min(SC) * min(PQ)) / 10`` in ``[0, 1]``.

    **Text--image editing (source + edited available):** matches the VIEScore ``tie`` setup
    used in GEdit-Bench: semantic criteria use **two** images (source then edited) and the
    editing instruction; perceptual criteria use the **edited** image only. Overall is
    ``sqrt(min(SC) * min(PQ)) / 10`` in ``[0, 1]``, with ``min`` taken over the sub-scores in
    each JSON ``score`` list, consistent with `VIEScore`_.

    .. _VIEScore: https://github.com/TIGER-AI-Lab/VIEScore

    Parameters
    ----------
    *args : Any
        Additional positional arguments.
    vlm : BaseVLM | None, optional
        Custom VLM instance. If provided, vlm_type and model_name are ignored.
    vlm_type : {"litellm", "transformers"}, optional
        VLM backend. Default is "litellm".
    model_name : str | None, optional
        Litellm model id or HuggingFace checkpoint id. **Required** when ``vlm`` is not
        provided (e.g. ``openai/gpt-4o``).
    vlm_kwargs : dict, optional
        Forwarded by ``get_vlm`` to ``LitellmVLM`` or ``TransformersVLM``. For local models,
        set ``model_load_kwargs`` for ``from_pretrained``; for litellm, pass extra API options.
    structured_output : bool, optional
        Use structured generation (litellm pydantic; transformers may use plain generation for
        multi-image). Default is True.
    device : str | torch.device | None, optional
        Device for transformers VLM.
    api_key : str | None, optional
        API key for litellm.
    call_type : str, optional
        Call type for the metric.
    **kwargs : Any
        Additional arguments.

    References
    ----------
    VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation (ACL 2024)
    https://arxiv.org/abs/2312.14867
    https://github.com/TIGER-AI-Lab/VIEScore

    GEdit-Bench (image editing evaluation)
    https://arxiv.org/abs/2504.17761

    Examples
    --------
    Same ``hosted`` / ``local`` pattern as :func:`~pruna.evaluation.metrics.vlm_base.get_vlm``.
    Multi-image ``tie`` paths call ``generate_with_image_lists`` on ``self.vlm`` internally.

    .. code-block:: python

        import torch

        from pruna.evaluation.metrics import VieScoreMetric

        hosted = VieScoreMetric(vlm_type="litellm", model_name="openai/gpt-4o")
        local = VieScoreMetric(
            vlm_type="transformers",
            model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            device="cpu",
            vlm_kwargs={"model_load_kwargs": {"torch_dtype": torch.float32}},
        )
    """

    scores: list[float]
    default_call_type: str = "y_x"
    higher_is_better: bool = True
    metric_name: str = "vie_score"

    def __init__(
        self,
        *args,
        vlm: BaseVLM | None = None,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str | None = None,
        vlm_kwargs: dict | None = None,
        structured_output: bool = True,
        device=None,
        api_key: str | None = None,
        call_type: str = SINGLE,
        **kwargs,
    ):
        super().__init__(device=device)
        self.structured_output = structured_output
        self.response_format = VIEScoreJsonOutput if structured_output else None

        self._init_vlm_scores(
            vlm=vlm,
            vlm_type=vlm_type,
            model_name=model_name,
            vlm_kwargs=vlm_kwargs,
            structured_output=structured_output,
            device=device,
            api_key=api_key,
            call_type=call_type,
        )

    def _score_single_image_t2i(self, image: Image.Image, prompt: str) -> float:
        """VIEScore ``t2i``: single-image SC (semantic + detail) and PQ (naturalness + artifacts).

        Matches the VIEScore paper's t2i evaluation: two SC sub-scores on 0--10 and two PQ
        sub-scores on 0--10, aggregated as ``sqrt(min(SC) * min(PQ)) / 10``.
        """
        sc_prompt = build_viescore_t2i_sc_prompt(prompt)
        pq_prompt = build_viescore_pq_prompt()

        rf = self.response_format if self.structured_output else None

        sc_raw = self.vlm.generate([image], [sc_prompt], response_format=rf)[0]
        pq_raw = self.vlm.generate([image], [pq_prompt], response_format=rf)[0]

        sc_list = viescore_min_scores_0_10(sc_raw)
        pq_list = viescore_min_scores_0_10(pq_raw)
        if len(sc_list) < 2:
            sc_list = sc_list + [0.0] * (2 - len(sc_list)) if sc_list else [0.0, 0.0]
        if len(pq_list) < 2:
            pq_list = pq_list + [0.0] * (2 - len(pq_list)) if pq_list else [0.0, 0.0]

        return viescore_tie_overall_unit(sc_list[:2], pq_list[:2])

    def _score_tie_gedit(self, source: Image.Image, edited: Image.Image, instruction: str) -> float:
        """VIEScore ``tie``: two-image SC, single-image PQ, overall geometric mean on 0--10 mins."""
        sc_prompt = build_viescore_tie_sc_prompt(instruction)
        pq_prompt = build_viescore_pq_prompt()

        rf = self.response_format if self.structured_output else None

        if hasattr(self.vlm, "generate_with_image_lists"):
            sc_raw = self.vlm.generate_with_image_lists(
                [[source, edited]],
                [sc_prompt],
                response_format=rf,
            )[0]
        else:
            raise RuntimeError("VLM backend must implement generate_with_image_lists for editing parity.")

        pq_raw = self.vlm.generate([edited], [pq_prompt], response_format=rf)[0]

        sc_list = viescore_min_scores_0_10(sc_raw)
        pq_list = viescore_min_scores_0_10(pq_raw)
        if len(sc_list) < 2:
            sc_list = sc_list + [0.0] * (2 - len(sc_list)) if sc_list else [0.0, 0.0]
        if len(pq_list) < 2:
            pq_list = pq_list + [0.0] * (2 - len(pq_list)) if pq_list else [0.0, 0.0]

        return viescore_tie_overall_unit(sc_list[:2], pq_list[:2])

    def update(self, x: list[Any] | torch.Tensor, gt: Any, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (prompts).
        gt : Any
            Per-sample auxiliary dicts (``prompt_with_auxiliaries_collate``), or tensor placeholders
            when aux is unused.
        outputs : torch.Tensor
            The output images.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = inputs[1] if len(inputs) > 1 and isinstance(inputs[1], list) else [""] * len(images)
        aux_list = auxiliary_dicts_from_gt(gt, len(images))

        for i, image in enumerate(images):
            prompt = prompts[i] if i < len(prompts) else ""
            aux = aux_list[i]
            source = pil_rgb_from_aux_image_bytes(aux, min_bytes_in_value_scan=100)

            if source is not None:
                self.scores.append(self._score_tie_gedit(source, image, prompt))
            else:
                self.scores.append(self._score_single_image_t2i(image, prompt))

    def compute(self) -> MetricResult:
        """
        Compute the VIEScore metric.

        Returns
        -------
        MetricResult
            The mean VIEScore across all updates.
        """
        return self.compute_mean_of_scores()
