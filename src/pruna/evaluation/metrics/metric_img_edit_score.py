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

"""
Image Edit Score metric.

VLM-based instruction-following score for image editing. Evaluates how well an edited image
follows the given editing instruction on a 0-10 scale. Related work: EditScore (arXiv:2509.23909),
ADIEE (ICCV 2025).

When the ``ImgEdit`` benchmark provides a per-sample ``judge_prompt`` and
``source_image_bytes`` in the auxiliaries, the metric mirrors the ImgEdit paper
evaluation protocol: the judge_prompt rubric (three 1-5 criterion scores) is
filled with the editing instruction, both source and edited images are shown to
the VLM, and the minimum of the three criterion scores is normalised to [0, 1] by
dividing by 5 (consistent with VIEScore methodology: the weakest criterion governs).
Without these auxiliaries the metric falls back to a single-image generic 0-10 prompt.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import (
    SINGLE,
    metric_data_processor,
)
from pruna.evaluation.metrics.vlm_base import (
    BaseVLM,
    StatefulVLMMeanScoresMetric,
    auxiliary_dicts_from_gt,
    prompts_from_y_x_inputs,
)
from pruna.evaluation.metrics.vlm_utils import (
    FloatOutput,
    VIEScoreJsonOutput,
    _process_images,
    get_score_from_response,
    pil_rgb_from_aux_image_bytes,
    viescore_min_scores_0_10,
)

_FALLBACK_QUESTION = (
    'On a scale of 0 to 10, how well does this edited image follow the instruction "{prompt}"? '
    "0 = instruction not followed at all, 10 = perfectly executed. Reply with a single number."
)

_JUDGE_JSON_SUFFIX = (
    '\n\nProvide your three criterion scores as JSON: {"score": [score1, score2, score3]} '
    "where each score is a number from 1 to 5."
)


@MetricRegistry.register("img_edit_score")
class ImageEditScoreMetric(StatefulVLMMeanScoresMetric):
    """
    Image Edit Score metric.

    VLM-based instruction-following score for image editing. Evaluates how well an edited image
    follows the given editing instruction. Higher scores indicate better editing quality.

    When auxiliaries contain ``judge_prompt`` and ``source_image_bytes`` (as provided
    by the ImgEdit benchmark), the metric passes **both** the source (before) and edited
    (after) images to the VLM together with the dataset-specific rubric. This matches
    the ImgEdit paper's evaluation protocol. Without these fields, it falls back to a
    single-image generic question.

    Related work: EditScore (arXiv:2509.23909), ADIEE (ICCV 2025).

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
        Use structured generation (litellm pydantic; transformers outlines when applicable).
        Default is True.
    device : str | torch.device | None, optional
        Device for transformers VLM.
    api_key : str | None, optional
        API key for litellm.
    call_type : str, optional
        Call type for the metric.
    **kwargs : Any
        Additional arguments.

    Examples
    --------
    Same ``hosted`` / ``local`` pattern as :func:`~pruna.evaluation.metrics.vlm_base.get_vlm`:

    .. code-block:: python

        import torch

        from pruna.evaluation.metrics import ImageEditScoreMetric

        hosted = ImageEditScoreMetric(vlm_type="litellm", model_name="openai/gpt-4o")
        local = ImageEditScoreMetric(
            vlm_type="transformers",
            model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            device="cpu",
            vlm_kwargs={"model_load_kwargs": {"torch_dtype": torch.float32}},
        )
    """

    scores: list[float]
    default_call_type: str = "y_x"
    higher_is_better: bool = True
    metric_name: str = "img_edit_score"

    def __init__(
        self,
        *args,
        vlm: BaseVLM | None = None,
        vlm_type: Literal["litellm", "transformers"] = "litellm",
        model_name: str | None = None,
        vlm_kwargs: dict | None = None,
        structured_output: bool = True,
        device: str | torch.device | None = None,
        api_key: str | None = None,
        call_type: str = SINGLE,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device)
        self.response_format = FloatOutput if structured_output else None

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

    def update(self, x: list[Any] | torch.Tensor, gt: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the metric with new batch data.

        When ``gt`` auxiliaries contain ``judge_prompt`` and ``source_image_bytes``, the
        metric uses the dataset rubric and a before/after two-image comparison. Otherwise
        it falls back to a single-image generic question.

        Parameters
        ----------
        x : List[Any] | torch.Tensor
            The input data (editing instructions / prompts).
        gt : torch.Tensor
            Auxiliaries per sample (may contain ``judge_prompt`` and ``source_image_bytes``).
        outputs : torch.Tensor
            The output (edited) images.
        """
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        images = _process_images(inputs[0])
        prompts = prompts_from_y_x_inputs(inputs, len(images))
        aux_list = auxiliary_dicts_from_gt(gt, len(images))

        for i, image in enumerate(images):
            prompt = prompts[i] if i < len(prompts) else ""
            aux_row = aux_list[i]

            judge_prompt = aux_row.get("judge_prompt", "") or ""
            source_image = pil_rgb_from_aux_image_bytes(aux_row, min_bytes_in_value_scan=100)

            if judge_prompt and source_image is not None:
                filled = judge_prompt.replace("<edit_prompt>", prompt).strip()
                question = filled + _JUDGE_JSON_SUFFIX
                try:
                    responses = self.vlm.generate_with_image_lists(
                        [[source_image, image]], [question], response_format=VIEScoreJsonOutput
                    )
                    raw = viescore_min_scores_0_10(responses[0])
                    if raw:
                        score = max(0.0, min(1.0, float(min(raw)) / 5.0))
                        self.scores.append(score)
                        continue
                except (NotImplementedError, AttributeError):
                    pass

            question = _FALLBACK_QUESTION.format(prompt=prompt)
            responses = self.vlm.generate([image], [question], response_format=self.response_format)
            self.scores.append(get_score_from_response(responses[0]))

    def compute(self) -> MetricResult:
        """
        Compute the image edit score.

        Returns
        -------
        MetricResult
            The mean image edit score across all updates.
        """
        return self.compute_mean_of_scores()
