"""Tests for VLM metrics (VQA, AlignmentScore, ImageEditScore, QAAccuracy, TextScore, VieScore)."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from pruna.evaluation.metrics.metric_alignment_score import AlignmentScoreMetric
from pruna.evaluation.metrics.metric_img_edit_score import ImageEditScoreMetric
from pruna.evaluation.metrics.metric_oneig_alignment import OneIGAlignmentMetric
from pruna.evaluation.metrics.metric_qa_accuracy import QAAccuracyMetric
from pruna.evaluation.metrics.metric_text_score import OneIGTextScoreMetric, TextScoreMetric
from pruna.evaluation.metrics.metric_vie_score import VieScoreMetric
from pruna.evaluation.metrics.metric_vqa import VQAMetric
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.vlm_base import BaseVLM, get_vlm
from pruna.evaluation.metrics.vlm_utils import yes_no_first_token_id_groups
from pruna.evaluation.vlm_benchmark_helpers import (
    BenchmarkVlmBatchOutcome,
    _safe_json,
    vlm_benchmark_batch_to_json_record,
)

SMOL_VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"

_ALL_VLM = (
    VQAMetric,
    AlignmentScoreMetric,
    ImageEditScoreMetric,
    QAAccuracyMetric,
    OneIGAlignmentMetric,
    TextScoreMetric,
    OneIGTextScoreMetric,
    VieScoreMetric,
)

_SLOW_SMOL_SUBSET = (
    VQAMetric,
    OneIGAlignmentMetric,
    ImageEditScoreMetric,
    VieScoreMetric,
)


def _dummy_image(batch: int = 1, size: int = 224) -> torch.Tensor:
    return torch.rand(batch, 3, size, size)


def _update_metric(metric: object, prompts: list, images: torch.Tensor) -> None:
    if isinstance(metric, OneIGAlignmentMetric):
        metric.update(
            prompts,
            [
                {
                    "questions": {"1": "Is there a cat?", "2": "Is it sleeping?"},
                    "dependencies": {"1": [0], "2": [1]},
                }
            ],
            images,
        )
    elif isinstance(metric, QAAccuracyMetric):
        metric.update(
            prompts,
            [{"questions": {"1": "Is there a cat?"}}],
            images,
        )
    elif isinstance(metric, (TextScoreMetric, OneIGTextScoreMetric)):
        metric.update(prompts, ["cat"], images)
    else:
        metric.update(prompts, images, images)


@pytest.mark.cpu
@pytest.mark.slow
@pytest.mark.parametrize("metric_cls", _SLOW_SMOL_SUBSET)
def test_vlm_metrics_transformers_smolvlm(metric_cls: type) -> None:
    """Smoke-test a subset with local SmolVLM (full matrix covered by litellm mock)."""
    metric = metric_cls(
        vlm_type="transformers",
        model_name=SMOL_VLM,
        device="cpu",
        structured_output=True,
    )
    images = _dummy_image(batch=1)
    prompts = ["a cat"]
    _update_metric(metric, prompts, images)
    result = metric.compute()
    assert result.name == metric.metric_name
    assert isinstance(result.result, float)
    if metric.higher_is_better:
        assert 0.0 <= result.result <= 1.0
    else:
        assert result.result >= 0.0


@pytest.mark.cpu
@pytest.mark.parametrize("metric_cls", _ALL_VLM)
def test_vlm_metrics_litellm_mocked(metric_cls: type) -> None:
    """Each VLM metric runs end-to-end with mocked litellm."""
    pytest.importorskip("litellm")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    if metric_cls in (AlignmentScoreMetric, VQAMetric, QAAccuracyMetric, OneIGAlignmentMetric):
        mock_response.choices[0].message.content = '{"answer": "Yes"}'
    else:
        mock_response.choices[0].message.content = '{"score": 8}'

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = mock_response

        metric = metric_cls(
            vlm_type="litellm",
            model_name="gpt-4o",
            device="cpu",
            structured_output=True,
        )
        images = _dummy_image(batch=1)
        prompts = ["a cat"]
        _update_metric(metric, prompts, images)
        result = metric.compute()

    assert result.name == metric.metric_name
    assert isinstance(result.result, float)
    assert mock_completion.called


@pytest.mark.cpu
def test_vlm_metrics_empty_compute_returns_zero() -> None:
    """No updates → compute is 0.0 (same for all stateful VLM metrics)."""
    metric = VQAMetric(
        vlm_type="transformers",
        model_name=SMOL_VLM,
        device="cpu",
        structured_output=True,
    )
    assert metric.compute().result == 0.0


@pytest.mark.cpu
def test_vlm_metrics_custom_vlm() -> None:
    """Custom VLM passed to VQAMetric is used instead of the default litellm backend."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.generate.return_value = ["Yes"]
    mock_vlm.score.return_value = [1.0]

    metric = VQAMetric(
        vlm=mock_vlm, vlm_type="litellm", device="cpu", structured_output=True
    )
    images = _dummy_image(batch=1)
    prompts = ["a cat"]
    metric.update(prompts, images, images)
    assert metric.compute().result == 1.0
    mock_vlm.score.assert_called()


@pytest.mark.cpu
def test_get_vlm_returns_custom() -> None:
    """get_vlm returns the provided VLM instance unchanged."""
    custom = MagicMock(spec=BaseVLM)
    out = get_vlm(vlm=custom, vlm_type="litellm", model_name="gpt-4o")
    assert out is custom


@pytest.mark.cpu
def test_yes_no_first_token_id_groups_disjoint() -> None:
    """Prefix token ids for Yes vs No should not overlap (avoids double-counting)."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    yes_ids, no_ids = yes_no_first_token_id_groups(tok)
    assert yes_ids and no_ids
    assert not (set(yes_ids) & set(no_ids))


@pytest.mark.cpu
def test_get_vlm_requires_model_name_without_vlm() -> None:
    """get_vlm raises ValueError when no model_name is given and no vlm is provided."""
    with pytest.raises(ValueError, match="model_name"):
        get_vlm(vlm=None, vlm_type="litellm")


@pytest.mark.cpu
@pytest.mark.parametrize(
    "metric_cls, expected_name, expected_result",
    [
        (TextScoreMetric, "text_score", 1.0),
        (OneIGTextScoreMetric, "oneig_text_score", 1.0),
    ],
)
def test_text_metrics_list_str_gt(
    metric_cls: type, expected_name: str, expected_result: float
) -> None:
    """Text metrics accept plain string ground-truth and return the expected score."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.generate.return_value = ["hello world"]

    metric = metric_cls(vlm=mock_vlm, vlm_type="litellm", device="cpu")
    images = _dummy_image(batch=1)
    metric.update(["a prompt"], ["hello world"], images)
    result = metric.compute()

    assert result.result == expected_result
    assert result.name == expected_name
    mock_vlm.generate.assert_called_once()


@pytest.mark.cpu
def test_text_score_result_in_zero_one_range() -> None:
    """TextScoreMetric must return a normalized score in [0, 1], not raw edit distance."""
    mock_vlm = MagicMock(spec=BaseVLM)
    # VLM OCR returns something very different from ground truth (high edit distance)
    mock_vlm.generate.return_value = ["completely wrong text abcdefghijklmnop"]

    metric = TextScoreMetric(vlm=mock_vlm, device="cpu")
    images = _dummy_image(batch=1)
    metric.update(["prompt"], ["hello"], images)
    result = metric.compute()

    assert 0.0 <= result.result <= 1.0, f"TextScoreMetric must return [0,1], got {result.result}"
    assert result.result < 0.5, f"Very different strings should score below 0.5, got {result.result}"


@pytest.mark.cpu
def test_text_score_perfect_match_is_one() -> None:
    """TextScoreMetric: identical OCR and GT -> score 1.0."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.generate.return_value = ["hello world"]

    metric = TextScoreMetric(vlm=mock_vlm, device="cpu")
    images = _dummy_image(batch=1)
    metric.update(["prompt"], ["hello world"], images)
    result = metric.compute()

    assert result.result == 1.0, f"Perfect match should give 1.0, got {result.result}"
    assert result.higher_is_better is True


@pytest.mark.cpu
def test_text_score_registry_aliases() -> None:
    """Registry aliases ocr_levenshtein and ocr_text_score resolve to the correct metric classes."""
    from pruna.evaluation.metrics.registry import MetricRegistry

    lev = MetricRegistry.get_metric("ocr_levenshtein", device="cpu", model_name="openai/gpt-4o")
    comp = MetricRegistry.get_metric("ocr_text_score", device="cpu", model_name="openai/gpt-4o")
    assert type(lev).__name__ == "TextScoreMetric"
    assert type(comp).__name__ == "OneIGTextScoreMetric"
    assert lev.metric_name == "text_score"
    assert comp.metric_name == "oneig_text_score"


@pytest.mark.cpu
def test_oneig_text_score_utils_golden_composite() -> None:
    """oneig_mean_text_score returns expected component values for a known input."""
    from pruna.evaluation.metrics.metric_text_score_utils import oneig_mean_text_score

    ed, cr, wac, composite = oneig_mean_text_score(
        edit_distances=[10.0],
        completion_ratios=[0.0],
        match_counts=[2],
        gt_totals=[4],
        language_mode="EN",
    )
    assert ed == 10.0
    assert cr == 0.0
    assert wac == 0.5
    assert composite == pytest.approx(0.95)

    _, _, _, zh = oneig_mean_text_score(
        edit_distances=[30.0],
        completion_ratios=[0.0],
        match_counts=[0],
        gt_totals=[1],
        language_mode="ZH",
    )
    assert zh == pytest.approx(0.4)


@pytest.mark.cpu
def test_qa_accuracy_all_or_nothing_partial_fail() -> None:
    """all_or_nothing: if any question scores 0, the image score is 0.0 (not a partial mean)."""
    mock_vlm = MagicMock(spec=BaseVLM)
    # First question Yes (1.0), second question No (0.0) → mean=0.5, all_or_nothing=0.0
    mock_vlm.score.return_value = [1.0, 0.0]

    metric = QAAccuracyMetric(vlm=mock_vlm, device="cpu", aggregation="all_or_nothing")
    metric.update(
        ["a prompt"],
        [{"questions": {"1": "Is there a cat?", "2": "Is it blue?"}}],
        _dummy_image(batch=1),
    )
    result = metric.compute()
    assert result.result == 0.0, f"Expected 0.0 for all_or_nothing with one No, got {result.result}"


@pytest.mark.cpu
def test_qa_accuracy_all_or_nothing_all_yes() -> None:
    """all_or_nothing: all Yes → score 1.0."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.score.return_value = [1.0, 1.0]

    metric = QAAccuracyMetric(vlm=mock_vlm, device="cpu", aggregation="all_or_nothing")
    metric.update(
        ["a prompt"],
        [{"questions": {"1": "Is there a cat?", "2": "Is it blue?"}}],
        _dummy_image(batch=1),
    )
    result = metric.compute()
    assert result.result == 1.0, f"Expected 1.0 for all_or_nothing with all Yes, got {result.result}"


@pytest.mark.cpu
def test_vie_score_uses_get_score_from_response() -> None:
    """VieScoreMetric must use get_score_from_response so FloatOutput pydantic objects work."""
    mock_vlm = MagicMock(spec=BaseVLM)
    # LitellmVLM returns model_dump_json() for structured outputs → JSON string
    mock_vlm.generate.return_value = ['{"score": 8.0}']

    metric = VieScoreMetric(vlm=mock_vlm, device="cpu", structured_output=True)
    metric.update(["a cat on a sofa"], _dummy_image(batch=1), _dummy_image(batch=1))
    result = metric.compute()

    # sem=8, qual=8 → sqrt(8 * 8) / 10 = 8/10 = 0.8
    assert abs(result.result - 0.8) < 0.01, f"Expected ~0.8, got {result.result}"


@pytest.mark.cpu
def test_qa_accuracy_all_or_nothing_ambiguous_score() -> None:
    """all_or_nothing: score exactly 0.5 (ambiguous) is treated as No → result 0.0."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.score.return_value = [0.5]

    metric = QAAccuracyMetric(vlm=mock_vlm, device="cpu", aggregation="all_or_nothing")
    metric.update(
        ["a prompt"],
        [{"questions": {"1": "Is there a cat?"}}],
        _dummy_image(batch=1),
    )
    result = metric.compute()
    assert result.result == 0.0, f"Score 0.5 should be treated as No (ambiguous), got {result.result}"


@pytest.mark.cpu
@pytest.mark.slow
def test_yes_no_token_ids_smolvlm_nonempty() -> None:
    """SmolVLM tokenizer must yield non-empty disjoint yes/no prefix ids for VQAScore scoring."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    yes_ids, no_ids = yes_no_first_token_id_groups(tok)
    assert len(yes_ids) > 0, "SmolVLM tokenizer has no 'Yes'-prefix token ids"
    assert len(no_ids) > 0, "SmolVLM tokenizer has no 'No'-prefix token ids"
    assert not (set(yes_ids) & set(no_ids)), "yes_ids and no_ids must be disjoint"


@pytest.mark.cpu
def test_img_edit_score_uses_prompt_from_x() -> None:
    """img_edit_score must score the edited image against the instruction from x, not gt."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.generate.return_value = ['{"score": 9}']

    metric = ImageEditScoreMetric(vlm=mock_vlm, device="cpu")
    pred = _dummy_image(batch=1)
    metric.update(
        ["replace the cat with a dog"],  # x = instruction
        pred,                             # gt = unused for y_x
        pred,                             # outputs = edited image
    )
    result = metric.compute()

    call_args = mock_vlm.generate.call_args
    prompt_sent = call_args[0][1][0]  # second positional arg = prompts list, first item
    assert "replace the cat with a dog" in prompt_sent, (
        f"Instruction not in VLM prompt. Got: {prompt_sent}"
    )
    assert abs(result.result - 0.9) < 0.01, f"Expected ~0.9, got {result.result}"


@pytest.mark.cpu
def test_vie_score_geditbench_gap_documented() -> None:
    """VieScoreMetric does not implement GEditBench 2-criterion SC scoring (known gap).

    This test fails if someone adds task_type support — at that point update GEditBench
    e2e tests and remove this sentinel.
    """
    import inspect

    sig = inspect.signature(VieScoreMetric.__init__)
    assert "task_type" not in sig.parameters, (
        "VieScoreMetric now has task_type — update GEditBench docs and e2e tests, then remove this sentinel."
    )


@pytest.mark.cpu
def test_litellm_logprob_aggregation_sums_all_yes_tokens() -> None:
    """LitellmVLM logprob scoring must sum all yes-prefix token probs, not return the first."""
    pytest.importorskip("litellm")
    import math
    from unittest.mock import MagicMock, patch

    import numpy as np
    from PIL import Image

    from pruna.evaluation.metrics.vlm_base import LitellmVLM

    # Simulate top_logprobs for first output token:
    # "Yes" → logprob=-2.303 (p≈0.10), " yes" → logprob=-2.996 (p≈0.05) → total p_yes≈0.15
    # "No"  → logprob=-1.609 (p≈0.20), " no"  → logprob=-2.303 (p≈0.10) → total p_no≈0.30
    # normalized: p_yes/(p_yes+p_no) ≈ 0.15/0.45 ≈ 0.333
    def make_top_logprob(token, logprob):
        t = MagicMock()
        t.token = token
        t.logprob = logprob
        return t

    first_tok = MagicMock()
    first_tok.top_logprobs = [
        make_top_logprob("Yes", math.log(0.10)),
        make_top_logprob(" yes", math.log(0.05)),
        make_top_logprob("No", math.log(0.20)),
        make_top_logprob(" no", math.log(0.10)),
        make_top_logprob("maybe", math.log(0.55)),
    ]

    mock_logprobs = MagicMock()
    mock_logprobs.content = [first_tok]

    mock_choice = MagicMock()
    mock_choice.logprobs = mock_logprobs
    mock_choice.message.content = "Yes"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("litellm.completion", return_value=mock_response):
        vlm = LitellmVLM(model_name="openai/gpt-4o")
        img = Image.fromarray(np.zeros((32, 32, 3), dtype="uint8"))
        score = vlm._score_with_logprobs(img, "Is there a cat?", "Yes")

    # Should be ~0.333 (p_yes=0.15 / (p_yes+p_no)=0.45), not just 0.10 (first match)
    assert 0.28 < score < 0.40, f"Expected ~0.333 (sum-normalized), got {score}"


@pytest.mark.cpu
@pytest.mark.slow
def test_vqa_probability_score_normalized() -> None:
    """P(Yes) from TransformersVLM.score use_probability=True is in [0, 1]."""
    pytest.importorskip("transformers")
    import numpy as np
    from PIL import Image

    from pruna.evaluation.metrics.vlm_base import TransformersVLM

    vlm = TransformersVLM(
        model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        device="cpu",
        use_outlines=False,
    )
    img = Image.fromarray(np.zeros((32, 32, 3), dtype="uint8"))
    scores = vlm.score([img], ["Is there a cat?"], ["Yes"], use_probability=True)
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0, f"P(Yes) must be in [0, 1], got {scores[0]}"


# ---------------------------------------------------------------------------
# vlm_benchmark_batch_to_json_record serialization tests
# ---------------------------------------------------------------------------


def test_vlm_benchmark_batch_to_json_record_serializes_batch() -> None:
    """Record includes prompts, pred shape, and metric fields."""
    mr = MetricResult(name="qa_accuracy", params={}, result=0.25, higher_is_better=True)
    outcome = BenchmarkVlmBatchOutcome(
        result=mr,
        prompts=["prompt"],
        auxiliaries=[{"path": "/tmp/x.png"}],
        pred=torch.zeros(1, 3, 8, 8),
    )
    rec = vlm_benchmark_batch_to_json_record(
        outcome,
        benchmark_key="GenEval",
        benchmark_name="GenEval",
        metric_name="qa_accuracy",
        vlm_type="transformers",
        model_name="m",
        device="cpu",
    )
    assert rec["inputs"]["prompts"] == ["prompt"]
    assert rec["pred"]["shape"] == [1, 3, 8, 8]
    assert rec["metric_result"]["result"] == 0.25


def test_safe_json_handles_bytes_without_expanding() -> None:
    """Bytes values in aux (e.g. source_image_bytes) are summarized, not expanded to str repr."""
    result = _safe_json({"source_image_bytes": b"\xff\xd8\xff" * 1000, "name": "test"})
    assert result["source_image_bytes"] == {"bytes_len": 3000}
    assert result["name"] == "test"


def test_vlm_benchmark_batch_to_json_record_preserves_null_question_slots() -> None:
    """Padded ``None`` question labels stay JSON null, not the string ``"None"``."""
    mr = MetricResult(name="oneig_alignment", params={}, result=1.0, higher_is_better=True)
    outcome = BenchmarkVlmBatchOutcome(
        result=mr,
        prompts=["p"],
        auxiliaries=[{"questions": {"1": "Are there boys?", "21": None}, "subset": "Anime_Stylization"}],
        pred=torch.zeros(1, 3, 8, 8),
    )
    rec = vlm_benchmark_batch_to_json_record(
        outcome,
        benchmark_key="OneIGAnimeStylization",
        benchmark_name="OneIG Anime Stylization",
        metric_name="oneig_alignment",
        vlm_type="transformers",
        model_name="m",
        device="cpu",
    )
    qs = rec["inputs"]["auxiliary_0"]["questions"]
    assert qs["1"] == "Are there boys?"
    assert qs["21"] is None
