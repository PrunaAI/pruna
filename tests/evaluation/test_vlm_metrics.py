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
from pruna.evaluation.metrics.vlm_base import BaseVLM, get_vlm

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
def test_qa_accuracy_aggregation_modes() -> None:
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.score.return_value = [1.0, 0.0]
    images = _dummy_image(batch=1)
    aux = [{"questions": {"1": "Q1", "2": "Q2"}}]

    mean_metric = QAAccuracyMetric(vlm=mock_vlm, vlm_type="litellm", device="cpu", aggregation="mean")
    mean_metric.update(["a prompt"], aux, images)
    assert mean_metric.compute().result == pytest.approx(0.5)

    strict_metric = QAAccuracyMetric(vlm=mock_vlm, vlm_type="litellm", device="cpu", aggregation="all_or_nothing")
    strict_metric.update(["a prompt"], aux, images)
    assert strict_metric.compute().result == pytest.approx(0.0)


@pytest.mark.cpu
def test_get_vlm_returns_custom() -> None:
    custom = MagicMock(spec=BaseVLM)
    out = get_vlm(vlm=custom, vlm_type="litellm", model_name="gpt-4o")
    assert out is custom


@pytest.mark.cpu
def test_get_vlm_requires_model_name_without_vlm() -> None:
    with pytest.raises(ValueError, match="model_name"):
        get_vlm(vlm=None, vlm_type="litellm")


@pytest.mark.cpu
@pytest.mark.parametrize(
    "metric_cls, expected_name, expected_result",
    [
        (TextScoreMetric, "text_score", 0.0),
        (OneIGTextScoreMetric, "oneig_text_score", 1.0),
    ],
)
def test_text_metrics_list_str_gt(
    metric_cls: type, expected_name: str, expected_result: float
) -> None:
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
def test_text_score_uses_normalized_edit_distance() -> None:
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.generate.side_effect = [["abxde"], ["ax"]]
    metric = TextScoreMetric(vlm=mock_vlm, vlm_type="litellm", device="cpu")

    metric.update(["p1"], ["abcde"], _dummy_image(batch=1))
    metric.update(["p2"], ["ab"], _dummy_image(batch=1))

    assert metric.scores == pytest.approx([0.2, 0.5])
    assert metric.compute().result == pytest.approx(0.35)


@pytest.mark.cpu
def test_text_score_registry_aliases() -> None:
    from pruna.evaluation.metrics.registry import MetricRegistry

    lev = MetricRegistry.get_metric("ocr_levenshtein", device="cpu", model_name="openai/gpt-4o")
    comp = MetricRegistry.get_metric("ocr_text_score", device="cpu", model_name="openai/gpt-4o")
    assert type(lev).__name__ == "TextScoreMetric"
    assert type(comp).__name__ == "OneIGTextScoreMetric"
    assert lev.metric_name == "text_score"
    assert comp.metric_name == "oneig_text_score"


@pytest.mark.cpu
def test_oneig_text_score_utils_golden_composite() -> None:
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
