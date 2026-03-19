"""Tests for VLM metrics (VQA, AlignmentScore, ImageEditScore, QAAccuracy, TextScore, VieScore)."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from pruna.evaluation.metrics.metric_alignment_score import AlignmentScoreMetric
from pruna.evaluation.metrics.metric_img_edit_score import ImageEditScoreMetric
from pruna.evaluation.metrics.metric_oneig_alignment import OneIGAlignmentMetric
from pruna.evaluation.metrics.metric_qa_accuracy import QAAccuracyMetric
from pruna.evaluation.metrics.metric_text_score import OneIGTextScoreMetric, TextScoreMetric
from pruna.evaluation.metrics.metric_viescore import VieScoreMetric
from pruna.evaluation.metrics.metric_vqa import VQAMetric
from pruna.evaluation.metrics.vlm_base import BaseVLM, get_vlm

SMOL_VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


def _dummy_image(batch: int = 1, size: int = 224) -> torch.Tensor:
    return torch.rand(batch, 3, size, size)


def _update_metric(metric: object, prompts: list, images: torch.Tensor) -> None:
    """Update metric with appropriate gt type per metric contract."""
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
@pytest.mark.parametrize(
    "metric_cls",
    [
        VQAMetric,
        AlignmentScoreMetric,
        ImageEditScoreMetric,
        QAAccuracyMetric,
        OneIGAlignmentMetric,
        TextScoreMetric,
        OneIGTextScoreMetric,
        VieScoreMetric,
    ],
)
@pytest.mark.parametrize("structured_output", [False, True])
def test_vlm_metrics_transformers_smolvlm(metric_cls: type, structured_output: bool) -> None:
    """Test each VLM metric with local SmolVLM-256M-Instruct."""
    metric = metric_cls(
        vlm_type="transformers",
        model_name=SMOL_VLM,
        device="cpu",
        structured_output=structured_output,
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
@pytest.mark.parametrize(
    "metric_cls",
    [
        VQAMetric,
        AlignmentScoreMetric,
        ImageEditScoreMetric,
        QAAccuracyMetric,
        OneIGAlignmentMetric,
        TextScoreMetric,
        OneIGTextScoreMetric,
        VieScoreMetric,
    ],
)
@pytest.mark.parametrize("structured_output", [False, True])
def test_vlm_metrics_litellm_mocked(metric_cls: type, structured_output: bool) -> None:
    """Test each VLM metric with mocked litellm API (requires litellm installed)."""
    pytest.importorskip("litellm")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    if metric_cls in (AlignmentScoreMetric, VQAMetric, QAAccuracyMetric, OneIGAlignmentMetric):
        mock_response.choices[0].message.content = (
            '{"answer": "Yes"}' if structured_output else "Yes"
        )
    else:
        mock_response.choices[0].message.content = (
            '{"score": 8}' if structured_output else "8"
        )

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = mock_response

        metric = metric_cls(
            vlm_type="litellm",
            model_name="gpt-4o",
            device="cpu",
            structured_output=structured_output,
        )
        images = _dummy_image(batch=1)
        prompts = ["a cat"]
        _update_metric(metric, prompts, images)
        result = metric.compute()

    assert result.name == metric.metric_name
    assert isinstance(result.result, float)
    assert mock_completion.called


@pytest.mark.cpu
@pytest.mark.parametrize(
    "metric_cls",
    [
        VQAMetric,
        AlignmentScoreMetric,
        ImageEditScoreMetric,
        QAAccuracyMetric,
        OneIGAlignmentMetric,
        TextScoreMetric,
        OneIGTextScoreMetric,
        VieScoreMetric,
    ],
)
@pytest.mark.parametrize("structured_output", [False, True])
def test_vlm_metrics_empty_score(metric_cls: type, structured_output: bool) -> None:
    """Test that empty compute returns 0.0."""
    metric = metric_cls(
        vlm_type="transformers",
        model_name=SMOL_VLM,
        device="cpu",
        structured_output=structured_output,
    )
    result = metric.compute()
    assert result.result == 0.0


@pytest.mark.cpu
@pytest.mark.parametrize("structured_output", [False, True])
def test_vlm_metrics_custom_vlm(structured_output: bool) -> None:
    """Test metrics with a custom VLM instance."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.generate.return_value = ["Yes"]
    mock_vlm.score.return_value = [1.0]

    metric = VQAMetric(
        vlm=mock_vlm, vlm_type="litellm", device="cpu", structured_output=structured_output
    )
    images = _dummy_image(batch=1)
    prompts = ["a cat"]
    metric.update(prompts, images, images)
    result = metric.compute()

    assert result.result == 1.0
    mock_vlm.score.assert_called()


@pytest.mark.cpu
def test_get_vlm_returns_custom() -> None:
    """Test get_vlm returns provided vlm as-is."""
    custom = MagicMock(spec=BaseVLM)
    out = get_vlm(vlm=custom, vlm_type="litellm", model_name="gpt-4o")
    assert out is custom


@pytest.mark.cpu
def test_text_score_with_list_str_gt() -> None:
    """Test TextScoreMetric accepts List[str] ground truth from text_score_collate."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.generate.return_value = ["hello world"]

    metric = TextScoreMetric(vlm=mock_vlm, vlm_type="litellm", device="cpu")
    images = _dummy_image(batch=1)
    metric.update(["a prompt"], ["hello world"], images)
    result = metric.compute()

    assert result.result == 0.0
    mock_vlm.generate.assert_called_once()


@pytest.mark.cpu
def test_oneig_text_score_with_list_str_gt() -> None:
    """OneIG composite is 1.0 when OCR exactly matches ground truth after preprocess."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.generate.return_value = ["hello world"]

    metric = OneIGTextScoreMetric(vlm=mock_vlm, vlm_type="litellm", device="cpu")
    images = _dummy_image(batch=1)
    metric.update(["a prompt"], ["hello world"], images)
    result = metric.compute()

    assert result.result == 1.0
    assert result.name == "oneig_text_score"
    mock_vlm.generate.assert_called_once()


@pytest.mark.cpu
def test_text_score_registry_aliases() -> None:
    """Descriptive OCR metric names are aliases for the same classes."""
    from pruna.evaluation.metrics.registry import MetricRegistry

    lev = MetricRegistry.get_metric("ocr_levenshtein", device="cpu")
    comp = MetricRegistry.get_metric("ocr_text_score", device="cpu")
    assert type(lev).__name__ == "TextScoreMetric"
    assert type(comp).__name__ == "OneIGTextScoreMetric"
    assert lev.metric_name == "text_score"
    assert comp.metric_name == "oneig_text_score"


@pytest.mark.cpu
def test_oneig_text_score_utils_golden_composite() -> None:
    """Reference composite matches OneIG ``text_score`` formula (EN cap)."""
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
@pytest.mark.integration
@pytest.mark.skip(reason="Requires OPENAI_API_KEY; run manually with: pytest -m integration")
@pytest.mark.parametrize("structured_output", [False, True])
def test_vlm_metrics_litellm_api(structured_output: bool) -> None:
    """Integration test with real litellm API (requires OPENAI_API_KEY)."""
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    metric = VQAMetric(
        vlm_type="litellm",
        model_name="gpt-4o",
        device="cpu",
        structured_output=structured_output,
    )
    images = _dummy_image(batch=1)
    prompts = ["a cat"]
    metric.update(prompts, images, images)
    result = metric.compute()
    assert 0.0 <= result.result <= 1.0
