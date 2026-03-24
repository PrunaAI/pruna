"""Tests for VLM metrics (VQA, AlignmentScore, ImageEditScore, QAAccuracy, TextScore, VieScore)."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import BaseModel

from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.evaluation.metrics.metric_alignment_score import AlignmentScoreMetric
from pruna.evaluation.metrics.metric_vlm_utils import VQAnswer, get_answer_from_response
from pruna.evaluation.metrics.vlm_base import BaseVLM, get_vlm
from pruna.evaluation.task import Task
from pruna.evaluation.metrics.metric_img_edit_score import ImageEditScoreMetric
from pruna.evaluation.metrics.metric_qa_accuracy import QAAccuracyMetric
from pruna.evaluation.metrics.metric_text_score import TextScoreMetric
from pruna.evaluation.metrics.metric_viescore import VieScoreMetric
from pruna.evaluation.metrics.metric_vqa import VQAMetric
from pruna.evaluation.metrics.vlm_base import TransformersVLM
from pruna.data.pruna_datamodule import PrunaDataModule

SMOL_VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


def _dummy_image(batch: int = 1, size: int = 224) -> torch.Tensor:
    return torch.rand(batch, 3, size, size)


def _update_metric(metric: object, prompts: list, images: torch.Tensor) -> None:
    """Update metric with appropriate gt type per metric contract."""
    if isinstance(metric, QAAccuracyMetric):
        metric.update(prompts, [["Is there a cat?"]], images)
    elif isinstance(metric, TextScoreMetric):
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
        TextScoreMetric,
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
        TextScoreMetric,
        VieScoreMetric,
    ],
)
@pytest.mark.parametrize("structured_output", [False, True])
def test_vlm_metrics_litellm_mocked(metric_cls: type, structured_output: bool) -> None:
    """Test each VLM metric with mocked litellm API (requires litellm installed)."""
    pytest.importorskip("litellm")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    if metric_cls in (AlignmentScoreMetric, VQAMetric, QAAccuracyMetric):
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
        TextScoreMetric,
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
def test_transformers_generate_routes_pydantic_response_format_to_outlines() -> None:
    """Structured Pydantic responses should use the outlines path for transformers backends."""
    vlm = TransformersVLM(model_name=SMOL_VLM, device="cpu", use_outlines=True)

    with (
        patch.object(vlm, "_load_model") as mock_load_model,
        patch.object(vlm, "_generate_with_outlines", return_value=['{"answer":"Yes"}']) as mock_outlines,
        patch.object(vlm, "_generate_standard", return_value=["fallback"]) as mock_standard,
    ):
        result = vlm.generate([MagicMock()], ["question"], response_format=VQAnswer)

    mock_load_model.assert_called_once()
    mock_outlines.assert_called_once()
    mock_standard.assert_not_called()
    assert result == ['{"answer":"Yes"}']


@pytest.mark.cpu
def test_transformers_outlines_result_serialization() -> None:
    """Outlines outputs should be normalized into strings parseable by existing helpers."""

    class DummySchema(BaseModel):
        answer: str

    schema_result = TransformersVLM._serialize_outlines_result(DummySchema(answer="Yes"))
    dict_result = TransformersVLM._serialize_outlines_result({"answer": "No"})

    assert get_answer_from_response(schema_result) == "Yes"
    assert get_answer_from_response(dict_result) == "No"


@pytest.mark.cpu
def test_evaluation_agent_update_stateful_metrics_with_stub_vlm() -> None:
    """Smoke-test the real agent stateful update path with a stub VLM-backed metric."""
    stub_vlm = MagicMock(spec=BaseVLM)
    stub_vlm.score.return_value = [1.0]
    metric = VQAMetric(vlm=stub_vlm, vlm_type="litellm", device="cpu")
    task = Task(request=[metric], datamodule=PrunaDataModule.from_string("LAION256"), device="cpu")
    agent = EvaluationAgent(task=task)
    agent.task.dataloader = [(["a cat"], torch.empty(0))]
    agent.device = "cpu"
    agent.device_map = None

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def run_inference(self, batch):
            return _dummy_image(batch=1)

    agent.update_stateful_metrics(FakeModel(), agent.task.get_single_stateful_metrics(), [])
    results = agent.compute_stateful_metrics(agent.task.get_single_stateful_metrics(), [])

    assert len(results) == 1
    assert results[0].name == "vqa"
    assert results[0].result == 1.0
    stub_vlm.score.assert_called_once()


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
