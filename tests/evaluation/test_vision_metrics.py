"""Vision metric tests split by dedicated metric PR branches."""

from unittest.mock import MagicMock

import pytest
import torch

from pruna.evaluation.metrics.metric_vie_score import VieScoreMetric
from pruna.evaluation.metrics.metric_vqa import VQAMetric
from pruna.evaluation.metrics.vlm_base import BaseVLM


def _dummy_image(batch: int = 1, size: int = 64) -> torch.Tensor:
    return torch.rand(batch, 3, size, size)


@pytest.mark.cpu
def test_vqa_uses_prompt_question_and_scores_yes_probability() -> None:
    """VQA asks prompt-grounded yes/no question and stores returned score."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.score.return_value = [0.7]

    metric = VQAMetric(vlm=mock_vlm, vlm_type="litellm", device="cpu", use_probability=True)
    images = _dummy_image()
    metric.update(["a cat"], images, images)

    result = metric.compute()
    assert result.name == "vqa"
    assert result.result == 0.7
    call = mock_vlm.score.call_args
    assert call[0][1] == ['Does this image show "a cat"?']


@pytest.mark.cpu
def test_vie_score_uses_json_score_lists() -> None:
    """VieScoreMetric parses JSON score lists and returns normalized value."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.generate.return_value = ['{"score": [8.0, 8.0], "reasoning": ""}']

    metric = VieScoreMetric(vlm=mock_vlm, device="cpu", structured_output=True)
    metric.update(["a cat on a sofa"], _dummy_image(), _dummy_image())
    result = metric.compute()

    assert abs(result.result - 0.8) < 0.01
