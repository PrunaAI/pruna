"""Vision metric tests split by dedicated metric PR branches."""

from unittest.mock import MagicMock

import pytest
import torch

from pruna.evaluation.metrics.metric_vqa import VQAMetric
from pruna.evaluation.metrics.vlm_base import BaseVLM


@pytest.mark.cpu
def test_vqa_uses_prompt_question_and_scores_yes_probability() -> None:
    """VQA asks prompt-grounded yes/no question and stores returned score."""
    mock_vlm = MagicMock(spec=BaseVLM)
    mock_vlm.score.return_value = [0.7]

    metric = VQAMetric(vlm=mock_vlm, vlm_type="litellm", device="cpu", use_probability=True)
    images = torch.rand(1, 3, 64, 64)
    metric.update(["a cat"], images, images)

    result = metric.compute()
    assert result.name == "vqa"
    assert result.result == 0.7
    call = mock_vlm.score.call_args
    assert call[0][1] == ['Does this image show "a cat"?']
