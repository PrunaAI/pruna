from __future__ import annotations

import pytest
import torch
import numpy as np


from pruna.evaluation.metrics.metric_vbench_background_consistency import VBenchBackgroundConsistency
from pruna.evaluation.metrics.metric_vbench_dynamic_degree import VBenchDynamicDegree


@pytest.mark.cuda
def test_metric_background_consistency():
    """Test metric background consistency."""
    # let's create a batch of 2 random RGB videos with 2 frames each that's 16 x 16 pixels.
    random_input_video_batch = torch.randn(2, 2, 3, 16,16)
    # let's create a batch of 2 all black RGB videos with 2 frames each that's 16 x 16 pixels.
    all_black_input_video_batch = torch.zeros(2, 2, 3, 16,16)
    metric = VBenchBackgroundConsistency()
    metric.update(random_input_video_batch, random_input_video_batch, random_input_video_batch)
    random_result = metric.compute()
    metric.reset()
    metric.update(all_black_input_video_batch, all_black_input_video_batch, all_black_input_video_batch)
    all_black_result = metric.compute()
    metric.reset()
    #  Background consistency checks for the cosine similarity between frames.
    #  Therefore we would expect a completely blacked out video (even though meaningless) to have a much higher score
    #  than a completely random set of frames.
    assert all_black_result.result >= random_result.result
    # Since the all black video is completely black, the cosine similarity between frames should be 1.0
    assert np.isclose(all_black_result.result, 1.0)

@pytest.mark.cuda
@pytest.mark.parametrize("model_fixture", ["wan_tiny_random"], indirect=["model_fixture"])
def test_metric_dynamic_degree_dynamic(model_fixture):
    """Test metric dynamic degree with an example sample from the vbench dataset that returns a dynamic video."""
    model, smash_config = model_fixture
    model.to("cuda")
    model.to(torch.float32)
    # this is a prompt from the vbench dataset under dynamic degree dimension.
    output_video = model("a dog running happily", num_inference_steps=10, output_type="pt").frames[0].unsqueeze(0)

    metric = VBenchDynamicDegree(interval=4)
    metric.update(output_video, output_video, output_video)
    result = metric.compute()
    # a video of a dog running ideally should have a dynamic degree of 1.0 (since it contains large movements)
    assert result.result == 1.0

@pytest.mark.cuda
def test_metric_dynamic_degree_static():
    """Test metric dynamic degree fail case."""
    # Testing for a lack of movement is much easier than testing for movement.
    # We create a completely black video to test the metric.
    video = torch.zeros(1,4,3,64,64) # a completely black video doesn't contain any movements
    metric = VBenchDynamicDegree(interval=1)
    metric.update(video, video, video)
    result = metric.compute()
    assert result.result == 0.0
