import pytest
import tempfile
from pathlib import Path

import numpy as np
import torch

from pruna.evaluation.artifactsavers.video_artifactsaver import VideoArtifactSaver
from pruna.evaluation.artifactsavers.utils import assign_artifact_saver
from pruna.evaluation.metrics.vbench_utils import load_video
from PIL import Image


def test_create_alias():
    """ Test that we can create an alias for an existing video."""
    with tempfile.TemporaryDirectory() as tmp_path:
        # First, we create a random video and save it.
        saver = VideoArtifactSaver(root=tmp_path, export_format="mp4")
        dummy_video = np.random.randint(0, 255, (10, 16, 16, 3), dtype=np.uint8)
        source_filename = saver.save_artifact(dummy_video, saving_kwargs={"fps": 5})

        # Then, we create an alias for the video.
        alias = saver.create_alias(source_filename, "alias_filename")
        # Finally, we reload the alias and check that it is the same as the original video.
        reloaded_alias_video = load_video(str(alias), return_type = "np")

        assert(reloaded_alias_video.shape == dummy_video.shape)
        assert alias.exists()
        assert alias.name.endswith(".mp4")


def test_assign_artifact_saver_video(tmp_path: Path):
    """ Test the artifact save is assigned correctly."""
    saver = assign_artifact_saver("video", root=tmp_path, export_format="mp4")
    assert isinstance(saver, VideoArtifactSaver)
    assert saver.export_format == "mp4"


def test_assign_artifact_saver_invalid():
    """ Test that we raise an error if the artifact saver is assigned incorrectly."""
    with pytest.raises(ValueError):
        assign_artifact_saver("text")

@pytest.mark.parametrize(
    "export_format, save_from_type, save_from_dtype",
    [pytest.param("gif", "np", "uint8"),
    pytest.param("gif", "np", "float32"),
    # Numpy doesn't have half precision, so we do not test for float16
    pytest.param("gif", "pt", "float32"),
    pytest.param("gif", "pt", "float16"),
    pytest.param("gif", "pt", "uint8"),
    # PIL doesnot support creating images from float numpy arrays, so we only test uint8.
    pytest.param("gif", "pil", "uint8"),
    pytest.param("mp4", "np", "uint8"),
    pytest.param("mp4", "np", "float32"),
    pytest.param("mp4", "pt", "float32"),
    pytest.param("mp4", "pt", "float16"),
    pytest.param("mp4", "pt", "uint8"),
    # PIL doesnot support creating images from float numpy arrays, so we only test uint8.
    pytest.param("mp4", "pil", "uint8"),]
)
def test_video_artifact_saver_tensor(export_format: str, save_from_type: str, save_from_dtype: str):
    """ Test that we can save a video from numpy, torch and PIL in mp4 and gif formats. """
    with tempfile.TemporaryDirectory() as tmp_path:
        saver = VideoArtifactSaver(root=tmp_path, export_format=export_format)
        # create a fake video:
        if save_from_type == "pt":
            # Unfortunately, neither torch nor numpy have one random generator function that can support all dtypes.
            # Therefore, we need to use different functions for int and float dtypes.
            if save_from_dtype == "uint8":
                dtype = getattr(torch, save_from_dtype)
                dummy_video = torch.randint(0, 256, (2, 3, 16, 16), dtype=dtype)
            else:
                dtype = getattr(torch, save_from_dtype)
                dummy_video = torch.randn(2, 3, 16, 16, dtype=dtype)
        elif save_from_type == "np":
            if save_from_dtype == "uint8":
                dtype = getattr(np, save_from_dtype)
                dummy_video = np.random.randint(0, 256, (2, 16, 16, 3), dtype=dtype)
            else:
                rng = np.random.default_rng()
                dtype = getattr(np, save_from_dtype)
                dummy_video = rng.random((2, 16, 16, 3), dtype=dtype)
        elif save_from_type == "pil":
            dtype = getattr(np, save_from_dtype)
            dummy_video = np.random.randint(0, 256, (2, 16, 16, 3), dtype=dtype)
            dummy_video = [Image.fromarray(frame.astype(np.uint8)) for frame in dummy_video]
        path = saver.save_artifact(dummy_video)
        assert path.exists()
        assert path.suffix == f".{export_format}"
