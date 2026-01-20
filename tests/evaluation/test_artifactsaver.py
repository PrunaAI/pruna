import pytest
import tempfile
from pathlib import Path

import numpy as np
import torch

from pruna.evaluation.artifactsavers.video_artifactsaver import VideoArtifactSaver
from pruna.evaluation.artifactsavers.utils import assign_artifact_saver
from pruna.evaluation.metrics.vbench_utils import load_video
from pruna.evaluation.artifactsavers.image_artifactsaver import ImageArtifactSaver
from PIL import Image


def test_create_alias():
    """ Test that we can create an alias for an existing video and image."""
    with tempfile.TemporaryDirectory() as tmp_path:

        # --- Test video artifact saver ---
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

        # --- Test image artifact saver ---
        saver = ImageArtifactSaver(root=tmp_path, export_format="png")
        dummy_image = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        source_filename = saver.save_artifact(dummy_image, saving_kwargs={"quality": 95})
        # Then, we create an alias for the image.
        alias = saver.create_alias(source_filename, "alias_filename")
        # Finally, we reload the alias and check that it is the same as the original image.
        reloaded_alias_image = np.array(Image.open(str(alias)))
        assert(reloaded_alias_image.shape == dummy_image.shape)
        assert alias.exists()
        assert alias.name.endswith(".png")

def test_assign_all_artifact_savers(tmp_path: Path):
    """ Test each artifact saver is assigned correctly."""
    saver = assign_artifact_saver("video", root=tmp_path, export_format="mp4")
    assert isinstance(saver, VideoArtifactSaver)
    assert saver.export_format == "mp4"
    saver = assign_artifact_saver("image", root=tmp_path, export_format="png")
    assert isinstance(saver, ImageArtifactSaver)
    assert saver.export_format == "png"

def test_assign_artifact_saver_invalid():
    """ Test that we raise an error if the artifact saver is assigned incorrectly."""
    with pytest.raises(ValueError):
        assign_artifact_saver("nonexistent_modality")

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

@pytest.mark.parametrize(
    "export_format, save_from_type, save_from_dtype",
    [
    # --- Test png format ---
    # numpy
    pytest.param("png", "np", "uint8"),
    pytest.param("png", "np", "float32"),
    # torch
    pytest.param("png", "pt", "float32"),
    pytest.param("png", "pt", "float16"),
    pytest.param("png", "pt", "uint8"),
    # PIL
    pytest.param("png", "pil", "uint8"),
    # --- Test jpg format ---
    # numpy
    pytest.param("jpg", "np", "uint8"),
    pytest.param("jpg", "np", "float32"),
    # torch
    pytest.param("jpg", "pt", "float32"),
    pytest.param("jpg", "pt", "float16"),
    pytest.param("jpg", "pt", "uint8"),
    # PIL
    pytest.param("jpg", "pil", "uint8"),
    # --- Test webp format ---
    # numpy
    pytest.param("webp", "np", "uint8"),
    pytest.param("webp", "np", "float32"),
    # torch
    pytest.param("webp", "pt", "float32"),
    pytest.param("webp", "pt", "float16"),
    pytest.param("webp", "pt", "uint8"),
    # PIL
    pytest.param("webp", "pil", "uint8"),
    # --- Test jpeg format ---
    # numpy
    pytest.param("jpeg", "np", "uint8"),
    pytest.param("jpeg", "np", "float32"),
    # torch
    pytest.param("jpeg", "pt", "float32"),
    pytest.param("jpeg", "pt", "float16"),
    pytest.param("jpeg", "pt", "uint8"),
    # PIL
    pytest.param("jpeg", "pil", "uint8"),
    ]
    )
def test_image_artifact_saver_tensor(export_format: str, save_from_type: str, save_from_dtype: str):
    """ Test that we can save an image from a tensor."""
    with tempfile.TemporaryDirectory() as tmp_path:
        saver = ImageArtifactSaver(root=tmp_path, export_format=export_format)
        # Create fake image:
        if save_from_type == "pt":
            #  Note: torch convention is (C, H, W)
            if save_from_dtype == "uint8":
                dtype = getattr(torch, save_from_dtype)
                dummy_image = torch.randint(0, 256, (3, 16, 16), dtype=dtype)
            else:
                dtype = getattr(torch, save_from_dtype)
                dummy_image = torch.randn(3, 16, 16, dtype=dtype)
        elif save_from_type == "np":
            # Note: Numpy arrays as images follow the convention (H, W, C)
            if save_from_dtype == "uint8":
                dtype = getattr(np, save_from_dtype)
                dummy_image = np.random.randint(0, 256, (16, 16, 3), dtype=dtype)
            else:
                rng = np.random.default_rng()
                dtype = getattr(np, save_from_dtype)
                dummy_image = rng.random((16, 16, 3), dtype=dtype)
        elif save_from_type == "pil":
            # Note: PIL images by default have shape (H, W, C) and are usually uint8 (standard for ".jpg", etc.)
            dtype = getattr(np, save_from_dtype)
            dummy_image = np.random.randint(0, 256, (16, 16, 3), dtype=dtype)
            dummy_image = Image.fromarray(dummy_image.astype(np.uint8))
        path = saver.save_artifact(dummy_image)
        assert path.exists()
        assert path.suffix == f".{export_format}"