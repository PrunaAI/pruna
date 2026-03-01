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

"""Tests for column_map support in collate functions."""

from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset
from PIL import Image

from pruna.data.collate import (
    audio_collate,
    image_classification_collate,
    image_generation_collate,
    prompt_collate,
    prompt_with_auxiliaries_collate,
    question_answering_collate,
    text_generation_collate,
)
from pruna.data.pruna_datamodule import PrunaDataModule


def _make_pil_image(size: int = 64) -> Image.Image:
    """Create a random RGB PIL image for testing."""
    return Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))


@pytest.fixture()
def bert_tokenizer():
    """Load bert-base-uncased tokenizer; skip test if unavailable."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("bert-base-uncased")
    except OSError:
        pytest.skip("bert-base-uncased tokenizer not available offline")


@pytest.mark.cpu
class TestColumnMapImageGeneration:
    """Test column_map for image_generation_collate."""

    def test_default_columns(self) -> None:
        """Default column names work without column_map."""
        batch = [
            {"image": _make_pil_image(), "text": "a cat"},
            {"image": _make_pil_image(), "text": "a dog"},
        ]
        texts, images = image_generation_collate(batch, img_size=32)
        assert texts == ["a cat", "a dog"]
        assert images.shape == (2, 3, 32, 32)

    def test_remapped_columns(self) -> None:
        """Remapped column names matching the dataset from issue #297."""
        batch = [
            {"chosen": _make_pil_image(), "prompt": "a sunset"},
            {"chosen": _make_pil_image(), "prompt": "a mountain"},
        ]
        texts, images = image_generation_collate(
            batch, img_size=32, column_map={"image": "chosen", "text": "prompt"}
        )
        assert texts == ["a sunset", "a mountain"]
        assert images.shape == (2, 3, 32, 32)

    def test_partial_remap(self) -> None:
        """Only some columns need remapping; others keep their defaults."""
        batch = [
            {"image": _make_pil_image(), "caption": "hello"},
            {"image": _make_pil_image(), "caption": "world"},
        ]
        texts, images = image_generation_collate(
            batch, img_size=32, column_map={"text": "caption"}
        )
        assert texts == ["hello", "world"]
        assert images.shape == (2, 3, 32, 32)


@pytest.mark.cpu
class TestColumnMapPrompt:
    """Test column_map for prompt collate functions."""

    def test_prompt_collate_remapped(self) -> None:
        """Remapped text column for prompt_collate."""
        batch = [{"caption": "hello"}, {"caption": "world"}]
        texts, none = prompt_collate(batch, column_map={"text": "caption"})
        assert texts == ["hello", "world"]
        assert none is None

    def test_prompt_with_auxiliaries_remapped(self) -> None:
        """Remapped text column for prompt_with_auxiliaries_collate."""
        batch = [
            {"caption": "hello", "category": "greeting"},
            {"caption": "world", "category": "noun"},
        ]
        texts, aux = prompt_with_auxiliaries_collate(batch, column_map={"text": "caption"})
        assert texts == ["hello", "world"]
        # auxiliary should contain category but NOT the remapped text column
        assert all("caption" not in d for d in aux)
        assert aux[0]["category"] == "greeting"


@pytest.mark.cpu
class TestColumnMapAudio:
    """Test column_map for audio_collate."""

    def test_audio_collate_remapped(self) -> None:
        """Remapped audio and sentence columns for audio_collate."""
        batch = [
            {"speech": {"path": "/tmp/a.wav"}, "transcription": "hello"},
            {"speech": {"path": "/tmp/b.wav"}, "transcription": "world"},
        ]
        paths, transcriptions = audio_collate(
            batch, column_map={"audio": "speech", "sentence": "transcription"}
        )
        assert paths == ["/tmp/a.wav", "/tmp/b.wav"]
        assert transcriptions == ["hello", "world"]


@pytest.mark.cpu
class TestColumnMapImageClassification:
    """Test column_map for image_classification_collate."""

    def test_remapped_columns(self) -> None:
        """Remapped image and label columns for image_classification_collate."""
        batch = [
            {"photo": _make_pil_image(), "class_id": 0},
            {"photo": _make_pil_image(), "class_id": 1},
        ]
        images, labels = image_classification_collate(
            batch, img_size=32, column_map={"image": "photo", "label": "class_id"}
        )
        assert images.shape == (2, 3, 32, 32)
        assert labels.tolist() == [0, 1]


@pytest.mark.cpu
class TestColumnMapTextGeneration:
    """Test column_map for text_generation_collate."""

    def test_remapped_columns(self, bert_tokenizer) -> None:
        """Remapped text column for text_generation_collate."""
        batch = [{"content": "hello world foo bar"}, {"content": "the quick brown fox"}]
        inputs, targets = text_generation_collate(
            batch, max_seq_len=16, tokenizer=bert_tokenizer, column_map={"text": "content"}
        )
        assert inputs.shape[0] == 2
        assert targets.shape[0] == 2


@pytest.mark.cpu
class TestColumnMapQuestionAnswering:
    """Test column_map for question_answering_collate."""

    def test_remapped_columns(self, bert_tokenizer) -> None:
        """Remapped question and answer columns for question_answering_collate."""
        batch = [
            {"query": "What is pruna?", "response": "A model optimization framework."},
            {"query": "Is it open source?", "response": "Yes it is."},
        ]
        q, a = question_answering_collate(
            batch,
            max_seq_len=16,
            tokenizer=bert_tokenizer,
            column_map={"question": "query", "answer": "response"},
        )
        assert q.shape[0] == 2
        assert a.shape[0] == 2


@pytest.mark.cpu
class TestColumnMapPipeline:
    """Test column_map flowing through the full PrunaDataModule pipeline."""

    def test_from_datasets_with_column_map(self) -> None:
        """column_map passed via collate_fn_args reaches the collate function."""
        images = [_make_pil_image(64) for _ in range(6)]
        prompts = [f"prompt {i}" for i in range(6)]
        ds = Dataset.from_dict({"chosen": images, "prompt": prompts})
        train_ds = ds.select(range(2))
        val_ds = ds.select(range(2, 4))
        test_ds = ds.select(range(4, 6))

        dm = PrunaDataModule.from_datasets(
            (train_ds, val_ds, test_ds),
            collate_fn="image_generation_collate",
            collate_fn_args={
                "img_size": 32,
                "column_map": {"image": "chosen", "text": "prompt"},
            },
        )
        texts, images_tensor = next(iter(dm.train_dataloader(batch_size=2)))
        assert len(texts) == 2
        assert images_tensor.shape == (2, 3, 32, 32)

    def test_from_datasets_without_column_map(self) -> None:
        """Default behavior unchanged when column_map is not provided."""
        images = [_make_pil_image(64) for _ in range(6)]
        captions = [f"caption {i}" for i in range(6)]
        ds = Dataset.from_dict({"image": images, "text": captions})
        train_ds = ds.select(range(2))
        val_ds = ds.select(range(2, 4))
        test_ds = ds.select(range(4, 6))

        dm = PrunaDataModule.from_datasets(
            (train_ds, val_ds, test_ds),
            collate_fn="image_generation_collate",
            collate_fn_args={"img_size": 32},
        )
        texts, images_tensor = next(iter(dm.train_dataloader(batch_size=2)))
        assert len(texts) == 2
        assert images_tensor.shape == (2, 3, 32, 32)
