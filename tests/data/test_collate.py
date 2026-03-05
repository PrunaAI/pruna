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


def _img(size: int = 64) -> Image.Image:
    return Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))


@pytest.fixture()
def bert_tokenizer():
    """Return a bert-base-uncased tokenizer, skip if unavailable."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("bert-base-uncased")
    except OSError:
        pytest.skip("bert-base-uncased tokenizer not available offline")


# ── image_generation_collate ─────────────────────────────────────


@pytest.mark.cpu
@pytest.mark.parametrize(
    "column_map, img_key, txt_key",
    [
        (None, "image", "text"),
        ({"image": "chosen", "text": "prompt"}, "chosen", "prompt"),
        ({"text": "caption"}, "image", "caption"),
    ],
    ids=["defaults", "full-remap", "partial-remap"],
)
def test_image_generation_collate(column_map, img_key, txt_key):
    """Test image_generation_collate with default, full, and partial column_map."""
    batch = [{img_key: _img(), txt_key: "hello"}, {img_key: _img(), txt_key: "world"}]
    texts, images = image_generation_collate(batch, img_size=32, column_map=column_map)
    assert texts == ["hello", "world"]
    assert images.shape == (2, 3, 32, 32)


# ── prompt collate functions ────────────────────────────────────


@pytest.mark.cpu
def test_prompt_collate_remapped():
    """Test prompt_collate with remapped text column."""
    batch = [{"caption": "hello"}, {"caption": "world"}]
    texts, none = prompt_collate(batch, column_map={"text": "caption"})
    assert texts == ["hello", "world"]
    assert none is None


@pytest.mark.cpu
def test_prompt_with_auxiliaries_remapped():
    """Test prompt_with_auxiliaries_collate with remapped text column."""
    batch = [
        {"caption": "hello", "category": "greeting"},
        {"caption": "world", "category": "noun"},
    ]
    texts, aux = prompt_with_auxiliaries_collate(batch, column_map={"text": "caption"})
    assert texts == ["hello", "world"]
    assert all("caption" not in d for d in aux)
    assert aux[0]["category"] == "greeting"


# ── audio_collate ────────────────────────────────────────────────


@pytest.mark.cpu
def test_audio_collate_remapped():
    """Test audio_collate with remapped audio and sentence columns."""
    batch = [
        {"speech": {"path": "/tmp/a.wav"}, "transcription": "hello"},
        {"speech": {"path": "/tmp/b.wav"}, "transcription": "world"},
    ]
    paths, transcriptions = audio_collate(
        batch, column_map={"audio": "speech", "sentence": "transcription"}
    )
    assert paths == ["/tmp/a.wav", "/tmp/b.wav"]
    assert transcriptions == ["hello", "world"]


# ── image_classification_collate ─────────────────────────────────


@pytest.mark.cpu
def test_image_classification_collate_remapped():
    """Test image_classification_collate with remapped image and label columns."""
    batch = [{"photo": _img(), "class_id": 0}, {"photo": _img(), "class_id": 1}]
    images, labels = image_classification_collate(
        batch, img_size=32, column_map={"image": "photo", "label": "class_id"}
    )
    assert images.shape == (2, 3, 32, 32)
    assert labels.tolist() == [0, 1]


# ── text_generation_collate ──────────────────────────────────────


@pytest.mark.cpu
def test_text_generation_collate_remapped(bert_tokenizer):
    """Test text_generation_collate with remapped text column."""
    batch = [{"content": "hello world foo bar"}, {"content": "the quick brown fox"}]
    inputs, targets = text_generation_collate(
        batch, max_seq_len=16, tokenizer=bert_tokenizer, column_map={"text": "content"}
    )
    assert inputs.shape[0] == 2
    assert targets.shape[0] == 2


# ── question_answering_collate ───────────────────────────────────


@pytest.mark.cpu
def test_question_answering_collate_remapped(bert_tokenizer):
    """Test question_answering_collate with remapped question and answer columns."""
    batch = [
        {"query": "What is pruna?", "response": "A model optimization framework."},
        {"query": "Is it open source?", "response": "Yes it is."},
    ]
    q, a = question_answering_collate(
        batch, max_seq_len=16, tokenizer=bert_tokenizer,
        column_map={"question": "query", "answer": "response"},
    )
    assert q.shape[0] == 2
    assert a.shape[0] == 2


# ── PrunaDataModule pipeline ────────────────────────────────────


@pytest.mark.cpu
@pytest.mark.parametrize(
    "column_map, img_key, txt_key",
    [
        ({"image": "chosen", "text": "prompt"}, "chosen", "prompt"),
        (None, "image", "text"),
    ],
    ids=["with-column-map", "without-column-map"],
)
def test_datamodule_pipeline(column_map, img_key, txt_key):
    """Test end-to-end PrunaDataModule pipeline with and without column_map."""
    ds = Dataset.from_dict({img_key: [_img(64) for _ in range(6)], txt_key: [f"p{i}" for i in range(6)]})
    collate_fn_args = {"img_size": 32}
    if column_map:
        collate_fn_args["column_map"] = column_map

    dm = PrunaDataModule.from_datasets(
        (ds.select(range(2)), ds.select(range(2, 4)), ds.select(range(4, 6))),
        collate_fn="image_generation_collate",
        collate_fn_args=collate_fn_args,
    )
    texts, images_tensor = next(iter(dm.train_dataloader(batch_size=2)))
    assert len(texts) == 2
    assert images_tensor.shape == (2, 3, 32, 32)
