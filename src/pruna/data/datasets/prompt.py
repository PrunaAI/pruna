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

from typing import Literal, Tuple

from datasets import Dataset, load_dataset

from pruna.data.utils import _prepare_test_only_prompt_dataset, define_sample_size_for_dataset
from pruna.logging.logger import pruna_logger

GenEvalCategory = Literal["single_object", "two_object", "counting", "colors", "position", "color_attr"]

PartiCategory = Literal[
    "Abstract",
    "Animals",
    "Artifacts",
    "Arts",
    "Food & Beverage",
    "Illustrations",
    "Indoor Scenes",
    "Outdoor Scenes",
    "People",
    "Produce & Plants",
    "Vehicles",
    "World Knowledge",
    "Basic",
    "Complex",
    "Fine-grained Detail",
    "Imagination",
    "Linguistic Structures",
    "Perspective",
    "Properties & Positioning",
    "Quantity",
    "Simple Detail",
    "Style & Format",
    "Writing & Symbols",
]


def setup_drawbench_dataset(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the DrawBench dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The DrawBench dataset.
    """
    ds = load_dataset("sayakpaul/drawbench", trust_remote_code=True)["train"]  # type: ignore[index]
    ds = ds.rename_column("Prompts", "text")
    pruna_logger.info("DrawBench is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


def setup_parti_prompts_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: PartiCategory | list[PartiCategory] | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the Parti Prompts dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.
    fraction : float
        The fraction of the dataset to use.
    train_sample_size : int | None
        The sample size to use for the train dataset (unused; train/val are dummy).
    test_sample_size : int | None
        The sample size to use for the test dataset.
    category : PartiCategory | list[PartiCategory] | None
        Filter by Category or Challenge.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The Parti Prompts dataset (dummy train, dummy val, test).
    """
    ds = load_dataset("nateraw/parti-prompts")["train"]  # type: ignore[index]

    if category is not None:
        categories = [category] if not isinstance(category, list) else category
        ds = ds.filter(lambda x: x["Category"] in categories or x["Challenge"] in categories)

    test_sample_size = define_sample_size_for_dataset(ds, fraction, test_sample_size)
    ds = ds.select(range(min(test_sample_size, len(ds))))
    ds = ds.rename_column("Prompt", "text")
    return _prepare_test_only_prompt_dataset(ds, seed, "PartiPrompts")


def _generate_geneval_question(entry: dict) -> list[str]:
    """Generate evaluation questions from GenEval metadata."""
    tag = entry.get("tag", "")
    include = entry.get("include", [])
    questions = []

    for obj in include:
        cls = obj.get("class", "")
        if "color" in obj:
            questions.append(f"Does the image contain a {obj['color']} {cls}?")
        elif "count" in obj:
            questions.append(f"Does the image contain exactly {obj['count']} {cls}(s)?")
        else:
            questions.append(f"Does the image contain a {cls}?")

    if tag == "position" and len(include) >= 2:
        a_cls = include[0].get("class", "")
        b_cls = include[1].get("class", "")
        pos = include[1].get("position")
        if pos and pos[0]:
            questions.append(f"Is the {b_cls} {pos[0]} the {a_cls}?")

    return questions


def setup_geneval_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: GenEvalCategory | list[GenEvalCategory] | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the GenEval benchmark dataset.

    License: MIT

    Parameters
    ----------
    seed : int
        The seed to use.
    fraction : float
        The fraction of the dataset to use.
    train_sample_size : int | None
        Unused; train/val are dummy.
    test_sample_size : int | None
        The sample size to use for the test dataset.
    category : GenEvalCategory | list[GenEvalCategory] | None
        Filter by category. Available: single_object, two_object, counting, colors, position, color_attr.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The GenEval dataset (dummy train, dummy val, test).
    """
    import json

    import requests

    url = "https://raw.githubusercontent.com/djghosh13/geneval/d927da8e42fde2b1b5cd743da4df5ff83c1654ff/prompts/evaluation_metadata.jsonl"
    response = requests.get(url)
    data = [json.loads(line) for line in response.text.splitlines()]

    if category is not None:
        categories = [category] if not isinstance(category, list) else category
        data = [entry for entry in data if entry.get("tag") in categories]

    records = []
    for entry in data:
        questions = _generate_geneval_question(entry)
        records.append(
            {
                "text": entry["prompt"],
                "tag": entry.get("tag", ""),
                "questions": questions,
                "include": entry.get("include", []),
            }
        )

    ds = Dataset.from_list(records)
    test_sample_size = define_sample_size_for_dataset(ds, fraction, test_sample_size)
    ds = ds.select(range(min(test_sample_size, len(ds))))
    return _prepare_test_only_prompt_dataset(ds, seed, "GenEval")


def setup_long_text_bench_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the Long Text Bench dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.
    fraction : float
        The fraction of the dataset to use.
    train_sample_size : int | None
        Unused; train/val are dummy.
    test_sample_size : int | None
        The sample size to use for the test dataset.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The Long Text Bench dataset (dummy train, dummy val, test).
    """
    ds = load_dataset("X-Omni/LongText-Bench")["train"]  # type: ignore[index]
    ds = ds.rename_column("text", "text_content")
    ds = ds.rename_column("prompt", "text")
    test_sample_size = define_sample_size_for_dataset(ds, fraction, test_sample_size)
    ds = ds.select(range(min(test_sample_size, len(ds))))
    return _prepare_test_only_prompt_dataset(ds, seed, "LongTextBench")


def setup_genai_bench_dataset(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the GenAI Bench dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The GenAI Bench dataset.
    """
    ds = load_dataset("BaiqiL/GenAI-Bench")["train"]  # type: ignore[index]
    ds = ds.rename_column("Prompt", "text")
    pruna_logger.info("GenAI-Bench is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds
