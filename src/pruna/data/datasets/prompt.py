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


ImgEditCategory = Literal["replace", "add", "remove", "adjust", "extract", "style", "background", "compose"]


def setup_imgedit_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: ImgEditCategory | list[ImgEditCategory] | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the ImgEdit benchmark dataset for image editing evaluation.

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
    category : ImgEditCategory | list[ImgEditCategory] | None
        Filter by edit type. Available: replace, add, remove, adjust, extract, style,
        background, compose. If None, returns all categories.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The ImgEdit dataset (dummy train, dummy val, test).
    """
    import json

    import requests

    instructions_url = "https://raw.githubusercontent.com/PKU-YuanGroup/ImgEdit/b3eb8e74d7cd1fd0ce5341eaf9254744a8ab4c0b/Benchmark/Basic/basic_edit.json"
    judge_prompts_url = "https://raw.githubusercontent.com/PKU-YuanGroup/ImgEdit/c14480ac5e7b622e08cd8c46f96624a48eb9ab46/Benchmark/Basic/prompts.json"

    instructions = json.loads(requests.get(instructions_url).text)
    judge_prompts = json.loads(requests.get(judge_prompts_url).text)

    categories = [category] if category is not None and not isinstance(category, list) else category
    records = []
    for _, instruction in instructions.items():
        edit_type = instruction.get("edit_type", "")

        if categories is not None and edit_type not in categories:
            continue

        records.append(
            {
                "text": instruction.get("prompt", ""),
                "category": edit_type,
                "image_id": instruction.get("id", ""),
                "judge_prompt": judge_prompts.get(edit_type, ""),
            }
        )

    ds = Dataset.from_list(records)
    test_sample_size = define_sample_size_for_dataset(ds, fraction, test_sample_size)
    ds = ds.select(range(min(test_sample_size, len(ds))))

    if len(ds) == 0:
        raise ValueError(f"No samples found for category '{category}'.")

    return _prepare_test_only_prompt_dataset(ds, seed, "ImgEdit")
