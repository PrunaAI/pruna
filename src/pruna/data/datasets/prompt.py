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


def setup_oneig_text_rendering_dataset(
    seed: int,
    num_samples: int | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the OneIG Text Rendering benchmark dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.
    num_samples : int | None
        Maximum number of samples to return. If None, returns all samples.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The OneIG Text Rendering dataset (dummy train, dummy val, test).
    """
    ds = load_dataset("OneIG-Bench/OneIG-Bench", "OneIG-Bench")["train"]  # type: ignore[index]
    ds = ds.filter(lambda x: x.get("category", "") in ("Text_Rendering", "Text Rendering"))

    def to_record(row: dict) -> dict:
        prompt = row.get("prompt_en", row.get("prompt", ""))
        return {
            "text": prompt,
            "text_content": row.get("class", ""),
        }

    records = [to_record(dict(row)) for row in ds]
    ds = Dataset.from_list(records).shuffle(seed=seed)

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    pruna_logger.info("OneIG Text Rendering is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


ONEIG_ALIGNMENT_CATEGORIES = ["Anime_Stylization", "Portrait", "General_Object"]


def setup_oneig_alignment_dataset(
    seed: int,
    category: str | None = None,
    num_samples: int | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the OneIG Alignment benchmark dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.
    category : str | None
        Filter by category. Available: Anime_Stylization, Portrait, General_Object.
    num_samples : int | None
        Maximum number of samples to return. If None, returns all samples.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The OneIG Alignment dataset (dummy train, dummy val, test).
    """
    import json

    import requests

    ds = load_dataset("OneIG-Bench/OneIG-Bench", "OneIG-Bench")["train"]  # type: ignore[index]

    questions_by_id: dict[str, dict] = {}
    url = "https://raw.githubusercontent.com/OneIG-Bench/OneIG-Benchmark/main/benchmark/alignment_questions.json"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = json.loads(response.text)
            items = data if isinstance(data, list) else [data]
            questions_by_id = {q["id"]: q for q in items if isinstance(q, dict) and "id" in q}
        except Exception:
            pass

    records = []
    for row in ds:
        row_id = row.get("id", "")
        row_category = row.get("category", "")

        if category is not None:
            if category not in ONEIG_ALIGNMENT_CATEGORIES:
                raise ValueError(f"Invalid category: {category}. Must be one of {ONEIG_ALIGNMENT_CATEGORIES}")
            if row_category != category:
                continue

        q_info = questions_by_id.get(row_id, {})
        records.append(
            {
                "text": row.get("prompt_en", row.get("prompt", "")),
                "category": row_category,
                "questions": q_info.get("questions", []),
                "dependencies": q_info.get("dependencies", []),
            }
        )

    ds = Dataset.from_list(records)
    ds = ds.shuffle(seed=seed)

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    pruna_logger.info("OneIG Alignment is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


DPGCategory = Literal["entity", "attribute", "relation", "global", "other"]


def setup_dpg_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: DPGCategory | list[DPGCategory] | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the DPG (Descriptive Prompt Generation) benchmark dataset.

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
    category : DPGCategory | list[DPGCategory] | None
        Filter by category. Available: entity, attribute, relation, global, other.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The DPG dataset (dummy train, dummy val, test).
    """
    import csv
    import io
    from collections import defaultdict

    import requests

    url = "https://raw.githubusercontent.com/TencentQQGYLab/ELLA/main/dpg_bench/dpg_bench.csv"
    response = requests.get(url)
    reader = csv.DictReader(io.StringIO(response.text))

    categories = [category] if category is not None and not isinstance(category, list) else category
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in reader:
        row_category = row.get("category_broad", row.get("category", ""))

        if categories is not None:
            if row_category not in categories:
                continue

        key = (row.get("text", ""), row_category)
        q = row.get("question_natural_language", "")
        if q and q not in grouped[key]:
            grouped[key].append(q)

    records = [{"text": text, "category_broad": cat, "questions": qs} for (text, cat), qs in grouped.items()]

    ds = Dataset.from_list(records)
    test_sample_size = define_sample_size_for_dataset(ds, fraction, test_sample_size)
    ds = ds.select(range(min(test_sample_size, len(ds))))
    return _prepare_test_only_prompt_dataset(ds, seed, "DPG")
