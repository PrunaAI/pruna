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

from typing import Tuple

from datasets import Dataset, load_dataset

from pruna.logging.logger import pruna_logger


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
    category: str | None = None,
    num_samples: int | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the Parti Prompts dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.
    category : str | None
        Filter by Category or Challenge. Available categories: Abstract, Animals, Artifacts,
        Arts, Food & Beverage, Illustrations, Indoor Scenes, Outdoor Scenes, People,
        Produce & Plants, Vehicles, World Knowledge. Available challenges: Basic, Complex,
        Fine-grained Detail, Imagination, Linguistic Structures, Perspective,
        Properties & Positioning, Quantity, Simple Detail, Style & Format, Writing & Symbols.
    num_samples : int | None
        Maximum number of samples to return. If None, returns all samples.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The Parti Prompts dataset (dummy train, dummy val, test).
    """
    ds = load_dataset("nateraw/parti-prompts")["train"]  # type: ignore[index]

    if category is not None:
        if isinstance(category, list):
            ds = ds.filter(
                lambda x: x["Category"] in category or x["Challenge"] in category
            )
        else:
            ds = ds.filter(
                lambda x: x["Category"] == category or x["Challenge"] == category
            )

    # Note: Not shuffling since these are test-only datasets

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    ds = ds.rename_column("Prompt", "text")
    pruna_logger.info("PartiPrompts is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


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
    import csv
    import io

    import requests

    url = "https://raw.githubusercontent.com/OneIG-Bench/OneIG-Benchmark/main/benchmark/text_rendering.csv"
    response = requests.get(url)
    reader = csv.DictReader(io.StringIO(response.text))

    records = []
    for row in reader:
        records.append({
            "text": row.get("prompt", ""),
            "text_content": row.get("text_content", row.get("text", "")),
        })

    ds = Dataset.from_list(records)
    # Note: Not shuffling since these are test-only datasets

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

    ds = load_dataset("OneIG-Bench/OneIG-Bench")["test"]  # type: ignore[index]

    url = "https://raw.githubusercontent.com/OneIG-Bench/OneIG-Benchmark/main/benchmark/alignment_questions.json"
    response = requests.get(url)
    questions_data = json.loads(response.text)

    questions_by_id = {q["id"]: q for q in questions_data}

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
        records.append({
            "text": row.get("prompt", ""),
            "category": row_category,
            "questions": q_info.get("questions", []),
            "dependencies": q_info.get("dependencies", []),
        })

    ds = Dataset.from_list(records)
    # Note: Not shuffling since these are test-only datasets

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    pruna_logger.info("OneIG Alignment is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


DPG_CATEGORIES = ["entity", "attribute", "relation", "global", "other"]


def setup_dpg_dataset(
    seed: int,
    category: str | None = None,
    num_samples: int | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the DPG (Descriptive Prompt Generation) benchmark dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.
    category : str | None
        Filter by category. Available: entity, attribute, relation, global, other.
    num_samples : int | None
        Maximum number of samples to return. If None, returns all samples.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The DPG dataset (dummy train, dummy val, test).
    """
    import csv
    import io

    import requests

    url = "https://raw.githubusercontent.com/TencentQQGYLab/ELLA/main/dpg_bench/prompts.csv"
    response = requests.get(url)
    reader = csv.DictReader(io.StringIO(response.text))

    records = []
    for row in reader:
        row_category = row.get("category", row.get("category_broad", ""))

        if category is not None:
            if category not in DPG_CATEGORIES:
                raise ValueError(f"Invalid category: {category}. Must be one of {DPG_CATEGORIES}")
            if row_category != category:
                continue

        records.append({
            "text": row.get("prompt", ""),
            "category_broad": row_category,
            "questions": row.get("questions", "").split("|") if row.get("questions") else [],
        })

    ds = Dataset.from_list(records)
    # Note: Not shuffling since these are test-only datasets

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    pruna_logger.info("DPG is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds
