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
        ds = ds.filter(lambda x: x["Category"] == category or x["Challenge"] == category)

    ds = ds.shuffle(seed=seed)

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


ONEIG_SUBSETS = ["text_rendering", "anime_alignment", "portrait_alignment", "object_alignment"]


def _load_oneig_text_rendering(seed: int) -> Dataset:
    """Load OneIG text rendering data from GitHub CSV."""
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
            "subset": "text_rendering",
            "text_content": row.get("text_content", row.get("text", "")),
        })

    return Dataset.from_list(records).shuffle(seed=seed)


def _load_oneig_alignment(seed: int, category: str | None = None) -> Dataset:
    """Load OneIG alignment data from HuggingFace + GitHub JSON."""
    import json

    import requests

    category_map = {
        "anime_alignment": "Anime_Stylization",
        "portrait_alignment": "Portrait",
        "object_alignment": "General_Object",
    }

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
            target_category = category_map.get(category, category)
            if row_category != target_category:
                continue

        subset_name = {v: k for k, v in category_map.items()}.get(row_category, "alignment")
        q_info = questions_by_id.get(row_id, {})
        records.append({
            "text": row.get("prompt", ""),
            "subset": subset_name,
            "category": row_category,
            "questions": q_info.get("questions", []),
            "dependencies": q_info.get("dependencies", []),
        })

    return Dataset.from_list(records).shuffle(seed=seed)


def setup_oneig_dataset(
    seed: int,
    subset: str | None = None,
    num_samples: int | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the OneIG benchmark dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.
    subset : str | None
        Filter by subset. Available: text_rendering, anime_alignment, portrait_alignment,
        object_alignment. If None, returns all subsets.
    num_samples : int | None
        Maximum number of samples to return. If None, returns all samples.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The OneIG dataset (dummy train, dummy val, test).
    """
    from datasets import concatenate_datasets

    if subset is not None and subset not in ONEIG_SUBSETS:
        raise ValueError(f"Invalid subset: {subset}. Must be one of {ONEIG_SUBSETS}")

    datasets_to_concat = []

    if subset is None or subset == "text_rendering":
        datasets_to_concat.append(_load_oneig_text_rendering(seed))

    if subset is None or subset in ["anime_alignment", "portrait_alignment", "object_alignment"]:
        alignment_subset = subset if subset in ["anime_alignment", "portrait_alignment", "object_alignment"] else None
        datasets_to_concat.append(_load_oneig_alignment(seed, alignment_subset))

    ds = concatenate_datasets(datasets_to_concat) if len(datasets_to_concat) > 1 else datasets_to_concat[0]
    ds = ds.shuffle(seed=seed)

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    pruna_logger.info("OneIG is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds
