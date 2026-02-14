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

    if len(ds) == 0:
        raise ValueError(f"No samples found for category '{category}'.")

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


IMGEDIT_CATEGORIES = ["replace", "add", "remove", "adjust", "extract", "style", "background", "compose"]


def setup_imgedit_dataset(
    seed: int,
    category: str | None = None,
    num_samples: int | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the ImgEdit benchmark dataset for image editing evaluation.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.
    category : str | None
        Filter by edit type. Available: replace, add, remove, adjust, extract, style,
        background, compose. If None, returns all categories.
    num_samples : int | None
        Maximum number of samples to return. If None, returns all samples.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The ImgEdit dataset (dummy train, dummy val, test).
    """
    import json

    import requests

    if category is not None and category not in IMGEDIT_CATEGORIES:
        raise ValueError(f"Invalid category: {category}. Must be one of {IMGEDIT_CATEGORIES}")

    instructions_url = "https://raw.githubusercontent.com/PKU-YuanGroup/ImgEdit/b3eb8e74d7cd1fd0ce5341eaf9254744a8ab4c0b/Benchmark/Basic/basic_edit.json"
    judge_prompts_url = "https://raw.githubusercontent.com/PKU-YuanGroup/ImgEdit/c14480ac5e7b622e08cd8c46f96624a48eb9ab46/Benchmark/Basic/prompts.json"

    instructions = json.loads(requests.get(instructions_url).text)
    judge_prompts = json.loads(requests.get(judge_prompts_url).text)

    records = []
    for _, instruction in instructions.items():
        edit_type = instruction.get("edit_type", "")

        if category is not None and edit_type != category:
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
    ds = ds.shuffle(seed=seed)

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    if len(ds) == 0:
        raise ValueError(f"No samples found for category '{category}'.")

    pruna_logger.info("ImgEdit is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


GEDIT_CATEGORIES = [
    "background_change",
    "color_alter",
    "material_alter",
    "motion_change",
    "ps_human",
    "style_change",
    "subject_add",
    "subject_remove",
    "subject_replace",
    "text_change",
    "tone_transfer",
]


def setup_gedit_dataset(
    seed: int,
    category: str | None = None,
    num_samples: int | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the GEditBench dataset for image editing evaluation.

    License: Apache 2.0

    Parameters
    ----------
    seed : int
        The seed to use.
    category : str | None
        Filter by task type. Available: background_change, color_alter, material_alter,
        motion_change, ps_human, style_change, subject_add, subject_remove, subject_replace,
        text_change, tone_transfer. If None, returns all categories.
    num_samples : int | None
        Maximum number of samples to return. If None, returns all samples.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The GEditBench dataset (dummy train, dummy val, test).
    """
    if category is not None and category not in GEDIT_CATEGORIES:
        raise ValueError(f"Invalid category: {category}. Must be one of {GEDIT_CATEGORIES}")

    task_type_map = {
        "subject_add": "subject-add",
        "subject_remove": "subject-remove",
        "subject_replace": "subject-replace",
    }

    ds = load_dataset("stepfun-ai/GEdit-Bench")["train"]  # type: ignore[index]
    ds = ds.filter(lambda x: x["instruction_language"] == "en")

    if category is not None:
        hf_task_type = task_type_map.get(category, category)
        ds = ds.filter(lambda x, tt=hf_task_type: x["task_type"] == tt)

    records = []
    for row in ds:
        task_type = row.get("task_type", "")
        category_name = {v: k for k, v in task_type_map.items()}.get(task_type, task_type)
        records.append(
            {
                "text": row.get("instruction", ""),
                "category": category_name,
            }
        )

    ds = Dataset.from_list(records)
    ds = ds.shuffle(seed=seed)

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    if len(ds) == 0:
        raise ValueError(f"No samples found for category '{category}'.")

    pruna_logger.info("GEditBench is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds
