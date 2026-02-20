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


def setup_hps_dataset(
    seed: int,
    category: str | None = None,
    num_samples: int | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the HPS (Human Preference Score) benchmark dataset.

    License: MIT

    Parameters
    ----------
    seed : int
        The seed to use.
    category : str | None
        Filter by category. Available categories: anime, concept-art, paintings, photo.
    num_samples : int | None
        Maximum number of samples to return. If None, returns all samples.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The HPS dataset (dummy train, dummy val, test).
    """
    import json

    from huggingface_hub import hf_hub_download

    hps_categories = ["anime", "concept-art", "paintings", "photo"]
    categories_to_load = [category] if category else hps_categories

    all_prompts = []
    for cat in categories_to_load:
        if cat not in hps_categories:
            raise ValueError(f"Invalid category: {cat}. Must be one of {hps_categories}")
        file_path = hf_hub_download("zhwang/HPDv2", f"{cat}.json", subfolder="benchmark", repo_type="dataset")
        with open(file_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            for prompt in prompts:
                all_prompts.append({"text": prompt, "category": cat})

    ds = Dataset.from_list(all_prompts)
    # Note: Not shuffling since these are test-only datasets

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    pruna_logger.info("HPS is a test-only dataset. Do not use it for training or validation.")
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
