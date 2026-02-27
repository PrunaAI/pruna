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

from typing import Literal, Tuple, get_args

from datasets import Dataset, load_dataset

from pruna.data.utils import _prepare_test_only_prompt_dataset, define_sample_size_for_dataset
from pruna.logging.logger import pruna_logger

GenEvalCategory = Literal["single_object", "two_object", "counting", "colors", "position", "color_attr"]
HPSCategory = Literal["anime", "concept-art", "paintings", "photo"]

OneIGCategory = Literal[
    "Anime_Stylization",
    "General_Object",
    "Knowledge_Reasoning",
    "Multilingualism",
    "Portrait",
    "Text_Rendering",
    "3d rendering",
    "Baroque",
    "Celluloid",
    "Chibi",
    "Chinese ink painting",
    "Cyberpunk",
    "Ghibli",
    "LEGO",
    "None",
    "PPT generation",
    "Pixar",
    "Rococo",
    "Ukiyo-e",
    "abstract expressionism",
    "advertising imagery",
    "art nouveau",
    "artistic renderings",
    "biology",
    "blackboard text",
    "chemistry",
    "clay",
    "comic",
    "common sense",
    "computer science",
    "crayon",
    "cubism",
    "fauvism",
    "floating-frame text",
    "geography",
    "graffiti",
    "graffiti-style text",
    "impasto",
    "impressionism",
    "line art",
    "long text rendering",
    "mathematics",
    "menu",
    "minimalism",
    "natural-scene text",
    "noir",
    "pencil sketch",
    "physics",
    "pixel art",
    "pointillism",
    "pop art",
    "poster design",
    "silvertone",
    "stone sculpture",
    "vintage",
    "vivid cold",
    "vivid warm",
    "watercolor",
]

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


def setup_hps_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: HPSCategory | list[HPSCategory] | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the HPS (Human Preference Score) benchmark dataset.

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
    category : HPSCategory | list[HPSCategory] | None
        Filter by category. Available: anime, concept-art, paintings, photo.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The HPS dataset (dummy train, dummy val, test).
    """
    import json

    from huggingface_hub import hf_hub_download

    categories_to_load = (
        list(get_args(HPSCategory)) if category is None else ([category] if not isinstance(category, list) else category)
    )

    all_prompts = []
    for cat in categories_to_load:
        file_path = hf_hub_download("zhwang/HPDv2", f"{cat}.json", subfolder="benchmark", repo_type="dataset")
        with open(file_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            for prompt in prompts:
                all_prompts.append({"text": prompt, "category": cat})

    ds = Dataset.from_list(all_prompts)
    test_sample_size = define_sample_size_for_dataset(ds, fraction, test_sample_size)
    ds = ds.select(range(min(test_sample_size, len(ds))))
    return _prepare_test_only_prompt_dataset(ds, seed, "HPS")


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


def _load_oneig_text_rendering(seed: int, class_filter: str | None = None) -> Dataset:
    """Load OneIG text rendering data from HuggingFace dataset."""
    ds = load_dataset("OneIG-Bench/OneIG-Bench", "OneIG-Bench")["train"]  # type: ignore[index]
    ds = ds.filter(lambda x: x.get("category", "") in ("Text_Rendering", "Text Rendering"))

    def to_record(row: dict) -> dict:
        prompt = row.get("prompt_en", row.get("prompt", ""))
        row_class = row.get("class", "None") or "None"
        return {
            "text": prompt,
            "subset": "Text_Rendering",
            "text_content": row_class if row_class != "None" else "",
            "category": row.get("category"),
            "class": row_class,
            "questions": [],
            "dependencies": [],
        }

    records = []
    for row in ds:
        row_dict = dict(row)
        if class_filter is not None and (row_dict.get("class") or "None") != class_filter:
            continue
        records.append(to_record(row_dict))

    return Dataset.from_list(records).shuffle(seed=seed)


def _load_oneig_alignment(seed: int, category: str | None = None, class_filter: str | None = None) -> Dataset:
    """Load OneIG alignment data from HuggingFace + GitHub JSON."""
    import json

    import requests

    ds = load_dataset("OneIG-Bench/OneIG-Bench", "OneIG-Bench")["train"]  # type: ignore[index]

    questions_by_id: dict[str, dict] = {}
    url = "https://raw.githubusercontent.com/OneIG-Bench/OneIG-Benchmark/main/benchmark/alignment_questions.json"
    response = requests.get(url)
    questions_by_id: dict[str, dict] = {}
    if response.status_code == 200:
        try:
            questions_data = json.loads(response.text)
            questions_by_id = {q["id"]: q for q in questions_data}
        except json.JSONDecodeError:
            pass

    alignment_cats = {"Anime_Stylization", "Portrait", "General_Object"}
    records = []
    for row in ds:
        row_id = row.get("id", "")
        row_category = row.get("category", "")
        row_class = row.get("class", "None") or "None"

        if row_category not in alignment_cats:
            continue
        if category is not None and row_category != category:
            continue
        if class_filter is not None and row_class != class_filter:
            continue

        q_info = questions_by_id.get(row_id, {})
        records.append(
            {
                "text": row.get("prompt_en", row.get("prompt", "")),
                "subset": row_category,
                "text_content": row_class if row_class != "None" else None,
                "category": row_category,
                "class": row_class,
                "questions": q_info.get("questions", []),
                "dependencies": q_info.get("dependencies", []),
            }
        )

    return Dataset.from_list(records).shuffle(seed=seed)


ONEIG_DATASET_CATEGORIES = frozenset(
    {"Anime_Stylization", "General_Object", "Knowledge_Reasoning", "Multilingualism", "Portrait", "Text_Rendering"}
)


def _load_oneig_generic(
    seed: int,
    category_filter: str | None = None,
    class_filter: str | None = None,
    config: str = "OneIG-Bench",
) -> Dataset:
    """Load OneIG data for Knowledge_Reasoning, Multilingualism, or any category without alignment questions."""
    ds = load_dataset("OneIG-Bench/OneIG-Bench", config)[  # type: ignore[index]
        "train"
    ]

    records = []
    for row in ds:
        row_category = row.get("category", "")
        row_class = row.get("class", "None") or "None"

        if category_filter is not None and row_category != category_filter:
            continue
        if class_filter is not None and row_class != class_filter:
            continue

        records.append(
            {
                "text": row.get("prompt_en", row.get("prompt", "")),
                "subset": row_category,
                "text_content": row_class if row_class != "None" else None,
                "category": row_category,
                "class": row_class,
                "questions": [],
                "dependencies": [],
            }
        )

    return Dataset.from_list(records).shuffle(seed=seed)


def setup_oneig_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: OneIGCategory | list[OneIGCategory] | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the OneIG benchmark dataset.

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
    category : OneIGCategory | list[OneIGCategory] | None
        Filter by dataset category (Anime_Stylization, Portrait, etc.) or class (fauvism,
        watercolor, etc.). If None, returns all subsets.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The OneIG dataset (dummy train, dummy val, test).
    """
    from datasets import concatenate_datasets

    categories = [category] if category is not None and not isinstance(category, list) else category
    single_filter = categories[0] if categories and len(categories) == 1 else None
    multi_filter = categories if categories and len(categories) > 1 else None

    alignment_categories = {"Anime_Stylization", "Portrait", "General_Object"}

    category_filter: str | None = None
    class_filter: str | None = None

    if single_filter in ONEIG_DATASET_CATEGORIES:
        category_filter = single_filter
    elif single_filter is not None:
        class_filter = single_filter

    load_text = single_filter is None or category_filter == "Text_Rendering" or class_filter
    load_align = single_filter is None or category_filter in alignment_categories or class_filter
    load_kr = single_filter is None or category_filter == "Knowledge_Reasoning"
    load_multi = single_filter is None or category_filter == "Multilingualism"

    datasets_to_concat = []
    if load_text:
        datasets_to_concat.append(_load_oneig_text_rendering(seed, class_filter or None))

    if load_align:
        align_cat = category_filter if category_filter in alignment_categories else None
        datasets_to_concat.append(_load_oneig_alignment(seed, category=align_cat, class_filter=class_filter or None))

    if load_kr:
        datasets_to_concat.append(_load_oneig_generic(seed, category_filter="Knowledge_Reasoning"))

    if load_multi:
        datasets_to_concat.append(_load_oneig_generic(seed, category_filter="Multilingualism", config="OneIG-Bench-ZH"))

    ds = concatenate_datasets(datasets_to_concat) if len(datasets_to_concat) > 1 else datasets_to_concat[0]

    if multi_filter:
        ds = ds.filter(
            lambda x: (
                x.get("category") in multi_filter or x.get("class") in multi_filter or x.get("subset") in multi_filter
            )
        )

    test_sample_size = define_sample_size_for_dataset(ds, fraction, test_sample_size)
    ds = ds.select(range(min(test_sample_size, len(ds))))

    if len(ds) == 0:
        raise ValueError(f"No samples found for category '{category}'. Check that the category exists and has data.")

    return _prepare_test_only_prompt_dataset(ds, seed, "OneIG")


def setup_oneig_text_rendering_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Setup OneIG Text Rendering benchmark (subset of OneIG)."""
    ds = _load_oneig_text_rendering(seed, None)
    n = define_sample_size_for_dataset(ds, fraction, test_sample_size)
    ds = ds.select(range(min(n, len(ds))))
    return _prepare_test_only_prompt_dataset(ds, seed, "OneIGTextRendering")


def setup_oneig_alignment_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: str | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Setup OneIG Alignment benchmark (subset of OneIG)."""
    ds = _load_oneig_alignment(seed, category=category, class_filter=None)
    n = define_sample_size_for_dataset(ds, fraction, test_sample_size)
    ds = ds.select(range(min(n, len(ds))))
    return _prepare_test_only_prompt_dataset(ds, seed, "OneIGAlignment")


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
