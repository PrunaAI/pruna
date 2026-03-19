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

from pruna.data.utils import stratify_dataset
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
ImgEditCategory = Literal["replace", "add", "remove", "adjust", "extract", "style", "background", "compose"]
GEditBenchCategory = Literal[
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
DPGCategory = Literal["entity", "attribute", "relation", "global", "other"]


def _to_oneig_record(row: dict, questions_by_key: dict[str, dict]) -> dict:
    """Convert OneIG row to unified record format."""
    row_category = row.get("category", "")
    row_class = row.get("class", "None") or "None"
    qd_name = _CATEGORY_TO_QD.get(row_category, "")
    lookup_key = f"{qd_name}_{row.get('id', '')}" if qd_name else ""
    q_info = questions_by_key.get(lookup_key, {})
    return {
        "text": row.get("prompt_en", row.get("prompt", "")),
        "subset": "Text_Rendering" if row_category in ("Text_Rendering", "Text Rendering") else row_category,
        "text_content": row_class if row_class != "None" else None,
        "category": row_category,
        "class": row_class,
        "questions": q_info.get("questions", {}),
        "dependencies": q_info.get("dependencies", {}),
    }


def setup_drawbench_dataset() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the DrawBench dataset.

    License: Apache 2.0

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

    ds = stratify_dataset(ds, sample_size=test_sample_size, fraction=fraction)
    ds = ds.rename_column("Prompt", "text")
    pruna_logger.info("PartiPrompts is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


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
    response = requests.get(url, timeout=30)
    response.raise_for_status()
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
            }
        )

    ds = Dataset.from_list(records)
    ds = stratify_dataset(ds, sample_size=test_sample_size, fraction=fraction)
    pruna_logger.info("GenEval is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


def setup_hps_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: HPSCategory | list[HPSCategory] | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the HPD (Human Preference Dataset) for the HPS (Human Preference Score) benchmark.

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
        The HPD dataset (dummy train, dummy val, test).
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
    ds = stratify_dataset(ds, sample_size=test_sample_size, fraction=fraction)
    pruna_logger.info("HPD is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


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
    ds = stratify_dataset(ds, sample_size=test_sample_size, fraction=fraction)
    pruna_logger.info("LongTextBench is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


def setup_genai_bench_dataset() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the GenAI Bench dataset.

    License: Apache 2.0

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The GenAI Bench dataset.
    """
    ds = load_dataset("BaiqiL/GenAI-Bench")["train"]  # type: ignore[index]
    ds = ds.rename_column("Prompt", "text")
    pruna_logger.info("GenAI-Bench is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


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
    response_instructions = requests.get(instructions_url, timeout=30)
    response_judge_prompts = requests.get(judge_prompts_url, timeout=30)
    response_instructions.raise_for_status()
    response_judge_prompts.raise_for_status()
    instructions: dict = json.loads(response_instructions.text)
    judge_prompts: dict = json.loads(response_judge_prompts.text)

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
    ds = stratify_dataset(ds, sample_size=test_sample_size, fraction=fraction)

    if len(ds) == 0:
        raise ValueError(f"No samples found for category '{category}'.")

    pruna_logger.info("ImgEdit is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


_CATEGORY_TO_QD: dict[str, str] = {
    "Anime_Stylization": "anime",
    "Portrait": "human",
    "General_Object": "object",
}

_ONEIG_ALIGNMENT_BASE = "https://raw.githubusercontent.com/OneIG-Bench/OneIG-Benchmark/41b49831e79e6dde5323618c164da1c4cf0f699d/scripts/alignment/Q_D"


def _fetch_oneig_alignment() -> dict[str, dict]:
    """Fetch alignment questions from per-category Q_D files (InferBench-style)."""
    import json

    import requests

    questions_by_key: dict[str, dict] = {}
    for qd_name in ("anime", "human", "object"):
        url = f"{_ONEIG_ALIGNMENT_BASE}/{qd_name}.json"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = json.loads(resp.text)
        for prompt_id, item in data.items():
            q = item.get("question", {})
            d = item.get("dependency", {})
            if isinstance(q, str):
                q = json.loads(q)
            if isinstance(d, str):
                d = json.loads(d)
            questions_by_key[f"{qd_name}_{prompt_id}"] = {"questions": q, "dependencies": d}
    return questions_by_key


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
    questions_by_key = _fetch_oneig_alignment()

    ds_raw = load_dataset("OneIG-Bench/OneIG-Bench", "OneIG-Bench")["train"]  # type: ignore[index]
    records = [_to_oneig_record(dict(row), questions_by_key) for row in ds_raw]
    ds = Dataset.from_list(records)

    if category is not None:
        categories = [category] if not isinstance(category, list) else category
        ds = ds.filter(
            lambda x: x.get("category") in categories or x.get("class") in categories or x.get("subset") in categories
        )

    ds = stratify_dataset(ds, sample_size=test_sample_size, fraction=fraction)

    if len(ds) == 0:
        raise ValueError(f"No samples found for category '{category}'. Check that the category exists and has data.")

    pruna_logger.info("OneIG is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


def setup_gedit_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: GEditBenchCategory | list[GEditBenchCategory] | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the GEditBench dataset for image editing evaluation.

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
    category : GEditBenchCategory | list[GEditBenchCategory] | None
        Filter by task type. Available: background_change, color_alter, material_alter,
        motion_change, ps_human, style_change, subject_add, subject_remove, subject_replace,
        text_change, tone_transfer. If None, returns all categories.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The GEditBench dataset (dummy train, dummy val, test).
    """
    task_type_map = {
        "subject_add": "subject-add",
        "subject_remove": "subject-remove",
        "subject_replace": "subject-replace",
    }
    task_type_to_category = {v: k for k, v in task_type_map.items()}

    ds = load_dataset("stepfun-ai/GEdit-Bench")["train"]  # type: ignore[index]
    ds = ds.filter(lambda x: x["instruction_language"] == "en")

    categories = [category] if category is not None and not isinstance(category, list) else category
    if categories is not None:
        hf_types = [task_type_map.get(c, c) for c in categories]
        ds = ds.filter(lambda x: x["task_type"] in hf_types)

    records = []
    for row in ds:
        task_type = row.get("task_type", "")
        category_name = task_type_to_category.get(task_type, task_type)
        records.append(
            {
                "text": row.get("instruction", ""),
                "category": category_name,
            }
        )

    ds = Dataset.from_list(records)
    ds = stratify_dataset(ds, sample_size=test_sample_size, fraction=fraction)

    if len(ds) == 0:
        raise ValueError(f"No samples found for category '{category}'.")

    pruna_logger.info("GEditBench is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds


def setup_dpg_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: DPGCategory | list[DPGCategory] | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the DPG (Dense Prompt Graph) benchmark dataset.

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
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))

    categories = [category] if category is not None and not isinstance(category, list) else category
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in reader:
        row_category = row.get("category_broad", row.get("category", ""))

        if categories is not None and row_category not in categories:
            continue

        key = (row.get("text", ""), row_category)
        q = row.get("question_natural_language", "")
        if q and q not in grouped[key]:
            grouped[key].append(q)

    records = [{"text": text, "category": cat, "questions": qs} for (text, cat), qs in grouped.items()]

    ds = Dataset.from_list(records)
    ds = stratify_dataset(ds, sample_size=test_sample_size, fraction=fraction)
    pruna_logger.info("DPG is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds
