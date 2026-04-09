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


def _warn_ignored_benchmark_seed(seed: int | None, *, dataset: str) -> None:
    if seed is not None:
        pruna_logger.warning(
            "%s: `seed` is ignored for this test-only benchmark; sampling does not shuffle the test split.",
            dataset,
        )


def _oneig_alignment_language_zh(row: dict) -> bool:
    """Return True when the official Q_D file for this row should use the ``*_zh`` graphs."""
    row_category = row.get("category", "")
    if row_category == "Multilingualism":
        return True
    lang = row.get("language") or row.get("lang")
    if isinstance(lang, str) and lang.lower() in {"zh", "zh-cn", "zh_cn", "chinese", "cn"}:
        return True
    if row.get("prompt_zh"):
        return True
    prompt = row.get("prompt")
    prompt_en = row.get("prompt_en")
    return bool(prompt and not (isinstance(prompt_en, str) and prompt_en.strip()))


def _oneig_qd_prefix(row: dict) -> str:
    """Map dataset ``category`` (+ language) to Q_D JSON stem (e.g. ``object``, ``anime_zh``)."""
    row_category = row.get("category", "")
    use_zh = _oneig_alignment_language_zh(row)
    if row_category == "Multilingualism":
        return "multilingualism_zh"
    base = _CATEGORY_TO_QD.get(row_category, "")
    if not base:
        return ""
    return f"{base}_zh" if use_zh else base


def _to_oneig_record(
    row: dict,
    questions_by_key: dict[str, dict],
    reasoning_gt_en: dict[str, str],
    reasoning_gt_zh: dict[str, str],
    reasoning_language: str = "EN",
) -> dict:
    """Convert OneIG row to unified record format.

    Parameters
    ----------
    row : dict
        Raw Hugging Face row (``category``, ``id``, ``class``). EN configs use ``prompt_en``; the
        ``OneIG-Bench-ZH`` **Multilingualism** split uses ``prompt_cn`` instead of ``prompt_en``.
    questions_by_key : dict[str, dict]
        Merged Q_D index keyed as ``{qd_stem}_{prompt_id}`` (see ``_fetch_oneig_alignment``).
    reasoning_gt_en : dict[str, str]
        Official ``gt_answer.json`` keyed by prompt id (e.g. ``"000"``).
    reasoning_gt_zh : dict[str, str]
        Official ``gt_answer_zh.json`` keyed by prompt id.
    reasoning_language : str, optional
        Which reasoning GT to use: ``"EN"`` or ``"ZH"``. Default is ``"EN"``.

    Returns
    -------
    dict
        Unified record including ``questions``, ``dependencies``, and ``reasoning_gt_answer`` when
        applicable (Knowledge_Reasoning only).
    """
    row_category = row.get("category", "")
    row_class = row.get("class", "None") or "None"
    prompt_id = str(row.get("id", ""))
    qd_prefix = _oneig_qd_prefix(row)
    lookup_key = f"{qd_prefix}_{prompt_id}" if qd_prefix else ""
    q_info = questions_by_key.get(lookup_key, {})
    text = row.get("prompt") or row.get("prompt_en") or row.get("prompt_cn") or ""
    reasoning_gt_answer: str | None = None
    if row_category == "Knowledge_Reasoning":
        if reasoning_language.upper() == "ZH":
            reasoning_gt_answer = reasoning_gt_zh.get(prompt_id)
        else:
            reasoning_gt_answer = reasoning_gt_en.get(prompt_id)
    return {
        "text": text,
        "subset": "Text_Rendering" if row_category in ("Text_Rendering", "Text Rendering") else row_category,
        "text_content": row_class if row_class != "None" else None,
        "category": row_category,
        "class": row_class,
        "questions": q_info.get("questions", {}),
        "dependencies": q_info.get("dependencies", {}),
        "reasoning_gt_answer": reasoning_gt_answer,
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
    seed: int | None = None,
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
    seed : int | None, optional
        Ignored; test order is deterministic. If not None, a warning is logged.
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
    _warn_ignored_benchmark_seed(seed, dataset="PartiPrompts")
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
    seed: int | None = None,
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
    seed : int | None, optional
        Ignored; test order is deterministic. If not None, a warning is logged.
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
    _warn_ignored_benchmark_seed(seed, dataset="GenEval")
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
    seed: int | None = None,
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
    seed : int | None, optional
        Ignored; test order is deterministic. If not None, a warning is logged.
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
    _warn_ignored_benchmark_seed(seed, dataset="HPS")
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
    seed: int | None = None,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the Long Text Bench dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int | None, optional
        Ignored; test order is deterministic. If not None, a warning is logged.
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
    _warn_ignored_benchmark_seed(seed, dataset="LongTextBench")
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
    seed: int | None = None,
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
    seed : int | None, optional
        Ignored; test order is deterministic. If not None, a warning is logged.
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
    _warn_ignored_benchmark_seed(seed, dataset="ImgEdit")
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

_ONEIG_BENCHMARK_REF = "41b49831e79e6dde5323618c164da1c4cf0f699d"
_ONEIG_RAW_BASE = f"https://raw.githubusercontent.com/OneIG-Bench/OneIG-Benchmark/{_ONEIG_BENCHMARK_REF}"
_ONEIG_ALIGNMENT_QD_URL = f"{_ONEIG_RAW_BASE}/scripts/alignment/Q_D"
_ONEIG_REASONING_GT_URL_EN = f"{_ONEIG_RAW_BASE}/scripts/reasoning/gt_answer.json"
_ONEIG_REASONING_GT_URL_ZH = f"{_ONEIG_RAW_BASE}/scripts/reasoning/gt_answer_zh.json"

_ONEIG_QD_JSON_STEMS: tuple[str, ...] = (
    "anime",
    "human",
    "object",
    "anime_zh",
    "human_zh",
    "object_zh",
    "multilingualism_zh",
)


def _fetch_oneig_alignment() -> dict[str, dict]:
    """Load OneIG question/dependency graphs from the official repo (HTTP, no on-disk cache).

    Fetches every ``scripts/alignment/Q_D/*.json`` file used by upstream ``alignment_score.py`` (EN + ZH),
    including ``multilingualism_zh.json``. Keys in the returned map are ``{stem}_{prompt_id}`` matching
    upstream file stems (e.g. ``object_012``, ``multilingualism_zh_000``).

    Returns
    -------
    dict[str, dict]
        ``prompt_id``-level ``questions`` and ``dependencies`` dicts (parsed from JSON strings when needed).

    Raises
    ------
    requests.HTTPError
        If any asset URL is missing or the response is not successful.
    """
    import json

    import requests

    questions_by_key: dict[str, dict] = {}
    for stem in _ONEIG_QD_JSON_STEMS:
        url = f"{_ONEIG_ALIGNMENT_QD_URL}/{stem}.json"
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
            questions_by_key[f"{stem}_{prompt_id}"] = {"questions": q, "dependencies": d}
    return questions_by_key


def _fetch_oneig_reasoning_gt() -> tuple[dict[str, str], dict[str, str]]:
    """Load official knowledge-reasoning reference answers (HTTP, no on-disk cache).

    Mirrors ``scripts/reasoning/gt_answer.json`` and ``gt_answer_zh.json`` from the same pinned commit as Q_D.
    Keys are prompt ids (``str``), values are answer strings; downstream metrics may slice filenames to the
    first three characters like ``reasoning_score.py``.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        ``(en_by_id, zh_by_id)``.

    Raises
    ------
    requests.HTTPError
        If any asset URL is missing or the response is not successful.
    """
    import json

    import requests

    def _load(url: str) -> dict[str, str]:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        raw = json.loads(resp.text)
        return {str(k): str(v) for k, v in raw.items()}

    return _load(_ONEIG_REASONING_GT_URL_EN), _load(_ONEIG_REASONING_GT_URL_ZH)


def _oneig_needs_zh_multilingualism_hub(category: OneIGCategory | list[OneIGCategory] | None) -> bool:
    """Whether ``OneIG-Bench-ZH`` must be loaded for ``Multilingualism`` rows."""
    if category is None:
        return True
    categories = [category] if not isinstance(category, list) else category
    return "Multilingualism" in categories


def setup_oneig_dataset(
    seed: int | None = None,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: OneIGCategory | list[OneIGCategory] | None = None,
    reasoning_language: str = "EN",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Setup the OneIG benchmark dataset.

    License: Apache 2.0

    Parameters
    ----------
    seed : int | None, optional
        Ignored; test order is deterministic. If not None, a warning is logged.
    fraction : float
        The fraction of the dataset to use.
    train_sample_size : int | None
        Unused; train/val are dummy.
    test_sample_size : int | None
        The sample size to use for the test dataset.
    category : OneIGCategory | list[OneIGCategory] | None
        Filter by dataset category (Anime_Stylization, Portrait, etc.) or class (fauvism,
        watercolor, etc.). If None, returns all subsets.
    reasoning_language : str, optional
        Which reasoning GT to use for Knowledge_Reasoning rows: ``"EN"`` or ``"ZH"``. Default is ``"EN"``.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        The OneIG dataset (dummy train, dummy val, test). Rows include ``questions`` and
        ``dependencies`` from official Q_D JSON (EN + ZH stems, including ``multilingualism_zh``),
        plus ``reasoning_gt_answer`` for ``Knowledge_Reasoning`` (language chosen by ``reasoning_language``).
        Rows cover EN categories from ``OneIG-Bench`` plus ``Multilingualism`` from ``OneIG-Bench-ZH``.
        Assets are downloaded over HTTP on each call (pinned commit ``_ONEIG_BENCHMARK_REF``); there is
        no local disk cache.

    Notes
    -----
    Non-multilingual prompts are loaded from the Hub config ``OneIG-Bench``; **Multilingualism** rows
    are taken only from ``OneIG-Bench-ZH`` (they use ``prompt_cn``). The ZH config is fetched only when
    the requested ``category`` is ``None`` (full suite) or explicitly includes ``Multilingualism``.
    Q_D / reasoning JSON URLs are defined next to ``_fetch_oneig_alignment`` and
    ``_fetch_oneig_reasoning_gt``.
    """
    _warn_ignored_benchmark_seed(seed, dataset="OneIG")
    questions_by_key = _fetch_oneig_alignment()
    reasoning_gt_en, reasoning_gt_zh = _fetch_oneig_reasoning_gt()

    ds_en = load_dataset("OneIG-Bench/OneIG-Bench", "OneIG-Bench")["train"]  # type: ignore[index]
    records = [
        _to_oneig_record(dict(row), questions_by_key, reasoning_gt_en, reasoning_gt_zh, reasoning_language)
        for row in ds_en
    ]
    if _oneig_needs_zh_multilingualism_hub(category):
        ds_zh = load_dataset("OneIG-Bench/OneIG-Bench", "OneIG-Bench-ZH")["train"]  # type: ignore[index]
        ds_zh_ml = ds_zh.filter(lambda r: r["category"] == "Multilingualism")
        records.extend(
            _to_oneig_record(dict(row), questions_by_key, reasoning_gt_en, reasoning_gt_zh, reasoning_language)
            for row in ds_zh_ml
        )
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


def _setup_oneig_subset_with_fixed_category(
    category: OneIGCategory,
    seed: int | None = None,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    reasoning_language: str = "EN",
) -> Tuple[Dataset, Dataset, Dataset]:
    return setup_oneig_dataset(
        seed=seed,
        fraction=fraction,
        train_sample_size=train_sample_size,
        test_sample_size=test_sample_size,
        category=category,
        reasoning_language=reasoning_language,
    )


def setup_oneig_anime_stylization_dataset(
    seed: int | None = None,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    reasoning_language: str = "EN",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load OneIG-Bench with ``category`` fixed to ``Anime_Stylization``.

    ``functools.partial`` is not used so ``get_literal_values_from_param`` does not unwrap to
    :func:`setup_oneig_dataset` and enumerate every ``OneIGCategory``.

    Parameters
    ----------
    seed : int | None, optional
        Ignored; see :func:`setup_oneig_dataset`.
    fraction : float
        Fraction of the subset to use.
    train_sample_size : int | None
        Unused; train/val are dummy.
    test_sample_size : int | None
        Test sample size cap for the subset.
    reasoning_language : str
        Passed to :func:`setup_oneig_dataset`.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        Dummy train, dummy val, and test split for this subset.
    """
    return _setup_oneig_subset_with_fixed_category(
        "Anime_Stylization",
        seed,
        fraction,
        train_sample_size,
        test_sample_size,
        reasoning_language,
    )


def setup_oneig_general_object_dataset(
    seed: int | None = None,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    reasoning_language: str = "EN",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load OneIG-Bench with ``category`` fixed to ``General_Object``.

    Parameters
    ----------
    seed : int | None, optional
        Ignored; see :func:`setup_oneig_dataset`.
    fraction : float
        Fraction of the subset to use.
    train_sample_size : int | None
        Unused; train/val are dummy.
    test_sample_size : int | None
        Test sample size cap for the subset.
    reasoning_language : str
        Passed to :func:`setup_oneig_dataset`.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        Dummy train, dummy val, and test split for this subset.
    """
    return _setup_oneig_subset_with_fixed_category(
        "General_Object",
        seed,
        fraction,
        train_sample_size,
        test_sample_size,
        reasoning_language,
    )


def setup_oneig_knowledge_reasoning_dataset(
    seed: int | None = None,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    reasoning_language: str = "EN",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load OneIG-Bench with ``category`` fixed to ``Knowledge_Reasoning``.

    Parameters
    ----------
    seed : int | None, optional
        Ignored; see :func:`setup_oneig_dataset`.
    fraction : float
        Fraction of the subset to use.
    train_sample_size : int | None
        Unused; train/val are dummy.
    test_sample_size : int | None
        Test sample size cap for the subset.
    reasoning_language : str
        Passed to :func:`setup_oneig_dataset`.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        Dummy train, dummy val, and test split for this subset.
    """
    return _setup_oneig_subset_with_fixed_category(
        "Knowledge_Reasoning",
        seed,
        fraction,
        train_sample_size,
        test_sample_size,
        reasoning_language,
    )


def setup_oneig_multilingualism_dataset(
    seed: int | None = None,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    reasoning_language: str = "EN",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load OneIG-Bench with ``category`` fixed to ``Multilingualism``.

    Parameters
    ----------
    seed : int | None, optional
        Ignored; see :func:`setup_oneig_dataset`.
    fraction : float
        Fraction of the subset to use.
    train_sample_size : int | None
        Unused; train/val are dummy.
    test_sample_size : int | None
        Test sample size cap for the subset.
    reasoning_language : str
        Passed to :func:`setup_oneig_dataset`.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        Dummy train, dummy val, and test split for this subset.
    """
    return _setup_oneig_subset_with_fixed_category(
        "Multilingualism",
        seed,
        fraction,
        train_sample_size,
        test_sample_size,
        reasoning_language,
    )


def setup_oneig_portrait_dataset(
    seed: int | None = None,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    reasoning_language: str = "EN",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load OneIG-Bench with ``category`` fixed to ``Portrait``.

    Parameters
    ----------
    seed : int | None, optional
        Ignored; see :func:`setup_oneig_dataset`.
    fraction : float
        Fraction of the subset to use.
    train_sample_size : int | None
        Unused; train/val are dummy.
    test_sample_size : int | None
        Test sample size cap for the subset.
    reasoning_language : str
        Passed to :func:`setup_oneig_dataset`.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        Dummy train, dummy val, and test split for this subset.
    """
    return _setup_oneig_subset_with_fixed_category(
        "Portrait",
        seed,
        fraction,
        train_sample_size,
        test_sample_size,
        reasoning_language,
    )


def setup_oneig_text_rendering_dataset(
    seed: int | None = None,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    reasoning_language: str = "EN",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load OneIG-Bench with ``category`` fixed to ``Text_Rendering``.

    Parameters
    ----------
    seed : int | None, optional
        Ignored; see :func:`setup_oneig_dataset`.
    fraction : float
        Fraction of the subset to use.
    train_sample_size : int | None
        Unused; train/val are dummy.
    test_sample_size : int | None
        Test sample size cap for the subset.
    reasoning_language : str
        Passed to :func:`setup_oneig_dataset`.

    Returns
    -------
    Tuple[Dataset, Dataset, Dataset]
        Dummy train, dummy val, and test split for this subset.
    """
    return _setup_oneig_subset_with_fixed_category(
        "Text_Rendering",
        seed,
        fraction,
        train_sample_size,
        test_sample_size,
        reasoning_language,
    )


def setup_gedit_dataset(
    seed: int | None = None,
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
    seed : int | None, optional
        Ignored; test order is deterministic. If not None, a warning is logged.
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
    _warn_ignored_benchmark_seed(seed, dataset="GEditBench")
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
    seed: int | None = None,
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
    seed : int | None, optional
        Ignored; test order is deterministic. If not None, a warning is logged.
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
    _warn_ignored_benchmark_seed(seed, dataset="DPG")
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
