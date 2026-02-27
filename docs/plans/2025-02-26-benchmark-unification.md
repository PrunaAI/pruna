# Benchmark PR Unification Plan

> **Goal:** Align all benchmark datasets with PartiPrompts. Minimize per-dataset code by extracting shared logic into a single helper. Same underlying flow everywhere.

**Principle:** PartiPrompts is the reference. Every dataset uses the same signature, same sampling logic, and the same `_prepare_test_only_prompt_dataset` helper.

---

## PartiPrompts reference pattern

```python
def setup_parti_prompts_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: PartiCategory | list[PartiCategory] | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    ds = load_dataset("nateraw/parti-prompts")["train"]
    if category is not None:
        categories = [category] if not isinstance(category, list) else category
        ds = ds.filter(lambda x: x["Category"] in categories or x["Challenge"] in categories)
    test_sample_size = define_sample_size_for_dataset(ds, fraction, test_sample_size)
    ds = ds.select(range(min(test_sample_size, len(ds))))
    ds = ds.rename_column("Prompt", "text")
    return _prepare_test_only_prompt_dataset(ds, seed, "PartiPrompts")
```

---

## 1. Shared helper in `pruna/data/utils.py`

**Add:**

```python
def _prepare_test_only_prompt_dataset(
    ds: Dataset,
    seed: int,
    dataset_name: str,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Shared tail for test-only prompt datasets: shuffle, return dummy train/val + test.
    All benchmark datasets use this.
    """
    ds = ds.shuffle(seed=seed)
    pruna_logger.info(f"{dataset_name} is a test-only dataset. Do not use it for training or validation.")
    return ds.select([0]), ds.select([0]), ds
```

**Effect:** Removes repeated `ds.shuffle(seed=seed); pruna_logger.info(...); return ds.select([0]), ds.select([0]), ds` from every setup function.

---

## 2. Unified signature (match PartiPrompts)

**All prompt/benchmark datasets** use the same param names, but **each dataset has its own Literal type** for `category`:

```python
# PartiPrompts (existing)
PartiCategory = Literal["Abstract", "Animals", ...]

# HPS
HPSCategory = Literal["anime", "concept-art", "paintings", "photo"]

# GenEval
GenEvalCategory = Literal["single_object", "two_object", "counting", "colors", "position", "color_attr"]

# ImgEdit
ImgEditCategory = Literal["replace", "add", "remove", "adjust", "extract", "style", "background", "compose"]

# GEditBench
GEditBenchCategory = Literal["background_change", ...]  # (full list per dataset)

# DPG
DPGCategory = Literal["entity", "attribute", "relation", "global", "other"]

# OneIG
OneIGCategory = Literal["text_rendering", "portrait_alignment"]
```

**Signature per dataset:**

```python
def setup_*_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: XCategory | list[XCategory] | None = None,  # dataset-specific Literal
) -> Tuple[Dataset, Dataset, Dataset]:
```

**Replace `num_samples` with `test_sample_size`** and use `define_sample_size_for_dataset(ds, fraction, test_sample_size)` everywhere.

**OneIG:** Use `category` with `OneIGCategory` instead of `subset` (align naming).

---

## 3. Per-dataset: minimal divergence

Each dataset keeps only:

1. **Load** – dataset-specific (HF, URL, JSON)
2. **Filter** – if category: filter by dataset-specific column(s)
3. **Sample** – `define_sample_size_for_dataset` + `ds.select` (same as PartiPrompts)
4. **Rename** – ensure `text` column (if needed)
5. **Return** – `_prepare_test_only_prompt_dataset(ds, seed, "DatasetName")`

**Example – HPS aligned with PartiPrompts (Literal for categories):**

```python
from typing import Literal, get_args

HPSCategory = Literal["anime", "concept-art", "paintings", "photo"]

def setup_hps_dataset(
    seed: int,
    fraction: float = 1.0,
    train_sample_size: int | None = None,
    test_sample_size: int | None = None,
    category: HPSCategory | list[HPSCategory] | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    categories_to_load = list(get_args(HPSCategory)) if category is None else (
        [category] if not isinstance(category, list) else category
    )

    all_prompts = []
    for cat in categories_to_load:
        file_path = hf_hub_download("zhwang/HPDv2", f"{cat}.json", ...)
        with open(file_path, "r", encoding="utf-8") as f:
            for prompt in json.load(f):
                all_prompts.append({"text": prompt, "category": cat})

    ds = Dataset.from_list(all_prompts)
    test_sample_size = define_sample_size_for_dataset(ds, fraction, test_sample_size)
    ds = ds.select(range(min(test_sample_size, len(ds))))
    return _prepare_test_only_prompt_dataset(ds, seed, "HPS")
```

**Example – GenEval / ImgEdit / DPG / GEditBench:** Same pattern: load → filter by category → build Dataset → `define_sample_size_for_dataset` + select → `_prepare_test_only_prompt_dataset`.

---

## 4. Datamodule passthrough

**Extend `from_string`** to pass `fraction`, `train_sample_size`, `test_sample_size` when the setup fn accepts them:

```python
for param in ("fraction", "train_sample_size", "test_sample_size"):
    if param in inspect.signature(setup_fn).parameters:
        setup_fn = partial(setup_fn, **{param: locals()[param]})
```

**OneIG:** Use `category` for subsets; no separate `subset` param. If OneIG must keep `subset` for backward compat, add passthrough for it.

---

## 5. Benchmark category registry + single test

**Option A – derive from Literal:** Use `get_literal_values_from_param(setup_fn, "category")` to get categories from each setup function's Literal type. No separate registry for categories.

**Option B – explicit registry** (for aux_keys mapping):

```python
BENCHMARK_CATEGORY_CONFIG: dict[str, tuple[str, list[str]]] = {
    # (first_category_for_test, aux_keys_in_batch)
    "PartiPrompts": ("Animals", ["Category", "Challenge"]),
    "HPS": ("anime", ["category"]),
    "GenEval": ("counting", ["tag"]),
    "DPG": ("entity", ["category_broad"]),
    "ImgEdit": ("replace", ["category"]),
    "GEditBench": ("background_change", ["category"]),
    "OneIG": ("text_rendering", ["subset"]),  # or ["category"] if aligned
}
```

**Test params:** Use `get_literal_values_from_param` to get `(dataset_name, categories[0])` for each setup that has a Literal `category` param. Categories come from the function signature, not the registry.

**Single parametrized test** (replaces all per-dataset category tests):

```python
@pytest.mark.parametrize("dataset_name, category", [
    (name, cat) for name, (cat, _) in BENCHMARK_CATEGORY_CONFIG.items()
    if name in base_datasets
])
def test_benchmark_category_filter(dataset_name: str, category: str) -> None:
    dm = PrunaDataModule.from_string(dataset_name, category=category, dataloader_args={"batch_size": 4})
    dm.limit_datasets(10)
    batch = next(iter(dm.test_dataloader()))
    prompts, auxiliaries = batch
    assert len(prompts) == 4
    assert all(isinstance(p, str) for p in prompts)
    _, aux_keys = BENCHMARK_CATEGORY_CONFIG[dataset_name]
    assert all(any(aux.get(k) == category for k in aux_keys) for aux in auxiliaries)
```

**Remove:** `test_geneval_with_category_filter`, `test_hps_with_category_filter`, `test_dpg_with_category_filter`, `test_imgedit_with_category_filter`, `test_geditbench_with_category_filter`, `test_oneig_*` (category variants). **Keep:** `test_long_text_bench_auxiliaries` (no category).

---

## 6. Execution order

1. **PartiPrompts branch:** Add `_prepare_test_only_prompt_dataset` in utils, refactor PartiPrompts to use it, add `BENCHMARK_CATEGORY_CONFIG`, add `test_benchmark_category_filter`, extend datamodule passthrough.
2. **Merge PartiPrompts into each branch.**
3. **Per branch:** Refactor each dataset to use the shared helper + unified signature; add its entries to `BENCHMARK_CATEGORY_CONFIG`; remove per-dataset tests.

---

## Summary

| Change | Where | Effect |
|--------|-------|--------|
| `_prepare_test_only_prompt_dataset` | utils.py | Single return path for all benchmarks |
| Unified signature | prompt.py | Same params everywhere |
| `define_sample_size_for_dataset` | All setups | Same sampling logic as PartiPrompts |
| `BENCHMARK_CATEGORY_CONFIG` | __init__.py | One registry, one test |
| Datamodule passthrough | pruna_datamodule.py | fraction, test_sample_size forwarded |

**Result:** PartiPrompts-style flow everywhere, minimal per-dataset code, shared helper and tests. Each dataset keeps its specific categories as a Literal in the function signature (type-safe, discoverable via `get_literal_values_from_param`).
