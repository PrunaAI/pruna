"""Tests for OneIG-Bench prompt loading (Q_D graphs and reasoning ground truth)."""

from __future__ import annotations

import pytest

from pruna.data.datasets import prompt as prompt_mod


def test_oneig_needs_zh_multilingualism_hub() -> None:
    """ZH config is pulled only for full suite or when Multilingualism is requested."""
    assert prompt_mod._oneig_needs_zh_multilingualism_hub(None) is True
    assert prompt_mod._oneig_needs_zh_multilingualism_hub("Multilingualism") is True
    assert prompt_mod._oneig_needs_zh_multilingualism_hub("Portrait") is False
    assert prompt_mod._oneig_needs_zh_multilingualism_hub(["Portrait", "General_Object"]) is False
    assert prompt_mod._oneig_needs_zh_multilingualism_hub(["Portrait", "Multilingualism"]) is True


def test_oneig_qd_prefix_multilingualism() -> None:
    """Multilingualism maps to the only upstream stem ``multilingualism_zh``."""
    row = {"category": "Multilingualism", "id": "000", "prompt_en": "x", "class": "None"}
    assert prompt_mod._oneig_qd_prefix(row) == "multilingualism_zh"


def test_oneig_qd_prefix_anime_zh_hint() -> None:
    """Rows marked Chinese use ``anime_zh`` when category is anime/stylization."""
    row = {
        "category": "Anime_Stylization",
        "id": "001",
        "prompt_en": "hello",
        "class": "None",
        "language": "zh",
    }
    assert prompt_mod._oneig_qd_prefix(row) == "anime_zh"


def test_to_oneig_record_multilingualism_fills_questions() -> None:
    """Synthetic Multilingualism row resolves Q_D from merged index."""
    qb = {"multilingualism_zh_000": {"questions": {"1": "现场是不是颁奖典礼？"}, "dependencies": {"1": [0]}}}
    row = {"category": "Multilingualism", "id": "000", "prompt_en": " awards ", "class": "None"}
    rec = prompt_mod._to_oneig_record(row, qb, {}, {})
    assert rec["questions"]["1"] == "现场是不是颁奖典礼？"
    assert rec["dependencies"]["1"] == [0]


def test_to_oneig_record_knowledge_reasoning_gt() -> None:
    """Knowledge_Reasoning rows attach official-style gt strings by id."""
    row = {
        "category": "Knowledge_Reasoning",
        "id": "000",
        "prompt_en": "Peaks chart",
        "class": "geography",
    }
    gt_en = {"000": "The world's five tallest peaks are Mount Everest"}
    gt_zh = {"000": "中文答案"}
    rec = prompt_mod._to_oneig_record(row, {}, gt_en, gt_zh)
    assert rec["reasoning_gt_answer_en"] == gt_en["000"]
    assert rec["reasoning_gt_answer_zh"] == gt_zh["000"]
    assert rec["questions"] == {}


def test_to_oneig_record_prefers_prompt_over_prompt_en() -> None:
    """When ``prompt`` is set it wins for the unified ``text`` field."""
    row = {
        "category": "General_Object",
        "id": "000",
        "prompt": "native",
        "prompt_en": "english",
        "class": "None",
    }
    rec = prompt_mod._to_oneig_record(row, {}, {}, {})
    assert rec["text"] == "native"


def test_to_oneig_record_uses_prompt_cn_for_zh_hub_rows() -> None:
    """``OneIG-Bench-ZH`` Multilingualism rows expose Chinese text as ``prompt_cn``."""
    row = {"category": "Multilingualism", "id": "000", "prompt_cn": "中文提示", "class": "None"}
    rec = prompt_mod._to_oneig_record(row, {}, {}, {})
    assert rec["text"] == "中文提示"


@pytest.mark.slow
def test_setup_oneig_lazyloads_zh_hub_only_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Portrait-only loads ``OneIG-Bench``; Multilingualism also loads ``OneIG-Bench-ZH``."""
    from datasets import load_dataset as real_load_dataset

    loaded: list[str] = []

    def tracking_load(*args: object, **kwargs: object):
        name = args[1] if len(args) > 1 else kwargs.get("name")
        loaded.append(str(name))
        return real_load_dataset(*args, **kwargs)

    monkeypatch.setattr(prompt_mod, "load_dataset", tracking_load)

    prompt_mod.setup_oneig_dataset(category="Portrait", test_sample_size=1)
    assert loaded == ["OneIG-Bench"]

    loaded.clear()
    prompt_mod.setup_oneig_dataset(category="Multilingualism", test_sample_size=1)
    assert loaded == ["OneIG-Bench", "OneIG-Bench-ZH"]


@pytest.mark.slow
def test_setup_oneig_knowledge_reasoning_loads_remote_gt() -> None:
    """Integration: first reasoning sample has non-empty EN gt from the hub JSON."""
    _train, _val, test = prompt_mod.setup_oneig_dataset(category="Knowledge_Reasoning", test_sample_size=1)
    row = test[0]
    assert row["reasoning_gt_answer_en"]
    assert isinstance(row["reasoning_gt_answer_en"], str)
    assert len(row["reasoning_gt_answer_en"]) > 20
