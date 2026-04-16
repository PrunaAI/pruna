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

"""Helpers for mine Replicate end-to-end scripts and benchmark JSON records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.benchmarks import BenchmarkRegistry
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.vlm_base import BaseVLM, VLM_METRIC_REGISTRY_NAMES

DEFAULT_SMOL = "HuggingFaceTB/SmolVLM-256M-Instruct"

_BENCHMARK_CATEGORY_KW: dict[str, dict[str, Any]] = {
    "GenEval": {"category": "single_object"},
    "ImgEdit": {"category": "replace"},
    "GEditBench": {"category": "background_change"},
}


def make_random_pred_images(batch_size: int, size: int = 224) -> torch.Tensor:
    """Return a float tensor batch of random RGB images in ``[0, 1]``."""
    return torch.rand(batch_size, 3, size, size)


def discover_vlm_benchmark_jobs(*, include_oneig_reasoning: bool = True) -> list[tuple[str, str, str]]:
    """
    List ``(benchmark_lookup_key, benchmark_display_name, metric_name)`` for VLM-backed paper metrics.

    Parameters
    ----------
    include_oneig_reasoning
        When True, include ``oneig_reasoning`` rows from benchmarks that use it (not in
        :data:`~pruna.evaluation.metrics.vlm_base.VLM_METRIC_REGISTRY_NAMES`).

    Returns
    -------
    list[tuple[str, str, str]]
        Sorted benchmark keys with each applicable metric name.
    """
    jobs: list[tuple[str, str, str]] = []
    for key in sorted(BenchmarkRegistry.list()):
        b = BenchmarkRegistry.get(key)
        for m in b.metrics:
            if m in VLM_METRIC_REGISTRY_NAMES:
                jobs.append((key, b.name, m))
            elif include_oneig_reasoning and m == "oneig_reasoning":
                jobs.append((key, b.name, m))
    return jobs


@dataclass(frozen=True)
class BenchmarkVlmBatchOutcome:
    """Last batch snapshot plus aggregated :class:`MetricResult` after multibatch ``update`` calls."""

    result: MetricResult
    prompts: list[Any]
    auxiliaries: list[Any]
    pred: torch.Tensor


def _metric_init_kwargs(
    metric_name: str,
    *,
    vlm_type: Literal["litellm", "transformers"],
    model_name: str | None,
    device: str | None,
    vlm: BaseVLM | None,
) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "vlm_type": vlm_type,
        "model_name": model_name,
        "device": device,
        "structured_output": True,
    }
    if vlm is not None:
        kw["vlm"] = vlm
    if metric_name == "qa_accuracy":
        kw["aggregation"] = "all_or_nothing"
    return kw


def run_benchmark_vlm_multibatch_with_preds(
    benchmark_key: str,
    metric_name: str,
    preds: list[torch.Tensor],
    *,
    vlm_type: Literal["litellm", "transformers"],
    model_name: str | None,
    device: str | None,
    vlm: BaseVLM | None = None,
) -> BenchmarkVlmBatchOutcome:
    """
    Run a stateful VLM metric over several dataloader batches with caller-supplied ``pred`` tensors.

    Parameters
    ----------
    benchmark_key
        ``PrunaDataModule.from_string`` lookup key (e.g. ``\"GenEval\"``).
    metric_name
        Registry metric name (e.g. ``\"qa_accuracy\"``).
    preds
        One pred tensor per batch, aligned with ``dm.test_dataloader()`` order after ``limit_datasets``.
    vlm_type
        ``\"litellm\"`` or ``\"transformers\"``.
    model_name
        Passed to :func:`~pruna.evaluation.metrics.vlm_base.get_vlm` when ``vlm`` is None.
    device
        Device for local VLMs and registry metrics.
    vlm
        Optional pre-built VLM (used by replicate scripts for ``transformers`` to share one load).

    Returns
    -------
    BenchmarkVlmBatchOutcome
        Aggregated metric result and the last batch's prompts, aux, and pred for JSON snapshots.

    Raises
    ------
    ValueError
        If ``preds`` length does not match the number of dataloader batches.
    """
    dm_kw: dict[str, Any] = {"dataloader_args": {"batch_size": 1}}
    dm_kw.update(_BENCHMARK_CATEGORY_KW.get(benchmark_key, {}))
    dm = PrunaDataModule.from_string(benchmark_key, **dm_kw)
    dm.limit_datasets(len(preds))
    batches = list(dm.test_dataloader())
    if len(batches) != len(preds):
        raise ValueError(
            f"preds length {len(preds)} does not match dataloader batches {len(batches)} "
            f"for benchmark_key={benchmark_key!r}."
        )

    mkw = _metric_init_kwargs(
        metric_name,
        vlm_type=vlm_type,
        model_name=model_name,
        device=device,
        vlm=vlm,
    )
    metric = MetricRegistry.get_metric(metric_name, **mkw)

    last_prompts: list[Any] = []
    last_aux: list[Any] = []
    last_pred = preds[-1]

    for (prompts, aux), pred in zip(batches, preds, strict=True):
        metric.update(prompts, aux, pred)
        last_prompts = prompts
        last_aux = aux

    result = metric.compute()
    return BenchmarkVlmBatchOutcome(
        result=result,
        prompts=last_prompts,
        auxiliaries=last_aux,
        pred=last_pred,
    )


def _short(obj: Any, max_len: int = 400) -> Any:
    if isinstance(obj, str) and len(obj) > max_len:
        return obj[:max_len] + "…"
    return obj


def _question_value_for_record(qt: Any, max_len: int = 200) -> Any:
    if qt is None:
        return None
    if isinstance(qt, str):
        return _short(qt, max_len)
    return _short(str(qt), max_len)


def _aux_for_record(aux: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in aux.items():
        if k == "questions" and isinstance(v, dict):
            out[k] = {qk: _question_value_for_record(qt, 200) for qk, qt in list(v.items())[:24]}
            if len(v) > 24:
                out["_truncated_questions"] = len(v) - 24
        else:
            out[k] = _short(v) if isinstance(v, str) else v
    return out


def safe_json_for_snapshot(obj: Any) -> Any:
    """Recursively JSON-safe view (bytes → length, tensors → shape)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return {"bytes_len": len(obj)}
    if isinstance(obj, dict):
        return {str(k): safe_json_for_snapshot(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json_for_snapshot(x) for x in obj]
    if isinstance(obj, torch.Tensor):
        return {"tensor_shape": list(obj.shape), "dtype": str(obj.dtype)}
    return str(obj)


def _metric_result_record(mr: MetricResult) -> dict[str, Any]:
    return {
        "name": mr.name,
        "result": float(mr.result),
        "higher_is_better": mr.higher_is_better,
        "metric_units": mr.metric_units,
    }


def vlm_benchmark_batch_to_json_record(
    outcome: BenchmarkVlmBatchOutcome,
    *,
    benchmark_key: str,
    benchmark_name: str,
    metric_name: str,
    vlm_type: str,
    model_name: str,
    device: str,
    pred_note: str | None = "random noise placeholder",
) -> dict[str, Any]:
    """
    Build a JSON-serializable record for mine runs and reports.

    Parameters
    ----------
    outcome
        Batch outcome from :func:`run_benchmark_vlm_multibatch_with_preds`.
    benchmark_key
        Benchmark lookup key (e.g. ``\"GenEval\"``).
    benchmark_name
        Human-readable benchmark name from :class:`~pruna.evaluation.benchmarks.BenchmarkRegistry`.
    metric_name
        Metric registry name.
    vlm_type
        ``\"litellm\"`` or ``\"transformers\"``.
    model_name
        Model id passed to the metric.
    device
        Evaluation device label.
    pred_note
        Short description of how ``pred`` was produced (e.g. Replicate model id).

    Returns
    -------
    dict[str, Any]
        JSON-safe nested dict suitable for writing to a ``.json`` file.
    """
    a0 = outcome.auxiliaries[0] if outcome.auxiliaries and isinstance(outcome.auxiliaries[0], dict) else {}
    pred_payload: dict[str, Any] = {
        "shape": list(outcome.pred.shape),
        "dtype": str(outcome.pred.dtype),
    }
    if pred_note is not None:
        pred_payload["note"] = pred_note
    record: dict[str, Any] = {
        "benchmark_lookup_key": benchmark_key,
        "benchmark_name": benchmark_name,
        "metric_name": metric_name,
        "dataset_name": benchmark_key,
        "vlm_type": vlm_type,
        "model_name": model_name,
        "device": device,
        "inputs": {
            "prompts": [_short(p, 500) for p in outcome.prompts],
            "auxiliary_0": _aux_for_record(a0) if isinstance(a0, dict) else safe_json_for_snapshot(a0),
        },
        "pred": pred_payload,
        "metric_result": _metric_result_record(outcome.result),
    }
    return safe_json_for_snapshot(record)
