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

"""VLM benchmark helpers: discovery, one-batch metric runs, JSON records (tests and mine scripts).

Hosted VLM runs use ``vlm_type="litellm"`` with :data:`DEFAULT_LITELLM` unless you pass another
``model_name``. Set ``OPENAI_API_KEY`` or ``LITELLM_API_KEY`` for API calls (see
:mod:`pruna.evaluation.metrics.vlm_base`). User-facing overview:
:doc:`Evaluate a model </docs_pruna/user_manual/evaluate>` (Vision-language judge metrics).
``mine/`` scripts that use Replicate or other hosts document their own tokens and are separate from
the LiteLLM judge path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.benchmarks import TASK_TYPE_TEXT_PLUS_IMAGE_IMAGE, BenchmarkRegistry
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.vlm_base import VLM_METRIC_REGISTRY_NAMES, BaseVLM
from pruna.evaluation.metrics.vlm_utils import VLM_AUX_IMAGE_BYTES_KEY_ORDER, pil_rgb_from_aux_image_bytes

DEFAULT_SMOL = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_LITELLM = "openai/gpt-4o"

_CATEGORY_DEFAULTS: dict[str, dict[str, Any]] = {
    "GenEval": {"category": "single_object"},
    "ImgEdit": {"category": "replace"},
    "GEditBench": {"category": "background_change"},
}


def discover_vlm_benchmark_jobs(include_oneig_reasoning: bool) -> list[tuple[str, str, str]]:
    """
    List ``(lookup_key, benchmark display name, metric_name)`` for VLM-backed paper metrics.

    Parameters
    ----------
    include_oneig_reasoning : bool
        If True, append ``oneig_reasoning`` for OneIG Knowledge Reasoning (LLM2CLIP, not SmolVLM).

    Returns
    -------
    list[tuple[str, str, str]]
        Sorted jobs for benchmarks that declare at least one metric in
        :data:`VLM_METRIC_REGISTRY_NAMES`, plus optional reasoning jobs.
    """
    jobs: list[tuple[str, str, str]] = []
    for key in sorted(BenchmarkRegistry.list()):
        b = BenchmarkRegistry.get(key)
        for m in b.metrics:
            if m in VLM_METRIC_REGISTRY_NAMES:
                jobs.append((key, b.name, m))
        if include_oneig_reasoning and "oneig_reasoning" in b.metrics:
            tup = (key, b.name, "oneig_reasoning")
            if tup not in jobs:
                jobs.append(tup)
    return jobs


def make_random_pred_images(batch_size: int, size: int = 224) -> torch.Tensor:
    """
    Return a random RGB batch (placeholder generations for smoke integration).

    Parameters
    ----------
    batch_size : int
        Number of images in the batch dimension.
    size : int, optional
        Height and width of each square image (default 224).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch_size, 3, size, size)`` with values in ``[0, 1)``.
    """
    return torch.rand(batch_size, 3, size, size)


_IMAGE_BYTES_FIELD_NAMES: tuple[str, ...] = VLM_AUX_IMAGE_BYTES_KEY_ORDER


def _pred_from_auxiliaries(auxiliaries: list[Any], size: int = 224, require_source_image: bool = False) -> torch.Tensor:
    """
    Build a pred tensor from auxiliaries, using source image bytes when available.

    Parameters
    ----------
    auxiliaries : list[Any]
        Per-sample dicts from ``prompt_with_auxiliaries_collate``.
    size : int, optional
        Target square resolution (default 224).
    require_source_image : bool, optional
        Raise :exc:`ValueError` instead of using random noise when no image bytes found.

    Returns
    -------
    torch.Tensor
        Shape ``(len(auxiliaries), 3, size, size)`` with values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``require_source_image=True`` and any aux entry lacks decodable image bytes.
    """
    from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

    transform = Compose([Resize(size), CenterCrop(size), ToTensor()])
    tensors = []
    for aux in auxiliaries:
        if not isinstance(aux, dict):
            if require_source_image:
                raise ValueError("require_source_image=True but auxiliary entry is not a dict.")
            tensors.append(torch.rand(3, size, size))
            continue

        pil = pil_rgb_from_aux_image_bytes(aux, min_bytes_in_value_scan=0)
        if pil is not None:
            try:
                tensors.append(transform(pil))
                continue
            except Exception:
                pass

        if require_source_image:
            raise ValueError(f"require_source_image=True but no decodable image bytes found (keys: {list(aux.keys())}).")
        tensors.append(torch.rand(3, size, size))
    return torch.stack(tensors)


def build_vlm_benchmark_metric(
    metric_name: str,
    benchmark_key: str,
    *,
    vlm_type: str,
    model_name: str,
    device: str,
    vlm: BaseVLM | None = None,
) -> Any:
    """
    Instantiate a metric for one benchmark VLM job.

    Parameters
    ----------
    metric_name : str
        Registry metric name (e.g. ``qa_accuracy``).
    benchmark_key : str
        Benchmark lookup key matching ``PrunaDataModule`` (e.g. ``GenEval``).
    vlm_type : str
        ``litellm`` or ``transformers`` when ``vlm`` is None.
    model_name : str
        Model id when ``vlm`` is None.
    device : str
        Device for metrics and optional local VLM.
    vlm : BaseVLM | None
        Pre-built VLM to reuse (e.g. session fixture); skips loading weights again.

    Returns
    -------
    Any
        A :class:`~pruna.evaluation.metrics.metric_stateful.StatefulMetric` instance.
    """
    if metric_name == "oneig_reasoning":
        return MetricRegistry.get_metric(metric_name, device=device)
    kw: dict[str, Any] = {
        "vlm_type": vlm_type,
        "model_name": model_name,
        "device": device,
        "structured_output": True,
    }
    if vlm is not None:
        kw["vlm"] = vlm
    if metric_name == "qa_accuracy" and benchmark_key == "GenEval":
        kw["aggregation"] = "all_or_nothing"
    return MetricRegistry.get_metric(metric_name, **kw)


@dataclass(frozen=True)
class BenchmarkVlmBatchOutcome:
    """
    Outputs from a single benchmark row plus metric score.

    Parameters
    ----------
    result : MetricResult
        Aggregated metric output.
    prompts : list[Any]
        Prompt batch from the dataloader.
    auxiliaries : list[Any]
        Auxiliary fields per row (e.g. questions).
    pred : torch.Tensor
        Predicted image batch passed to the metric.
    """

    result: MetricResult
    prompts: list[Any]
    auxiliaries: list[Any]
    pred: torch.Tensor


def run_benchmark_vlm_batch_full(
    benchmark_key: str,
    metric_name: str,
    *,
    num_samples: int = 1,
    vlm_type: str = "transformers",
    model_name: str = DEFAULT_SMOL,
    device: str = "cpu",
    vlm: BaseVLM | None = None,
) -> BenchmarkVlmBatchOutcome:
    """
    Load ``num_samples`` test batches, run one VLM metric, return result and last batch tensors.

    Parameters
    ----------
    benchmark_key : str
        Dataset lookup key for :meth:`PrunaDataModule.from_string`.
    metric_name : str
        Registry metric name.
    num_samples : int, optional
        Number of dataset samples to evaluate (default 1). Each sample is a separate
        ``metric.update()`` call so state accumulates correctly across samples.
    vlm_type : str, optional
        ``litellm`` or ``transformers`` when ``vlm`` is None (default ``transformers``).
    model_name : str, optional
        Model id when ``vlm`` is None (default HuggingFace SmolVLM).
    device : str, optional
        Device string (default ``cpu``).
    vlm : BaseVLM | None, optional
        Pre-built VLM to reuse.

    Returns
    -------
    BenchmarkVlmBatchOutcome
        Aggregated result across all samples, plus prompts/auxiliaries/pred from the last batch.
    """
    dm_kw: dict[str, Any] = {"dataloader_args": {"batch_size": 1}}
    dm_kw.update(_CATEGORY_DEFAULTS.get(benchmark_key, {}))
    dm = PrunaDataModule.from_string(benchmark_key, **dm_kw)
    dm.limit_datasets(num_samples)
    metric = build_vlm_benchmark_metric(
        metric_name,
        benchmark_key,
        vlm_type=vlm_type,
        model_name=model_name,
        device=device,
        vlm=vlm,
    )
    is_edit_benchmark = BenchmarkRegistry.get(benchmark_key).task_type == TASK_TYPE_TEXT_PLUS_IMAGE_IMAGE
    last_prompts: list[Any] = []
    last_auxiliaries: list[Any] = []
    last_pred: torch.Tensor = make_random_pred_images(1)
    for prompts, auxiliaries in dm.test_dataloader():
        if auxiliaries:
            pred = _pred_from_auxiliaries(auxiliaries, require_source_image=is_edit_benchmark)
        else:
            pred = make_random_pred_images(len(prompts))
        metric.update(prompts, auxiliaries, pred)
        last_prompts, last_auxiliaries, last_pred = prompts, auxiliaries, pred
    mr = metric.compute()
    return BenchmarkVlmBatchOutcome(result=mr, prompts=last_prompts, auxiliaries=last_auxiliaries, pred=last_pred)


def run_benchmark_vlm_batch_with_pred(
    benchmark_key: str,
    metric_name: str,
    pred: torch.Tensor,
    *,
    vlm_type: str = "transformers",
    model_name: str = DEFAULT_SMOL,
    device: str = "cpu",
    vlm: BaseVLM | None = None,
) -> BenchmarkVlmBatchOutcome:
    """
    Load one test batch, score provided image batch with a VLM metric (no random placeholders).

    Parameters
    ----------
    benchmark_key : str
        Dataset lookup key for :meth:`PrunaDataModule.from_string`.
    metric_name : str
        Registry metric name.
    pred : torch.Tensor
        Predicted images, shape ``(N, 3, H, W)`` with values in ``[0, 1]`` (float) or ``[0, 255]``
        (handled by :mod:`~pruna.evaluation.metrics.vlm_utils`). ``N`` must match the batch size.
    vlm_type : str, optional
        ``litellm`` or ``transformers`` when ``vlm`` is None (default ``transformers``).
    model_name : str, optional
        Model id when ``vlm`` is None (default HuggingFace SmolVLM).
    device : str, optional
        Torch device string (default ``cpu``).
    vlm : BaseVLM | None, optional
        Pre-built VLM to reuse.

    Returns
    -------
    BenchmarkVlmBatchOutcome
        Result, prompts, auxiliaries, and the ``pred`` tensor passed in.

    Raises
    ------
    ValueError
        If ``pred`` batch dimension does not match the number of prompts in the batch.
    """
    dm_kw: dict[str, Any] = {"dataloader_args": {"batch_size": 1}}
    dm_kw.update(_CATEGORY_DEFAULTS.get(benchmark_key, {}))
    dm = PrunaDataModule.from_string(benchmark_key, **dm_kw)
    dm.limit_datasets(1)
    prompts, auxiliaries = next(iter(dm.test_dataloader()))
    if pred.shape[0] != len(prompts):
        raise ValueError(f"pred batch size {pred.shape[0]} does not match prompt batch size {len(prompts)}")
    metric = build_vlm_benchmark_metric(
        metric_name,
        benchmark_key,
        vlm_type=vlm_type,
        model_name=model_name,
        device=device,
        vlm=vlm,
    )
    metric.update(prompts, auxiliaries, pred)
    mr = metric.compute()
    return BenchmarkVlmBatchOutcome(result=mr, prompts=prompts, auxiliaries=auxiliaries, pred=pred)


def run_benchmark_vlm_multibatch_with_preds(
    benchmark_key: str,
    metric_name: str,
    preds: list[torch.Tensor],
    *,
    vlm_type: str = "transformers",
    model_name: str = DEFAULT_SMOL,
    device: str = "cpu",
    vlm: BaseVLM | None = None,
) -> BenchmarkVlmBatchOutcome:
    """
    Score N pre-built pred tensors against N dataset batches, accumulating into one metric.

    Loads ``len(preds)`` batches from the dataset (one per pred tensor) and calls
    ``metric.update()`` once per batch so state accumulates correctly before ``compute()``.

    Parameters
    ----------
    benchmark_key : str
        Dataset lookup key for :meth:`PrunaDataModule.from_string`.
    metric_name : str
        Registry metric name.
    preds : list of torch.Tensor
        Pre-built predicted image tensors, one per dataset sample. Each tensor must
        have shape ``(1, 3, H, W)`` to match ``batch_size=1``.
    vlm_type : str, optional
        ``litellm`` or ``transformers`` when ``vlm`` is None.
    model_name : str, optional
        Model id when ``vlm`` is None.
    device : str, optional
        Torch device string.
    vlm : BaseVLM | None, optional
        Pre-built VLM to reuse.

    Returns
    -------
    BenchmarkVlmBatchOutcome
        Aggregated result across all samples; prompts/auxiliaries/pred from the last batch.
    """
    n = len(preds)
    if n == 0:
        raise ValueError("preds must contain at least one tensor")
    dm_kw: dict[str, Any] = {"dataloader_args": {"batch_size": 1}}
    dm_kw.update(_CATEGORY_DEFAULTS.get(benchmark_key, {}))
    dm = PrunaDataModule.from_string(benchmark_key, **dm_kw)
    dm.limit_datasets(n)
    metric = build_vlm_benchmark_metric(
        metric_name,
        benchmark_key,
        vlm_type=vlm_type,
        model_name=model_name,
        device=device,
        vlm=vlm,
    )
    last_prompts: list[Any] = []
    last_auxiliaries: list[Any] = []
    for i, (prompts, auxiliaries) in enumerate(dm.test_dataloader()):
        if i >= n:
            break
        pred_i = preds[i]
        if pred_i.shape[0] != len(prompts):
            raise ValueError(f"preds[{i}] batch size {pred_i.shape[0]} does not match prompt batch size {len(prompts)}")
        metric.update(prompts, auxiliaries, pred_i)
        last_prompts, last_auxiliaries = prompts, auxiliaries
    mr = metric.compute()
    return BenchmarkVlmBatchOutcome(result=mr, prompts=last_prompts, auxiliaries=last_auxiliaries, pred=preds[-1])


def run_benchmark_metric_batch(
    benchmark_key: str,
    metric_name: str,
    *,
    vlm_type: str = "transformers",
    model_name: str = DEFAULT_SMOL,
    device: str = "cpu",
    vlm: BaseVLM | None = None,
) -> MetricResult:
    """
    Load one test batch from the benchmark, run one VLM metric, return :class:`MetricResult`.

    Uses random ``pred`` tensors as placeholder generations (same as the ``mine`` store script).

    Parameters
    ----------
    benchmark_key : str
        Dataset name for :meth:`PrunaDataModule.from_string`.
    metric_name : str
        Metric to run.
    vlm_type : str
        Backend when ``vlm`` is not provided.
    model_name : str
        Checkpoint or litellm id when ``vlm`` is not provided.
    device : str
        Torch device string.
    vlm : BaseVLM | None
        Optional shared VLM instance for faster multi-benchmark runs.

    Returns
    -------
    MetricResult
        Aggregated score from :meth:`~pruna.evaluation.metrics.metric_stateful.StatefulMetric.compute`.
    """
    return run_benchmark_vlm_batch_full(
        benchmark_key,
        metric_name,
        vlm_type=vlm_type,
        model_name=model_name,
        device=device,
        vlm=vlm,
    ).result


def _short(obj: Any, max_len: int = 400) -> Any:
    if isinstance(obj, str) and len(obj) > max_len:
        return obj[:max_len] + "…"
    return obj


def _question_value_for_record(qt: Any, max_len: int = 200) -> Any:
    r"""Serialize a single question label for JSON; keep dataset padding as null, not the string ``\"None\"``."""
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


def _safe_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return {"bytes_len": len(obj)}
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
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
    Build a JSON-serializable snapshot of one benchmark batch, preds, and metric output.

    Parameters
    ----------
    outcome : BenchmarkVlmBatchOutcome
        Batch prompts, auxiliaries, ``pred`` tensor, and computed :class:`MetricResult`.
    benchmark_key : str
        Registry / datamodule lookup key (e.g. ``GenEval``).
    benchmark_name : str
        Human-readable benchmark name.
    metric_name : str
        Metric id used for this run.
    vlm_type : str
        Backend id (e.g. ``transformers``).
    model_name : str
        Model id or litellm route.
    device : str
        Torch device string.
    pred_note : str | None, optional
        Short note stored next to ``pred`` shape (placeholder generations in integration).

    Returns
    -------
    dict[str, Any]
        Nested dict safe for ``json.dumps`` (strings truncated; tensors summarized).

    Examples
    --------
    >>> from pruna.evaluation.metrics.result import MetricResult
    >>> import torch
    >>> mr = MetricResult(name="m", params={}, result=1.0, higher_is_better=True)
    >>> bo = BenchmarkVlmBatchOutcome(
    ...     result=mr,
    ...     prompts=["hi"],
    ...     auxiliaries=[{}],
    ...     pred=torch.zeros(1, 3, 2, 2),
    ... )
    >>> rec = vlm_benchmark_batch_to_json_record(
    ...     bo,
    ...     benchmark_key="K",
    ...     benchmark_name="K",
    ...     metric_name="m",
    ...     vlm_type="transformers",
    ...     model_name="x",
    ...     device="cpu",
    ... )
    >>> rec["metric_result"]["result"]
    1.0
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
            "auxiliary_0": _aux_for_record(a0) if isinstance(a0, dict) else _safe_json(a0),
        },
        "pred": pred_payload,
        "metric_result": _metric_result_record(outcome.result),
    }
    return _safe_json(record)
