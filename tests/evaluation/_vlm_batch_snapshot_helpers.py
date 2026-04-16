"""Test-only helpers: placeholder ``pred`` tensors from aux + JSON snapshot records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.vlm_utils import pil_rgb_from_aux_image_bytes


def pred_tensor_from_auxiliaries(
    auxiliaries: list[Any],
    size: int = 224,
    *,
    require_source_image: bool = False,
) -> torch.Tensor:
    """Build a float pred batch from aux dicts (tests only; uses :func:`pil_rgb_from_aux_image_bytes`)."""
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


@dataclass(frozen=True)
class BenchmarkVlmBatchOutcome:
    """Minimal batch + metric result for snapshot tests."""

    result: MetricResult
    prompts: list[Any]
    auxiliaries: list[Any]
    pred: torch.Tensor


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
    """Build a JSON-serializable snapshot (used only in tests)."""
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
