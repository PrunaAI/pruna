#!/usr/bin/env python
"""Manual smoke checks for VLM metrics."""

from __future__ import annotations

import argparse
import os
import sys
from unittest.mock import MagicMock

import torch
from PIL import Image

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.evaluation.metrics.metric_vlm_utils import FloatOutput, VQAnswer, get_answer_from_response
from pruna.evaluation.metrics.metric_vqa import VQAMetric
from pruna.evaluation.metrics.vlm_base import BaseVLM, LitellmVLM, TransformersVLM
from pruna.evaluation.task import Task


def _dummy_image(size: int = 64) -> Image.Image:
    tensor = torch.rand(3, size, size)
    arr = (tensor.numpy() * 255).astype("uint8").transpose(1, 2, 0)
    return Image.fromarray(arr)


def run_stub_smoke() -> int:
    """Run a fast offline smoke test through the agent stateful path."""
    stub_vlm = MagicMock(spec=BaseVLM)
    stub_vlm.score.return_value = [1.0]
    metric = VQAMetric(vlm=stub_vlm, vlm_type="litellm", device="cpu")
    task = Task(request=[metric], datamodule=PrunaDataModule.from_string("LAION256"), device="cpu")
    agent = EvaluationAgent(task=task)
    agent.task.dataloader = [(["a cat"], torch.empty(0))]
    agent.device = "cpu"
    agent.device_map = None

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def run_inference(self, batch):
            return torch.rand(1, 3, 64, 64)

    agent.update_stateful_metrics(FakeModel(), agent.task.get_single_stateful_metrics(), [])
    results = agent.compute_stateful_metrics(agent.task.get_single_stateful_metrics(), [])
    if len(results) != 1 or results[0].result != 1.0:
        print("stub smoke failed", file=sys.stderr)
        return 1
    print("stub smoke ok:", results[0].name, results[0].result)
    return 0


def run_transformers_smoke(model_name: str) -> int:
    """Run a manual structured-output smoke check against a local transformers VLM."""
    vlm = TransformersVLM(model_name=model_name, device="cpu", use_outlines=True)
    response = vlm.generate([_dummy_image()], ["Answer yes or no: is there an object?"], response_format=VQAnswer)[0]
    print("raw:", response)
    print("parsed:", get_answer_from_response(response))
    return 0


def run_litellm_smoke(model_name: str) -> int:
    """Run a manual structured-output smoke check against the API backend."""
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("LITELLM_API_KEY"):
        print("OPENAI_API_KEY or LITELLM_API_KEY is required for --litellm", file=sys.stderr)
        return 1
    vlm = LitellmVLM(model_name=model_name)
    response = vlm.generate(
        [_dummy_image()],
        ["Score this image from 0 to 10 and respond with JSON."],
        response_format=FloatOutput,
    )[0]
    print("raw:", response)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual smoke checks for VLM metrics.")
    parser.add_argument("--stub", action="store_true", help="Run an offline stub smoke test.")
    parser.add_argument("--transformers", action="store_true", help="Run a local transformers smoke test.")
    parser.add_argument("--litellm", action="store_true", help="Run an API-backed litellm smoke test.")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct", help="Model name to use.")
    args = parser.parse_args()

    if args.stub:
        return run_stub_smoke()
    if args.transformers:
        return run_transformers_smoke(args.model)
    if args.litellm:
        return run_litellm_smoke(args.model)
    parser.error("Select one of --stub, --transformers, or --litellm")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
