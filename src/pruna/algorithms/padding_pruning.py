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

from __future__ import annotations

import functools
import inspect
from collections.abc import Iterable
from typing import Any

from ConfigSpace import OrdinalHyperparameter

from pruna.algorithms.base.pruna_base import PrunaAlgorithmBase
from pruna.algorithms.base.tags import AlgorithmTag
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_diffusers_model
from pruna.engine.save import SAVE_FUNCTIONS


class PaddingPruner(PrunaAlgorithmBase):
    """
    Implement Padding Pruning for Diffusers pipelines.

    Padding Pruning removes unused padding tokens from the prompt embedding of diffusers pipelines.
    """

    algorithm_name: str = "padding_pruning"
    group_tags: list[AlgorithmTag] = [AlgorithmTag.PRUNER]
    references: dict[str, str] = {}
    tokenizer_required: bool = True
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda", "accelerate"]
    dataset_required: bool = False
    save_fn = SAVE_FUNCTIONS.reapply
    compatible_before: Iterable[str | AlgorithmTag] = ["qkv_diffusers"]
    compatible_after: Iterable[str | AlgorithmTag] = [
        AlgorithmTag.CACHER,
        "hyper",
        "torch_compile",
        "stable_fast",
        "hqq_diffusers",
        "diffusers_int8",
        "torchao",
        "flash_attn3",
        "ring_attn",
    ]

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for the Prompt Pruner.

        Returns
        -------
        list
            A list of hyperparameters.
        """
        return [
            OrdinalHyperparameter(
                "min_sequence_length",
                sequence=[32, 64, 128, 256],
                default_value=64,
                meta=dict(desc="Minimum sequence length used to embed a prompt."),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a diffusers pipeline with a max_sequence_length parameter.

        Parameters
        ----------
        model : Any
            The model instance to check.

        Returns
        -------
        bool
            True if the model is a diffusers pipeline with a max_sequence_length parameter.
        """
        if not is_diffusers_model(model):
            return False
        signature = inspect.signature(model.__call__)
        return "max_sequence_length" in signature.parameters

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Apply Prompt Pruning to the pipeline.

        Parameters
        ----------
        model : Any
            The pipeline to apply prompt pruning to.
        smash_config : SmashConfigPrefixWrapper
            Configuration settings for the pruning.

        Returns
        -------
        Any
            The pipeline with Prompt Pruning enabled.
        """
        min_sequence_length = smash_config["min_sequence_length"]
        model.padding_pruning_helper = PaddingPruningHelper(model, min_sequence_length, smash_config.tokenizer)
        model.padding_pruning_helper.enable()
        return model

    def import_algorithm_packages(self) -> dict[str, Any]:
        """
        Import necessary algorithm packages.

        Returns
        -------
        dict
            An empty dictionary as no packages are imported in this implementation.
        """
        return dict()


class PaddingPruningHelper:
    """
    Helper for Prompt Pruning.

    Parameters
    ----------
    pipe : Any
        The diffusers pipeline to wrap.
    min_tokens : int
        The minimum number of tokens to embed a prompt.
    tokenizer : Any
        The tokenizer of the pipeline.
    """

    def __init__(self, pipe: Any, min_tokens: int, tokenizer: Any) -> None:
        self.pipe = pipe
        self.min_tokens = min_tokens
        self.tokenizer = tokenizer

    def enable(self) -> None:
        """Enable prompt pruning by wrapping the pipe."""
        self.wrap_pipe(self.pipe)

    def disable(self) -> None:
        """Disable prompt pruning by unwrapping the pipe."""
        if self.pipe_call:
            self.pipe.__call__ = self.pipe_call

    def wrap_pipe(self, pipe: Any) -> None:
        """
        Wrap the call method of the pipe to adjust the max sequence length.

        Parameters
        ----------
        pipe : Any
            The diffusers pipeline to wrap.
        """
        pipe_call = pipe.__call__
        self.pipe_call = pipe_call
        signature = inspect.signature(pipe_call)
        default_max_sequence_length = signature.parameters["max_sequence_length"].default

        @functools.wraps(pipe_call)
        def wrapped_call(*args, **kwargs):  # noqa: ANN201
            # while a natural approach would be to remove all padding tokens,
            # we found this to degrade the quality of the generated images
            # for this reason, we usually round to the nearest order of two
            # and use this as the max sequence length

            # the min_tokens parameter controls the minimum for the max sequence length
            min_sequence_length = self.min_tokens
            # we use the default value as the maximum value for the max sequence length
            max_sequence_length = kwargs.get("max_sequence_length", default_max_sequence_length)

            prompts = self._extract_prompts(args, kwargs)
            max_num_tokens = max(len(self.tokenizer.encode(p)) for p in prompts)

            sequence_length = min_sequence_length
            while max_num_tokens > sequence_length:
                sequence_length *= 2
            if sequence_length >= max_sequence_length:
                sequence_length = max_sequence_length
            kwargs["max_sequence_length"] = sequence_length
            return pipe_call(*args, **kwargs)

        pipe.__call__ = wrapped_call

    def _extract_prompts(self, args: Any, kwargs: Any) -> list[str]:
        """Extract the prompts from the args and kwargs of the pipe call."""
        prompts: list[str] = []

        # the first arguments of diffusers pipelines are usually the prompts
        for arg in args:
            if isinstance(arg, str):
                prompts.append(arg)
            elif isinstance(arg, list):
                if len(arg) > 0 and isinstance(arg[0], str):
                    prompts.extend(arg)
            else:
                break

        for kwarg in kwargs:
            if kwarg.startswith("prompt"):
                prompt = kwargs[kwarg]
                if isinstance(prompt, str):
                    prompts.append(prompt)
                elif isinstance(prompt, list):
                    prompts.extend(prompt)
            if kwarg.startswith("negative_prompt"):
                negative_prompt = kwargs[kwarg]
                if isinstance(negative_prompt, str):
                    prompts.append(negative_prompt)
                elif isinstance(negative_prompt, list):
                    prompts.extend(negative_prompt)
        return prompts
