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
from typing import Any, Callable, Dict, List, Optional, Tuple

from ConfigSpace import OrdinalHyperparameter

from pruna.algorithms.caching import PrunaCacher
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import is_flux_pipeline


class FORACacher(PrunaCacher):
    """
    Implement FORA for the Flux pipeline.

    FORA reuses the outputs of the transformer blocks for N steps before recomputing them.
    Different from the official implementation, this implementation exposes a start step parameter
    that allows to obtain a higher fidelity to the base model.
    """

    algorithm_name: str = "fora"
    references: dict[str, str] = {"Paper": "https://arxiv.org/abs/2407.01425"}
    tokenizer_required: bool = False
    processor_required: bool = False
    runs_on: list[str] = ["cpu", "cuda", "accelerate"]
    dataset_required: bool = False
    compatible_algorithms: dict[str, list[str]] = dict(
        compiler=["stable_fast", "torch_compile"],
        quantizer=["diffusers_int8", "hqq_diffusers", "torchao"],
        factorizer=["qkv_diffusers"],
    )

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for the FORA cacher.

        Returns
        -------
        list
            A list of hyperparameters.
        """
        return [
            OrdinalHyperparameter(
                "interval",
                sequence=range(1, 6),
                default_value=2,
                meta=dict(desc="Interval at which the outputs are computed. Higher is faster, but reduces quality."),
            ),
            OrdinalHyperparameter(
                "start_step",
                sequence=range(11),
                default_value=2,
                meta=dict(desc="How many steps to wait before starting to cache."),
            ),
        ]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the provided model is a Flux pipeline.

        Parameters
        ----------
        model : Any
            The model instance to check.

        Returns
        -------
        bool
            True if the model is a Flux pipeline, False otherwise.
        """
        return is_flux_pipeline(model)

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Apply FORA caching to the model.

        Parameters
        ----------
        model : Any
            The model to cache.
        smash_config : SmashConfigPrefixWrapper
            Configuration settings for the caching.

        Returns
        -------
        Any
            The model with FORA caching enabled.
        """
        model.cache_helper = CacheHelper(model, smash_config["interval"], smash_config["start_step"])
        model.cache_helper.enable()
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


class CacheHelper:
    """
    Cache helper for FORA.

    Parameters
    ----------
    pipe : Any
        The diffusers pipeline to wrap.
    interval : int
        The interval at which the transformer outputs are computed.
    start_step : int
        The step at which the interval caching starts.
    """

    def __init__(self, pipe: Any, interval: int, start_step: int) -> None:
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.interval = interval
        self.start_step = start_step

        # the cache schedule contains the caching decision for each step
        # 1 = compute outputs for step, 0 = reuse cached outputs
        self.cache_schedule: List[int] = []
        self.step: int = 0

        # Store original methods to be able to restore them later
        self.transformer_forward: Optional[Callable] = None
        self.pipe_call: Optional[Callable] = None
        self.double_stream_blocks_forward: Dict[int, Callable] = {}
        self.single_stream_blocks_forward: Dict[int, Callable] = {}

        # Use seperate caches for the two different transformer block types
        self.double_stream_blocks_cache: Dict[int, Tuple[Any, Any]] = {}
        self.single_stream_blocks_cache: Dict[int, Any] = {}

    def get_cache_schedule(self, num_steps: int) -> list[int]:
        """
        Get the cache schedule for the given number of steps.

        Parameters
        ----------
        num_steps : int
            The number of inference steps.

        Returns
        -------
        List[int]
            Cache schedule (1=compute step, 0=use cache).
        """
        cache_schedule = [1] * min(self.start_step, num_steps)
        num_remaining_steps = num_steps - self.start_step
        cache_schedule += [0] * num_remaining_steps
        for i in range(self.start_step, num_steps, self.interval):
            cache_schedule[i] = 1
        return cache_schedule

    def enable(self) -> None:
        """Enable caching by wrapping the pipe and transformer."""
        self.reset()
        self.wrap_pipe(self.pipe)
        self.wrap_transformer(self.transformer)
        for idx, block in enumerate(self.transformer.transformer_blocks):
            self.wrap_double_stream_block(block, idx)
        for idx, block in enumerate(self.transformer.single_transformer_blocks):
            self.wrap_single_stream_block(block, idx)

    def disable(self) -> None:
        """Disable caching by unwrapping the pipe and transformer."""
        if self.pipe_call:
            self.pipe.__call__ = self.pipe_call
        if self.transformer_forward:
            self.transformer.forward = self.transformer_forward
        for idx, block in enumerate(self.transformer.transformer_blocks):
            block.forward = self.double_stream_blocks_forward[idx]
        for idx, block in enumerate(self.transformer.single_transformer_blocks):
            block.forward = self.single_stream_blocks_forward[idx]
        self.reset()

    def set_params(self, interval: Optional[int] = None, start_step: Optional[int] = None) -> None:
        """
        Set caching parameters.

        Parameters
        ----------
        interval : Optional[int]
            The interval at which the transformer outputs are computed.
        start_step : Optional[int]
            The step at which caching starts.
        """
        if interval is not None:
            if not isinstance(interval, int):
                raise ValueError("Interval must be an integer.")
            if interval < 1:
                raise ValueError("Interval must be at least 1.")
            self.interval = interval
        if start_step is not None:
            if not isinstance(start_step, int):
                raise ValueError("start_step must be an integer.")
            if start_step < 0:
                raise ValueError("start_step must be non-negative.")
            self.start_step = start_step

    def wrap_pipe(self, pipe: Any) -> None:
        """
        Wrap the call method of the pipe to reset the cache and the step.

        Parameters
        ----------
        pipe : Any
            The diffusers pipeline to wrap.
        """
        pipe_call = pipe.__call__
        self.pipe_call = pipe_call

        @functools.wraps(pipe_call)
        def wrapped_call(*args, **kwargs):  # noqa: ANN201
            self.reset()
            return pipe_call(*args, **kwargs)

        pipe.__call__ = wrapped_call

    def wrap_transformer(self, transformer: Any) -> None:
        """
        Wrap the forward method of the transformer to manage timestep and to adjust schedule.

        Parameters
        ----------
        transformer : Any
            The transformer module of the Flux pipeline.
        """
        transformer_forward = transformer.forward
        self.transformer_forward = transformer_forward

        @functools.wraps(transformer_forward)
        def wrapped_forward(*args, **kwargs):  # noqa: ANN201
            num_steps = len(self.pipe.scheduler.timesteps)
            if self.step == 0:
                self.cache_schedule = self.get_cache_schedule(num_steps)
            result = transformer_forward(*args, **kwargs)
            self.step += 1
            return result

        transformer.forward = wrapped_forward

    def wrap_double_stream_block(self, block: Any, layer: int) -> None:
        """
        Wrap the forward method of a double stream transformer block.

        Parameters
        ----------
        block : Any
            The double stream transformer block to wrap.
        layer : int
            The layer of the double stream transformer block.
        """
        block_forward = block.forward
        self.double_stream_blocks_forward[layer] = block_forward

        @functools.wraps(block_forward)
        def wrapped_forward(*args, **kwargs):  # noqa: ANN201
            if self.cache_schedule[self.step]:
                output = self.double_stream_blocks_forward[layer](*args, **kwargs)
                self.double_stream_blocks_cache[layer] = output
                return output
            else:
                return self.double_stream_blocks_cache[layer]

        block.forward = wrapped_forward

    def wrap_single_stream_block(self, block: Any, layer: int) -> None:
        """
        Wrap the forward method of a single stream transformer block.

        Parameters
        ----------
        block : Any
            The single stream transformer block to wrap.
        layer : int
            The layer of the single stream transformer block.
        """
        block_forward = block.forward
        self.single_stream_blocks_forward[layer] = block_forward

        @functools.wraps(block_forward)
        def wrapped_forward(*args, **kwargs):  # noqa: ANN201
            if self.cache_schedule[self.step]:
                result = self.single_stream_blocks_forward[layer](*args, **kwargs)
                self.single_stream_blocks_cache[layer] = result
                return result
            else:
                return self.single_stream_blocks_cache[layer]

        block.forward = wrapped_forward

    def reset(self) -> None:
        """Clear the caches and reset the timestep."""
        self.double_stream_blocks_cache.clear()
        self.single_stream_blocks_cache.clear()
        self.step = 0
