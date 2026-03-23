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

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, List, Literal

import PIL.Image
import torch
from rapidata import RapidataClient
from rapidata.rapidata_client.benchmark.rapidata_benchmark import RapidataBenchmark
from torch import Tensor
from torchvision.utils import save_image

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.metrics.async_mixin import AsyncEvaluationMixin
from pruna.evaluation.metrics.context_mixin import EvaluationContextMixin
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.result import CompositeMetricResult
from pruna.evaluation.metrics.utils import PAIRWISE, SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_RAPIDATA = "rapidata"


# We don't use the MetricRegistry here
# because we need to instantiate the Metric directly with benchmark and leaderboards.
class RapidataMetric(StatefulMetric, AsyncEvaluationMixin, EvaluationContextMixin):
    """
    Evaluate models with human feedback via the Rapidata platform https://www.rapidata.ai/.

    Parameters
    ----------
    call_type : str
        How to extract inputs from (x, gt, outputs). Only "single" is supported.
    client : RapidataClient | None
        The Rapidata client to use. If None, a new one is created.
    rapidata_client_id : str | None
        The client ID of the Rapidata client.
        If none, the credentials are read from the environment variable RAPIDATA_CLIENT_ID.
        If credentials are not found in the environment variable, you will be prompted to login via browser.
    rapidata_client_secret : str | None
        The client secret of the Rapidata client.
        If none, the credentials are read from the environment variable RAPIDATA_CLIENT_SECRET.
        If credentials are not found in the environment variable, you will be prompted to login via browser.
    *args :
        Additional arguments passed to StatefulMetric.
    **kwargs : Any
        Additional keyword arguments passed to StatefulMetric.

    Examples
    --------
    Standalone usage::
        metric = RapidataMetric()
        # OR metric = RapidataMetric.from_benchmark_id("69bc528fa858d3fbc1ea1475")

        metric.create_benchmark("my_bench", prompts)
        metric.create_request("Quality", instruction="Which image looks better?")

        metric.set_current_context("model_a")
        metric.update(prompts, ground_truths, outputs_a)
        metric.compute()

        metric.set_current_context("model_b")
        metric.update(prompts, ground_truths, outputs_b)
        metric.compute()

        # wait for human votes
        overall = metric.retrieve_results()
    """

    media_cache: List[torch.Tensor | PIL.Image.Image | str]
    prompt_cache: List[str]
    # With every metric higher is actually better,
    # Because for negative questions like "Which image has more errors?"
    # We create the leaderboard with inverse_ranking=True, which reverses the ranking.
    higher_is_better: bool = True
    default_call_type: str = "x_y"
    metric_name: str = METRIC_RAPIDATA

    def __init__(
        self,
        call_type: str = SINGLE,
        client: RapidataClient | None = None,
        rapidata_client_id: str | None = None,
        rapidata_client_secret: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.client = client or RapidataClient(
            client_id=rapidata_client_id,
            client_secret=rapidata_client_secret,
        )
        if call_type.startswith(PAIRWISE):
            raise ValueError("RapidataMetric does not support pairwise metrics. Use a single metric instead.")
        self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
        self.add_state("media_cache", default=[])
        self.add_state("prompt_cache", default=[])
        self.benchmark: RapidataBenchmark | None = None
        self.current_context: str | None = None

    @classmethod
    def from_rapidata_benchmark(
        cls,
        benchmark: RapidataBenchmark | str,
        rapidata_client_id: str | None = None,
        rapidata_client_secret: str | None = None
    ) -> RapidataMetric:
        """
        Create a RapidataMetric from an existing RapidataBenchmark.

        Parameters
        ----------
        benchmark : RapidataBenchmark | str
            The benchmark to attach to. Can be a RapidataBenchmark object or a string (benchmark ID).
        rapidata_client_id : str | None
            The client ID of the Rapidata client.
        rapidata_client_secret : str | None
            The client secret of the Rapidata client.

        Returns
        -------
        RapidataMetric
            The created metric.
        """
        metric = cls(
            rapidata_client_id=rapidata_client_id,
            rapidata_client_secret=rapidata_client_secret,
        )
        if isinstance(benchmark, RapidataBenchmark):
            metric.benchmark = benchmark
        elif isinstance(benchmark, str):
            metric.benchmark = metric.client.mri.get_benchmark_by_id(benchmark)
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}. Expected a RapidataBenchmark or a string.")
        return metric

    def create_benchmark(
        self,
        name: str,
        data: list[str] | PrunaDataModule,
        data_assets: list[str] | None = None,
        split: Literal["test", "val", "train"] = "test",
        **kwargs,
    ) -> None:
        """
        Register a new benchmark on the Rapidata platform.

        The benchmark defines the prompt pool. Any data submitted to
        leaderboards later must be drawn from this pool.

        Prompts can be provided as a list of strings or as a PrunaDataModule.
        When using a list of strings, you can optionally pass data_assets as a list of file paths or URLs.
        When using a PrunaDataModule, data assets are extracted automatically from the datamodule, if available.

        Parameters
        ----------
        name : str
            The name of the benchmark.
        data : list[str] | PrunaDataModule
            The prompts or dataset to benchmark against.
        data_assets : list[str] | None
            The assets to attach to the prompts.
            For instance, if you wish to benchmark an image editing model,
            you can pass the original images as data_assets.
        split : str, optional
            Which split to use when data is a PrunaDataModule. Default is "test".
        **kwargs : Any
            Additional keyword arguments passed to the Rapidata API.
        """
        if self.benchmark is not None:
            raise ValueError("Benchmark already created. Use from_benchmark() to attach to an existing one.")

        # Rapidata benchmarks only accept a list of string,
        # so we need to convert the PrunaDataModule to a list of strings.
        if isinstance(data, PrunaDataModule):
            split_map = {"test": data.test_dataset, "val": data.val_dataset, "train": data.train_dataset}
            dataset = split_map[split]
            # PrunaDataModule dataset loaders always renames the prompts column to "text"
            if hasattr(dataset, "column_names") and "text" in dataset.column_names:
                data = list(dataset["text"])
                data_assets = None  # When using a PrunaDataModule, we need to get the data assets from the datamodule.
                if "image" in dataset.column_names:
                    images = list(dataset["image"])  # Pruna text to image datasets always have an "image" column.
                    #  Rapidata only accepts file paths or URLs, so we need to convert the images to file paths.
                    data_assets = self._prepare_media_for_upload(images)
            else:
                raise ValueError(
                    "Could not extract prompts from dataset.\n "
                    "Expected a 'text' column. Please use a suitable dataset from Pruna \
                    or pass a list[str] directly instead."
                )

        self.benchmark = self.client.mri.create_new_benchmark(name, prompts=data, prompt_assets=data_assets, **kwargs)

    def create_async_request(
        self,
        name: str,
        instruction: str,
        show_prompt: bool = False,
        show_prompt_assets: bool = False,
        **kwargs,
    ) -> None:
        """
        Add a leaderboard (evaluation criterion) to the benchmark.

        Each leaderboard defines a single instruction that human raters see
        when comparing model outputs (e.g. "Which image has higher quality?"
        or "Which image is more aligned with the prompt?").

        You can create multiple leaderboards to evaluate different quality dimensions.
        Must be called after :meth:`create_benchmark` (or after attaching a
        benchmark via :meth:`from_rapidata_benchmark`).

        Parameters
        ----------
        name : str
            The name of the leaderboard.
        instruction : str
            The evaluation instruction shown to human raters.
        show_prompt : bool, optional
            Whether to show the prompt to raters. Default is False.
        show_prompt_assets : bool, optional
            Whether to show the prompt assets to raters. Default is False.
        **kwargs : Any
            Additional keyword arguments passed to the Rapidata API.
        """
        self._require_benchmark()
        self.benchmark.create_leaderboard(name, instruction, show_prompt, show_prompt_assets, **kwargs)

    def set_current_context(self, model_name: str, **kwargs) -> None:
        """
        Set which model is currently being evaluated.

        Call this before the :meth:`update` / :meth:`compute` cycle for each
        model. At least two models must be submitted before meaningful
        human comparison can begin.

        Parameters
        ----------
        model_name : str
            The name of the model to evaluate.
        **kwargs : Any
            Additional keyword arguments.
        """
        self.current_context = model_name
        self.reset()  # Clear the cache for the new model.

    def update(self, x: List[Any] | Tensor, gt: List[Any] | Tensor, outputs: Any) -> None:
        """
        Accumulate model outputs for the current model.

        Parameters
        ----------
        x : List[Any] | Tensor
            The input data (prompts).
        gt : List[Any] | Tensor
            The ground truth data.
        outputs : Any
            The model outputs (generated media).
        """
        self._require_benchmark()
        self._require_context()
        inputs = metric_data_processor(x, gt, outputs, self.call_type)
        self.prompt_cache.extend(inputs[0])
        self.media_cache.extend(inputs[1])

    def compute(self) -> None:
        """
        Submit the accumulated outputs for the current model to Rapidata.

        Converts cached media to uploadable file paths if necessary (saving tensors and
        PIL images to a temporary directory), submits them to the benchmark,
        and cleans up temporary files.

        This method does **not** return a result — human evaluation is
        asynchronous. Use :meth:`retrieve_results` or
        :meth:`retrieve_granular_results` once enough votes have been
        collected.
        """
        self._require_benchmark()
        self._require_context()
        if not self.media_cache:
            raise ValueError("No data accumulated. Call update() before compute().")

        media = self._prepare_media_for_upload()

        #  Ignoring the type error because _require_context() has already been called, but ty can't see it.
        self.benchmark.evaluate_model(
            self.current_context,  # type: ignore[arg-type]
            media=media,
            prompts=self.prompt_cache,
        )

        self._cleanup_temp_media()

        pruna_logger.warning(
            "Sent evaluation request for model '%s' to Rapidata.\n "
            "It may take a while to collect votes from human raters.\n "
            "Use retrieve_results() to check scores later, "
            "or monitor progress at: "
            "https://app.rapidata.ai/mri/benchmarks/%s",
            self.current_context,
            self.benchmark.id,
        )

    def on_context_change(self) -> None:
        """Reset the cache when the context changes."""
        self.reset()

    @staticmethod
    def _is_not_ready_error(exc: Exception) -> bool:
        """
        Search for a ValidationError in the exception chain.

        When the benchmark is not finished yet, the API throws a pydantic ValidationError
        we are catching it and returning None to indicate that the benchmark is not ready yet,
        rather than straight up failing with an exception.
        """
        return "ValidationError" in type(exc).__name__

    def _fetch_standings(self, api_call, *args, **kwargs):
        """
        Barebones API call wrapper that catches ValidationError and returns None if the benchmark is not ready yet.

        Since the core logic between the overall and granular standings is the same,
        we can use a single function to fetch the standings.

        Parameters
        ----------
        api_call : callable
            The API call to make.
        *args : Any
            Additional arguments passed to the API call.
        **kwargs : Any
            Additional keyword arguments passed to the API call.
        """
        try:
            return api_call(*args, **kwargs)
        except Exception as e:
            if not self._is_not_ready_error(e):
                raise
            return None

    def _fetch_overall_standings(self, *args, **kwargs) -> tuple[CompositeMetricResult | None, bool]:
        """
        Retrieve overall standings for the benchmark.

        Returns a tuple where the first element is the composite score of all leaderboards in the benchmark,
        and the second element is a boolean indicating whether the benchmark is finished yet.

        Parameters
        ----------
        *args : Any
            Additional arguments passed to the Rapidata API.
        **kwargs : Any
            Additional keyword arguments passed to the Rapidata API.

        Returns
        -------
        CompositeMetricResult | None
            The overall standings or None if the benchmark is not finished yet.
        """
        standings = self._fetch_standings(self.benchmark.get_overall_standings, *args, **kwargs)
        if standings is None:
            return None, False
        return CompositeMetricResult(
            name=self.metric_name,
            params={},
            result=dict(zip(standings["name"], standings["score"])),
            higher_is_better=self.higher_is_better,
        ), True

    def _fetch_granular_standings(self, *args, **kwargs) -> tuple[List[CompositeMetricResult] | None, bool]:
        """
        Retrieve standings for all leaderboards.

        Returns a tuple where the first element is a list of results, one per leaderboard,
        and the second element is a boolean indicating whether all of the leaderboards (the benchmark) is finished yet.

        Parameters
        ----------
        *args : Any
            Additional arguments passed to the Rapidata API.
        **kwargs : Any
            Additional keyword arguments passed to the Rapidata API.

        Returns
        -------
        List[CompositeMetricResult] | None
            A list of results, one per leaderboard, or None if the benchmark is not finished yet.
        """
        results = []
        all_finished = True
        for leaderboard in self.benchmark.leaderboards:
            standings = self._fetch_standings(leaderboard.get_standings, *args, **kwargs)
            if standings is None:
                all_finished = False
                continue
            results.append(CompositeMetricResult(
                name=leaderboard.name,
                params={"instruction": leaderboard.instruction},
                result=dict(zip(standings["name"], standings["score"])),
                higher_is_better=self.higher_is_better,
            ))
        return results, all_finished

    def _fetch_with_retry_option(
        self,
        fetch_fn: Callable,
        is_blocking: bool,
        timeout: float,
        poll_interval: float,
        *args,
        **kwargs,
    ) -> CompositeMetricResult | List[CompositeMetricResult] | None:
        """
        Wait for  the results or return whatever we have as is from the benchmark.

        If is_blocking is True, it will poll until the results are ready or the timeout is reached.
        If is_blocking is False, it will return the results immediately if they are ready,
        otherwise it will return None and log a warning.

        Parameters
        ----------
        fetch_fn : callable
            The function to fetch the standings from the benchmark.
        is_blocking : bool
            Whether to block and wait for the results to be ready.
        timeout : float
            The maximum time to wait for the results to be ready.
        poll_interval : float
            The interval in seconds to poll for the results.
        *args : Any
            Additional arguments passed to the Rapidata API.
        **kwargs : Any
            Additional keyword arguments passed to the Rapidata API.
        """
        deadline = time.monotonic() + timeout
        while True:
            result, is_finished = fetch_fn(*args, **kwargs)
            if is_finished:  # The benchmark is finished, we don't need to check anything else, just return the result.
                return result
            if not is_blocking:  # The benchmark is not finished yet, but the user doesn't want to keep on polling.
                pruna_logger.warning(
                    "The benchmark hasn't finished yet. "
                    "Please wait for more votes and try again."
                )
                return result  # Return whatever we have as is.
            if time.monotonic() + poll_interval > deadline:  # The timeout is reached, we raise an exception.
                raise TimeoutError(
                    f"Benchmark results not ready after {timeout:.0f}s. "
                    f"Monitor at: https://app.rapidata.ai/mri/benchmarks/{self.benchmark.id}"
                )
            pruna_logger.info("Results not ready yet, retrying in %ds...", poll_interval)
            time.sleep(poll_interval)

    def retrieve_async_results(
        self,
        is_granular: bool = False,
        is_blocking: bool = False,
        timeout: float = 3600,
        poll_interval: float = 30,
        *args,
        **kwargs,
    ) -> List[CompositeMetricResult] | CompositeMetricResult | None:
        """
        Retrieve standings from the benchmark.

        Parameters
        ----------
        is_granular : bool, optional
            If True, return per-leaderboard results (partial results
            are returned for any leaderboard that is ready).
            If False, return overall aggregated standings.
        is_blocking : bool, optional
            If True, poll until results are ready or *timeout* is reached.
        timeout : float, optional
            Maximum seconds to wait when blocking. Default is 3600.
        poll_interval : float, optional
            Seconds between polling attempts when blocking. Default is 30.
        *args : Any
            Additional arguments passed to the Rapidata API.
        **kwargs : Any
            Additional keyword arguments passed to the Rapidata API.

        Returns
        -------
        List[CompositeMetricResult] | CompositeMetricResult | None
            Granular returns a list (possibly partial), overall returns
            a single result or None if not ready.

        Raises
        ------
        TimeoutError
            If *is_blocking* is True and results are not ready within *timeout*.
        """
        self._require_benchmark()
        fetch_fn = self._fetch_granular_standings if is_granular else self._fetch_overall_standings
        return self._fetch_with_retry_option(fetch_fn, is_blocking, timeout, poll_interval, **kwargs)

    def _require_benchmark(self) -> None:
        """Raise if no benchmark has been created or attached."""
        if self.benchmark is None:
            raise ValueError(
                "No benchmark configured. "
                "Call create_benchmark(), or use from_benchmark() / from_benchmark_id()."
            )

    def _prepare_media_for_upload(self, media: list[torch.Tensor | PIL.Image.Image | str] | None = None) -> list[str]:
        """
        Convert cached media to file paths that Rapidata can upload.

        Handles three cases:
        - str: assumed to be a URL or file path, passed through as-is
        - PIL.Image: saved to a temporary file
        - torch.Tensor: saved to a temporary file

        Parameters
        ----------
        media : list[torch.Tensor | PIL.Image.Image | str] | None
            The media to prepare for upload. If None, the media cache is used.

        Returns
        -------
        list[str]
            A list of URLs or file paths.
        """
        self._temp_dir = Path(tempfile.mkdtemp(prefix="rapidata_"))
        media_paths = []

        for i, item in enumerate(media or self.media_cache):
            if isinstance(item, str):
                media_paths.append(item)
            elif isinstance(item, PIL.Image.Image):
                path = self._temp_dir / f"{i}.png"
                item.save(path)
                media_paths.append(str(path))
            elif isinstance(item, torch.Tensor):
                path = self._temp_dir / f"{i}.png"
                tensor = item.float()
                if tensor.max() > 1.0:
                    tensor = tensor / 255.0
                save_image(tensor, path)
                media_paths.append(str(path))
            else:
                raise TypeError(
                    f"Unsupported media type: {type(item)}. "
                    "Expected str (URL/path), PIL.Image, or torch.Tensor."
                )

        return media_paths

    def _cleanup_temp_media(self) -> None:
        """Remove temporary files created for upload."""
        if hasattr(self, "_temp_dir") and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
