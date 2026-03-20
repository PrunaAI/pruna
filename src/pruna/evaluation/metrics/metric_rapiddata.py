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
from pathlib import Path
from typing import Any, List, Literal

import PIL.Image
import torch
from rapidata import RapidataClient
from rapidata.rapidata_client.benchmark.rapidata_benchmark import RapidataBenchmark
from torch import Tensor
from torchvision.utils import save_image

from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.metrics.async_mixin import AsyncEvaluationMixin
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.result import CompositeMetricResult
from pruna.evaluation.metrics.utils import PAIRWISE, SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.logging.logger import pruna_logger

METRIC_RAPIDATA = "rapidata"


# We don't use the MetricRegistry here
# because we need to instantiate the Metric directly with benchmark and leaderboards.
class RapidataMetric(StatefulMetric, AsyncEvaluationMixin):
    """
    Evaluate models with human feedback via the Rapidata platform.

    Parameters
    ----------
    call_type : str
        How to extract inputs from (x, gt, outputs). Default is "single".
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
    default_call_type: str = "x_y"
    higher_is_better: bool = True
    metric_name: str = METRIC_RAPIDATA
    runs_on: List[str] = ["cpu", "cuda"]

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
        self.current_benchmarked_model: str | None = None

    @classmethod
    def from_benchmark(
        cls,
        benchmark: RapidataBenchmark,
        rapidata_client_id: str | None = None,
        rapidata_client_secret: str | None = None
    ) -> RapidataMetric:
        """
        Create a RapidataMetric from an existing RapidataBenchmark.

        Parameters
        ----------
        benchmark : RapidataBenchmark
            The benchmark to attach to.
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
        metric.benchmark = benchmark
        return metric

    @classmethod
    def from_benchmark_id(
        cls,
        benchmark_id: str,
        rapidata_client_id: str | None = None,
        rapidata_client_secret: str | None = None,
    ) -> RapidataMetric:
        """
        Create a RapidataMetric from an existing benchmark ID.

        Parameters
        ----------
        benchmark_id : str
            The ID of the benchmark on the Rapidata platform.
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
        metric.benchmark = metric.client.mri.get_benchmark_by_id(benchmark_id)
        return metric

    def create_benchmark(
        self,
        name: str,
        data: list[str] | PrunaDataModule,
        split: Literal["test", "val", "train"] = "test",
        **kwargs,
    ) -> None:
        """
        Register a new benchmark on the Rapidata platform.

        The benchmark defines the prompt pool. Any data submitted to
        leaderboards later must be drawn from this pool.

        Parameters
        ----------
        name : str
            The name of the benchmark.
        data : list[str] | PrunaDataModule
            The prompts or dataset to benchmark against.
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
            else:
                raise ValueError(
                    "Could not extract prompts from dataset.\n "
                    "Expected a 'text' column. Please use a suitable dataset from Pruna \
                    or pass a list[str] directly instead."
                )

        self.benchmark = self.client.mri.create_new_benchmark(name, prompts=data, **kwargs)

    def create_request(
        self,
        name: str,
        instruction: str,
        show_prompt: bool = False,
        **kwargs,
    ) -> None:
        """
        Add a leaderboard (evaluation criterion) to the benchmark.

        Each leaderboard defines a single instruction that human raters see
        when comparing model outputs (e.g. "Which image has higher quality?"
        or "Which image is more aligned with the prompt?").

        You can create multiple leaderboards to evaluate different quality dimensions.
        Must be called after :meth:`create_benchmark` (or after attaching a
        benchmark via :meth:`from_benchmark` / :meth:`from_benchmark_id`).

        Parameters
        ----------
        name : str
            The name of the leaderboard.
        instruction : str
            The evaluation instruction shown to human raters.
        show_prompt : bool, optional
            Whether to show the prompt to raters. Default is False.
        **kwargs : Any
            Additional keyword arguments passed to the Rapidata API.
        """
        self._require_benchmark()
        self.benchmark.create_leaderboard(name, instruction, show_prompt, **kwargs)

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
        self.current_benchmarked_model = model_name
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
        self._require_model()
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
        self._require_model()
        if not self.media_cache:
            raise ValueError("No data accumulated. Call update() before compute().")

        media = self._prepare_media_for_upload()

        #  Ignoring the type error because _require_model() has already been called, but ty can't see it.
        self.benchmark.evaluate_model(
            self.current_benchmarked_model,  # type: ignore[arg-type]
            media=media,
            prompts=self.prompt_cache,
        )

        self._cleanup_temp_media()

        pruna_logger.info(
            "Sent evaluation request for model '%s' to Rapidata.\n "
            "It may take a while to collect votes from human raters.\n "
            "Use retrieve_results() to check scores later, "
            "or monitor progress at: "
            "https://app.rapidata.ai/mri/benchmarks/%s",
            self.current_benchmarked_model,
            self.benchmark.id,
        )

    def retrieve_results(self, *args, **kwargs) -> CompositeMetricResult | None:
        """
        Retrieve aggregated standings across all leaderboards.

        Parameters
        ----------
        *args : Any
            Additional arguments passed to the Rapidata API.
        **kwargs : Any
            Additional keyword arguments passed to the Rapidata API.

        Returns
        -------
        CompositeMetricResult | None
            The overall standings, or None if not enough votes yet.
        """
        self._require_benchmark()

        try:
            standings = self.benchmark.get_overall_standings(*args, **kwargs)
        except Exception as e:
            if "ValidationError" in type(e).__name__:
                pruna_logger.warning(
                    "The benchmark hasn't finished yet.\n "
                    "Please wait for more votes and try again.\n "
                    "Skipping."
                )
                return None
            raise

        scores = dict(zip(standings["name"], standings["score"]))
        return CompositeMetricResult(
            name=self.metric_name,
            params={},
            result=scores,
            higher_is_better=self.higher_is_better,
        )

    def retrieve_granular_results(self, **kwargs) -> List[CompositeMetricResult]:
        """
        Retrieve per-leaderboard results.

        Each leaderboard produces a separate CompositeMetricResult containing
        scores for all evaluated models.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to the Rapidata API.

        Returns
        -------
        List[CompositeMetricResult]
            A list of results, one per leaderboard.
        """
        self._require_benchmark()

        results = []
        for leaderboard in self.benchmark.leaderboards:
            try:
                standings = leaderboard.get_standings(**kwargs)
            except Exception as e:
                if "ValidationError" in type(e).__name__:
                    pruna_logger.warning(
                        "Leaderboard '%s' does not have results yet.\n "
                        "Not enough votes have been collected. Skipping.",
                        leaderboard.name,
                    )
                    continue
                raise

            scores = dict(zip(standings["name"], standings["score"]))
            result = CompositeMetricResult(
                name=leaderboard.name,
                params={"instruction": leaderboard.instruction},
                result=scores,
                higher_is_better=not leaderboard.inverse_ranking,
            )
            results.append(result)
        return results

    def _require_benchmark(self) -> None:
        """Raise if no benchmark has been created or attached."""
        if self.benchmark is None:
            raise ValueError(
                "No benchmark configured. "
                "Call create_benchmark(), or use from_benchmark() / from_benchmark_id()."
            )

    def _require_model(self) -> None:
        """Raise if no model context has been set."""
        if self.current_benchmarked_model is None:
            raise ValueError(
                "No model set. Call set_current_context() first."
            )

    def _prepare_media_for_upload(self) -> list[str]:
        """
        Convert cached media to file paths that Rapidata can upload.

        Handles three cases:
        - str: assumed to be a URL or file path, passed through as-is
        - PIL.Image: saved to a temporary file
        - torch.Tensor: saved to a temporary file

        Returns
        -------
        list[str]
            A list of URLs or file paths.
        """
        self._temp_dir = Path(tempfile.mkdtemp(prefix="rapidata_"))
        media_paths = []

        for i, item in enumerate(self.media_cache):
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
