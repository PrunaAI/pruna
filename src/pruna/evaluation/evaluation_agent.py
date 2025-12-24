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

import json
import tempfile
from pathlib import Path
from traceback import print_tb
from typing import Any, List, Literal

import torch
from torch import Tensor
from tqdm.auto import tqdm

from pruna.config.smash_config import SmashConfig
from pruna.config.utils import is_empty_config
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.data.utils import move_batch_to_device
from pruna.engine.pruna_model import PrunaModel
from pruna.engine.utils import get_device, move_to_device, safe_memory_cleanup, set_to_best_available_device
from pruna.evaluation.artifactsavers.utils import assign_artifact_saver
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import ensure_device_consistency, get_device_map, group_metrics_by_inheritance
from pruna.evaluation.task import Task
from pruna.logging.logger import pruna_logger

OUTPUT_DIR = tempfile.mkdtemp(prefix="inference_outputs")


class EvaluationAgent:
    """
    Entry point for evaluating a model.

    Parameters
    ----------
    task : Task, optional
        Configuration object that defines how to evaluate the model.
    request : str | List[str | BaseMetric | StatefulMetric], optional
        The user request to evaluate. Required if task is not provided.
    datamodule : PrunaDataModule, optional
        The dataloader to use for the evaluation. Required if task is not provided.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    save_artifacts : bool, optional
        Whether to save the artifacts. Default is False.
    root_dir : str | Path | None, optional
        The directory to save the artifacts. Default is None.
    num_samples_per_input : int, optional
        The number of samples to generate per input. Default is 1.
    seed_strategy : Literal["per_sample", "no_seed"], optional
        The seed strategy to use. Default is "no_seed".
    global_seed : int | None, optional
        The global seed to use. Default is None.
    saving_kwargs : dict, optional
        The kwargs to pass to the artifact saver. Default is an empty dict.
    """

    def __init__(
        self,
        task: Task | None = None,
        *,
        request: str | List[str | BaseMetric | StatefulMetric] | None = None,
        datamodule: PrunaDataModule | None = None,
        device: str | torch.device | None = None,
        save_artifacts: bool = False,
        root_dir: str | Path | None = None,
        num_samples_per_input: int = 1,
        seed_strategy: Literal["per_sample", "no_seed"] = "no_seed",
        global_seed: int | None = None,
        artifact_saver_export_format: str | None = None,
        save_in_out_metadata: bool = False,
        saving_kwargs: dict = dict(),
    ) -> None:
        if task is not None:
            if request is not None or datamodule is not None or device is not None:
                raise ValueError(
                    "Cannot specify both 'task' parameter and direct parameters (request, datamodule, device). "
                    "Use either the 'task' parameter or the new direct parameters."
                )
            self.task = task
        else:
            if request is None or datamodule is None:
                raise ValueError("When not using 'task' parameter, both 'request' and 'datamodule' must be provided.")
            self.task = Task(request=request, datamodule=datamodule, device=device)
        self.first_model_results: List[MetricResult] = []
        self.subsequent_model_results: List[MetricResult] = []
        self.seed_strategy = seed_strategy
        self.num_samples_per_input = num_samples_per_input
        self.global_seed = global_seed
        self.device = set_to_best_available_device(self.task.device)
        self.cache: List[Tensor] = []
        self.evaluation_for_first_model: bool = True
        self.save_in_out_metadata: bool = save_in_out_metadata
        self.save_artifacts: bool = save_artifacts
        if save_artifacts:
            self.root_dir = root_dir if root_dir is not None else OUTPUT_DIR
            self.artifact_saver = assign_artifact_saver(self.task.modality, self.root_dir, artifact_saver_export_format)
            # for miscellaneous saving kwargs like fps, etc.
            self.saving_kwargs = saving_kwargs

    def evaluate(self, model: Any) -> List[MetricResult]:
        """
        Evaluate models using different metric types.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.

        Returns
        -------
        List[MetricResult]
            The results of the model.
        """
        results = []
        model = self.prepare_model(model)

        # Separate metrics by execution strategy
        single_stateful_metrics = self.task.get_single_stateful_metrics()
        pairwise_metrics = self.task.get_pairwise_stateful_metrics()
        stateless_metrics = self.task.get_stateless_metrics()

        # Update and compute stateful metrics.
        pruna_logger.info("Evaluating stateful metrics.")
        with torch.no_grad():
            self.update_stateful_metrics(model, single_stateful_metrics, pairwise_metrics)
        results.extend(self.compute_stateful_metrics(single_stateful_metrics, pairwise_metrics))

        # Compute stateless metrics.
        pruna_logger.info("Evaluating isolated inference metrics.")
        results.extend(self.compute_stateless_metrics(model, stateless_metrics))

        # Move model back to the original device.
        move_to_device(model, self.device, device_map=self.device_map)
        pruna_logger.info(f"Evaluation run has finished. Moved model to {self.device}")

        safe_memory_cleanup()
        if self.evaluation_for_first_model:
            self.first_model_results = results
            self.evaluation_for_first_model = False
            if self.task.is_pairwise_evaluation():
                pruna_logger.info(
                    "The cache has been populated with the current model.\n"
                    "All future evaluations with this agent will use this cache to evaluate pairwise metrics."
                )
        else:
            self.subsequent_model_results = results
        return results

    def prepare_model(self, model: Any) -> PrunaModel:
        """
        Prepare the model for evaluation by wrapping it in a PrunaModel and moving it to the cpu if it is not already.

        Parameters
        ----------
        model : Any
            The model to evaluate.

        Returns
        -------
        PrunaModel
            The model.
        """
        if hasattr(model, "smash_config"):
            is_base = is_empty_config(model.smash_config)
            model_type = "base" if is_base else "smashed"
            pruna_logger.info(f"Evaluating a {model_type} model.")
            if not is_base and self.task.is_pairwise_evaluation() and self.evaluation_for_first_model:
                pruna_logger.warning(
                    "You have requested an evaluation task with pairwise metrics, \n"
                    "But the base model hasn't been evaluated yet. \n "
                    "Pairwise metrics will cache the smashed model outputs. \n"
                    "Ensure this is intentional, as typically the base model outputs are cached for comparison."
                )

        else:
            smash_config = SmashConfig(device=get_device(model))
            model = PrunaModel(model, smash_config=smash_config)
            pruna_logger.info("Evaluating a base model.")
            is_base = True

        model.inference_handler.log_model_info()
        if (
            "batch_size" in self.task.datamodule.dataloader_args
            and self.task.datamodule.dataloader_args["batch_size"] != model.smash_config.batch_size
            and not is_base
            and model.smash_config.is_batch_size_locked()
        ):
            pruna_logger.warning(
                "Batch size mismatch between evaluation datamodule and smashed model's smash config. "
                "This may lead to incorrect metric computation due to compression algorithms being batch size specific. "
                "Adjust the datamodule creation to match the smashed model's batch size, e.g., "
                "datamodule = PrunaDataModule.from_string(dataset_name, dataloader_args={'batch_size': %d})",
                model.smash_config.batch_size,
            )

        ensure_device_consistency(model, self.task)
        model_device = get_device(model)

        # The device map is set before smashing, so for the base models, we need to set it here.
        if model_device == "accelerate" and is_base and model.smash_config.device_map is None:
            model.smash_config.device_map = get_device_map(model)

        self.device = self.task.device
        # Keeping the device map to move model back to the original device, when the agent is finished.
        self.device_map = get_device_map(model)
        model.set_to_eval()
        #  Setup seeding for inference.
        model.inference_handler.configure_seed(self.seed_strategy, self.global_seed)

        return model

    def update_stateful_metrics(
        self, model: PrunaModel, single_stateful_metrics: List[StatefulMetric], pairwise_metrics: List[StatefulMetric]
    ) -> None:
        """
        Update stateful metrics.

        This method processes each batch of data by running inference on the model to obtain outputs.
        The outputs are then used to update both single and pairwise stateful metrics.

        - Single stateful metrics are updated as usual with the current batch outputs.
        - Pairwise metrics are only updated if the cache is already populated, ensuring that
        the necessary data from the first model is available for comparison.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        single_stateful_metrics : List[StatefulMetric]
            The single stateful metrics to update.
        pairwise_metrics : List[StatefulMetric]
            The pairwise metrics to update.
        """
        if not single_stateful_metrics and not pairwise_metrics:
            return

        move_to_device(model, self.device, device_map=self.device_map)
        for batch_idx, batch in enumerate(tqdm(self.task.dataloader, desc="Processing batches", unit="batch")):
            for sample_idx in range(self.num_samples_per_input):
                processed_outputs = model.run_inference(batch)
                if self.save_artifacts:
                    canonical_paths = []
                    # We have to save the artifacts for each sample in the batch.
                    for processed_output in processed_outputs:
                        canonical_path = self.artifact_saver.save_artifact(processed_output)
                        canonical_paths.append(canonical_path)
                    # Create aliases for the prompts if the user wants to save the artifacts with the prompt name.
                    # For doing that, the user needs to set the saving_kwargs["save_as_prompt_name"] to True.
                    self._maybe_create_input_output_metadata(batch, canonical_paths, sample_idx, batch_idx)

                batch = move_batch_to_device(batch, self.device)
                processed_outputs = move_batch_to_device(processed_outputs, self.device)
                (x, gt) = batch
                # Non-pairwise (aka single) metrics have regular update.
                for stateful_metric in single_stateful_metrics:
                    stateful_metric.update(x, gt, processed_outputs)
                    if self.save_artifacts and stateful_metric.create_alias:
                        # The evaluation agent saves the artifacts with a canonical filenaming convention.
                        # If the user wants to save the artifact with a different filename,
                        # here we give them the option to create an alias for the file.
                        for prompt_idx, prompt in enumerate(x):
                            if self.artifact_saver.export_format is None:
                                raise ValueError(
                                    "Export format is not set. Please set the export format for the artifact saver."
                                )
                            alias_filename = stateful_metric.create_filename(
                                filename=prompt, idx=sample_idx, file_extension=self.artifact_saver.export_format
                            )
                            self.artifact_saver.create_alias(canonical_paths[prompt_idx], alias_filename)

                # Cache outputs once in the agent for pairwise metrics to save compute time and memory.
                if self.task.is_pairwise_evaluation():
                    if self.num_samples_per_input > 1:
                        raise ValueError("Pairwise evaluation with multiple samples per input is not supported.")
                    if self.evaluation_for_first_model:
                        self.cache.append(processed_outputs)
                    else:
                        for pairwise_metric in pairwise_metrics:
                            pairwise_metric.update(x, self.cache[batch_idx], processed_outputs)

    def compute_stateful_metrics(
        self, single_stateful_metrics: List[StatefulMetric], pairwise_metrics: List[StatefulMetric]
    ) -> List[MetricResult]:
        """
        Compute stateful metrics.

        Parameters
        ----------
        single_stateful_metrics : List[StatefulMetric]
            The single stateful metrics to compute.
        pairwise_metrics : List[StatefulMetric]
            The pairwise metrics to compute.

        Returns
        -------
        List[MetricResult]
            The results of the stateful metrics.
        """
        results = []
        for stateful_metric in single_stateful_metrics:
            results.append(stateful_metric.compute())
            stateful_metric.reset()

        if not self.evaluation_for_first_model and self.task.is_pairwise_evaluation():
            for pairwise_metric in pairwise_metrics:
                results.append(pairwise_metric.compute())
                pairwise_metric.reset()
        return results

    def compute_stateless_metrics(self, model: PrunaModel, stateless_metrics: List[Any]) -> List[MetricResult]:
        """
        Compute stateless metrics.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        stateless_metrics : List[Any]
            The stateless metrics to compute.

        Returns
        -------
        Dict[str, Any]
            The results of the stateless metrics.
        """
        results = []
        parent_to_children, children_of_base = group_metrics_by_inheritance(stateless_metrics)
        for (parent, _), children in parent_to_children.items():
            # Get the metrics that share a common parent to share inference computation by calling the parent metric.
            raw_results = parent.compute(children[0], model, self.task.dataloader)
            for child in children:
                results.append(
                    MetricResult.from_results_dict(child.metric_name, dict(children[0].__dict__), raw_results)
                )
        for metric in children_of_base:
            results.append(metric.compute(model, self.task.dataloader))
        return results

    def _maybe_create_input_output_metadata(self, batch, canonical_paths, sample_idx, batch_idx):
        """
        Write prompt-level metadata for saved artifacts.

        If ``save_prompt_metadata`` is enabled, this function appends one JSONL
        record per prompt to ``metadata.jsonl`` in the run output directory.
        The canonical filename is used as the stable identifier.

        Args:
            batch: Batch tuple where the first element contains the prompts.
            canonical_paths: List of canonical file paths corresponding to each
                prompt in the batch.
            sample_idx: Index of the current sample within the evaluation run.
            batch_idx: Index of the batch within the evaluation dataloader loop.

        Returns:
        -------
            None
        """
        if not self.save_in_out_metadata:
            return

        (x, _) = batch  # x = prompts

        metadata_path = Path(self.root_dir) / "metadata.jsonl"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with metadata_path.open("a", encoding="utf-8") as f:
            for prompt_idx, prompt in enumerate(x):
                record = {
                    # stable ID (file actually on disk)
                    "file": Path(canonical_paths[prompt_idx]).name,
                    # full path
                    "canonical_path": str(canonical_paths[prompt_idx]),
                    # original prompt
                    "prompt": str(prompt),
                    "sample_idx": sample_idx,
                    "batch_idx": batch_idx,
                    "prompt_idx": prompt_idx,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
