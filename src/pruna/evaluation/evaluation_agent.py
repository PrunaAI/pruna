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

from typing import Any, List

import torch
from torch import Tensor

from pruna.config.utils import is_empty_config
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.engine.pruna_model import PrunaModel
from pruna.engine.utils import safe_memory_cleanup, set_to_best_available_device
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import group_metrics_by_inheritance
from pruna.evaluation.task import Task
from pruna.logging.logger import pruna_logger


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
    """

    def __init__(
        self,
        task: Task | None = None,
        *,
        request: str | List[str | BaseMetric | StatefulMetric] | None = None,
        datamodule: PrunaDataModule | None = None,
        device: str | torch.device | None = None,
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
        self.device = set_to_best_available_device(self.task.device)
        self.cache: List[Tensor] = []
        self.evaluation_for_first_model: bool = True

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

        model.move_to_device("cpu")
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
        Prepare the model for evaluation by wrapping it in a PrunaModel if it is not already.

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
            model = PrunaModel(model)
            pruna_logger.info("Evaluating a base model.")

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

        model.move_to_device(self.device)
        for batch_idx, batch in enumerate(self.task.dataloader):
            processed_outputs = model.run_inference(batch, self.device)

            (x, gt) = batch
            # Non-pairwise (aka single) metrics have regular update.
            for stateful_metric in single_stateful_metrics:
                stateful_metric.update(x, gt, processed_outputs)

            # Cache outputs once in the agent for pairwise metrics to save compute time and memory.
            if self.task.is_pairwise_evaluation():
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
