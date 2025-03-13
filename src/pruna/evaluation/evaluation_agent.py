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

from typing import Any, Dict, List

import torch
from torch import Tensor

from pruna.config.utils import is_empty_config
from pruna.engine.pruna_model import PrunaModel
from pruna.engine.utils import safe_memory_cleanup
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.task import Task
from pruna.logging.logger import pruna_logger


class EvaluationAgent:
    """
    Entry point for evaluating a model.

    Parameters
    ----------
    task : Task
        Configuration object that defines how to evaluate the model.
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        self.first_model_results: Dict[str, Any] = {}
        self.subsequent_model_results: Dict[str, Any] = {}
        self.device = self.task.device
        self.cache: List[Tensor] = []
        self.evaluation_for_first_model: bool = True

    def evaluate(self, model: Any) -> Dict[str, Any]:
        """
        Evaluate models using different metric types.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.

        Returns
        -------
        Dict[str, Any]
            The results of the model.
        """
        results = {}

        model = self.prepare_model(model)

        # Separate metrics by execution strategy
        single_stateful_metrics = self.task.get_single_stateful_metrics()
        pairwise_metrics = self.task.get_pairwise_stateful_metrics()
        stateless_metrics = self.task.get_stateless_metrics()

        # Update and compute stateful metrics.
        pruna_logger.info("Evaluating stateful metrics.")
        with torch.no_grad():
            self.update_stateful_metrics(model, single_stateful_metrics, pairwise_metrics)
        results.update(self.compute_stateful_metrics(single_stateful_metrics, pairwise_metrics))

        # Compute stateless metrics.
        pruna_logger.info("Evaluating isolated inference metrics.")
        results.update(self.compute_stateless_metrics(model, stateless_metrics))

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

        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.item()
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
    ) -> Dict[str, Any]:
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
        Dict[str, Any]
            The results of the stateful metrics.
        """
        results = {}
        for stateful_metric in single_stateful_metrics:
            results[f"{stateful_metric.metric_name}_{stateful_metric.call_type}"] = stateful_metric.compute()
            stateful_metric.reset()

        if not self.evaluation_for_first_model and self.task.is_pairwise_evaluation():
            for pairwise_metric in pairwise_metrics:
                results[f"{pairwise_metric.metric_name}_{pairwise_metric.call_type}"] = pairwise_metric.compute()
                pairwise_metric.reset()
        return results

    def compute_stateless_metrics(self, model: PrunaModel, stateless_metrics: List[Any]) -> Dict[str, Any]:
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
        results = {}
        for stateless_metric in stateless_metrics:
            results.update(stateless_metric.compute(model, self.task.dataloader))
        return results
