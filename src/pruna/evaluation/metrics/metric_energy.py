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

from typing import Any, Dict, cast
from warnings import warn

import torch
from codecarbon import EmissionsTracker
from torch.utils.data import DataLoader

from pruna.engine.pruna_model import PrunaModel
from pruna.engine.utils import set_to_best_available_device
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.result import MetricResult
from pruna.logging.logger import pruna_logger

ENERGY_CONSUMED = "energy_consumed"
CO2_EMISSIONS = "co2_emissions"


class EnvironmentalImpactStats(BaseMetric):
    """
    Internal metric for evaluating energy usage during model inference.

    This metric is not intended for direct use by end users. It serves as a shared computation
    utility for evaluating environmental impact across energy-related child metrics.

    It estimates two key performance indicators related to sustainability:
    1. CO₂ Emissions: Estimated carbon emissions generated during inference, measured in kilograms (kg).
    2. Energy Consumption: Estimated total energy used by the hardware during inference,
    measured in kilowatt-hours (kWh).

    These values are typically derived from GPU power draw and runtime statistics, and are returned
    as raw results to be used by child metric classes that wrap them into standardized `MetricResult` objects.

    Parameters
    ----------
    n_iterations : int, default=100
        The number of batches to evaluate the model. Note that the energy consumption and CO2 emissions
        are not averaged and will therefore increase with this argument.
    n_warmup_iterations : int, default=10
        The number of warmup batches to evaluate the model.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    """

    def __init__(
        self, n_iterations: int = 100, n_warmup_iterations: int = 10, device: str | torch.device | None = None
    ) -> None:
        self.n_iterations = n_iterations
        self.n_warmup_iterations = n_warmup_iterations
        self.device = set_to_best_available_device(device)

    @torch.no_grad()
    def compute(self, model: PrunaModel, dataloader: DataLoader) -> Dict[str, Any] | MetricResult:
        """
        Compute the energy metrics of a model.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        dataloader : DataLoader
            The dataloader to evaluate the model on.

        Returns
        -------
        dict
            The CO2 emissions and energy consumption of the model.
        """
        # Saving the model to disk to measure loading energy later
        save_path = model.smash_config.cache_dir + "/metrics_save"
        model.save_pretrained(save_path)

        tracker = EmissionsTracker(project_name="pruna", measure_power_secs=0.1)
        tracker.start()

        # Measure the loading energy
        tracker.start_task("Loading model")
        temp_model = model.__class__.from_pretrained(
            save_path,
        )
        tracker.stop_task()
        del temp_model

        model.set_to_eval()
        model.move_to_device(self.device)

        batch = next(iter(dataloader))
        batch = model.inference_handler.move_inputs_to_device(batch, self.device)
        inputs = model.inference_handler.prepare_inputs(batch)

        # Warmup
        for _ in range(self.n_warmup_iterations):
            model(inputs, **model.inference_handler.model_args)

        tracker.start_task("Inference")
        for _ in range(self.n_iterations):
            model(inputs, **model.inference_handler.model_args)
        tracker.stop_task()

        # Make sure all the operations are finished before stopping the tracker
        if self.device == "cuda" or str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
        tracker.stop()

        emissions_data = self._collect_emissions_data(tracker)

        return emissions_data

    def _collect_emissions_data(self, tracker: EmissionsTracker) -> Dict[str, Any]:
        emissions_data = {}
        for task_name, task in tracker._tasks.items():
            emissions_data[f"{task_name}_emissions"] = self._get_data(task.emissions_data, "emissions", task_name)
            emissions_data[f"{task_name}_energy_consumed"] = self._get_data(
                task.emissions_data, "energy_consumed", task_name
            )

        emissions_data[CO2_EMISSIONS] = self._get_data(tracker.final_emissions_data, "emissions", "tracker")
        emissions_data[ENERGY_CONSUMED] = self._get_data(tracker.final_emissions_data, "energy_consumed", "tracker")

        return emissions_data

    def _get_data(self, source: Any, attribute: str, name: str) -> float:
        try:
            return getattr(source, attribute)
        except AttributeError as e:
            pruna_logger.error(f"Could not get {attribute} data for {name}")
            pruna_logger.error(e)
            return 0


@MetricRegistry.register(ENERGY_CONSUMED)
class EnergyConsumedMetric(EnvironmentalImpactStats):
    """
    View over EnvironmentalImpactStats with energy consumed as primary metric.

    Parameters
    ----------
    n_iterations : int, default=100
        The number of batches to evaluate the model. Note that the energy consumption and CO2 emissions
        are not averaged and will therefore increase with this argument.
    n_warmup_iterations : int, default=10
        The number of warmup batches to evaluate the model.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    """

    higher_is_better: bool = False
    metric_name: str = ENERGY_CONSUMED
    metric_units: str = "kWh"

    def compute(self, model: PrunaModel, dataloader: DataLoader) -> MetricResult:
        """
        Compute the energy consumed by a model.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        dataloader : DataLoader
            The dataloader to evaluate the model on.

        Returns
        -------
        MetricResult
            The energy consumed by the model.
        """
        # Note: This runs separate inference if called directly.
        # Use EvaluationAgent to share computation across environmental impact metrics.
        raw_results = super().compute(model, dataloader)
        return MetricResult.from_results_dict(self.metric_name, self.__dict__.copy(), cast(Dict[str, Any], raw_results))


@MetricRegistry.register(CO2_EMISSIONS)
class CO2EmissionsMetric(EnvironmentalImpactStats):
    """
    View over EnvironmentalImpactStats with CO2 emissions as primary metric.

    Parameters
    ----------
    n_iterations : int, default=100
        The number of batches to evaluate the model. Note that the energy consumption and CO2 emissions
        are not averaged and will therefore increase with this argument.
    n_warmup_iterations : int, default=10
        The number of warmup batches to evaluate the model.
    device : str | torch.device | None, optional
        The device to be used, e.g., 'cuda' or 'cpu'. Default is None.
        If None, the best available device will be used.
    """

    higher_is_better: bool = False
    metric_name: str = CO2_EMISSIONS
    metric_units: str = "kgCO2e"

    def compute(self, model: PrunaModel, dataloader: DataLoader) -> MetricResult:
        """
        Compute the CO2 emissions of a model.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        dataloader : DataLoader
            The dataloader to evaluate the model on.

        Returns
        -------
        MetricResult
            The CO2 emissions of the model.
        """
        # Note: This runs separate inference if called directly.
        # Use EvaluationAgent to share computation across environmental impact metrics.
        raw_results = super().compute(model, dataloader)
        return MetricResult.from_results_dict(self.metric_name, self.__dict__.copy(), cast(Dict[str, Any], raw_results))


class EnergyMetric:
    """
    Deprecated class.

    Parameters
    ----------
    *args : Any
        Arguments for EnvironmentalImpactStats.
    **kwargs : Any
        Keyword arguments for EnvironmentalImpactStats.
    """

    def __new__(cls, *args, **kwargs):
        """Forwards to EnvironmentalImpactStats."""
        warn(
            "Class EnergyMetric is deprecated and will be removed in 'v0.2.8' release. \n"
            "It has been replaced by EnvironmentalImpactStats, \n"
            "which is a shared parent class for 'EnergyConsumedMetric' and 'CO2EmissionsMetric'. \n"
            "In the future please use 'EnergyConsumedMetric' or 'CO2EmissionsMetric' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return EnvironmentalImpactStats(*args, **kwargs)
