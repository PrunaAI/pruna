from __future__ import annotations

import shutil
import tempfile
from abc import abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Any

from pruna import PrunaModel, SmashConfig, smash
from pruna.algorithms.pruna_base import PrunaAlgorithmBase
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.engine.utils import _resolve_cuda_device, get_device, move_to_device, safe_memory_cleanup
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.task import Task
from pruna.logging.logger import pruna_logger


class AlgorithmTesterBase:
    """Base class for testing algorithms."""

    def __init__(self):
        self._saving_path = Path(tempfile.mkdtemp(prefix="pruna_saved_model_"))

    @property
    @abstractmethod
    def models(self) -> list[str]:
        """Some models to test for this algorithm."""
        pass

    @property
    @abstractmethod
    def reject_models(self) -> list[str]:
        """Some models to reject for this algorithm."""
        pass

    @property
    @abstractmethod
    def allow_pickle_files(self) -> bool:
        """Whether to allow pickle files in the saving path."""
        pass

    @property
    @abstractmethod
    def algorithm_class(self) -> type[PrunaAlgorithmBase]:
        """The algorithm class to test."""
        pass

    @property
    @abstractmethod
    def metrics(self) -> list[str]:
        """The metrics to evaluate the algorithm."""
        pass

    def final_teardown(self, smash_config: SmashConfig) -> None:
        """Teardown the test, remove the saved model and clean up the files in any case."""
        # reset this smash config cache dir, this should not be shared across runs
        smash_config.cleanup_cache_dir()

        if self._saving_path.exists():
            shutil.rmtree(self._saving_path)

        # clean up the leftovers
        safe_memory_cleanup()

    def execute_save(self, smashed_model: PrunaModel) -> None:
        """Save the smashed model."""
        smashed_model.save_pretrained(str(self._saving_path))
        assert len(list(self._saving_path.iterdir())) > 0
        if self.allow_pickle_files:
            self.assert_no_pickle_files()
        move_to_device(smashed_model, "cpu")

    def assert_no_pickle_files(self) -> None:
        """Check for pickle files in the saving path if pickle files are not expected."""
        for file in self._saving_path.iterdir():
            assert file.suffix != ".pkl", "Pickle files found in directory"

    @classmethod
    def compatible_devices(cls) -> list[str]:
        """Get the compatible devices for the algorithm."""
        return cls.algorithm_class.runs_on

    @classmethod
    def get_algorithm_name(cls) -> str:
        """Get the algorithm name."""
        return cls.algorithm_class.algorithm_name

    @classmethod
    def get_algorithm_group(cls) -> str:
        """Get the algorithm group."""
        return cls.algorithm_class.algorithm_group

    @classmethod
    def get_metrics(cls, device: str) -> list[BaseMetric | StatefulMetric]:
        """Get the metrics to evaluate the algorithm."""
        metrics = cls.metrics
        return cls.get_metric_instances(metrics, device)

    @classmethod
    def get_metric_instances(cls, metrics: list[str], device: str) -> list[BaseMetric | StatefulMetric]:
        """Get the metric instances."""
        if not metrics:
            raise ValueError("No metrics provided")
        metric_instances = [MetricRegistry.get_metric(metric) for metric in metrics]
        for metric in metric_instances:
            if hasattr(metric, "n_iterations"):
                metric.n_warmup_iterations = 1
                metric.n_iterations = 1
            # Try setting device or calling .to(device)
            with suppress(Exception):
                metric.device = device
            with suppress(Exception):
                metric.to(device)

            # Handle nested metric.metric if exists
            wrapped = getattr(metric, "metric", None)
            if wrapped is not None:
                with suppress(Exception):
                    wrapped.device = device
                with suppress(Exception):
                    wrapped.to(device)
        return metric_instances

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Fast hook to verify algorithm application after smashing."""
        pass

    def pre_smash_hook(self, model: PrunaModel) -> None:
        """Fast hook to get information about the base model before smashing (if required)."""
        pass

    def execute_load(self) -> PrunaModel:
        """Load the smashed model."""
        model = PrunaModel.from_pretrained(str(self._saving_path))
        assert isinstance(model, PrunaModel)
        self.post_smash_hook(model)
        assert _resolve_cuda_device(model.smash_config.device) == _resolve_cuda_device(get_device(model))
        return model

    def execute_smash(self, model: Any, smash_config: SmashConfig) -> PrunaModel:
        """Execute the smash operation."""
        self.pre_smash_hook(model)
        smashed_model = smash(model, smash_config=smash_config)
        assert isinstance(smashed_model, PrunaModel)
        self.post_smash_hook(smashed_model)
        assert _resolve_cuda_device(get_device(smashed_model)) == _resolve_cuda_device(smash_config["device"])
        return smashed_model

    def execute_evaluation(self, model: Any, datamodule: PrunaDataModule, device: str) -> None:
        """Execute the evaluation operation."""
        metrics = self.get_metrics(device=device)
        datamodule.limit_datasets(5)
        task = Task(request=metrics, datamodule=datamodule, device=device)
        evaluation_agent = EvaluationAgent(task=task)
        results = evaluation_agent.evaluate(model=model)
        for result in results:
            pruna_logger.info(result)

    def prepare_smash_config(self, smash_config: SmashConfig, device: str) -> None:
        """Prepare the smash config for the test."""
        smash_config["device"] = device
        smash_config[self.get_algorithm_group()] = self.get_algorithm_name()

        if hasattr(self, "hyperparameters"):
            for key, value in self.hyperparameters.items():
                smash_config[key] = value
