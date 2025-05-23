import os
import shutil
from abc import abstractmethod
from typing import Any

from pruna import PrunaModel, SmashConfig, smash
from pruna.algorithms.pruna_base import PrunaAlgorithmBase
from pruna.engine.utils import get_device, move_to_device, safe_memory_cleanup


class AlgorithmTesterBase:
    """Base class for testing algorithms."""

    saving_path: str = "saved_model/"

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

    @classmethod
    def final_teardown(cls, smash_config: SmashConfig) -> None:
        """Teardown the test, remove the saved model and clean up the files in any case."""
        # reset this smash config cache dir, this should not be shared across runs
        smash_config.cleanup_cache_dir()

        # remove the saved model
        model_path = "saved_model/"
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        # clean up the leftovers
        safe_memory_cleanup()

    @classmethod
    def execute_save(cls, smashed_model: PrunaModel) -> None:
        """Save the smashed model."""
        smashed_model.save_pretrained(cls.saving_path)
        assert len(os.listdir(cls.saving_path)) > 0
        if cls.allow_pickle_files:
            cls.assert_no_pickle_files()
        move_to_device(smashed_model, "cpu")

    @classmethod
    def assert_no_pickle_files(cls) -> None:
        """Check for pickle files in the saving path if pickle files are not expected."""
        for file in os.listdir(cls.saving_path):
            assert not file.endswith(".pkl"), "Pickle files found in directory"

    @classmethod
    def compatible_devices(cls) -> list[str]:
        """Get the compatible devices for the algorithm."""
        return cls.algorithm_class.compatible_devices()

    @classmethod
    def get_algorithm_name(cls) -> str:
        """Get the algorithm name."""
        return cls.algorithm_class.algorithm_name

    @classmethod
    def get_algorithm_group(cls) -> str:
        """Get the algorithm group."""
        return cls.algorithm_class.algorithm_group

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Fast hook to verify algorithm application after smashing."""
        pass

    def pre_smash_hook(self, model: PrunaModel) -> None:
        """Fast hook to get information about the base model before smashing (if required)."""
        pass

    def execute_load(self) -> PrunaModel:
        """Load the smashed model."""
        model = PrunaModel.from_pretrained(self.saving_path)
        assert isinstance(model, PrunaModel)
        self.post_smash_hook(model)

    def execute_smash(self, model: Any, smash_config: SmashConfig) -> PrunaModel:
        """Execute the smash operation."""
        self.pre_smash_hook(model)
        smashed_model = smash(model, smash_config=smash_config)
        assert isinstance(smashed_model, PrunaModel)
        self.post_smash_hook(smashed_model)
        assert get_device(smashed_model) == smash_config["device"]
        return smashed_model

    def prepare_smash_config(self, smash_config: SmashConfig, device: str) -> None:
        """Prepare the smash config for the test."""
        smash_config["device"] = device
        smash_config[self.get_algorithm_group()] = self.get_algorithm_name()

        if hasattr(self, "hyperparameters"):
            for key, value in self.hyperparameters.items():
                smash_config[key] = value
