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
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Literal

from transformers import Pipeline

from pruna.algorithms.base.tags import AlgorithmTag
from pruna.config.smash_config import SUPPORTED_DEVICES, SmashConfig, SmashConfigPrefixWrapper
from pruna.config.smash_space import SMASH_SPACE
from pruna.engine.save import (
    SAVE_BEFORE_SMASH_CACHE_DIR,
    SAVE_FUNCTIONS,
    save_pruna_model,
)
from pruna.logging.logger import pruna_logger


class PrunaAlgorithmBase(ABC):
    """Base class for Pruna algorithms."""

    def __init__(self) -> None:
        self.hyperparameters = self.get_hyperparameters()
        SMASH_SPACE.register_algorithm(self.algorithm_name, self.hyperparameters)

        if (self.incompatible_before or self.incompatible_after) and (self.compatible_before or self.compatible_after):
            raise ValueError("Algorithms need to provide a list of incompatible or compatible algorithms, not both.")

        assert all(device in SUPPORTED_DEVICES for device in self.runs_on)

    def __init_subclass__(cls, **kwargs):
        """Intercept the instantiation of subclasses of the PrunaAlgorithmBase class."""
        super().__init_subclass__(**kwargs)

        # skip if subclass stayed abstract
        impl = cls.__dict__.get("import_algorithm_packages")
        if impl is None:
            return

        # Skip if we already wrapped it (multiple inheritance chains, reloads, etc.)
        if getattr(impl, "__wrapped__", None) is not None:
            return

        # Replace the function with the wrapped version
        cls.import_algorithm_packages = wrap_handle_imports(impl)

    def compatible_devices(self) -> list[str]:
        """
        Return the compatible devices for the algorithm.

        Returns
        -------
        list[str]
            The compatible devices for the algorithm.
        """
        return self.runs_on

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Subclasses need to provide a name for the algorithm."""
        pass

    @property
    def required_install(self) -> str | None:
        """Subclasses need to provide extra requirements for the algorithm."""
        return None

    @property
    @abstractmethod
    def references(self) -> None | dict[str, str]:
        """References likes papers or GitHub repository for the algorithm."""
        pass

    @property
    @abstractmethod
    def runs_on(self) -> list[str]:
        """Subclasses need to provide a list of devices the algorithm can run on."""
        pass

    @property
    @abstractmethod
    def save_fn(self) -> SAVE_FUNCTIONS | None:
        """Subclasses need to provide a save_fn for the algorithm."""
        pass

    @property
    @abstractmethod
    def tokenizer_required(self) -> bool:
        """Subclasses need to request a tokenizer for the algorithm."""
        pass

    @property
    @abstractmethod
    def processor_required(self) -> bool:
        """Subclasses need to request a processor for the algorithm."""
        pass

    @property
    @abstractmethod
    def dataset_required(self) -> bool:
        """Subclasses need to request a dataset for the algorithm."""
        pass

    @property
    @abstractmethod
    def group_tags(self) -> list[AlgorithmTag]:
        """Subclasses need to provide a list of group tags for the algorithm."""
        pass

    @property
    def incompatible_before(self) -> list[str]:
        """Subclasses need to provide a list of algorithms that can not be executed before the current algorithm."""
        return []

    @property
    def incompatible_after(self) -> list[str]:
        """Subclasses need to provide a list of algorithms that can not be executed after the current algorithm."""
        return []

    @property
    def compatible_before(self) -> list[str]:
        """Subclasses need to provide a list of algorithms that can be executed before the current algorithm."""
        return []

    @property
    def compatible_after(self) -> list[str]:
        """Subclasses need to provide a list of algorithms that can be executed after the current algorithm."""
        return []

    @abstractmethod
    def model_check_fn(self, model: Any) -> bool:
        """
        Provide a model check function for the algorithm.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is supported, False otherwise.
        """
        pass

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Provide a algorithm packages for the algorithm.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        return dict()

    def get_hyperparameters(self) -> list:
        """
        Configure all algorithm-specific hyperparameters with ConfigSpace.

        Returns
        -------
        list
            The hyperparameters for the algorithm.
        """
        return []

    def get_model_dependent_hyperparameter_defaults(
        self, model: Any, smash_config: SmashConfig | SmashConfigPrefixWrapper
    ) -> Any:
        """
        Get default values for unconstrained hyperparameters based on the model and configuration.

        Subclasses can override this method to provide default values for their unconstrained hyperparameters.

        Parameters
        ----------
        model : Any
            The model to get the default hyperparameters from.
        smash_config : SmashConfig
            The SmashConfig object.

        Returns
        -------
        Any
            The default unconstrained hyperparameters values for the algorithm.
        """
        return None

    def pre_smash_hook(self, model: Any, smash_config: SmashConfig) -> None:
        """
        Perform any necessary actions before the smashing process begins.

        This method is called before any algorithm is applied to the model. It allows algorithms
        to e.g. perform preparatory steps that need to happen before the actual model adaptation.

        Parameters
        ----------
        model : Any
            The model to be smashed.
        smash_config : SmashConfig
            The SmashConfig object containing the algorithm settings.
        """
        prefix = self.algorithm_name + "_"
        wrapped_config = SmashConfigPrefixWrapper(smash_config, prefix)
        self._pre_smash_hook(model, wrapped_config)

    def _pre_smash_hook(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> None:
        """
        Function to be overridden by an algorithm to perform any pre-smash actions.

        Wrapped by the pre_smash_hook method to handle smash_config prefix.

        Parameters
        ----------
        model : Any
            The model to be smashed.
        smash_config : SmashConfig
            The SmashConfig object containing the algorithm settings.
        """
        pass

    @abstractmethod
    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """Apply the algorithm to the model."""
        pass

    def _apply_to_model_within_transformers_pipeline(
        self, pipeline: Pipeline, smash_config: SmashConfigPrefixWrapper
    ) -> Pipeline:
        """Apply the algorithm to the model."""
        pipeline.model = self._apply(pipeline.model, smash_config)
        return pipeline

    def apply(self, model: Any, smash_config: SmashConfig) -> Any:
        """
        Wrap the apply algorithm for e.g. saving callbacks.

        Parameters
        ----------
        model : Any
            The model to apply the algorithm to.
        smash_config : SmashConfig
            The SmashConfig object containing the save and load functions.

        Returns
        -------
        Any
            The model after the algorithm has been applied.
        """
        if self.save_fn == SAVE_FUNCTIONS.save_before_apply and smash_config._prepare_saving:
            save_dir = smash_config.cache_dir / SAVE_BEFORE_SMASH_CACHE_DIR
            save_pruna_model(model, save_dir, smash_config)

        # save algorithms to reapply after loading
        if self.save_fn == SAVE_FUNCTIONS.save_before_apply or self.save_fn == SAVE_FUNCTIONS.reapply:
            smash_config.reapply_after_load[self.algorithm_group] = self.algorithm_name

        # if the registered save function is None, the original saving function remains
        if self.save_fn is not None and self.save_fn != SAVE_FUNCTIONS.reapply:
            smash_config.save_fns.append(self.save_fn.name)

        prefix = self.algorithm_name + "_"
        wrapped_config = SmashConfigPrefixWrapper(smash_config, prefix)
        return self._apply(model, wrapped_config)

    def _expand(self, items: Iterable[str | AlgorithmTag] | None) -> list[str]:
        """Expand algorithms/tags -> concrete algorithm names."""
        if not items:
            return []

        # avoid circular import
        from pruna.algorithms import PRUNA_ALGORITHMS

        out: list[str] = []
        for it in items:
            if isinstance(it, str):
                out.append(it)
            else:
                out.extend(PRUNA_ALGORITHMS.get_algorithms_by_tag(it))
        return out

    def _expanded_set(self, items: Iterable[str | AlgorithmTag] | None) -> set[str]:
        return set(self._expand(items))

    def get_incompatible_algorithms(self) -> list[str]:
        """Algorithms incompatible with this one."""
        ib = self._expanded_set(self.incompatible_before)
        ia = self._expanded_set(self.incompatible_after)

        if ib or ia:
            # If any explicit incompatibility is provided, use the intersection.
            return sorted(ib & ia)

        # Otherwise, derive from compatible lists by exclusion.
        cb = self._expanded_set(self.compatible_before)
        ca = self._expanded_set(self.compatible_after)
        allowed = cb | ca
        all_algorithms = set(SMASH_SPACE.get_all_algorithms())
        return sorted(all_algorithms - allowed)

    def _required_execution_order(self, side: Literal["before", "after"]) -> list[str]:
        compat = getattr(self, f"compatible_{side}")
        opposite = "after" if side == "before" else "before"
        fallback = getattr(self, f"incompatible_{opposite}")
        return self._expand(compat or fallback)

    def get_required_before(self) -> list[str]:
        """Algorithms required to run before this one."""
        return self._required_execution_order("before")

    def get_required_after(self) -> list[str]:
        """Algorithms required to run after this one."""
        return self._required_execution_order("after")


def wrap_handle_imports(func):
    """
    Wrap the import_algorithm_packages method to handle import errors in a unified and user-friendly way.

    Parameters
    ----------
    func : Callable
        The function to wrap.

    Returns
    -------
    Callable
        The wrapped function.
    """

    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        try:
            result = func(self, *args, **kwargs)
            return result
        except Exception as e:
            if self.required_install is not None:
                pruna_logger.debug(str(e))
                exception_message = (
                    f"Could not import necessary packages for {self.algorithm_name}. ",
                    f"To use {self.algorithm_name}, follow the installation instructions: {self.required_install}.",
                )
            else:
                exception_message = str(e)
                pruna_logger.error(
                    (
                        f"Could not import necessary packages for {self.algorithm_name}.",
                        "Please verify your pruna installation.",
                    )
                )
        raise ImportError(exception_message)

    _wrapper.__wrapped__ = func  # mark the original (helps avoid double-wrapping)
    return _wrapper
