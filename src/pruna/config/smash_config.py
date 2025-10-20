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

import atexit
import json
import shutil
import tempfile
from functools import singledispatchmethod
from pathlib import Path
from typing import Any, Dict, Union
from warnings import warn

import numpy as np
import torch
from ConfigSpace import Configuration, ConfigurationSpace
from transformers import AutoProcessor, AutoTokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from pruna.config.smash_space import SMASH_SPACE
from pruna.data.pruna_datamodule import PrunaDataModule, TokenizerMissingError
from pruna.engine.utils import set_to_best_available_device
from pruna.logging.logger import pruna_logger

ADDITIONAL_ARGS = [
    "batch_size",
    "device",
    "device_map",
    "cache_dir",
    "save_fns",
    "load_fns",
    "reapply_after_load",
]

TOKENIZER_SAVE_PATH = "tokenizer/"
PROCESSOR_SAVE_PATH = "processor/"
SMASH_CONFIG_FILE_NAME = "smash_config.json"
SUPPORTED_DEVICES = ["cpu", "cuda", "mps", "accelerate"]
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pruna"


class SmashConfig:
    """
    Wrapper class to hold a ConfigSpace Configuration object as a Smash configuration.

    Parameters
    ----------
    max_batch_size : int, optional
        Deprecated. The number of batches to process at once. Default is 1.
    batch_size : int, optional
        The number of batches to process at once. Default is 1.
    device : str | torch.device | None, optional
        The device to be used for smashing, options are "cpu", "cuda", "mps", "accelerate". Default is None.
        If None, the best available device will be used.
    cache_dir_prefix : str, optional
        The prefix for the cache directory. If None, a default cache directory will be created.
    configuration : Configuration, optional
        The configuration to be used for smashing. If None, a default configuration will be created.
    """

    def __init__(
        self,
        configuration: Dict[str, Any] | Configuration | None = None,
        batch_size: int = 1,
        device: str | torch.device | None = None,
        cache_dir_prefix: str | Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self._configuration: Configuration = (
            configuration if isinstance(configuration, Configuration) else SMASH_SPACE.get_default_configuration()
        )
        self.config_space: ConfigurationSpace = self._configuration.config_space
        self.batch_size = batch_size
        self.device = set_to_best_available_device(device)
        self.device_map = None

        self.cache_dir_prefix = Path(cache_dir_prefix)
        if not self.cache_dir_prefix.exists():
            self.cache_dir_prefix.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(tempfile.mkdtemp(dir=cache_dir_prefix))

        self.save_fns: list[str] = []
        self.load_fns: list[str] = []
        self.reapply_after_load: dict[str, str | None] = {}
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.processor: ProcessorMixin | None = None
        self.data: PrunaDataModule | None = None
        self._target_module: Any | None = None

        # internal variable *to save time* by avoiding compilers saving models for inference-only smashing
        self._prepare_saving = True

        # internal variable to overwrite the graph-induced order of algorithms if desired
        self._algorithm_order: list[str] | None = None

        # internal variable to indicated that a model has been smashed for a specific batch size
        self.__locked_batch_size = False

        # ensure the cache directory is deleted on program exit
        atexit.register(self.cleanup_cache_dir)

        # if configuration was provided as a dictionary, add it as if the user called config.add()
        if isinstance(configuration, (dict, list)):
            self.add(configuration)

    @classmethod
    def from_list(
        cls,
        configuration: list[str],
        batch_size: int = 1,
        device: str | torch.device | None = None,
        cache_dir_prefix: str | Path = DEFAULT_CACHE_DIR,
    ) -> SmashConfig:
        """
        Create a SmashConfig from a Configuration object.
        """
        return cls(configuration=configuration, batch_size=batch_size, device=device, cache_dir_prefix=cache_dir_prefix)

    @classmethod
    def from_dict(
        cls,
        configuration: Dict[str, Any],
        batch_size: int = 1,
        device: str | torch.device | None = None,
        cache_dir_prefix: str | Path = DEFAULT_CACHE_DIR,
    ) -> SmashConfig:
        """
        Create a SmashConfig from a dictionary.

        Parameters
        ----------
        configuration : Dict[str, Any]
            The dictionary to create the SmashConfig from.
        batch_size : int, optional
            The batch size to use for the SmashConfig. Default is 1.
        device : str | torch.device | None, optional
            The device to use for the SmashConfig. Default is None.
        cache_dir_prefix : str | Path, optional
            The prefix for the cache directory. Default is DEFAULT_CACHE_DIR.

        Returns
        -------
        SmashConfig
            The SmashConfig object instantiated from the dictionary.
        """
        return cls(configuration=configuration, batch_size=batch_size, device=device, cache_dir_prefix=cache_dir_prefix)

    def __del__(self) -> None:
        """Delete the SmashConfig object."""
        self.cleanup_cache_dir()

    def __eq__(self, other: Any) -> bool:
        """Check if two SmashConfigs are equal."""
        if not isinstance(other, self.__class__):
            return False

        return (
            self._configuration == other._configuration
            and self.batch_size == other.batch_size
            and self.device == other.device
            and self.cache_dir_prefix == other.cache_dir_prefix
            and self.save_fns == other.save_fns
            and self.load_fns == other.load_fns
            and self.reapply_after_load == other.reapply_after_load
        )

    def cleanup_cache_dir(self) -> None:
        """Clean up the cache directory."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

    def reset_cache_dir(self) -> None:
        """Reset the cache directory."""
        self.cleanup_cache_dir()
        self.cache_dir = Path(tempfile.mkdtemp(dir=self.cache_dir_prefix))

    def load_from_json(self, path: str | Path) -> None:
        """
        Load a SmashConfig from a JSON file.

        Parameters
        ----------
        path : str| Path
            The file path to the JSON file containing the configuration.
        """
        config_path = Path(path) / SMASH_CONFIG_FILE_NAME
        json_string = config_path.read_text()
        config_dict = json.loads(json_string)

        # check device compatibility
        if "device" in config_dict:
            config_dict["device"] = set_to_best_available_device(config_dict["device"])

        # support deprecated load_fn
        if "load_fn" in config_dict:
            value = config_dict.pop("load_fn")
            config_dict["load_fns"] = [value]

        # support deprecated max batch size argument
        if "max_batch_size" in config_dict:
            config_dict["batch_size"] = config_dict.pop("max_batch_size")

        for name in ADDITIONAL_ARGS:
            if name not in config_dict:
                pruna_logger.warning(f"Argument {name} not found in config file. Skipping...")
                continue

            # do not load the old cache directory
            if name == "cache_dir":
                if name in config_dict:
                    del config_dict[name]
                continue

            setattr(self, name, config_dict.pop(name))

        self._configuration = Configuration(SMASH_SPACE, values=config_dict)

        tokenizer_path = Path(path) / TOKENIZER_SAVE_PATH
        if tokenizer_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        processor_path = Path(path) / PROCESSOR_SAVE_PATH
        if processor_path.exists():
            self.processor = AutoProcessor.from_pretrained(str(processor_path))

    def save_to_json(self, path: str | Path) -> None:
        """
        Save the SmashConfig to a JSON file, including additional keys.

        Parameters
        ----------
        path : str| Path]
            The file path where the JSON file will be saved.
        """
        config_dict = dict(self._configuration)
        for key, value in config_dict.items():
            config_dict[key] = convert_numpy_types(value)

        for name in ADDITIONAL_ARGS:
            config_dict[name] = getattr(self, name)

        # do not save the old cache directory or device
        if "cache_dir" in config_dict:
            del config_dict["cache_dir"]

        # Save the updated dictionary back to a JSON file
        config_path = Path(path) / SMASH_CONFIG_FILE_NAME
        config_path.write_text(json.dumps(config_dict, indent=4))

        if self.tokenizer:
            self.tokenizer.save_pretrained(str(Path(path) / TOKENIZER_SAVE_PATH))
        if self.processor:
            self.processor.save_pretrained(str(Path(path) / PROCESSOR_SAVE_PATH))
        if self.data is not None:
            pruna_logger.info("Data detected in smash config, this will be detached and not reloaded...")

    def load_dict(self, config_dict: dict) -> None:
        """
        Load a dictionary of hyperparameters into the SmashConfig.

        Parameters
        ----------
        config_dict : dict
            The dictionary to load into the SmashConfig.

        Examples
        --------
        >>> config = SmashConfig()
        >>> config.load_dict({'cacher': 'deepcache', 'deepcache_interval': 4})
        >>> config
        SmashConfig(
         'cacher': 'deepcache',
         'deepcache_interval': 4,
        )
        """
        # check device compatibility
        if "device" in config_dict:
            config_dict["device"] = set_to_best_available_device(config_dict["device"])

        # since this function is only used for loading algorithm settings, we will ignore additional arguments
        filtered_config_dict = {k: v for k, v in config_dict.items() if k not in ADDITIONAL_ARGS}
        discarded_args = [k for k in config_dict if k in ADDITIONAL_ARGS]
        if discarded_args:
            pruna_logger.info(f"Discarded arguments: {discarded_args}")

        for k, v in filtered_config_dict.items():
            self[k] = v

    def flush_configuration(self) -> None:
        """
        Remove all algorithm hyperparameters from the SmashConfig.

        Examples
        --------
        >>> config = SmashConfig()
        >>> config['cacher'] = 'deepcache'
        >>> config.flush_configuration()
        >>> config
        SmashConfig()
        """
        self._configuration = SMASH_SPACE.get_default_configuration()

        # flush also saving / load functionality associated with a specific configuration
        self.save_fns = []
        self.load_fns = []
        self.reapply_after_load = {}

        # reset potentially previously used cache directory
        self.reset_cache_dir()

    def __get_dataloader(self, dataloader_name: str, **kwargs) -> torch.utils.data.DataLoader | None:
        if self.data is None:
            return None

        if "batch_size" in kwargs and kwargs["batch_size"] != self.batch_size:
            pruna_logger.warning(
                f"Batch size {kwargs['batch_size']} is not the same as the batch size {self.batch_size}"
                f"set in the SmashConfig. Using the {self.batch_size}."
            )
        kwargs["batch_size"] = self.batch_size
        return getattr(self.data, dataloader_name)(**kwargs)

    def train_dataloader(self, **kwargs) -> torch.utils.data.DataLoader | None:
        """
        Getter for the train DataLoader instance.

        Parameters
        ----------
        **kwargs : dict
            Any additional arguments used when loading data, overriding the default values provided in the constructor.
            Examples: img_size: int would override the collate function default for image generation,
            while batch_size: int, shuffle: bool, pin_memory: bool, ... would override the dataloader defaults.

        Returns
        -------
        torch.utils.data.DataLoader | None
            The DataLoader instance associated with the SmashConfig.
        """
        return self.__get_dataloader("train_dataloader", **kwargs)

    def val_dataloader(self, **kwargs) -> torch.utils.data.DataLoader | None:
        """
        Getter for the validation DataLoader instance.

        Parameters
        ----------
        **kwargs : dict
            Any additional arguments used when loading data, overriding the default values provided in the constructor.
            Examples: img_size: int would override the collate function default for image generation,
            while batch_size: int, shuffle: bool, pin_memory: bool, ... would override the dataloader defaults.

        Returns
        -------
        torch.utils.data.DataLoader | None
            The DataLoader instance associated with the SmashConfig.
        """
        return self.__get_dataloader("val_dataloader", **kwargs)

    def test_dataloader(self, **kwargs) -> torch.utils.data.DataLoader | None:
        """
        Getter for the test DataLoader instance.

        Parameters
        ----------
        **kwargs : dict
            Any additional arguments used when loading data, overriding the default values provided in the constructor.
            Examples: img_size: int would override the collate function default for image generation,
            while batch_size: int, shuffle: bool, pin_memory: bool, ... would override the dataloader defaults.

        Returns
        -------
        torch.utils.data.DataLoader | None
            The DataLoader instance associated with the SmashConfig.
        """
        return self.__get_dataloader("test_dataloader", **kwargs)

    @singledispatchmethod
    def add_data(self, arg):
        """
        Add data to the SmashConfig.

        Parameters
        ----------
        arg : Any
            The argument to be used.
        """
        pruna_logger.error("Unsupported argument type for .add_data() SmashConfig function")
        raise NotImplementedError()

    @add_data.register
    def _(self, dataset_name: str, *args, **kwargs) -> None:
        try:
            kwargs["tokenizer"] = self.tokenizer
            self.data = PrunaDataModule.from_string(dataset_name, *args, **kwargs)
        except TokenizerMissingError:
            raise ValueError(
                f"Tokenizer is required for {dataset_name} but not provided. "
                "Please provide a tokenizer with smash_config.add_tokenizer()."
            ) from None

    @add_data.register(list)
    def _(self, datasets: list, collate_fn: str, *args, **kwargs) -> None:
        try:
            kwargs["tokenizer"] = self.tokenizer
            self.data = PrunaDataModule.from_datasets(datasets, collate_fn, *args, **kwargs)
        except TokenizerMissingError:
            raise ValueError(
                f"Tokenizer is required for {collate_fn} but not provided. "
                "Please provide a tokenizer with smash_config.add_tokenizer()."
            ) from None

    @add_data.register(tuple)
    def _(self, datasets: tuple, collate_fn: str, *args, **kwargs) -> None:
        try:
            kwargs["tokenizer"] = self.tokenizer
            self.data = PrunaDataModule.from_datasets(datasets, collate_fn, *args, **kwargs)
        except TokenizerMissingError:
            raise ValueError(
                f"Tokenizer is required for {collate_fn} but not provided. "
                "Please provide a tokenizer with smash_config.add_tokenizer()."
            ) from None

    @add_data.register(PrunaDataModule)
    def _(self, datamodule: PrunaDataModule) -> None:
        self.data = datamodule

    def add_tokenizer(self, tokenizer: str | PreTrainedTokenizerBase) -> None:
        """
        Add a tokenizer to the SmashConfig.

        Parameters
        ----------
        tokenizer : str | transformers.AutoTokenizer
            The tokenizer to be added to the SmashConfig.
        """
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

    def add_processor(self, processor: str | ProcessorMixin) -> None:
        """
        Add a processor to the SmashConfig.

        Parameters
        ----------
        processor : str | transformers.AutoProcessor
            The processor to be added to the SmashConfig.
        """
        if isinstance(processor, str):
            self.processor = AutoProcessor.from_pretrained(processor)
        else:
            self.processor = processor

    def add_target_module(self, target_module: Any) -> None:
        """
        Add a target module to prune to the SmashConfig.

        Parameters
        ----------
        target_module : Any
            The target module to prune.
        """
        if self["pruner"] is None:
            pruna_logger.error("No pruner selected, target module is only supported by torch_structured pruner.")
            raise
        elif self["pruner"] != "torch_structured":
            pruna_logger.error("Target module is only supported for torch_structured pruner.")
            raise
        self._target_module = target_module

    def lock_batch_size(self) -> None:
        """Lock the batch size in the SmashConfig."""
        self.__locked_batch_size = True

    def is_batch_size_locked(self) -> bool:
        """
        Check if the batch size is locked in the SmashConfig.

        Returns
        -------
        bool
            True if the batch size is locked, False otherwise.
        """
        return self.__locked_batch_size

    def __getitem__(self, name: str) -> Any:
        """
        Get a configuration value from the configuration.

        Parameters
        ----------
        name : str
            The name of the configuration setting.

        Returns
        -------
        Any
            Configuration value for the given name
        """
        if name in ADDITIONAL_ARGS:
            return getattr(self, name)
        else:
            return_value = self._configuration.__getitem__(name)
            # config space internally holds numpy types
            # we convert this to native python types for printing and handing arguments to pruna algorithms
            return convert_numpy_types(return_value)

    def __setitem__(self, name: str, value: Any) -> None:
        """
        Set a configuration value for a given name.

        Parameters
        ----------
        name : str
            The name of the configuration setting.
        value : Any
            The value to set for the configuration setting.
        """
        if name in ADDITIONAL_ARGS:
            return setattr(self, name, value)
        else:
            # support old way of activating algorithms
            if value in SMASH_SPACE.get_all_algorithms():
                pruna_logger.warning(f"Setting {name} to {value} is deprecated. Please use config.add({value}).")
            else:
                pruna_logger.warning(f"Setting {name} deprecated. Please use config.add(dict({name}={value})).")
            self.add(value)

    def add(self, request: Any) -> None:
        """
        Add a value to the SmashConfig.

        Parameters
        ----------
        request : Any
            The value to add to the SmashConfig.
        """
        # TODO: add example to docstring
        # request wants to activate a single algorithm
        if isinstance(request, str):
            self._configuration[request] = True
        if isinstance(request, list):
            if not all(isinstance(item, str) for item in request):
                raise ValueError("Request must be a list of algorithm names.")
            for item in request:
                self._configuration[item] = True
        if isinstance(request, dict):
            if all(isinstance(item, dict) for item in request.values()):
                for key, value in request.items():
                    for k, v in value.items():
                        if not k.startswith(key):
                            k = f"{key}_{k}"
                        self._configuration[k] = v
            else:
                for key, value in request.items():
                    self._configuration[key] = value
        return

    def __getattr__(self, attr: str) -> object:  # noqa: D105
        if attr == "_data":
            return self.__dict__.get("_data")
        elif attr == "_configuration":
            return self.__dict__.get("_configuration")
        return_value = getattr(self._configuration, attr)
        # config space internally holds numpy types
        # we convert this to native python types for printing and handing arguments to pruna algorithms
        return convert_numpy_types(return_value)

    def __str__(self) -> str:  # noqa: D105
        header = "SmashConfig("
        lines = []
        for alg in self.get_active_algorithms():
            lines.append(f"  {alg}")
            if len(self._configuration.config_space.children_of[alg]) > 0:
                for child in self._configuration.config_space.children_of[alg]:
                    child_name = child.name
                    child_value = self._configuration[child_name]
                    lines.append(f"      {child_name.removeprefix(alg + '_')}: {convert_numpy_types(child_value)!r},")
            else:
                lines.append("      -")
        end = ")"
        return "\n".join([header, *lines, end])

    def __repr__(self) -> str:  # noqa: D105
        return self.__str__()

    def get_active_algorithms(self) -> list[str]:
        """Get all active algorithms in this smash config."""
        all_algorithms = self.config_space.get_all_algorithms()
        return [k for k, v in self._configuration.items() if v and k in all_algorithms]

    def overwrite_algorithm_order(self, algorithm_order: list[str]) -> None:
        """Overwrite the graph-induced order of algorithms if desired."""
        if not set(algorithm_order) == set(self.get_active_algorithms()):
            raise ValueError("All active algorithms must be contained in the given algorithm order.")
        self._algorithm_order = algorithm_order

    def disable_saving(self) -> None:
        """Disable the saving of the SmashConfig."""
        pruna_logger.info("Disabling the preparation of saving, smashed model will not be saveable.")
        self._prepare_saving = False


class SmashConfigPrefixWrapper:
    """
    Wrapper for SmashConfig to add a prefix to the config keys.

    Parameters
    ----------
    base_config : Union[SmashConfig, "SmashConfigPrefixWrapper"]
        The base SmashConfig or SmashConfigPrefixWrapper object.
    prefix : str
        The prefix to add to the config keys.
    """

    def __init__(self, base_config: Union[SmashConfig, "SmashConfigPrefixWrapper"], prefix: str) -> None:
        self._base_config = base_config
        self._prefix = prefix

    def __getitem__(self, key: str) -> Any:
        """
        Intercept `wrapped[key]` and prepend the prefix.

        Parameters
        ----------
        key : str
            The key to get from the config.

        Returns
        -------
        Any
            The value from the config.
        """
        parent_hyperparameters = self._base_config.get_parent_hyperparameters()
        if key in ADDITIONAL_ARGS + parent_hyperparameters:
            return self._base_config[key]
        actual_key = self._prefix + key
        return self._base_config[actual_key]

    def __getattr__(self, attr: str) -> Any:
        """
        Called *only* if `attr` is not found as a normal attribute on `self`. Fallback to the base_config's attribute.

        Parameters
        ----------
        attr : str
            The attribute to get from the config.

        Returns
        -------
        Any
            The value from the config.
        """
        return getattr(self._base_config, attr)

    def lock_batch_size(self) -> None:
        """Lock the batch size in the SmashConfig."""
        self._base_config.lock_batch_size()


def convert_numpy_types(input_value: Any) -> Any:
    """
    Convert numpy types in the dictionary to native Python types.

    Parameters
    ----------
    input_value : Any
        A value that may be of numpy types (e.g., np.bool_, np.int_).

    Returns
    -------
    Any
        A new value where all numpy types are converted to native Python types.
    """
    if isinstance(input_value, np.generic):
        return input_value.item()
    else:
        return input_value
