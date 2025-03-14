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

import atexit
import json
import os
import shutil
import tempfile
from functools import singledispatchmethod
from typing import Any, Union
from warnings import warn

import numpy as np
import torch
import transformers
from ConfigSpace import Configuration, ConfigurationSpace
from transformers import AutoProcessor, AutoTokenizer

from pruna.config.smash_space import ALGORITHM_GROUPS, SMASH_SPACE
from pruna.data.pruna_datamodule import PrunaDataModule, TokenizerMissingError
from pruna.logging.logger import pruna_logger

ADDITIONAL_ARGS = [
    "max_batch_size",
    "device",
    "cache_dir",
    "save_fns",
    "load_fn",
    "reapply_after_load",
]

TOKENIZER_SAVE_PATH = "tokenizer/"
PROCESSOR_SAVE_PATH = "processor/"
SMASH_CONFIG_FILE_NAME = "smash_config.json"


class SmashConfig:
    """
    Wrapper class to hold a ConfigSpace Configuration object as a Smash configuration.

    Parameters
    ----------
    max_batch_size : int, optional
        The maximum number of batches to process at once. Default is 1.
    device : str, optional
        The device to be used for smashing, e.g., 'cuda' or 'cpu'. Default is 'cuda'.
    cache_dir_prefix : str, optional
        The prefix for the cache directory. If None, a default cache directory will be created.
    configuration : Configuration, optional
        The configuration to be used for smashing. If None, a default configuration will be created.
    """

    def __init__(
        self,
        max_batch_size: int = 1,
        device: str = "cuda",
        cache_dir_prefix: str = os.path.join(os.path.expanduser("~"), ".cache", "pruna"),
        configuration: Configuration | None = None,
    ) -> None:
        SMASH_SPACE.gather_algorithm_buffer()
        self._configuration: Configuration = (
            SMASH_SPACE.get_default_configuration() if configuration is None else configuration
        )
        self.config_space: ConfigurationSpace = self._configuration.config_space
        self.max_batch_size = max_batch_size
        self.device = device

        self.cache_dir_prefix = cache_dir_prefix
        if not os.path.exists(cache_dir_prefix):
            os.makedirs(cache_dir_prefix)
        self.cache_dir = tempfile.mkdtemp(dir=cache_dir_prefix)

        self.save_fns: list[str] = []
        self.load_fn: str | None = None
        self.reapply_after_load: dict[str, str | None] = {algorithm: None for algorithm in ALGORITHM_GROUPS}
        self.tokenizer = None
        self.processor = None
        self._data: PrunaDataModule | None = None

        # internal variable *to save time* by avoiding compilers saving models for inference-only smashing
        self._prepare_saving = True

        # ensure the cache directory is deleted on program exit
        atexit.register(self.cleanup_cache_dir)

    def __del__(self) -> None:
        """Delete the SmashConfig object."""
        self.cleanup_cache_dir()

    def __eq__(self, other: Any) -> bool:
        """Check if two SmashConfigs are equal."""
        if not isinstance(other, self.__class__):
            return False

        return (
            self._configuration == other._configuration
            and self.max_batch_size == other.max_batch_size
            and self.device == other.device
            and self.cache_dir_prefix == other.cache_dir_prefix
            and self.save_fns == other.save_fns
            and self.load_fn == other.load_fn
            and self.reapply_after_load == other.reapply_after_load
        )

    def cleanup_cache_dir(self) -> None:
        """Clean up the cache directory."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def reset_cache_dir(self) -> None:
        """Reset the cache directory."""
        self.cleanup_cache_dir()
        self.cache_dir = tempfile.mkdtemp(dir=self.cache_dir_prefix)

    def load_from_json(self, path: str) -> None:
        """
        Load a SmashConfig from a JSON file.

        Parameters
        ----------
        path : str
            The file path to the JSON file containing the configuration.
        """
        with open(os.path.join(path, SMASH_CONFIG_FILE_NAME), "r") as f:
            json_string = f.read()
            config_dict = json.loads(json_string)

        for name in ADDITIONAL_ARGS:
            # do not load the old cache directory
            if name == "cache_dir":
                config_dict.pop(name)
                continue
            setattr(self, name, config_dict.pop(name))

        self._configuration = Configuration(SMASH_SPACE, values=config_dict)

        if os.path.exists(os.path.join(path, TOKENIZER_SAVE_PATH)):
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, TOKENIZER_SAVE_PATH))

        if os.path.exists(os.path.join(path, PROCESSOR_SAVE_PATH)):
            self.processor = AutoProcessor.from_pretrained(os.path.join(path, PROCESSOR_SAVE_PATH))

    def save_to_json(self, path: str) -> None:
        """
        Save the SmashConfig to a JSON file, including additional keys.

        Parameters
        ----------
        path : str
            The file path where the JSON file will be saved.
        """
        config_dict = dict(self._configuration)
        for key, value in config_dict.items():
            config_dict[key] = convert_numpy_types(value)

        for name in ADDITIONAL_ARGS:
            config_dict[name] = getattr(self, name)

        # Save the updated dictionary back to a JSON file
        with open(os.path.join(path, SMASH_CONFIG_FILE_NAME), "w") as f:
            json.dump(config_dict, f, indent=4)

        if self.tokenizer:
            self.tokenizer.save_pretrained(os.path.join(path, TOKENIZER_SAVE_PATH))
        if self.processor:
            self.processor.save_pretrained(os.path.join(path, PROCESSOR_SAVE_PATH))
        if self._data is not None:
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
        # since this function is only used for loading algorithm settings, we will ignore additional arguments
        filtered_config_dict = {k: v for k, v in config_dict.items() if k not in ADDITIONAL_ARGS}
        discarded_args = [k for k in config_dict.keys() if k in ADDITIONAL_ARGS]
        if discarded_args:
            pruna_logger.info(f"Discarded arguments: {discarded_args}")

        # first load the algorithm settings
        # otherwise fine-grained hyperparameters will not be active yet and we can not set them
        # lambda returns False for keys in ALGORITHM_GROUPS (and False sorts before True)
        for k, v in sorted(filtered_config_dict.items(), key=lambda item: item[0] not in ALGORITHM_GROUPS):
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
        self.load_fn = None
        self.reapply_after_load = {algorithm: None for algorithm in ALGORITHM_GROUPS}

        # reset potentiallypreviously used cache directory
        self.reset_cache_dir()

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
        if self._data is None:
            return None
        else:
            return self._data.train_dataloader(**kwargs)

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
        DataLoader
            The DataLoader instance associated with the SmashConfig.
        """
        if self._data is None:
            return None
        else:
            return self._data.val_dataloader(**kwargs)

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
        DataLoader
            The DataLoader instance associated with the SmashConfig.
        """
        if self._data is None:
            return None
        else:
            return self._data.test_dataloader(**kwargs)

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
            self._data = PrunaDataModule.from_string(dataset_name, *args, **kwargs)
        except TokenizerMissingError:
            raise ValueError(
                f"Tokenizer is required for {dataset_name} but not provided. "
                "Please provide a tokenizer with smash_config.add_tokenizer()."
            ) from None

    @add_data.register(list)
    def _(self, datasets: list, collate_fn: str, *args, **kwargs) -> None:
        try:
            kwargs["tokenizer"] = self.tokenizer
            self._data = PrunaDataModule.from_datasets(datasets, collate_fn, *args, **kwargs)
        except TokenizerMissingError:
            raise ValueError(
                f"Tokenizer is required for {collate_fn} but not provided. "
                "Please provide a tokenizer with smash_config.add_tokenizer()."
            ) from None

    @add_data.register(tuple)
    def _(self, datasets: tuple, collate_fn: str, *args, **kwargs) -> None:
        try:
            kwargs["tokenizer"] = self.tokenizer
            self._data = PrunaDataModule.from_datasets(datasets, collate_fn, *args, **kwargs)
        except TokenizerMissingError:
            raise ValueError(
                f"Tokenizer is required for {collate_fn} but not provided. "
                "Please provide a tokenizer with smash_config.add_tokenizer()."
            ) from None

    @add_data.register(PrunaDataModule)
    def _(self, datamodule: PrunaDataModule) -> None:
        self._data = datamodule

    def add_tokenizer(self, tokenizer: str | transformers.AutoTokenizer) -> None:
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

    def add_processor(self, processor: str | transformers.AutoProcessor) -> None:
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

    def check_argument_compatibility(self, algorithm_name: str) -> None:
        """
        Check if the SmashConfig has the required arguments (tokenizer, processor, dataset) for an algorithm.

        Parameters
        ----------
        algorithm_name : str
            The algorithm name that is about to be activated.
        """
        algorithm_requirements = SMASH_SPACE.model_requirements[algorithm_name]
        if algorithm_requirements["tokenizer_required"] and self.tokenizer is None:
            raise ValueError(
                f"{algorithm_name} requires a tokenizer. Please provide it with smash_config.add_tokenizer()."
            )
        if algorithm_requirements["processor_required"] and self.processor is None:
            raise ValueError(
                f"{algorithm_name} requires a processor. Please provide it with smash_config.add_processor()."
            )
        if algorithm_requirements["dataset_required"] and self._data is None:
            raise ValueError(f"{algorithm_name} requires a dataset. Please provide it with smash_config.add_data().")

    def get_tokenizer_name(self) -> str | None:
        """
        Get a tokenizer object from a tokenizer name.

        Returns
        -------
        str | None
            The name of the tokenizer to use.
        """
        if self.tokenizer is None:
            return None
        if hasattr(self.tokenizer, "tokenizer"):
            return self.tokenizer.tokenizer.name_or_path
        else:
            return self.tokenizer.name_or_path

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

        Examples
        --------
        >>> config = SmashConfig()
        >>> config["quantizer"] = "awq"
        >>> config["quantizer"]
        "awq"
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

        Returns
        -------
        None
            This method updates the internal configuration state but does not return a value.

        Examples
        --------
        >>> config = SmashConfig()
        >>> config["quantizer"] = "awq"
        >>> config["quantizer"]
        "awq"
        """
        deprecated = False
        deprecated_algorithm_groups = [
            "quantizers",
            "pruners",
            "distillers",
            "cachers",
            "recoverers",
            "compilers",
            "batchers",
        ]
        if name in ADDITIONAL_ARGS:
            return setattr(self, name, value)
        elif name in ALGORITHM_GROUPS + deprecated_algorithm_groups:
            # deprecation logic for assignment of plural algorithm groups, e.g. quantizers
            if name in deprecated_algorithm_groups:
                new_algorithm_group = name[:-1]
                warn(f"The algorithm group {name} is deprecated.", DeprecationWarning, stacklevel=2)
                name = new_algorithm_group
                deprecated = True
            # deprecation logic for assignment of algorithms as lists
            if isinstance(value, list):
                if len(value) == 0:
                    value = None
                else:
                    value = value[0]
                deprecated = True
                warn("Assigning algorithms as lists is deprecated...", DeprecationWarning, stacklevel=2)
            # deprecating old method names
            deprecated_algorithm_names = {
                "llm_lora": "text_to_text_perp",
                "llm-lora": "text_to_text_perp",
                "text_to_text_lora": "text_to_text_perp",
                "text_to_image_lora": "text_to_image_perp",
                "torch-structured": "torch_structured",
                "torch-unstructured": "torch_unstructured",
                "llm-int8": "llm_int8",
                "diffusers2": "stable_fast",
                "x-fast": "x_fast",
                "cgenerate": "c_generate",
                "ctranslate": "c_translate",
                "cwhisper": "c_whisper",
                "ws2t": "whisper_s2t",
                "step_caching": "deepcache",
            }
            if value in list(deprecated_algorithm_names.keys()):
                warn(
                    f"The {value} method has been renamed to {deprecated_algorithm_names[value]}.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                value = deprecated_algorithm_names[value]
                deprecated = True
            ###
            # end of deprecation logic for assignment of algorithms as lists
            if value is not None:
                self.check_argument_compatibility(value)
            if deprecated:
                warn(f"Continuing with setting smash_config['{name}'] = '{value}'.", DeprecationWarning, stacklevel=2)
            self._configuration.__setitem__(name, value)
        else:
            # isolating prefix behavior here for easy removal later
            deprecated_prefixes = ["quant_", "prune_", "comp_", "recov_", "distill_", "cache_", "batch_"]

            def remove_starting_prefix(s: str) -> str:
                for prefix in deprecated_prefixes:
                    if s.startswith(prefix):
                        warn(
                            f"The {prefix} prefix is deprecated. " f"Please use the {s[len(prefix) :]} instead.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        return s[len(prefix) :]
                return s

            name = remove_starting_prefix(name)
            # deprecation logic over
            return self._configuration.__setitem__(name, value)

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
        values = dict(self._configuration)
        header = "SmashConfig("
        lines = [
            f"  '{k}': {convert_numpy_types(values[k])!r},"
            for k in sorted(values, key=self._configuration.config_space.index_of.get)  # type: ignore
            # determine whether hyperparameter is conditionally active
            if values[k] is not None or len(self._configuration.config_space.parents_of[k]) > 0
        ]
        end = ")"
        return "\n".join([header, *lines, end])

    def __repr__(self) -> str:  # noqa: D105
        return self.__str__()


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
        if key in ADDITIONAL_ARGS + ALGORITHM_GROUPS:
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
