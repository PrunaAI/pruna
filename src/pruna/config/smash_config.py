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
import inspect
import json
import os
import shutil
import tempfile
import warnings
from copy import deepcopy
from functools import singledispatchmethod
from typing import Any, Union
from warnings import warn

import numpy as np
import torch
from ConfigSpace import Configuration, ConfigurationSpace
from transformers import AutoProcessor, AutoTokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from pruna.config.smash_space import ALGORITHM_GROUPS, SMASH_SPACE
from pruna.data.pruna_datamodule import PrunaDataModule, TokenizerMissingError
from pruna.logging.logger import pruna_logger

warnings.filterwarnings("default", category=DeprecationWarning)

ADDITIONAL_ARGS = [
    "cache_dir",
    "save_fns",
    "load_fns",
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
        Deprecated. The number of batches to process at once. Default is 1.
    batch_size : int, optional
        The number of batches to process at once. Default is 1.
    device : str, optional
        The device to be used for smashing, e.g., 'cuda' or 'cpu'. Default is 'cuda'.
    cache_dir_prefix : str, optional
        The prefix for the cache directory. If None, a default cache directory will be created.
    configuration : Configuration, optional
        The configuration to be used for smashing. If None, a default configuration will be created.
    """

    def __init__(
        self,
        max_batch_size: int | None = None,
        batch_size: int = 1,
        device: str = "cuda",
        cache_dir_prefix: str = os.path.join(os.path.expanduser("~"), ".cache", "pruna"),
        configuration: Configuration | None = None,
    ) -> None:
        SMASH_SPACE.gather_algorithm_buffer()
        self._configuration: Configuration = (
            SMASH_SPACE.get_default_configuration() if configuration is None else configuration
        )
        self.config_space: ConfigurationSpace = self._configuration.config_space

        if max_batch_size is not None:
            warn(
                "max_batch_size is deprecated and will be removed in v0.2.8. Please use batch_size instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            batch_size = max_batch_size

        self.cache_dir_prefix = cache_dir_prefix
        if not os.path.exists(cache_dir_prefix):
            os.makedirs(cache_dir_prefix)
        self.cache_dir = tempfile.mkdtemp(dir=cache_dir_prefix)

        self.save_fns: list[str] = []
        self.load_fns: list[str] = []
        self.reapply_after_load: dict[str, str | None] = dict.fromkeys(ALGORITHM_GROUPS)
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.processor: ProcessorMixin | None = None
        self.data: PrunaDataModule | None = None

        # ensure the cache directory is deleted on program exit
        atexit.register(self.cleanup_cache_dir)

        # register defaults of environment configuration
        self.configure_environment(batch_size=batch_size, device=device)

    def __del__(self) -> None:
        """Delete the SmashConfig object."""
        self.cleanup_cache_dir()

    def __eq__(self, other: Any) -> bool:
        """Check if two SmashConfigs are equal."""
        if not isinstance(other, self.__class__):
            return False

        smash_config_args = deepcopy(ADDITIONAL_ARGS)
        smash_config_args.remove("cache_dir")

        args_equal = all(getattr(self, arg) == getattr(other, arg) for arg in smash_config_args)
        return args_equal and self._configuration == other._configuration

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

        # do not save the old cache directory or device
        if "cache_dir" in config_dict:
            del config_dict["cache_dir"]

        # Save the updated dictionary back to a JSON file
        with open(os.path.join(path, SMASH_CONFIG_FILE_NAME), "w") as f:
            json.dump(config_dict, f, indent=4)

        if self.tokenizer:
            self.tokenizer.save_pretrained(os.path.join(path, TOKENIZER_SAVE_PATH))
        if self.processor:
            self.processor.save_pretrained(os.path.join(path, PROCESSOR_SAVE_PATH))
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
        # since this function is only used for loading algorithm settings, we will ignore additional arguments
        filtered_config_dict = {k: v for k, v in config_dict.items() if k not in ADDITIONAL_ARGS}
        discarded_args = [k for k in config_dict if k in ADDITIONAL_ARGS]
        if discarded_args:
            pruna_logger.info(f"Discarded arguments: {discarded_args}")

        # first load the algorithm settings
        # otherwise fine-grained hyperparameters will not be active yet and we can not set them
        # lambda returns False for keys in ALGORITHM_GROUPS (and False sorts before True)
        for k, v in sorted(filtered_config_dict.items(), key=lambda item: item[0] not in ALGORITHM_GROUPS):
            self.__setitem__(k, v)

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
        self.reapply_after_load = dict.fromkeys(ALGORITHM_GROUPS)

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

    def configure_environment(
        self,
        device: str = "cuda",
        batch_size: int = 1,
        saveable_model: bool = True,
        calibration_samples: int = 16,
        call_function_name: str = "__call__",
        max_seq_len: int = 512,
        diffusion_steps_arg_name: str = "num_inference_steps",
        diffusion_backbone_attr_name: str = "model",
        diffusion_backbone_call_method: str = "__call__",
        enable_cpu_offload: bool = False,
        external_logging: str = "none",
    ) -> None:
        """
        Provide additional information to adapt smashing to a custom pipelines or specific inference constraints.

        Parameters
        ----------
        device : str, default="cuda"
            The device for which the model is optimized. Options are "cuda" and "cpu".
        batch_size : int, default=1
            The inference batch size to optimize the model for. Attention: some algorithms optimize the model for exactly
            this batch size and changing this at inference time might lead to less performance gains.
        saveable_model : bool, default=True
            Whether the model needs to be saveable after optimization, e.g. with ``smashed_model.save_pretrained()``.
            Setting this to false can lead to faster smashing.
        calibration_samples : int, default=16
            The number of calibration samples to be used for algorithms that perform calibration.
        call_function_name : str, default="__call__"
            The name of the "call" function of a given base model.
        max_seq_len : int, default=512
            The maximum sequence length for text generation models.
        diffusion_steps_arg_name : str, default="num_inference_steps"
            The name of the argument that specifies the number of diffusion steps for diffusion models.
        diffusion_backbone_attr_name : str, default="model"
            The name of the attribute that points to the backbone of a diffusion model.
        diffusion_backbone_call_method : str, default="__call__"
            The name of the method that calls the backbone of a diffusion model.
        enable_cpu_offload : bool, default=False
            Whether the CPU offload can be enabled if required by the algorithm.
        external_logging : str, default="none"
            Whether external logging is set up. Options are "none", "wandb", "tensorboard".

        Examples
        --------
        >>> from pruna import SmashConfig
        >>> config = SmashConfig()
        >>> config.configure_environment(saveable_model=True, calibration_samples=16)
        >>> # or
        >>> config["saveable_model"] = False
        >>> config["calibration_samples"] = 64
        """
        if device in ["cuda", "cpu"]:
            self.device = device
        else:
            pruna_logger.error("device must be 'cuda' or 'cpu'.")

        # aggregate checks for boolean arguments
        for arg in ["saveable_model", "enable_cpu_offload"]:
            if isinstance(locals()[arg], bool):
                setattr(self, arg, locals()[arg])
            else:
                pruna_logger.error(f"{arg} must be a boolean.")

        # aggregate checks for positive integer arguments
        for arg in ["batch_size", "calibration_samples"]:
            if isinstance(locals()[arg], int) and locals()[arg] > 0:
                setattr(self, arg, locals()[arg])
            else:
                pruna_logger.error(f"{arg} must be a positive integer.")
                # internal variable to indicated that a model has been smashed for a specific batch size
                # set this on the first initialization of these attributes
                if arg == "batch_size" and not hasattr(self, "__locked_batch_size"):
                    self.__locked_batch_size = False

        # aggregate checks for string arguments
        for arg in ["call_function_name", "diffusion_steps_arg_name", "diffusion_backbone_attr_name"]:
            if isinstance(locals()[arg], str):
                setattr(self, arg, locals()[arg])
            else:
                pruna_logger.error(f"{arg} must be a string.")

        if isinstance(external_logging, str) and external_logging in ["none", "wandb", "tensorboard"]:
            self.external_logging = external_logging
        else:
            pruna_logger.error("external_logging must be a string from ['none', 'wandb', 'tensorboard'].")

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
        updated_name: str | None = name
        if updated_name in ENV_ARGUMENTS:
            current_settings = {arg: getattr(self, arg) for arg in ENV_ARGUMENTS if arg != updated_name}
            self.configure_environment(**{updated_name: value, **current_settings})
        elif updated_name in ADDITIONAL_ARGS:
            return setattr(self, updated_name, value)
        else:
            updated_name, value = self.check_deprecation(updated_name, value)
            if updated_name is not None:
                return self._configuration.__setitem__(updated_name, value)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: D105
        if name == "_prepare_saving":
            warn(
                "The _prepare_saving attribute is deprecated and will be removed in v0.2.8. Please configure this "
                "setting by calling smash_config.configure_environment(saveable_model=True/False).",
                DeprecationWarning,
                stacklevel=2,
            )
            self.saveable_model = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, attr: str) -> object:  # noqa: D105
        if attr in ADDITIONAL_ARGS:
            return self.__dict__.get(attr)
        if attr == "_data":
            return self.__dict__.get("_data")
        if attr == "_configuration":
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

    def check_deprecation(self, name: str | None, value: Any) -> tuple[str | None, Any]:
        """
        Check for deprecation of the given name and value before setting an attribute or item.

        Parameters
        ----------
        name : str
            The name of the attribute or item to set.
        value : Any
            The value to set for the attribute or item.

        Returns
        -------
        tuple[str | None, Any]
            The updated name and value of the attribute or item, returns None if the item no longer needs to be set.
        """
        if name in [
            "whisper_s2t_batch_size",
            "ifw_batch_size",
            "higgs_example_batch_size",
            "diffusers_higgs_example_batch_size",
            "torch_compile_batch_size",
        ]:
            warn(
                f"{name} is deprecated and will be removed in v0.2.8. Use SmashConfig(batch_size={value}) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.batch_size = value
            name = None

        if name in ["diffusers_int8_enable_fp32_cpu_offload", "llm_int8_enable_fp32_cpu_offload"]:
            warn(
                (
                    f"{name} is deprecated and will be removed in v0.2.8. "
                    f"Use smash_config.configure_environment(enable_cpu_offload={value})."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            self.enable_cpu_offload = value
            name = None

        elif name == "torch_structured_calibration_samples":
            warn(
                (
                    f"{name} is deprecated and will be removed in v0.2.8. "
                    "Use smash_config.configure_environment(calibration_samples={value})."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            self.calibration_samples = value
            name = None

        elif name in ["torch_compile_max_kv_cache_size", "torch_compile_seqlen_manual_cuda_graph"]:
            warn(
                f"{name} is deprecated and will be removed in v0.2.8. They are now determined automatically.",
                DeprecationWarning,
                stacklevel=2,
            )
            name = None

        elif name == "_prepare_saving":
            warn(
                "The _prepare_saving attribute is deprecated and will be removed in v0.2.8. Please configure this "
                "setting by calling smash_config.configure_environment(saveable_model=True/False).",
                DeprecationWarning,
                stacklevel=2,
            )
            self.saveable_model = value
            name = None

        return name, value


ENV_ARGUMENTS = list(inspect.signature(SmashConfig.configure_environment).parameters)
ENV_ARGUMENTS.remove("self")
ADDITIONAL_ARGS.extend(ENV_ARGUMENTS)


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
