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
import json
import os
import shutil
import tempfile
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, List

import torch
import transformers
from huggingface_hub import upload_large_folder

from pruna.config.smash_config import SMASH_CONFIG_FILE_NAME, SmashConfig
from pruna.engine.load import (
    LOAD_FUNCTIONS,
    PICKLED_FILE_NAME,
    PIPELINE_INFO_FILE_NAME,
    SAVE_BEFORE_SMASH_CACHE_DIR,
)
from pruna.engine.model_checks import get_helpers
from pruna.logging.logger import pruna_logger


def save_pruna_model(model: Any, model_path: str, smash_config: SmashConfig) -> None:
    """
    Save the model to the specified directory.

    Parameters
    ----------
    model : Any
        The model to save.
    model_path : str
        The directory to save the model to.
    smash_config : SmashConfig
        The SmashConfig object containing the save and load functions.
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # in the case of no specialized save functions, we use the model's original save function
    if len(smash_config.save_fns) == 0:
        pruna_logger.debug("Using model's original save function...")
        save_fn = original_save_fn

    # if save-before-move was the last operation, we simply move the already saved files, we have delt with them before
    elif smash_config.save_fns[-1] == SAVE_FUNCTIONS.save_before_apply.name:
        pruna_logger.debug("Moving saved model...")
        save_fn = save_before_apply

    # if the original save function was overwritten *once*, we can use the new save function
    elif len(smash_config.save_fns) == 1:
        pruna_logger.debug(f"Using new save function {smash_config.save_fns[-1]}...")
        save_fn = SAVE_FUNCTIONS[smash_config.save_fns[-1]]
        pruna_logger.debug(
            f"Overwriting original load function {smash_config.load_fn} with {smash_config.save_fns[-1]}..."
        )

    # in the case of multiple, specialized save functions, we default to pickled
    else:
        pruna_logger.debug(f"Several save functions stacked: {smash_config.save_fns}, defaulting to pickled")
        save_fn = SAVE_FUNCTIONS.pickled
        smash_config.load_fn = LOAD_FUNCTIONS.pickled.name
    # execute selected save function
    save_fn(model, model_path, smash_config)

    # save smash config (includes tokenizer and processor)
    smash_config.save_to_json(model_path)


def save_pruna_model_to_hub(
    model: Any,
    smash_config: SmashConfig,
    repo_id: str,
    folder_path: str | Path | None = None,
    *,
    revision: str | None = None,
    private: bool = False,
    allow_patterns: List[str] | str | None = None,
    ignore_patterns: List[str] | str | None = None,
    num_workers: int | None = None,
    print_report: bool = True,
    print_report_every: int = 60,
) -> None:
    """
    Save the model to the specified directory.

    Parameters
    ----------
    model : Any
        The model to save.
    smash_config : SmashConfig
        The SmashConfig object containing the save and load functions.
    repo_id : str
        The repository ID.
    folder_path : str | Path | None, optional
        The folder path to save the model to.
    revision : str | None, optional
        The revision of the model.
    private : bool, optional
        Whether the model is private.
    allow_patterns : List[str] | str | None, optional
        The allow patterns.
    ignore_patterns : List[str] | str | None, optional
        The ignore patterns.
    num_workers : int | None, optional
        The number of workers.
    print_report : bool, optional
        Whether to print the report.
    print_report_every : int, optional
        The print report every.

    Returns
    -------
    None
    """
    # Create a temporary directory within the specified folder path to store the model files
    with tempfile.TemporaryDirectory(dir=folder_path) as temp_dir:
        # Save the model and its configuration to the temporary directory
        save_pruna_model(model=model, model_path=temp_dir, smash_config=smash_config)

        # Open and load the model configuration and smash configuration data from their respective files
        model_config_path = Path(temp_dir) / "config.json"
        smash_config_path = Path(temp_dir) / "smash_config.json"
        with model_config_path.open() as config_file, smash_config_path.open() as smash_config_file:
            model_config = json.load(config_file)
            smash_config_data = json.load(smash_config_file)

        # Format the content for the README using the template and the loaded configuration data
        template_path = Path(__file__).parent / "model_card_template.md"
        template = template_path.read_text()
        content = template.format(
            repo_id=repo_id,
            model_config=json.dumps(model_config, indent=4),
            smash_config=json.dumps(smash_config_data, indent=4),
            library_name=smash_config_data["load_fn"],
        )

        # Define the path for the README file and write the formatted content to it
        readme_path = Path(temp_dir) / "README.md"
        readme_path.write_text(content)

        # Upload the contents of the temporary directory to the specified repository on the hub
        upload_large_folder(
            repo_id=repo_id,
            folder_path=temp_dir,
            repo_type="model",
            revision=revision,
            private=private,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            num_workers=num_workers,
            print_report=print_report,
            print_report_every=print_report_every,
        )


def original_save_fn(model: Any, model_path: str, smash_config: SmashConfig) -> None:
    """
    Save the model to the specified directory.

    Parameters
    ----------
    model : Any
        The model to save.
    model_path : str
        The directory to save the model to.
    smash_config : SmashConfig
        The SmashConfig object containing the save and load functions.
    """
    # catch any huggingface diffuser or transformer model and record which load function to use
    if "diffusers" in model.__module__:
        smash_config.load_fn = LOAD_FUNCTIONS.diffusers.name
        model.save_pretrained(model_path)

    elif "transformers" in model.__module__:
        smash_config.load_fn = LOAD_FUNCTIONS.transformers.name
        model.save_pretrained(model_path)

        # if the model is a transformers pipeline, we additionally save the pipeline info
        if isinstance(model, transformers.Pipeline):
            save_pipeline_info(model, model_path)

    # otherwise, resort to pickled saving
    else:
        save_pickled(model, model_path, smash_config)
        smash_config.load_fn = LOAD_FUNCTIONS.pickled.name


def save_pipeline_info(pipeline_obj: Any, save_directory: str) -> None:
    """
    Save pipeline information to a JSON file in the specified directory for easy loading.

    Parameters
    ----------
    pipeline_obj : Any
        The pipeline object to save.
    save_directory : str
        The directory to save the pipeline information to.
    """
    pruna_logger.info(f"Detected pipeline, saving info to {PIPELINE_INFO_FILE_NAME}")
    info = {
        "pipeline_type": type(pipeline_obj).__name__,
        "task": pipeline_obj.task,
    }

    filepath = os.path.join(save_directory, PIPELINE_INFO_FILE_NAME)

    with open(filepath, "w") as f:
        json.dump(info, f)


def save_before_apply(model: Any, model_path: str, smash_config: SmashConfig) -> None:
    """
    Save the model by moving already saved, temporary files into the model path.

    Parameters
    ----------
    model : Any
        The model to save.
    model_path : str
        The directory to save the model to.
    smash_config : SmashConfig
        The SmashConfig object containing the save and load functions.
    """
    save_dir = os.path.join(smash_config.cache_dir, SAVE_BEFORE_SMASH_CACHE_DIR)

    # load old smash config to get load_fn assigned previously
    # load json directly from file
    with open(os.path.join(save_dir, SMASH_CONFIG_FILE_NAME), "r") as f:
        old_smash_config = json.load(f)
    smash_config.load_fn = deepcopy(old_smash_config["load_fn"])
    del old_smash_config

    # move files in save dir into model path
    for file in os.listdir(save_dir):
        shutil.move(os.path.join(save_dir, file), os.path.join(model_path, file))


def save_pickled(model: Any, model_path: str, smash_config: SmashConfig) -> None:
    """
    Save the model by pickling it.

    Parameters
    ----------
    model : Any
        The model to save.
    model_path : str
        The directory to save the model to.
    smash_config : SmashConfig
        The SmashConfig object containing the save and load functions.
    """
    # helpers can not be pickled, we will disable and just reapply them later
    smash_helpers = get_helpers(model)
    for helper in smash_helpers:
        getattr(model, helper).disable()
    torch.save(model, os.path.join(model_path, PICKLED_FILE_NAME))
    smash_config.load_fn = LOAD_FUNCTIONS.pickled.name


def save_model_hqq(model: Any, model_path: str, smash_config: SmashConfig) -> None:
    """
    Save the model with HQQ functionality.

    Parameters
    ----------
    model : Any
        The model to save.
    model_path : str
        The directory to save the model to.
    smash_config : SmashConfig
        The SmashConfig object containing the save and load functions.
    """
    from hqq.engine.hf import HQQModelForCausalLM
    from hqq.models.hf.base import AutoHQQHFModel

    if isinstance(model, HQQModelForCausalLM):
        model.save_quantized(model_path)
    else:
        AutoHQQHFModel.save_quantized(model, model_path)

    smash_config.load_fn = LOAD_FUNCTIONS.hqq.name


def save_model_hqq_diffusers(model: Any, model_path: str, smash_config: SmashConfig) -> None:
    """
    Save the pipeline by saving the quantized model with HQQ, and rest of the pipeline with diffusers.

    Parameters
    ----------
    model : Any
        The model to save.
    model_path : str
        The directory to save the model to.
    smash_config : SmashConfig
        The SmashConfig object containing the save and load functions.
    """
    from pruna.algorithms.quantization.hqq_diffusers import (
        HQQDiffusersQuantizer,
        construct_base_class,
    )

    hf_quantizer = HQQDiffusersQuantizer()
    auto_hqq_hf_diffusers_model = construct_base_class(hf_quantizer.import_algorithm_packages())
    if hasattr(model, "transformer"):
        # save the backbone
        auto_hqq_hf_diffusers_model.save_quantized(model.transformer, os.path.join(model_path, "backbone_quantized"))
        transformer_backup = model.transformer
        model.transformer = None
        # save the rest of the pipeline
        model.save_pretrained(model_path)
        model.transformer = transformer_backup
    elif hasattr(model, "unet"):
        # save the backbone
        auto_hqq_hf_diffusers_model.save_quantized(model.unet, os.path.join(model_path, "backbone_quantized"))
        unet_backup = model.unet
        model.unet = None
        # save the rest of the pipeline
        model.save_pretrained(model_path)
        model.unet = unet_backup
    else:
        auto_hqq_hf_diffusers_model.save_quantized(model, model_path)
    smash_config.load_fn = LOAD_FUNCTIONS.hqq_diffusers.name


def save_quantized(model: Any, model_path: str, smash_config: SmashConfig) -> None:
    """
    Save the model by saving the quantized model.

    Parameters
    ----------
    model : Any
        The model to save.
    model_path : str
        The directory to save the model to.
    smash_config : SmashConfig
        The SmashConfig object containing the save and load functions.
    """
    model.save_quantized(model_path)
    smash_config.load_fn = LOAD_FUNCTIONS.awq_quantized.name


def reapply(model: Any, model_path: str, smash_config: SmashConfig) -> None:
    """
    Reapply the model.

    Parameters
    ----------
    model : Any
        The model to reapply.
    model_path : str
        The directory to reapply the model to.
    smash_config : SmashConfig
        The SmashConfig object containing the save and load functions.
    """
    raise ValueError("Reapply function is not a save function to call directly")


class SAVE_FUNCTIONS(Enum):  # noqa: N801
    """
    Enumeration of save functions for different model types.

    This enum provides callable functions for saving different types of models,
    including pickled models, IPEX LLM models, HQQ models, AWQ quantized models,
    and models that need to be saved before applying transformations.

    Parameters
    ----------
    value : callable
        The save function to be called.
    names : str
        The name of the enum member.
    module : str
        The module where the enum is defined.
    qualname : str
        The qualified name of the enum.
    type : type
        The type of the enum.
    start : int
        The start index for auto-numbering enum values.

    Examples
    --------
    >>> SAVE_FUNCTIONS.pickled(model, save_path, smash_config)
    # Model saved to disk in pickled format
    """

    pickled = partial(save_pickled)
    hqq = partial(save_model_hqq)
    hqq_diffusers = partial(save_model_hqq_diffusers)
    awq_quantized = partial(save_quantized)
    save_before_apply = partial(save_before_apply)
    reapply = partial(reapply)

    def __call__(self, *args, **kwargs) -> None:
        """
        Call the save function.

        Parameters
        ----------
        args : Any
            The arguments to pass to the save function.
        kwargs : Any
            The keyword arguments to pass to the save function.
        """
        if self.value is not None:
            self.value(*args, **kwargs)
