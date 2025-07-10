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

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from huggingface_hub import constants
from tqdm.auto import tqdm as base_tqdm

from pruna.config.smash_config import SmashConfig
from pruna.engine.handler.handler_utils import register_inference_handler
from pruna.engine.load import load_pruna_model, load_pruna_model_from_hub
from pruna.engine.save import save_pruna_model, save_pruna_model_to_hub
from pruna.engine.utils import get_nn_modules, move_to_device, set_to_best_available_device, set_to_eval
from pruna.logging.filter import apply_warning_filter
from pruna.telemetry import increment_counter, track_usage


class PrunaModel:
    """
    A pruna class wrapping any model.

    Parameters
    ----------
    model : Any
        The model to be held by this class.
    smash_config : SmashConfig | None
        Smash configuration.
    """

    def __init__(
        self,
        model: Any,
        smash_config: SmashConfig | None = None,
    ) -> None:
        self.model: Any | None = model
        self.smash_config = smash_config if smash_config is not None else SmashConfig()
        self.inference_handler = register_inference_handler(self.model)

    @track_usage
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the smashed model.

        Parameters
        ----------
        *args : Any
            Arguments to pass to the model.
        **kwargs : Any
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The output of the model's prediction.
        """
        if self.model is None:
            raise ValueError("No more model available, this model is likely destroyed.")
        else:
            with torch.no_grad():
                return self.model.__call__(*args, **kwargs)

    def run_inference(
        self, batch: Any, device: torch.device | str | None = None
    ) -> Any:
        """
        Run inference on the model.

        Parameters
        ----------
        batch : Any
            The batch to run inference on.
        device : torch.device | str | None
            The device to run inference on. If None, the best available device will be used.

        Returns
        -------
        Any
            The processed output.
        """
        device = set_to_best_available_device(device)
        batch = self.inference_handler.move_inputs_to_device(batch, device)  # type: ignore

        if not isinstance(batch, tuple):
            batch = (batch, {})
        prepared_inputs = self.inference_handler.prepare_inputs(batch)

        if prepared_inputs is None:
            outputs = self(**self.inference_handler.model_args)
        elif isinstance(prepared_inputs, dict):
            outputs = self(**prepared_inputs, **self.inference_handler.model_args)
        else:
            outputs = self(prepared_inputs, **self.inference_handler.model_args)
        outputs = self.inference_handler.process_output(outputs)
        return outputs

    def is_instance(self, instance_type: type) -> bool:
        """
        Compare the model to the given instance type.

        Parameters
        ----------
        instance_type : type
            The type to compare the model to.

        Returns
        -------
        bool
            True if the model is an instance of the given type, False otherwise.
        """
        return isinstance(self.model, instance_type)

    def __getattr__(self, attr: str) -> Any:
        """
        Forward attribute access to the underlying model.

        Parameters
        ----------
        attr : str
            The name of the attribute to access.

        Returns
        -------
        Any
            The value of the requested attribute in the underlying model.
        """
        if self.model is None:
            raise ValueError("No more model available, this model is likely destroyed.")
        else:
            return getattr(self.model, attr)

    def __delattr__(self, attr: str) -> None:
        """
        Delete an attribute from the model.

        Parameters
        ----------
        attr : str
            The attribute to delete.
        """
        delattr(self.model, attr)

    def get_nn_modules(self) -> dict[str | None, torch.nn.Module]:
        """
        Get the nn.Module instances in the model.

        Returns
        -------
        dict[str | None, torch.nn.Module]
            A dictionary of the nn.Module instances in the model.
        """
        return get_nn_modules(self.model)

    def set_to_eval(self) -> None:
        """Set the model to evaluation mode."""
        set_to_eval(self.model)

    def move_to_device(self, device: str) -> None:
        """
        Move the model to a specific device.

        Parameters
        ----------
        device : str
            The device to move the model to.
        """
        move_to_device(self.model, device)

    def save_pretrained(self, model_path: str) -> None:
        """
        Save the smashed model to the specified model path.

        Parameters
        ----------
        model_path : str
            The path to the directory where the model will be saved.
        """
        save_pruna_model(self.model, model_path, self.smash_config)
        increment_counter("save_pretrained", success=True, smash_config=repr(self.smash_config))

    def save_to_hub(
        self,
        repo_id: str,
        *,
        model_path: str | None = None,
        revision: str | None = None,
        private: bool = False,
        allow_patterns: List[str] | str | None = None,
        ignore_patterns: List[str] | str | None = None,
        num_workers: int | None = None,
        print_report: bool = False,
        print_report_every: int = 0,
    ) -> None:
        """
        Save the model to the specified repository.

        Parameters
        ----------
        repo_id : str
            The repository ID to save the model to.
        model_path : str | None
            The path to the directory where the model will be saved.
            If None, the model will only be saved to the the Hugging Face Hub.
        revision : str | None
            The revision to save the model to.
        private : bool
            Whether to save the model as a private repository.
        allow_patterns : List[str] | str | None
            The patterns to allow to save the model.
        ignore_patterns : List[str] | str | None
            The patterns to ignore to save the model.
        num_workers : int | None
            The number of workers to use to save the model.
        print_report : bool
            Whether to print the report of the saved model.
        print_report_every : int
            The number of steps to print the report of the saved model.
        """
        save_pruna_model_to_hub(
            instance=self,
            model=self.model,
            smash_config=self.smash_config,
            repo_id=repo_id,
            model_path=model_path,
            revision=revision,
            private=private,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            num_workers=num_workers,
            print_report=print_report,
            print_report_every=print_report_every,
        )
        increment_counter("save_to_hub", success=True, smash_config=repr(self.smash_config))

    @staticmethod
    @track_usage
    def from_pretrained(model_path: str, verbose: bool = False, **kwargs: Any) -> PrunaModel:
        """
        Load a `PrunaModel` from the specified model path.

        Parameters
        ----------
        model_path : str
            The path to the model directory containing necessary configuration and model files.
        verbose : bool, optional
            Whether to apply warning filters to suppress warnings. Defaults to False.
        **kwargs : dict
            Additional keyword arguments to pass to the model loading function, such as specific settings or parameters.

        Returns
        -------
        PrunaModel
            The loaded `PrunaModel` instance.
        """
        if not verbose:
            apply_warning_filter()

        model, smash_config = load_pruna_model(model_path, **kwargs)

        if not isinstance(model, PrunaModel):
            model = PrunaModel(model=model, smash_config=smash_config)
        else:
            model.smash_config = smash_config
        return model

    @staticmethod
    @track_usage
    def from_hub(
        repo_id: str,
        revision: Optional[str] = None,
        cache_dir: Union[str, Path, None] = None,
        local_dir: Union[str, Path, None] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        user_agent: Optional[Union[Dict, str]] = None,
        proxies: Optional[Dict] = None,
        etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
        force_download: bool = False,
        token: Optional[Union[bool, str]] = None,
        local_files_only: bool = False,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        max_workers: int = 8,
        tqdm_class: Optional[base_tqdm] = None,
        headers: Optional[Dict[str, str]] = None,
        endpoint: Optional[str] = None,
        # Deprecated args
        local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
        resume_download: Optional[bool] = None,
        **kwargs,
    ) -> PrunaModel:
        """
        Load a `PrunaModel` from the specified repository.

        Parameters
        ----------
        repo_id : str
            The repository ID to load the model from.
        revision : str | None, optional
            The revision of the model to load.
        cache_dir : str | Path | None, optional
            The directory to cache the model in.
        local_dir : str | Path | None, optional
            The local directory to save the model in.
        library_name : str | None, optional
            The name of the library to use to load the model.
        library_version : str | None, optional
            The version of the library to use to load the model.
        user_agent : str | Dict | None, optional
            The user agent to use to load the model.
        proxies : Dict | None, optional
            The proxies to use to load the model.
        etag_timeout : float, optional
            The timeout for the etag.
        force_download : bool, optional
            Whether to force the download of the model.
        token : str | bool | None, optional
            The token to use to access the repository.
        local_files_only : bool, optional
            Whether to only load the model from the local files.
        allow_patterns : List[str] | str | None, optional
            The patterns to allow to load the model.
        ignore_patterns : List[str] | str | None, optional
            The patterns to ignore to load the model.
        max_workers : int, optional
            The maximum number of workers to use to load the model.
        tqdm_class : tqdm | None, optional
            The tqdm class to use to load the model.
        headers : Dict[str, str] | None, optional
            The headers to use to load the model.
        endpoint : str | None, optional
            The endpoint to use to load the model.
        local_dir_use_symlinks : bool | Literal["auto"], optional
            Whether to use symlinks to load the model.
        resume_download : bool | None, optional
            Whether to resume the download of the model.
        **kwargs : Any, optional
            Additional keyword arguments to pass to the model loading function.

        Returns
        -------
        PrunaModel
            The loaded `PrunaModel` instance.
        """
        model, smash_config = load_pruna_model_from_hub(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
            proxies=proxies,
            etag_timeout=etag_timeout,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
            tqdm_class=tqdm_class,
            headers=headers,
            endpoint=endpoint,
            **kwargs,
            # Deprecated args
            local_dir_use_symlinks=local_dir_use_symlinks,
            resume_download=resume_download,
        )
        if not isinstance(model, PrunaModel):
            model = PrunaModel(model=model, smash_config=smash_config)
        else:
            model.smash_config = smash_config
        return model

    def destroy(self) -> None:
        """Destroy model."""
        pass
