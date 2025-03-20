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

from pathlib import Path
from typing import Any, List, Tuple

import torch

from pruna.config.smash_config import SmashConfig
from pruna.engine.handler.handler_utils import register_inference_handler
from pruna.engine.load import load_pruna_model, load_pruna_model_from_hub
from pruna.engine.save import save_pruna_model, save_pruna_model_to_hub
from pruna.engine.utils import get_nn_modules, move_to_device, set_to_eval
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
        self, batch: Tuple[List[str] | torch.Tensor, ...], device: torch.device | str
    ) -> Any:
        """
        Run inference on the model.

        Parameters
        ----------
        batch : Tuple[List[str] | torch.Tensor, ...]
            The batch to run inference on.
        device : torch.device | str
            The device to run inference on.

        Returns
        -------
        Any
            The processed output.
        """
        batch = self.inference_handler.move_inputs_to_device(batch, device)  # type: ignore
        prepared_inputs = self.inference_handler.prepare_inputs(batch)
        outputs = self(prepared_inputs, **self.inference_handler.model_args)
        outputs = self.inference_handler.process_output(outputs)
        return outputs

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

    def move_to_device(self, device: str | torch.device) -> None:
        """
        Move the model to a specific device.

        Parameters
        ----------
        device : str | torch.device
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
        increment_counter(
            "save_pretrained", success=True, smash_config=repr(self.smash_config)
        )

    def save_to_hub(
        self,
        repo_id: str,
        folder_path: str | Path,
        *,
        revision: str | None = None,
        private: bool = False,
        allow_patterns: List[str] | str | None = None,
        ignore_patterns: List[str] | str | None = None,
        num_workers: int | None = None,
        print_report: bool = False,
        print_report_every: int = 0,
    ) -> None:
        """Save the model to the specified repository.

        Parameters
        ----------
        repo_id : str
            The repository ID to save the model to.
        folder_path : str | Path
            The folder path to save the model to.
        """
        save_pruna_model_to_hub(
            model=self.model,
            smash_config=self.smash_config,
            repo_id=repo_id,
            folder_path=folder_path,
            revision=revision,
            private=private,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            num_workers=num_workers,
            print_report=print_report,
            print_report_every=print_report_every,
        )
        increment_counter(
            "save_to_hub", success=True, smash_config=repr(self.smash_config)
        )

    @staticmethod
    @track_usage
    def from_pretrained(model_path: str, verbose: bool = False, **kwargs: Any) -> Any:
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
    def from_hub(repo_id: str, token: str | None = None) -> Any:
        """
        Load a `PrunaModel` from the specified repository.
        """
        model, smash_config = load_pruna_model_from_hub(repo_id, token)
        return PrunaModel(model=model, smash_config=smash_config)

    def destroy(self) -> None:
        """Destroy model."""
        pass
