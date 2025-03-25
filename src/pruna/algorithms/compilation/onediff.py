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

import inspect
import os
from typing import Any, Dict

from ConfigSpace import Constant

from pruna.algorithms.compilation import PrunaCompiler
from pruna.config.smash_config import SmashConfigPrefixWrapper
from pruna.engine.model_checks import get_diffusers_transformer_models


class OneDiffCompiler(PrunaCompiler):
    """
    Implement OneDiff compilation using the onediff library.

    OneDiff achieves acceleration by converting diffusion model modules into optimized static graphs via PyTorch
    module compilation. This process fuses operations, applies low-level GPU kernel optimizations, and supports
    dynamic input shapes without the overhead of re-compilation.
    """

    algorithm_name = "onediff"
    references = {"GitHub": "https://github.com/siliconflow/onediff"}
    tokenizer_required = False
    processor_required = False
    run_on_cpu = False
    run_on_cuda = True
    dataset_required = False
    compatible_algorithms = dict(quantizer=["half"])
    required_install = "``pip install pruna[onediff]``"

    def get_hyperparameters(self) -> list:
        """
        Get the hyperparameters for the algorithm.

        Returns
        -------
        list
            The hyperparameters.
        """
        return [Constant("backend", value="nexfort")]

    def model_check_fn(self, model: Any) -> bool:
        """
        Check if the model is a valid model for the algorithm.

        Parameters
        ----------
        model : Any
            The model to check.

        Returns
        -------
        bool
            True if the model is a valid model for the algorithm, False otherwise.
        """
        transformer_models = get_diffusers_transformer_models()

        if isinstance(model, tuple(transformer_models)):
            return True

        for _, attr_value in inspect.getmembers(model):
            if isinstance(attr_value, tuple(transformer_models)):
                return True

        return False

    def _apply(self, model: Any, smash_config: SmashConfigPrefixWrapper) -> Any:
        """
        Compile the model.

        Parameters
        ----------
        model : Any
            The model to compile.
        smash_config : SmashConfigPrefixWrapper
            The configuration for the compilation.

        Returns
        -------
        Any
            The compiled model.
        """
        imported_modules = self.import_algorithm_packages()
        transformer_models = get_diffusers_transformer_models()

        if isinstance(model, tuple(transformer_models)):
            model.onediff_compiler = OnediffWrapper(model, smash_config, imported_modules)
            model = model.onediff_compiler.compile()

        else:
            for attr_name, attr_value in inspect.getmembers(model):
                if isinstance(attr_value, tuple(transformer_models)):
                    working_model = getattr(model, attr_name)
                    working_model.onediff_compiler = OnediffWrapper(working_model, smash_config, imported_modules)
                    working_model = working_model.onediff_compiler.compile()
                    setattr(model, attr_name, working_model)
        return model

    def import_algorithm_packages(self) -> Dict[str, Any]:
        """
        Import the algorithm packages.

        Returns
        -------
        Dict[str, Any]
            The algorithm packages.
        """
        try:
            from onediff.infer_compiler import compile as compile_onediff

            os.environ["NEXFORT_FUSE_TIMESTEP_EMBEDDING"] = "0"
            os.environ["NEXFORT_FX_FORCE_TRITON_SDPA"] = "1"
        except ImportError:
            raise ImportError(f"Onediff is not installed. Please install it using {self.required_install}.")

        return dict(compile=compile_onediff)


class OnediffWrapper:
    """
    Compilation Wrapper for Onediff compilation.

    Parameters
    ----------
    model : Any
        The model to be compiled.
    smash_config : SmashConfigPrefixWrapper
        Configuration dictionary for the compilation process.
    imported_modules : Dict[str, Any]
        Dictionary containing the imported modules.
    """

    def __init__(self, model: Any, smash_config: SmashConfigPrefixWrapper, imported_modules: Dict[str, Any]) -> None:
        self.model = model
        self.smash_config = smash_config
        self.imported_modules = imported_modules

    def compile(self) -> Any:
        """
        Compile the model using the onediff compiler.

        Returns
        -------
        Any
            The compiled model.
        """
        self.model = self.imported_modules["compile"](
            self.model, backend=self.smash_config["backend"], options='{"mode": "O3"}'
        )
        return self.model
