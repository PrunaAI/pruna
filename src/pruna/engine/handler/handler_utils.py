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

import inspect
from typing import Any

from pruna.engine.handler.handler_diffuser import DiffuserHandler
from pruna.engine.handler.handler_inference import InferenceHandler
from pruna.engine.handler.handler_pipeline import PipelineHandler
from pruna.engine.handler.handler_standard import StandardHandler
from pruna.engine.handler.handler_transformer import TransformerHandler

HANDLER_EXCEPTIONS: dict[type[InferenceHandler], list[str]] = {
    TransformerHandler: ["AutoHQQHFModel", "TranslatorWrapper", "GeneratorWrapper", "GPTQ"],
    DiffuserHandler: ["AutoHQQHFDiffusersModel"],
}


def register_inference_handler(model: Any) -> InferenceHandler:
    """
    Register an inference handler for the model. The handler is chosen based on the model type.

    Parameters
    ----------
    model : Any
        The model to register a handler for.

    Returns
    -------
    InferenceHandler
        The registered handler.
    """
    handler = scan_for_exceptions(model)

    if handler is not None:
        return handler

    model_module = model._orig_mod.__module__ if hasattr(model, "_orig_mod") else model.__module__

    # Prefer diffusers handler first to avoid routing diffusers pipelines to generic pipeline handler
    if "diffusers" in model_module:
        return DiffuserHandler(call_signature=inspect.signature(model.__call__))

    # Transformers models and pipelines
    if "transformers" in model_module and "Pipeline" in type(model).__name__:
        # Specific check for text generation pipelines
        if "TextGeneration" in type(model).__name__:
            return PipelineHandler(pipeline=model)
        else:
            raise ValueError("Unsupported pipeline type. Only text generation pipelines are currently supported.")
    elif "transformers" in model_module:
        return TransformerHandler()
    else:
        return StandardHandler()


def scan_for_exceptions(model: Any) -> InferenceHandler | None:
    """
    Scan for exceptions in the model and return the appropriate handler.

    Parameters
    ----------
    model : Any
        The model to scan for exceptions.

    Returns
    -------
    InferenceHandler | None
        The handler if an exception is found, otherwise None.
    """
    # instead of checking with isinstance for the class itself we check the module name
    # this avoids directly importing external packages
    for handler, model_classes in HANDLER_EXCEPTIONS.items():
        for model_class in model_classes:
            if model.__class__.__name__ == "OptimizedModule":  # torch_compile abstracts over the model class.
                name = model._orig_mod.__class__.__name__
            else:
                name = model.__class__.__name__
            if model_class in name:
                return handler()
    return None
