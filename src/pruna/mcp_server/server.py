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

"""
Model Context Protocol server exposing pruna's optimization catalogue to an LLM.

The server has two tiers of tools. Discovery and validation tools are cheap, read-only and
need neither a GPU nor a loaded model, so they are always available. Tools that actually
optimize a model are opt-in and must be enabled explicitly with ``--allow-execute``.

Run it over stdio with::

    pruna-mcp
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    _MCP_IMPORT_ERROR: ImportError | None = None
except ImportError as error:  # pragma: no cover - depends on the installed extras
    # Deferred rather than raised, so that importing this module stays possible without the
    # extra. The console script imports it before main() runs, so raising here would bury the
    # explanation under a traceback that a client shows verbatim to the user.
    _MCP_IMPORT_ERROR = error

from . import introspection, validation

INSTRUCTIONS = """
pruna is a model-optimization framework. Its entry point is `smash(model, SmashConfig)`,
where a `SmashConfig` activates one or more optimization algorithms.

Recommended workflow:

1. Call `list_tags` and `list_algorithms` to find the algorithms relevant to a goal such as
   quantization or caching.
2. Call `describe_algorithm` on each candidate. It returns every hyperparameter with its
   allowed values and default, plus which algorithms it may run before or after.
3. Call `validate_config` with the intended combination. This needs no GPU and reports
   missing pip extras, missing tokenizers and incompatible algorithm pairs immediately.
4. Only then call `smash_model`, which loads a model and optimizes it. This tool exists only
   when the server was started with --allow-execute.

Hyperparameters are always keyed by their fully qualified name, prefixed with the algorithm
name, for example `hqq_weight_bits`. Omit any hyperparameter to keep its default.
""".strip()


def _require_mcp() -> None:
    """
    Fail with actionable guidance when the mcp extra is missing.

    Raises
    ------
    ImportError
        If the server was imported without pruna[mcp] installed.
    """
    if _MCP_IMPORT_ERROR is not None:
        raise ImportError(
            "The pruna MCP server requires the 'mcp' package, which is not installed. "
            f"Install it with: pip install 'pruna[mcp]'  (original error: {_MCP_IMPORT_ERROR})"
        ) from _MCP_IMPORT_ERROR


def _require_execution(allow_execute: bool) -> None:
    """
    Guard the execution tier behind an explicit opt-in flag.

    Parameters
    ----------
    allow_execute : bool
        Whether the server was started with execution enabled.

    Raises
    ------
    ToolError
        If execution tools were called on a read-only server.
    """
    if not allow_execute:
        raise ToolError(
            "This server runs in read-only mode. Restart it with --allow-execute to enable "
            "smash_model and evaluate_model. Use validate_config to check a configuration first."
        )


def build_server(allow_execute: bool = False) -> MCPServer:
    """
    Construct the MCP server and register every tool.

    Parameters
    ----------
    allow_execute : bool
        Whether to register the tools that load and optimize models. These are omitted
        entirely when False, so a client can discover that execution is unavailable.

    Returns
    -------
    MCPServer
        The configured server, ready to be run over stdio.
    """
    _require_mcp()
    server = MCPServer(name="pruna", instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)
    executing = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False)

    @server.tool(
        annotations=read_only,
        description="List the algorithm categories, such as quantizer or cacher, with descriptions.",
    )
    def list_tags() -> dict[str, Any]:
        """
        List every algorithm category with a description of what it does.

        Returns
        -------
        dict[str, Any]
            The categories under the key "tags", each with a name and a description.
        """
        tags = introspection.list_tags()
        return {"tags": tags, "count": len(tags)}

    @server.tool(
        annotations=read_only,
        description="List all optimization algorithms. Filter by category, e.g. tag='quantizer'.",
    )
    def list_algorithms(tag: str | None = None) -> dict[str, Any]:
        """
        List the available optimization algorithms and what each one needs to run.

        Parameters
        ----------
        tag : str | None
            Optional category filter, for example "quantizer" or "cacher". Call list_tags for
            the full set. If None, every algorithm is returned.

        Returns
        -------
        dict[str, Any]
            The algorithm summaries under the key "algorithms", the applied filter under "tag"
            and their number under "count".
        """
        try:
            algorithms = introspection.list_algorithms(tag)
        except ValueError as error:
            raise ToolError(str(error)) from None
        return {"algorithms": algorithms, "tag": tag, "count": len(algorithms)}

    @server.tool(
        annotations=read_only,
        description=(
            "Describe one algorithm in full: every hyperparameter with allowed values and "
            "defaults, order constraints, and requirements. Call this before composing a config."
        ),
    )
    def describe_algorithm(name: str) -> dict[str, Any]:
        """
        Describe a single algorithm, including all of its hyperparameters.

        Parameters
        ----------
        name : str
            The algorithm name, as returned by list_algorithms.

        Returns
        -------
        dict[str, Any]
            The full algorithm description, including per-hyperparameter names, types,
            defaults and allowed ranges.
        """
        try:
            return introspection.describe_algorithm(name)
        except KeyError:
            available = len(introspection.list_algorithms())
            raise ToolError(
                f"Unknown algorithm '{name}'. Call list_algorithms to see the {available} available names."
            ) from None

    @server.tool(
        annotations=read_only,
        description="List evaluation benchmarks and the metrics each one computes.",
    )
    def list_benchmarks(task_type: str | None = None) -> dict[str, Any]:
        """
        List the built-in evaluation benchmarks.

        Parameters
        ----------
        task_type : str | None
            Optional filter such as "text_to_image". If None, every benchmark is returned.

        Returns
        -------
        dict[str, Any]
            The benchmarks under the key "benchmarks", each including the metrics it evaluates.
        """
        benchmarks = introspection.list_benchmarks(task_type)
        return {"benchmarks": benchmarks, "count": len(benchmarks)}

    @server.tool(
        annotations=read_only,
        description=(
            "List the named datasets an evaluation can run on, e.g. 'LAION256' or 'WikiText'. "
            "Pass one of these names as the 'data' argument of evaluate_model."
        ),
    )
    def list_datasets() -> dict[str, Any]:
        """
        List the named datasets available to an evaluation task.

        Returns
        -------
        dict[str, Any]
            The dataset names under the key "datasets", accepted verbatim by evaluate_model.
        """
        datasets = introspection.list_datasets()
        return {"datasets": datasets, "count": len(datasets)}

    @server.tool(
        annotations=read_only,
        description="List every evaluation metric name available to an evaluation task.",
    )
    def list_metrics() -> dict[str, Any]:
        """
        List the available evaluation metrics.

        Returns
        -------
        dict[str, Any]
            The sorted metric names usable with evaluate_model, under the key "metrics".
        """
        metrics = introspection.list_metrics()
        return {"metrics": metrics, "count": len(metrics)}

    @server.tool(
        annotations=read_only,
        description=(
            "Validate a proposed configuration WITHOUT loading a model. Reports missing pip "
            "extras, missing tokenizers/processors/datasets and incompatible algorithm pairs. "
            "Cheap and safe to call repeatedly while iterating."
        ),
    )
    def validate_config(
        algorithms: list[str],
        hyperparameters: dict[str, Any] | None = None,
        device: str | None = None,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        """
        Check whether a proposed configuration will work, without loading a model.

        Parameters
        ----------
        algorithms : list[str]
            Algorithm names to activate, as returned by list_algorithms.
        hyperparameters : dict[str, Any] | None
            Hyperparameter values keyed by fully qualified name, e.g. {"hqq_weight_bits": 4}.
            Omitted hyperparameters keep their defaults.
        device : str | None
            Optional target device such as "cpu" or "cuda".
        batch_size : int
            Inference batch size to record on the configuration.

        Returns
        -------
        dict[str, Any]
            A report with a valid flag, any errors with hints, the resolved configuration,
            the requirements the algorithms impose, and the checks that still need a model.
        """
        return validation.validate_config(algorithms, hyperparameters, device, batch_size)

    if allow_execute:

        @server.tool(
            annotations=executing,
            description=(
                "Load a base model and optimize it. This downloads weights and needs a device, "
                "so it is slow. Call validate_config first."
            ),
        )
        def smash_model(
            model_path: str,
            algorithms: list[str],
            hyperparameters: dict[str, Any] | None = None,
            loader: str = "auto",
            device: str | None = None,
            experimental: bool = False,
            save_to: str | None = None,
            batch_size: int = 1,
        ) -> dict[str, Any]:
            """
            Load a base model, optimize it with the given algorithms, and optionally save it.

            The optimized model only exists in memory, so pass ``save_to`` to write it to disk
            for later use.

            Parameters
            ----------
            model_path : str
                Hugging Face model id or local directory of the base model.
            algorithms : list[str]
                Algorithm names to apply, as returned by list_algorithms.
            hyperparameters : dict[str, Any] | None
                Hyperparameter values keyed by fully qualified name. Omitted hyperparameters
                keep their defaults.
            loader : str
                How to load the model: "transformers", "diffusers", or "auto" to try
                transformers first and fall back to diffusers.
            device : str | None
                Optional target device such as "cpu" or "cuda".
            experimental : bool
                Skip the strict compatibility checks that ``smash`` normally enforces.
            save_to : str | None
                Directory to save the optimized model to. The result is discarded if None.
            batch_size : int
                Inference batch size to record on the configuration.

            Returns
            -------
            dict[str, Any]
                The algorithms applied, the resolved configuration, and the save location if
                the model was written to disk.
            """
            _require_execution(allow_execute)
            from pruna.engine.load import load_diffusers_model, load_transformers_model
            from pruna.engine.save import save_pruna_model
            from pruna.smash import smash

            loaders = {
                "transformers": load_transformers_model,
                "diffusers": load_diffusers_model,
            }
            if loader == "auto":
                try:
                    model = load_transformers_model(model_path)
                except Exception:  # A diffusers checkpoint is not loadable by transformers
                    model = load_diffusers_model(model_path)
            elif loader in loaders:
                model = loaders[loader](model_path)
            else:
                raise ValueError(f"Unknown loader '{loader}'. Use 'auto', 'transformers' or 'diffusers'.")

            config = validation.build_config(algorithms, hyperparameters, device, batch_size)
            optimized = smash(model, config, experimental=experimental)

            saved_to = None
            if save_to is not None:
                save_pruna_model(optimized, save_to, config)
                saved_to = save_to

            return {
                "applied_algorithms": config.get_active_algorithms(),
                "resolved_hyperparameters": {
                    name: {
                        parameter.name: validation.resolve_hyperparameter(config, name, parameter)
                        for parameter in _algorithm_registry()[name].hyperparameters
                    }
                    for name in config.get_active_algorithms()
                },
                "saved_to": saved_to,
            }

        @server.tool(
            annotations=executing,
            description="Evaluate a model with the pruna evaluation agent. Needs a device and is slow.",
        )
        def evaluate_model(
            model_path: str,
            request: str | list[str],
            data: str,
            device: str | None = None,
        ) -> dict[str, Any]:
            """
            Load a model and evaluate it, returning the computed metric results.

            Parameters
            ----------
            model_path : str
                Hugging Face model id or local directory of the model to evaluate.
            request : str | list[str]
                A metric name, a list of metric names, or one of the special requests
                "image_generation_quality" and "text_generation_quality".
            data : str
                Name of the dataset to evaluate on, for example "LAION256", "ImageNet" or
                "WikiText".
            device : str | None
                Optional device to evaluate on.

            Returns
            -------
            dict[str, Any]
                The metric results reported by the evaluation agent.
            """
            _require_execution(allow_execute)
            from pruna.data.pruna_datamodule import PrunaDataModule
            from pruna.engine.load import load_diffusers_model, load_transformers_model
            from pruna.evaluation.evaluation_agent import EvaluationAgent

            try:
                model = load_transformers_model(model_path)
            except Exception:  # A diffusers checkpoint is not loadable by transformers
                model = load_diffusers_model(model_path)

            # from_string is pruna's own entry point for a named dataset; the constructor takes
            # dataset objects and has no data_name argument. An unknown name would raise a plain
            # KeyError whose text the SDK discards, so it is turned into guidance here.
            try:
                datamodule = PrunaDataModule.from_string(data)
            except Exception as error:
                raise ToolError(
                    f"Could not load dataset '{data}': {error}. "
                    f"Call list_datasets for the {len(introspection.list_datasets())} available names."
                ) from None

            agent = EvaluationAgent(request=request, datamodule=datamodule, device=device)
            results = agent.evaluate(model, model_name=model_path)

            return {"metrics": [str(result) for result in results]}

    return server


def _algorithm_registry() -> Any:
    """
    Import and return the algorithm registry.

    Returns
    -------
    Any
        The ``AlgorithmRegistry`` class from pruna.
    """
    from pruna import AlgorithmRegistry

    return AlgorithmRegistry


def main() -> None:
    """
    Command line entry point for the pruna MCP server.

    Parses arguments and runs the server over stdio.
    """
    parser = argparse.ArgumentParser(prog="pruna-mcp", description="Model Context Protocol server for pruna.")
    parser.add_argument(
        "--allow-execute",
        action="store_true",
        help="Register smash_model and evaluate_model. Off by default because they load models and need a device.",
    )
    arguments = parser.parse_args()

    try:
        server = build_server(allow_execute=arguments.allow_execute)
    except ImportError as error:
        # A client launches this as a subprocess and shows whatever lands on stderr, so a clean
        # one-line explanation is far more useful than an import traceback.
        sys.stderr.write(f"pruna-mcp: {error}\n")
        raise SystemExit(1) from None

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
