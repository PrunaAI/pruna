"""Tests for MCP tool registration and the read-only/execution split."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from pruna.mcp_server.server import build_server  # noqa: E402

READ_ONLY_TOOLS = {
    "list_tags",
    "list_algorithms",
    "describe_algorithm",
    "list_benchmarks",
    "list_metrics",
    "list_datasets",
    "validate_config",
}
EXECUTION_TOOLS = {"smash_model", "evaluate_model"}


def _tool_names(allow_execute: bool = False) -> set[str]:
    """Return the names of every tool registered on a server."""
    import anyio

    server = build_server(allow_execute=allow_execute)
    return {tool.name for tool in anyio.run(server.list_tools)}


@pytest.mark.cpu
@pytest.mark.requires_mcp
class TestToolRegistration:
    """Tests for which tools the server exposes."""

    def test_every_documented_tool_is_registered(self) -> None:
        """The documented tool list and the code must not drift apart."""
        import re
        from pathlib import Path

        docs = Path(__file__).parents[2] / "docs" / "user_manual" / "mcp_server.rst"
        documented = set(re.findall(r"^``([a-z_]+)``$", docs.read_text(encoding="utf-8"), re.M))
        assert documented <= _tool_names(allow_execute=True), (
            f"documented but not registered: {sorted(documented - _tool_names(allow_execute=True))}"
        )

    def test_read_only_mode_registers_the_discovery_tier(self) -> None:
        """The safe tools should always be available."""
        assert READ_ONLY_TOOLS <= _tool_names(allow_execute=False)

    def test_execution_tools_are_absent_by_default(self) -> None:
        """Tools that load models must not be exposed without an explicit opt-in."""
        assert not (EXECUTION_TOOLS & _tool_names(allow_execute=False))

    def test_execution_tools_appear_when_enabled(self) -> None:
        """Opting in should add the execution tier alongside the discovery tier."""
        names = _tool_names(allow_execute=True)
        assert EXECUTION_TOOLS <= names
        assert READ_ONLY_TOOLS <= names

    def test_tools_carry_input_schemas(self) -> None:
        """Every tool must expose a schema so a client can call it."""
        import anyio

        server = build_server(allow_execute=False)
        for tool in anyio.run(server.list_tools):
            assert tool.input_schema["type"] == "object"

    def test_discovery_tools_are_annotated_read_only(self) -> None:
        """Clients rely on the read-only hint to avoid prompting on safe calls."""
        import anyio

        server = build_server(allow_execute=False)
        for tool in anyio.run(server.list_tools):
            assert tool.annotations is not None
            assert tool.annotations.read_only_hint is True


@pytest.mark.cpu
@pytest.mark.requires_mcp
class TestToolExecution:
    """Tests for tools actually returning data."""

    def test_list_tags_returns_the_catalogue(self) -> None:
        """Calling the server should yield the algorithm categories."""
        import anyio

        result = anyio.run(lambda: build_server().call_tool("list_tags", {}))
        assert result.is_error is False
        assert result.structured_content["count"] > 0
        assert all({"name", "description"} <= set(tag) for tag in result.structured_content["tags"])

    def test_every_tool_returns_one_coherent_json_object(self) -> None:
        """
        Each tool must answer with a single content block holding a JSON object.

        A bare list return is serialized by the SDK as one content block per element, so a
        catalogue of 48 algorithms would arrive as 48 fragments for the model to reassemble.
        Returning a named object keeps every tool's shape uniform.
        """
        import anyio

        server = build_server()
        calls = {
            "list_tags": {},
            "list_algorithms": {},
            "list_algorithms_filtered": {"tag": "quantizer"},
            "describe_algorithm": {"name": "torch_compile"},
            "list_benchmarks": {},
            "list_metrics": {},
            "list_datasets": {},
            "validate_config": {"algorithms": ["torch_compile"]},
        }
        for label, arguments in calls.items():
            name = "list_algorithms" if label == "list_algorithms_filtered" else label
            result = anyio.run(lambda n=name, a=arguments: server.call_tool(n, a))
            assert result.is_error is False, f"{label} reported an error"
            assert len(result.content) == 1, f"{label} fragmented its answer into {len(result.content)} blocks"
            assert isinstance(result.structured_content, dict), f"{label} did not return an object"
            assert "result" not in result.structured_content, f"{label} fell back to the SDK's list wrapper"

    def test_validate_config_reports_its_result(self) -> None:
        """The validator tool should return a structured report."""
        import anyio

        result = anyio.run(
            lambda: build_server().call_tool("validate_config", {"algorithms": ["deepcache"]})
        )
        assert result.is_error is False
        assert result.structured_content["valid"] is True

    def test_describe_unknown_algorithm_keeps_the_error_message(self) -> None:
        """
        An unknown algorithm name is a routine mistake, so the guidance must reach the model.

        The SDK turns a plain exception into a generic message and drops the detail, so the
        tools raise ToolError deliberately to preserve it.
        """
        import anyio
        from mcp.server.mcpserver.exceptions import ToolError

        with pytest.raises(ToolError) as caught:
            anyio.run(lambda: build_server().call_tool("describe_algorithm", {"name": "nope"}))

        message = str(caught.value)
        assert "nope" in message
        assert "list_algorithms" in message

    def test_invalid_tag_keeps_the_error_message(self) -> None:
        """An unknown category should list the valid ones rather than fail opaquely."""
        import anyio
        from mcp.server.mcpserver.exceptions import ToolError

        with pytest.raises(ToolError) as caught:
            anyio.run(lambda: build_server().call_tool("list_algorithms", {"tag": "not_a_tag"}))

        assert "not_a_tag" in str(caught.value)

@pytest.mark.cpu
@pytest.mark.requires_mcp
class TestExecutionTierContracts:
    """
    Tests for the execution tools that do not need a model to run.

    These tools call deep into pruna behind function-level imports, so a wrong argument name or
    a renamed factory would not surface until a model was downloaded. These tests bind the
    arguments to the real signatures instead.
    """

    @staticmethod
    def _bind(function, **arguments):
        """Return the parameter names the function would actually accept."""
        import inspect

        return set(inspect.signature(function).parameters)

    def test_smash_accepts_the_configuration_the_tool_builds(self) -> None:
        """The config the tool builds has to be what smash expects."""
        from pruna.smash import smash

        assert {"model", "smash_config", "experimental"} <= self._bind(smash)

    def test_saving_accepts_the_optimized_model_and_config(self) -> None:
        """save_pruna_model takes the model, a path and the config, in that order."""
        from pruna.engine.save import save_pruna_model

        assert self._bind(save_pruna_model) >= {"model", "model_path", "smash_config"}

    def test_loaders_take_a_path(self) -> None:
        """Both loaders accept a bare path, which is what the tool passes."""
        from pruna.engine.load import load_diffusers_model, load_transformers_model

        for loader in (load_transformers_model, load_diffusers_model):
            assert self._bind(loader) >= {"path"}

    def test_a_named_dataset_resolves_through_the_public_factory(self) -> None:
        """
        evaluate_model must build its datamodule through PrunaDataModule.from_string.

        The constructor takes dataset objects and has no data_name argument, so calling it
        directly with a name raises a TypeError that the SDK would reduce to a bare
        "Error executing tool evaluate_model".
        """
        from pruna.data.pruna_datamodule import PrunaDataModule

        assert "data_name" not in self._bind(PrunaDataModule.__init__)
        assert "dataset_name" in self._bind(PrunaDataModule.from_string)

    def test_evaluation_agent_accepts_the_tool_keyword_arguments(self) -> None:
        """The agent takes request, datamodule and device as keywords."""
        from pruna.evaluation.evaluation_agent import EvaluationAgent

        assert {"request", "datamodule", "device"} <= self._bind(EvaluationAgent.__init__)
        assert {"model", "model_name"} <= self._bind(EvaluationAgent.evaluate)


@pytest.mark.cpu
@pytest.mark.requires_mcp
class TestMissingExtra:
    """Tests for a pruna installed without the mcp extra."""

    def test_the_package_imports_without_the_extra(self) -> None:
        """
        Importing the module must not require the mcp package.

        The console script imports it before main() runs, so an import-time raise would replace
        the guidance with a traceback that a client shows to the user verbatim.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import pruna.mcp_server.server as s; print(s._MCP_IMPORT_ERROR is None)",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "True" in result.stdout

    def test_building_without_the_extra_explains_how_to_install_it(self) -> None:
        """The failure has to name the extra, not just the missing module."""
        blocker = tempfile.mkdtemp()
        Path(blocker, "sitecustomize.py").write_text(
            "import sys\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'mcp' or name.startswith('mcp.'):\n"
            "            raise ImportError(\"No module named 'mcp'\")\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n",
            encoding="utf-8",
        )
        environment = {**os.environ, "PYTHONPATH": blocker}
        result = subprocess.run(
            [sys.executable, "-c", "import pruna.mcp_server.server as s; s.build_server()"],
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
        assert result.returncode != 0
        assert "pruna[mcp]" in result.stderr, result.stderr[-500:]

    def test_the_console_script_reports_the_missing_extra_cleanly(self) -> None:
        """A user wiring pruna-mcp into a CLI should get one line, not a traceback."""
        blocker = tempfile.mkdtemp()
        Path(blocker, "sitecustomize.py").write_text(
            "import sys\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'mcp' or name.startswith('mcp.'):\n"
            "            raise ImportError(\"No module named 'mcp'\")\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, "-m", "pruna.mcp_server.server"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": blocker},
            timeout=120,
        )
        assert result.returncode == 1
        assert "pruna[mcp]" in result.stderr
        assert "Traceback" not in result.stderr, result.stderr[-500:]
