import importlib.util
import inspect
import os
import subprocess
from typing import Any, Callable

import numpydoc_validation
import pytest
from docutils.core import publish_doctree
from docutils.nodes import literal_block, section, title

from pruna import SmashConfig
from pruna.engine.utils import get_device, move_to_device, safe_memory_cleanup


def device_parametrized(cls: Any) -> Any:
    """Decorator that adds device parameterization to all test methods in the AlgorithmTesterBase."""
    return pytest.mark.parametrize(
        "device",
        [
            pytest.param("cuda", marks=pytest.mark.cuda),
            pytest.param("accelerate", marks=pytest.mark.cuda),
            pytest.param("cpu", marks=pytest.mark.cpu),
        ],
    )(cls)


def get_instances_from_module(module: Any) -> list[tuple[Any, str]]:
    """Get all tester instances from a module and expand their model parametrizations."""

    def process_fn(cls: Any, model: str) -> dict[str, Any]:
        return [cls(), model]

    return collect_tester_instances(module, process_fn, "models")


def get_negative_examples_from_module(module: Any) -> list[tuple[Any, str]]:
    """Get all negative examples from a module."""

    def process_fn(cls: Any, model: str) -> dict[str, Any]:
        return [cls.get_algorithm_group(), cls.get_algorithm_name(), model]

    return collect_tester_instances(module, process_fn, "reject_models")


def collect_tester_instances(
    module: Any, process_fn: Callable[[Any, str], list[Any]], model_attr: str
) -> list[tuple[Any, str]]:
    """Collect all model classes from a module and process them with a function."""
    parametrizations = []
    for _, cls in vars(module).items():
        if (
            inspect.isclass(cls)
            and module.__name__ in cls.__module__
            and "AlgorithmTesterBase" not in cls.__name__
        ):
            model_parametrizations = getattr(cls, model_attr)
            markers = getattr(cls, "pytestmark", [])
            if not isinstance(markers, list):
                markers = [markers]
            for model in model_parametrizations:
                parameters = process_fn(cls, model)
                idx = f"{cls.__name__}_{model}"
                parametrizations.append(
                    pytest.param(*parameters, marks=markers, id=idx)
                )
    return parametrizations


def run_full_integration(
    algorithm_tester: Any, device: str, model_fixture: tuple[Any, SmashConfig]
) -> None:
    """Run the full integration test."""
    try:
        model, smash_config = model_fixture[0], model_fixture[1]
        if device not in algorithm_tester.compatible_devices():
            pytest.skip(
                f"Algorithm {algorithm_tester.get_algorithm_name()} is not compatible with {device}"
            )
        algorithm_tester.prepare_smash_config(smash_config, device)
        move_to_device(model, device=smash_config["device"])
        assert device == get_device(model)
        smashed_model = algorithm_tester.execute_smash(model, smash_config)
        algorithm_tester.execute_save(smashed_model)
        safe_memory_cleanup()
        algorithm_tester.execute_load()
    finally:
        algorithm_tester.final_teardown(smash_config)


def check_docstrings_content(file: str) -> None:
    """
    Given an import statement, check if the docstrings are valid in imported module.

    Note: The numpy validation module only accepts import statements, not file paths.

    Parameters
    ----------
    file : str
        The import statement to check.
    """
    n_invalid, report = numpydoc_validation.validate_recursive(
        file, checks={"all", "ES01", "SA01", "EX01"}, exclude=set()
    )
    if n_invalid != 0:
        raise ValueError(report)


def get_all_imports(package: str) -> set[str]:
    """
    Get all modules in a package and return a set of import strings.

    Example:
        If package is "pruna.algorithms.compilation" and there is a file
        "pruna/algorithms/compilation/stable_fast.py", the returned set will
        contain "pruna.algorithms.compilation.stable_fast".

    Parameters
    ----------
    package : str
        The package name to get the modules from.

    Returns
    -------
    set[str]
        A set of import strings for all modules in the package.

    Raises
    ------
    ValueError
        If the package cannot be found.
    """
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise ValueError(f"Package '{package}' not found.")

    imports = set()

    # Loop over all directories associated with the package (in case of namespace packages)
    for location in spec.submodule_search_locations:
        for root, _, files in os.walk(location):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    # Get the relative path from the package location
                    relative_path = os.path.relpath(file_path, location)
                    # Remove the .py extension
                    module_path = relative_path[:-3]
                    # Replace OS-specific path separators with dots to form the import string
                    module_name = module_path.replace(os.sep, ".")

                    # ignore __init__.py at the root of the package
                    if module_name == "__init__":
                        continue

                    # Handle __init__.py: its module is the package (or subpackage) itself.
                    if file == "__init__.py":
                        module_name = module_name.rsplit(".", 1)[
                            0
                        ]  # Remove the trailing '__init__'

                    # Combine the base package name with the module path.
                    # If module_name is empty (i.e. __init__.py at the root), just use the package name.
                    full_import = f"{package}.{module_name}" if module_name else package
                    imports.add(full_import)
    return imports


def run_script_successfully(script_file: str) -> None:
    """Run the script and return the result."""
    result = subprocess.run(["python", script_file], capture_output=True, text=True)
    run_ruff_linting(script_file)
    os.remove(script_file)

    assert result.returncode == 0, f"Notebook failed with error:\n{result.stderr}"


def convert_notebook_to_script(notebook_file: str, expected_script_file: str) -> None:
    """Convert the notebook to a Python script."""
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "script",
            "--TemplateExporter.exclude_raw=True",
            notebook_file,
        ],
        check=True,
    )

    # Handle possible incorrect extension
    generated_script = notebook_file.replace(".ipynb", ".txt")
    if os.path.exists(generated_script):
        os.rename(generated_script, expected_script_file)
    elif not os.path.exists(expected_script_file):
        raise FileNotFoundError("Converted script not found!")

    # Read the script, filter out lines starting with '!'
    with open(expected_script_file, "r") as file:
        lines = file.readlines()

    with open(expected_script_file, "w") as file:
        for line in lines:
            if not line.lstrip().startswith(
                "get_ipython"
            ) and not line.lstrip().startswith("!"):
                file.write(line)


def run_ruff_linting(file_path: str) -> None:
    """Run ruff on the file."""
    # Error codes to check:
    # F401: Unused imports
    # F841: Unused variables (detected by pyflakes, part of flake8)
    error_codes = ["F401", "F841"]

    # Run ruff on the file with the --select flag to only check these specific errors
    result = subprocess.run(
        ["ruff", "check", file_path, f"--select={','.join(error_codes)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise AssertionError(
            f"Linting errors found:\n{result.stdout}\nRuff error output:\n{result.stderr}"
        )


def extract_python_code_blocks(rst_file_path: str, output_dir: str) -> None:
    """Extract code blocks from first-level sections of an rst file, skipping blocks with the `noextract` class."""
    # Read the content of the .rst file
    with open(rst_file_path, "r") as file:
        rst_content = file.read()

    # Parse the content into a document tree
    document = publish_doctree(rst_content)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    def extract_code_blocks_from_node(node: Any, section_name: str) -> None:
        section_code_file = os.path.join(output_dir, f"{section_name}_code.py")
        with open(section_code_file, "w") as code_file:
            for block in node.traverse(literal_block):
                # Skip code blocks marked with the 'noextract' class
                if "noextract" in block.attributes.get("classes", []):
                    continue
                if "python" in block.attributes.get("classes", []):
                    code_file.write(block.astext() + "\n")

    # Process only first-level sections (sections whose parent is the document)
    for sec in document.traverse(section):
        if sec.parent is not document:
            continue  # Skip subsections
        section_title_node = sec.next_node(title)
        if section_title_node:
            section_title = section_title_node.astext().replace(" ", "_").lower()
            extract_code_blocks_from_node(sec, section_title)

    print(f"Code blocks extracted and written to {output_dir}")
