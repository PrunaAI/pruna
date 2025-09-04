from __future__ import annotations

import inspect
from typing import Any

from ConfigSpace import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from pruna.algorithms import PRUNA_ALGORITHMS
from pruna.algorithms.pruna_base import PrunaAlgorithmBase


def generate_algorithm_desc(obj: PrunaAlgorithmBase, name_suffix: str = "") -> str:
    """
    Generate a Markdown description of a Pruna algorithm from its instance.

    Parameters
    ----------
    obj: PrunaAlgorithmBase
        The instance of the Pruna algorithm to generate the description for.
    name_suffix: str, optional
        A suffix to add to the algorithm name in the header.

    Returns
    -------
    str
        A Markdown description of the Pruna algorithm.
    """
    header = get_header_and_description(obj, name_suffix)
    references_str = get_references(obj)
    required_inputs_str = get_required_inputs(obj)
    compatible_devices_str = get_compatible_devices(obj)
    compatible_algorithms_str = get_compatible_algorithms(obj)
    required_install_str = get_required_install(obj)

    static_info = "\n".join(
        [
            f"| **References**: {references_str}.",
            f"| **Can be applied on**: {compatible_devices_str}.",
            f"| **Required**: {required_inputs_str}.",
            f"| **Compatible with**: {compatible_algorithms_str}.",
            f"| **Required install**: {required_install_str}." if required_install_str else "",
        ]
    )

    table_rows, hyperparameter_counter = get_table_rows(obj)

    # Only add the table if there are hyperparameters to show.
    if hyperparameter_counter > 0:
        table = format_grid_table(table_rows)
        markdown = f"{header}\n\n{static_info}\n\n{table}"
    else:
        markdown = f"{header}\n\n{static_info}"
    return markdown


def get_algorithm_description(cls: Any) -> str:
    """Extract the description segment from the docstring of a class."""
    doc = inspect.getdoc(cls)
    assert doc is not None, "No docstring found for class"
    lines = doc.splitlines()
    assert len(lines) > 2, "No description found in docstring"
    segment_lines = []
    for line in lines[2:]:
        if not line.strip():
            break
        segment_lines.append(line.strip())
    return " ".join(segment_lines)


def format_grid_table(rows: list[list[str]]) -> str:
    """Given a list of rows (each row is a list of cell strings), returns a grid table."""
    num_cols = len(rows[0])
    col_widths = [max(len(row[i]) for row in rows) for i in range(num_cols)]
    total_widths = [w + 2 for w in col_widths]

    horizontal_border = "+" + "+".join("-" * width for width in total_widths) + "+"
    header_line = "|" + "|".join(" " + rows[0][i].ljust(col_widths[i]) + " " for i in range(num_cols)) + "|"
    header_separator = "+" + "+".join("=" * width for width in total_widths) + "+"

    data_lines = []
    for row in rows[1:]:
        row_line = "|" + "|".join(" " + row[i].ljust(col_widths[i]) + " " for i in range(num_cols)) + "|"
        data_lines.append(row_line)
        data_lines.append(horizontal_border)

    table_lines = [horizontal_border, header_line, header_separator] + data_lines
    return "\n".join(table_lines)


def get_header_and_description(obj: PrunaAlgorithmBase, name_suffix: str = "") -> str:
    """Get the header and description of a Pruna algorithm."""
    algo = obj.algorithm_name
    description = get_algorithm_description(obj)
    header = f"``{algo}`` {name_suffix}\n{'~' * (len(algo) + len(name_suffix) + 5)}\n"
    header += f"{description}\n"
    return header


def get_references(obj: PrunaAlgorithmBase) -> str:
    """Get the references of a Pruna algorithm."""
    references = obj.references
    if references:
        return ", ".join(f"`{key} <{value}>`__" for key, value in references.items())
    else:
        return "None"


def get_required_inputs(obj: PrunaAlgorithmBase) -> str:
    """Get the required inputs of a Pruna algorithm."""
    required_inputs = []
    if obj.tokenizer_required:
        required_inputs.append("Tokenizer")
    if obj.processor_required:
        required_inputs.append("Processor")
    if obj.dataset_required:
        required_inputs.append("Dataset")
    return ", ".join(required_inputs) if required_inputs else "None"


def get_compatible_devices(obj: PrunaAlgorithmBase) -> str:
    """Get the compatible devices of a Pruna algorithm."""
    compatible_devices = []
    for device in obj.runs_on:
        name_map = {
            "cpu": "CPU",
            "cuda": "CUDA",
            "mps": "MPS",
            "accelerate": "Accelerate distributed",
        }
        compatible_devices.append(name_map[device])
    return ", ".join(compatible_devices) if compatible_devices else "None"


def get_compatible_algorithms(obj: PrunaAlgorithmBase) -> str:
    """Get the compatible algorithms of a Pruna algorithm."""
    compatible_algorithms = []
    for algorithms in obj.compatible_algorithms.values():
        compatible_algorithms.extend([f"``{a}``" for a in algorithms])
    return ", ".join(compatible_algorithms) if compatible_algorithms else "None"


def get_required_install(obj: PrunaAlgorithmBase) -> str | None:
    """Get the required install of a Pruna algorithm."""
    required_install = obj.required_install
    base_install = "``pip install pruna[full]``"

    # Break the long command into multiple code blocks with line continuation for better RST rendering
    pro_install = (
        "``pip install pruna_pro[full]`` "
        "``--extra-index-url https://prunaai.pythonanywhere.com/`` "
        "``--extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/cn/``"
    )

    if required_install:
        if obj.algorithm_name == "gptq":
            return required_install
        if "pro" in required_install:
            required_install += f" or {pro_install}"
        else:
            required_install += f" or {base_install}"
        return required_install
    else:
        return None


def get_table_rows(obj: PrunaAlgorithmBase) -> tuple[list[list[str]], int]:
    """Get the table rows of a Pruna algorithm hyperparameter section."""
    rows = []
    # Table header row
    rows.append(["**Parameter**", "**Default**", "**Options**", "**Description**"])
    hyperparameter_counter = 0

    for hp in obj.get_hyperparameters():
        if isinstance(hp, Constant):
            continue  # Skip constant hyperparameters

        param_name = f"``{obj.algorithm_name}_{hp.name}``"
        assert hp.meta is not None and "desc" in hp.meta
        description = hp.meta["desc"]

        if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            default = str(hp.default_value)
            values = f"Range {hp.lower} to {hp.upper}"
        elif isinstance(hp, OrdinalHyperparameter):
            choices = ", ".join(str(v) for v in hp.sequence[:-1])
            values = f"{choices} or {hp.sequence[-1]}"
            default = str(hp.default_value)
        elif isinstance(hp, CategoricalHyperparameter):
            values = ", ".join(str(v) for v in hp.choices)
            default = str(hp.default_value)
        else:
            raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")

        rows.append([param_name, default, values, description])
        hyperparameter_counter += 1
    return rows, hyperparameter_counter


if __name__ == "__main__":
    # Collect all algorithms into a single file.
    with open("compression.rst", "w") as f:
        for algorithm_group in PRUNA_ALGORITHMS.values():
            for algorithm in algorithm_group.values():
                f.write(generate_algorithm_desc(algorithm))
                f.write("\n\n")
