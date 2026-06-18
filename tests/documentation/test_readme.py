import os
import re
from pathlib import Path

import pytest

from ..common import run_script_successfully

README_PATH = Path(__file__).parent.parent.parent / "README.md"
SCRIPT_PATH = Path("test_readme_snippets.py")


def extract_python_code_blocks(readme_path: Path) -> str:
    """Extracts all Python code blocks from a Markdown README file."""
    content = README_PATH.read_text(encoding="utf-8")

    # Match fenced code blocks marked as python (```python ... ```)
    code_blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
    return "\n\n".join(code_blocks)  # Concatenate all code snippets


# The README's Python blocks are separate blocks, but together they form one example
# and this test executes them as a single script. That example uses stable_fast, so
# the whole README snippet test requires the stable-fast extra. If we later add
# independent snippets, we should split extraction/marking per example.
@pytest.mark.cuda
@pytest.mark.requires_stable_fast
def test_readme_code_blocks() -> None:
    """Writes all extracted Python code blocks to a single file, lints it with flake8, and executes it."""
    code = extract_python_code_blocks(README_PATH)

    SCRIPT_PATH.write_text(code, encoding="utf-8")

    # Run flake8 linting
    run_script_successfully(SCRIPT_PATH)
