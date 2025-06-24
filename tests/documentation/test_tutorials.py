import os
import shutil
from pathlib import Path
import glob
import pytest

from ..common import convert_notebook_to_script, run_script_successfully

TUTORIAL_PATH = Path(os.path.dirname(__file__)).parent.parent / "docs"


@pytest.mark.parametrize(
    "notebook_name",
    [
        pytest.param(notebook_name, marks=(pytest.mark.cuda, pytest.mark.high))
        for notebook_name in glob.glob(str(TUTORIAL_PATH / "tutorials" / "*.ipynb"))
    ],
)
def test_notebook_execution(notebook_name: str) -> None:
    """Test to ensure the notebook runs without errors."""
    NOTEBOOK_FILE = str(TUTORIAL_PATH / "tutorials" / notebook_name)
    COPY_NOTEBOOK_FILE = f"{notebook_name}.ipynb"
    EXPECTED_SCRIPT_FILE = f"{notebook_name}.py"
    shutil.copy(NOTEBOOK_FILE, COPY_NOTEBOOK_FILE)

    convert_notebook_to_script(COPY_NOTEBOOK_FILE, EXPECTED_SCRIPT_FILE)
    run_script_successfully(EXPECTED_SCRIPT_FILE)

    os.remove(COPY_NOTEBOOK_FILE)
