import os
import shutil
from pathlib import Path

import pytest

from ..common import convert_notebook_to_script, run_script_successfully

TUTORIAL_PATH = Path(os.path.dirname(__file__)).parent.parent / "docs"


@pytest.mark.parametrize(
    "notebook_name",
    [
        pytest.param("asr_tutorial", marks=(pytest.mark.cuda, pytest.mark.high)),
        pytest.param("flux_small", marks=(pytest.mark.cuda, pytest.mark.high)),
        pytest.param("asr_whisper", marks=(pytest.mark.cuda, pytest.mark.high)),
        pytest.param("llms", marks=(pytest.mark.cuda, pytest.mark.high)),
        pytest.param("cv_cpu", marks=(pytest.mark.cpu, pytest.mark.high)),
        pytest.param("sana_diffusers_int8", marks=(pytest.mark.cuda, pytest.mark.high)),
        pytest.param("sd_deepcache", marks=(pytest.mark.cuda, pytest.mark.high)),
        pytest.param("evaluation_agent_cmmd", marks=(pytest.mark.cuda, pytest.mark.high)),
        pytest.param("portable_compilation", marks=(pytest.mark.cuda, pytest.mark.high)),  
    ],
)
def test_notebook_execution(notebook_name: str) -> None:
    """Test to ensure the notebook runs without errors."""
    NOTEBOOK_FILE = str(TUTORIAL_PATH / "tutorials" / f"{notebook_name}.ipynb")
    COPY_NOTEBOOK_FILE = f"{notebook_name}.ipynb"
    EXPECTED_SCRIPT_FILE = f"{notebook_name}.py"
    shutil.copy(NOTEBOOK_FILE, COPY_NOTEBOOK_FILE)

    convert_notebook_to_script(COPY_NOTEBOOK_FILE, EXPECTED_SCRIPT_FILE)
    run_script_successfully(EXPECTED_SCRIPT_FILE)

    os.remove(COPY_NOTEBOOK_FILE)
