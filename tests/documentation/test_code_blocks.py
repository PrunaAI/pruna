import os
import shutil
from pathlib import Path

import pytest

from ..common import extract_python_code_blocks, run_script_successfully

TUTORIAL_PATH = Path(os.path.dirname(__file__)).parent.parent / "docs"


@pytest.mark.parametrize(
    "rst_name",
    [
        pytest.param(f"user_manual/{path.stem}", marks=pytest.mark.cuda)
        for path in TUTORIAL_PATH.glob("user_manual/*.rst")
        if path.stem not in ["dataset"]
    ]
    + [
        pytest.param(f"contributions/{path.stem}", marks=pytest.mark.cuda)
        for path in TUTORIAL_PATH.glob("contributions/*.rst")
    ]
)
def test_codeblocks_cuda(rst_name: str) -> None:
    """Test to ensure the notebook runs without errors."""
    rst_file_path = str(TUTORIAL_PATH / f"{rst_name}.rst")
    output_dir = "tmp/code_blocks"
    extract_python_code_blocks(rst_file_path, output_dir)

    for file in sorted(os.listdir(output_dir)):
        file_path = os.path.join(output_dir, file)
        run_script_successfully(file_path)

    shutil.rmtree(output_dir)
