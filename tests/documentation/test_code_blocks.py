import os
import shutil
from pathlib import Path

import pytest

from ..common import extract_python_code_blocks, run_script_successfully

TUTORIAL_PATH = Path(__file__).parent.parent.parent / "docs"

@pytest.mark.parametrize(
    "rst_path",
    (TUTORIAL_PATH / "user_manual").glob("*.rst"),
    ids=lambda p: p.stem,
)
def test_codeblocks_cuda(rst_path, tmp_path):
    out_dir = tmp_path / "blocks"          # unique per test instance
    extract_python_code_blocks(rst_path, out_dir)

    for script in sorted(out_dir.iterdir()):
        run_script_successfully(script)

    shutil.rmtree(out_dir)
