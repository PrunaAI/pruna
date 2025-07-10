import os
import shutil
import asyncio
import torch
from pathlib import Path

import pytest

from ..common import extract_python_code_blocks, run_script_successfully

TUTORIAL_PATH = Path(os.path.dirname(__file__)).parent.parent / "docs"

def pytest_generate_tests(metafunc):
    if "rst_path" in metafunc.fixturenames and "script_content" in metafunc.fixturenames:
        rst_files = list((TUTORIAL_PATH / "user_manual").glob("*.rst"))
        all_script_params = []
        all_ids = []
        for rst_file in rst_files:
            # Use a temporary directory to extract code blocks
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                extract_python_code_blocks(rst_file, tmpdir_path)
                scripts = sorted(tmpdir_path.iterdir())
                for idx, script in enumerate(scripts):
                    script_content = script.read_text()
                    # Apply device compatibility modifications
                    all_script_params.append((rst_file, script_content))
                    all_ids.append(f"{rst_file.stem}_block{idx+1}")
        metafunc.parametrize(
            ("rst_path", "script_content"),
            all_script_params,
            ids=all_ids
        )

async def run_script_async(script_path):
    # Wrap the synchronous function in a thread to allow async execution
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run_script_successfully, script_path)


@pytest.mark.usefixtures("tmp_path")
@pytest.mark.asyncio
async def test_codeblock_cuda(rst_path, script_content, tmp_path):
    # Each test gets a unique script file
    script_file = tmp_path / "script.py"
    script_file.write_text(script_content)

    await run_script_async(script_file)

    # Clean up
    if script_file.exists():
        script_file.unlink()
