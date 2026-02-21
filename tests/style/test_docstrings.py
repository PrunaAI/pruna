import pytest

from ..common import check_docstrings_content, get_all_imports


@pytest.mark.style
@pytest.mark.parametrize("file", get_all_imports("pruna"))
def test_docstrings(file: str) -> None:
    """
    Test all docstrings in the pruna directory.

    Parameters
    ----------
    file : str
        The import statement to check.
    """
    # Skip metrics_vlm module as it uses a different docstring pattern for VLM parameters
    if "metrics_vlm" in file:
        pytest.skip("metrics_vlm uses custom VLM parameter documentation")
    check_docstrings_content(file)
