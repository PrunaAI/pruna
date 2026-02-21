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
    check_docstrings_content(file)
