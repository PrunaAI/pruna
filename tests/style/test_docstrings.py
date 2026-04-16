import pytest

from ..common import check_docstrings_content, get_all_imports

_VENDOR_ONEIG_LLM2VEC_PREFIX = "pruna.evaluation.metrics.vendor.oneig_llm2vec"


def _pruna_modules_for_docstring_check() -> list[str]:
    """All ``pruna`` modules except vendored OneIG LLM2Vec (upstream docstrings)."""
    return [
        m
        for m in get_all_imports("pruna")
        if not m.startswith(_VENDOR_ONEIG_LLM2VEC_PREFIX)
    ]


@pytest.mark.style
@pytest.mark.parametrize("file", _pruna_modules_for_docstring_check())
def test_docstrings(file: str) -> None:
    """
    Test all docstrings in the pruna directory.

    Parameters
    ----------
    file : str
        The import statement to check.
    """
    check_docstrings_content(file)
