import pytest

from pruna.algorithms.kvpress import PRESS_TYPES


@pytest.fixture()
def kvpress_modules():
    """Import kvpress and return all press classes."""
    import kvpress

    return {name: getattr(kvpress, name) for name in PRESS_TYPES}


def test_press_types_exist(kvpress_modules):
    """Verify all PRESS_TYPES are valid classes in the kvpress module."""
    for name, cls in kvpress_modules.items():
        assert isinstance(cls, type), f"{name} is not a class"


def test_press_types_accept_compression_ratio(kvpress_modules):
    """Verify all PRESS_TYPES can be constructed with compression_ratio."""
    for name, cls in kvpress_modules.items():
        press = cls(compression_ratio=0.5)
        assert press.compression_ratio == 0.5, f"{name} did not set compression_ratio"


def test_press_kwargs_forwarded(kvpress_modules):
    """Verify press_kwargs are forwarded to the press constructor."""
    snap = kvpress_modules["SnapKVPress"](compression_ratio=0.4, window_size=32, kernel_size=3)
    assert snap.window_size == 32
    assert snap.kernel_size == 3
