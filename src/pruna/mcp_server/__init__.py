# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model Context Protocol server exposing pruna to an LLM.

The server itself lives in :mod:`pruna.mcp_server.server` and is imported lazily, so that
importing this package does not require the ``mcp`` dependency to be installed. The
metadata helpers in :mod:`pruna.mcp_server.introspection` and
:mod:`pruna.mcp_server.validation` are always importable.
"""

from typing import Any

# Both names are provided lazily by __getattr__, which ruff cannot follow.
__all__ = ["build_server", "main"]  # noqa: F822

_LAZY: dict[str, tuple[str, str]] = {
    "build_server": ("pruna.mcp_server.server", "build_server"),
    "main": ("pruna.mcp_server.server", "main"),
}


def __getattr__(name: str) -> Any:
    """
    Resolve a server entry point lazily.

    Parameters
    ----------
    name : str
        The attribute being accessed.

    Returns
    -------
    Any
        The requested server attribute.

    Raises
    ------
    AttributeError
        If the attribute is not a lazy entry point.
    """
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attribute = _LAZY[name]
    return getattr(import_module(module_name), attribute)
