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

import os
import sys
import warnings
from typing import Any


def apply_warning_filter() -> None:
    """Apply the warning filter globally."""
    warnings.filterwarnings("ignore")


def remove_warning_filter() -> None:
    """Remove the warning filter globally."""
    warnings.filterwarnings("default")


def is_notebook() -> bool:
    """
    Check if the code is running in a Jupyter notebook.

    Returns
    -------
    bool
        True if the code is running in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        # Jupyter notebook or qtconsole result in True, terminal result in False
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False  # Probably standard Python interpreter


class SuppressOutput:
    """Context manager to suppress output in console or Jupyter notebook."""

    def __enter__(self) -> "SuppressOutput":
        """Enter the context manager."""
        self._is_notebook = is_notebook()
        if self._is_notebook:
            # In Jupyter, use io.capture_output()
            from IPython.utils import io

            self._suppressor = io.capture_output()
            self._suppressor.__enter__()
        else:
            # In standard Python, redirect sys.stdout to os.devnull
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the context manager."""
        if self._is_notebook:
            # Exit the io.capture_output() context
            self._suppressor.__exit__(exc_type, exc_value, traceback)
        else:
            # Close the devnull file and restore sys.stdout
            sys.stdout.close()
            sys.stdout = self._original_stdout
