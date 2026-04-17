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


from abc import ABC


class EvaluationContextMixin(ABC):
    """
    Mixin for metrics that evaluate multiple models sequentially.

    Provides a current_context property that tracks which model is being
    evaluated. Setting a new context triggers on_context_change(), which
    subclasses can override to reset state between models.
    """

    _current_context: str | None = None

    @property
    def current_context(self) -> str | None:
        """
        Return the current context.

        Returns
        -------
        str | None
            The current context.
        """
        return self._current_context

    @current_context.setter
    def current_context(self, value: str | None) -> None:
        """
        Set the current context.

        Parameters
        ----------
        value : str
            The new context.
        """
        if self._current_context != value:
            self._current_context = value
            self.on_context_change()

    def on_context_change(self) -> None:
        """Hook called when the context changes. Override to reset state."""
        pass

    def _require_context(self) -> None:
        """Raise if no context has been set."""
        if self._current_context is None:
            raise ValueError("No context set. Set current_context first.")
