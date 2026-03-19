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


from abc import ABC, abstractmethod
from typing import Any


class AsyncEvaluationMixin(ABC):
    """
    Mixin for metrics that submit to external evaluation services and retrieve results asynchronously.

    Subclasses implement create_request() to set up an evaluation
    (e.g., create a leaderboard) and retrieve_results() to retrieve
    outcomes (e.g., standings from human evaluators).
    """

    @abstractmethod
    def create_request(self, *args, **kwargs) -> Any:
        """
        Create/configure an evaluation request on the external service.

        Parameters
        ----------
        *args :
            Variable length argument list.
        **kwargs :
            Arbitrary keyword arguments.
        """

    @abstractmethod
    def retrieve_results(self, *args, **kwargs) -> Any:
        """
        Retrieve results from the external service.

        Parameters
        ----------
        *args :
            Variable length argument list.
        **kwargs :
            Arbitrary keyword arguments.
        """
