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

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any, Dict

import pruna.algorithms as algorithms
from pruna.config.smash_space import ALGORITHM_GROUPS

# PRUNA_METHODS holds instances of all available methods for each algorithm group
PRUNA_ALGORITHMS: Dict[str, Dict[str, Any]] = dict()

for algorithm_group in ALGORITHM_GROUPS:
    PRUNA_ALGORITHMS[algorithm_group] = dict()

# iterate through all coarse-grained algorithm types
for finder, algorithm_types, ispkg in pkgutil.iter_modules(algorithms.__path__):
    if ispkg:
        # iterate through all algorithms within an algorithm type
        for _finder, sub_name, _ispkg in pkgutil.iter_modules([str(Path(algorithms.__path__[0]) / algorithm_types)]):
            module = importlib.import_module(f"{algorithms.__name__}.{algorithm_types}.{sub_name}")
            # discover the algorithm class in the file
            for obj_name, obj in inspect.getmembers(module, inspect.isclass):
                # Check whether the algorithm is a grandchild of PrunaAlgo in order to skip the algorithm base classes
                class_names = [cls.__name__ for cls in obj.__mro__]
                # usually algorithms are grandchildren of PrunaAlgo, except for cases like ctranslate/cgenerate/cwhisper
                if "PrunaAlgorithmBase" in class_names and class_names.index("PrunaAlgorithmBase") != 1:
                    # instantiation of the algorithm makes it ready to call and registers HPs
                    PRUNA_ALGORITHMS[obj.algorithm_group][obj.algorithm_name] = obj()
