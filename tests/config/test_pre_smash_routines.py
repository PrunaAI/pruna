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

import itertools
from unittest.mock import Mock, patch, MagicMock
from typing import Any

import pytest
import torch
import networkx as nx

from pruna import SmashConfig
from pruna.config.pre_smash_routines import (
    ensure_device_consistency,
    check_model_compatibility,
    check_algorithm_packages_availability,
    check_argument_compatibility,
    check_algorithm_availability,
    execute_algorithm_pre_smash_hooks,
    check_algorithm_cross_compatibility,
    determine_algorithm_order,
    construct_algorithm_directed_graph,
)


class TestEnsureDeviceConsistency:
    """Test suite for ensure_device_consistency function."""

    def test_device_consistency_matching_devices(self):
        """Test device consistency when model and smash_config devices match. Device consistency check should pass without errors."""
        model = Mock()
        smash_config = Mock()
        smash_config.device = "cuda"
        
        with patch("pruna.config.pre_smash_routines.get_device", return_value="cuda"):
            with patch("pruna.config.pre_smash_routines.pruna_logger") as mock_logger:
                ensure_device_consistency(model, smash_config)
                mock_logger.debug.assert_called_with("Device consistency check passed.")

    def test_device_consistency_accelerate_matching(self):
        """
        Test device consistency when both model and config use accelerate. Should handle accelerate device mapping correctly.
        """
        model = Mock()
        smash_config = Mock()
        smash_config.device = "accelerate"
        
        mock_device_map = {"layer1": 0, "layer2": 1}
        
        with patch("pruna.config.pre_smash_routines.get_device", return_value="accelerate"):
            with patch("pruna.config.pre_smash_routines.get_device_map", return_value=mock_device_map):
                with patch("pruna.config.pre_smash_routines.pruna_logger") as mock_logger:
                    ensure_device_consistency(model, smash_config)
                    assert smash_config.device_map == mock_device_map
                    mock_logger.debug.assert_called_with("Device consistency check passed.")

    def test_device_consistency_accelerate_cpu_offloading_error(self):
        """
        Test device consistency error when accelerate uses CPU offloading. Should raise ValueError for CPU offloading.
        """
        model = Mock()
        smash_config = Mock()
        smash_config.device = "accelerate"
        
        mock_device_map = {"layer1": 0, "layer2": "cpu"}
        
        with patch("pruna.config.pre_smash_routines.get_device", return_value="accelerate"):
            with patch("pruna.config.pre_smash_routines.get_device_map", return_value=mock_device_map):
                with pytest.raises(ValueError, match="Device map indicates CPU offloading"):
                    ensure_device_consistency(model, smash_config)

    def test_device_consistency_different_compatible_devices(self):
        """Test device consistency when devices differ but are compatible. Should move model to smash_config device."""
        model = Mock()
        smash_config = Mock()
        smash_config.device = "cuda:1"
        
        with patch("pruna.config.pre_smash_routines.get_device", return_value="cuda:0"):
            with patch("pruna.config.pre_smash_routines.move_to_device") as mock_move:
                with patch("pruna.config.pre_smash_routines.pruna_logger") as mock_logger:
                    ensure_device_consistency(model, smash_config)
                    mock_move.assert_called_once_with(model, "cuda:1")
                    mock_logger.warning.assert_called_once()

    def test_device_consistency_accelerate_mismatch(self):
        """Test device consistency when one uses accelerate and other doesn't. Should update smash_config device to match model."""
        model = Mock()
        smash_config = Mock()
        smash_config.device = "accelerate"
        
        with patch("pruna.config.pre_smash_routines.get_device", return_value="cuda"):
            with patch("pruna.config.pre_smash_routines.pruna_logger") as mock_logger:
                ensure_device_consistency(model, smash_config)
                assert smash_config.device == "cuda"
                mock_logger.warning.assert_called_once()


class TestCheckModelCompatibility:
    """Test suite for check_model_compatibility function."""

    def test_model_compatibility_success(self):
        """Test successful model compatibility check. Should pass without errors when model is compatible."""
        model = Mock()
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1"]
        
        mock_algorithm1 = Mock()
        mock_algorithm1.algorithm_name = "algorithm1"
        mock_algorithm1.model_check_fn.return_value = True
        mock_algorithm1.runs_on = ["cuda", "cpu"]
        
        mock_algorithms = {"algorithm1": mock_algorithm1}
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            with patch("pruna.config.pre_smash_routines.get_device", return_value="cuda"):
                check_model_compatibility(model, smash_config)

    def test_model_compatibility_model_check_failure(self):
        """Test model compatibility failure when model check fails. Should raise ValueError when model is not compatible."""
        model = Mock()
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1"]
        
        mock_algorithm1 = Mock()
        mock_algorithm1.algorithm_name = "algorithm1"
        mock_algorithm1.model_check_fn.return_value = False
        
        mock_algorithms = {"algorithm1": mock_algorithm1}
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            with pytest.raises(ValueError, match="Model is not compatible with algorithm1"):
                check_model_compatibility(model, smash_config)

    def test_model_compatibility_device_incompatibility(self):
        """Test model compatibility failure when device is incompatible. Should raise ValueError when device is not supported."""
        model = Mock()
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1"]
        
        mock_algorithm1 = Mock()
        mock_algorithm1.algorithm_name = "algorithm1"
        mock_algorithm1.model_check_fn.return_value = True
        mock_algorithm1.runs_on = ["cpu"]
        
        mock_algorithms = {"algorithm1": mock_algorithm1}
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            with patch("pruna.config.pre_smash_routines.get_device", return_value="cuda"):
                with pytest.raises(ValueError, match="algorithm1 is not compatible with model device cuda"):
                    check_model_compatibility(model, smash_config)


class TestCheckAlgorithmPackagesAvailability:
    """Test suite for check_algorithm_packages_availability function."""

    def test_algorithm_packages_availability_success(self):
        """Test successful algorithm packages availability check. Should call import_algorithm_packages for each active algorithm."""
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1"]
        
        mock_algorithm1 = Mock()
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", {"algorithm1": mock_algorithm1}):
            check_algorithm_packages_availability(smash_config)
            mock_algorithm1.import_algorithm_packages.assert_called_once()

    def test_algorithm_packages_availability_import_error(self):
        """Test algorithm packages availability when import fails. Should wrap import error correctly."""
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1"]
        
        mock_algorithm1 = Mock()
        mock_algorithm1.import_algorithm_packages.side_effect = ImportError("Something went wrong")
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", {"algorithm1": mock_algorithm1}):
            with pytest.raises(ImportError, match="Could not import necessary packages for algorithm1"):
                check_algorithm_packages_availability(smash_config)


class TestCheckArgumentCompatibility:
    """Test suite for check_argument_compatibility function."""

    def test_argument_compatibility_missing_tokenizer(self):
        """Test argument compatibility failure when tokenizer is missing."""
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1"]
        smash_config.tokenizer = None
        smash_config._target_module = None
        
        mock_algorithm1 = Mock()
        mock_algorithm1.tokenizer_required = True
        mock_algorithm1.processor_required = False
        mock_algorithm1.dataset_required = False
        
        mock_algorithms = {"algorithm1": mock_algorithm1}
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            with pytest.raises(ValueError, match="algorithm1 requires a tokenizer"):
                check_argument_compatibility(smash_config)

    def test_argument_compatibility_missing_processor(self):
        """Test argument compatibility failure when processor is missing."""
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1"]
        smash_config.tokenizer = Mock()
        smash_config.processor = None
        smash_config._target_module = None
        
        mock_algorithm1 = Mock()
        mock_algorithm1.tokenizer_required = False
        mock_algorithm1.processor_required = True
        mock_algorithm1.dataset_required = False
        
        mock_algorithms = {"algorithm1": mock_algorithm1}
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            with pytest.raises(ValueError, match="algorithm1 requires a processor"):
                check_argument_compatibility(smash_config)

    def test_argument_compatibility_missing_dataset(self):
        """Test argument compatibility failure when dataset is missing."""
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1"]
        smash_config.tokenizer = Mock()
        smash_config.processor = Mock()
        smash_config.data = None
        smash_config._target_module = None
        
        mock_algorithm1 = Mock()
        mock_algorithm1.tokenizer_required = False
        mock_algorithm1.processor_required = False
        mock_algorithm1.dataset_required = True
        
        mock_algorithms = {"algorithm1": mock_algorithm1}
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            with pytest.raises(ValueError, match="algorithm1 requires a dataset"):
                check_argument_compatibility(smash_config)


class TestCheckAlgorithmAvailability:
    """Test suite for check_algorithm_availability function."""

    def test_algorithm_availability_pruna_pro_error(self):
        """Test algorithm availability failure for pruna_pro algorithms."""
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1"]
        
        mock_algorithm1 = Mock()
        mock_algorithm1.__module__ = "pruna_pro.algorithms.algorithm1"
        
        mock_algorithms = {"algorithm1": mock_algorithm1}
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            with pytest.raises(RuntimeError, match="Algorithm algorithm1 is unavailable with pruna.smash"):
                check_algorithm_availability(smash_config)


class TestCheckAlgorithmCrossCompatibility:
    """Test suite for check_algorithm_cross_compatibility function."""

    def test_cross_compatibility_success(self):
        """Test successful algorithm cross-compatibility check."""
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1", "algorithm2"]
        
        mock_algorithm1 = Mock()
        mock_algorithm1.get_incompatible_algorithms.return_value = []
        
        mock_algorithm2 = Mock()
        mock_algorithm2.get_incompatible_algorithms.return_value = []
        
        mock_algorithms = {"algorithm1": mock_algorithm1, "algorithm2": mock_algorithm2}
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            check_algorithm_cross_compatibility(smash_config)

    def test_cross_compatibility_incompatible_algorithms(self):
        """Test algorithm cross-compatibility failure for incompatible algorithms. Should raise ValueError when algorithms are incompatible."""
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1", "algorithm2"]
        
        mock_algorithm1 = Mock()
        mock_algorithm1.get_incompatible_algorithms.return_value = []
        
        mock_algorithm2 = Mock()
        mock_algorithm2.get_incompatible_algorithms.return_value = ["algorithm1"]
        
        mock_algorithms = {"algorithm1": mock_algorithm1, "algorithm2": mock_algorithm2}
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            with pytest.raises(ValueError, match="Algorithm algorithm1 is incompatible with algorithm2"):
                check_algorithm_cross_compatibility(smash_config)


class TestDetermineAlgorithmOrder:
    """Test suite for determine_algorithm_order function."""

    def test_determine_algorithm_order_success(self):
        """Test successful algorithm order determination. Should return topologically sorted algorithm order."""
        smash_config = Mock()
        
        mock_graph = nx.DiGraph()
        mock_graph.add_nodes_from(["algorithm1", "algorithm2"])
        mock_graph.add_edge("algorithm1", "algorithm2")
        
        with patch("pruna.config.pre_smash_routines.construct_algorithm_directed_graph", return_value=mock_graph):
            order = determine_algorithm_order(smash_config)
            assert order == ["algorithm1", "algorithm2"]

    def test_determine_algorithm_order_cyclic_dependency(self):
        """Test algorithm order determination with cyclic dependencies. Should raise NetworkXUnfeasible for cyclic dependencies."""
        smash_config = Mock()
        
        mock_graph = nx.DiGraph()
        mock_graph.add_nodes_from(["algorithm1", "algorithm2"])
        mock_graph.add_edge("algorithm1", "algorithm2")
        mock_graph.add_edge("algorithm2", "algorithm1")
        
        with patch("pruna.config.pre_smash_routines.construct_algorithm_directed_graph", return_value=mock_graph):
            with pytest.raises(nx.NetworkXUnfeasible):
                determine_algorithm_order(smash_config)


class TestConstructAlgorithmDirectedGraph:
    """Test suite for construct_algorithm_directed_graph function."""

    def test_construct_graph_no_dependencies(self):
        """Test graph construction with no algorithm dependencies. Should create graph with nodes but no edges."""
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1", "algorithm2"]
        
        mock_algorithm1 = Mock()
        mock_algorithm1.get_required_before.return_value = []
        mock_algorithm1.get_required_after.return_value = []
        
        mock_algorithm2 = Mock()
        mock_algorithm2.get_required_before.return_value = []
        mock_algorithm2.get_required_after.return_value = []
        
        mock_algorithms = {"algorithm1": mock_algorithm1, "algorithm2": mock_algorithm2}
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            graph = construct_algorithm_directed_graph(smash_config)
            
            assert set(graph.nodes()) == {"algorithm1", "algorithm2"}
            assert len(graph.edges()) == 0

    def test_construct_graph_with_required_before(self):
        """Test graph construction with required_before dependencies. Should create graph with edges for required_before dependencies."""
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1", "algorithm2"]
        
        mock_algorithm1 = Mock()
        mock_algorithm1.get_required_before.return_value = []
        mock_algorithm1.get_required_after.return_value = []
        
        mock_algorithm2 = Mock()
        mock_algorithm2.get_required_before.return_value = ["algorithm1"]
        mock_algorithm2.get_required_after.return_value = []
        
        mock_algorithms = {"algorithm1": mock_algorithm1, "algorithm2": mock_algorithm2}
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            graph = construct_algorithm_directed_graph(smash_config)
            
            assert set(graph.nodes()) == {"algorithm1", "algorithm2"}
            assert ("algorithm1", "algorithm2") in graph.edges()

    def test_construct_graph_with_required_after(self):
        """Test graph construction with required_after dependencies. Should create graph with edges for required_after dependencies."""
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1", "algorithm2"]
        
        mock_algorithm1 = Mock()
        mock_algorithm1.get_required_before.return_value = []
        mock_algorithm1.get_required_after.return_value = []
        
        mock_algorithm2 = Mock()
        mock_algorithm2.get_required_before.return_value = []
        mock_algorithm2.get_required_after.return_value = ["algorithm1"]
        
        mock_algorithms = {"algorithm1": mock_algorithm1, "algorithm2": mock_algorithm2}
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            graph = construct_algorithm_directed_graph(smash_config)
            
            assert set(graph.nodes()) == {"algorithm1", "algorithm2"}
            assert ("algorithm2", "algorithm1") in graph.edges()

    def test_construct_graph_complex_dependencies(self):
        """Test graph construction with complex algorithm dependencies. Should create graph with multiple dependency relationships."""
        smash_config = Mock()
        smash_config.get_active_algorithms.return_value = ["algorithm1", "algorithm2", "algorithm3"]
        
        mock_algorithm1 = Mock()
        mock_algorithm1.get_required_before.return_value = []
        mock_algorithm1.get_required_after.return_value = []
        
        mock_algorithm2 = Mock()
        mock_algorithm2.get_required_before.return_value = ["algorithm1"]
        mock_algorithm2.get_required_after.return_value = ["algorithm3"]
        
        mock_algorithm3 = Mock()
        mock_algorithm3.get_required_before.return_value = []
        mock_algorithm3.get_required_after.return_value = []
        
        mock_algorithms = {
            "algorithm1": mock_algorithm1,
            "algorithm2": mock_algorithm2,
            "algorithm3": mock_algorithm3
        }
        
        with patch("pruna.config.pre_smash_routines.PRUNA_ALGORITHMS", mock_algorithms):
            graph = construct_algorithm_directed_graph(smash_config)
            
            assert set(graph.nodes()) == {"algorithm1", "algorithm2", "algorithm3"}
            assert ("algorithm1", "algorithm2") in graph.edges()
            assert ("algorithm2", "algorithm3") in graph.edges()
