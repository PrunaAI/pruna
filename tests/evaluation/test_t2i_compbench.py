# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import tempfile
from pathlib import Path

import pytest

from pruna.evaluation.metrics.metric_t2i_compbench import T2ICompBench
from pruna.evaluation.metrics.registry import MetricRegistry


@pytest.fixture
def mock_results_dir() -> Path:
    """Create a temporary directory with mock T2I-CompBench(++) results."""
    with tempfile.TemporaryDirectory() as tdir:
        root = Path(tdir)

        # Directory structure
        (root / "annotation_blip").mkdir(parents=True, exist_ok=True)
        (root / "annotation_clip").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "annotation_obj_detection_2d").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "annotation_obj_detection_3d").mkdir(parents=True, exist_ok=True)
        (root / "annotation_num").mkdir(parents=True, exist_ok=True)
        (root / "annotation_3_in_1").mkdir(parents=True, exist_ok=True)

        # Files (means: 0.85, 0.75, 0.65, 0.55, 0.45, 0.35) -> overall (equal weights) = 0.60
        payloads = [
            ("annotation_blip/vqa_result.json", [{"answer": 0.8}, {"answer": 0.9}]),                    # 0.85
            ("annotation_clip/vqa_result.json", [{"answer": 0.7}, {"answer": 0.8}]),                    # 0.75
            ("labels/annotation_obj_detection_2d/vqa_result.json", [{"answer": 0.6}, {"answer": 0.7}]), # 0.65
            ("labels/annotation_obj_detection_3d/vqa_result.json", [{"answer": 0.5}, {"answer": 0.6}]), # 0.55
            ("annotation_num/vqa_result.json", [{"answer": 0.4}, {"answer": 0.5}]),                     # 0.45
            ("annotation_3_in_1/vqa_result.json", [{"answer": 0.3}, {"answer": 0.4}]),                  # 0.35
        ]
        for rel, data in payloads:
            fp = root / rel
            with fp.open("w", encoding="utf-8") as f:
                json.dump(data, f)

        yield root


def test_init_basic(mock_results_dir: Path) -> None:
    metric = T2ICompBench(results_dir=mock_results_dir)
    assert metric.metric_name == "t2i_compbench"
    assert metric.higher_is_better is True
    assert metric.call_type == "y"
    assert metric.results_dir == mock_results_dir
    assert metric.weights is None


def test_invalid_ctor_args() -> None:
    # results_dir is required positional arg -> missing raises TypeError
    with pytest.raises(TypeError):
        T2ICompBench()  # type: ignore[misc]

    with pytest.raises(ValueError, match="Results directory does not exist"):
        T2ICompBench(results_dir="/not/a/real/path")

    with tempfile.NamedTemporaryFile() as tmp:
        with pytest.raises(ValueError, match="Results path is not a directory"):
            T2ICompBench(results_dir=tmp.name)


def test_weights_validation_and_filtering(mock_results_dir: Path) -> None:
    # valid weights are preserved
    weights = {
        "attribute_binding": 0.3,
        "non_spatial": 0.2,
        "spatial_2d": 0.2,
        "spatial_3d": 0.1,
        "numeracy": 0.1,
        "complex": 0.1,
    }
    metric = T2ICompBench(results_dir=mock_results_dir, weights=weights)
    assert metric.weights == weights

    # invalid keys are filtered out -> results in {}
    metric2 = T2ICompBench(results_dir=mock_results_dir, weights={"nope": 1.0})
    assert metric2.weights == {}


def test_compute_equal_weights(mock_results_dir: Path) -> None:
    metric = T2ICompBench(results_dir=mock_results_dir)
    res = metric.compute()

    assert res.name == "t2i_compbench"
    assert isinstance(res.result, float)
    assert res.result == pytest.approx(0.60, abs=1e-6)

    details = res.params.get("details", {})
    assert set(details.get("available_categories", [])) == {
        "attribute_binding",
        "non_spatial",
        "spatial_2d",
        "spatial_3d",
        "numeracy",
        "complex",
    }
    assert details.get("missing", []) == []


def test_compute_with_custom_weights(mock_results_dir: Path) -> None:
    # Only two categories weighted -> overall = (0.85 + 0.75) / 2 = 0.80
    metric = T2ICompBench(
        results_dir=mock_results_dir,
        weights={"attribute_binding": 0.5, "non_spatial": 0.5},
    )
    res = metric.compute()
    assert isinstance(res.result, float)
    assert res.result == pytest.approx(0.80, abs=1e-6)

    w_used = res.params["details"].get("weights_used")
    if w_used is not None:
        positive = {k for k, v in w_used.items() if v > 0}
        assert positive == {"attribute_binding", "non_spatial"}


def test_missing_some_files(mock_results_dir: Path) -> None:
    # Remove one category; compute should succeed and mark missing
    (mock_results_dir / "annotation_3_in_1" / "vqa_result.json").unlink()
    metric = T2ICompBench(results_dir=mock_results_dir)
    res = metric.compute()

    assert isinstance(res.result, float)
    assert 0.0 <= res.result <= 1.0
    assert "complex" in res.params["details"].get("missing", [])


def test_malformed_json_is_handled(mock_results_dir: Path) -> None:
    # Corrupt numeracy file; it should be treated as missing, not crash
    bad = mock_results_dir / "annotation_num" / "vqa_result.json"
    bad.write_text("{not: valid json", encoding="utf-8")

    metric = T2ICompBench(results_dir=mock_results_dir)
    res = metric.compute()

    assert isinstance(res.result, float)
    assert "numeracy" in res.params["details"].get("missing", [])


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cuda", marks=pytest.mark.cuda),
        pytest.param("cpu", marks=pytest.mark.cpu),
    ],
)
def test_device_kwarg_is_accepted_but_ignored(mock_results_dir: Path, device: str) -> None:
    metric = T2ICompBench(results_dir=mock_results_dir, device=device)
    res = metric.compute()
    assert isinstance(res.result, float)
    assert 0.0 <= res.result <= 1.0
    assert not hasattr(metric, "device")


def test_ignores_other_kwargs(mock_results_dir: Path) -> None:
    metric = T2ICompBench(results_dir=mock_results_dir, extra_arg="foo")
    assert not hasattr(metric, "extra_arg")


def test_metric_registry() -> None:
    metric = MetricRegistry.get_metric("t2i_compbench", results_dir="/tmp")
    assert isinstance(metric, T2ICompBench)


def test_is_pairwise_flag() -> None:
    metric = T2ICompBench(results_dir="/tmp")
    assert not metric.is_pairwise()
    metric.call_type = "y_gt"
    assert not metric.is_pairwise()
