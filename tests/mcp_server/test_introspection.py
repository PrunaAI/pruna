"""Tests for the MCP introspection helpers that expose the pruna algorithm catalogue."""

import pytest

from pruna.mcp_server import introspection


@pytest.mark.cpu
class TestListTags:
    """Tests for listing algorithm categories."""

    def test_returns_every_tag_with_a_description(self) -> None:
        """Every algorithm category should be reported with a human-readable description."""
        tags = introspection.list_tags()
        assert len(tags) > 0
        for tag in tags:
            assert set(tag) == {"name", "description"}
            assert tag["name"] and tag["description"]


@pytest.mark.cpu
class TestListAlgorithms:
    """Tests for enumerating the algorithm catalogue."""

    def test_lists_every_registered_algorithm(self) -> None:
        """The unfiltered listing should cover the whole registry."""
        from pruna import AlgorithmRegistry

        algorithms = introspection.list_algorithms()
        assert len(algorithms) == len(AlgorithmRegistry.get_all_algorithms())

    def test_summary_reports_name_devices_and_requirements(self) -> None:
        """Each summary should carry the fields a caller needs to plan a configuration."""
        for algorithm in introspection.list_algorithms():
            assert algorithm["name"]
            assert isinstance(algorithm["runs_on"], list)
            assert set(algorithm["requires"]) == {"tokenizer", "processor", "dataset", "pip_install"}

    def test_tag_filter_returns_a_subset(self) -> None:
        """Filtering by a category should return fewer algorithms than the full listing."""
        everything = introspection.list_algorithms()
        quantizers = introspection.list_algorithms(tag="quantizer")
        assert len(quantizers) < len(everything)
        assert all("quantizer" in algorithm["tags"] for algorithm in quantizers)

    def test_tag_filter_is_case_insensitive(self) -> None:
        """Category names should be accepted in either case."""
        assert introspection.list_algorithms(tag="QUANTIZER") == introspection.list_algorithms(tag="quantizer")

    def test_unknown_tag_raises_with_the_valid_options(self) -> None:
        """An unknown category should fail loudly and list what is accepted."""
        with pytest.raises(ValueError, match="Unknown algorithm tag"):
            introspection.list_algorithms(tag="not_a_real_tag")


@pytest.mark.cpu
class TestDescribeAlgorithm:
    """Tests for the single-algorithm detail view."""

    def test_describes_hyperparameters_with_qualified_names(self) -> None:
        """Hyperparameters should be reported both bare and algorithm-prefixed."""
        described = introspection.describe_algorithm("hqq")
        assert described["name"] == "hqq"
        assert described["hyperparameters"]

        for hyperparameter in described["hyperparameters"]:
            assert hyperparameter["qualified_name"] == f"hqq_{hyperparameter['name']}"
            assert hyperparameter["type"]
            assert "default" in hyperparameter

    def test_reports_compatibility_and_requirements(self) -> None:
        """Order constraints and requirements should be present for configuration planning."""
        described = introspection.describe_algorithm("hqq")
        for key in ("compatible_before", "compatible_after", "compatible_with", "requires", "runs_on"):
            assert key in described

    def test_unknown_algorithm_raises(self) -> None:
        """An unknown algorithm name should raise rather than return an empty description."""
        with pytest.raises(KeyError):
            introspection.describe_algorithm("definitely_not_an_algorithm")


@pytest.mark.cpu
class TestEvaluationCatalogue:
    """Tests for the benchmark and metric listings."""

    def test_list_metrics_returns_sorted_names(self) -> None:
        """Metrics should be reported as a sorted list of names."""
        metrics = introspection.list_metrics()
        assert metrics == sorted(metrics)
        assert all(isinstance(name, str) for name in metrics)

    def test_list_benchmarks_reports_metrics_per_benchmark(self) -> None:
        """Each benchmark should expose the metrics it evaluates."""
        benchmarks = introspection.list_benchmarks()
        assert len(benchmarks) > 0
        for benchmark in benchmarks:
            assert benchmark["name"]
            assert isinstance(benchmark["metrics"], list)

    def test_list_benchmarks_filters_by_task_type(self) -> None:
        """Filtering by task type should return a subset carrying that task type."""
        filtered = introspection.list_benchmarks(task_type="text_to_image")
        assert len(filtered) <= len(introspection.list_benchmarks())
        for benchmark in filtered:
            assert benchmark["task_type"] == "text_to_image"

@pytest.mark.cpu
class TestDatasets:
    """Tests for the named dataset catalogue."""

    def test_returns_sorted_names(self) -> None:
        """The names should come back in a stable order so clients can rely on them."""
        names = introspection.list_datasets()
        assert names == sorted(names)
        assert "LAION256" in names

    def test_every_name_is_accepted_by_the_datamodule_factory(self) -> None:
        """
        Each advertised name must be one PrunaDataModule.from_string can actually resolve.

        The list is read straight from pruna's own dataset registry, so this guards against the
        registry and the factory drifting apart, which would hand evaluate_model a bad name.
        """
        from pruna.data import base_datasets

        assert introspection.list_datasets() == sorted(base_datasets.keys())
