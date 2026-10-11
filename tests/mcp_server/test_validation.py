"""Tests for model-free configuration validation used by the MCP server."""

import json

import pytest

from pruna.mcp_server import introspection, validation

# Subjects chosen so assertions do not depend on which optional extras are installed.
# Each is a real registered algorithm in pruna's catalogue.
CORE_ALGORITHM = "torch_compile"  # needs no tokenizer, processor or dataset, and no pip extra
QUANTIZER_ALGORITHM = "quanto"  # core-importable, and exposes a tunable weight-bit hyperparameter
EXTRA_ONLY_ALGORITHM = "stable_fast"  # importable only with pruna[stable-fast], absent from the dev extra
TOKENIZER_ALGORITHM = "padding_pruning"  # always requires a tokenizer to be attached


def _is_importable(algorithm_name: str) -> bool:
    """
    Report whether an algorithm's own dependencies can be imported.

    Parameters
    ----------
    algorithm_name : str
        The algorithm to probe.

    Returns
    -------
    bool
        True if the algorithm's imports succeed.
    """
    from pruna import AlgorithmRegistry

    try:
        AlgorithmRegistry[algorithm_name].import_algorithm_packages()
    except ImportError:
        return False
    return True


@pytest.mark.cpu
class TestValidateConfig:
    """Tests for the GPU-free configuration validator."""

    def test_core_config_reports_no_errors(self) -> None:
        """An algorithm with no unmet requirements should validate cleanly."""
        report = validation.validate_config(algorithms=[CORE_ALGORITHM])
        assert report["errors"] == []
        assert report["valid"] is True
        assert report["active_algorithms"] == [CORE_ALGORITHM]

    def test_missing_pip_extra_is_reported(self) -> None:
        """An algorithm behind an uninstalled extra should be reported, not swallowed."""
        if _is_importable(EXTRA_ONLY_ALGORITHM):
            pytest.skip(f"{EXTRA_ONLY_ALGORITHM} is installed in this environment")

        report = validation.validate_config(algorithms=[EXTRA_ONLY_ALGORITHM])
        assert report["valid"] is False
        assert any(
            error["check"] == "check_algorithm_packages_availability" for error in report["errors"]
        )
        assert EXTRA_ONLY_ALGORITHM in report["requirements"]["pip_install"] or any(
            EXTRA_ONLY_ALGORITHM in error["message"] for error in report["errors"]
        )

    def test_missing_tokenizer_is_reported(self) -> None:
        """An algorithm that needs a tokenizer should say so when none is attached."""
        report = validation.validate_config(algorithms=[TOKENIZER_ALGORITHM])
        assert report["valid"] is False
        assert any(error["check"] == "check_argument_compatibility" for error in report["errors"])
        assert TOKENIZER_ALGORITHM in report["requirements"]["tokenizer"]

    def test_resolves_a_value_for_every_declared_hyperparameter(self) -> None:
        """Every declared hyperparameter should appear in the resolved configuration."""
        from pruna import AlgorithmRegistry

        declared = {hp.name for hp in AlgorithmRegistry[CORE_ALGORITHM].hyperparameters}
        resolved = validation.validate_config(algorithms=[CORE_ALGORITHM])["resolved_hyperparameters"]
        assert set(resolved[CORE_ALGORITHM]) == declared

    def test_resolved_configuration_is_json_serializable(self) -> None:
        """The report has to survive the JSON encoding an MCP client applies."""
        json.dumps(validation.validate_config(algorithms=[CORE_ALGORITHM, QUANTIZER_ALGORITHM]))

    def test_untouched_hyperparameter_reports_its_default(self) -> None:
        """A hyperparameter that was never set should fall back to its declared default."""
        from pruna import AlgorithmRegistry

        report = validation.validate_config(algorithms=[CORE_ALGORITHM])
        for hyperparameter in AlgorithmRegistry[CORE_ALGORITHM].hyperparameters:
            resolved = report["resolved_hyperparameters"][CORE_ALGORITHM][hyperparameter.name]
            assert resolved == validation._to_jsonable(hyperparameter.default_value)

    def test_explicit_hyperparameter_is_respected(self) -> None:
        """A configured hyperparameter must win over the default."""
        report = validation.validate_config(
            algorithms=[CORE_ALGORITHM],
            hyperparameters={f"{CORE_ALGORITHM}_max_kv_cache_size": 512},
        )
        resolved = report["resolved_hyperparameters"][CORE_ALGORITHM]
        assert resolved[f"{CORE_ALGORITHM}_max_kv_cache_size"] == 512

    def test_unknown_algorithm_is_reported_not_raised(self) -> None:
        """A bad configuration should come back as a report, not an exception."""
        report = validation.validate_config(algorithms=["not_a_real_algorithm"])
        assert report["valid"] is False
        assert report["errors"][0]["check"] == "build_config"

    def test_errors_carry_a_hint(self) -> None:
        """Every reported error should explain what went wrong."""
        for error in validation.validate_config(algorithms=["not_a_real_algorithm"])["errors"]:
            assert set(error) == {"check", "message", "hint"}
            assert error["message"]

    def test_report_keeps_its_shape_when_the_config_cannot_be_built(self) -> None:
        """
        A rejected hyperparameter value must still produce the full report.

        A client reads a fixed set of keys, so the failure path has to carry every key the
        success path does rather than a shorter dict.
        """
        from pruna import AlgorithmRegistry

        algorithm = AlgorithmRegistry[QUANTIZER_ALGORITHM]
        hyperparameter = algorithm.hyperparameters[0]

        report = validation.validate_config(
            algorithms=[QUANTIZER_ALGORITHM],
            hyperparameters={f"{QUANTIZER_ALGORITHM}_{hyperparameter.name}": object()},
        )
        success = validation.validate_config(algorithms=[CORE_ALGORITHM])

        assert set(report) == set(success)
        assert report["valid"] is False
        assert report["errors"][0]["check"] == "build_config"
        assert report["execution_order"] is None
        assert report["notes"]

    def test_a_rejected_hyperparameter_value_is_explained(self) -> None:
        """The message has to name the offending value so the model can correct it."""
        report = validation.validate_config(
            algorithms=[QUANTIZER_ALGORITHM],
            hyperparameters={f"{QUANTIZER_ALGORITHM}_weight_bits": "definitely-not-a-value"},
        )
        assert "weight_bits" in report["errors"][0]["message"]
        assert "describe_algorithm" in report["errors"][0]["hint"]

    def test_reports_the_execution_order(self) -> None:
        """The report should state the order the algorithms will actually be applied in."""
        report = validation.validate_config(algorithms=[CORE_ALGORITHM, QUANTIZER_ALGORITHM])
        order = report["execution_order"]
        assert order is not None
        assert set(order) == {CORE_ALGORITHM, QUANTIZER_ALGORITHM}

    def test_execution_order_respects_declared_constraints(self) -> None:
        """The reported order must match what pruna's own ordering routine produces."""
        from pruna import SmashConfig
        from pruna.config.pre_smash_routines import determine_algorithm_order

        algorithms = [CORE_ALGORITHM, QUANTIZER_ALGORITHM]
        config = SmashConfig(dict.fromkeys(algorithms, True))
        expected = determine_algorithm_order(None, config)
        assert validation.validate_config(algorithms=algorithms)["execution_order"] == expected

    def test_report_notes_what_still_needs_a_model(self) -> None:
        """The report should be explicit about the checks it could not perform."""
        report = validation.validate_config(algorithms=[CORE_ALGORITHM])
        assert any("model" in note.lower() for note in report["notes"])

    def test_reports_requirements_of_the_selected_algorithms(self) -> None:
        """The requirements block should reflect only the selected algorithms."""
        requirements = validation.validate_config(algorithms=[CORE_ALGORITHM])["requirements"]
        assert set(requirements) == {"tokenizer", "processor", "dataset", "supported_devices", "pip_install"}
        assert requirements["tokenizer"] == []


@pytest.mark.cpu
class TestRoundTrip:
    """Tests that what the server advertises can actually be configured."""

    @staticmethod
    def _importable_names() -> list[str]:
        """Return the registry names whose own dependencies import successfully."""
        from pruna import AlgorithmRegistry

        names = []
        for algorithm in AlgorithmRegistry.get_all_algorithms():
            try:
                algorithm.import_algorithm_packages()
            except Exception:  # an uninstalled extra or a missing optional dependency
                continue
            names.append(algorithm.algorithm_name)
        return names

    def test_every_advertised_default_is_accepted_back(self) -> None:
        """
        Feeding describe_algorithm's own defaults into validate_config must succeed.

        This is the contract the model relies on: the server states a default for each
        hyperparameter, so that exact value has to be one pruna will accept.
        """
        from pruna import SmashConfig

        checked = 0
        for name in self._importable_names():
            described = introspection.describe_algorithm(name)
            payload: dict[str, object] = {name: True}
            for hyperparameter in described["hyperparameters"]:
                payload[hyperparameter["qualified_name"]] = hyperparameter["default"]
            try:
                SmashConfig(payload)
            except Exception as error:  # noqa: BLE001 - the assertion reports the detail
                pytest.fail(f"{name}: describe_algorithm advertised a default pruna rejected: {error}")
            checked += 1

        assert checked > 10, f"only {checked} algorithms were exercised, too few to be meaningful"

    def test_every_advertised_allowed_value_is_accepted_back(self) -> None:
        """
        Every value describe_algorithm lists as allowed must be one pruna accepts.

        Allowed values are read off the ConfigSpace hyperparameter, so this guards the
        attribute the reader trusts is the authoritative one.
        """
        from pruna import SmashConfig

        exercised = 0
        for name in self._importable_names():
            described = introspection.describe_algorithm(name)
            for hyperparameter in described["hyperparameters"]:
                allowed = hyperparameter.get("allowed")
                if not allowed:
                    continue
                for value in allowed:
                    payload: dict[str, object] = {name: True, hyperparameter["qualified_name"]: value}
                    try:
                        SmashConfig(payload)
                    except Exception as error:  # noqa: BLE001 - the assertion reports the detail
                        pytest.fail(
                            f"{name}.{hyperparameter['qualified_name']}: describe_algorithm advertised "
                            f"{value!r} as allowed but pruna rejected it: {error}"
                        )
                    exercised += 1

        assert exercised > 20, f"only {exercised} values were exercised, too few to be meaningful"


@pytest.mark.cpu
class TestIncompatiblePairs:
    """Tests for the static pairwise compatibility check."""

    @staticmethod
    def _registry_names() -> list[str]:
        """Return every registered algorithm name."""
        from pruna import AlgorithmRegistry

        return [algorithm.algorithm_name for algorithm in AlgorithmRegistry.get_all_algorithms()]

    def test_single_algorithm_has_no_pairs(self) -> None:
        """A lone algorithm cannot conflict with anything."""
        assert validation.find_incompatible_pairs([CORE_ALGORITHM]) == []

    def test_reported_pairs_are_genuinely_incompatible(self) -> None:
        """Every reported pair must be absent from both compatibility lists."""
        from pruna import AlgorithmRegistry

        for first, second in validation.find_incompatible_pairs(self._registry_names()):
            other = AlgorithmRegistry[second]
            assert first not in other.get_compatible_algorithms()
            assert first not in other.get_disjointly_compatible_algorithms()

    def test_compatible_algorithms_are_not_reported(self) -> None:
        """Algorithms declared mutually compatible must not be flagged."""
        from pruna import AlgorithmRegistry

        for first, second in validation.find_incompatible_pairs(self._registry_names()):
            assert not (
                first in AlgorithmRegistry[second].get_compatible_algorithms()
                or second in AlgorithmRegistry[first].get_compatible_algorithms()
            )

    def test_incompatible_pair_surfaces_as_an_error(self) -> None:
        """An incompatible combination should make the whole report invalid."""
        found = validation.find_incompatible_pairs(self._registry_names())
        if not found:
            pytest.skip("no algorithm pair in the registry is declared incompatible")

        report = validation.validate_config(algorithms=list(found[0]))
        assert report["valid"] is False
        assert any(error["check"] == "cross_compatibility" for error in report["errors"])


@pytest.mark.cpu
class TestBuildConfig:
    """Tests for constructing a SmashConfig from flat tool arguments."""

    def test_activates_the_named_algorithms(self) -> None:
        """Every named algorithm should end up active."""
        config = validation.build_config([CORE_ALGORITHM, QUANTIZER_ALGORITHM])
        assert set(config.get_active_algorithms()) == {CORE_ALGORITHM, QUANTIZER_ALGORITHM}

    def test_records_batch_size(self) -> None:
        """The requested batch size should be stored on the configuration."""
        assert validation.build_config([CORE_ALGORITHM], batch_size=4).batch_size == 4

    def test_records_device_when_given(self) -> None:
        """An explicit device should be stored on the configuration."""
        assert str(validation.build_config([CORE_ALGORITHM], device="cpu").device) == "cpu"