import pytest

from pruna.algorithms import AlgorithmRegistry


def test_compatibility_symmetry():
    pruna_algorithms = AlgorithmRegistry._registry
    
    asymmetric_pairs = []
    
    for alg_a_name, alg_a in pruna_algorithms.items():
        for alg_b_name, alg_b in pruna_algorithms.items():
            # Check symmetric relationship: if A is in B's compatible_after, 
            # then B should be in A's compatible_before
            if alg_b_name in alg_a.get_algorithms_to_run_after() and alg_a_name not in alg_b.get_algorithms_to_run_before():
                asymmetric_pairs.append(
                    f"{alg_a_name} lists {alg_b_name} in compatible_after, "
                    f"but {alg_b_name} does not list {alg_a_name} in compatible_before"
                )
            
            # Check symmetric relationship: if A is in B's compatible_before,
            # then B should be in A's compatible_after
            if alg_b_name in alg_a.get_algorithms_to_run_before():
                if alg_a_name not in alg_b.get_algorithms_to_run_after():
                    asymmetric_pairs.append(
                        f"{alg_a_name} lists {alg_b_name} in compatible_before, "
                        f"but {alg_b_name} does not list {alg_a_name} in compatible_after"
                    )
            
            # Check symmetric relationship: if A is in B's disjointly_compatible_after,
            # then B should be in A's disjointly_compatible_before
            if alg_b_name in alg_a.get_algorithms_to_run_after_disjointly() and alg_a_name not in alg_b.get_algorithms_to_run_before_disjointly():
                asymmetric_pairs.append(
                    f"{alg_a_name} lists {alg_b_name} in disjointly_compatible_after, "
                    f"but {alg_b_name} does not list {alg_a_name} in disjointly_compatible_before"
                )

            # Check symmetric relationship: if A is in B's disjointly_compatible_before,
            # then B should be in A's disjointly_compatible_after
            if alg_b_name in alg_a.get_algorithms_to_run_before_disjointly() and alg_a_name not in alg_b.get_algorithms_to_run_after_disjointly():
                asymmetric_pairs.append(
                    f"{alg_a_name} lists {alg_b_name} in disjointly_compatible_before, "
                    f"but {alg_b_name} does not list {alg_a_name} in disjointly_compatible_after"
                )
    
    assert len(asymmetric_pairs) == 0, (
        f"Found {len(asymmetric_pairs)} asymmetric compatibility relationships:\n" +
        "\n".join(f"  - {pair}" for pair in asymmetric_pairs)
    )


def test_disjointly_compatible_algorithms_have_target_modules():
    """Enforce that all algorithms with disjointly compatible algorithms have a target_modules hyperparameter."""
    pruna_algorithms = AlgorithmRegistry._registry

    missing_target_modules = []
    
    for alg_name, alg in pruna_algorithms.items():
        if alg.get_disjointly_compatible_algorithms():
            alg_hyperparameters = alg.get_hyperparameters()
            has_target_modules = any(param.name == "target_modules" for param in alg_hyperparameters)
            if not has_target_modules:
                missing_target_modules.append(alg_name)
    
    assert not missing_target_modules, (
        f"Found {len(missing_target_modules)} algorithms with disjointly compatible algorithms "
        f"but no target_modules hyperparameter: {missing_target_modules}"
    )


def test_no_redundant_disjoint_compatibility():
    """Enforce that no algorithm lists a disjointly compatible algorithm that is already marked as compatible in the same order."""
    pruna_algorithms = AlgorithmRegistry._registry

    redundant_disjoint_compatibility = {}

    for alg_name, alg in pruna_algorithms.items():
        redundant_before = list(set(alg.get_algorithms_to_run_before()) & set(alg.get_algorithms_to_run_before_disjointly()))
        redundant_after = list(set(alg.get_algorithms_to_run_after()) & set(alg.get_algorithms_to_run_after_disjointly()))
        if redundant_before or redundant_after:
            redundant_disjoint_compatibility[alg_name] = {}
        if redundant_before:
            redundant_disjoint_compatibility[alg_name]["before"] = redundant_before
        if redundant_after:
            redundant_disjoint_compatibility[alg_name]["after"] = redundant_after

    if not redundant_disjoint_compatibility:
        return

    report_header = f"Found redundant disjoint compatibility relationships in {len(redundant_disjoint_compatibility)} algorithms:"
    per_algorithm_report = [
        f"- {alg_name}: before: {redundancy.get('before', [])} - after: {redundancy.get('after', [])}"
        for alg_name, redundancy in redundant_disjoint_compatibility.items()
    ]
    report_detail = "\n".join(per_algorithm_report)
    raise ValueError(f"{report_header}\n{report_detail}")
