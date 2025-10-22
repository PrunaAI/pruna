import pytest

from pruna.algorithms import AlgorithmRegistry



def test_compatibility_symmetry():
    pruna_algorithms = AlgorithmRegistry._registry
    
    asymmetric_pairs = []
    
    for alg_a_name, alg_a in pruna_algorithms.items():
        for alg_b_name, alg_b in pruna_algorithms.items():
            if alg_a_name == alg_b_name:
                continue
            
            # Check symmetric relationship: if A is in B's compatible_after, 
            # then B should be in A's compatible_before
            if alg_b_name in alg_a.get_algorithms_to_run_after():
                if alg_a_name not in alg_b.get_algorithms_to_run_before():
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
    
    assert len(asymmetric_pairs) == 0, (
        f"Found {len(asymmetric_pairs)} asymmetric compatibility relationships:\n" +
        "\n".join(f"  - {pair}" for pair in asymmetric_pairs)
    )

