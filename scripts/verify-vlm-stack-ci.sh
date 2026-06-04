#!/usr/bin/env bash
# Run CI-equivalent lint + tests for each VLM stack branch before pushing.
#
# Usage:
#   ./scripts/verify-vlm-stack-ci.sh
#   ./scripts/verify-vlm-stack-ci.sh feat/vlm-pr-3b-oneig-alignment
#
# Lint: ruff + ty + docstring checks on evaluation.metrics (see tests.yaml).
# Tests: pytest with the same markers as CI base matrix (cpu, no_extras, …).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MARK='cpu and not slow and not style and no_extras'
export PRUNA_CI_CPU_ONLY=1

INFRA_TEST_TEMPLATE="$ROOT/scripts/test_vlm_base_infrastructure_infra.py"

branches=(
  feat/vlm-pr-1-vendor
  feat/vlm-pr-2-infrastructure
  feat/vlm-pr-3a-qa-accuracy
  feat/vlm-pr-3b-oneig-alignment
  feat/vlm-pr-3c-text-score-pair
  feat/vlm-pr-3d-oneig-reasoning
  feat/vlm-pr-4a-vqa
  feat/vlm-pr-4b-vie-score
  feat/vlm-pr-4c-img-edit-score
  feat/vlm-pr-5-e2e-tests
)

if [[ $# -gt 0 ]]; then
  branches=("$@")
fi

# Branches before e2e use infrastructure-only VLM tests (no forward imports).
needs_infra_test_only() {
  case "$1" in
    feat/vlm-pr-2-infrastructure|feat/vlm-pr-3a-qa-accuracy|feat/vlm-pr-3b-oneig-alignment|feat/vlm-pr-3c-text-score-pair|feat/vlm-pr-3d-oneig-reasoning|feat/vlm-pr-4a-vqa|feat/vlm-pr-4b-vie-score|feat/vlm-pr-4c-img-edit-score)
      return 0
      ;;
  esac
  return 1
}

tests_for_branch() {
  local b=$1
  if needs_infra_test_only "$b" || [[ "$b" == "feat/vlm-pr-2-infrastructure" ]]; then
    if [[ -f tests/evaluation/test_vlm_base_infrastructure.py ]]; then
      echo "tests/evaluation/test_vlm_base_infrastructure.py"
    fi
  elif git cat-file -e "$b:tests/evaluation/test_vlm_base_infrastructure.py" 2>/dev/null; then
    echo "tests/evaluation/test_vlm_base_infrastructure.py"
  fi
  if git cat-file -e "$b:tests/evaluation/test_text_metrics.py" 2>/dev/null; then
    echo "tests/evaluation/test_text_metrics.py"
  fi
  if git cat-file -e "$b:tests/evaluation/_vlm_batch_snapshot_helpers.py" 2>/dev/null; then
    echo "tests/evaluation/_vlm_batch_snapshot_helpers.py"
  fi
}

run_lint() {
  echo "--- ruff (src/pruna) ---"
  uv run ruff check src/pruna

  echo "--- ty (src/pruna) ---"
  uv run ty check src/pruna

  echo "--- docstring style (evaluation.metrics) ---"
  uv run pytest -m style -q tests/style/test_docstrings.py -k "evaluation.metrics" --maxfail=3
}

run_tests() {
  local b=$1
  local -a paths=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && paths+=("$line")
  done < <(tests_for_branch "$b")

  if [[ ${#paths[@]} -eq 0 ]]; then
    echo "(no VLM-specific tests on this branch; skipping pytest)"
    return 0
  fi

  echo "--- pytest (${paths[*]}) ---"
  uv run pytest -q --tb=line -m "$MARK" --maxfail=3 "${paths[@]}"
}

orig_branch=$(git branch --show-current 2>/dev/null || echo main)
failed=()

for b in "${branches[@]}"; do
  echo ""
  echo "========== $b =========="
  git checkout "$b" --quiet

  if needs_infra_test_only "$b" && [[ -f "$INFRA_TEST_TEMPLATE" ]]; then
    cp "$INFRA_TEST_TEMPLATE" tests/evaluation/test_vlm_base_infrastructure.py
  fi

  if ! uv sync --extra dev --quiet 2>/dev/null; then
    uv sync --extra dev
  fi

  if run_lint && run_tests "$b"; then
    echo "PASS $b"
  else
    echo "FAIL $b"
    failed+=("$b")
  fi
done

git checkout "$orig_branch" --quiet 2>/dev/null || true

echo ""
if [[ ${#failed[@]} -eq 0 ]]; then
  echo "All branches passed lint and tests."
  exit 0
fi
echo "Failed:"
printf '  - %s\n' "${failed[@]}"
exit 1
