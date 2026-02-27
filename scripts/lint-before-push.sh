#!/usr/bin/env bash
# Run the same lint checks as CI before pushing.
# Usage: ./scripts/lint-before-push.sh [path]
#   path: optional, e.g. src/pruna/data/ to scope checks (default: src/pruna/)

set -e
SCOPE="${1:-src/pruna/}"

echo "=== Ruff check ==="
uv run ruff check "$SCOPE"
echo "=== Ruff format check ==="
uv run ruff format --check "$SCOPE"
echo "=== Ty type checker ==="
uv run ty check src/pruna
echo "=== Pytest style ==="
uv run pytest -m "style" -q
echo "=== All lint checks passed ==="
