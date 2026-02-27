#!/usr/bin/env bash
# Run the same lint checks as CI before pushing.
# Usage: ./scripts/lint-before-push.sh

set -e

echo "=== Ruff check ==="
uv run ruff check src/pruna/

echo "=== Ruff format check ==="
uv run ruff format --check src/pruna/

echo "=== Ty type checker ==="
uv run ty check src/pruna

echo "=== Pytest style (docstrings) ==="
uv run pytest -m "style" -q

echo "=== All lint checks passed ==="
