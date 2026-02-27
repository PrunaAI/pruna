#!/usr/bin/env bash
set -e
echo "=== Ruff check ==="
uv run ruff check src/pruna/
echo "=== Ruff format check ==="
uv run ruff format --check src/pruna/
echo "=== Ty type checker ==="
uv run ty check src/pruna
echo "=== Pytest style ==="
uv run pytest -m "style" -q
echo "=== All lint checks passed ==="
