# GitHub Workflows

This directory contains GitHub Actions workflows for the Pruna project.

## Workflow Overview

### Reusable Workflows

#### Linting (`linting.yaml`)
- **Triggers**: Changes to source code, configuration files, or linting rules
- **Reusable**: Can be called by other workflows
- **Concurrency**: Uses shared concurrency group to prevent multiple simultaneous runs
- **Purpose**: Centralized code quality checks (ruff, mypy, docstring validation)
- **Outputs**: Provides success status to calling workflows

### Individual Workflows

#### CPU Tests (`cpu_tests.yaml`)
- **Triggers**: Changes to source code, tests, or dependencies
- **Concurrency**: Cancels in-progress runs
- **Job Order**:
  1. **Linting** (reusable workflow)
  2. **CPU tests** (runs after linting passes)
- **Purpose**: Fast CPU-only tests with early code quality validation

#### Installation Tests (`installation.yaml`)
- **Triggers**: Changes to package configuration or source code
- **Concurrency**: Cancels in-progress runs
- **Job Order**:
  1. **Linting** (reusable workflow)
  2. **Installation tests** (runs after linting passes)
- **Purpose**: Verify package installation across different platforms and Python versions

#### Documentation Generation (`documentation.yaml`)
- **Triggers**: Changes to algorithms or documentation generation scripts
- **Concurrency**: Cancels in-progress runs
- **Job Order**:
  1. **Linting** (reusable workflow)
  2. **Documentation generation** (runs after linting passes)
- **Purpose**: Generate algorithm documentation with code quality validation

#### Package Build (`package_build.yaml`)
- **Triggers**: Changes to source code or package configuration
- **Concurrency**: Cancels in-progress runs
- **Job Order**:
  1. **Linting** (reusable workflow)
  2. **Package build** (runs after linting passes)
- **Purpose**: Build and package the project with code quality validation

## Concurrency Controls

All workflows use concurrency controls to prevent resource waste:

```yaml
concurrency:
  group: ci-${{ github.repository }}-[workflow-name]-${{ github.ref }}
  cancel-in-progress: true
```

This ensures that:
- Only one workflow run per branch is active at a time
- New commits cancel in-progress runs to avoid outdated results
- Resources are used efficiently

## Reusable Workflow Pattern

The linting workflow is designed as a **reusable workflow** that can be called by other workflows:

```yaml
jobs:
  linting:
    uses: ./.github/workflows/linting.yaml
```

This approach provides:
- **DRY principle**: No code duplication across workflows
- **Centralized maintenance**: Linting logic is maintained in one place
- **Consistency**: All workflows use the same linting checks
- **Flexibility**: Linting can run standalone or as part of other workflows

### Shared Concurrency Optimization

The linting workflow uses a **shared concurrency group** to prevent multiple simultaneous runs:

```yaml
concurrency:
  group: ci-${{ github.repository }}-linting
  cancel-in-progress: true
```

This ensures:
- **Single linting run**: Even when multiple workflows trigger simultaneously, only one linting job runs
- **Resource efficiency**: No duplicate linting processes
- **Consistent results**: All workflows get the same linting status
- **Faster CI**: Eliminates redundant linting checks

## Job Ordering Strategy

All workflows follow a **fast-feedback-first** approach:

1. **Linting** (reusable workflow, fast, catches code quality issues early)
2. **Main tests** (CPU tests, installation, documentation, package build)

This strategy:
- **Saves resources**: Fails fast on code quality issues before running expensive tests
- **Provides quick feedback**: Developers get linting results quickly
- **Prevents waste**: No point running tests if code doesn't meet quality standards
- **Maintains consistency**: All workflows use the same linting checks
- **Optimizes performance**: Shared concurrency prevents duplicate linting runs

## Usage

- **Automatic**: Workflows run automatically on relevant file changes
- **Manual**: Some workflows support `workflow_dispatch` for manual triggering
- **Branch Protection**: Use these workflows in branch protection rules for required status checks
- **Standalone Linting**: The linting workflow can run independently for quick code quality checks
