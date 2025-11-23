# GitHub Actions Workflows

This directory contains GitHub Actions workflows for CI/CD, testing, building, and releasing.

## Workflows Overview

### 1. `ci.yml` - Continuous Integration
**Triggers**: Push to main/master, Pull requests

**What it does**:
- Runs comprehensive test suite
- Runs EGGROLL-specific tests
- Runs basic example
- Verifies package imports
- Checks package version

**Status**: ✅ Active

### 2. `test.yml` - Legacy Test Suite
**Triggers**: Push to main/master, Pull requests

**What it does**:
- Runs comprehensive tests
- Runs EGGROLL tests
- Optional MNIST test

**Status**: ✅ Active (may be consolidated into ci.yml)

### 3. `build.yml` - Package Build Verification
**Triggers**: Push to main/master (when code changes), Pull requests

**What it does**:
- Builds the package (wheel + sdist)
- Verifies package can be installed
- Checks package contents
- Uploads build artifacts

**Status**: ✅ Active

### 4. `release.yml` - PyPI Release
**Triggers**: 
- GitHub Release published
- Manual workflow dispatch

**What it does**:
- Extracts version from tag or input
- Verifies version matches pyproject.toml
- Builds package
- Publishes to TestPyPI (manual) or PyPI (release)
- Updates release notes

**Status**: ✅ Active (requires PyPI API tokens)

### 5. `docs.yml` - Documentation Deployment
**Triggers**: Push to main (when docs change)

**What it does**:
- Builds MkDocs documentation
- Deploys to GitHub Pages

**Status**: ✅ Active

## Setup Required

### For PyPI Releases

1. **Create PyPI API tokens**:
   - Production: https://pypi.org/manage/account/token/
   - Test: https://test.pypi.org/manage/account/token/

2. **Add GitHub Secrets**:
   - `PYPI_API_TOKEN` - Production PyPI token
   - `TEST_PYPI_API_TOKEN` - TestPyPI token

3. See `.github/RELEASE_SETUP.md` for detailed instructions

## Usage

### Running Tests Locally

```bash
# Install dependencies
uv sync

# Run tests
uv run python examples/test_comprehensive.py
uv run python examples/test_eggroll.py
```

### Building Package Locally

```bash
# Install build tools
uv pip install build

# Build package
python -m build

# Verify installation
uv pip install dist/*.whl
uv run python -c "from eggroll_trainer import EGGROLLTrainer"
```

### Creating a Release

1. Update version in `pyproject.toml`
2. Commit and push
3. Create GitHub Release with tag `v<version>`
4. Workflow automatically publishes to PyPI

## Workflow Status

All workflows use:
- ✅ Latest GitHub Actions versions
- ✅ `uv` for fast dependency management
- ✅ Python 3.12
- ✅ Ubuntu latest

## Troubleshooting

See `.github/RELEASE_SETUP.md` for release troubleshooting.

For CI issues, check:
- Workflow logs in GitHub Actions tab
- Test failures in test output
- Build errors in build logs

