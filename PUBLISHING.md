# Publishing Guide

## Automated Publishing (Recommended)

The repository includes GitHub Actions workflows for automated publishing:

### Release Process

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # Patch: 0.1.0 → 0.1.1
   version = "0.2.0"  # Minor: 0.1.0 → 0.2.0
   version = "1.0.0"  # Major: 0.1.0 → 1.0.0
   ```

2. **Commit and push**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.1"
   git push origin main
   ```

3. **Create GitHub Release**:
   - Go to GitHub → Releases → Draft a new release
   - Tag: `v0.1.1` (must match version in pyproject.toml)
   - Title: `Release v0.1.1`
   - Description: Release notes
   - Publish release

4. **Automated workflow** will:
   - Build the package
   - Verify version matches
   - Publish to PyPI
   - Update release notes

### Manual Publishing (Alternative)

If you need to publish manually:

```bash
# Install build tools
pip install build twine

# Build distribution
python -m build

# Test on TestPyPI first
twine upload --repository testpypi dist/*

# If successful, publish to PyPI
twine upload dist/*
```

## Pre-publishing Checklist

- [x] `.gitignore` - Excludes build artifacts, cache files, data
- [x] `LICENSE` - MIT license included
- [x] `README.md` - Comprehensive documentation
- [x] `CONTRIBUTING.md` - Contribution guidelines
- [x] `pyproject.toml` - Proper metadata and dependencies
- [x] `MANIFEST.in` - Package manifest
- [ ] Update version in `pyproject.toml` for releases
- [ ] Update CHANGELOG.md (if exists)
- [ ] Run tests: `uv run python examples/test_comprehensive.py`
- [ ] Verify build: `python -m build`

## PyPI API Tokens

For automated publishing, set up PyPI API tokens:

1. **TestPyPI Token** (for testing):
   - Go to https://test.pypi.org/manage/account/token/
   - Create API token
   - Add as `TEST_PYPI_API_TOKEN` secret in GitHub repository

2. **PyPI Token** (for production):
   - Go to https://pypi.org/manage/account/token/
   - Create API token
   - Add as `PYPI_API_TOKEN` secret in GitHub repository

## Version Bumping

Update version in `pyproject.toml`:
- **Patch**: 0.1.0 → 0.1.1 (bug fixes)
- **Minor**: 0.1.0 → 0.2.0 (new features)
- **Major**: 0.1.0 → 1.0.0 (breaking changes)

## CI/CD Workflows

The repository includes:

- **`.github/workflows/ci.yml`** - Runs tests on push/PR
- **`.github/workflows/build.yml`** - Builds and verifies package
- **`.github/workflows/release.yml`** - Publishes to PyPI on release
- **`.github/workflows/test.yml`** - Legacy test workflow
- **`.github/workflows/docs.yml`** - Builds and deploys documentation

All workflows use `uv` for fast dependency management.

