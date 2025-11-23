# Publishing Guide

## Pre-publishing Checklist

- [x] `.gitignore` - Excludes build artifacts, cache files, data
- [x] `LICENSE` - MIT license included
- [x] `README.md` - Comprehensive documentation
- [x] `CONTRIBUTING.md` - Contribution guidelines
- [x] `pyproject.toml` - Proper metadata and dependencies
- [x] `MANIFEST.in` - Package manifest
- [ ] Update version in `pyproject.toml` for releases
- [ ] Update repository URLs in `pyproject.toml` with actual GitHub URL

## Building the Package

```bash
# Install build tools
pip install build twine

# Build distribution
python -m build

# This creates:
# - dist/eggroll_trainer-0.1.0.tar.gz (source distribution)
# - dist/eggroll_trainer-0.1.0-py3-none-any.whl (wheel)
```

## Publishing to PyPI

```bash
# Test on TestPyPI first
twine upload --repository testpypi dist/*

# If successful, publish to PyPI
twine upload dist/*
```

## Version Bumping

Update version in `pyproject.toml`:
- Patch: 0.1.0 → 0.1.1 (bug fixes)
- Minor: 0.1.0 → 0.2.0 (new features)
- Major: 0.1.0 → 1.0.0 (breaking changes)

## Git Tags

After publishing:
```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

