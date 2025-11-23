# Release Setup Guide

This guide explains how to set up automated PyPI releases using GitHub Actions.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org if you don't have one
2. **TestPyPI Account**: Create an account at https://test.pypi.org for testing
3. **GitHub Repository**: This repository must be on GitHub

## Setting Up PyPI API Tokens

### 1. Create PyPI API Token (Production)

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Token name: `eggroll-trainer-github-actions`
4. Scope: **Entire account** (or project-specific if preferred)
5. Click "Add token"
6. **Copy the token immediately** (you won't see it again!)

### 2. Create TestPyPI API Token (Testing)

1. Go to https://test.pypi.org/manage/account/token/
2. Click "Add API token"
3. Token name: `eggroll-trainer-testpypi`
4. Scope: **Entire account**
5. Click "Add token"
6. **Copy the token immediately**

### 3. Add Secrets to GitHub Repository

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add these secrets:

   - **Name**: `PYPI_API_TOKEN`
     **Value**: Your PyPI API token (from step 1)
   
   - **Name**: `TEST_PYPI_API_TOKEN`
     **Value**: Your TestPyPI API token (from step 2)

## Release Process

### Automated Release (Recommended)

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # Increment as needed
   ```

2. **Commit and push**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.1"
   git push origin main
   ```

3. **Create GitHub Release**:
   - Go to GitHub → **Releases** → **Draft a new release**
   - **Tag version**: `v0.1.1` (must match version in pyproject.toml, with 'v' prefix)
   - **Release title**: `Release v0.1.1`
   - **Description**: Add release notes
   - Click **Publish release**

4. **Workflow runs automatically**:
   - Builds the package
   - Verifies version matches tag
   - Publishes to PyPI
   - Updates release notes

### Manual Release (Alternative)

If you need to test on TestPyPI first:

1. Go to **Actions** → **Release to PyPI** → **Run workflow**
2. Enter version (e.g., `0.1.1`)
3. Click **Run workflow**
4. This will publish to TestPyPI for testing
5. Once verified, create a GitHub Release to publish to production PyPI

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **Patch** (0.1.0 → 0.1.1): Bug fixes
- **Minor** (0.1.0 → 0.2.0): New features, backward compatible
- **Major** (0.1.0 → 1.0.0): Breaking changes

## Workflow Files

- **`.github/workflows/ci.yml`** - Runs tests on every push/PR
- **`.github/workflows/build.yml`** - Builds and verifies package
- **`.github/workflows/release.yml`** - Publishes to PyPI on release
- **`.github/workflows/test.yml`** - Legacy test workflow
- **`.github/workflows/docs.yml`** - Builds and deploys documentation

## Troubleshooting

### "Version mismatch" error

The tag version must match the version in `pyproject.toml`:
- Tag: `v0.1.1`
- pyproject.toml: `version = "0.1.1"`

### "Authentication failed" error

Check that:
1. PyPI API tokens are set correctly in GitHub Secrets
2. Token names match exactly: `PYPI_API_TOKEN` and `TEST_PYPI_API_TOKEN`
3. Tokens haven't expired (they don't expire, but check if regenerated)

### "Package already exists" error

The version already exists on PyPI. Either:
1. Bump the version number
2. Delete the existing version (if it was a mistake)

## Verification

After release, verify:

1. **Package is on PyPI**: https://pypi.org/project/eggroll-trainer/
2. **Can be installed**: `pip install eggroll-trainer==0.1.1`
3. **Imports work**: `python -c "from eggroll_trainer import EGGROLLTrainer"`

## Security Notes

- Never commit API tokens to the repository
- Use GitHub Secrets for all sensitive data
- Rotate tokens if compromised
- Use project-specific tokens if possible (more secure than account-wide)

