# GitHub Actions CI/CD Workflows

This directory contains automated workflows for continuous integration and deployment.

## Workflows

### `test.yml` - Automated Testing

**Triggers:**
- Pull requests to `main` branch
- Pushes to `main` branch

**Jobs:**

#### 1. Fast Tests
- **Runtime**: ~2-5 minutes
- **Python Version**: 3.10
- **Test Strategy**: Excludes slow and GPU tests for fast feedback
- **Command**: `pytest -m "not slow and not gpu"`
- **Features**:
  - Fails fast on first 5 failures (`--maxfail=5`)
  - Shows 10 slowest tests (`--durations=10`)
  - Uses `uv` for 10x faster dependency installation
  - Caches dependencies for subsequent runs

#### 2. Linting
- **Runtime**: ~30 seconds
- **Tools**: Ruff (linting and formatting)
- **Strategy**: Non-blocking (continues on errors)
- **Checks**:
  - Code style violations
  - Common bugs and anti-patterns
  - Import sorting
  - Format compliance

## Test Markers

Tests use pytest markers to categorize by speed and requirements:

- `slow` - Tests that load large models or run for >5 seconds
- `gpu` - Tests requiring GPU/CUDA
- `integration` - End-to-end integration tests

**Fast tests** (run in CI): All tests except `slow` and `gpu`  
**Local full tests**: `pytest tests/`  
**Local fast tests**: `pytest tests/ -m "not slow"`

## Running Locally

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run the same tests as CI
pytest tests/ -v -m "not slow and not gpu" --tb=short

# Run linting
ruff check ragcun/ scripts/ tests/
ruff format --check ragcun/ scripts/ tests/
```

## CI/CD Performance

**Optimizations:**
- ✅ Uses `uv` instead of `pip` (10-100x faster)
- ✅ Caches pip/uv dependencies between runs
- ✅ Skips slow model-loading tests
- ✅ Skips GPU-requiring tests
- ✅ Fails fast on 5 errors to save time
- ✅ Parallel linting job

**Expected Runtime:**
- First run (cold cache): ~5 minutes
- Subsequent runs (warm cache): ~2 minutes
- Linting: ~30 seconds

## Adding More Workflows

To add additional workflows (deployment, release, etc.):

1. Create a new `.yml` file in this directory
2. Follow the same structure as `test.yml`
3. Use appropriate triggers (`on:` section)
4. Test locally with `act` (GitHub Actions local runner)

## Troubleshooting

### Tests fail in CI but pass locally
- Check Python version matches (3.10)
- Verify all dependencies are in `pyproject.toml`
- Look for environment-specific issues (paths, etc.)

### CI is too slow
- Add more tests to `slow` marker: `@pytest.mark.slow`
- Consider splitting into more parallel jobs
- Check if dependencies can be further optimized

### Linting failures
- Run locally: `ruff check --fix ragcun/`
- Auto-format: `ruff format ragcun/`
- See `pyproject.toml` for ruff configuration

## Status Badges

Add to your `README.md`:

```markdown
![Tests](https://github.com/ctn/ragcun/workflows/Tests/badge.svg?branch=main)
```

## Further Reading

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [uv Documentation](https://github.com/astral-sh/uv)
- [Ruff Documentation](https://github.com/astral-sh/ruff)

