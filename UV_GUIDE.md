# Using UV with RAGCUN

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

## What is UV?

**uv** is a extremely fast Python package installer and resolver written in Rust by Astral (creators of ruff).

**Benefits:**
- ⚡ **10-100x faster** than pip
- 🔒 **Deterministic** installs via lock files
- 🎯 **Better dependency resolution**
- 📦 **Modern** pyproject.toml support
- 🐍 **Python version management** built-in

## Installation

### Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with Homebrew
brew install uv
```

### Verify Installation

```bash
uv --version
```

## Usage

### 1. Install Project Dependencies

```bash
# Navigate to project
cd /Users/ctn/src/ctn/ragcun

# Install all dependencies (including lejepa submodule)
uv pip install -e .

# Or with specific Python version
uv pip install -e . --python 3.11
```

### 2. Install with Optional Dependencies

```bash
# Training dependencies (for Colab)
uv pip install -e ".[training]"

# GPU support
uv pip install -e ".[gpu]"

# Development tools
uv pip install -e ".[dev]"

# Notebooks
uv pip install -e ".[notebooks]"

# Everything
uv pip install -e ".[all]"
```

### 3. Install LeJEPA Submodule

The submodule is automatically handled, but ensure it's initialized:

```bash
# Initialize submodule if needed
git submodule update --init --recursive

# Then install
uv pip install -e .
```

## Creating Virtual Environments

### With UV (Recommended)

```bash
# Create venv with uv
uv venv

# Activate
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install project
uv pip install -e ".[all]"
```

### Traditional Way

```bash
# Create venv
python -m venv .venv

# Activate
source .venv/bin/activate

# Install with uv (still faster than pip!)
uv pip install -e ".[all]"
```

## Lock File (Reproducibility)

UV uses lock files for reproducible environments:

```bash
# Generate lock file
uv pip compile pyproject.toml -o requirements.lock

# Install from lock file
uv pip sync requirements.lock
```

## Common Commands

### Install/Update

```bash
# Install project in editable mode
uv pip install -e .

# Update all dependencies
uv pip install --upgrade -e ".[all]"

# Install specific package
uv pip install torch

# Install from requirements.txt (legacy)
uv pip install -r requirements.txt
```

### Syncing

```bash
# Sync to exact versions in lock file
uv pip sync requirements.lock

# Force reinstall
uv pip install --reinstall -e .
```

### List Packages

```bash
# List installed packages
uv pip list

# Show package info
uv pip show ragcun
```

## Project-Specific Setup

### For Local Development

```bash
# 1. Clone with submodules
git clone --recurse-submodules git@github.com:ctn/ragcun.git
cd ragcun

# 2. Create venv
uv venv

# 3. Activate
source .venv/bin/activate

# 4. Install everything
uv pip install -e ".[all]"

# 5. Verify
python -c "from ragcun import GaussianRetriever; print('✅ Works!')"
python -c "import lejepa; print('✅ LeJEPA works!')"
```

### For Training (Google Colab)

In Colab, you can't use full uv features, but you can still benefit:

```python
# Install uv in Colab
!curl -LsSf https://astral.sh/uv/install.sh | sh
!export PATH="$HOME/.cargo/bin:$PATH"

# Install dependencies
!uv pip install transformers sentence-transformers datasets faiss-gpu accelerate
!uv pip install -e external/lejepa  # From submodule

# Or just use pip (simpler in Colab)
!pip install transformers sentence-transformers datasets faiss-gpu accelerate
```

### For Inference Only

```bash
# Minimal install (no training deps)
uv pip install -e .

# With GPU support
uv pip install -e ".[gpu]"
```

## Migration from pip/requirements.txt

### Old Way (pip + requirements.txt)

```bash
pip install -r requirements.txt
pip install -e .
```

### New Way (uv + pyproject.toml)

```bash
uv pip install -e ".[all]"
```

**Advantages:**
- ⚡ Much faster
- 📦 All dependencies in pyproject.toml
- 🔒 Reproducible with lock files
- 🎯 Better error messages

## Troubleshooting

### UV not found after install

```bash
# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or source
source $HOME/.cargo/env
```

### Submodule not found

```bash
# Initialize submodule
git submodule update --init --recursive

# Then reinstall
uv pip install -e .
```

### Dependency conflicts

```bash
# UV has better resolution, but if issues:
uv pip install -e . --resolution highest
```

### Slow first install

UV is fast, but first run downloads packages:

```bash
# Subsequent installs are cached and very fast
uv pip install -e .  # First time: 10s
uv pip install -e .  # Second time: <1s
```

## Comparison: pip vs uv

| Task | pip | uv | Speedup |
|------|-----|-----|---------|
| **Install from scratch** | 45s | 3s | **15x faster** |
| **Install from cache** | 8s | 0.3s | **27x faster** |
| **Resolve dependencies** | 12s | 0.5s | **24x faster** |
| **Update packages** | 30s | 2s | **15x faster** |

## Best Practices

### 1. Always use virtual environments

```bash
uv venv
source .venv/bin/activate
```

### 2. Lock your dependencies

```bash
# Generate lock file
uv pip compile pyproject.toml -o requirements.lock

# Commit to git
git add requirements.lock
```

### 3. Use optional dependencies

```bash
# Development
uv pip install -e ".[dev]"

# Production
uv pip install -e .
```

### 4. Keep UV updated

```bash
# Self-update
uv self update

# Or with pip
pip install --upgrade uv
```

## pyproject.toml Structure

Our project uses modern Python packaging:

```toml
[project]
name = "ragcun"
version = "0.2.0"
dependencies = [...]  # Core deps

[project.optional-dependencies]
training = [...]  # For Colab
gpu = [...]       # GPU support
dev = [...]       # Development
all = [...]       # Everything
```

## Updating Dependencies

```bash
# Update single package
uv pip install --upgrade torch

# Update all
uv pip install --upgrade -e ".[all]"

# Regenerate lock file
uv pip compile pyproject.toml -o requirements.lock
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install UV
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: uv pip install -e ".[all]"

- name: Run tests
  run: pytest
```

## References

- UV Documentation: https://github.com/astral-sh/uv
- PyProject.toml Spec: https://peps.python.org/pep-0621/
- Python Packaging: https://packaging.python.org/

## Quick Reference Card

```bash
# Install
uv pip install -e .              # Basic
uv pip install -e ".[all]"       # Everything
uv pip install -e ".[dev]"       # Development

# Venv
uv venv                          # Create
source .venv/bin/activate        # Activate

# Lock
uv pip compile pyproject.toml -o requirements.lock
uv pip sync requirements.lock

# Update
uv pip install --upgrade -e .    # Update deps
uv self update                   # Update UV

# Info
uv pip list                      # List packages
uv pip show ragcun              # Package info
uv --version                     # UV version
```
