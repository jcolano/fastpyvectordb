# Publishing FastPyVectorDB to PyPI

This guide walks you through publishing FastPyVectorDB to PyPI so users can install it with:

```bash
pip install fastpyvectordb
```

## Prerequisites

### 1. Create PyPI Accounts

You'll need accounts on both PyPI (production) and TestPyPI (testing):

1. **PyPI** (production): https://pypi.org/account/register/
2. **TestPyPI** (testing): https://test.pypi.org/account/register/

### 2. Create API Tokens

For secure uploads, create API tokens instead of using passwords:

**For PyPI:**
1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name it (e.g., "fastpyvectordb-upload")
4. Scope: "Entire account" (for first upload) or specific project later
5. Copy and save the token (starts with `pypi-`)

**For TestPyPI:**
1. Go to https://test.pypi.org/manage/account/token/
2. Same process as above

### 3. Install Build Tools

```bash
pip install build twine
```

---

## Step-by-Step Publishing Process

### Step 1: Verify Package Name Availability

Check if "fastpyvectordb" is available on PyPI:

```bash
# Check PyPI
pip index versions fastpyvectordb

# Or visit: https://pypi.org/project/fastpyvectordb/
```

If the name is taken, you'll need to choose a different name and update `pyproject.toml`:
```toml
[project]
name = "your-unique-name"
```

### Step 2: Update Version Number

Before each release, update the version in two places:

**pyproject.toml:**
```toml
[project]
version = "0.1.0"  # Update this
```

**fastpyvectordb/__init__.py:**
```python
__version__ = "0.1.0"  # Update this
```

Follow semantic versioning:
- `0.1.0` → `0.1.1` for bug fixes
- `0.1.0` → `0.2.0` for new features
- `0.1.0` → `1.0.0` for major/breaking changes

### Step 3: Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf dist/ build/ *.egg-info/
```

### Step 4: Build the Package

```bash
# Build source distribution and wheel
python -m build
```

This creates two files in `dist/`:
- `fastpyvectordb-0.1.0.tar.gz` (source distribution)
- `fastpyvectordb-0.1.0-py3-none-any.whl` (wheel)

### Step 5: Verify the Build

```bash
# Check the distribution
twine check dist/*

# List contents of the wheel
unzip -l dist/*.whl
```

### Step 6: Test on TestPyPI First (Recommended)

**Upload to TestPyPI:**
```bash
twine upload --repository testpypi dist/*
```

When prompted:
- Username: `__token__`
- Password: Your TestPyPI API token (including the `pypi-` prefix)

**Or use a config file** (`~/.pypirc`):
```ini
[testpypi]
username = __token__
password = pypi-your-test-token-here

[pypi]
username = __token__
password = pypi-your-production-token-here
```

**Test the installation:**
```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ fastpyvectordb

# Test it works
python -c "import fastpyvectordb; print(fastpyvectordb.__version__)"

# Cleanup
deactivate
rm -rf test_env
```

### Step 7: Upload to PyPI (Production)

Once testing is successful:

```bash
twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: Your PyPI API token

### Step 8: Verify the Release

```bash
# Wait a minute for PyPI to process
pip install fastpyvectordb

# Verify
python -c "import fastpyvectordb; print(fastpyvectordb.__version__)"
```

Visit your package page: https://pypi.org/project/fastpyvectordb/

---

## Automating with GitHub Actions (Optional)

Create `.github/workflows/publish.yml` for automated releases:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

**Setup:**
1. Go to your GitHub repo → Settings → Secrets and variables → Actions
2. Add a new secret: `PYPI_API_TOKEN` with your PyPI token
3. Create a release on GitHub to trigger the workflow

---

## Quick Reference

### One-Time Setup
```bash
# Install tools
pip install build twine

# Create ~/.pypirc (optional, for convenience)
cat > ~/.pypirc << 'EOF'
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE

[testpypi]
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
EOF

chmod 600 ~/.pypirc
```

### Each Release
```bash
# 1. Update version in pyproject.toml and fastpyvectordb/__init__.py

# 2. Clean and build
rm -rf dist/ build/ *.egg-info/
python -m build

# 3. Check
twine check dist/*

# 4. Upload to TestPyPI (optional but recommended)
twine upload --repository testpypi dist/*

# 5. Upload to PyPI
twine upload dist/*
```

---

## Troubleshooting

### "Package name already exists"
Choose a different name in `pyproject.toml`.

### "Invalid API token"
- Ensure you're using `__token__` as the username
- Include the full token (starting with `pypi-`)
- Check you're using the right token (TestPyPI vs PyPI)

### "File already exists"
You cannot overwrite an existing version. Bump the version number.

### Package imports fail after install
- Ensure all necessary files are included
- Check `[tool.setuptools]` in pyproject.toml
- Verify with `unzip -l dist/*.whl`

### Missing dependencies
Ensure all runtime dependencies are listed in `pyproject.toml`:
```toml
dependencies = [
    "numpy>=1.24.0",
    "hnswlib>=0.8.0",
]
```

---

## Checklist Before Publishing

- [ ] Version updated in `pyproject.toml` and `__init__.py`
- [ ] README.md is up to date
- [ ] All tests pass
- [ ] LICENSE file exists
- [ ] Dependencies are correct in `pyproject.toml`
- [ ] Package installs correctly from TestPyPI
- [ ] Package imports and basic functionality works

---

## Links

- PyPI: https://pypi.org/
- TestPyPI: https://test.pypi.org/
- Packaging Guide: https://packaging.python.org/tutorials/packaging-projects/
- Twine Documentation: https://twine.readthedocs.io/
