# Development Scripts

This directory contains scripts that replicate CI operations locally, making it easy to debug CI failures and run checks during development.

## 🔧 Available Scripts

### `ci_local.sh` - Complete CI Pipeline
Runs the complete CI pipeline locally, identical to what runs in GitHub Actions.

```bash
./scripts/ci_local.sh
```

**What it does:**
1. Code Quality Checks (linting, formatting, type checking)
2. Unit Tests (fast feedback)
3. Full Test Suite with Coverage
4. Package Build & Validation

**When to use:** Before committing to ensure all CI checks will pass.

---

### `test_full.sh` - Complete Test Suite
Runs all tests with coverage reporting.

```bash
./scripts/test_full.sh
```

**What it does:**
- Runs unit, integration, and performance tests
- Generates coverage reports (HTML, XML, terminal)
- Checks coverage threshold (currently 60%)
- Creates detailed coverage analysis

**Outputs:**
- `htmlcov/index.html` - Interactive HTML coverage report
- `coverage.xml` - XML coverage report for CI
- Terminal coverage summary

---

### `test_unit.sh` - Fast Unit Tests
Runs only unit tests for quick feedback during development.

```bash
./scripts/test_unit.sh
```

**What it does:**
- Excludes integration, performance, and slow tests
- Stops on first failure (`-x` flag)
- Shows test durations
- Fast feedback loop for development

**When to use:** During development to quickly validate changes.

---

### `test_coverage.sh` - Detailed Coverage Analysis
Provides comprehensive coverage analysis with detailed reporting.

```bash
./scripts/test_coverage.sh
```

**What it does:**
- Runs complete test suite with detailed coverage
- Generates all coverage formats (HTML, XML, JSON)
- Shows modules with lowest coverage
- Provides actionable coverage improvement suggestions
- Fails if coverage is below 85% (target for Phase 2)

**Outputs:**
- Detailed coverage breakdown by module
- Missing lines analysis
- Coverage improvement recommendations

---

### `lint.sh` - Code Quality Checks
Runs all code quality and style checks.

```bash
./scripts/lint.sh
```

**What it does:**
- **Black**: Code formatting check
- **isort**: Import sorting check
- **flake8**: Syntax errors and critical issues
- **mypy**: Type checking
- **bandit**: Security vulnerability scanning

**Auto-fix suggestions:**
```bash
# Fix formatting issues
black hidden_regime/ tests/ examples/
isort hidden_regime/ tests/ examples/
```

---

### `build_check.sh` - Package Build & Validation
Builds and validates the Python package.

```bash
./scripts/build_check.sh
```

**What it does:**
- Cleans previous build artifacts
- Validates README.md
- Builds source distribution (`.tar.gz`) and wheel (`.whl`)
- Validates package metadata with `twine`
- Tests package installation in isolated environment
- Verifies package imports work correctly

**Outputs:**
- `dist/` directory with built packages
- Package size analysis
- Installation validation

---

## 🎯 Development Workflow

### Quick Development Loop
```bash
# During development - fast feedback
./scripts/test_unit.sh

# Before committing - full validation  
./scripts/ci_local.sh
```

### Debugging CI Failures

If CI fails in GitHub Actions:

1. **Replicate locally:**
   ```bash
   ./scripts/ci_local.sh
   ```

2. **Run specific checks:**
   ```bash
   ./scripts/lint.sh          # Code quality issues
   ./scripts/test_unit.sh      # Fast test feedback
   ./scripts/test_coverage.sh  # Coverage analysis
   ./scripts/build_check.sh    # Package build issues
   ```

3. **Fix issues and re-test:**
   ```bash
   # Fix the issues, then verify
   ./scripts/ci_local.sh
   ```

### Coverage Analysis Workflow

```bash
# Get detailed coverage analysis
./scripts/test_coverage.sh

# Open HTML report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 🔍 Script Details

### Virtual Environment Handling
All scripts automatically detect and activate the virtual environment at `$HOME/hidden-regime-pyenv/bin/activate` if it exists.

### Error Handling
- Scripts use `set -e` to exit immediately on any error
- Colorized output for better readability
- Clear success/failure indicators
- Detailed error reporting

### Windows Compatibility
- CI workflow handles Windows differently (no bash scripts)
- Local development on Windows should use WSL or Git Bash
- PowerShell equivalents could be added if needed

## 📊 Coverage Target Progression

- **Current Target**: 60% (recently reduced from 80%)
- **Phase 2 Target**: 85% (comprehensive coverage plan)
- **Final Target**: 90%+ (production-ready)

Run `./scripts/test_coverage.sh` to see current coverage levels and improvement opportunities.

---

## 🚀 Integration with CI

These scripts are integrated into the GitHub Actions CI pipeline:

- **Linux/macOS**: Uses scripts directly (`./scripts/test_full.sh`)
- **Windows**: Uses equivalent pytest commands inline
- **Debugging**: CI failure notifications include script commands for local reproduction

This ensures that **any CI failure can be reproduced locally** by running the same script that failed in CI.