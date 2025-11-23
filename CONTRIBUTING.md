# Contributing to Hidden Regime

Thank you for your interest in contributing to Hidden Regime! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Development Workflow](#development-workflow)
5. [Testing](#testing)
6. [Code Quality Standards](#code-quality-standards)
7. [Documentation](#documentation)
8. [Submitting Changes](#submitting-changes)
   - [Example Review Checklist](#example-review-checklist)

---

## Code of Conduct

Be respectful, professional, and collaborative. This is an educational project focused on advancing understanding of market regime detection.

---

## Getting Started

### Prerequisites

- **Python 3.9+**
- **Git** for version control
- **Virtual environment** (recommended)

### First-Time Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/hidden-regime.git
   cd hidden-regime/working
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv $HOME/hidden-regime-pyenv
   source $HOME/hidden-regime-pyenv/bin/activate  # Linux/Mac
   # OR
   $HOME/hidden-regime-pyenv/Scripts/activate  # Windows
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e .[dev,mcp,visualization]
   ```

4. **Verify installation:**
   ```bash
   python -c "import hidden_regime; print(hidden_regime.__version__)"
   pytest tests/ -v --tb=short
   ```

---

## Development Setup

### Virtual Environment

**CRITICAL:** Always activate the virtual environment before running Python commands:

```bash
source $HOME/hidden-regime-pyenv/bin/activate
```

Add this to your shell profile (`.bashrc`, `.zshrc`) for convenience:
```bash
alias hr-env='source $HOME/hidden-regime-pyenv/bin/activate'
```

### Project Structure

```
working/
├── hidden_regime/        # Main package source code
│   ├── models/           # HMM algorithms
│   ├── pipeline/         # Orchestration framework
│   ├── interpreter/      # Regime labeling
│   ├── data/             # Data loading
│   ├── config/           # Configuration
│   └── ...
├── tests/                # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test fixtures
├── examples/             # Example scripts
├── scripts/              # Development scripts
└── docs/                 # Documentation
```

### Development Scripts

Located in `scripts/` directory:

- `test_unit.sh` - Fast unit tests only (~10s)
- `test_full.sh` - Complete test suite with coverage (~2min)
- `test_coverage.sh` - Detailed coverage report
- `lint.sh` - Run all code quality checks
- `build_check.sh` - Build and validate package
- `ci_local.sh` - Full CI pipeline simulation

---

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# OR
git checkout -b fix/issue-number-description
```

### 2. Make Your Changes

Follow the architectural principles:

**Core Principles:**

1. **Pipeline is THE interface** - Always use factory functions (`create_financial_pipeline()`, etc.)
2. **Model = Pure mathematics ONLY** - No financial domain knowledge in HMM model
3. **ALL financial knowledge in Interpreter** - Regime labeling, characteristics, statistics
4. **Separation of concerns** - Each component has a single, well-defined responsibility

**Example - Adding a new feature:**

```python
# ❌ DON'T - Direct component instantiation
from hidden_regime.models import HiddenMarkovModel
model = HiddenMarkovModel(config)

# ✅ DO - Use factory functions
import hidden_regime as hr
pipeline = hr.create_financial_pipeline('AAPL', n_states=3)
```

### 3. Write Tests

**Test coverage target: >60% overall**

Create tests in `tests/` directory following the structure:

```python
# tests/unit/test_your_feature.py
import pytest
from hidden_regime import create_financial_pipeline

def test_your_new_feature():
    """Test description following Google docstring style."""
    # Arrange
    pipeline = create_financial_pipeline('SPY', n_states=3)

    # Act
    result = pipeline.update()

    # Assert
    assert 'regime_name' in result.columns
    assert len(result) > 0
```

**Test markers:**
```python
@pytest.mark.unit           # Fast isolated tests
@pytest.mark.integration    # Component interaction tests
@pytest.mark.models         # HMM-specific tests
@pytest.mark.data           # Data loading tests
@pytest.mark.network        # Tests requiring internet
```

### 4. Run Tests Locally

```bash
# Fast unit tests (run frequently during development)
./scripts/test_unit.sh

# Full test suite (run before committing)
./scripts/test_full.sh

# Run specific test file
pytest tests/unit/test_your_feature.py -v

# Run with specific marker
pytest -m "unit" -v
```

### 5. Check Code Quality

```bash
# Run all quality checks
./scripts/lint.sh

# Individual tools:
black hidden_regime/ tests/ examples/        # Format code
isort hidden_regime/ tests/ examples/        # Sort imports
flake8 hidden_regime/                        # Linting
mypy hidden_regime/                          # Type checking
bandit -r hidden_regime/ -c pyproject.toml   # Security scan
```

---

## Testing

### Test Philosophy

**Hidden Regime follows the Test Pyramid:**

- **70% Unit Tests** - Fast, isolated component tests
- **20% Integration Tests** - Component interaction validation
- **10% E2E Tests** - Full pipeline verification

### Test Structure

```
tests/
├── unit/
│   ├── test_models/           # HMM algorithms, initialization
│   ├── test_interpreter/      # Regime labeling, characteristics
│   ├── test_signal_generation/ # Trading signal logic
│   ├── test_data/             # Data loading and validation
│   └── test_config/           # Configuration validation
├── integration/
│   ├── test_pipeline_e2e.py   # Full pipeline tests
│   ├── test_component_pairs.py # Component interactions
│   └── test_temporal.py       # Temporal isolation tests
└── fixtures/
    └── sample_data.py         # Shared test data
```

### Writing Effective Tests

**1. Test actual behavior, not mocked responses:**

```python
# ❌ DON'T - Over-mocking
@patch('hidden_regime.models.HiddenMarkovModel.fit')
def test_model(mock_fit):
    mock_fit.return_value = None
    # This doesn't test anything!

# ✅ DO - Test real behavior
def test_model():
    pipeline = create_financial_pipeline('SPY', n_states=3)
    result = pipeline.update()
    assert result['predicted_state'].dtype == 'int64'
```

**2. Use fixtures for shared setup:**

```python
# tests/conftest.py
@pytest.fixture
def sample_price_data():
    """Provide sample OHLC price data."""
    return pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100),
        'Open': 100 + np.random.randn(100),
        'High': 102 + np.random.randn(100),
        'Low': 98 + np.random.randn(100),
        'Close': 100 + np.random.randn(100),
        'Volume': 1000000
    }).set_index('Date')
```

**3. Test edge cases and boundary conditions:**

```python
def test_empty_data_handling():
    """Ensure model handles empty data gracefully."""
    with pytest.raises(ValueError, match="No data provided"):
        pipeline = create_financial_pipeline('SPY')
        pipeline.model.fit(pd.DataFrame())

def test_extreme_volatility():
    """Test model with extreme market conditions."""
    # Create data with 50% daily swings
    extreme_data = create_extreme_volatility_data()
    pipeline = create_financial_pipeline('TEST')
    result = pipeline.update()  # Should not crash
```

**4. Validate integration points:**

```python
def test_model_to_interpreter_integration():
    """Ensure model output matches interpreter input schema."""
    pipeline = create_financial_pipeline('SPY', n_states=3)
    result = pipeline.update()

    # Model must provide these fields for interpreter
    assert 'predicted_state' in result.columns
    assert 'confidence' in result.columns
    assert 'emission_means' in result.columns
```

### Test Coverage

**Minimum coverage targets:**
- **Overall:** 60%
- **Critical paths (HMM algorithms, data pipeline):** 80%+

**Generate coverage report:**

```bash
./scripts/test_coverage.sh
# Opens htmlcov/index.html with detailed line-by-line coverage
```

**Coverage is configured in `pyproject.toml`:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "unit: Fast isolated unit tests",
    "integration: Integration tests",
    "network: Tests requiring internet connection",
]

[tool.coverage.run]
source = ["hidden_regime"]
omit = [
    "*/tests/*",
    "*/examples/*",
    "*/__pycache__/*",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
```

---

## Code Quality Standards

### Code Formatting

**Black** (line length 88, opinionated formatter):

```bash
black hidden_regime/ tests/ examples/
```

Configuration in `pyproject.toml`:
```toml
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
```

### Import Sorting

**isort** (Black-compatible profile):

```bash
isort hidden_regime/ tests/ examples/
```

Configuration:
```toml
[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

### Linting

**Flake8** (syntax errors, undefined names):

```bash
flake8 hidden_regime/
```

Configuration in `.flake8`:
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist,.eggs
```

### Type Checking

**MyPy** (static type checking):

```bash
mypy hidden_regime/
```

**Type hints required for:**
- All public API functions
- Component interfaces
- Configuration classes

```python
# Example with proper type hints
from typing import Dict, List, Optional
import pandas as pd

def predict_regime(
    data: pd.DataFrame,
    n_states: int = 3,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Predict market regimes from price data.

    Args:
        data: OHLC price data with DatetimeIndex
        n_states: Number of HMM states (2-5)
        config: Optional configuration overrides

    Returns:
        DataFrame with regime predictions and confidence scores

    Raises:
        ValueError: If data is empty or n_states out of range
    """
    ...
```

### Security Scanning

**Bandit** (security issue detection):

```bash
bandit -r hidden_regime/ -c pyproject.toml
```

### Running All Quality Checks

```bash
./scripts/lint.sh
```

This runs:
1. Black formatting check
2. isort import sorting check
3. Flake8 linting
4. MyPy type checking
5. Bandit security scan

**All checks must pass before submitting a PR.**

---

## Documentation

### Code Documentation

**Use Google-style docstrings:**

```python
def calculate_regime_statistics(
    returns: np.ndarray,
    states: np.ndarray,
    n_states: int
) -> Dict[int, Dict[str, float]]:
    """
    Calculate statistical characteristics for each regime.

    Computes mean return, volatility, win rate, max drawdown, and Sharpe ratio
    for each identified market regime.

    Args:
        returns: Array of log returns
        states: Array of regime state assignments (0, 1, 2, ...)
        n_states: Number of distinct states in the model

    Returns:
        Dictionary mapping state index to statistics:
        {
            0: {
                'mean_return': 0.001,
                'volatility': 0.02,
                'win_rate': 0.55,
                'max_drawdown': -0.15,
                'sharpe_ratio': 0.87
            },
            ...
        }

    Raises:
        ValueError: If returns and states have different lengths
        ValueError: If n_states doesn't match unique states in array

    Example:
        >>> returns = np.array([0.01, -0.02, 0.015, ...])
        >>> states = np.array([2, 0, 2, ...])
        >>> stats = calculate_regime_statistics(returns, states, n_states=3)
        >>> print(stats[2]['sharpe_ratio'])
        1.23

    Note:
        Sharpe ratio calculated using risk-free rate of 0.
    """
    ...
```

### Module Documentation

Each module should have a `README.md` explaining:
- Purpose and responsibility
- Key classes/functions
- Usage examples
- Integration points

### Documentation Files

When adding new features, update:
- Module `README.md` files
- Main `README.md` (if user-facing)
- `ARCHITECTURE.md` (if changing design)
- Example scripts in `examples/`

### Examples

Provide working examples for new features:

```python
# examples/advanced/new_feature_example.py
"""
Example: Using New Feature
===========================

This example demonstrates how to use the new feature for regime detection.
"""

import hidden_regime as hr

# Create pipeline with new feature
pipeline = hr.create_financial_pipeline(
    'AAPL',
    n_states=3,
    new_feature_param=True  # Enable new feature
)

# Run analysis
result = pipeline.update()

# Print results
print(f"Feature output: {result['new_column'].iloc[-1]}")
```

---

## Submitting Changes

### Pre-Submission Checklist

Before creating a pull request:

- [ ] All tests pass: `./scripts/test_full.sh`
- [ ] Code quality checks pass: `./scripts/lint.sh`
- [ ] Coverage maintained >42%: `./scripts/test_coverage.sh`
- [ ] New features have tests
- [ ] New features have documentation
- [ ] No breaking changes (or marked clearly if required)
- [ ] Commit messages are clear and descriptive
- [ ] **If adding/modifying examples:** See [Example Review Checklist](#example-review-checklist) below

### Example Review Checklist

**CRITICAL:** All examples must follow these requirements to pass architectural compliance tests.

When adding or modifying example files in `examples/`, ensure:

#### 🏭 Architecture Compliance

- [ ] **Factory Pattern REQUIRED** - Use `hr.create_*_pipeline()` or `component_factory.create_*_component()`
- [ ] **No Direct Instantiation** - Never use `HiddenMarkovModel()`, `FinancialDataLoader()`, etc.
- [ ] **Test Compliance** - Run `pytest tests/architecture/test_factory_pattern.py -v` and ensure it passes
- [ ] **No Deprecation Warnings** - Example runs without `FutureWarning` messages

#### ✅ Code Quality

- [ ] **Runs Successfully** - Test the example from command line: `python examples/path/to/example.py`
- [ ] **Error Handling** - Graceful degradation if network/data unavailable
- [ ] **Output Files** - Creates output in `examples/output/` (not repository root)
- [ ] **Runtime** - Completes in reasonable time (< 5 minutes for most examples)
- [ ] **Dependencies** - Uses only dependencies in `pyproject.toml`

#### 📝 Documentation

- [ ] **Docstring** - Module-level docstring explaining purpose and usage
- [ ] **Usage Example** - Shows how to run: `python examples/.../example.py`
- [ ] **Parameters** - Clearly documented configuration options
- [ ] **Expected Output** - Describes what the example produces

#### 🎯 Best Practices

- [ ] **Follows v2.0 Patterns** - Uses current recommended API, not deprecated code
- [ ] **Helpful Comments** - Explains WHY, not just WHAT
- [ ] **Realistic Example** - Demonstrates practical use case
- [ ] **Appropriate Complexity** - Matches directory level (quickstart/intermediate/advanced)

#### 🔒 Protected Files

- [ ] **Check Blog-Referenced** - Do NOT modify files marked with 🔒 in `examples/README.md`
- [ ] **Case Studies** - `case_study_2008`, `case_study_dotcom_2000`, `case_study_covid_2020` are PROTECTED
- [ ] **Notebooks** - All 7 notebooks in `examples/notebooks/` are PROTECTED

#### Example: Factory Pattern Compliance

```python
# ✅ CORRECT - Uses factory pattern
import hidden_regime as hr

pipeline = hr.create_financial_pipeline('SPY', n_states=3)
result = pipeline.update()

# ❌ WRONG - Direct instantiation (WILL FAIL TESTS)
from hidden_regime.models.hmm import HiddenMarkovModel
model = HiddenMarkovModel(config)  # Architectural test failure!
```

#### Running Compliance Tests

```bash
# Test factory pattern compliance for all examples
pytest tests/architecture/test_factory_pattern.py -v

# Test specific example
pytest tests/architecture/test_factory_pattern.py::test_examples_use_factory_pattern -v

# View violations
python -c "from pathlib import Path; import ast; print([str(p) for p in Path('examples').rglob('*.py')])"
```

### Creating a Pull Request

1. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub** with:
   - Clear title describing the change
   - Description explaining what and why
   - Reference any related issues
   - Screenshots/examples if applicable

3. **PR Template:**
   ```markdown
   ## Description
   Brief description of the changes.

   ## Motivation
   Why is this change needed?

   ## Changes
   - Added X feature
   - Fixed Y bug
   - Updated Z documentation

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Manual testing performed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Tests pass locally
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented)
   ```

4. **Review Process:**
   - Maintainers will review your code
   - Address feedback by pushing new commits
   - Once approved, your PR will be merged

### Git Commit Guidelines

**Commit message format:**

```
type(scope): Short description

Longer description if needed explaining what and why.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style/formatting
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**

```bash
git commit -m "feat(interpreter): Add crisis regime detection

Implements crisis regime detection using extreme negative returns
and high volatility thresholds. Crisis regimes are characterized by
>3 sigma negative returns.

Closes #45"
```

```bash
git commit -m "fix(models): Fix numerical underflow in forward algorithm

Use logsumexp for numerical stability in forward-backward algorithm.
Prevents underflow errors with long time series.

Fixes #67"
```

### Version Bumps

Version managed by `setuptools-scm` based on git tags.

**Semantic Versioning:**
- `vX.0.0` - Major (breaking changes)
- `v0.X.0` - Minor (new features, backward compatible)
- `v0.0.X` - Patch (bug fixes)

---

## Advanced Topics

### Adding a New Component

1. **Create component class** inheriting from appropriate interface:
   ```python
   # hidden_regime/new_component/base.py
   from hidden_regime.pipeline.interfaces import ComponentInterface

   class NewComponent(ComponentInterface):
       def update(self, input_data: pd.DataFrame) -> pd.DataFrame:
           """Implement component logic."""
           ...
   ```

2. **Add configuration class:**
   ```python
   # hidden_regime/config/new_component.py
   from dataclasses import dataclass

   @dataclass
   class NewComponentConfig:
       param1: int = 10
       param2: str = "default"
   ```

3. **Register in factory:**
   ```python
   # hidden_regime/factories/components.py
   factory.register_component('NewComponentConfig', create_new_component)
   ```

4. **Add tests:**
   ```python
   # tests/unit/test_new_component/test_base.py
   def test_new_component_initialization():
       ...
   ```

5. **Document:**
   - Create `hidden_regime/new_component/README.md`
   - Add example to `examples/`
   - Update main `README.md` if user-facing

### Architecture Guidelines

**Key principles:**

1. **Pipeline Architecture** - All components fit into pipeline flow:
   ```
   Data → Observation → Model → Interpreter → Signal Generator → Report
   ```

2. **DataFrame-First Design** - All components exchange pandas DataFrames

3. **Factory Pattern** - Create components via factory functions, not direct instantiation

4. **Separation of Concerns**:
   - **Model**: Pure mathematics, HMM algorithms
   - **Interpreter**: Financial domain knowledge, regime labeling
   - **Signal Generator**: Trading logic
   - **Report**: Output generation

5. **Configuration Over Code** - Use dataclass configs for all parameters

6. **Temporal Isolation** - Prevent lookahead bias in backtesting

---

## Getting Help

- **Documentation**: Module READMEs and `ARCHITECTURE.md`
- **Examples**: See `examples/` directory
- **Issues**: Ask questions by creating a GitHub issue
- **Architecture Questions**: See `ARCHITECTURE.md` for design decisions

---

## Recognition

Contributors will be acknowledged in:
- `README.md` acknowledgments section
- Release notes for their contributions
- Git commit history

Thank you for contributing to Hidden Regime! 🚀

---

**Hidden Regime** - Quantitative market regime detection for systematic trading.

*Last updated: November 2025*
