# Architectural Compliance Tests

This directory contains tests that enforce the core architectural principles of Hidden Regime.

## Purpose

Unlike functional tests (unit/integration), **architectural tests verify structural integrity** and design patterns. They act as guardrails to prevent architectural degradation over time.

## Test Categories

### 1. Model Purity (`test_model_purity.py`)

**Principle:** Model components should contain ONLY mathematical logic, with zero financial domain knowledge.

**What it tests:**
- No financial terminology in `hidden_regime/models/` code
- HMM API is domain-agnostic (outputs states, not "bull"/"bear")
- Model configuration has no financial parameters

**Why it matters:**
- Enables model reuse across domains (beyond finance)
- Clear separation of concerns
- Concentrates all financial logic in one layer (Interpreter)

**Example violation:**
```python
# BAD - Financial concept in Model
class HiddenMarkovModel:
    def get_bull_bear_states(self):  # ❌ Financial term in model
        ...

# GOOD - Domain-agnostic output
class HiddenMarkovModel:
    def predict_states(self):  # ✅ Generic state prediction
        return states  # [0, 1, 2, ...] - no financial meaning
```

### 2. Factory Pattern (`test_factory_pattern.py`)

**Principle:** Components should be created via factory functions, not direct instantiation.

**What it tests:**
- Examples use `component_factory.create_*()` instead of direct `Component()`
- Factory functions exist and are accessible
- Direct instantiation triggers deprecation warnings
- Documentation demonstrates factory pattern

**Why it matters:**
- Consistent component creation
- Easier to add validation, caching, logging in future
- Users learn correct patterns from examples

**Example violation:**
```python
# BAD - Direct instantiation in example
model = HiddenMarkovModel(config)  # ❌ Anti-pattern

# GOOD - Factory pattern
from hidden_regime.factories import component_factory
model = component_factory.create_model_component(config)  # ✅
```

### 3. Interpreter Separation (`test_interpreter_separation.py`)

**Principle:** Models output mathematical states; Interpreters add financial semantics.

**What it tests:**
- Model outputs contain NO regime labels ("bull", "bear")
- Interpreter outputs contain regime labels and financial metrics
- No financial calculations in Model layer

**Why it matters:**
- State IDs are arbitrary (state 0 might be "bull" or "bear" depending on data)
- Interpretation must be data-driven
- Enables model reuse with different interpretations

**Example:**
```python
# Model output (domain-agnostic)
model_output = {
    'predicted_state': [0, 1, 2, 1, 0, ...],  # Just numbers
    'emission_means': [...],
    'emission_stds': [...]
}

# Interpreter output (financial semantics)
interpreter_output = {
    'regime_labels': ['Bear', 'Sideways', 'Bull', ...],  # Financial meaning
    'confidence': [0.85, 0.92, ...],
    'regime_characteristics': [...]  # Sharpe, win rate, etc.
}
```

## Running Tests

```bash
# Run all architectural tests
pytest tests/architecture/ -v

# Run specific test file
pytest tests/architecture/test_model_purity.py -v

# Run only architecture tests (using marker)
pytest -m architecture -v

# Include in CI pipeline
pytest tests/architecture/ --tb=short
```

## Test Markers

All tests are marked with `@pytest.mark.architecture` to enable selective running:

```python
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "architecture: Architectural compliance tests",
]
```

## Fixing Violations

When an architectural test fails, it means the codebase has deviated from core design principles. **Do not disable the test** - fix the violation.

### Common Fixes

**Model Purity Violation:**
1. Move financial logic from `models/` to `interpreter/`
2. Replace financial terms with generic mathematical terms
3. Update tests to verify domain-agnostic behavior

**Factory Pattern Violation:**
1. Replace `Component(config)` with `component_factory.create_component(config)`
2. Add import: `from hidden_regime.factories import component_factory`
3. Update documentation/examples

**Interpreter Separation Violation:**
1. Ensure Model returns only states and parameters
2. Move regime labeling logic to Interpreter
3. Add data-driven interpretation based on emission parameters

## Adding New Tests

When adding architectural tests:

1. **Focus on structure, not functionality**
   - Good: "Does this module import financial concepts?"
   - Bad: "Does this function return the correct result?"

2. **Make violations actionable**
   - Provide clear error messages
   - Show code examples of correct patterns
   - Link to architecture documentation

3. **Mark tests appropriately**
   ```python
   @pytest.mark.architecture
   def test_new_architectural_constraint():
       ...
   ```

4. **Document the principle**
   - Add to this README
   - Explain WHY the constraint exists
   - Provide examples

## Integration with CI

These tests run in CI to catch architectural violations before merge:

```yaml
# .github/workflows/ci.yml
- name: Run Architectural Tests
  run: pytest tests/architecture/ -v --tb=short
```

Failures in architectural tests should **block merges** until violations are fixed.

## References

- `ARCHITECTURE.md` - Detailed architectural principles
- `ROADMAP.md` - Architectural evolution plan
- `CLAUDE.md` - Development guidelines

---

**Last Updated:** 2025-11-21
