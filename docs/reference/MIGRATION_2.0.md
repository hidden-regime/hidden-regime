# Migration Guide: v1.x to v2.0

**Last Updated**: 2025-11-23

This guide covers the breaking changes introduced in Hidden Regime v2.0 and how to migrate your code from v1.x.

---

## Overview of v2.0 Changes

Version 2.0 completes the architecture migration to a unified **Interpreter + Signal Generation** pattern, removing all legacy backward compatibility code.

### Key Changes

1. **Removed Legacy Components**
   - `FinancialAnalysis` component (replaced by `FinancialInterpreter`)
   - `AnalysisConfig` and `FinancialAnalysisConfig` (replaced by `InterpreterConfiguration`)
   - `analysis` parameter in Pipeline
   - `analysis_config` parameter in factory methods

2. **New Architecture**
   ```
   Data → Observation → Model → Interpreter → SignalGenerator → Report
   ```

3. **Recommended Approach**
   - Use factory functions instead of direct instantiation
   - Use `create_financial_pipeline()` for most use cases

---

## Breaking Changes

### 1. Removed: `FinancialAnalysis` Component

**Old (v1.x):**
```python
from hidden_regime.analysis.financial import FinancialAnalysis
from hidden_regime.config import FinancialAnalysisConfig

config = FinancialAnalysisConfig(n_states=3)
analysis = FinancialAnalysis(config)
```

**New (v2.0):**
```python
import hidden_regime as hr

# Use factory function (recommended)
pipeline = hr.create_financial_pipeline('AAPL', n_states=3)

# Or create interpreter directly
from hidden_regime.config import InterpreterConfiguration
from hidden_regime.interpreter.financial import FinancialInterpreter

config = InterpreterConfiguration(n_states=3)
interpreter = FinancialInterpreter(config)
```

---

### 2. Removed: `analysis_config` Parameter

**Old (v1.x):**
```python
from hidden_regime.factories import pipeline_factory

pipeline = pipeline_factory.create_pipeline(
    data_config=data_config,
    observation_config=obs_config,
    model_config=model_config,
    analysis_config=analysis_config  # ❌ No longer exists
)
```

**New (v2.0):**
```python
from hidden_regime.factories import pipeline_factory

pipeline = pipeline_factory.create_pipeline(
    data_config=data_config,
    observation_config=obs_config,
    model_config=model_config,
    interpreter_config=interpreter_config,  # ✅ Use this instead
    signal_generator_config=signal_config   # ✅ Optional
)
```

---

### 3. Removed: Pipeline `analysis` Parameter

**Old (v1.x):**
```python
from hidden_regime.pipeline import Pipeline

pipeline = Pipeline(
    data=data_component,
    observation=obs_component,
    model=model_component,
    analysis=analysis_component  # ❌ No longer exists
)
```

**New (v2.0):**
```python
from hidden_regime.pipeline import Pipeline

pipeline = Pipeline(
    data=data_component,
    observation=obs_component,
    model=model_component,
    interpreter=interpreter_component,    # ✅ Use this instead
    signal_generator=signal_component     # ✅ Optional
)
```

---

### 4. Changed: Configuration Imports

**Old (v1.x):**
```python
from hidden_regime.config import (
    AnalysisConfig,              # ❌ Removed
    FinancialAnalysisConfig,     # ❌ Removed
)
```

**New (v2.0):**
```python
from hidden_regime.config import (
    InterpreterConfiguration,        # ✅ Use this
    SignalGenerationConfiguration,   # ✅ Use this
)
```

---

### 5. Changed: Column Names in Output

Some output column names have been standardized:

**Old (v1.x):**
- `regime_name`
- `regime_return`
- `regime_volatility`

**New (v2.0):**
- `regime_label`
- `expected_return`
- `expected_volatility`

**Note**: The MCP tools still support both naming conventions for backward compatibility.

---

## Migration Checklist

### Step 1: Update Imports

Replace any imports of removed components:

```python
# ❌ Remove these
from hidden_regime.analysis.financial import FinancialAnalysis
from hidden_regime.config import AnalysisConfig, FinancialAnalysisConfig

# ✅ Use these instead
from hidden_regime.config import InterpreterConfiguration, SignalGenerationConfiguration
from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.signal_generation.financial import FinancialSignalGenerator
```

### Step 2: Switch to Factory Functions

The easiest migration path is to use factory functions:

```python
import hidden_regime as hr

# Simple regime detection
pipeline = hr.create_financial_pipeline('AAPL', n_states=3)

# Run pipeline
result = pipeline.update()

# Access results
print(result['regime_label'].iloc[-1])  # Note: regime_label, not regime_name
```

### Step 3: Update Pipeline Creation

If you're creating pipelines manually, update the parameters:

```python
from hidden_regime.factories import pipeline_factory
from hidden_regime.config import (
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    InterpreterConfiguration,
    SignalGenerationConfiguration
)

# Create configs
data_config = FinancialDataConfig(ticker='AAPL')
obs_config = FinancialObservationConfig()
model_config = HMMConfig(n_states=3)
interpreter_config = InterpreterConfiguration(n_states=3)
signal_config = SignalGenerationConfiguration()

# Create pipeline with new architecture
pipeline = pipeline_factory.create_pipeline(
    data_config=data_config,
    observation_config=obs_config,
    model_config=model_config,
    interpreter_config=interpreter_config,     # Changed
    signal_generator_config=signal_config      # New (optional)
)
```

### Step 4: Update Column Name References

If your code references specific column names, update them:

```python
# ❌ Old
df['regime_name']
df['regime_return']

# ✅ New
df['regime_label']
df['expected_return']
```

### Step 5: Update Tests

Update any tests that use the old API:

```python
# ❌ Old test
def test_analysis():
    from hidden_regime.analysis.financial import FinancialAnalysis
    analysis = FinancialAnalysis(config)
    # ...

# ✅ New test
def test_interpreter():
    from hidden_regime.interpreter.financial import FinancialInterpreter
    interpreter = FinancialInterpreter(config)
    # ...
```

---

## Example Migration

### Before (v1.x)

```python
from hidden_regime.config import (
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    FinancialAnalysisConfig
)
from hidden_regime.factories import pipeline_factory

# Create configs
data_config = FinancialDataConfig(ticker='SPY')
obs_config = FinancialObservationConfig()
model_config = HMMConfig(n_states=3)
analysis_config = FinancialAnalysisConfig(n_states=3)

# Create pipeline
pipeline = pipeline_factory.create_pipeline(
    data_config=data_config,
    observation_config=obs_config,
    model_config=model_config,
    analysis_config=analysis_config
)

# Run pipeline
result = pipeline.update()

# Access results
print(result['regime_name'].iloc[-1])
```

### After (v2.0)

```python
import hidden_regime as hr

# Simple approach - use factory function
pipeline = hr.create_financial_pipeline('SPY', n_states=3)

# Run pipeline
result = pipeline.update()

# Access results
print(result['regime_label'].iloc[-1])
```

Or if you need more control:

```python
from hidden_regime.config import (
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    InterpreterConfiguration,
    SignalGenerationConfiguration
)
from hidden_regime.factories import pipeline_factory

# Create configs
data_config = FinancialDataConfig(ticker='SPY')
obs_config = FinancialObservationConfig()
model_config = HMMConfig(n_states=3)
interpreter_config = InterpreterConfiguration(n_states=3)
signal_config = SignalGenerationConfiguration()

# Create pipeline
pipeline = pipeline_factory.create_pipeline(
    data_config=data_config,
    observation_config=obs_config,
    model_config=model_config,
    interpreter_config=interpreter_config,
    signal_generator_config=signal_config
)

# Run pipeline
result = pipeline.update()

# Access results
print(result['regime_label'].iloc[-1])
```

---

## Benefits of v2.0

1. **Cleaner Architecture**
   - Single responsibility principle maintained
   - Clear separation: Interpreter adds domain knowledge, SignalGenerator creates trading signals
   - No confusing dual API paths

2. **Simpler Codebase**
   - 66.6KB of legacy code removed
   - No backward compatibility complexity
   - Easier to maintain and extend

3. **Better Performance**
   - Reduced code complexity leads to faster imports
   - Cleaner call paths

4. **Future-Proof**
   - Architecture designed for extensibility
   - Easy to add new interpreter types
   - Easy to add new signal generation strategies

---

## Getting Help

If you encounter issues migrating to v2.0:

1. Check the [CHANGELOG.md](../../CHANGELOG.md) for detailed changes
2. Review the [ARCHITECTURE.md](../../ARCHITECTURE.md) for architecture details
3. Look at the updated examples in `examples/`
4. Open an issue on GitHub if you need assistance

---

## Timeline

- **v1.x**: Legacy architecture with backward compatibility
- **v2.0**: Clean architecture with removed backward compatibility
- **Future**: Continue building on v2.0 foundation

---

**Last Updated**: 2025-11-23
