# Utils Module

The utils module provides common utilities, custom exceptions, and helper functions used throughout the hidden-regime package.

## Overview

Utility functions support core functionality across all modules:

```
Core Modules → Utilities → Consistent Behavior
     ↓             ↓              ↓
  Operations   Exceptions   Error Handling
  Data Trans   Formatting   Regime Mapping
  Analysis     Validation   Conversions
```

## Core Components

### Custom Exceptions

Hierarchical exception system for precise error handling.

```python
from hidden_regime.utils import (
    HiddenRegimeError,
    DataLoadError,
    ConfigurationError,
    HMMTrainingError
)

try:
    # Data loading
    data = loader.load('INVALID_TICKER')
except DataLoadError as e:
    print(f"Data loading failed: {e}")

try:
    # Configuration
    config = HMMConfig(n_states=1)  # Invalid
except ConfigurationError as e:
    print(f"Configuration error: {e}")

try:
    # Model training
    model.fit(insufficient_data)
except HMMTrainingError as e:
    print(f"Training failed: {e}")
```

**Exception Hierarchy:**

```
HiddenRegimeError (base)
├── DataLoadError          # Network errors, invalid tickers, API failures
├── DataQualityError       # Excessive missing values, price anomalies
├── ValidationError        # Invalid date ranges, malformed parameters
├── ConfigurationError     # Invalid settings, incompatible configs
├── HMMTrainingError       # Convergence failures, insufficient data
├── HMMInferenceError      # Prediction failures, invalid observations
├── AnalysisError          # Performance calculation failures
└── ReportGenerationError  # Template errors, visualization failures
```

### State Mapping Utilities

Map HMM states to financial regimes based on actual characteristics.

```python
from hidden_regime.utils import (
    map_states_to_financial_regimes,
    get_regime_characteristics
)
import numpy as np

# HMM emission means (log returns)
emission_means = np.array([-0.005, 0.001, 0.134])

# Map to financial regimes
mapping = map_states_to_financial_regimes(emission_means, n_states=3)
print(mapping)
# {0: 'Bear', 1: 'Sideways', 2: 'Euphoric'}

# Get expected characteristics
bear_char = get_regime_characteristics('Bear')
print(bear_char)
# {
#     'expected_return': -0.002,
#     'expected_volatility': 0.025,
#     'expected_duration': 8.0
# }
```

**Key Functions:**
- `map_states_to_financial_regimes()` - Map states to regimes
- `get_regime_characteristics()` - Expected regime characteristics
- `create_consistent_regime_labels()` - Standard regime labels
- `apply_regime_mapping_to_analysis()` - Apply mapping to DataFrame

### Conversion Utilities

Convert between log returns and percentage changes.

```python
from hidden_regime.utils import (
    percent_change_to_log_return,
    log_return_to_percent_change
)

# Percentage to log return
pct_change = 0.05  # 5% increase
log_return = percent_change_to_log_return(pct_change)
print(f"{pct_change:.2%} → {log_return:.6f}")
# 5.00% → 0.048790

# Log return to percentage
log_ret = 0.048790
pct = log_return_to_percent_change(log_ret)
print(f"{log_ret:.6f} → {pct:.2%}")
# 0.048790 → 5.00%
```

### Formatting Utilities

Consistent display formatting across reports and visualizations.

```python
from hidden_regime.utils.formatting import (
    format_strategy_name,
    format_percentage,
    format_currency
)

# Strategy names
strategy = 'ta_rsi'
formatted = format_strategy_name(strategy)
print(formatted)  # 'Ta Rsi'

# Percentages
value = 0.0523
formatted = format_percentage(value, decimals=2)
print(formatted)  # '5.23%'

# Currency
amount = 123456.78
formatted = format_currency(amount, decimals=2)
print(formatted)  # '$123,456.78'
```

## Regime Mapping

### Threshold-Based Classification

States are classified based on actual return characteristics:

| Daily Return | Classification |
|--------------|----------------|
| < -3% | **Crisis** |
| -3% to -0.5% | **Bear** |
| -0.5% to +1% | **Sideways** |
| +1% to +5% | **Bull** |
| > +5% | **Euphoric** |

```python
# Emission means from HMM
emission_means = np.array([-0.035, -0.008, 0.002, 0.015, 0.062])

# Map to regimes
mapping = map_states_to_financial_regimes(emission_means, n_states=5)
print(mapping)
# {
#     0: 'Crisis',
#     1: 'Bear',
#     2: 'Sideways',
#     3: 'Bull',
#     4: 'Euphoric'
# }
```

### Duplicate Regime Resolution

When multiple states fall in the same category, qualifiers are added:

```python
# Two bull states
emission_means = np.array([-0.008, 0.015, 0.035])

mapping = map_states_to_financial_regimes(emission_means, n_states=3)
print(mapping)
# {
#     0: 'Bear',
#     1: 'Weak Bull',
#     2: 'Strong Bull'
# }
```

**Qualifiers:**
- 2 duplicates: Weak/Strong
- 3 duplicates: Weak/Moderate/Strong
- 4+ duplicates: Level 1/Level 2/Level 3/...

### Regime Characteristics

Default expected characteristics for each regime type:

```python
from hidden_regime.utils import get_regime_characteristics

# Crisis regime
crisis = get_regime_characteristics('Crisis')
print(crisis)
# {
#     'expected_return': -0.005,      # -0.5% daily
#     'expected_volatility': 0.040,   # 4.0% volatility
#     'expected_duration': 5.0         # 5 days average
# }

# Bull regime
bull = get_regime_characteristics('Bull')
print(bull)
# {
#     'expected_return': 0.001,       # 0.1% daily
#     'expected_volatility': 0.018,   # 1.8% volatility
#     'expected_duration': 12.0        # 12 days average
# }
```

**Available Regimes:**
- Crisis: Extreme negative returns, high volatility, brief duration
- Bear: Negative returns, elevated volatility, moderate duration
- Sideways: Near-zero returns, low volatility, long duration
- Bull: Positive returns, moderate volatility, moderate duration
- Euphoric: Strong positive returns, elevated volatility, brief duration

### Consistent Regime Labels

Standard regime labels for different state counts:

```python
from hidden_regime.utils.state_mapping import create_consistent_regime_labels

# 2 states
labels = create_consistent_regime_labels(2)
print(labels)  # ['Bear', 'Bull']

# 3 states
labels = create_consistent_regime_labels(3)
print(labels)  # ['Bear', 'Sideways', 'Bull']

# 4 states
labels = create_consistent_regime_labels(4)
print(labels)  # ['Crisis', 'Bear', 'Sideways', 'Bull']

# 5 states
labels = create_consistent_regime_labels(5)
print(labels)  # ['Crisis', 'Bear', 'Sideways', 'Bull', 'Euphoric']
```

## Exception Handling

### Specific Exception Handling

```python
from hidden_regime.utils import (
    DataLoadError,
    DataQualityError,
    ValidationError,
    HMMTrainingError
)

try:
    # Data loading
    data = loader.load(ticker='AAPL')
except DataLoadError as e:
    # Handle network issues, invalid tickers
    print(f"Failed to load data: {e}")
    data = load_fallback_data()
except DataQualityError as e:
    # Handle missing values, anomalies
    print(f"Data quality issues: {e}")
    data = clean_data(data)

try:
    # Model training
    model.fit(data)
except HMMTrainingError as e:
    # Handle convergence failures
    print(f"Training failed: {e}")
    model = retry_with_different_init()

try:
    # Validation
    validate_date_range(start, end)
except ValidationError as e:
    # Handle invalid parameters
    print(f"Validation error: {e}")
    start, end = use_default_dates()
```

### Base Exception Catch-All

```python
from hidden_regime.utils import HiddenRegimeError

try:
    # Any hidden-regime operation
    result = pipeline.update()
except HiddenRegimeError as e:
    # Catch any package-specific error
    print(f"Hidden Regime error: {e}")
except Exception as e:
    # Catch unexpected errors
    print(f"Unexpected error: {e}")
```

## Usage Examples

### Example 1: Exception Handling in Pipeline

```python
from hidden_regime.utils import (
    DataLoadError,
    HMMTrainingError,
    AnalysisError
)
import hidden_regime as hr

try:
    # Create pipeline
    pipeline = hr.create_financial_pipeline('AAPL', n_states=3)

    # Run analysis
    result = pipeline.update()

except DataLoadError:
    print("Could not load data. Using cached data instead.")
    result = load_cached_results()

except HMMTrainingError:
    print("HMM training failed. Trying with different initialization.")
    pipeline = hr.create_financial_pipeline('AAPL', n_states=3)
    result = pipeline.update()

except AnalysisError:
    print("Analysis failed. Generating minimal report.")
    result = generate_minimal_analysis()
```

### Example 2: State Mapping

```python
from hidden_regime.utils import (
    map_states_to_financial_regimes,
    get_regime_characteristics
)
import numpy as np

# After training HMM
emission_means = np.array([-0.008, 0.002, 0.012])
emission_stds = np.array([0.025, 0.012, 0.018])

# Map states to regimes
mapping = map_states_to_financial_regimes(
    emission_means,
    n_states=3,
    validate=True
)

# Print regime information
for state_id, regime_name in mapping.items():
    print(f"State {state_id}: {regime_name}")
    print(f"  Observed Return: {emission_means[state_id]:.4f}")
    print(f"  Observed Volatility: {emission_stds[state_id]:.4f}")

    # Get expected characteristics
    expected = get_regime_characteristics(regime_name)
    print(f"  Expected Return: {expected['expected_return']:.4f}")
    print(f"  Expected Duration: {expected['expected_duration']:.1f} days")
```

### Example 3: Return Conversions

```python
from hidden_regime.utils import (
    percent_change_to_log_return,
    log_return_to_percent_change
)
import numpy as np
import pandas as pd

# Convert price data to log returns
prices = pd.Series([100, 105, 103, 108])
pct_returns = prices.pct_change().dropna()

# To log returns
log_returns = pct_returns.apply(percent_change_to_log_return)

print("Percentage Returns:")
print(pct_returns.apply(lambda x: f"{x:.2%}"))

print("\nLog Returns:")
print(log_returns.apply(lambda x: f"{x:.6f}"))

# Convert back
reconstructed = log_returns.apply(log_return_to_percent_change)
print("\nReconstructed:")
print(reconstructed.apply(lambda x: f"{x:.2%}"))
```

### Example 4: Formatting for Display

```python
from hidden_regime.utils.formatting import (
    format_strategy_name,
    format_percentage,
    format_currency
)

# Strategy comparison results
results = {
    'ta_rsi': {'return': 0.125, 'capital': 112500},
    'hmm_regime_following': {'return': 0.187, 'capital': 118700},
    'buy_and_hold': {'return': 0.095, 'capital': 109500}
}

# Format for display
print("Strategy Performance")
print("-" * 50)
for strategy, metrics in results.items():
    name = format_strategy_name(strategy)
    ret = format_percentage(metrics['return'])
    cap = format_currency(metrics['capital'])
    print(f"{name:25s} {ret:>8s} {cap:>12s}")

# Output:
# Ta Rsi                    12.50%     $112,500
# Hmm Regime Following      18.70%     $118,700
# Buy And Hold               9.50%     $109,500
```

### Example 5: Validation with Warnings

```python
from hidden_regime.utils.state_mapping import map_states_to_financial_regimes
import numpy as np
import warnings

# Unusual emission means
emission_means = np.array([-0.001, 0.0, 0.001])  # Very similar

# This will generate validation warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    mapping = map_states_to_financial_regimes(
        emission_means,
        n_states=3,
        validate=True
    )

    # Check warnings
    if w:
        print(f"Received {len(w)} validation warnings:")
        for warning in w:
            print(f"  - {warning.message}")

    # Warnings might include:
    # - Small spread between regime returns
    # - Bear regime with positive return
    # - Sideways classification for all states
```

## Best Practices

### 1. Use Specific Exceptions

```python
# Better: Specific exception handling
from hidden_regime.utils import DataLoadError

try:
    data = loader.load(ticker='AAPL')
except DataLoadError:
    # Handle data loading specifically
    data = load_cached_data()

# Rather than: Catch all exceptions
try:
    data = loader.load(ticker='AAPL')
except Exception:
    # Too broad
    pass
```

### 2. Validate Regime Mappings

```python
# Always validate mappings
mapping = map_states_to_financial_regimes(
    emission_means,
    n_states=3,
    validate=True  # Enable validation
)

# Check mapping makes sense
for state_id, regime_name in mapping.items():
    if 'Bear' in regime_name and emission_means[state_id] > 0:
        print(f"Warning: Bear regime has positive return")
```

### 3. Use Return Conversions Correctly

```python
from hidden_regime.utils import (
    percent_change_to_log_return,
    log_return_to_percent_change
)

# For HMM input (use log returns)
log_returns = prices.pct_change().apply(percent_change_to_log_return)
hmm.fit(log_returns)

# For display (convert to percentages)
display_returns = log_returns.apply(log_return_to_percent_change)
print(f"Return: {display_returns.mean():.2%}")
```

### 4. Format Consistently

```python
from hidden_regime.utils.formatting import format_percentage, format_currency

# Use formatting utilities for consistency
def print_results(results):
    for metric, value in results.items():
        if 'return' in metric or 'sharpe' in metric:
            print(f"{metric}: {format_percentage(value)}")
        elif 'capital' in metric or 'pnl' in metric:
            print(f"{metric}: {format_currency(value)}")
        else:
            print(f"{metric}: {value}")
```

### 5. Handle Missing Regime Types

```python
from hidden_regime.utils import get_regime_characteristics

def get_safe_characteristics(regime_name):
    """Get characteristics with fallback."""
    chars = get_regime_characteristics(regime_name)

    # Always returns dict with default values
    # No need to check for KeyError
    return chars

# Safe to use
chars = get_safe_characteristics('Custom Regime')
# Returns defaults if regime unknown
```

## Module Structure

```
utils/
├── __init__.py           # Public API
├── exceptions.py         # Custom exception hierarchy
├── state_mapping.py      # Regime mapping utilities
├── formatting.py         # Display formatting
└── regime_mapping.py     # Additional regime utilities
```

## Related Modules

- **[models](../models/README.md)**: Models (use exceptions, state mapping)
- **[data](../data/README.md)**: Data loaders (raise data exceptions)
- **[analysis](../analysis/README.md)**: Analysis (use regime mapping)
- **[reports](../reports/README.md)**: Reports (use formatting)

## Key Concepts

### Exception Hierarchy

Organized exception system enables precise error handling:

```python
# Specific handling
try:
    model.fit(data)
except HMMTrainingError:
    # Handle training-specific issues
    pass
except HMMInferenceError:
    # Handle prediction-specific issues
    pass

# Or catch category
except HiddenRegimeError:
    # Handle any package error
    pass
```

### Regime Mapping Philosophy

States are mapped based on **actual characteristics**, not **arbitrary numbering**:

```python
# Traditional approach (forced ordering)
# State 0 = Bear, State 1 = Sideways, State 2 = Bull
# Problems: What if State 0 has positive returns?

# Hidden Regime approach (characteristic-based)
emission_means = np.array([0.015, -0.008, 0.002])
mapping = map_states_to_financial_regimes(emission_means, 3)
# {0: 'Bull', 1: 'Bear', 2: 'Sideways'}
# Maps based on actual returns, not state indices
```

### Return Conversions

Log returns are used internally for mathematical properties:

```python
# Why log returns?
# 1. Additive: log(1+r1) + log(1+r2) = log(1+r_total)
# 2. Symmetric: +10% gain = log(1.1), -10% loss = log(0.9)
# 3. Normal approximation: Better for HMM Gaussian emissions

# Always convert for display
internal_log_return = 0.0488
display_pct = log_return_to_percent_change(internal_log_return)
print(f"Return: {display_pct:.2%}")  # "Return: 5.00%"
```

---

For complete examples using utilities, see all example scripts in `examples/` directory in the project root.