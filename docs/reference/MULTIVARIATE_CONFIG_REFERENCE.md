# Multivariate HMM Configuration Reference

## Quick Reference Table

| Parameter | Type | Default | Multivariate Only? | Purpose |
|-----------|------|---------|-------------------|---------|
| `observation_mode` | ObservationMode | UNIVARIATE | No | UNIVARIATE or MULTIVARIATE |
| `observed_signal` | str | "log_return" | No | Single feature for univariate |
| `observed_signals` | List[str] | None | Yes | List of features for multivariate |
| `n_states` | int | 3 | No | Number of regime states (2-5 typical) |
| `initialization_method` | str | "kmeans" | No | How to initialize parameters: kmeans/gmm/quantile |
| `max_iterations` | int | 100 | No | Max EM training iterations |
| `tolerance` | float | 1e-6 | No | Convergence threshold |
| `random_seed` | int | None | No | For reproducibility |

---

## Valid Feature Names for Multivariate

### Core Financial (4)
- `log_return` - Daily log returns
- `return_ratio` - Return ratio (price change relative to previous close)
- `average_price` - OHLC average: (O+H+L+C)/4
- `price_change` - Simple price difference

### Volatility Measures (2)
- `volatility` - Rolling volatility (default 20-day window)
- `realized_vol` - Realized volatility (sum of squared returns)

### Advanced Momentum Metrics (4)
- `momentum_strength` - Strength of recent momentum
- `trend_persistence` - How consistent trend is
- `volatility_context` - Volatility relative to regime
- `directional_consistency` - Consistency of price direction

### Technical Indicators (4)
- `rsi` - Relative Strength Index (14-period default)
- `macd` - MACD signal line
- `bollinger_bands` - Bollinger Bands bandwidth
- `moving_average` - Distance from 20-day MA

### Volume Indicators (4)
- `volume_sma` - Simple moving average of volume
- `volume_ratio` - Current volume vs average
- `volume_change` - Log change in volume
- `price_volume_trend` - Price-volume trend

---

## Configuration Examples

### Minimal (Factory Default)

```python
import hidden_regime as hr

pipeline = hr.create_multivariate_pipeline('SPY')
# Uses: n_states=3, features=['log_return', 'realized_vol']
```

### Basic with Custom States

```python
pipeline = hr.create_multivariate_pipeline('SPY', n_states=2)
# 2-state model (Bull/Bear) with default features
```

### Custom Features

```python
pipeline = hr.create_multivariate_pipeline(
    'SPY',
    features=['log_return', 'momentum_strength'],
    n_states=3
)
```

### Advanced Explicit Configuration

```python
from hidden_regime import HMMConfig, ObservationMode

config = HMMConfig(
    # Observation configuration
    observation_mode=ObservationMode.MULTIVARIATE,
    observed_signals=['log_return', 'realized_vol', 'volume_ratio'],

    # Model configuration
    n_states=4,
    initialization_method='gmm',  # Use GMM instead of KMeans

    # Training configuration
    max_iterations=200,            # More iterations for complex data
    tolerance=1e-8,                # Stricter convergence
    random_seed=42                 # Reproducible results
)

from hidden_regime.models import HiddenMarkovModel
model = HiddenMarkovModel(config)
```

---

## Initialization Methods Comparison

### KMeans (Default)

```python
config = HMMConfig(
    observation_mode=ObservationMode.MULTIVARIATE,
    observed_signals=['log_return', 'realized_vol'],
    initialization_method='kmeans'
)
```

**Pros:**
- Data-driven discovery of regime structure
- Fast and reliable
- Works well for most financial data
- Depends on actual observations, not arbitrary thresholds

**Cons:**
- Requires sklearn
- Slightly slower than quantile
- Assumes spherical clusters (not always true)

**Use when:** General purpose, default choice

---

### GMM (Gaussian Mixture Model)

```python
config = HMMConfig(
    observation_mode=ObservationMode.MULTIVARIATE,
    observed_signals=['log_return', 'realized_vol'],
    initialization_method='gmm'
)
```

**Pros:**
- More flexible than KMeans (non-spherical clusters)
- Captures natural cluster shapes
- Often converges faster

**Cons:**
- More computationally expensive
- May overfit with limited data
- Requires sklearn

**Use when:** Clusters have different shapes, complex regime structure

---

### Quantile (Lightweight)

```python
config = HMMConfig(
    observation_mode=ObservationMode.MULTIVARIATE,
    observed_signals=['log_return', 'realized_vol'],
    initialization_method='quantile'
)
```

**Pros:**
- Fast (no sklearn required)
- No external dependencies
- Good for quick experimentation
- Includes regularization for stability

**Cons:**
- Less sophisticated initialization
- May not discover all regime structure
- Can struggle with complex data

**Use when:** Speed critical, no sklearn available, quick testing

---

## Feature Selection Guidance

### Decision Tree

```
Do your features have regime-dependent correlation?
├─ YES → Use multivariate (e.g., returns + volatility)
└─ NO → Use univariate (e.g., just returns)

Are features complementary?
├─ YES → They measure different things (use both)
├─ NO → They measure the same thing (use one)
└─ UNCERTAIN → Check correlation
            If corr > 0.95 → Use one
            If corr < 0.95 → Use both

Do you have enough data?
├─ YES (>200 obs for 2 features) → Multivariate is OK
└─ NO (<100 obs) → Stick with univariate
```

### Feature Pair Recommendations

**For Trend Detection:**
- ✓ `log_return` + `trend_persistence`
- ✓ `log_return` + `directional_consistency`
- ✓ `momentum_strength` + `volatility_context`

**For Volatility Regimes:**
- ✓ `log_return` + `realized_vol` (Default, recommended)
- ✓ `volatility` + `volume_ratio`
- ✓ `volatility_context` + `price_volume_trend`

**For Mean-Reversion:**
- ✓ `log_return` + `rsi`
- ✓ `price_change` + `bollinger_bands`
- ✓ `momentum_strength` + `moving_average`

**To Avoid:**
- ✗ `log_return` + `price_change` (nearly identical)
- ✗ `return_ratio` + `log_return` (highly correlated)
- ✗ Two volume measures together (redundant)
- ✗ Multiple technical indicators (often correlated)

---

## Performance Tuning

### Default Settings

```python
HMMConfig(
    n_states=3,
    max_iterations=100,
    tolerance=1e-6,
    initialization_method='kmeans'
)
```

**Best for:** Most financial data, 2-5 year time series

### Conservative (Stable)

```python
HMMConfig(
    n_states=3,
    max_iterations=50,
    tolerance=1e-4,
    initialization_method='quantile'
)
```

**Best for:** Real-time trading, fast inference needed

### Aggressive (Complex Data)

```python
HMMConfig(
    n_states=4,
    max_iterations=200,
    tolerance=1e-8,
    initialization_method='gmm'
)
```

**Best for:** Research, complex regime structure, 10+ year data

---

## Multivariate-Specific Parameters

### Observation Mode

```python
from hidden_regime import ObservationMode

# Explicit selection
observation_mode=ObservationMode.UNIVARIATE   # Default, single feature
observation_mode=ObservationMode.MULTIVARIATE # Multiple features
```

**Validation Rules:**
- UNIVARIATE: `observed_signal` required, `observed_signals` must be None
- MULTIVARIATE: `observed_signals` required (1+ signals), `observed_signal` must be default

### Observed Signals

```python
# Univariate
observed_signal='log_return'  # Single feature string

# Multivariate - choose from valid features
observed_signals=[
    'log_return',
    'realized_vol'
]

# Up to 10 features allowed
observed_signals=[
    'log_return',
    'realized_vol',
    'momentum_strength',
    'volume_ratio'
]
```

---

## Output Interpretation

### Standard Output Columns

```python
result.columns  # All DataFrames include:
# - regime_label: Human-readable regime (Bull, Bear, Sideways, etc.)
# - state: Regime index (0, 1, 2, ...)
# - confidence: Probability of regime (0-1)
# - win_rate: % positive return days in regime
# - max_drawdown: Maximum peak-to-trough decline
# - sharpe_ratio: Risk-adjusted return
# (plus many more...)
```

### Multivariate-Specific Output Columns

```python
# When using multivariate pipeline, additional columns:

# Raw metrics
'multivariate_eigenvalue_ratio'         # Largest/smallest eigenvalue
'multivariate_pca_explained_variance'   # First PC importance (0-1)
'multivariate_avg_feature_correlation'  # Average absolute correlation (0-1)
'multivariate_condition_number'         # Covariance matrix condition
'multivariate_covariance_trace'         # Sum of eigenvalues

# Interpreted metrics (human-readable)
'multivariate_variance_concentration'   # Isotropic/Moderate/High/Extreme
'multivariate_correlation_regime'       # Uncorrelated/Low/Moderate/High
```

### Interpreting Eigenvalue Ratio

| Ratio | Regime | Interpretation |
|-------|--------|----------------|
| <1.5  | Isotropic | Risk balanced across dimensions |
| 1.5-3 | Moderate | One direction somewhat dominant |
| 3-10  | High | One direction strongly dominant |
| >10   | Extreme | Risk concentrated in one direction |

---

## Migration from Univariate to Multivariate

### If You Have Existing Univariate Code

```python
# Before (univariate)
pipeline = hr.create_financial_pipeline('SPY', n_states=3)
result = pipeline.update()

# After (multivariate)
pipeline = hr.create_multivariate_pipeline('SPY', n_states=3)
result = pipeline.update()  # Same interface, additional columns
```

**Result:** All existing columns remain, new multivariate columns added

### Backward Compatibility Helper

```python
from hidden_regime import HMMConfig

# For code using old dual-parameter style
config = HMMConfig.from_legacy_config(
    n_states=3,
    observed_signals=['log_return', 'realized_vol']
)
# Automatically detects multivariate mode
```

---

## Validation and Error Handling

### Configuration Validation

```python
# These raise ConfigurationError immediately:

# 1. Setting observed_signal in MULTIVARIATE mode
HMMConfig(
    observation_mode=ObservationMode.MULTIVARIATE,
    observed_signal='log_return',  # ❌ Error: forbidden in multivariate
    observed_signals=['log_return', 'realized_vol']
)

# 2. Missing observed_signals in MULTIVARIATE mode
HMMConfig(
    observation_mode=ObservationMode.MULTIVARIATE,
    observed_signals=None  # ❌ Error: required in multivariate
)

# 3. Too many features
HMMConfig(
    observation_mode=ObservationMode.MULTIVARIATE,
    observed_signals=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']  # ❌ >10
)
```

### Factory Validation

```python
# Feature validation happens at factory time
try:
    pipeline = hr.create_multivariate_pipeline(
        'SPY',
        features=['log_return', 'nonexistent_feature']
    )
except ConfigurationError as e:
    print(e)  # Clear message showing valid features
```

---

## Performance Considerations

### Data Requirements

| Model | Min Observations | Recommended |
|-------|-----------------|-------------|
| Univariate (1 feature) | 50 | 100+ |
| Multivariate (2 features) | 100 | 200+ |
| Multivariate (3 features) | 150 | 300+ |
| Multivariate (4+ features) | 200+ | 500+ |

### Computation Time Estimates

| Operation | Time |
|-----------|------|
| Univariate training (1yr data, 3 states) | <100ms |
| Multivariate training (1yr data, 3 states, 2 features) | 200-400ms |
| Multivariate training (1yr data, 4 states, 3 features) | 400-800ms |
| Multivariate prediction (250 obs) | 10-50ms |
| Multivariate interpretation | 50-100ms |

### Memory Usage

| Model | Memory |
|-------|--------|
| Univariate parameters | ~1KB |
| Multivariate (2 features) | ~2KB |
| Multivariate (4 features) | ~5KB |
| Full pipeline with history | 1-10MB |

---

## Summary

- **Multivariate configuration is explicit** - `observation_mode` makes intent clear
- **Feature validation catches errors early** - Invalid features rejected at factory
- **Pipeline handles scaling automatically** - No need to preprocess
- **Output includes eigenvalue interpretation** - Financial meaning of covariance structure
- **Performance tuning available** - Initialization and iteration methods
- **Fully backward compatible** - Existing univariate code unchanged

For more information, see:
- User Guide: `docs/guides/MULTIVARIATE_HMM_GUIDE.md`
- Examples: `examples/advanced/multivariate_*.py`
- Main Documentation: `README.md`
