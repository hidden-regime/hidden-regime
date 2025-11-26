# Multivariate Hidden Markov Models: Complete Guide

## Table of Contents

1. [When to Use Multivariate vs Univariate](#when-to-use)
2. [Feature Selection Best Practices](#feature-selection)
3. [Configuration and Setup](#configuration)
4. [Understanding Multivariate Output](#understanding-output)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Topics](#advanced)

---

## When to Use Multivariate vs Univariate {#when-to-use}

### Use Multivariate When:

1. **Multiple complementary signals** - You have features that provide different perspectives on regime
   - Example: `log_return` (direction) + `realized_vol` (magnitude)
   - Why: Covariance between them reveals regime structure (correlation changes with regime)

2. **You want regime-conditional relationships** - How features relate depends on the regime
   - Example: In crisis, returns and volatility are highly correlated (negative)
   - In normal markets, they're less correlated
   - Multivariate model captures this regime-specific structure

3. **Your features have regime-dependent correlation** - The correlation itself is informative
   - Bull markets: Positive correlation between returns and volume
   - Bear markets: Return spikes disconnected from volume
   - Model learns these transitions automatically

4. **You need regime characterization beyond returns** - Single signal is insufficient
   - Example: Distinguish "high volatility bull" from "low volatility bull"
   - Multivariate model captures both return AND volatility dynamics per regime

### DO NOT Use Multivariate When:

1. **Limited data** - Multivariate models need more observations
   - Univariate: ~50-100 obs minimum
   - Multivariate: ~200-300 obs recommended (covariance requires more data)

2. **Features are highly correlated** - Covariance becomes nearly singular
   - Example: `log_return` and `price_change` are nearly identical
   - Model struggles with numerical stability
   - Solution: Choose one OR decorrelate with PCA preprocessing

3. **One feature dominates others** - Information imbalance
   - Example: `log_return` (range 0.001-0.02) + `raw_volume` (range 1M-10M)
   - Model can't learn relationships (before feature standardization)
   - Note: Pipeline applies automatic StandardScaler to fix this

4. **Interpretability is critical** - Univariate is simpler
   - Multivariate: "Regime has eigenvalue_ratio=3.2 with high correlation"
   - Univariate: "Bull regime has 0.8% mean return, 12% volatility"

5. **Real-time prediction** - Latency requirements strict
   - Multivariate inference slightly slower (covariance computation)
   - Rarely significant unless millisecond-critical

---

## Feature Selection Best Practices {#feature-selection}

### Good Feature Pairs

✓ **`log_return` + `realized_vol`** (Recommended Default)
- **Why:** Returns measure direction; volatility measures magnitude
- **Covariance insight:** Crisis regimes show negative correlation (returns drop as vol spikes)
- **Example:**
  ```python
  pipeline = hr.create_multivariate_pipeline(
      'SPY',
      features=['log_return', 'realized_vol'],
      n_states=3
  )
  ```

✓ **`log_return` + `momentum_strength`**
- **Why:** Returns + momentum capture trend and strength
- **Covariance insight:** Trending regimes show correlated momentum; reversals show divergence
- **Use case:** Momentum trading, trend-following strategies

✓ **`log_return` + `directional_consistency`**
- **Why:** Directional consistency measures how often price moves one direction
- **Covariance insight:** Trending regimes show high consistency; choppy regimes show low consistency
- **Use case:** Trend vs range-bound detection

✓ **`volatility` + `volume_ratio`**
- **Why:** Volatility + trading activity (relative to average)
- **Covariance insight:** Liquid regimes have correlated volume and volatility changes
- **Use case:** Liquidity-sensitive trading

### Bad Feature Pairs

✗ **`log_return` + `price_change`**
- **Why:** Nearly identical (just different units)
- **Problem:** Perfect multicollinearity, covariance singular
- **Fix:** Use one or the other, not both

✗ **`return_ratio` + `log_return`**
- **Why:** Highly correlated (different parameterizations of same thing)
- **Problem:** Redundant information, covariance near-singular
- **Fix:** Use one or the other

✗ **`volume_change` + `raw_volume`**
- **Why:** Volume features tend to be highly correlated
- **Problem:** Limited independent information
- **Fix:** Choose one

✗ **`rsi` + `moving_average`**
- **Why:** Technical indicators often measure similar things
- **Problem:** Limited independent information
- **Fix:** Use fundamental features (returns, volatility) + one technical indicator

### Feature Scaling

**Good News:** Pipeline automatically applies `StandardScaler` if features have different scales.

```python
# These work fine - automatic scaling handles scale differences
pipeline = hr.create_multivariate_pipeline(
    'SPY',
    features=['log_return', 'volume_ratio'],  # Scales: 0.001-0.02 vs 0.5-2.0
    n_states=3
)
# Pipeline detects scale mismatch and applies StandardScaler automatically
```

**What to avoid:**

```python
# DON'T manually scale before pipeline - pipeline does it
df['log_return_scaled'] = (df['log_return'] - df['log_return'].mean()) / df['log_return'].std()
df['volume_scaled'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()
pipeline = hr.create_multivariate_pipeline(..., features=['log_return_scaled', 'volume_scaled'])
# Redundant - pipeline already does this internally
```

---

## Configuration and Setup {#configuration}

### Basic Configuration

```python
import hidden_regime as hr
from hidden_regime import ObservationMode, HMMConfig

# Option 1: Using convenience factory (recommended for most users)
pipeline = hr.create_multivariate_pipeline(
    ticker='SPY',
    n_states=3,
    features=['log_return', 'realized_vol']  # Feature validation happens here
)
result = pipeline.update()  # Full pipeline execution
```

### Advanced Configuration

```python
from hidden_regime import HMMConfig, ObservationMode

# Create explicit config for fine control
config = HMMConfig(
    n_states=3,
    observation_mode=ObservationMode.MULTIVARIATE,  # Explicit!
    observed_signals=['log_return', 'realized_vol'],
    initialization_method='kmeans',  # or 'gmm', 'quantile', 'random'
    max_iterations=100,
    tolerance=1e-6,
    random_seed=42  # For reproducibility
)

# Create model directly
model = HiddenMarkovModel(config)

# Fit on data
model.fit(your_dataframe)

# Predict and get covariance matrices
predictions = model.predict(your_dataframe)
# predictions includes:
#  - predicted_state: Most likely state (0, 1, 2, ...)
#  - confidence: Probability of that state
#  - emission_means: (n_states, n_features) array
#  - emission_covs: (n_states, n_features, n_features) array
```

### Initialization Methods

**KMeans (Default):**
```python
config = HMMConfig(
    observation_mode=ObservationMode.MULTIVARIATE,
    observed_signals=['log_return', 'realized_vol'],
    initialization_method='kmeans'
)
# Pros: Data-driven, discovers regime structure from data
# Cons: Requires sklearn, slightly slower
# Use when: General purpose (most cases)
```

**GMM (Gaussian Mixture Model):**
```python
config = HMMConfig(
    observation_mode=ObservationMode.MULTIVARIATE,
    observed_signals=['log_return', 'realized_vol'],
    initialization_method='gmm'
)
# Pros: More flexible than KMeans, captures non-spherical clusters
# Cons: Most computational cost
# Use when: Clusters have different shapes
```

**Quantile (Fast, Fallback):**
```python
config = HMMConfig(
    observation_mode=ObservationMode.MULTIVARIATE,
    observed_signals=['log_return', 'realized_vol'],
    initialization_method='quantile'
)
# Pros: Fast, no external dependencies
# Cons: Less sophisticated initialization
# Use when: Speed critical or sklearn unavailable
```

---

## Understanding Multivariate Output {#understanding-output}

### Output Columns

When you run multivariate pipeline, you get standard interpretation columns PLUS multivariate-specific metrics:

**Standard Interpretation Columns:**
- `regime_label` - Human-readable regime name (Bull, Bear, Sideways, etc.)
- `state` - State index (0, 1, 2, ...)
- `confidence` - Probability of assigned state (0-1)

**Multivariate-Specific Columns:**
- `multivariate_eigenvalue_ratio` - Ratio of largest to smallest eigenvalue
  - Near 1: Isotropic (balanced risk across dimensions)
  - 3-10: Moderate concentration (one direction somewhat dominant)
  - >10: Extreme concentration (one direction dominates)

- `multivariate_pca_explained_variance` - First principal component importance (0-1)
  - 0.5: First PC captures 50% of variance (features equally important)
  - 0.95: First PC captures 95% of variance (one direction dominates)

- `multivariate_avg_feature_correlation` - Average absolute correlation (0-1)
  - 0.0: Features independent
  - 0.5: Moderate relationship
  - 1.0: Perfect correlation

- `multivariate_variance_concentration` - Human-readable eigenvalue interpretation
  - "Isotropic" - Balanced risk
  - "Moderate Concentration" - One direction somewhat dominant
  - "High Concentration" - One direction strongly dominant
  - "Extreme Concentration" - One direction dominates

- `multivariate_correlation_regime` - Human-readable correlation interpretation
  - "Uncorrelated" - Features move independently
  - "Low Correlation" - Weak relationship
  - "Moderate Correlation" - Clear relationship
  - "High Correlation" - Features move together

### Example Output Interpretation

```python
import hidden_regime as hr

pipeline = hr.create_multivariate_pipeline('SPY', n_states=2)
result = pipeline.update()

# Get latest row
latest = result.iloc[-1]

print(f"Current Regime: {latest['regime_label']}")
print(f"Confidence: {latest['confidence']:.1%}")
print(f"Eigenvalue Ratio: {latest['multivariate_eigenvalue_ratio']:.2f}")
print(f"Concentration: {latest['multivariate_variance_concentration']}")
print(f"Feature Correlation: {latest['multivariate_correlation_regime']}")
print(f"PCA Explained Var: {latest['multivariate_pca_explained_variance']:.1%}")

# Interpretation example:
# Current Regime: Bull
# Confidence: 87.5%
# Eigenvalue Ratio: 2.3
# Concentration: Moderate Concentration
# Feature Correlation: Moderate Correlation
# PCA Explained Var: 67.3%
#
# This means: Bull regime with one direction somewhat dominant (2.3 ratio),
# features have moderate relationship (0.45 correlation)
```

---

## Troubleshooting {#troubleshooting}

### Issue: ConfigurationError: Invalid features requested

```python
pipeline = hr.create_multivariate_pipeline(
    'SPY',
    features=['log_return', 'super_cool_feature']
)
# ConfigurationError: Invalid features requested: ['super_cool_feature']
# Valid features are: [...]
```

**Solution:** Check available features
```python
valid_features = [
    'log_return', 'return_ratio', 'average_price', 'price_change',
    'volatility', 'realized_vol',
    'momentum_strength', 'trend_persistence', 'volatility_context', 'directional_consistency',
    'rsi', 'macd', 'bollinger_bands', 'moving_average',
    'volume_sma', 'volume_ratio', 'volume_change', 'price_volume_trend',
]
# Use features from this list
```

### Issue: HMM did not converge after max_iterations

```
UserWarning: HMM did not converge after 100 iterations. Final log-likelihood: -245.3,
Last improvement: 0.05. Consider: Increase max_iterations...
```

**Solutions:**
1. **Increase max_iterations:**
   ```python
   config = HMMConfig(
       observation_mode=ObservationMode.MULTIVARIATE,
       observed_signals=['log_return', 'realized_vol'],
       max_iterations=200  # Increased from 100
   )
   ```

2. **Check for feature scale mismatch:**
   ```python
   # Pipeline should auto-scale, but verify
   data['log_return'].std()  # e.g., 0.015
   data['volume'].std()  # e.g., 1000000
   # Scale difference > 1000x will warn
   ```

3. **Try different initialization:**
   ```python
   config = HMMConfig(
       initialization_method='gmm'  # Try GMM instead of KMeans
   )
   ```

4. **Reduce n_states:**
   ```python
   # Maybe 3 states is too many for your data
   # Try 2 states first
   pipeline = hr.create_multivariate_pipeline('SPY', n_states=2)
   ```

### Issue: Singular covariance matrix or numerical instability

```
LinAlgError: Singular matrix or numerical instability detected
```

**Solutions:**
1. **Check for highly correlated features:**
   ```python
   data[['log_return', 'realized_vol']].corr()
   # If correlation > 0.95, features are redundant - remove one
   ```

2. **Ensure sufficient data:**
   ```python
   # Multivariate needs more observations
   # Rule of thumb: at least 100-200 obs per feature dimension
   len(data)  # Should be > 300 for 2 features
   ```

3. **Try different initialization:**
   ```python
   config = HMMConfig(
       initialization_method='quantile'  # More regularization
   )
   ```

---

## Advanced Topics {#advanced}

### Multivariate vs Univariate Comparison

```python
import hidden_regime as hr

# Create both pipelines
uni_pipeline = hr.create_financial_pipeline('SPY', n_states=2)  # Default univariate
multi_pipeline = hr.create_multivariate_pipeline('SPY', n_states=2)

# Get results
uni_result = uni_pipeline.update()
multi_result = multi_pipeline.update()

# Compare
print("Univariate columns:", uni_result.columns.tolist())
# regime_label, state, confidence, ...

print("Multivariate columns:", multi_result.columns.tolist())
# regime_label, state, confidence, multivariate_eigenvalue_ratio, ..., multivariate_correlation_regime
```

### Regime Characterization from Eigenvalues

The eigenvalue decomposition reveals how variance is distributed:

```python
# From multivariate interpretation
eigenvalue_ratio = result['multivariate_eigenvalue_ratio']
pca_explained = result['multivariate_pca_explained_variance']

# Interpretation:
if eigenvalue_ratio < 2.0:
    # Risk spread across multiple dimensions
    # "Diversified volatility regime"
    # Good for hedging, diverse risk sources

elif eigenvalue_ratio > 5.0:
    # One direction dominates
    # "Concentrated risk regime"
    # Vulnerable to shocks in dominant direction
    # Hedge against that specific risk

# PCA explained variance:
if pca_explained > 0.9:
    # One factor explains 90% of variance
    # Classic trend market

elif pca_explained < 0.5:
    # Multiple factors equally important
    # Complex, multi-directional risk
```

### Custom Feature Engineering

If you want to use features beyond the standard set:

```python
# Add custom features to data BEFORE pipeline
data['my_custom_feature'] = some_calculation(data)

# They won't be auto-generated, but you can include them
# if they're in the dataframe
```

---

## Summary

- **Multivariate is powerful** for capturing regime-specific relationships
- **Choose complementary features** (return + volatility, not return + price_change)
- **Pipeline handles scaling automatically** - don't worry about scale mismatches
- **Monitor eigenvalue ratio and correlation** - they reveal regime character
- **Start with default features** (`log_return` + `realized_vol`) - proven combination
- **Ensure sufficient data** - multivariate needs more observations than univariate
- **Increase max_iterations if convergence warning** - common for complex data

For more information, see:
- Main documentation: `working/README.md`
- Architecture: `working/ARCHITECTURE.md`
- Examples: `working/examples/advanced/multivariate_*.py`
