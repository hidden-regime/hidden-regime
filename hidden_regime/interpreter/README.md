# Interpreter Module

**VERSION 2.0.0** - Central Location for ALL Financial Domain Knowledge

---

## Overview

The Interpreter module is the **exclusive home for all financial domain knowledge** in Hidden Regime. It translates pure mathematical HMM outputs (states 0, 1, 2...) into meaningful financial regime labels (Bull, Bear, Sideways, Crisis) along with comprehensive financial characteristics.

### Core Principle

**Principle #3: ALL financial knowledge in the Interpreter**

The model outputs mathematics. The interpreter adds meaning.

```
┌─────────────────┐         ┌──────────────────────────┐
│   HMM Model     │────────▶│    Interpreter           │
│  (Mathematics)  │         │  (Financial Knowledge)   │
│                 │         │                          │
│ States: 0,1,2   │────────▶│ Labels: Bear,Bull,Side.. │
│ Means: μ₀,μ₁,μ₂ │────────▶│ Returns: +15%, -8%, +2%  │
│ Stds: σ₀,σ₁,σ₂  │────────▶│ Volatility: 25%, 35%, 15%│
│                 │         │ Sharpe: 1.2, -0.8, 0.3   │
│                 │         │ Win Rate: 65%, 35%, 52%  │
└─────────────────┘         └──────────────────────────┘
```

---

## Module Contents

### Core Files

1. **`regime_types.py`** - Financial domain definitions
   - `RegimeType` enum (BULLISH, BEARISH, SIDEWAYS, CRISIS, MIXED)
   - `RegimeProfile` dataclass (complete financial characteristics)
   - `REGIME_TYPE_COLORS` (colorblind-safe color scheme)

2. **`base.py`** - Abstract interpreter interface
   - `BaseInterpreter` class
   - Template method pattern for regime interpretation
   - Common utilities for all interpreters

3. **`financial.py`** - Financial regime interpretation
   - `FinancialInterpreter` implementation
   - Data-driven regime labeling
   - Threshold-based labeling (alternative method)

---

## Regime Types

### RegimeType Enum

Semantic financial classifications of HMM states:

```python
from hidden_regime.interpreter import RegimeType

class RegimeType(Enum):
    BULLISH = "bullish"    # Strong positive returns, moderate volatility
    BEARISH = "bearish"    # Negative returns, often high volatility
    SIDEWAYS = "sideways"  # Low returns, typically low volatility
    CRISIS = "crisis"      # Extreme volatility, negative returns
    MIXED = "mixed"        # Unclear financial characteristics
```

### Visual Representation

```python
from hidden_regime.interpreter import REGIME_TYPE_COLORS

# Colorblind-safe color scheme (ColorBrewer2 diverging)
REGIME_TYPE_COLORS = {
    RegimeType.BULLISH: "#4575b4",   # Blue
    RegimeType.BEARISH: "#d73027",   # Red
    RegimeType.SIDEWAYS: "#fee08b",  # Yellow
    RegimeType.CRISIS: "#a50026",    # Dark Red
    RegimeType.MIXED: "#9970ab",     # Purple
}
```

**Why Colorblind-Safe?**
- Accessible to users with deuteranopia (red-green colorblindness)
- Maintains distinction in grayscale printing
- Follows scientific visualization best practices

---

## Regime Profile

### Complete Financial Characterization

The `RegimeProfile` dataclass contains **ALL financial domain knowledge** about a regime:

```python
from hidden_regime.interpreter import RegimeProfile, RegimeType

profile = RegimeProfile(
    # State identification
    state_id=2,                         # HMM state index
    regime_type=RegimeType.BULLISH,     # Semantic classification
    color="#4575b4",                    # Visualization color

    # Return characteristics
    mean_daily_return=0.0012,           # +0.12% per day
    daily_volatility=0.018,             # 1.8% daily std dev
    annualized_return=0.30,             # ~30% annual
    annualized_volatility=0.286,        # ~29% annual vol

    # Regime behavior
    persistence_days=15.3,              # Lasts ~15 days on average
    regime_strength=0.82,               # Strong regime distinction
    confidence_score=0.91,              # High classification confidence

    # Trading characteristics
    win_rate=0.65,                      # 65% positive return days
    max_drawdown=-0.08,                 # 8% max drawdown in regime
    return_skewness=0.3,                # Slight positive skew
    return_kurtosis=4.2,                # Moderate tail risk

    # Transition behavior
    avg_duration=15.3,                  # Average duration
    transition_volatility=0.025,        # Volatility during transitions

    # Data-driven label
    regime_type_str="Strong Bull"       # Human-readable label
)
```

### Accessing Profile Data

```python
# Get display name
display_name = profile.get_display_name()
# Returns: "Strong Bull" (or falls back to regime_type.value)

# Financial metrics
sharpe = profile.annualized_return / profile.annualized_volatility
# Sharpe ratio: 1.05

sortino = profile.annualized_return / (profile.daily_volatility * np.sqrt(252))
# Sortino ratio (simplified)

calmar = profile.annualized_return / abs(profile.max_drawdown)
# Calmar ratio: 3.75
```

---

## FinancialInterpreter

### Data-Driven Regime Labeling

The `FinancialInterpreter` assigns regime labels based on **actual data characteristics**, not hard-coded rules:

```python
from hidden_regime.interpreter import FinancialInterpreter
from hidden_regime.config import InterpreterConfiguration

# Configure interpreter
config = InterpreterConfiguration(
    n_states=3,
    interpretation_method='data_driven',  # or 'threshold'
    force_regime_labels=None              # None = automatic labeling
)

# Create interpreter
interpreter = FinancialInterpreter(config)

# Interpret model output
interpreted_output = interpreter.update(model_output)
```

### Interpretation Methods

#### 1. Data-Driven Labeling (Recommended)

**Method:** Analyzes emission parameters to classify regimes

```python
config = InterpreterConfiguration(
    interpretation_method='data_driven',
    n_states=3
)

# Classification algorithm:
# 1. Sort states by mean return (low to high)
# 2. Analyze volatility characteristics
# 3. Assign labels based on return+volatility combination:
#    - High volatility + negative returns → "Bear" or "Crisis"
#    - Low returns + low volatility → "Sideways"
#    - Positive returns + moderate vol → "Bull"
```

**Example Classifications:**

| State | Mean Return (Annual) | Volatility (Annual) | Assigned Label |
|-------|---------------------|---------------------|----------------|
| 0 | -15% | 35% | Bear |
| 1 | +2% | 15% | Sideways |
| 2 | +25% | 22% | Bull |

#### 2. Threshold-Based Labeling

**Method:** Uses fixed return thresholds

```python
config = InterpreterConfiguration(
    interpretation_method='threshold',
    n_states=3
)

# Threshold algorithm:
# - Daily return < -0.5%  → "Bear"
# - Daily return > +1.0%  → "Bull"
# - Otherwise            → "Sideways"
```

**When to Use:**
- Need consistent labels across different time periods
- Comparing results with specific regime definitions
- Regulatory or reporting requirements

#### 3. Manual Override (Expert Knowledge)

**Method:** Explicitly specify regime labels

```python
config = InterpreterConfiguration(
    force_regime_labels=['Crisis', 'Bear', 'Sideways', 'Bull'],
    n_states=4
)

# Labels assigned by state index:
# State 0 → "Crisis"
# State 1 → "Bear"
# State 2 → "Sideways"
# State 3 → "Bull"
```

**When to Use:**
- Domain expertise about specific market periods
- Matching historical regime definitions
- Educational demonstrations

---

## Interpreter Configuration

### InterpreterConfiguration

```python
from hidden_regime.config import InterpreterConfiguration

config = InterpreterConfiguration(
    # Required
    n_states=3,                         # Number of HMM states

    # Interpretation method
    interpretation_method='data_driven',  # 'data_driven', 'threshold', or 'manual'

    # Manual override (optional)
    force_regime_labels=None,            # List[str] or None

    # Regime characteristics
    min_regime_duration_days=2,          # Minimum regime duration filter
    regime_persistence_threshold=0.7,    # Persistence threshold (0-1)

    # Visualization
    color_scheme='default',              # 'default', 'sequential', or 'custom'
    custom_colors=None,                  # Dict[str, str] hex colors

    # Advanced
    calculate_regime_profiles=True,      # Compute full financial profiles
    include_transition_analysis=True,    # Analyze regime transitions
)
```

### Factory Methods

```python
# Default financial interpretation
config = InterpreterConfiguration.create_default_financial(n_states=3)

# Conservative interpretation (stricter thresholds)
config = InterpreterConfiguration.create_conservative(n_states=3)

# Aggressive interpretation (more granular regimes)
config = InterpreterConfiguration.create_aggressive(n_states=4)
```

---

## Usage Examples

### Example 1: Basic Interpretation

```python
import hidden_regime as hr

# Create pipeline (interpreter created automatically)
pipeline = hr.create_financial_pipeline('SPY', n_states=3)

# Run analysis
result = pipeline.update()

# Access interpreted outputs
interpreter_output = pipeline.component_outputs['interpreter']

regime_labels = interpreter_output['regime_label']    # Series of regime names
current_regime = interpreter_output['regime_label'].iloc[-1]  # Current regime
confidence = interpreter_output['confidence'].iloc[-1]        # Confidence score

print(f"Current Regime: {current_regime} (Confidence: {confidence:.1%})")
```

### Example 2: Custom Interpretation Method

```python
from hidden_regime.config import (
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    InterpreterConfiguration
)
from hidden_regime.factories import pipeline_factory

# Configure interpreter
interpreter_config = InterpreterConfiguration(
    n_states=4,
    interpretation_method='threshold',  # Use threshold method
    min_regime_duration_days=5,         # Filter short-lived regimes
)

# Create pipeline
pipeline = pipeline_factory.create_pipeline(
    data_config=FinancialDataConfig(ticker='AAPL'),
    observation_config=FinancialObservationConfig.create_default_financial(),
    model_config=HMMConfig(n_states=4),
    interpreter_config=interpreter_config
)

result = pipeline.update()
```

### Example 3: Manual Regime Labels

```python
from hidden_regime.config import InterpreterConfiguration
from hidden_regime.interpreter import FinancialInterpreter

# Specify exact labels (expert knowledge)
config = InterpreterConfiguration(
    n_states=4,
    force_regime_labels=['Deep Bear', 'Bear', 'Neutral', 'Bull']
)

interpreter = FinancialInterpreter(config)

# Interpret model output
interpreted = interpreter.update(model_output)

# Verify labels
labels = interpreted['regime_label'].unique()
# Returns: ['Deep Bear', 'Bear', 'Neutral', 'Bull']
```

### Example 4: Accessing Regime Profiles

```python
from hidden_regime.interpreter import FinancialInterpreter
from hidden_regime.config import InterpreterConfiguration

config = InterpreterConfiguration(
    n_states=3,
    calculate_regime_profiles=True  # Enable full profiles
)

interpreter = FinancialInterpreter(config)
result = interpreter.update(model_output)

# Access regime profiles (if available)
if hasattr(interpreter, '_regime_profiles') and interpreter._regime_profiles:
    for state_id, profile in interpreter._regime_profiles.items():
        print(f"\nState {state_id}: {profile['label']}")
        print(f"  Annual Return: {profile['mean_return']:.1%}")
        print(f"  Annual Volatility: {profile['volatility']:.1%}")
```

### Example 5: Visualization with Colors

```python
import matplotlib.pyplot as plt
from hidden_regime.interpreter import REGIME_TYPE_COLORS, RegimeType

# Get interpreted output
interpreted = interpreter.update(model_output)

# Plot with regime colors
fig, ax = plt.subplots(figsize=(12, 6))

for regime_type in RegimeType:
    mask = interpreted['regime_type'] == regime_type.value
    if mask.any():
        data = interpreted[mask]
        ax.scatter(
            data.index,
            data['price'],
            c=REGIME_TYPE_COLORS[regime_type],
            label=regime_type.value.capitalize(),
            alpha=0.6
        )

ax.legend()
ax.set_title('Price with Regime Classification')
plt.show()
```

---

## Extending the Interpreter

### Creating a Custom Interpreter

For specialized domains (crypto, forex, commodities), create a custom interpreter:

```python
from hidden_regime.interpreter.base import BaseInterpreter
from hidden_regime.config import InterpreterConfiguration
from typing import Dict
import pandas as pd

class CryptoInterpreter(BaseInterpreter):
    """Interpreter for cryptocurrency markets."""

    def _assign_regime_labels(self, model_output: pd.DataFrame) -> Dict[int, str]:
        """Assign crypto-specific regime labels.

        Crypto regimes differ from equities:
        - "FOMO" (Fear Of Missing Out) - rapid price increases
        - "Capitulation" - panic selling
        - "Accumulation" - sideways with increasing volume
        - "Distribution" - sideways with decreasing volume
        """
        # Extract emission parameters
        first_row = model_output.iloc[0]
        emission_means = first_row.get("emission_means")
        emission_stds = first_row.get("emission_stds")

        if emission_means is None:
            return self._default_labels(self.config.n_states)

        labels = {}
        n_states = self.config.n_states

        # Crypto-specific classification
        for state_idx in range(n_states):
            mean_return = emission_means[state_idx]
            volatility = emission_stds[state_idx]

            # Classify based on crypto market dynamics
            if mean_return > 0.05 and volatility > 0.08:
                labels[state_idx] = "FOMO"
            elif mean_return < -0.03 and volatility > 0.10:
                labels[state_idx] = "Capitulation"
            elif abs(mean_return) < 0.01 and volatility < 0.04:
                labels[state_idx] = "Accumulation"
            elif abs(mean_return) < 0.02:
                labels[state_idx] = "Distribution"
            else:
                labels[state_idx] = f"State {state_idx}"

        return labels

    def _get_regime_type(self, regime_label: str) -> str:
        """Map crypto regime labels to types."""
        crypto_types = {
            "FOMO": "bullish",
            "Accumulation": "sideways",
            "Distribution": "sideways",
            "Capitulation": "bearish"
        }
        return crypto_types.get(regime_label, "mixed")

# Usage
crypto_config = InterpreterConfiguration(n_states=4)
crypto_interpreter = CryptoInterpreter(crypto_config)

result = crypto_interpreter.update(model_output)
```

---

## Interpreter Output Format

### DataFrame Schema

The interpreter adds the following columns to model output:

```python
# Input (from Model)
model_output = pd.DataFrame({
    'timestamp': pd.DatetimeIndex,    # Trading dates
    'state': int,                     # HMM state index (0, 1, 2, ...)
    'confidence': float,              # Max probability (0-1)
    'state_probabilities': np.ndarray,# All state probabilities
    'emission_means': np.ndarray,     # Model parameters (optional)
    'emission_stds': np.ndarray,      # Model parameters (optional)
})

# Output (after Interpreter)
interpreted_output = pd.DataFrame({
    # Original columns preserved
    'timestamp': pd.DatetimeIndex,
    'state': int,
    'confidence': float,
    'state_probabilities': np.ndarray,

    # NEW: Interpreter adds financial meaning
    'regime_label': str,              # "Bull", "Bear", "Sideways", etc.
    'regime_type': str,               # "bullish", "bearish", "sideways", "crisis", "mixed"
    'regime_color': str,              # Hex color for visualization
    'regime_strength': float,         # Strength of regime (0-1)

    # Optional: Full regime profiles
    'regime_profile': RegimeProfile,  # Complete financial characteristics
})
```

---

## Key Concepts

### Why Separate Interpreter from Model?

**Without Separation (Bad):**
```python
# Model knows about finance - BAD!
class HiddenMarkovModel:
    def predict(self, observations):
        states = self._viterbi(observations)
        # ❌ Model shouldn't know about "bull" or "bear"
        regimes = ['Bull' if s == 2 else 'Bear' for s in states]
        return regimes
```

**With Separation (Good):**
```python
# Model knows only math - GOOD!
class HiddenMarkovModel:
    def predict(self, observations):
        states = self._viterbi(observations)
        # ✅ Returns integers: [0, 1, 2, 2, 1, 0]
        return states

# Interpreter adds domain knowledge
class FinancialInterpreter:
    def update(self, model_output):
        states = model_output['state']
        # ✅ Translates: 0→Bear, 1→Sideways, 2→Bull
        regimes = self._map_states_to_labels(states)
        return regimes
```

**Benefits:**
1. **Reusability**: Same HMM model works for stocks, crypto, forex, commodities
2. **Testability**: Can test model math independently of financial logic
3. **Clarity**: Clear separation between "what happened" (states) vs "what it means" (regimes)
4. **Extensibility**: Add new domains without touching model code

### Data-Driven vs Heuristic Labeling

**Heuristic (Old Approach):**
```python
# Fixed rules - inflexible
if mean_return > 0.01:
    return "Bull"
elif mean_return < -0.01:
    return "Bear"
else:
    return "Sideways"
```

**Data-Driven (New Approach v2.0.0):**
```python
# Learns from actual data
emission_means = [-0.015, 0.001, 0.012]  # From HMM training
emission_stds = [0.030, 0.015, 0.020]

# State 0: μ=-1.5%, σ=3.0% → "Bear" (negative, high vol)
# State 1: μ=+0.1%, σ=1.5% → "Sideways" (low return, low vol)
# State 2: μ=+1.2%, σ=2.0% → "Bull" (positive, moderate vol)
```

**Why Data-Driven?**
- Adapts to actual market conditions
- No arbitrary thresholds
- Regime definitions emerge from data
- More robust across different assets and time periods

---

## Best Practices

### 1. Let Data Drive Labels

```python
# ✅ GOOD: Use data-driven interpretation
config = InterpreterConfiguration(
    interpretation_method='data_driven',
    n_states=3
)

# ⚠️ USE SPARINGLY: Manual override
config = InterpreterConfiguration(
    force_regime_labels=['Custom Bear', 'Custom Bull', ...],
    n_states=3
)
# Only use manual override for specific research needs
```

### 2. Match States to Complexity

```python
# Simple markets or shorter periods
config = InterpreterConfiguration(n_states=3)  # Bear, Sideways, Bull

# Complex markets or longer periods
config = InterpreterConfiguration(n_states=4)  # Crisis, Bear, Sideways, Bull

# Very complex analysis
config = InterpreterConfiguration(n_states=5)  # Deep Bear, Bear, Neutral, Bull, Strong Bull
```

### 3. Validate Regime Assignments

```python
# After interpretation, validate results
interpreted = interpreter.update(model_output)

# Check label distribution
label_counts = interpreted['regime_label'].value_counts()
print("Regime Distribution:")
print(label_counts)

# Check if any regime is too rare
rare_regimes = label_counts[label_counts < len(interpreted) * 0.05]
if not rare_regimes.empty:
    print(f"Warning: Rare regimes detected: {rare_regimes.index.tolist()}")
    print("Consider reducing n_states or adjusting configuration")
```

### 4. Use Appropriate Color Schemes

```python
# Default: Colorblind-safe (recommended)
config = InterpreterConfiguration(color_scheme='default')

# Custom colors (ensure accessibility)
config = InterpreterConfiguration(
    color_scheme='custom',
    custom_colors={
        'Bear': '#d73027',
        'Sideways': '#fee08b',
        'Bull': '#4575b4'
    }
)
```

---

## Module Structure

```
interpreter/
├── __init__.py           # Public API exports
├── regime_types.py       # RegimeType, RegimeProfile, REGIME_TYPE_COLORS
├── base.py              # BaseInterpreter (abstract class)
└── financial.py          # FinancialInterpreter (implementation)
```

---

## Related Modules

- **[models](../models/README.md)**: Produces the mathematical states that interpreter translates
- **[pipeline](../pipeline/README.md)**: Orchestrates interpreter in the complete flow
- **[analysis](../analysis/README.md)**: Uses interpreter output for performance analysis
- **[visualization](../visualization/README.md)**: Visualizes regimes using interpreter colors
- **[signal_generation](../signal_generation/README.md)**: Generates trading signals from interpreted regimes

---

## Migration from v1.x

### Import Changes

**v1.x (OLD):**
```python
from hidden_regime.financial import RegimeType, RegimeProfile
```

**v2.0.0 (NEW):**
```python
from hidden_regime.interpreter import RegimeType, RegimeProfile
```

### Component Interface Changes

**v1.x (OLD):**
```python
from hidden_regime.pipeline import AnalysisComponent  # DELETED in v2.0.0

class MyAnalyzer(AnalysisComponent):
    def update(self, **kwargs):
        ...
```

**v2.0.0 (NEW):**
```python
from hidden_regime.pipeline import InterpreterComponent

class MyInterpreter(InterpreterComponent):
    def update(self, model_output):
        ...
```

---

## Summary

The Interpreter module is the **sole repository of financial domain knowledge** in Hidden Regime. It ensures:

- ✅ **Clean Separation**: Models stay pure math, interpreters handle finance
- ✅ **Extensibility**: Easy to add new domains (crypto, forex, commodities)
- ✅ **Data-Driven**: Regime labels emerge from actual market behavior
- ✅ **Comprehensive**: Full financial profiles beyond simple labels
- ✅ **Consistent**: Standardized regime types and visualization colors

By centralizing all financial knowledge in this module, we maintain architectural clarity and make the codebase easier to understand, test, and extend.

---

**Version:** 2.0.0
**Last Updated:** 2025-11-16
**Principle:** ALL financial domain knowledge lives here
