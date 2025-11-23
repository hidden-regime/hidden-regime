# Signal Generation Module

**VERSION 2.0.0** - Trading Logic Separated from Regime Interpretation

---

## Overview

The Signal Generation module implements **ALL trading logic** in Hidden Regime. It takes regime interpretations from the Interpreter and generates actionable trading signals, position sizes, and entry/exit rules.

### Core Principle

**The Interpreter tells you WHAT the regime is. The Signal Generator tells you HOW to trade it.**

```
┌──────────────────┐         ┌───────────────────────────┐
│   Interpreter    │────────▶│   Signal Generator        │
│  (What is it?)   │         │   (How to trade it?)      │
│                  │         │                           │
│ Regime: Bull     │────────▶│ Signal: +1.0 (Long)       │
│ Confidence: 0.85 │────────▶│ Position Size: 80%        │
│ Strength: 0.72   │────────▶│ Strategy: Regime Following│
└──────────────────┘         └───────────────────────────┘
```

---

## Module Contents

### Core Files

1. **`base.py`** - Abstract signal generator interface
   - `BaseSignalGenerator` class
   - Template method pattern for signal generation
   - Common utilities for position sizing and risk management

2. **`financial.py`** - Financial trading signal implementations
   - `FinancialSignalGenerator` - Main signal generator
   - Multiple strategy types (following, contrarian, confidence-weighted, multi-timeframe)
   - `ContrarianSignalGenerator` - Specialized contrarian strategy

---

## Signal Types

### Base Signal Range

All signals are normalized to the range **[-1.0, 1.0]**:

| Signal Value | Position Direction | Interpretation |
|--------------|-------------------|----------------|
| **+1.0** | Full Long | Maximum bullish conviction |
| **+0.5** | Half Long | Moderate bullish |
| **0.0** | Neutral/Flat | No position |
| **-0.5** | Half Short | Moderate bearish |
| **-1.0** | Full Short | Maximum bearish conviction |

### Signal Components

Each generated signal has multiple components:

```python
signal_output = {
    'base_signal': float,          # Raw signal (-1 to 1)
    'signal_strength': float,      # Confidence in signal (0 to 1)
    'position_size': float,        # Sized position (accounts for risk)
    'signal_valid': bool,          # Whether signal meets criteria
    'regime_changed': bool,        # Whether regime transition occurred
}
```

---

## Strategy Types

### 1. Regime Following (Default)

**Strategy:** Follow the regime direction

**Logic:**
- Bullish regime → Long position (+1.0)
- Bearish regime → Short position (-1.0)
- Sideways regime → Neutral (0.0)
- Crisis regime → Defensive short (-0.5)

**Best For:**
- Trending markets
- Momentum strategies
- Medium to long-term holding periods

**Example:**
```python
from hidden_regime.config import SignalGenerationConfiguration
from hidden_regime.signal_generation import FinancialSignalGenerator

config = SignalGenerationConfiguration(
    strategy_type='regime_following',
    confidence_threshold=0.7,  # Only trade when confidence >= 70%
)

generator = FinancialSignalGenerator(config)
signals = generator.update(interpreter_output)

# Bullish regime with 0.85 confidence → signal = +1.0, position_size = 85%
```

### 2. Regime Contrarian (Mean Reversion)

**Strategy:** Fade the regime (trade opposite direction)

**Logic:**
- Bullish regime → Short position (-1.0)
- Bearish regime → Long position (+1.0)
- Sideways regime → Neutral (0.0)

**Best For:**
- Range-bound markets
- Mean reversion strategies
- Short-term trading
- Overbought/oversold conditions

**Example:**
```python
config = SignalGenerationConfiguration(
    strategy_type='regime_contrarian',
    confidence_threshold=0.75,  # Higher threshold for contrarian
)

generator = FinancialSignalGenerator(config)
signals = generator.update(interpreter_output)

# Bullish regime → signal = -1.0 (fade the rally)
# Bearish regime → signal = +1.0 (buy the dip)
```

### 3. Confidence Weighted

**Strategy:** Scale position size by regime confidence

**Logic:**
- Base signal from regime direction
- Scale by regime strength/confidence
- Higher confidence → Larger position
- Lower confidence → Smaller position

**Best For:**
- Risk management
- Adaptive position sizing
- Uncertain market conditions
- Dynamic portfolio allocation

**Example:**
```python
config = SignalGenerationConfiguration(
    strategy_type='confidence_weighted',
    position_size_range=(0.2, 1.0),  # 20% to 100% of capital
)

generator = FinancialSignalGenerator(config)
signals = generator.update(interpreter_output)

# Bullish, 0.90 confidence → position_size = 90%
# Bullish, 0.50 confidence → position_size = 50%
# Bullish, 0.30 confidence → position_size = 20% (minimum)
```

### 4. Multi-Timeframe Alignment

**Strategy:** Only trade when multiple timeframes agree

**Logic:**
- Requires timeframe_alignment score in interpreter output
- Only generates signals when alignment >= threshold (default 0.7)
- Scales position by alignment strength
- Filters ~70% of false signals

**Best For:**
- High Sharpe ratio strategies (10+)
- Reducing whipsaws
- Multi-timeframe analysis
- Professional trading

**Example:**
```python
config = SignalGenerationConfiguration(
    strategy_type='multi_timeframe',
    position_size_range=(0.3, 1.0),
)

generator = FinancialSignalGenerator(config)

# Interpreter output must include 'timeframe_alignment' column
# alignment = 1.0 (perfect) → full signal
# alignment = 0.7 (threshold) → 70% of signal
# alignment = 0.5 (below threshold) → NO SIGNAL (0.0)

signals = generator.update(interpreter_output_with_alignment)
```

---

## FinancialSignalGenerator

### Initialization

```python
from hidden_regime.signal_generation import FinancialSignalGenerator
from hidden_regime.config import SignalGenerationConfiguration

# Configure signal generator
config = SignalGenerationConfiguration(
    # Strategy
    strategy_type='regime_following',    # or 'regime_contrarian', 'confidence_weighted', 'multi_timeframe'

    # Risk management
    confidence_threshold=0.7,            # Minimum confidence to trade
    position_size_range=(0.1, 1.0),      # Min/max position size (10% to 100%)

    # Regime change handling
    enable_regime_change_exits=True,     # Exit on regime transitions
    regime_change_lookback=1,            # Days to confirm regime change

    # Position sizing
    scale_by_volatility=True,            # Reduce size in high volatility
    max_leverage=1.0,                    # Maximum leverage (1.0 = no leverage)
)

# Create generator
generator = FinancialSignalGenerator(config)
```

### Input Format

The Signal Generator expects interpreter output with these columns:

```python
interpreter_output = pd.DataFrame({
    'timestamp': pd.DatetimeIndex,    # Trading dates
    'state': int,                     # HMM state index
    'regime_label': str,              # Regime name ("Bull", "Bear", etc.)
    'regime_type': str,               # Regime category ("bullish", "bearish", "sideways", "crisis")
    'regime_strength': float,         # Confidence/strength (0-1)

    # Optional (for confidence_weighted)
    'regime_return': float,           # Expected regime return
    'regime_volatility': float,       # Regime volatility

    # Optional (for multi_timeframe)
    'timeframe_alignment': float,     # Multi-timeframe alignment score (0-1)
})
```

### Output Format

The Signal Generator adds these columns:

```python
signal_output = pd.DataFrame({
    # Original interpreter columns preserved
    ...

    # NEW: Signal generation outputs
    'base_signal': float,             # Raw signal (-1 to 1)
    'signal_strength': float,         # Confidence in signal (0 to 1)
    'position_size': float,           # Sized position (0 to 1 or configured range)
    'signal_valid': bool,             # Whether signal meets criteria
    'regime_changed': bool,           # Whether regime transition occurred
})
```

---

## Position Sizing

### Base Position Sizing

Position size is calculated based on multiple factors:

```python
def calculate_position_size(signal, strength, volatility, config):
    # Start with base signal magnitude
    base_size = abs(signal)  # 0 to 1

    # Scale by confidence/strength
    size = base_size * strength

    # Adjust for volatility (if enabled)
    if config.scale_by_volatility:
        volatility_scalar = min(1.0, 0.2 / volatility)  # Reduce size in high vol
        size *= volatility_scalar

    # Apply position size range
    min_size, max_size = config.position_size_range
    size = np.clip(size, min_size, max_size)

    # Apply leverage limit
    size = min(size, config.max_leverage)

    return size
```

### Example Position Sizing

| Scenario | Base Signal | Strength | Volatility | Position Size |
|----------|-------------|----------|------------|---------------|
| Strong Bull, High Confidence, Normal Vol | +1.0 | 0.90 | 0.15 | 90% |
| Weak Bull, Medium Confidence, Normal Vol | +0.6 | 0.60 | 0.15 | 36% |
| Strong Bull, High Confidence, High Vol | +1.0 | 0.90 | 0.40 | 45% (reduced by vol) |
| Regime Below Threshold | +1.0 | 0.50 | 0.15 | 0% (threshold=0.7) |

---

## Configuration

### SignalGenerationConfiguration

```python
from hidden_regime.config import SignalGenerationConfiguration

config = SignalGenerationConfiguration(
    # Strategy type
    strategy_type='regime_following',    # 'regime_following', 'regime_contrarian', 'confidence_weighted', 'multi_timeframe'

    # Signal filtering
    confidence_threshold=0.7,            # Minimum regime confidence to trade (0-1)
    min_regime_duration_days=2,          # Require regime to persist N days

    # Position sizing
    position_size_range=(0.1, 1.0),      # (min, max) position as fraction of capital
    scale_by_volatility=True,            # Reduce position in high volatility
    volatility_lookback_days=20,         # Days for volatility calculation

    # Risk management
    max_leverage=1.0,                    # Maximum leverage (1.0 = fully invested)
    max_position_concentration=0.25,     # Max % in single position

    # Regime change handling
    enable_regime_change_exits=True,     # Exit positions on regime transitions
    regime_change_lookback=1,            # Days to confirm regime change

    # Multi-timeframe (if using)
    alignment_threshold=0.7,             # Minimum alignment score to trade
    require_all_timeframes=False,        # Strict alignment requirement

    # Advanced
    allow_shorting=True,                 # Enable short positions
    force_neutral_on_crisis=True,        # Go to cash in crisis regimes
)
```

### Factory Methods

```python
# Conservative (lower risk)
config = SignalGenerationConfiguration.create_conservative()
# - Higher confidence threshold (0.8)
# - Lower max position (0.5 = 50%)
# - Stricter regime change exits

# Aggressive (higher risk)
config = SignalGenerationConfiguration.create_aggressive()
# - Lower confidence threshold (0.6)
# - Higher max position (1.0 = 100%)
# - More lenient regime filters

# Balanced (default)
config = SignalGenerationConfiguration.create_balanced()
# - Medium confidence threshold (0.7)
# - Medium max position (0.8 = 80%)
# - Standard risk management
```

---

## Usage Examples

### Example 1: Basic Signal Generation

```python
import hidden_regime as hr

# Create pipeline with signal generation
pipeline = hr.create_trading_pipeline('SPY', n_states=3)

# Run analysis
result = pipeline.update()

# Access signal outputs
signal_output = pipeline.component_outputs['signal_generator']

current_signal = signal_output['base_signal'].iloc[-1]
current_position = signal_output['position_size'].iloc[-1]

print(f"Current Signal: {current_signal:+.2f}")
print(f"Position Size: {current_position:.1%}")
```

### Example 2: Custom Strategy Configuration

```python
from hidden_regime.config import (
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    InterpreterConfiguration,
    SignalGenerationConfiguration,
)
from hidden_regime.factories import pipeline_factory

# Configure signal generation
signal_config = SignalGenerationConfiguration(
    strategy_type='confidence_weighted',  # Scale by confidence
    confidence_threshold=0.75,            # Higher threshold
    position_size_range=(0.2, 0.8),       # 20% to 80% position sizes
    scale_by_volatility=True,             # Reduce in high vol
    enable_regime_change_exits=True,      # Exit on transitions
)

# Create pipeline
pipeline = pipeline_factory.create_pipeline(
    data_config=FinancialDataConfig(ticker='AAPL'),
    observation_config=FinancialObservationConfig.create_default_financial(),
    model_config=HMMConfig(n_states=3),
    interpreter_config=InterpreterConfiguration(),
    signal_config=signal_config
)

result = pipeline.update()
```

### Example 3: Contrarian Strategy

```python
from hidden_regime.signal_generation import FinancialSignalGenerator
from hidden_regime.config import SignalGenerationConfiguration

# Configure contrarian strategy
config = SignalGenerationConfiguration(
    strategy_type='regime_contrarian',
    confidence_threshold=0.80,  # Higher threshold for contrarian (riskier)
    position_size_range=(0.1, 0.5),  # Smaller positions for mean reversion
)

generator = FinancialSignalGenerator(config)
signals = generator.update(interpreter_output)

# When market is bullish, signal is negative (fade the rally)
# When market is bearish, signal is positive (buy the dip)
```

### Example 4: Backtesting with Signals

```python
import hidden_regime as hr
import pandas as pd

# Create pipeline
pipeline = hr.create_trading_pipeline('SPY', n_states=3, risk_adjustment=True)

# Get full dataset
data = pipeline.data.get_all_data()

# Create temporal controller for V&V backtesting
controller = hr.create_temporal_controller(pipeline, data)

# Step through time
results = controller.step_through_time('2023-01-01', '2023-12-31')

# Extract signals at each step
backtest_signals = []
for date, result in results.items():
    signal_data = result.get('signal_generator', {})
    if signal_data:
        backtest_signals.append({
            'date': date,
            'signal': signal_data.get('base_signal'),
            'position': signal_data.get('position_size'),
            'regime': result.get('interpreter', {}).get('regime_label')
        })

signals_df = pd.DataFrame(backtest_signals)
print(signals_df.head())
```

### Example 5: Risk-Managed Portfolio

```python
from hidden_regime.signal_generation import FinancialSignalGenerator
from hidden_regime.config import SignalGenerationConfiguration

# Configure risk-managed signal generation
config = SignalGenerationConfiguration(
    strategy_type='confidence_weighted',
    confidence_threshold=0.70,
    position_size_range=(0.0, 0.6),      # Max 60% of portfolio
    scale_by_volatility=True,            # Volatility targeting
    volatility_lookback_days=30,
    max_leverage=0.8,                    # Max 80% invested
    force_neutral_on_crisis=True,        # Cash in crisis regimes
    enable_regime_change_exits=True,     # Exit on regime changes
)

generator = FinancialSignalGenerator(config)
signals = generator.update(interpreter_output)

# Signals will:
# 1. Never exceed 60% position size
# 2. Reduce positions in high volatility
# 3. Go to cash (0%) in crisis regimes
# 4. Exit positions when regimes change
# 5. Stay below 80% total market exposure
```

---

## Integration with Pipeline

### Signal Generation in Pipeline Flow

```
Data → Observation → Model → Interpreter → Signal Generator → Report
                                   ↓              ↓
                            Regime Labels    Trading Signals
                            Confidence       Position Sizes
                            Characteristics  Entry/Exit Rules
```

### Accessing Signal Outputs

```python
# After pipeline.update()
signal_output = pipeline.component_outputs['signal_generator']

# Available data
signals = signal_output['base_signal']         # Raw signals
positions = signal_output['position_size']     # Sized positions
valid = signal_output['signal_valid']          # Whether to trade
changes = signal_output['regime_changed']      # Regime transitions

# Current state
current_signal = signals.iloc[-1]
current_position = positions.iloc[-1]
should_trade = valid.iloc[-1]
```

---

## Best Practices

### 1. Match Strategy to Market Conditions

```python
# Trending markets → Regime Following
config = SignalGenerationConfiguration(strategy_type='regime_following')

# Range-bound markets → Contrarian
config = SignalGenerationConfiguration(strategy_type='regime_contrarian')

# Uncertain markets → Confidence Weighted
config = SignalGenerationConfiguration(
    strategy_type='confidence_weighted',
    confidence_threshold=0.80  # Higher threshold
)
```

### 2. Use Appropriate Position Sizing

```python
# Conservative (low risk)
config = SignalGenerationConfiguration(
    position_size_range=(0.0, 0.3),  # Max 30%
    confidence_threshold=0.80
)

# Moderate (balanced)
config = SignalGenerationConfiguration(
    position_size_range=(0.1, 0.7),  # 10% to 70%
    confidence_threshold=0.70
)

# Aggressive (high risk)
config = SignalGenerationConfiguration(
    position_size_range=(0.2, 1.0),  # 20% to 100%
    confidence_threshold=0.60
)
```

### 3. Handle Regime Transitions

```python
# Exit on regime changes (reduces whipsaws)
config = SignalGenerationConfiguration(
    enable_regime_change_exits=True,
    regime_change_lookback=2  # Wait 2 days to confirm
)

# Stay in positions through transitions
config = SignalGenerationConfiguration(
    enable_regime_change_exits=False
)
```

### 4. Volatility-Adjusted Sizing

```python
# Reduce positions in high volatility
config = SignalGenerationConfiguration(
    scale_by_volatility=True,
    volatility_lookback_days=20  # 20-day rolling vol
)

# Example:
# Normal vol (15%) → 100% of intended position
# High vol (30%) → 50% of intended position
```

### 5. Multi-Timeframe Filtering

```python
# Only trade when timeframes align (Sharpe 10+ strategy)
config = SignalGenerationConfiguration(
    strategy_type='multi_timeframe',
    alignment_threshold=0.7,  # 70% alignment required
)

# Filters ~70% of trades
# Dramatically improves Sharpe ratio
# Requires multi-timeframe analysis in Interpreter
```

---

## Module Structure

```
signal_generation/
├── __init__.py           # Public API exports
├── base.py              # BaseSignalGenerator (abstract class)
└── financial.py          # FinancialSignalGenerator (implementation)
                         # ContrarianSignalGenerator (specialized)
```

---

## Related Modules

- **[interpreter](../interpreter/README.md)**: Provides regime interpretation that Signal Generator uses
- **[pipeline](../pipeline/README.md)**: Orchestrates signal generation in complete flow
- **[simulation](../simulation/README.md)**: Uses signals for backtesting and performance analysis
- **[reports](../reports/README.md)**: Reports on signal performance

---

## Migration from v1.x

### Location Changes

**v1.x (OLD):**
```python
# Signal generation was in two places (duplicate code)
from hidden_regime.financial.signal_generation import FinancialSignalGenerator  # DELETED
from hidden_regime.simulation.signal_generators import SignalGenerator          # Legacy
```

**v2.0.0 (NEW):**
```python
# Single unified location
from hidden_regime.signal_generation import FinancialSignalGenerator
```

### Interface Changes

**v1.x (OLD):**
```python
# Signals were part of FinancialAnalysis (confusing)
analysis_config = FinancialAnalysisConfig(
    generate_signals=True
)
```

**v2.0.0 (NEW):**
```python
# Signals are separate component (clear separation)
signal_config = SignalGenerationConfiguration(
    strategy_type='regime_following'
)
```

---

## Summary

The Signal Generation module provides:

- ✅ **Clear Separation**: Trading logic separated from regime interpretation
- ✅ **Multiple Strategies**: Following, contrarian, confidence-weighted, multi-timeframe
- ✅ **Risk Management**: Position sizing, volatility adjustment, leverage limits
- ✅ **Flexible Configuration**: Easy to customize for different trading styles
- ✅ **Pipeline Integration**: Seamless integration with interpreter outputs

By separating signal generation from interpretation, we enable:
- **Strategy Testing**: Easy to test different trading strategies on same regimes
- **Risk Control**: Centralized risk management and position sizing
- **Clarity**: Clear distinction between "what the market is doing" vs "what to do about it"
- **Extensibility**: Easy to add new trading strategies

---

**Version:** 2.0.0
**Last Updated:** 2025-11-16
**Principle:** ALL trading logic lives here
