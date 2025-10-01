# Observations Module

The observations module transforms raw data into model-ready features through configurable observation generators, enabling flexible feature engineering for regime detection.

## Overview

In the pipeline architecture, the observation component sits between data loading and model training:

```
Raw Data (OHLCV) → Observation Generation → Model-Ready Features
     ↓                      ↓                        ↓
  Price, Volume      Log Returns, Indicators    Feature Matrix
```

The observation layer provides:
- **Feature transformation**: Convert raw prices to returns, indicators, etc.
- **Feature engineering**: Create regime-relevant signals
- **Extensibility**: Add custom feature generators
- **Configuration**: Control which features to generate

## Core Classes

### BaseObservationGenerator

Abstract base class for all observation generators.

```python
from hidden_regime.observations import BaseObservationGenerator

class CustomObservationGenerator(BaseObservationGenerator):
    def update(self, data: pd.DataFrame) -> pd.DataFrame:
        # Transform data to observations
        observations = data.copy()
        observations['custom_feature'] = self._compute_feature(data)
        return observations

    def plot(self, **kwargs):
        # Visualization logic
        pass
```

### FinancialObservationGenerator

Specialized generator for financial time series with built-in financial features.

```python
from hidden_regime.observations import FinancialObservationGenerator
from hidden_regime.config import FinancialObservationConfig

# Configure feature generation
config = FinancialObservationConfig(
    generators=['log_return', 'volatility', 'rsi'],
    volatility_window=20,
    normalize_features=True
)

# Create generator
obs_gen = FinancialObservationGenerator(config)

# Generate observations
observations = obs_gen.update(price_data)
```

## Built-in Financial Features

### Core Transformations

#### log_return
Natural logarithm of price returns.

```python
config = FinancialObservationConfig(generators=['log_return'])
```

**Formula**: `log(price_t / price_t-1)`

**Use Case**: Primary feature for HMM regime detection

#### return_ratio
Simple percentage returns.

```python
config = FinancialObservationConfig(generators=['return_ratio'])
```

**Formula**: `(price_t - price_t-1) / price_t-1`

#### price_change
Absolute price changes.

```python
config = FinancialObservationConfig(generators=['price_change'])
```

**Formula**: `price_t - price_t-1`

#### average_price
OHLC average price.

```python
config = FinancialObservationConfig(generators=['average_price'])
```

**Formula**: `(open + high + low + close) / 4`

### Volatility Measures

#### volatility
Rolling standard deviation of returns.

```python
config = FinancialObservationConfig(
    generators=['volatility'],
    volatility_window=20  # 20-period window
)
```

**Use Case**: Regime classification (high volatility = bear/crisis)

#### volatility_context
Normalized volatility relative to historical range.

```python
config = FinancialObservationConfig(generators=['volatility_context'])
```

**Formula**: Current volatility / historical max volatility

**Use Case**: Identify unusual volatility periods

### Regime-Relevant Features

#### momentum_strength
Strength of current price momentum.

```python
config = FinancialObservationConfig(generators=['momentum_strength'])
```

**Use Case**: Distinguish bull vs. sideways regimes

#### trend_persistence
How consistently price moves in one direction.

```python
config = FinancialObservationConfig(generators=['trend_persistence'])
```

**Use Case**: Identify trend vs. range-bound regimes

#### directional_consistency
Alignment of recent returns with overall trend.

```python
config = FinancialObservationConfig(generators=['directional_consistency'])
```

**Use Case**: Detect regime transitions

### Technical Indicators

#### rsi
Relative Strength Index (14-period default).

```python
config = FinancialObservationConfig(
    generators=['rsi'],
    rsi_period=14
)
```

**Range**: 0-100
- **Above 70**: Overbought (possible regime change)
- **Below 30**: Oversold (possible regime change)

#### macd
Moving Average Convergence Divergence.

```python
config = FinancialObservationConfig(generators=['macd'])
```

**Components**: MACD line, signal line, histogram

**Use Case**: Trend strength and momentum

#### bollinger_bands
Price bands based on standard deviations.

```python
config = FinancialObservationConfig(
    generators=['bollinger_bands'],
    bollinger_window=20,
    bollinger_std=2.0
)
```

**Components**: Upper band, middle band, lower band

**Use Case**: Volatility and price extreme detection

#### moving_average
Simple moving average.

```python
config = FinancialObservationConfig(
    generators=['moving_average'],
    ma_window=50
)
```

**Use Case**: Trend identification, support/resistance

### Volume Indicators

#### volume_sma
Moving average of trading volume.

```python
config = FinancialObservationConfig(
    generators=['volume_sma'],
    volume_window=20
)
```

**Use Case**: Identify unusual volume periods

#### volume_ratio
Current volume relative to average.

```python
config = FinancialObservationConfig(generators=['volume_ratio'])
```

**Formula**: `current_volume / average_volume`

**Use Case**: Volume spikes indicating regime changes

#### price_volume_trend
Combines price direction with volume.

```python
config = FinancialObservationConfig(generators=['price_volume_trend'])
```

**Use Case**: Confirm price movements with volume

## Configuration

### FinancialObservationConfig

```python
from hidden_regime.config import FinancialObservationConfig

config = FinancialObservationConfig(
    # Feature selection
    generators=['log_return', 'volatility', 'rsi'],

    # Feature engineering parameters
    volatility_window=20,
    rsi_period=14,
    ma_window=50,
    bollinger_window=20,
    bollinger_std=2.0,

    # Normalization
    normalize_features=False,

    # Missing data handling
    handle_missing=True,
    fill_method='forward'
)
```

### Factory Methods

```python
# Default financial configuration
config = FinancialObservationConfig.create_default_financial()
# Includes: log_return, volatility

# Comprehensive configuration
config = FinancialObservationConfig.create_comprehensive()
# Includes: log_return, volatility, rsi, macd, bollinger_bands

# Minimal configuration
config = FinancialObservationConfig(generators=['log_return'])
```

## Usage Examples

### Basic Usage

```python
from hidden_regime.observations import FinancialObservationGenerator
from hidden_regime.config import FinancialObservationConfig
from hidden_regime.data import FinancialDataLoader

# Load data
loader = FinancialDataLoader()
data = loader.load('AAPL', '2023-01-01', '2024-01-01')

# Configure observations
config = FinancialObservationConfig(
    generators=['log_return', 'volatility', 'rsi']
)

# Generate observations
obs_gen = FinancialObservationGenerator(config)
observations = obs_gen.update(data)

print(observations.columns)
# ['log_return', 'volatility', 'rsi']
```

### Pipeline Integration

```python
import hidden_regime as hr

# Observation config is handled internally
pipeline = hr.create_financial_pipeline('SPY', n_states=3)

# Or with custom observation config
from hidden_regime.config import FinancialObservationConfig

obs_config = FinancialObservationConfig(
    generators=['log_return', 'volatility', 'momentum_strength']
)

pipeline = hr.create_financial_pipeline(
    'SPY',
    observation_config_overrides=obs_config.__dict__
)
```

### Custom Features

```python
from hidden_regime.observations import FinancialObservationGenerator
from hidden_regime.config import FinancialObservationConfig

class CustomObservations(FinancialObservationGenerator):
    def _get_builtin_generator(self, name):
        if name == 'my_custom_feature':
            return self._generate_my_feature

        return super()._get_builtin_generator(name)

    def _generate_my_feature(self, data):
        # Custom feature logic
        return data['close'] / data['close'].rolling(10).mean()

# Use custom generator
config = FinancialObservationConfig(
    generators=['log_return', 'my_custom_feature']
)
obs_gen = CustomObservations(config)
```

### Multiple Features

```python
config = FinancialObservationConfig(
    generators=[
        'log_return',
        'volatility',
        'rsi',
        'macd',
        'bollinger_bands',
        'volume_ratio'
    ],
    volatility_window=20,
    rsi_period=14
)

obs_gen = FinancialObservationGenerator(config)
observations = obs_gen.update(data)

# Observations now contains all requested features
print(observations.shape)
# (N, 6+) depending on multi-column features like bollinger_bands
```

## Feature Engineering Best Practices

### 1. Start Simple

```python
# Start with minimal features
config = FinancialObservationConfig(generators=['log_return'])

# Add features incrementally
config = FinancialObservationConfig(
    generators=['log_return', 'volatility']
)
```

### 2. Match Features to Use Case

```python
# For HMM regime detection (recommended)
config = FinancialObservationConfig(
    generators=['log_return', 'volatility']
)

# For trading strategies
config = FinancialObservationConfig(
    generators=['log_return', 'rsi', 'macd']
)

# For research/analysis
config = FinancialObservationConfig.create_comprehensive()
```

### 3. Handle Missing Values

```python
config = FinancialObservationConfig(
    generators=['log_return', 'volatility'],
    handle_missing=True,
    fill_method='forward'  # or 'backward', 'interpolate'
)
```

### 4. Consider Normalization

```python
# Normalize when combining different-scale features
config = FinancialObservationConfig(
    generators=['log_return', 'rsi', 'volume_ratio'],
    normalize_features=True  # Scale to [0, 1]
)
```

## Visualization

```python
# Generate observations
observations = obs_gen.update(data)

# Plot observations
fig = obs_gen.plot(data=data, observations=observations)
plt.show()
```

## Module Structure

```
observations/
├── __init__.py          # Public API exports
├── base.py              # BaseObservationGenerator
└── financial.py         # FinancialObservationGenerator
```

## Related Modules

- **[data](../data/README.md)**: Data loading (input to observations)
- **[models](../models/README.md)**: Models (use observations as input)
- **[config](../config/README.md)**: Configuration classes
- **[pipeline](../pipeline/README.md)**: Pipeline integration

## Key Concepts

### Why Observations?

Rather than feeding raw prices directly to models, we transform them into **observations** that are:

1. **Stationary**: Returns are more stationary than prices
2. **Normalized**: Features on similar scales
3. **Informative**: Features that reveal regime characteristics
4. **Model-appropriate**: Features suited to HMM assumptions

### Feature Selection

For regime detection, prioritize features that vary by regime:

| Regime | Log Return | Volatility | RSI | Volume |
|--------|-----------|------------|-----|--------|
| **Bull** | Positive | Moderate | >50 | Normal |
| **Bear** | Negative | High | <50 | High |
| **Sideways** | Near 0 | Low | ~50 | Low |
| **Crisis** | Very Negative | Extreme | <30 | Extreme |

---

For complete examples using observations, see `examples/` directory in the project root.