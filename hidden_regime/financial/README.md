# Financial Module

The financial module provides specialized components that understand financial markets natively, offering domain-specific regime characterization and analysis beyond generic pipeline components.

## Overview

While the core pipeline modules (data, models, analysis) are generic, the financial module provides **financial-first** implementations optimized for trading and market analysis:

```
Generic Pipeline → Financial Specialization → Trading Insights
       ↓                    ↓                        ↓
   HMM States       Financial Regimes          Position Sizing
   Probabilities    Market Conditions          Risk Assessment
```

## Core Components

### FinancialRegimeCharacterizer

Characterizes HMM states as financial market regimes with domain expertise.

```python
from hidden_regime.financial import FinancialRegimeCharacterizer, RegimeProfile

# Create characterizer
characterizer = FinancialRegimeCharacterizer(n_regimes=3)

# Characterize regime
profile = characterizer.characterize_regime(
    mean_return=0.0015,
    volatility=0.020,
    duration_stats={'avg': 18, 'median': 15}
)

print(f"Regime Type: {profile.regime_type}")  # e.g., "Bull Market"
print(f"Strength: {profile.strength}")         # e.g., "Moderate"
print(f"Confidence: {profile.confidence_score}")  # 0.0-1.0
print(f"Trading Implication: {profile.trading_implication}")
```

**RegimeProfile Attributes:**
- `regime_type`: Bull, Bear, Sideways, Crisis
- `strength`: Weak, Moderate, Strong, Extreme
- `confidence_score`: Model confidence (0-1)
- `risk_level`: Low, Medium, High, Extreme
- `trading_implication`: Suggested position direction
- `typical_duration`: Expected regime length
- `volatility_regime`: Low, Normal, High, Extreme

### FinancialRegimeAnalysis

High-level financial analysis orchestrator.

```python
from hidden_regime.financial import FinancialRegimeAnalysis, FinancialRegimeConfig

# Configure
config = FinancialRegimeConfig.create_comprehensive_analysis(
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    n_regimes=3
)

# Run analysis
analyzer = FinancialRegimeAnalysis(config)
results = analyzer.run_complete_analysis()

# Access results
print(f"Current Regime: {results.current_regime_info.regime_type}")
print(f"Trading Signal: {results.trading_signal}")

# Regime profiles
for regime_id, profile in results.regime_profiles.items():
    print(f"Regime {regime_id}: {profile.regime_type}")
    print(f"  Return: {profile.annualized_return:.1%}")
    print(f"  Risk: {profile.risk_level}")
```

### FinancialRegimeConfig

Configuration for financial-specific analysis.

```python
from hidden_regime.financial import FinancialRegimeConfig

config = FinancialRegimeConfig(
    # Data settings
    ticker='SPY',
    start_date='2023-01-01',
    end_date='2024-01-01',

    # Model settings
    n_regimes=3,
    training_days=252,

    # Analysis settings
    include_technical_indicators=True,
    generate_trading_signals=True,

    # Simulation
    initial_capital=100000,
    risk_tolerance='moderate'  # conservative, moderate, aggressive
)
```

**Factory Methods:**
```python
# Comprehensive analysis
config = FinancialRegimeConfig.create_comprehensive_analysis(
    ticker='AAPL',
    n_regimes=3
)

# Quick analysis
config = FinancialRegimeConfig.create_quick_analysis(ticker='SPY')

# Trading focus
config = FinancialRegimeConfig.create_trading_focused(
    ticker='QQQ',
    risk_tolerance='aggressive'
)
```

## Regime Characterization

### Regime Types

The characterizer maps HMM states to financial regimes:

| Mean Return | Volatility | Classification | Trading Implication |
|-------------|------------|----------------|---------------------|
| > +1.0% | Low-Moderate | **Strong Bull** | Maximum long exposure |
| +0.3% to +1.0% | Moderate | **Bull Market** | Long bias |
| -0.3% to +0.3% | Low | **Sideways** | Neutral, range trading |
| -1.0% to -0.3% | High | **Bear Market** | Defensive, short bias |
| < -1.0% | Very High | **Crisis** | Cash, hedging |

### Risk Levels

Based on volatility and return distribution:

- **Low**: Volatility < 1.5%, stable returns
- **Medium**: Volatility 1.5-2.5%, normal range
- **High**: Volatility 2.5-4.0%, elevated risk
- **Extreme**: Volatility > 4.0%, crisis conditions

### Trading Implications

Characterizer provides actionable guidance:

```python
profile = characterizer.characterize_regime(...)

if profile.trading_implication == 'LONG':
    position_size = 1.0 if profile.strength == 'Strong' else 0.5
elif profile.trading_implication == 'SHORT':
    position_size = -0.5 if profile.strength == 'Strong' else -0.2
else:  # NEUTRAL
    position_size = 0.2  # Small long bias
```

## Signal Generation

### Financial Signal Generator

Generate trading signals from regime profiles.

```python
from hidden_regime.financial.signal_generation import FinancialSignalGenerator

generator = FinancialSignalGenerator(
    risk_tolerance='moderate',
    max_position_size=1.0
)

signal = generator.generate_signal(
    regime_profile=current_regime_profile,
    market_context=market_data
)

print(f"Position: {signal.position_size:.1%}")
print(f"Action: {signal.action}")  # BUY, SELL, HOLD
print(f"Confidence: {signal.confidence:.2f}")
print(f"Stop Loss: {signal.stop_loss:.4f}")
print(f"Take Profit: {signal.take_profit:.4f}")
```

## Usage Examples

### Example 1: Characterize Regime

```python
from hidden_regime.financial import FinancialRegimeCharacterizer

characterizer = FinancialRegimeCharacterizer(n_regimes=3)

# Characterize based on statistics
profile = characterizer.characterize_regime(
    mean_return=0.0012,      # 0.12% daily
    volatility=0.018,        # 1.8% volatility
    duration_stats={'avg': 20, 'median': 18}
)

print(f"{profile.regime_type} ({profile.strength})")
print(f"Risk: {profile.risk_level}")
print(f"Suggested Action: {profile.trading_implication}")
```

### Example 2: Complete Financial Analysis

```python
from hidden_regime.financial import (
    FinancialRegimeAnalysis,
    FinancialRegimeConfig
)

# Configure
config = FinancialRegimeConfig.create_comprehensive_analysis(
    ticker='AAPL',
    n_regimes=4
)

# Run analysis
analyzer = FinancialRegimeAnalysis(config)
results = analyzer.run_complete_analysis()

# Use results
if results.analysis_success:
    print(f"Current: {results.current_regime_info.regime_type}")

    # View all regime profiles
    for rid, profile in results.regime_profiles.items():
        print(f"Regime {rid}:")
        print(f"  Type: {profile.regime_type}")
        print(f"  Return: {profile.annualized_return:.1%}")
        print(f"  Sharpe: {profile.sharpe_ratio:.2f}")
```

### Example 3: Trading Signal Generation

```python
from hidden_regime.financial import FinancialRegimeCharacterizer
from hidden_regime.financial.signal_generation import FinancialSignalGenerator

# Characterize current regime
characterizer = FinancialRegimeCharacterizer(n_regimes=3)
current_profile = characterizer.characterize_regime(
    mean_return=current_stats['mean'],
    volatility=current_stats['vol'],
    duration_stats=current_stats['duration']
)

# Generate signal
generator = FinancialSignalGenerator(risk_tolerance='moderate')
signal = generator.generate_signal(
    regime_profile=current_profile,
    market_context={'price': 150.0, 'volume': 1000000}
)

# Execute based on signal
if signal.action == 'BUY':
    place_order(symbol='AAPL', quantity=signal.position_size * 100)
```

## Best Practices

### 1. Use Financial Characterization

```python
# Better: Use financial characterization
from hidden_regime.financial import FinancialRegimeCharacterizer

characterizer = FinancialRegimeCharacterizer(n_regimes=3)
profile = characterizer.characterize_regime(...)

# Rather than: Generic state interpretation
# generic_label = f"State {state_id}"
```

### 2. Consider Risk Tolerance

```python
# Adjust signal generation to risk profile
config = FinancialRegimeConfig(
    ticker='SPY',
    risk_tolerance='conservative'  # Reduces position sizes
)
```

### 3. Use Complete Analysis

```python
# Comprehensive analysis includes characterization
config = FinancialRegimeConfig.create_comprehensive_analysis(...)
analyzer = FinancialRegimeAnalysis(config)
results = analyzer.run_complete_analysis()
```

## Module Structure

```
financial/
├── __init__.py                # Public API
├── regime_characterizer.py    # FinancialRegimeCharacterizer
├── analysis.py                # FinancialRegimeAnalysis
├── config.py                  # FinancialRegimeConfig
└── signal_generation.py       # Signal generation utilities
```

## Related Modules

- **[analysis](../analysis/README.md)**: Generic analysis (financial builds on this)
- **[models](../models/README.md)**: HMM models (provide raw states)
- **[simulation](../simulation/README.md)**: Trading simulation (uses signals)
- **[data](../data/README.md)**: Financial data loading

## Key Concepts

### Financial-First Design

Rather than treating markets as generic data:

**Generic Approach:**
```python
# State 0: mean=-0.01, vol=0.025
# State 1: mean=0.0, vol=0.015
# State 2: mean=0.01, vol=0.020
```

**Financial-First Approach:**
```python
# Bear Market: Declining prices, high uncertainty
# Sideways: Range-bound, low volatility
# Bull Market: Rising prices, moderate volatility
```

The financial module provides the translation layer that converts statistical properties into actionable market intelligence.

---

For complete examples using financial components, see `examples/case_study_comprehensive.py` and `examples/financial_case_study.py`.