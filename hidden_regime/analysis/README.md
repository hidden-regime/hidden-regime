# Analysis Module

The analysis module interprets model outputs and computes performance metrics, providing financial domain knowledge and actionable insights from regime detection results.

## Overview

The analysis layer sits between model outputs and reporting, transforming raw regime predictions into meaningful financial metrics:

```
Model Output (States) → Analysis → Financial Insights
        ↓                  ↓              ↓
   HMM States        Regime Stats    Trading Signals
   Probabilities     Performance     Risk Metrics
```

## Core Components

### FinancialAnalysis

Main analysis component for interpreting HMM regime predictions in financial context.

```python
from hidden_regime.analysis import FinancialAnalysis
from hidden_regime.config import FinancialAnalysisConfig

# Configure analysis
config = FinancialAnalysisConfig(
    compute_regime_stats=True,
    indicator_comparisons=True,
    performance_metrics=True
)

# Create analyzer
analyzer = FinancialAnalysis(config)

# Analyze model output
result = analyzer.update(
    model_output=hmm_predictions,
    raw_data=price_data
)

print(result)  # Human-readable summary
```

**Key Features:**
- Regime statistics (mean return, volatility, duration)
- Current regime identification with confidence
- Regime transition detection
- Performance metrics by regime
- Optional technical indicator comparison

### RegimePerformanceAnalyzer

Computes performance metrics specific to each regime.

```python
from hidden_regime.analysis import RegimePerformanceAnalyzer

analyzer = RegimePerformanceAnalyzer()

metrics = analyzer.compute_regime_metrics(
    returns=log_returns,
    regimes=regime_labels,
    dates=dates
)

# Access metrics by regime
for regime_id, stats in metrics.items():
    print(f"{regime_id}:")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {stats['win_rate']:.1%}")
    print(f"  Avg Return: {stats['avg_return']:.4f}")
```

**Computed Metrics:**
- Sharpe ratio
- Sortino ratio
- Win rate
- Average return
- Maximum drawdown
- Volatility
- Return skewness/kurtosis

### IndicatorPerformanceComparator

Compares HMM regime detection with technical indicators.

```python
from hidden_regime.analysis import IndicatorPerformanceComparator

comparator = IndicatorPerformanceComparator()

comparison = comparator.compare_methods(
    hmm_regimes=hmm_states,
    indicator_signals={
        'RSI': rsi_signals,
        'MACD': macd_signals,
        'SMA': sma_signals
    },
    returns=log_returns
)

# View comparison
print(f"HMM Sharpe: {comparison['hmm_sharpe']:.2f}")
print(f"RSI Sharpe: {comparison['rsi_sharpe']:.2f}")
print(f"Agreement: {comparison['hmm_rsi_agreement']:.1%}")
```

**Comparison Metrics:**
- Performance comparison (returns, Sharpe, etc.)
- Signal agreement/disagreement
- Timing analysis (lead/lag)
- False positive/negative rates

## Analysis Configuration

### FinancialAnalysisConfig

```python
from hidden_regime.config import FinancialAnalysisConfig

config = FinancialAnalysisConfig(
    # Core analysis
    compute_regime_stats=True,
    performance_metrics=True,

    # Technical indicators
    indicator_comparisons=True,
    indicators_to_compare=['RSI', 'MACD', 'SMA_Crossover'],

    # Regime interpretation
    n_states=3,  # Number of regimes
    regime_labels=['Bear', 'Sideways', 'Bull'],

    # Performance calculation
    risk_free_rate=0.02,  # Annual risk-free rate
    trading_days_per_year=252
)
```

### Factory Methods

```python
# Comprehensive analysis (all features enabled)
config = FinancialAnalysisConfig.create_comprehensive_financial()

# Basic analysis (regime stats only)
config = FinancialAnalysisConfig.create_basic()

# Trading-focused (performance metrics emphasized)
config = FinancialAnalysisConfig.create_trading_focused()
```

## Analysis Features

### Regime Statistics

Computed for each detected regime:

```python
regime_stats = {
    'Bear': {
        'mean_return': -0.015,      # Average daily return
        'volatility': 0.025,         # Daily volatility
        'frequency': 0.30,           # % of time in regime
        'avg_duration': 15.2,        # Average regime duration (days)
        'median_duration': 12.0,
        'max_duration': 42,
        'min_duration': 3,
        'n_occurrences': 18,         # Number of regime instances
        'sharpe_ratio': -0.85,       # Risk-adjusted return
        'win_rate': 0.35,            # % of positive return days
        'annualized_return': -0.85,  # Annualized
        'annualized_volatility': 0.40
    },
    # ... other regimes
}
```

### Current Regime Analysis

```python
current_regime = {
    'regime': 'Bull',
    'regime_id': 2,
    'confidence': 0.87,              # Probability
    'days_in_regime': 12,            # Current duration
    'expected_total_duration': 18.5, # Historical average
    'probability_transition_soon': 0.23,  # Estimated
    'characteristics': {
        'direction': 'Upward',
        'volatility': 'Moderate',
        'strength': 'Strong'
    }
}
```

### Regime Transitions

```python
transitions = [
    {
        'date': '2023-03-15',
        'from_regime': 'Sideways',
        'to_regime': 'Bull',
        'confidence_before': 0.72,
        'confidence_after': 0.89,
        'trigger': 'Strong momentum shift',
        'price_change': 0.032  # 3.2% move
    },
    # ... more transitions
]
```

### Performance Attribution

Breakdown of returns by regime:

```python
attribution = {
    'total_return': 0.15,           # 15% total
    'regime_contributions': {
        'Bear': -0.05,              # -5% from bear regimes
        'Sideways': 0.02,           # +2% from sideways
        'Bull': 0.18                # +18% from bull regimes
    },
    'time_in_regime': {
        'Bear': 0.25,               # 25% of time
        'Sideways': 0.40,           # 40% of time
        'Bull': 0.35                # 35% of time
    },
    'return_per_day_in_regime': {
        'Bear': -0.0008,
        'Sideways': 0.0002,
        'Bull': 0.0020
    }
}
```

## Technical Indicator Comparison

### Supported Indicators

When `indicator_comparisons=True`:

**Trend Indicators:**
- SMA (Simple Moving Average) crossover
- EMA (Exponential Moving Average) crossover
- MACD (Moving Average Convergence Divergence)

**Momentum Indicators:**
- RSI (Relative Strength Index)
- Stochastic Oscillator
- CCI (Commodity Channel Index)

**Volatility Indicators:**
- Bollinger Bands
- ATR (Average True Range)
- Keltner Channels

### Comparison Analysis

```python
comparison = {
    'method_performance': {
        'HMM': {'sharpe': 1.25, 'return': 0.15, 'volatility': 0.12},
        'RSI': {'sharpe': 0.85, 'return': 0.10, 'volatility': 0.12},
        'MACD': {'sharpe': 0.95, 'return': 0.11, 'volatility': 0.12}
    },
    'signal_agreement': {
        'HMM_RSI': 0.68,        # 68% agreement
        'HMM_MACD': 0.72,       # 72% agreement
        'RSI_MACD': 0.81        # 81% agreement
    },
    'timing_analysis': {
        'HMM_leads_RSI': 2.3,   # HMM leads by 2.3 days average
        'HMM_leads_MACD': 1.8
    },
    'false_signals': {
        'HMM': {'false_positive': 0.12, 'false_negative': 0.15},
        'RSI': {'false_positive': 0.18, 'false_negative': 0.22}
    }
}
```

## Case Study Analysis

For temporal backtesting and evolution tracking:

```python
from hidden_regime.analysis.case_study import CaseStudyAnalyzer

analyzer = CaseStudyAnalyzer()

# Analyze regime evolution over time
evolution = analyzer.analyze_temporal_evolution(
    regime_history=historical_regimes,
    performance_history=historical_returns,
    dates=dates
)

# Results include:
# - Regime stability over time
# - Parameter drift detection
# - Performance consistency
# - Regime change patterns
```

## Signal Attribution

Track which signals contributed to performance:

```python
from hidden_regime.analysis.signal_attribution import SignalAttributor

attributor = SignalAttributor()

attribution = attributor.attribute_returns(
    returns=portfolio_returns,
    signals=trading_signals,
    regimes=regime_states
)

# Results:
# - Return contribution per signal type
# - Success rate by regime
# - Signal timing quality
```

## Usage Examples

### Example 1: Basic Regime Analysis

```python
from hidden_regime.analysis import FinancialAnalysis
from hidden_regime.config import FinancialAnalysisConfig

# Configure
config = FinancialAnalysisConfig(
    compute_regime_stats=True,
    n_states=3
)

# Analyze
analyzer = FinancialAnalysis(config)
result = analyzer.update(
    model_output=hmm_output,
    raw_data=price_data
)

# Result is human-readable summary string
print(result)
```

### Example 2: Comprehensive Analysis with Indicators

```python
from hidden_regime.config import FinancialAnalysisConfig

config = FinancialAnalysisConfig.create_comprehensive_financial()

analyzer = FinancialAnalysis(config)
result = analyzer.update(
    model_output=hmm_output,
    raw_data=price_data
)

# Access detailed results
stats = analyzer._last_analysis['regime_statistics']
comparison = analyzer._last_analysis['indicator_comparison']
```

### Example 3: Performance Comparison

```python
from hidden_regime.analysis import RegimePerformanceAnalyzer

analyzer = RegimePerformanceAnalyzer()

# Compare regime performance
metrics = analyzer.compute_regime_metrics(
    returns=returns,
    regimes=regimes,
    dates=dates
)

# Find best regime
best_regime = max(metrics.items(), key=lambda x: x[1]['sharpe_ratio'])
print(f"Best regime: {best_regime[0]} (Sharpe: {best_regime[1]['sharpe_ratio']:.2f})")
```

### Example 4: HMM vs. Indicators

```python
from hidden_regime.analysis import IndicatorPerformanceComparator

comparator = IndicatorPerformanceComparator()

comparison = comparator.compare_methods(
    hmm_regimes=hmm_states,
    indicator_signals={'RSI': rsi, 'MACD': macd},
    returns=returns
)

# Print comparison
for method, perf in comparison['method_performance'].items():
    print(f"{method}: Sharpe={perf['sharpe']:.2f}, Return={perf['return']:.1%}")
```

### Example 5: Pipeline Integration

```python
import hidden_regime as hr

# Analysis is automatic in pipeline
pipeline = hr.create_financial_pipeline(
    'AAPL',
    n_states=3
)

result = pipeline.update()  # Includes analysis

# Access detailed analysis
analysis_output = pipeline.component_outputs['analysis']
```

## Best Practices

### 1. Choose Appropriate Analysis Level

```python
# For quick analysis
config = FinancialAnalysisConfig.create_basic()

# For comprehensive research
config = FinancialAnalysisConfig.create_comprehensive_financial()

# For trading
config = FinancialAnalysisConfig.create_trading_focused()
```

### 2. Provide Raw Data When Available

```python
# Better: Includes price context
result = analyzer.update(
    model_output=hmm_output,
    raw_data=price_data  # Include this!
)

# Limited: Only regime stats
result = analyzer.update(model_output=hmm_output)
```

### 3. Use Appropriate Risk-Free Rate

```python
config = FinancialAnalysisConfig(
    risk_free_rate=0.05,  # Current rate environment
    trading_days_per_year=252
)
```

### 4. Validate Regime Labels

```python
# Ensure labels match number of states
config = FinancialAnalysisConfig(
    n_states=4,
    regime_labels=['Crisis', 'Bear', 'Sideways', 'Bull']
)
```

## Module Structure

```
analysis/
├── __init__.py                    # Public API
├── financial.py                   # FinancialAnalysis
├── performance.py                 # RegimePerformanceAnalyzer
├── indicator_comparison.py        # IndicatorPerformanceComparator
├── technical_indicators.py        # TechnicalIndicatorAnalyzer
├── case_study.py                  # Case study analysis
├── regime_evolution.py            # Regime evolution tracking
└── signal_attribution.py          # Signal attribution
```

## Related Modules

- **[models](../models/README.md)**: Model outputs (input to analysis)
- **[observations](../observations/README.md)**: Feature data used in analysis
- **[visualization](../visualization/README.md)**: Visualizing analysis results
- **[simulation](../simulation/README.md)**: Simulation performance analysis
- **[reports](../reports/README.md)**: Reporting analysis results

## Key Concepts

### Why Analysis Layer?

The analysis layer provides:

1. **Domain Knowledge**: Interprets mathematical outputs in financial context
2. **Actionable Insights**: Converts regime probabilities to trading signals
3. **Performance Metrics**: Risk-adjusted returns, drawdowns, etc.
4. **Comparative Analysis**: Benchmark against traditional methods
5. **Context**: Explains what regimes mean for trading

### Regime Characteristics

Analysis translates HMM states into meaningful financial regimes:

| State | Mean Return | Volatility | Interpretation | Trading Implication |
|-------|-------------|------------|----------------|---------------------|
| 0 | Negative | High | **Bear Market** | Reduce exposure, hedge |
| 1 | Near Zero | Low | **Sideways** | Range trading, neutral |
| 2 | Positive | Moderate | **Bull Market** | Increase exposure, momentum |
| 3 | Very Negative | Extreme | **Crisis** | Cash, defensive positions |

---

For complete examples using analysis, see `examples/` directory in the project root.