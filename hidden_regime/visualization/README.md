# Visualization Module

The visualization module provides comprehensive plotting capabilities for regime detection analysis, including HMM visualizations, technical indicator comparisons, performance analytics, and animations.

## Overview

Hidden Regime's visualization framework offers:

- **Regime Visualizations**: Timeline plots, heatmaps, transition matrices
- **Indicator Comparisons**: HMM vs. technical indicators side-by-side
- **Performance Analytics**: Returns, drawdowns, regime-specific metrics
- **Interactive Plots**: Advanced plotting classes for custom analysis
- **Animations**: Regime evolution over time
- **Data Collection**: Track decision-making and model evolution

## Quick Start

###  Basic Regime Plot

```python
from hidden_regime.visualization import plot_returns_with_regimes
import matplotlib.pyplot as plt

# Plot returns colored by regime
fig = plot_returns_with_regimes(
    returns=data['log_return'],
    regime_labels=predicted_regimes,
    dates=data['date']
)
plt.show()
```

### Regime Comparison

```python
from hidden_regime.visualization import plot_hmm_vs_indicators_comparison

# Compare HMM with RSI and MACD
fig = plot_hmm_vs_indicators_comparison(
    data=price_data,
    hmm_states=predicted_regimes,
    indicators=['RSI', 'MACD']
)
plt.show()
```

## Core Plotting Functions

### setup_financial_plot_style()

Apply consistent financial chart styling.

```python
from hidden_regime.visualization import setup_financial_plot_style

setup_financial_plot_style(style='professional')
# Options: 'professional', 'classic', 'academic'
```

### plot_returns_with_regimes()

Visualize returns colored by detected regime.

```python
from hidden_regime.visualization import plot_returns_with_regimes

fig = plot_returns_with_regimes(
    returns=log_returns,
    regime_labels=regimes,
    dates=dates,
    figsize=(14, 6),
    title='Market Returns by Regime'
)
```

### plot_regime_heatmap()

Heatmap showing regime probabilities over time.

```python
from hidden_regime.visualization import plot_regime_heatmap

# State probabilities from HMM
state_probs = hmm.predict_proba(returns)

fig = plot_regime_heatmap(
    state_probabilities=state_probs,
    dates=dates,
    regime_names=['Bear', 'Sideways', 'Bull']
)
```

### plot_regime_statistics()

Bar charts comparing regime characteristics.

```python
from hidden_regime.visualization import plot_regime_statistics

fig = plot_regime_statistics(
    regime_stats={
        0: {'mean_return': -0.01, 'volatility': 0.025, 'duration': 15},
        1: {'mean_return': 0.0, 'volatility': 0.015, 'duration': 25},
        2: {'mean_return': 0.01, 'volatility': 0.020, 'duration': 18}
    },
    regime_names=['Bear', 'Sideways', 'Bull']
)
```

## Indicator Comparison Plots

### plot_price_with_regimes_and_indicators()

Comprehensive view of price, regimes, and indicators.

```python
from hidden_regime.visualization import plot_price_with_regimes_and_indicators

fig = plot_price_with_regimes_and_indicators(
    data=ohlcv_data,
    regimes=hmm_regimes,
    indicators=['RSI', 'MACD', 'Bollinger_Bands']
)
```

**Displays:**
- Price chart with regime backgrounds
- RSI with overbought/oversold levels
- MACD with signal line
- Bollinger Bands
- Volume bars

### plot_hmm_vs_indicators_comparison()

Side-by-side comparison of HMM and technical indicators.

```python
from hidden_regime.visualization import plot_hmm_vs_indicators_comparison

fig = plot_hmm_vs_indicators_comparison(
    data=price_data,
    hmm_states=hmm_regimes,
    indicators=['RSI', 'MACD', 'SMA_Crossover'],
    comparison_period='6M'
)
```

**Shows:**
- HMM regime timeline
- Indicator signal timelines
- Agreement/disagreement analysis
- Regime change timing comparison

### plot_indicator_performance_dashboard()

Performance metrics for each indicator vs. HMM.

```python
from hidden_regime.visualization import plot_indicator_performance_dashboard

fig = plot_indicator_performance_dashboard(
    hmm_regimes=hmm_states,
    indicator_signals=indicator_dict,
    returns=log_returns
)
```

**Metrics:**
- Accuracy vs. HMM
- Signal timing
- Return attribution
- False positive/negative rates

### create_regime_transition_visualization()

Visualize regime transitions and timing.

```python
from hidden_regime.visualization import create_regime_transition_visualization

fig = create_regime_transition_visualization(
    regimes=regime_sequence,
    dates=dates,
    transition_points=True
)
```

## Advanced Plotting Classes

### RegimePlotter

Comprehensive regime analysis plotting.

```python
from hidden_regime.visualization import RegimePlotter

plotter = RegimePlotter()

# Create multi-panel regime analysis
fig = plotter.create_comprehensive_analysis(
    data=price_data,
    regimes=hmm_states,
    state_probs=state_probabilities,
    transition_matrix=transition_matrix
)

# Individual plots
fig = plotter.plot_regime_timeline(regimes, dates)
fig = plotter.plot_transition_matrix(transition_matrix)
fig = plotter.plot_regime_durations(regimes, dates)
```

### PerformancePlotter

Performance and risk metrics visualization.

```python
from hidden_regime.visualization import PerformancePlotter

plotter = PerformancePlotter()

# Performance dashboard
fig = plotter.create_performance_dashboard(
    returns=strategy_returns,
    regimes=detected_regimes,
    benchmark_returns=market_returns
)

# Individual metrics
fig = plotter.plot_cumulative_returns(returns)
fig = plotter.plot_drawdowns(returns)
fig = plotter.plot_rolling_sharpe(returns)
fig = plotter.plot_regime_specific_performance(returns, regimes)
```

### ComparisonPlotter

Compare multiple regime detection methods.

```python
from hidden_regime.visualization import ComparisonPlotter

plotter = ComparisonPlotter()

# Compare HMM vs. indicators
fig = plotter.create_comparison_dashboard(
    hmm_regimes=hmm_states,
    indicator_signals={'RSI': rsi_signals, 'MACD': macd_signals},
    returns=log_returns
)

# Correlation analysis
fig = plotter.plot_method_correlation(
    method_outputs={'HMM': hmm_states, 'RSI': rsi_signals}
)
```

### InteractivePlotter

Interactive plotting capabilities (requires additional dependencies).

```python
from hidden_regime.visualization import InteractivePlotter

plotter = InteractivePlotter()

# Create interactive regime plot
fig = plotter.create_interactive_regime_plot(
    data=price_data,
    regimes=hmm_states,
    include_indicators=True
)
```

## Utility Functions

### create_regime_timeline_plot()

Simple regime timeline visualization.

```python
from hidden_regime.visualization import create_regime_timeline_plot

fig = create_regime_timeline_plot(
    regimes=regime_sequence,
    dates=dates,
    labels=['Bear', 'Sideways', 'Bull']
)
```

### create_multi_asset_regime_comparison()

Compare regimes across multiple assets.

```python
from hidden_regime.visualization import create_multi_asset_regime_comparison

fig = create_multi_asset_regime_comparison(
    asset_regimes={
        'AAPL': aapl_regimes,
        'GOOGL': googl_regimes,
        'MSFT': msft_regimes
    },
    dates=common_dates
)
```

### get_regime_colors()

Get consistent regime color scheme.

```python
from hidden_regime.visualization import get_regime_colors

colors = get_regime_colors(['Bear', 'Sideways', 'Bull'])
# Returns: ['#d32f2f', '#f57c00', '#388e3c']
```

**Color Schemes:**
- `'classic'`: Red, Orange, Green
- `'professional'`: Darker variants
- `'colorblind_safe'`: Colorblind-friendly palette

### format_financial_axis()

Format axes for financial data.

```python
from hidden_regime.visualization import format_financial_axis
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
# ... plot data ...
format_financial_axis(ax, date_format='%Y-%m', rotation=45)
```

## Data Collection Visualization

### DataCollectionVisualizationSuite

Visualize collected simulation and decision data.

```python
from hidden_regime.visualization import DataCollectionVisualizationSuite

viz = DataCollectionVisualizationSuite(collected_data)

# Comprehensive dashboard
fig = viz.create_complete_visualization_suite()

# Individual visualizations
fig = viz.plot_decision_evolution()
fig = viz.plot_model_parameter_evolution()
fig = viz.plot_signal_attribution()
```

**Use Cases:**
- Track how model decisions evolve over time
- Analyze parameter stability
- Debug simulation behavior
- Identify regime detection patterns

## Animations

### Creating Regime Evolution Animations

```python
from hidden_regime.visualization.animations import create_regime_animation

animation = create_regime_animation(
    data=historical_data,
    regimes=regime_sequence,
    dates=dates,
    output_path='regime_evolution.gif',
    fps=5
)
```

**Features:**
- Animated regime transitions
- Rolling window regime detection
- Parameter evolution
- Performance tracking

## Styling and Customization

### Custom Color Schemes

```python
# Define custom colors
custom_colors = {
    'Bear': '#FF0000',
    'Sideways': '#FFFF00',
    'Bull': '#00FF00'
}

# Use in plots
fig = plot_returns_with_regimes(
    returns=returns,
    regime_labels=regimes,
    colors=custom_colors
)
```

### Plot Styling

```python
# Apply consistent styling
setup_financial_plot_style(style='professional')

# Or customize
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
```

## Usage Examples

### Example 1: Complete Regime Analysis

```python
from hidden_regime.visualization import RegimePlotter
import matplotlib.pyplot as plt

# Create plotter
plotter = RegimePlotter()

# Generate comprehensive analysis
fig = plotter.create_comprehensive_analysis(
    data=price_data,
    regimes=hmm_states,
    state_probs=state_probabilities,
    transition_matrix=transition_matrix,
    regime_names=['Bear', 'Sideways', 'Bull']
)

# Save figure
fig.savefig('regime_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Example 2: HMM vs. Indicators

```python
from hidden_regime.visualization import plot_hmm_vs_indicators_comparison

# Compare HMM with multiple indicators
fig = plot_hmm_vs_indicators_comparison(
    data=ohlcv_data,
    hmm_states=hmm_regimes,
    indicators=['RSI', 'MACD', 'SMA_Crossover', 'Bollinger_Bands']
)

plt.savefig('hmm_vs_indicators.png', dpi=300)
plt.show()
```

### Example 3: Performance Dashboard

```python
from hidden_regime.visualization import PerformancePlotter

plotter = PerformancePlotter()

# Create performance dashboard
fig = plotter.create_performance_dashboard(
    returns=strategy_returns,
    regimes=detected_regimes,
    benchmark_returns=spy_returns,
    title='Regime-Based Strategy Performance'
)

plt.show()
```

### Example 4: Multi-Asset Comparison

```python
from hidden_regime.visualization import create_multi_asset_regime_comparison

# Compare regime detection across assets
fig = create_multi_asset_regime_comparison(
    asset_regimes={
        'AAPL': aapl_hmm_states,
        'GOOGL': googl_hmm_states,
        'MSFT': msft_hmm_states,
        'SPY': spy_hmm_states
    },
    dates=common_dates,
    title='Tech Stock Regime Comparison'
)

plt.show()
```

## Best Practices

### 1. Use Consistent Styling

```python
from hidden_regime.visualization import setup_financial_plot_style

# Apply at start of notebook/script
setup_financial_plot_style(style='professional')
```

### 2. Save High-Resolution Figures

```python
fig.savefig('analysis.png', dpi=300, bbox_inches='tight')
# For presentations: dpi=150
# For publications: dpi=300-600
```

### 3. Label Regimes Clearly

```python
regime_names = ['Bear Market', 'Sideways', 'Bull Market']

# Use names throughout visualizations
fig = plot_returns_with_regimes(
    returns=returns,
    regime_labels=regimes,
    regime_names=regime_names
)
```

### 4. Include Dates

```python
# Always include dates for context
fig = plot_regime_heatmap(
    state_probabilities=probs,
    dates=dates,  # Important!
    regime_names=names
)
```

## Module Structure

```
visualization/
├── __init__.py                    # Public API exports
├── plotting.py                    # Core plotting functions
├── indicators.py                  # Indicator comparison plots
├── advanced_plots.py              # Advanced plotting classes
├── animations.py                  # Animation capabilities
└── data_collection_plots.py       # Data collection viz
```

## Related Modules

- **[models](../models/README.md)**: HMM models (generate regime data for plotting)
- **[analysis](../analysis/README.md)**: Analysis results (input to visualizations)
- **[simulation](../simulation/README.md)**: Simulation results (performance plots)
- **[reports](../reports/README.md)**: Report generation (includes plots)

## Dependencies

- **matplotlib** >= 3.4.0: Core plotting
- **seaborn**: Enhanced statistical plots
- **pandas**: Data handling for plots

**Optional:**
- **plotly**: Interactive plots (InteractivePlotter)
- **imageio**: Animation creation

---

For complete examples using visualization, see `examples/` directory in the project root.