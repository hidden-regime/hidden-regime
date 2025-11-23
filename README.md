# Hidden Regime

**Pipeline-based market regime detection using Hidden Markov Models for quantitative finance.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Hidden Regime is a Python package for detecting and analyzing market regimes using Hidden Markov Models (HMMs). Built on a **pipeline architecture** that ensures temporal data isolation and rigorous backtesting, Hidden Regime provides three levels of abstraction:

1. **Simple Pipelines** - Quick regime detection with sensible defaults
2. **Financial Pipelines** - Comprehensive analysis with regime characterization and trading signals
3. **Market Event Studies** - High-level framework for analyzing regime behavior during market events (crashes, bubbles, sector rotations)

All analysis flows through a consistent pipeline: **Data → Observation → Model → Analysis → Report**, ensuring reproducibility and enabling rigorous verification & validation (V&V) for backtesting.

**FOR EDUCATIONAL PURPOSES ONLY.** _This is not financial advice and should not be considered as such._

Hidden Regime is a mathematical tool designed for educational purposes only to explore financial concepts and analysis techniques. It is not financial advice, and its outputs should not be used to make investment decisions. Always consult with a qualified financial professional before making any investment decisions.

## Features

- **Pipeline Architecture**: Modular Data → Observation → Model → Analysis → Report flow with temporal isolation for V&V backtesting
- **Three Abstraction Levels**: Simple pipelines, financial pipelines, and market event studies
- **MarketEventStudy Framework**: High-level API for analyzing regime behavior during market events with multi-ticker support
- **Hidden Markov Models**: 2-5 state HMMs with Baum-Welch training and Viterbi inference
- **Regime Characterization**: Automatic financial analysis (returns, volatility, win rates, drawdowns, regime strength)
- **Temporal Analysis**: Step through historical periods day-by-day with rigorous data isolation
- **Financial Data Integration**: Robust yfinance data loading with comprehensive validation
- **Visualization**: Regime plots, animations, snapshots, and interactive charts
- **Trading Simulation**: Backtest regime-based strategies with risk management
- **Technical Indicator Comparison**: Compare HMM regime detection against traditional indicators
- **Reporting**: Generate markdown reports with analysis results and recommendations
- **🆕 MCP Server Integration**: Use regime detection directly in AI assistants like Claude Desktop via Model Context Protocol

## AI Assistant Integration (Claude Desktop)

Hidden Regime now includes an **MCP (Model Context Protocol) server** that brings regime detection to AI assistants like Claude Desktop. Ask Claude natural language questions and get sophisticated HMM regime analysis automatically.

### Quick Setup

```bash
# 1. Install with MCP extras
pip install hidden-regime[mcp]

# 2. Add to Claude Desktop config
# macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
# Linux: ~/.config/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "hidden-regime": {
      "command": "/path/to/python",
      "args": ["-m", "fastmcp", "run", "/path/to/hidden_regime_mcp/server.py"]
    }
  }
}

# 3. Restart Claude Desktop and start asking questions!
```

### What You Can Do

**Ask Natural Language Questions:**
```
"What's SPY's current market regime?"
"Analyze NVDA's regime behavior over the past 2 years"
"Compare regimes across SPY, QQQ, and IWM"
"What trading strategies suit QQQ's current regime?"
```

**Use Expert-Level Prompts:**
- 📊 **regime-quick-check** - Fast market health check for daily decisions
- 🔍 **regime-deep-dive** - Comprehensive multi-tool analysis with historical context
- 💼 **regime-strategy-advisor** - Trading strategy recommendations based on regime + risk
- 📈 **regime-multi-asset-comparison** - Cross-asset regime comparison and diversification
- ⚠️ **regime-risk-assessment** - Quantitative risk scoring (0-100) with mitigation strategies
- 📚 **regime-historical-analogs** - Find similar past patterns and predict outcomes
- 💰 **regime-portfolio-review** - Portfolio-wide regime analysis with rebalancing recommendations

**Available Interfaces:**
- ✅ **Tools** (3) - Core regime detection operations
- ✅ **Resources** (2) - URI-based data access (`regime://SPY/current`)
- ✅ **Prompts** (7) - Pre-configured expert analysis workflows
- ✅ **Logging** - Structured logging showing data fetching, model training, and caching

### Learn More

See **[MCP Integration Guide](docs/integrations/mcp/README.md)** for:
- Complete setup instructions for all platforms
- Detailed documentation of all tools and prompts
- Usage examples and scenarios
- Troubleshooting guide

See **[MCP Roadmap](docs/integrations/mcp/roadmap.md)** for:
- Implementation status and roadmap
- Planned features and enhancements
- Contributing to MCP development

## Installation

```bash
pip install hidden-regime
```

## Quick Start

Hidden Regime provides three levels of abstraction - choose based on your needs:

### Level 1: Simple Regime Detection

Quick regime detection with sensible defaults:

```python
import hidden_regime as hr

# Create and run pipeline
pipeline = hr.create_simple_regime_pipeline('AAPL', n_states=3)
result = pipeline.update()
print(result)  # Shows current regime and confidence
```

### Level 2: Financial Analysis Pipeline

Comprehensive analysis with regime characterization and trading signals:

```python
import hidden_regime as hr

# Create financial analysis pipeline
pipeline = hr.create_financial_pipeline(
    ticker='SPY',
    n_states=3,
    start_date='2023-01-01',
    end_date='2024-01-01'
)

# Run analysis
result = pipeline.update()
print(result)  # Shows regime, confidence, returns, volatility, win rates, etc.
```

### Level 3: Market Event Study

Analyze regime behavior during market events (crashes, bubbles, etc.):

```python
import hidden_regime as hr

# Create market event study
study = hr.MarketEventStudy(
    ticker='QQQ',
    training_start='2018-01-01',
    training_end='2019-12-31',
    analysis_start='2020-01-01',
    analysis_end='2020-12-31',
    n_states=3,
    key_events={'2020-03-23': 'Market Bottom'},
    output_dir='output/covid_study'
)

# Run complete analysis
study.run(create_snapshots=True, create_animations=True)
study.print_summary()
study.export_results(format='csv')
```

## Core Concepts

### Market Regimes

Financial markets exhibit distinct behavioral phases:

| Regime | Characteristics | Typical Duration |
|--------|-----------------|------------------|
| **Bull** | Positive returns, moderate volatility | Weeks to months |
| **Bear** | Negative returns, high volatility | Days to weeks |
| **Sideways** | Near-zero returns, low volatility | Days to weeks |
| **Crisis** | Very negative returns, extreme volatility | Days |

Hidden Regime uses HMMs to automatically detect these regimes from price data.

### Pipeline Architecture

```
Data Loading → Observation Generation → Model Training → Analysis → Reporting
     ↓                  ↓                      ↓            ↓           ↓
  yfinance        Log Returns              HMM Fit     Regime Stats   Markdown
  Validation      Features              State Inference  Performance   Plots
```

Every analysis in Hidden Regime flows through this pipeline. Each component has a clear responsibility:

- **Data**: Loads and validates price data from yfinance with temporal isolation
- **Observation**: Transforms prices into statistical features (log returns by default)
- **Model**: Trains HMM using Baum-Welch and performs Viterbi inference for regime prediction
- **Analysis**: Characterizes regimes (returns, volatility, win rates, drawdowns, regime strength)
- **Report**: Exports results as markdown reports and visualizations

**You never need to manually chain these components** - the pipeline handles all data flow automatically.

### When to Use What

Choose the right API for your task:

| **Use Case** | **Recommended API** | **Example** |
|-------------|---------------------|-------------|
| Quick regime detection | `create_simple_regime_pipeline()` | Identify current market regime |
| Financial analysis | `create_financial_pipeline()` | Regime returns, volatility, trading signals |
| Market event study | `MarketEventStudy` | Analyze COVID crash, dot-com bubble, etc. |
| Trading strategies | `create_trading_pipeline()` | Regime-based position sizing |
| Academic research | `create_research_pipeline()` | Comprehensive indicator comparison |
| Temporal backtesting | `create_temporal_controller()` | Day-by-day simulation with V&V isolation |
| Full customization | Manual pipeline creation | Custom components and configs |

## Examples

The `examples/` directory contains working demonstrations. **Most users should start with the financial pipeline or MarketEventStudy examples.**

### Recommended Starting Point
- **`case_study_covid_2020.py`** - **MarketEventStudy showcase** (135 lines, multi-ticker COVID crash analysis)
- **`01_real_market_analysis.py`** - Financial pipeline usage with real market data

### Getting Started
- **`00_basic_regime_detection.py`** - Manual component usage (understanding pipeline internals)
- **`01_real_market_analysis.py`** - Using `create_financial_pipeline()` API

### Market Event Studies
- **`case_study_covid_2020.py`** - COVID crash (QQQ vs CCL, fastest crisis regime in history)
- **`case_study_template.py`** - Template for creating new event studies

### Advanced Examples
- **`02_regime_comparison_analysis.py`** - Compare different regime models
- **`03_trading_strategy_demo.py`** - Regime-based trading strategies
- **`04_multi_stock_comparative_study.py`** - Multi-asset comparative analysis
- **`05_advanced_analysis_showcase.py`** - Advanced visualization and analysis
- **`initialization_methods_demo.py`** - KMeans, Random, and Custom initialization with transfer learning

### Running Examples

```bash
# Recommended: Start with MarketEventStudy
python examples/case_study_covid_2020.py

# Or use financial pipeline
python examples/01_real_market_analysis.py

# Basic example (manual component usage)
python examples/00_basic_regime_detection.py
```

## Documentation

- **[Data Pipeline](hidden_regime/data/README.md)**: Data loading, validation, and preprocessing
- **[Models](hidden_regime/models/README.md)**: HMM implementation and algorithms
- **Examples**: See `examples/` directory for working code

## MarketEventStudy Framework

The `MarketEventStudy` class provides a high-level API for analyzing regime behavior during market events. It encapsulates the entire workflow: data loading, model training, temporal analysis, visualization, and metrics computation.

### Basic Usage

```python
import hidden_regime as hr

study = hr.MarketEventStudy(
    ticker='QQQ',                    # Single ticker or list of tickers
    training_start='2018-01-01',     # Train on pre-event period
    training_end='2019-12-31',
    analysis_start='2020-01-01',     # Analyze event period
    analysis_end='2020-12-31',
    n_states=3,                      # Number of regime states
    key_events={                     # Optional: dates for snapshots
        '2020-02-19': 'Market Peak',
        '2020-03-23': 'Market Bottom'
    },
    output_dir='output/covid_study'
)

# Run complete analysis
study.run(
    create_snapshots=True,           # PNG snapshots at key dates
    create_animations=True,          # GIF showing regime evolution
    snapshot_window_days=90,         # Window size for snapshots
    animation_fps=5                  # Animation frame rate
)

# Print summary metrics
study.print_summary()

# Export results
study.export_results(format='csv')   # or 'json'
```

### Multi-Ticker Analysis

Compare regime behavior across multiple assets:

```python
study = hr.MarketEventStudy(
    ticker=['QQQ', 'CCL', 'XLE'],    # Compare tech, travel, energy
    training_start='2018-01-01',
    training_end='2019-12-31',
    analysis_start='2020-01-01',
    analysis_end='2020-12-31',
    n_states=3
)

study.run()

# Get metrics for specific ticker
qqq_metrics = study.get_metrics('QQQ')
print(f"Detection lag: {qqq_metrics['detection_lag_days']} days")
print(f"Crisis days: {qqq_metrics['crisis_days']}")
```

### What It Does

1. **Trains HMM** on pre-event period (isolates training from analysis)
2. **Steps through time** day-by-day during event period (temporal V&V isolation)
3. **Creates visualizations** at key event dates (e.g., market peak, bottom, recovery)
4. **Computes metrics** (detection lag, regime stability, crisis duration)
5. **Exports results** to CSV/JSON for further analysis

### Generated Outputs

- **Snapshots**: `{ticker}_snapshot_{date}.png` - Price + regime overlay at key dates
- **Animations**: `{ticker}_full_analysis.gif` - Full regime evolution over time
- **Metrics CSV**: `regime_history.csv` - Complete regime timeline with confidence scores
- **Console Summary**: Detection lag, regime durations, transition counts

See `examples/case_study_covid_2020.py` for a complete working example (135 lines).

## Configuration

Hidden Regime uses dataclass-based configuration for flexibility:

```python
from hidden_regime.config import HMMConfig, FinancialDataConfig

# Configure HMM
hmm_config = HMMConfig(
    n_states=3,
    max_iterations=100,
    tolerance=1e-6,
    initialization_method='kmeans',
    random_seed=42
)

# Configure data loading
data_config = FinancialDataConfig(
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    use_ohlc_average=True
)

# Create pipeline with custom configs
from hidden_regime.factories import pipeline_factory

pipeline = pipeline_factory.create_pipeline(
    data_config=data_config,
    model_config=hmm_config,
    # ... other configs
)
```

## Initialization Methods

Hidden Regime supports three approaches for initializing HMM parameters:

### 1. KMeans (Default - Recommended)

Data-driven clustering approach that automatically discovers regime structure:

```python
config = HMMConfig(
    n_states=3,
    initialization_method='kmeans',  # This is the default
)
```

**When to use:**
- ✓ Default choice for most applications
- ✓ No prior knowledge about regime characteristics
- ✓ Sufficient historical data (>200 observations)

### 2. Random (Quantile-Based)

Simple initialization using data quantiles:

```python
config = HMMConfig(
    n_states=3,
    initialization_method='random',
)
```

**When to use:**
- ✓ Fallback when sklearn not available
- ✓ Small datasets where KMeans struggles
- ✓ Quick prototyping

### 3. Custom (Expert-Specified)

Specify exact starting parameters based on domain knowledge:

```python
# Option 1: Direct specification
config = HMMConfig(
    n_states=3,
    initialization_method='custom',
    custom_emission_means=[-0.015, 0.0, 0.012],  # Bear, Sideways, Bull
    custom_emission_stds=[0.025, 0.015, 0.020],
    # Optional: custom_transition_matrix, custom_initial_probs
)

# Option 2: Convenience factory
config = HMMConfig.from_regime_specs([
    {'mean': -0.015, 'std': 0.025},  # Bear
    {'mean': 0.0, 'std': 0.015},     # Sideways
    {'mean': 0.012, 'std': 0.020},   # Bull
])
```

**When to use:**
- ✓ Transfer learning (use trained params from similar asset)
- ✓ Incorporating expert domain knowledge
- ✓ Research reproducibility
- ✓ Testing specific regime hypotheses

**Note:** Custom parameters are **starting values** - Baum-Welch training will update them based on actual data.

See `examples/initialization_methods_demo.py` for complete demonstrations including transfer learning workflows.

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest --cov=hidden_regime --cov-report=html tests/
```

## Dependencies

- **pandas** >= 2.0.0 - Data manipulation
- **numpy** >= 2.0.0 - Numerical computing
- **scipy** >= 1.7.0 - Scientific computing
- **yfinance** >= 0.2.0 - Financial data
- **matplotlib** >= 3.4.0 - Visualization
- **ta** >= 0.10.2 - Technical indicators

## Project Structure

```
hidden_regime/
├── analysis/       # Regime analysis and indicator comparison
├── config/         # Configuration dataclasses
├── data/          # Data loading and validation
├── factories/     # Pipeline and component factories
├── financial/     # Financial-specific utilities
├── models/        # HMM implementation
├── observations/  # Observation generation
├── pipeline/      # Core pipeline architecture
├── reports/       # Report generation
├── simulation/    # Trading simulation
├── utils/         # Utility functions
└── visualization/ # Plotting and charts
```

## Use Cases

### Market Event Analysis

Analyze regime behavior during crashes, bubbles, or sector rotations:

```python
import hidden_regime as hr

study = hr.MarketEventStudy(
    ticker=['SPY', 'QQQ', 'TLT'],
    training_start='2018-01-01',
    training_end='2019-12-31',
    analysis_start='2020-01-01',
    analysis_end='2020-12-31',
    n_states=3,
    key_events={'2020-03-23': 'Market Bottom'}
)

study.run(create_snapshots=True, create_animations=True)
study.print_summary()
```

### Regime-Based Trading

Detect current market regime and adjust strategies accordingly:

```python
import hidden_regime as hr

pipeline = hr.create_trading_pipeline('SPY', n_states=4, risk_adjustment=True)
result = pipeline.update()

# Access regime information
current_regime = result['regime_name'].iloc[-1]
confidence = result['confidence'].iloc[-1]
print(f"Current regime: {current_regime} ({confidence:.1%} confidence)")
```

### Research and Analysis

Analyze historical regime behavior with comprehensive indicators:

```python
import hidden_regime as hr

pipeline = hr.create_research_pipeline('BTC-USD', comprehensive_analysis=True)
result = pipeline.update()

# Result includes regime characterization, technical indicators, and performance metrics
```

### Temporal Backtesting

Test regime-based strategies with rigorous V&V isolation:

```python
import hidden_regime as hr

pipeline = hr.create_financial_pipeline('AAPL', n_states=3)
data = pipeline.data.get_all_data()

# Step through time day-by-day
controller = hr.create_temporal_controller(pipeline, data)
results = controller.step_through_time('2023-01-01', '2023-12-31')
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Hidden Regime in your research, please cite:

```bibtex
@software{hidden_regime,
  title = {Hidden Regime: Market Regime Detection using Hidden Markov Models},
  author = {aoaustin},
  year = {2025},
  url = {https://github.com/hidden-regime/hidden-regime}
}
```

## Documentation

### Core Documentation

- **[Quickstart Guide](docs/guides/quickstart.md)** - Get started in 5 minutes
- **[Architecture Overview](ARCHITECTURE.md)** - System design and pipeline flow
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute, test, and develop
- **[Changelog](CHANGELOG.md)** - Version history and release notes

### User Guides

- **[Configuration Guide](docs/guides/configuration.md)** - Configure all components
- **[Troubleshooting](docs/guides/troubleshooting.md)** - Common issues and solutions
- **Setup Guides**: [Linux](docs/guides/SETUP_LINUX.md) | [macOS](docs/guides/SETUP_MACOS.md) | [Windows](docs/guides/SETUP_WINDOWS.md)

### Advanced Topics

- **[Advanced Features](docs/advanced/features.md)** - Multi-timeframe detection, temporal analysis
- **[Trading Applications](docs/advanced/trading.md)** - Regime-based trading strategies
- **[Case Studies](docs/advanced/case-studies.md)** - Historical market event analysis
- **[Optimization Guide](docs/advanced/optimization.md)** - Performance tuning
- **[Community Examples](docs/advanced/community-examples.md)** - User-contributed strategies

### Reference

- **[Mathematical Foundations](docs/reference/mathematical-foundations.md)** - HMM theory and algorithms
- **[Deployment Guide](docs/reference/deployment.md)** - Production deployment
- **[Error Handling](docs/reference/error-handling.md)** - Exception handling reference

### Integrations

- **[MCP Server](docs/integrations/mcp/README.md)** - Claude Desktop integration
- **[Experimental Features](docs/experimental/README.md)** - QuantConnect and other experiments

### Module Documentation

See module READMEs in `hidden_regime/*/README.md` for detailed component documentation.

## Support

- **Examples**: Working code in `examples/` directory
- **Issues**: Report bugs and request features on GitHub
- **Website**: [hiddenregime.com](https://hiddenregime.com)

## Acknowledgments

Built with inspiration from academic research in regime-switching models and modern quantitative finance practices.

---

**Hidden Regime** - Quantitative market regime detection for systematic trading.

--- 

**FOR EDUCATIONAL PURPOSES ONLY.** _This is not financial advice and should not be considered as such._

 Hidden Regime is a mathematical tool designed for educational purposes only to explore financial concepts and analysis techniques. It is not financial advice, and its outputs should not be used to make investment decisions. Always consult with a qualified financial professional before making any investment decisions.
