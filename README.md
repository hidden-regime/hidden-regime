# Hidden Regime

**Market regime detection using Hidden Markov Models for quantitative finance.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Hidden Regime is a Python package for detecting and analyzing market regimes using Hidden Markov Models (HMMs). It provides a complete pipeline for regime-based trading analysis, from data loading through model training to trading simulation and reporting.

**FOR EDUCATIONAL PURPOSES ONLY.** _This is not financial advice and should not be considered as such._

 Hidden Regime is a mathematical tool designed for educational purposes only to explore financial concepts and analysis techniques. It is not financial advice, and its outputs should not be used to make investment decisions. Always consult with a qualified financial professional before making any investment decisions.

## Features

- **Hidden Markov Models**: 2-5 state HMMs with Baum-Welch training and Viterbi inference
- **Financial Data Pipeline**: Robust data loading with yfinance integration and comprehensive validation
- **Pipeline Architecture**: Modular Data → Observation → Model → Analysis → Report flow
- **Technical Indicator Comparison**: Compare HMM regime detection against traditional indicators (using `ta` library)
- **Trading Simulation**: Backtest regime-based strategies with risk management and performance analytics
- **Case Studies**: Temporal analysis framework for evaluating models over time
- **Visualization**: Comprehensive plotting for regimes, indicators, and performance metrics
- **Data Collection**: Track simulation decisions and model evolution for detailed analysis
- **Reporting**: Generate markdown reports with analysis results and recommendations

## Installation

```bash
pip install hidden-regime
```

## Quick Start

### Basic Regime Detection

```python
import hidden_regime as hr

# Create a simple regime detection pipeline
pipeline = hr.create_simple_regime_pipeline('AAPL', n_states=3)

# Run analysis
result = pipeline.update()
print(result)
```

### Financial Analysis Pipeline

```python
import hidden_regime as hr

# Create pipeline with financial analysis
pipeline = hr.create_financial_pipeline(
    ticker='SPY',
    n_states=3,
    start_date='2023-01-01',
    end_date='2024-01-01'
)

# Update and get results
result = pipeline.update()
```

### Direct HMM Usage

```python
from hidden_regime.models import HiddenMarkovModel, HMMConfig
from hidden_regime.data import FinancialDataLoader

# Load data
loader = FinancialDataLoader()
data = loader.load('AAPL', '2023-01-01', '2024-01-01')

# Train HMM
config = HMMConfig.for_market_data(conservative=True)
hmm = HiddenMarkovModel(config=config)
hmm.fit(data['log_return'].values)

# Analyze regimes
analysis = hmm.analyze_regimes(data['log_return'].values)
print(f"Detected {hmm.n_states} regimes")
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

## Examples

The `examples/` directory contains working demonstrations:

### Getting Started
- **`00_basic_regime_detection.py`** - 3-state HMM with synthetic data
- **`01_real_market_analysis.py`** - Real stock data analysis

### Core Use Cases (v1.0.0 Requirements)
1. **2-state HMM**: See `00_basic_regime_detection.py` (modify `n_states=2`)
2. **3-state HMM**: See `00_basic_regime_detection.py` or `01_real_market_analysis.py`
3. **Current regime analysis**: See `01_real_market_analysis.py`
4. **Pipeline example**: All examples use the pipeline architecture
5. **Simple case study**: See `case_study_basic.py`
6. **Comprehensive case study**: See `case_study_comprehensive.py` (HMM vs indicators)

### Advanced Examples
- **`02_regime_comparison_analysis.py`** - Compare different regime models
- **`03_trading_strategy_demo.py`** - Regime-based trading strategies
- **`04_multi_stock_comparative_study.py`** - Multi-asset analysis
- **`05_advanced_analysis_showcase.py`** - Advanced visualization and analysis

### Running Examples

```bash
# Navigate to project directory
cd /path/to/hidden-regime

# Run basic example
python examples/00_basic_regime_detection.py

# Run with real market data
python examples/01_real_market_analysis.py

# Run comprehensive case study
python examples/case_study_comprehensive.py
```

## Documentation

- **[Data Pipeline](hidden_regime/data/README.md)**: Data loading, validation, and preprocessing
- **[Models](hidden_regime/models/README.md)**: HMM implementation and algorithms
- **Examples**: See `examples/` directory for working code

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

### Regime-Based Trading

Detect market regimes and adjust trading strategies accordingly:

```python
pipeline = hr.create_trading_pipeline('SPY', n_states=4, risk_adjustment=True)
result = pipeline.update()

# Use result to inform trading decisions
```

### Research and Analysis

Analyze historical regime behavior across multiple assets:

```python
pipeline = hr.create_research_pipeline('BTC-USD', comprehensive_analysis=True)
result = pipeline.update()
```

### Backtesting

Test regime-based strategies over time:

```python
from hidden_regime.pipeline import TemporalController

pipeline = hr.create_financial_pipeline('AAPL')
data = pipeline.data.get_all_data()

controller = hr.create_temporal_controller(pipeline, data)
results = controller.step_through_time('2023-01-01', '2024-01-01')
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

## Support

- **Documentation**: See module READMEs in `hidden_regime/*/README.md`
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
