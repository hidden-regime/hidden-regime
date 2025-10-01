# Hidden Regime Examples

This directory contains 12 working examples demonstrating the capabilities of the Hidden Regime package v1.0.0. All examples have been tested and verified to execute successfully.

## 🚀 Quick Start

```bash
# Activate the virtual environment
source /home/aoaustin/hidden-regime-pyenv/bin/activate

# Run any example
python examples/00_basic_regime_detection.py
```

All examples generate output in the `output/` directory including:
- Analysis reports (Markdown format)
- Visualizations (PNG images)
- Performance metrics (JSON)
- Data files (CSV/Parquet)

## 📚 Example Categories

### Beginner Examples

#### 00_basic_regime_detection.py
**Runtime**: ~30 seconds
**Purpose**: Minimal working example showing the core pipeline workflow

Demonstrates:
- Loading financial data with `FinancialDataLoader`
- Generating observations with `FinancialObservationGenerator`
- Training a 3-state HMM model
- Running basic financial analysis
- Generating analysis reports

**Run**:
```bash
python examples/00_basic_regime_detection.py
```

#### 01_real_market_analysis.py
**Runtime**: ~60 seconds
**Purpose**: Regime detection on real market data with error handling

Demonstrates:
- Robust data loading with fallback to sample data
- Regime change detection in actual market conditions
- Professional analysis report generation
- Publication-quality visualizations
- Graceful handling of data availability issues

**Run**:
```bash
python examples/01_real_market_analysis.py
```

### Intermediate Examples

#### 02_regime_comparison_analysis.py
**Runtime**: ~20 seconds
**Purpose**: Compare regime patterns across different assets

Demonstrates:
- Multi-asset regime detection
- Synchronous regime change identification
- Cross-market correlation analysis
- Divergence and convergence pattern detection
- Portfolio analysis applications

**Run**:
```bash
python examples/02_regime_comparison_analysis.py
```

#### 03_trading_strategy_demo.py
**Runtime**: ~15 seconds
**Purpose**: Build practical trading strategies based on regime detection

Demonstrates:
- Regime-based position sizing
- Entry and exit signal generation
- Performance metrics calculation
- Realistic trading simulation with transaction costs
- Strategy performance reporting

**Run**:
```bash
python examples/03_trading_strategy_demo.py
```

#### 04_multi_stock_comparative_study.py
**Runtime**: ~180 seconds
**Purpose**: Comprehensive regime analysis across multiple stocks

Demonstrates:
- Batch processing of multiple tickers
- Cross-stock regime correlation analysis
- Sector-based comparative studies
- Market regime consensus identification
- Institutional-quality comparative reporting

**Run**:
```bash
python examples/04_multi_stock_comparative_study.py
```

### Advanced Examples

#### 05_advanced_analysis_showcase.py
**Runtime**: ~15 seconds
**Purpose**: Complete working pipeline with all components

Demonstrates:
- Full data → observation → model → analysis pipeline
- Comprehensive financial metrics
- Report generation with markdown output
- Proper handling of edge cases
- Current API best practices

**Run**:
```bash
python examples/05_advanced_analysis_showcase.py
```

#### improved_features.py
**Runtime**: ~120 seconds
**Purpose**: Enhanced features for improved regime detection

Demonstrates:
- **Momentum strength**: Bull/Bear momentum detection
- **Trend persistence**: Sideways regime identification
- **Volatility context**: Crisis period detection
- **Directional consistency**: Regime characterization
- Baseline vs enhanced feature comparison
- Configuration-driven feature selection

**Run**:
```bash
python examples/improved_features.py
```

### Case Study Examples

#### case_study.py
**Runtime**: ~30 seconds
**Purpose**: Main case study orchestrator implementing 4-phase workflow

Demonstrates:
- **Phase 1 - Training**: Train HMM on historical data
- **Phase 2 - Evolution**: Daily regime updates over evaluation period
- **Phase 3 - Visualization**: Charts and animations
- **Phase 4 - Analysis**: Compare vs buy-and-hold and technical indicators
- Proper temporal isolation
- Comprehensive performance analysis

**Run**:
```bash
python examples/case_study.py
```

#### case_study_basic.py
**Runtime**: ~30 seconds
**Purpose**: Simple financial regime analysis with unified API

Demonstrates:
- `FinancialRegimeAnalysis` unified entry point
- Quick analysis configuration
- 3-month analysis on single stock
- Simplified workflow for rapid testing

**Run**:
```bash
python examples/case_study_basic.py
```

#### case_study_comprehensive.py
**Runtime**: ~60 seconds
**Purpose**: Full-featured analysis with all advanced options

Demonstrates:
- Comprehensive analysis configuration
- Intelligent signal generation
- Full trading simulation
- Detailed performance metrics
- Advanced reporting capabilities

**Run**:
```bash
python examples/case_study_comprehensive.py
```

#### case_study_multi_asset.py
**Runtime**: ~120 seconds
**Purpose**: Comparative analysis across multiple assets

Demonstrates:
- Batch case study execution
- Cross-asset regime comparison
- Sector analysis capabilities
- Asset class comparative studies
- Multi-ticker output organization

**Run**:
```bash
python examples/case_study_multi_asset.py
```

#### financial_case_study.py
**Runtime**: ~90 seconds
**Purpose**: Financial-first architecture showcase

Demonstrates:
- Data-driven regime characterization (not naive state assumptions)
- Intelligent signal generation using regime characteristics
- Single-asset optimized position sizing (100% allocation)
- Zero transaction cost defaults (retail-friendly)
- Unified configuration and analysis

**Run**:
```bash
python examples/financial_case_study.py
```

## 🎯 Running All Examples

To verify all examples work in your environment:

```bash
# Activate environment
source /home/aoaustin/hidden-regime-pyenv/bin/activate

# Run all examples (this will take ~10 minutes)
for example in examples/*.py; do
    echo "Running $example..."
    python "$example"
done
```

## 📊 Example Output Structure

All examples generate organized output:

```
output/
├── plots/                          # Visualizations (PNG)
│   ├── regime_analysis_*.png
│   ├── performance_comparison_*.png
│   └── enhanced_features_*.png
├── reports/                        # Analysis reports (Markdown)
│   ├── *_analysis_report.md
│   └── *_case_study_report.md
├── data/                          # Data exports (JSON/CSV/Parquet)
│   ├── evolution_results.json
│   ├── performance_metrics.json
│   └── regime_history.parquet
└── animations/                    # Time-lapse animations (GIF)
    └── *_regime_evolution.gif
```

## 🏗️ Pipeline Architecture

All examples follow the consistent pipeline pattern:

```python
# 1. Data Loading
from hidden_regime.data.financial import FinancialDataLoader
from hidden_regime.config.data import FinancialDataConfig

data_config = FinancialDataConfig(ticker="AAPL", ...)
loader = FinancialDataLoader(data_config)
raw_data = loader.update()

# 2. Observation Generation
from hidden_regime.observations.financial import FinancialObservationGenerator
from hidden_regime.config.observation import FinancialObservationConfig

obs_config = FinancialObservationConfig(generators=["log_return"])
obs_gen = FinancialObservationGenerator(obs_config)
observations = obs_gen.update(raw_data)

# 3. Model Training
from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.config.model import HMMConfig

model_config = HMMConfig(n_states=3)
hmm_model = HiddenMarkovModel(config=model_config)
model_output = hmm_model.update(observations)

# 4. Financial Analysis
from hidden_regime.analysis.financial import FinancialAnalysis
from hidden_regime.config.analysis import FinancialAnalysisConfig

analysis_config = FinancialAnalysisConfig()
financial_analysis = FinancialAnalysis(analysis_config)
analysis_results = financial_analysis.update(
    model_output,
    raw_data,
    model_component=hmm_model
)

# 5. Report Generation
from hidden_regime.reporting.financial import FinancialReportGenerator
from hidden_regime.config.reporting import FinancialReportConfig

report_config = FinancialReportConfig(output_dir="./output")
report_gen = FinancialReportGenerator(report_config)
report_gen.update(analysis_results, model_output, raw_data)
```

## 🔧 Common Configuration

### Data Configuration
```python
from hidden_regime.config.data import FinancialDataConfig

config = FinancialDataConfig(
    ticker="AAPL",
    start_date="2024-01-01",
    end_date="2024-12-31",
    source="yfinance",
    frequency="business_days"
)
```

### Model Configuration
```python
from hidden_regime.config.model import HMMConfig

config = HMMConfig(
    n_states=3,              # Number of regimes
    max_iterations=1000,     # EM algorithm iterations
    tolerance=1e-6,          # Convergence tolerance
    random_seed=42           # Reproducibility
)
```

### Analysis Configuration
```python
from hidden_regime.config.analysis import FinancialAnalysisConfig

config = FinancialAnalysisConfig(
    enable_trading_simulation=True,
    initial_capital=100000.0,
    position_sizing="regime_based",
    stop_loss_pct=0.05
)
```

## 📖 Key Concepts

### Regime Detection
The HMM identifies distinct market regimes based on return patterns:
- **Bearish**: Negative mean returns, high volatility
- **Sideways**: Near-zero returns, low volatility
- **Bullish**: Positive returns, moderate volatility
- **Crisis**: Extreme negative returns, extreme volatility

### Data-Driven Regime Labeling
Unlike naive approaches, regimes are characterized by **actual financial metrics**:
- Mean return and volatility per regime
- Win rate and duration statistics
- Risk-adjusted return ratios
- Maximum drawdown per regime

### Trading Simulation
Examples demonstrate realistic trading strategies:
- Regime-based position sizing
- Transaction cost modeling
- Stop-loss risk management
- Performance attribution by regime

## 🎓 Learning Path

**Beginners**: Start with examples 00 → 01 → 02

**Traders**: Focus on examples 03 → case_study_basic → financial_case_study

**Researchers**: Explore examples 04 → 05 → improved_features

**Production Users**: Study case_study_comprehensive for complete workflows

## ⚠️ Important Notes

1. **Data Requirements**: Most examples use yfinance for data. Internet connection required for real market data.

2. **Runtime**: Case study and multi-asset examples can take 2-3 minutes. Examples with 30-180s timeouts are expected.

3. **Output Directory**: Examples create `output/` directory automatically. Add to `.gitignore` if committing to version control.

4. **Model Component Parameter**: When using `FinancialAnalysis.update()`, always pass `model_component=hmm_model` for data-driven regime interpretation.

5. **Virtual Environment**: Always activate the environment before running examples.

## 🐛 Troubleshooting

### Data Loading Issues
```python
# Examples handle this gracefully with fallback to sample data
# Check internet connection if real market data fails
```

### Missing Dependencies
```bash
pip install yfinance pandas numpy matplotlib seaborn scipy
```

### Output Directory Permissions
```bash
chmod -R 755 output/
```

## 🌟 Next Steps

After running the examples:
1. Modify configurations for your specific assets
2. Experiment with different regime numbers (2-6 states)
3. Adjust position sizing and risk parameters
4. Integrate regime detection into your trading workflow
5. Explore the generated reports and visualizations

For API documentation, see the module READMEs in:
- `hidden_regime/data/README.md`
- `hidden_regime/models/README.md`
- `hidden_regime/analysis/README.md`
- `hidden_regime/reporting/README.md`

## 📝 Example Summary Table

| Example | Runtime | Complexity | Purpose |
|---------|---------|------------|---------|
| 00_basic_regime_detection.py | 30s | Beginner | Minimal working example |
| 01_real_market_analysis.py | 60s | Beginner | Real market data with fallbacks |
| 02_regime_comparison_analysis.py | 20s | Intermediate | Multi-asset comparison |
| 03_trading_strategy_demo.py | 15s | Intermediate | Trading strategy development |
| 04_multi_stock_comparative_study.py | 180s | Intermediate | Batch multi-ticker analysis |
| 05_advanced_analysis_showcase.py | 15s | Advanced | Complete pipeline showcase |
| improved_features.py | 120s | Advanced | Enhanced feature engineering |
| case_study.py | 30s | Intermediate | 4-phase case study system |
| case_study_basic.py | 30s | Beginner | Quick unified API example |
| case_study_comprehensive.py | 60s | Advanced | Full-featured analysis |
| case_study_multi_asset.py | 120s | Advanced | Multi-asset case studies |
| financial_case_study.py | 90s | Advanced | Financial-first architecture |

**Total Runtime**: ~10 minutes for all examples

---

*All examples tested and verified working on 2025-09-30*