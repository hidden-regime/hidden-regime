# Hidden Regime Examples

This directory contains comprehensive examples demonstrating the Hidden Regime package capabilities. Examples are organized by difficulty level and use case for easier navigation and learning.

## 📁 Directory Structure

```
examples/
├── quickstart/              # Beginner examples (5-60 seconds runtime)
├── intermediate/            # Intermediate examples (15-180 seconds)
├── advanced/                # Advanced topics (60-180 seconds)
├── case_studies/            # Real-world market analysis
├── portfolios/              # QuantConnect trading strategies
├── notebooks/               # Jupyter tutorials (🔒 BLOG-REFERENCED)
└── templates/               # Reusable templates
```

---

## 🚀 Quickstart Examples

**Perfect for:** First-time users, quick prototyping

### `minimal_10_lines.py` ✨ NEW
**Runtime:** ~5 seconds

The absolute simplest way to use hidden-regime. Detect market regimes in just 10 lines of code.

```bash
python examples/quickstart/minimal_10_lines.py
```

### `00_recommended_usage.py`
**Runtime:** ~30 seconds

Factory pattern example (v2.0+). Shows the **RECOMMENDED** way to create pipelines.

**Demonstrates:**
- `create_financial_pipeline()` factory method
- Deprecation-free pipeline creation
- Best practices for v2.0+

```bash
python examples/quickstart/00_recommended_usage.py
```

### `01_real_market_analysis.py`
**Runtime:** ~60 seconds

Regime detection on real market data with robust error handling.

**Demonstrates:**
- Loading real market data with fallbacks
- Regime change detection
- Professional analysis reports
- Publication-quality visualizations

```bash
python examples/quickstart/01_real_market_analysis.py
```

### `02_regime_comparison_analysis.py`
**Runtime:** ~20 seconds

Compare regime patterns across multiple assets.

**Demonstrates:**
- Multi-asset regime detection
- Synchronous regime changes
- Cross-market correlation
- Portfolio applications

```bash
python examples/quickstart/02_regime_comparison_analysis.py
```

---

## 🎯 Intermediate Examples

**Perfect for:** Building trading strategies, multi-asset analysis

### `03_trading_strategy_demo.py`
**Runtime:** ~15 seconds

Build practical trading strategies based on regime detection.

**Demonstrates:**
- Regime-based position sizing
- Entry/exit signal generation
- Performance metrics calculation
- Transaction cost modeling

```bash
python examples/intermediate/03_trading_strategy_demo.py
```

### `04_multi_stock_comparative_study.py`
**Runtime:** ~180 seconds

Comprehensive regime analysis across multiple stocks.

**Demonstrates:**
- Batch processing of multiple tickers
- Cross-stock regime correlation
- Sector-based comparative studies
- Market regime consensus

```bash
python examples/intermediate/04_multi_stock_comparative_study.py
```

### `05_advanced_analysis_showcase.py`
**Runtime:** ~15 seconds

Complete pipeline with all components.

**Demonstrates:**
- Full data → observation → model → analysis pipeline
- Comprehensive financial metrics
- Edge case handling
- Current API best practices

```bash
python examples/intermediate/05_advanced_analysis_showcase.py
```

---

## 🔬 Advanced Examples

**Perfect for:** Power users, researchers, custom implementations

### `initialization_methods_demo.py`
**Runtime:** ~90 seconds

Demonstrates the three HMM initialization approaches.

**Demonstrates:**
- **KMeans** (DEFAULT): Data-driven clustering
- **Random**: Quantile-based initialization
- **Custom**: User-specified parameters
- Transfer learning (SPY → AAPL)
- Convenience factory: `from_regime_specs()`

```bash
python examples/advanced/initialization_methods_demo.py
```

### `multi_timeframe_example.py`
**Runtime:** ~45 seconds

Multi-timeframe regime detection (daily, weekly, monthly).

**Demonstrates:**
- Independent HMM models per timeframe
- Alignment scoring (0.3 = misaligned, 1.0 = perfect)
- False signal filtering (~70% reduction)
- Confidence-based position sizing

```bash
python examples/advanced/multi_timeframe_example.py
```

### `improved_features.py`
**Runtime:** ~120 seconds

Enhanced feature engineering for improved regime detection.

**Demonstrates:**
- Momentum strength detection
- Trend persistence features
- Volatility context
- Directional consistency
- Baseline vs enhanced comparison

```bash
python examples/advanced/improved_features.py
```

### `financial_case_study.py`
**Runtime:** ~90 seconds

Financial-first architecture showcase.

**Demonstrates:**
- Data-driven regime characterization
- Intelligent signal generation
- Optimized single-asset position sizing
- Unified configuration and analysis

```bash
python examples/advanced/financial_case_study.py
```

### `regime_quality_validation.py` ✨ NEW
**Runtime:** ~120 seconds

Assess and validate regime detection quality.

**Demonstrates:**
- Quality metrics programmatic access
- Log-likelihood interpretation
- Regime persistence and duration
- Configuration comparison
- Quality issue identification

```bash
python examples/advanced/regime_quality_validation.py
```

---

## 📊 Case Studies

**Perfect for:** Real-world market analysis, research validation

### 🔒 BLOG-REFERENCED (DO NOT MODIFY)

These files are directly referenced on hiddenregime.com and **must remain unchanged**:

- ✅ `case_study_2008_financial_crisis.py`
- ✅ `case_study_dotcom_2000.py`
- ✅ `case_study_covid_2020.py`

### `case_study_2008_financial_crisis.py` 🔒
**Runtime:** ~180 seconds | **PROTECTED**

2008 Financial Crisis regime analysis.

**Period:** 2007-2009 (crisis emergence → collapse → recovery)
**Assets:** SPY, XLF, TLT, GLD
**Events:** Bear Stearns, Lehman, TARP, Market Trough, Fed QE

```bash
python examples/case_studies/case_study_2008_financial_crisis.py
```

### `case_study_dotcom_2000.py` 🔒
**Runtime:** ~180 seconds | **PROTECTED**

Dot-Com Bubble collapse analysis.

**Period:** 2000-2002 (bubble burst → capitulation)
**Assets:** QQQ, MSFT, INTC, CSCO, AMZN, SPY
**Events:** NASDAQ Peak, Initial Crash, 9/11, Bear Bottom

```bash
python examples/case_studies/case_study_dotcom_2000.py
```

### `case_study_covid_2020.py` 🔒
**Runtime:** ~180 seconds | **PROTECTED**

COVID-19 Crisis regime detection.

**Period:** 2020 (fastest market crash in history)
**Assets:** QQQ, CCL, WMT, AMZN, DIS, INTC
**Events:** Market Peak, WHO Pandemic, Fed Rate Cut, Bottom, CARES Act

```bash
python examples/case_studies/case_study_covid_2020.py
```

### Other Case Studies

#### `case_study_multi_asset.py`
**Runtime:** ~120 seconds

Comparative analysis across multiple assets.

```bash
python examples/case_studies/case_study_multi_asset.py
```

#### `market_event_study_tsla_2024.py`
**Runtime:** ~90 seconds

TSLA 2024 market event study with visualization pipeline.

```bash
python examples/case_studies/market_event_study_tsla_2024.py
```

#### Helper Scripts
- `analyze_2008_regime_statistics.py` - Extract 2008 regime stats
- `analyze_dotcom_regime_statistics.py` - Extract dotcom regime stats
- `analyze_timeframe_persistence.py` - Analyze timeframe persistence

---

## 💼 Portfolio Strategies (QuantConnect)

**Perfect for:** Live trading, QuantConnect backtesting

**Note:** These are QuantConnect algorithms, not standalone Python scripts. They require the QuantConnect environment to run.

### `quantconnect_simple_regime_following.py` ✨ NEW
**A minimal regime-following strategy for learning QuantConnect integration.**

**Strategy:** Long equities in Bull, defensive (bonds/gold) in Bear
**Expected Sharpe:** 0.9-1.2

### `all_weather_regime_portfolio.py`
**Ray Dalio-inspired all-weather portfolio with regime adaptation.**

**Assets:** SPY, QQQ, EFA, TLT, IEF, GLD, DBC, VNQ
**Expected Sharpe:** 1.2-1.5
**Max Drawdown:** <15%

**Strategy:**
- 4-state regime detection (Bull/Bear/Sideways/Crisis)
- Regime-adaptive allocation across asset classes
- Monthly rebalancing or on regime change

### `momentum_regime_rotation.py`
**Momentum rotation filtered by regime state.**

**Universe:** 10 sector ETFs
**Expected Sharpe:** 1.3-1.7
**Max Drawdown:** <20%

**Strategy:**
- Rank sectors by 6-month momentum
- Filter by regime (long only in Bull)
- Top 3 sectors, equal-weighted
- Defensive assets in Bear regime

### `volatility_targeting_regime.py`
**Constant volatility targeting with regime-adaptive targets.**

**Assets:** SPY, QQQ, IWM
**Target Vol:** 10% (regime-adjusted)
**Expected Sharpe:** 1.4-1.8

**Strategy:**
- Maintain constant 10% volatility
- Adjust leverage based on realized vol
- Lower vol target in Bear regimes
- Risk parity weighting

---

## 📓 Jupyter Notebooks

**Perfect for:** Interactive learning, mathematical foundations

### 🔒 ALL NOTEBOOKS ARE BLOG-REFERENCED (DO NOT MODIFY)

All 7 notebooks are directly referenced on hiddenregime.com and **must not be modified**:

- ✅ `01_Why_Log_Returns_Mathematical_Foundation.ipynb`
- ✅ `02_HMM_Basics_Understanding_Regime_Detection.ipynb`
- ✅ `03_Full_Pipeline_Advanced_Analysis.ipynb`
- ✅ `04_Advanced_Trading_Applications.ipynb`
- ✅ `05_multi_timeframe_regime_detection.ipynb`
- ✅ `HMM101.ipynb`
- ✅ `PipelineCheckout.ipynb`

### Notebooks Overview

**`01_Why_Log_Returns_Mathematical_Foundation.ipynb`** 🔒
Mathematical foundation for using log returns in regime detection.

**`02_HMM_Basics_Understanding_Regime_Detection.ipynb`** 🔒
Comprehensive introduction to Hidden Markov Models.

**`03_Full_Pipeline_Advanced_Analysis.ipynb`** 🔒
End-to-end pipeline walkthrough with advanced analysis.

**`04_Advanced_Trading_Applications.ipynb`** 🔒
Building production-ready trading strategies.

**`05_multi_timeframe_regime_detection.ipynb`** 🔒
Multi-timeframe analysis with visualization.

**`HMM101.ipynb`** 🔒
Beginner-friendly HMM tutorial.

**`PipelineCheckout.ipynb`** 🔒
Pipeline architecture and best practices.

---

## 📝 Templates

### `case_study_template.py`
Reusable template for creating custom market event case studies.

**Use for:**
- Analyzing any market event (crashes, bubbles, policy changes)
- Quick setup with MarketEventStudy API
- Customizable key event dates and snapshots

```bash
python examples/templates/case_study_template.py
```

---

## 🎓 Learning Paths

### Absolute Beginner
```
1. quickstart/minimal_10_lines.py           (5s)
2. quickstart/00_recommended_usage.py       (30s)
3. quickstart/01_real_market_analysis.py    (60s)
4. notebooks/HMM101.ipynb
```

### Traders & Portfolio Managers
```
1. quickstart/02_regime_comparison_analysis.py
2. intermediate/03_trading_strategy_demo.py
3. case_studies/case_study_covid_2020.py
4. portfolios/quantconnect_simple_regime_following.py
5. portfolios/all_weather_regime_portfolio.py
```

### Researchers & Quants
```
1. intermediate/04_multi_stock_comparative_study.py
2. advanced/initialization_methods_demo.py
3. advanced/multi_timeframe_example.py
4. advanced/regime_quality_validation.py
5. notebooks/01-04 (all mathematical foundations)
```

### QuantConnect Users
```
1. portfolios/quantconnect_simple_regime_following.py
2. portfolios/all_weather_regime_portfolio.py
3. portfolios/momentum_regime_rotation.py
4. portfolios/volatility_targeting_regime.py
```

---

## ⚡ Quick Commands

### Run all quickstart examples
```bash
for f in examples/quickstart/*.py; do
    echo "Running $f..."
    python "$f"
done
```

### Run all case studies
```bash
for f in examples/case_studies/case_study_*.py; do
    echo "Running $f..."
    python "$f"
done
```

### Run a specific example
```bash
python examples/quickstart/minimal_10_lines.py
python examples/case_studies/case_study_covid_2020.py
python examples/advanced/regime_quality_validation.py
```

---

## 📊 Example Summary Table

| Example | Category | Runtime | Complexity | New |
|---------|----------|---------|------------|-----|
| `minimal_10_lines.py` | Quickstart | 5s | Beginner | ✨ |
| `00_recommended_usage.py` | Quickstart | 30s | Beginner | |
| `01_real_market_analysis.py` | Quickstart | 60s | Beginner | |
| `02_regime_comparison_analysis.py` | Quickstart | 20s | Beginner | |
| `03_trading_strategy_demo.py` | Intermediate | 15s | Intermediate | |
| `04_multi_stock_comparative_study.py` | Intermediate | 180s | Intermediate | |
| `05_advanced_analysis_showcase.py` | Intermediate | 15s | Intermediate | |
| `initialization_methods_demo.py` | Advanced | 90s | Advanced | |
| `multi_timeframe_example.py` | Advanced | 45s | Advanced | |
| `improved_features.py` | Advanced | 120s | Advanced | |
| `financial_case_study.py` | Advanced | 90s | Advanced | |
| `regime_quality_validation.py` | Advanced | 120s | Advanced | ✨ |
| `case_study_2008_financial_crisis.py` 🔒 | Case Study | 180s | Intermediate | |
| `case_study_dotcom_2000.py` 🔒 | Case Study | 180s | Intermediate | |
| `case_study_covid_2020.py` 🔒 | Case Study | 180s | Intermediate | |
| `case_study_multi_asset.py` | Case Study | 120s | Advanced | |
| `market_event_study_tsla_2024.py` | Case Study | 90s | Intermediate | |
| `quantconnect_simple_regime_following.py` | Portfolio | N/A | Beginner | ✨ |
| `all_weather_regime_portfolio.py` | Portfolio | N/A | Advanced | |
| `momentum_regime_rotation.py` | Portfolio | N/A | Advanced | |
| `volatility_targeting_regime.py` | Portfolio | N/A | Advanced | |

**Legend:**
🔒 = Blog-referenced (DO NOT MODIFY)
✨ = New in this update

---

## 🏭 Factory Pattern (REQUIRED)

**All examples MUST use the factory pattern for component creation.** Direct instantiation is deprecated and will trigger architectural compliance failures.

### ✅ Correct (Factory Pattern)

```python
import hidden_regime as hr
from hidden_regime.factories import component_factory
from hidden_regime.config.data import FinancialDataConfig

# RECOMMENDED: Use high-level factory functions
pipeline = hr.create_financial_pipeline('SPY', n_states=3)

# OR: Use component factory for fine-grained control
data_config = FinancialDataConfig(ticker='SPY')
data_loader = component_factory.create_data_component(data_config)
```

### ❌ Incorrect (Direct Instantiation - DEPRECATED)

```python
# DON'T DO THIS - Will fail architectural tests
from hidden_regime.data.financial import FinancialDataLoader
from hidden_regime.models.hmm import HiddenMarkovModel

data_loader = FinancialDataLoader(data_config)  # ❌ WRONG
model = HiddenMarkovModel(model_config)        # ❌ WRONG
```

### Why Factory Pattern?

1. **Deprecation Safety**: Direct instantiation will show `FutureWarning`
2. **Architectural Compliance**: Examples must pass `test_factory_pattern.py`
3. **Best Practices**: Follows documented v2.0 architecture
4. **Future-Proof**: Easier migration when internals change

### Available Factories

**High-Level Pipeline Factories** (Recommended):
- `hr.create_simple_regime_pipeline()` - Basic regime detection
- `hr.create_financial_pipeline()` - Full financial analysis
- `hr.create_trading_pipeline()` - Trading strategies with signals
- `hr.create_research_pipeline()` - Academic research features
- `hr.create_temporal_controller()` - V&V backtesting

**Component-Level Factories** (Advanced):
```python
from hidden_regime.factories import component_factory

data = component_factory.create_data_component(data_config)
obs = component_factory.create_observation_component(obs_config)
model = component_factory.create_model_component(model_config)
interp = component_factory.create_interpreter_component(interp_config)
```

---

## ⚠️ Important Notes

### Blog-Referenced Files (Protected)

The following files are directly referenced on hiddenregime.com and **MUST NOT BE MODIFIED**:

**Case Studies:**
- `case_studies/case_study_2008_financial_crisis.py`
- `case_studies/case_study_dotcom_2000.py`
- `case_studies/case_study_covid_2020.py`

**Notebooks:**
- All 7 notebooks in `notebooks/` directory

Any changes to these files will break blog links and tutorials.

### Data Requirements

Most examples use yfinance for data. Internet connection required for real market data. Examples gracefully fall back to sample data when network is unavailable.

### Output Directory

Examples create `output/` directory automatically. Add to `.gitignore` if committing to version control.

### Runtime Estimates

Runtime estimates are approximate and depend on:
- Network speed (for data downloads)
- CPU performance
- Number of assets analyzed

Case studies can take 2-3 minutes for full analysis.

### QuantConnect Examples

Portfolio strategies in `portfolios/` are QuantConnect algorithms. They:
- Require QuantConnect environment to run
- Cannot be executed as standalone Python scripts
- Serve as templates for QuantConnect backtesting

---

## 🐛 Troubleshooting

### Data Loading Issues
```python
# Examples handle this gracefully with fallback to sample data
# Check internet connection if real market data fails
```

### Missing Dependencies
```bash
pip install hidden-regime yfinance pandas numpy matplotlib seaborn scipy
```

### Output Directory Permissions
```bash
chmod -R 755 output/
```

---

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

---

## 📖 Additional Resources

- **Main Documentation:** See root `README.md`
- **API Reference:** `API_REFERENCE.md`
- **QuantConnect Guide:** `quantconnect_templates/README.md`
- **Best Practices:** `BEST_PRACTICES_AND_FAQ.md`
- **Blog & Tutorials:** https://hiddenregime.com

---

*Last Updated: 2025-11-17*
*Examples reorganized into subdirectories for better navigation*
*All examples tested and verified working*
