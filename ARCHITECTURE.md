# Hidden Regime Architecture

**Version 2.0.0** - Pipeline-Based Market Regime Detection with Rigorous Separation of Concerns

---

## Table of Contents

1. [Overview](#overview)
2. [Architectural Principles](#architectural-principles)
3. [Core Architecture](#core-architecture)
4. [Separation of Concerns](#separation-of-concerns)
5. [Data Flow](#data-flow)
6. [Module Organization](#module-organization)
7. [How to Use the Library](#how-to-use-the-library)
8. [Extension Points](#extension-points)
9. [Design Patterns](#design-patterns)

---

## Overview

Hidden Regime is built on a **pipeline architecture** that ensures clean separation between mathematical modeling, data handling, and domain knowledge. The architecture enforces a strict principle: **models are pure mathematics, interpreters contain ALL financial domain knowledge, and data handlers manage only data operations**.

### Key Characteristics

- **Pipeline-Based**: All workflows follow Data → Observation → Model → Interpreter → Signal → Report
- **Temporally Safe**: Built-in V&V (Verification & Validation) isolation prevents look-ahead bias
- **Domain-Agnostic Core**: Mathematical models know nothing about finance
- **Extensible**: Add new domains (crypto, forex, commodities) without touching the model layer
- **Configuration-Driven**: Type-safe dataclass configurations for all components

---

## Architectural Principles

### Principle #1: Pipeline is THE Interface

**Users should ALWAYS use factory functions** to create pipelines, never instantiate components directly.

```python
# ✅ CORRECT: Use factory functions
import hidden_regime as hr
pipeline = hr.create_financial_pipeline('AAPL', n_states=3)

# ❌ AVOID: Direct instantiation (deprecated in v2.0.0)
from hidden_regime.data import FinancialDataLoader
loader = FinancialDataLoader(config)  # Shows FutureWarning
```

**Why?** Factories ensure:
- Consistent component wiring
- Proper configuration
- Future-proof against internal changes
- Easier upgrades

### Principle #2: MarketEventStudy = Convenience Wrapper

`MarketEventStudy` is a high-level API for train/test analysis that internally uses the pipeline architecture.

```python
# High-level API for market event analysis
study = hr.MarketEventStudy(
    ticker='QQQ',
    training_start='2018-01-01',
    training_end='2019-12-31',
    analysis_start='2020-01-01',
    analysis_end='2020-12-31',
    n_states=3
)
study.run()
```

### Principle #3: ALL Financial Knowledge in Interpreter

**The Interpreter module contains ALL financial domain concepts:**
- State → Label mapping (0 → "Bear", 1 → "Sideways", 2 → "Bull")
- Regime characterization (win rates, drawdowns, volatility)
- Financial statistics (Sharpe ratios, max drawdown, Sortino ratio)
- Market context interpretation

```python
# Interpreter translates model outputs to financial meaning
model_output = {'states': [0, 1, 2, 2, 1], ...}
interpreter_output = interpreter.update(model_output)
# Returns: {'regime_labels': {0: 'Bear', 1: 'Sideways', 2: 'Bull'}, ...}
```

### Principle #4: Model = Pure Mathematics ONLY

**The HMM model knows ONLY mathematics:**
- States: 0, 1, 2... (integers, not "bull"/"bear")
- Emission parameters: means, stds (Gaussian distributions)
- Transition probabilities (Markov chains)
- ❌ NO financial concepts like "bull", "bear", "crisis"

```python
# Model outputs are pure math
hmm_output = {
    'states': np.array([0, 1, 2, 2, 1]),           # Integer state sequence
    'emission_means': np.array([-0.01, 0.0, 0.02]), # Mathematical means
    'emission_stds': np.array([0.03, 0.01, 0.02]),  # Mathematical stds
    'transition_matrix': np.array([...]),           # Probability matrix
}
# No mention of "bull", "bear", or any financial terms!
```

---

## Core Architecture

### Component Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                         USER API LAYER                              │
│  • create_financial_pipeline()                                     │
│  • create_trading_pipeline()                                       │
│  • MarketEventStudy                                                │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ↓
┌────────────────────────────────────────────────────────────────────┐
│                       FACTORY LAYER                                 │
│  • PipelineFactory → creates complete pipelines                    │
│  • ComponentFactory → creates individual components                │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ↓
┌────────────────────────────────────────────────────────────────────┐
│                    PIPELINE ORCHESTRATION                           │
│  Pipeline.update() coordinates the complete data flow              │
└────────────────────────────┬───────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ↓                   ↓                   ↓
    ┌────────┐         ┌────────────┐      ┌──────────┐
    │  Data  │────────▶│Observation │─────▶│  Model   │
    │Component│         │ Component  │      │Component │
    └────────┘         └────────────┘      └──────────┘
         │                                       │
         │ OHLC prices                State predictions
         │ Timestamps                 Emission params
         │                            Transition matrix
         │                                       │
         ↓                                       ↓
    yfinance API                      ┌──────────────────┐
    Validation                        │   Interpreter    │
    Temporal slicing                  │   Component      │
                                      └────────┬─────────┘
                                               │
                                     Regime labels (Bull/Bear)
                                     Financial characteristics
                                     Performance metrics
                                               │
                                               ↓
                                      ┌────────────────────┐
                                      │ Signal Generator   │
                                      │    Component       │
                                      └────────┬───────────┘
                                               │
                                     Trading signals
                                     Position sizing
                                     Entry/exit rules
                                               │
                                               ↓
                                      ┌────────────────────┐
                                      │     Report         │
                                      │    Component       │
                                      └────────────────────┘
                                               │
                                      Markdown reports
                                      Visualizations
                                      Metrics
```

### Layer Responsibilities

#### User API Layer
- **Purpose**: Simple, intuitive interface for end users
- **Components**: Factory functions, MarketEventStudy
- **Principle**: Hide complexity, provide sensible defaults

#### Factory Layer
- **Purpose**: Component construction and wiring
- **Components**: PipelineFactory, ComponentFactory
- **Principle**: Enforce Principle #1 (Pipeline is THE interface)

#### Pipeline Orchestration Layer
- **Purpose**: Coordinate data flow between components
- **Components**: Pipeline class, TemporalController
- **Principle**: Single `.update()` call triggers entire flow

#### Component Layer
- **Purpose**: Modular, reusable processing units
- **Components**: Data, Observation, Model, Interpreter, Signal, Report
- **Principle**: Each component has ONE clear responsibility

---

## Separation of Concerns

### Layer 1: Data (Pure Data Operations)

**Location:** `hidden_regime/data/`

**Responsibility:** Load, validate, and manage price data

**Knowledge Boundaries:**
- ✅ OHLC data formats
- ✅ Date handling and validation
- ✅ Data quality assessment
- ✅ yfinance API integration
- ❌ NO mathematical models
- ❌ NO financial interpretations
- ❌ NO regime concepts

```python
# Data component outputs
{
    'price': pd.Series,      # Price data (OHLC avg or Close)
    'date': pd.DatetimeIndex,# Trading dates
    'volume': pd.Series,     # Trading volume (optional)
}
```

### Layer 2: Observations (Feature Engineering)

**Location:** `hidden_regime/observations/`

**Responsibility:** Transform raw data into model inputs

**Knowledge Boundaries:**
- ✅ Log return calculations
- ✅ Feature normalization
- ✅ Statistical transformations
- ✅ Rolling window operations
- ❌ NO model-specific logic
- ❌ NO financial interpretations
- ❌ NO regime labels

```python
# Observation component outputs
{
    'log_returns': np.ndarray,   # Log returns: ln(price_t / price_t-1)
    'features': np.ndarray,      # Additional features (optional)
    'dates': pd.DatetimeIndex,   # Aligned timestamps
}
```

### Layer 3: Models (Pure Mathematics)

**Location:** `hidden_regime/models/`

**Responsibility:** Mathematical state estimation

**Knowledge Boundaries:**
- ✅ Hidden Markov Model algorithms
- ✅ Baum-Welch training
- ✅ Viterbi inference
- ✅ Gaussian emissions
- ✅ Transition matrices
- ❌ NO financial terms ("bull", "bear", "crisis")
- ❌ NO regime interpretations
- ❌ NO trading logic

**Key Files:**
- `hmm.py` - Core HMM implementation (2,326 lines)
- `algorithms.py` - Baum-Welch, Viterbi, Forward-Backward
- `utils.py` - Parameter initialization, validation
- `multitimeframe.py` - Multi-timeframe regime detection

```python
# Model component outputs (PURE MATH)
{
    'states': np.ndarray,              # Integer states: [0, 1, 2, ...]
    'state_probabilities': np.ndarray, # P(state|data) for each time
    'emission_means': np.ndarray,      # μ₀, μ₁, μ₂, ... (mathematical means)
    'emission_stds': np.ndarray,       # σ₀, σ₁, σ₂, ... (mathematical stds)
    'transition_matrix': np.ndarray,   # A[i,j] = P(state_t+1=j | state_t=i)
    'log_likelihood': float,           # Model fit quality
}
```

### Layer 4: Interpreter (ALL Financial Domain Knowledge)

**Location:** `hidden_regime/interpreter/`

**Responsibility:** Translate mathematical outputs to financial meaning

**Knowledge Boundaries:**
- ✅ State → Regime label mapping
- ✅ Regime characterization (returns, volatility, drawdowns)
- ✅ Financial statistics (Sharpe, Sortino, Calmar)
- ✅ Win rates, persistence, regime strength
- ✅ Market context and interpretation
- ✅ RegimeType enum (Bear, Bull, Sideways, Crisis)
- ✅ RegimeProfile dataclass
- ❌ NO mathematical model logic
- ❌ NO trading signals (that's Signal Generator's job)

**Key Files:**
- `regime_types.py` - RegimeType enum, RegimeProfile dataclass, REGIME_TYPE_COLORS
- `financial.py` - FinancialInterpreter with comprehensive regime analysis
- `base.py` - BaseInterpreter interface

```python
# Interpreter component outputs (FINANCIAL DOMAIN)
{
    'regime_labels': {0: 'Bear', 1: 'Sideways', 2: 'Bull'},
    'current_regime': 'Bull',
    'confidence': 0.87,
    'regime_profiles': {
        0: RegimeProfile(
            regime_type=RegimeType.BEAR,
            mean_return=-0.015,
            volatility=0.030,
            win_rate=0.35,
            max_drawdown=-0.25,
            sharpe_ratio=-0.85,
            ...
        ),
        1: RegimeProfile(...),
        2: RegimeProfile(...),
    },
    'regime_strength': 0.72,
    'persistence_days': 12.5,
}
```

### Layer 5: Signal Generation (Trading Logic)

**Location:** `hidden_regime/signal_generation/`

**Responsibility:** Generate trading signals based on regime interpretation

**Knowledge Boundaries:**
- ✅ Position signals (long/short/neutral)
- ✅ Position sizing strategies
- ✅ Entry/exit rules
- ✅ Risk management logic
- ❌ NO regime interpretation (uses Interpreter outputs)
- ❌ NO model logic

```python
# Signal Generator outputs
{
    'signals': np.ndarray,       # 1=long, 0=neutral, -1=short
    'position_sizes': np.ndarray,# Position sizing based on regime
    'entry_dates': List[date],   # Entry points
    'exit_dates': List[date],    # Exit points
}
```

### Layer 6: Reporting (Output Generation)

**Location:** `hidden_regime/reports/`

**Responsibility:** Generate markdown reports and visualizations

**Knowledge Boundaries:**
- ✅ Markdown formatting
- ✅ Metric aggregation
- ✅ Visualization generation
- ❌ NO business logic
- ❌ NO computations (uses outputs from other components)

---

## Data Flow

### Complete Pipeline Flow

```
1. DATA LOADING
   FinancialDataLoader.update()
   ↓
   • Fetch OHLC data from yfinance
   • Validate data quality
   • Handle missing values
   • Apply temporal filtering (if backtesting)
   ↓
   OUTPUT: DataFrame with price, date, (volume)

2. OBSERVATION GENERATION
   FinancialObservationGenerator.update(data)
   ↓
   • Calculate log returns: ln(price_t / price_t-1)
   • Generate additional features (optional)
   • Normalize if needed
   ↓
   OUTPUT: np.ndarray of log returns

3. MODEL TRAINING/INFERENCE
   HiddenMarkovModel.fit(observations)      [First call only]
   HiddenMarkovModel.predict(observations)  [All calls]
   ↓
   • Initialize parameters (KMeans/Random/Custom)
   • Train HMM using Baum-Welch (EM algorithm)
   • Predict states using Viterbi algorithm
   • Compute emission parameters and transition matrix
   ↓
   OUTPUT: {states, emission_means, emission_stds, transition_matrix, ...}

4. INTERPRETATION
   FinancialInterpreter.update(model_output)
   ↓
   • Map states to regime labels (data-driven or heuristic)
     - State 0 (lowest mean) → Bear
     - State 1 (middle mean) → Sideways
     - State 2 (highest mean) → Bull
   • Characterize each regime:
     - Calculate win rate, max drawdown, Sharpe ratio
     - Compute persistence and regime strength
   • Assess current regime and confidence
   ↓
   OUTPUT: {regime_labels, regime_profiles, current_regime, confidence, ...}

5. SIGNAL GENERATION (Optional)
   FinancialSignalGenerator.update(interpreter_output)
   ↓
   • Generate trading signals based on regime
   • Size positions based on confidence and regime
   • Apply risk management rules
   ↓
   OUTPUT: {signals, position_sizes, entry_dates, exit_dates}

6. REPORTING (Optional)
   MarkdownReportGenerator.update(all_outputs)
   ↓
   • Aggregate metrics from all components
   • Create visualizations (price + regime overlay)
   • Format markdown report
   ↓
   OUTPUT: Markdown report string, saved plots
```

### Temporal Backtesting Flow (V&V Isolation)

```
TemporalController.step_through_time(start_date, end_date)
↓
FOR each date in [start_date, end_date]:
    1. Create TemporalDataStub with data up to date
       • ONLY historical data visible to pipeline
       • Prevents look-ahead bias

    2. Replace pipeline.data with stub

    3. pipeline.update()
       • Runs complete flow with historical data only
       • Model sees only past returns
       • No future information leakage

    4. Record results at this timestamp

    5. Move to next date
↓
OUTPUT: Time series of decisions with rigorous V&V
```

---

## Module Organization

### Directory Structure

```
hidden_regime/
├── analysis/                 # High-level analysis workflows
│   ├── market_event_study.py    # MarketEventStudy framework
│   ├── financial.py             # FinancialAnalysis (DEPRECATED name, legacy)
│   ├── performance.py           # Performance analytics
│   ├── indicator_comparison.py  # HMM vs technical indicators
│   └── technical_indicators.py  # Technical indicator calculations
│
├── config/                   # Configuration dataclasses (12 files)
│   ├── base.py                  # BaseConfig
│   ├── data.py                  # DataConfig, FinancialDataConfig
│   ├── model.py                 # ModelConfig, HMMConfig
│   ├── interpreter.py           # InterpreterConfig
│   ├── analysis.py              # AnalysisConfig (legacy)
│   └── ...
│
├── data/                     # Data loading and validation
│   ├── financial.py             # FinancialDataLoader (yfinance)
│   ├── collectors.py            # Data collectors for temporal analysis
│   ├── exporters.py             # Export utilities
│   └── session_pool.py          # Connection pooling
│
├── factories/                # Component and pipeline factories
│   ├── pipeline.py              # PipelineFactory + convenience functions
│   └── components.py            # ComponentFactory
│
├── interpreter/              # **NEW in v2.0.0** - ALL financial domain knowledge
│   ├── regime_types.py          # RegimeType, RegimeProfile, REGIME_TYPE_COLORS
│   ├── financial.py             # FinancialInterpreter
│   └── base.py                  # BaseInterpreter interface
│
├── models/                   # Pure mathematical HMM implementation
│   ├── hmm.py                   # HiddenMarkovModel (2,326 lines)
│   ├── algorithms.py            # Baum-Welch, Viterbi algorithms
│   ├── utils.py                 # Parameter initialization (960 lines)
│   ├── multitimeframe.py        # Multi-timeframe regime detection
│   └── config.py                # HMMConfig
│
├── observations/             # Feature generation from raw data
│   ├── base.py                  # BaseObservationGenerator
│   └── financial.py             # FinancialObservationGenerator
│
├── pipeline/                 # Core pipeline architecture
│   ├── core.py                  # Pipeline class
│   ├── interfaces.py            # Component interfaces
│   ├── temporal.py              # TemporalController, TemporalDataStub
│   └── schemas.py               # Output validation schemas
│
├── reports/                  # Report generation
│   ├── markdown.py              # MarkdownReportGenerator
│   ├── comprehensive.py         # Comprehensive reports
│   └── case_study.py            # Case study reports
│
├── signal_generation/        # **NEW in v2.0.0** - Trading signal logic
│   ├── base.py                  # BaseSignalGenerator
│   └── financial.py             # FinancialSignalGenerator
│
├── simulation/               # Trading backtesting and simulation
│   ├── trading_engine.py        # Trading simulation
│   ├── performance_analytics.py # Performance metrics
│   └── risk_management.py       # Risk management
│
├── utils/                    # Utilities and helpers
│   ├── exceptions.py            # Exception classes
│   ├── state_mapping.py         # State mapping utilities
│   └── ...
│
└── visualization/            # Plotting and charting
    ├── plotting.py              # Core plotting functions
    ├── advanced_plots.py        # Advanced visualizations
    ├── animations.py            # Animated regime evolution
    └── indicators.py            # Indicator comparison plots
```

### Module Relationships

```
┌─────────────────────────────────────────────────────────┐
│                     analysis/                            │
│  High-level workflows (MarketEventStudy, comparisons)   │
│  Uses: pipeline, interpreter, models, reports           │
└─────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────┐
│                    factories/                            │
│  Creates pipelines and components                       │
│  Uses: ALL modules (wires components together)          │
└─────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────┐
│                     pipeline/                            │
│  Orchestrates component flow                            │
│  Uses: Component interfaces only (polymorphism)         │
└─────────────────────────────────────────────────────────┘
                             ↓
         ┌───────────────────┼──────────────────────┐
         ↓                   ↓                      ↓
    ┌────────┐        ┌─────────────┐      ┌──────────────┐
    │  data/ │───────▶│observations/│─────▶│   models/    │
    └────────┘        └─────────────┘      └──────────────┘
         │                                          │
         │                                          ↓
         │                                  ┌──────────────┐
         │                                  │ interpreter/ │
         │                                  └──────────────┘
         │                                          │
         │                                          ↓
         │                                  ┌──────────────────┐
         │                                  │signal_generation/│
         │                                  └──────────────────┘
         │                                          │
         ↓                                          ↓
    ┌────────────────────────────────────────────────────┐
    │               reports/                              │
    │  (receives outputs from all components)            │
    └────────────────────────────────────────────────────┘
         │
         ↓
    ┌─────────────────┐
    │ visualization/  │
    │  (used by all)  │
    └─────────────────┘
```

---

## How to Use the Library

### Level 1: Simple Regime Detection (Quickstart)

**Use Case:** Identify current market regime with minimal configuration

```python
import hidden_regime as hr

# Create and run pipeline
pipeline = hr.create_simple_regime_pipeline('AAPL', n_states=3)
result = pipeline.update()
print(result)  # Shows current regime and confidence
```

**What Happens:**
1. Loads recent AAPL data from yfinance
2. Calculates log returns
3. Trains 3-state HMM
4. Identifies current regime
5. Returns summary string

### Level 2: Financial Analysis Pipeline

**Use Case:** Comprehensive regime analysis with performance metrics

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
print(result)  # Includes regime stats, performance, confidence

# Access detailed outputs
regime_labels = pipeline.component_outputs['interpreter']['regime_labels']
current_regime = pipeline.component_outputs['interpreter']['current_regime']
confidence = pipeline.component_outputs['interpreter']['confidence']
```

**What Happens:**
1. Loads SPY data for specified period
2. Calculates log returns and features
3. Trains HMM and predicts regimes
4. **Interpreter characterizes each regime** (NEW in v2.0.0)
   - Computes Sharpe ratio, win rate, drawdowns
   - Determines regime persistence
   - Calculates regime strength
5. Returns comprehensive analysis

### Level 3: Market Event Study

**Use Case:** Analyze regime behavior during market events (crashes, bubbles)

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

**What Happens:**
1. Trains HMM on 2018-2019 data
2. Steps through 2020 day-by-day (temporal isolation)
3. Creates snapshots at key events
4. Generates animated regime evolution
5. Exports detailed metrics

### Level 4: Trading Strategy Development

**Use Case:** Develop regime-based trading strategies

```python
import hidden_regime as hr

# Create trading pipeline
pipeline = hr.create_trading_pipeline('SPY', n_states=4, risk_adjustment=True)
result = pipeline.update()

# Access trading signals
signals = pipeline.component_outputs['signal_generator']['signals']
position_sizes = pipeline.component_outputs['signal_generator']['position_sizes']

# Backtest with temporal isolation
data = pipeline.data.get_all_data()
controller = hr.create_temporal_controller(pipeline, data)

results = controller.step_through_time('2023-01-01', '2023-12-31')
# Each result contains decisions made with ONLY historical data
```

### Level 5: Custom Pipeline Configuration

**Use Case:** Full control over all components

```python
from hidden_regime.config import (
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    InterpreterConfiguration,
)
from hidden_regime.factories import pipeline_factory

# Configure each component
data_config = FinancialDataConfig(
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    use_ohlc_average=True
)

obs_config = FinancialObservationConfig.create_default_financial()

model_config = HMMConfig(
    n_states=4,
    max_iterations=100,
    tolerance=1e-6,
    initialization_method='kmeans',
    random_seed=42
)

interpreter_config = InterpreterConfiguration(
    labeling_method='data_driven',  # or 'heuristic'
    min_regime_duration_days=3
)

# Create custom pipeline
pipeline = pipeline_factory.create_pipeline(
    data_config=data_config,
    observation_config=obs_config,
    model_config=model_config,
    interpreter_config=interpreter_config
)

result = pipeline.update()
```

### Level 6: Streaming Data Ingestion (QuantConnect, Real-Time APIs)

**Use Case:** Integrate with streaming data sources like QuantConnect LEAN engine or real-time market data feeds

**Key Principle:** Users pass only the latest data bar; the pipeline accumulates and processes internally through the mandatory financial pipeline.

```python
import pandas as pd
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.data.financial import FinancialDataLoader

# Create data component
config = FinancialDataConfig(ticker='AAPL')
data_loader = FinancialDataLoader(config)

# Simulate streaming data (e.g., from QuantConnect OnData callback)
# Each call represents one new bar arriving
new_bar = pd.DataFrame({
    'Open': [150.0],
    'High': [151.0],
    'Low': [149.5],
    'Close': [150.5],
    'Volume': [1000000]
}, index=pd.DatetimeIndex(['2024-01-15']))

# Ingest with public interface (NEW in v2.0.0)
pipeline.update(data=new_bar)

# Inspect accumulated data at any time
accumulated = pipeline.data.get_data()
print(f"Total bars: {len(accumulated)}")
print(f"Columns: {list(accumulated.columns)}")  # Includes mandatory pipeline columns
```

**What Happens Internally:**

1. **Validation**: Input DataFrame checked for OHLCV columns and DatetimeIndex
2. **Mandatory Pipeline**: New data processed through:
   - OHLC average calculation: `(O+H+L+C)/4`
   - Percentage change calculation
   - Log returns calculation
3. **Accumulation**: New data appended to internal accumulator (with duplicate detection)
4. **State Consistency**: Cache invalidated, `_last_data` updated
5. **Ready for Pipeline**: Accumulated data passed downstream to Observation component

**QuantConnect Example:**

```python
from hidden_regime.quantconnect import HiddenRegimeAlgorithm

class MyAlgorithm(HiddenRegimeAlgorithm):
    def OnData(self, data):
        # Get the latest bar for SPY
        spy_bar = data['SPY']

        if spy_bar is None:
            return

        # Convert to DataFrame format for pipeline
        bar_df = pd.DataFrame({
            'Open': [spy_bar.Open],
            'High': [spy_bar.High],
            'Low': [spy_bar.Low],
            'Close': [spy_bar.Close],
            'Volume': [spy_bar.Volume]
        }, index=pd.DatetimeIndex([spy_bar.Time]))

        # Update regime detection with latest bar
        # Pipeline internally accumulates and retrains on rolling window
        self.update_regime('SPY')  # Handles data ingestion internally
```

**Design Rationale:**

- **User-Friendly**: Users pass only new data; pipeline handles accumulation internally
- **Data Quality**: All data passes through mandatory financial pipeline
- **Transparent**: Users can inspect accumulated data with `pipeline.data.get_data()`
- **General-Purpose**: Works for any streaming source (QuantConnect, Alpaca, IB, custom APIs)
- **Efficient**: Only new data is processed; cache invalidation ensures correctness

---

## Extension Points

### Adding a New Domain (e.g., Cryptocurrency)

The architecture supports new domains without modifying core models:

**Step 1: Create Domain-Specific Interpreter**

```python
# hidden_regime/interpreter/crypto.py
from hidden_regime.interpreter.base import BaseInterpreter
from hidden_regime.interpreter.regime_types import RegimeType, RegimeProfile

class CryptoInterpreter(BaseInterpreter):
    """Interpreter for cryptocurrency markets."""

    def _map_states_to_regimes(self, states, emission_params):
        """Map HMM states to crypto-specific regime labels."""
        # Crypto has different characteristics than stocks
        # e.g., "FOMO", "Capitulation", "Accumulation", "Distribution"
        ...

    def _characterize_regime(self, regime_type, returns, volatility):
        """Characterize regimes for crypto markets."""
        # Crypto-specific metrics
        # e.g., on-chain metrics, hash rate, funding rates
        ...
```

**Step 2: Register in Factory**

```python
# hidden_regime/factories/components.py
from hidden_regime.interpreter.crypto import CryptoInterpreter

component_factory.register_interpreter('crypto', CryptoInterpreter)
```

**Step 3: Use in Pipeline**

```python
import hidden_regime as hr

pipeline = hr.create_pipeline(
    data_config=FinancialDataConfig(ticker='BTC-USD'),
    observation_config=FinancialObservationConfig.create_default_financial(),
    model_config=HMMConfig(n_states=4),  # Same model!
    interpreter_config=CryptoInterpreterConfig()  # New interpreter
)
```

**Key Point:** The HMM model remains unchanged - it still outputs states 0, 1, 2, 3. Only the interpreter changes to provide crypto-specific domain knowledge.

### Adding a New Data Source

**Step 1: Implement DataComponent Interface**

```python
# hidden_regime/data/alpha_vantage.py
from hidden_regime.pipeline.interfaces import DataComponent

class AlphaVantageDataLoader(DataComponent):
    def get_all_data(self):
        """Load data from Alpha Vantage API."""
        ...

    def update(self, current_date=None):
        """Update with temporal filtering."""
        ...

    def plot(self, **kwargs):
        """Visualize data."""
        ...
```

**Step 2: Register in Factory**

```python
component_factory.register_data_source('alpha_vantage', AlphaVantageDataLoader)
```

### Adding a New Model Type

**Step 1: Implement ModelComponent Interface**

```python
# hidden_regime/models/gaussian_mixture.py
from hidden_regime.pipeline.interfaces import ModelComponent

class GaussianMixtureModel(ModelComponent):
    def fit(self, observations):
        """Train GMM on observations."""
        ...

    def predict(self, observations):
        """Predict cluster assignments."""
        ...
```

**Step 2: Register in Factory**

```python
component_factory.register_model('gmm', GaussianMixtureModel)
```

---

## Design Patterns

### 1. Pipeline Pattern

Coordinates data flow through multiple processing stages.

```python
class Pipeline:
    def update(self, current_date=None):
        # Stage 1: Load data
        data = self.data.update(current_date)

        # Stage 2: Generate observations
        observations = self.observation.update(data)

        # Stage 3: Train/predict model
        self.model.fit(observations)
        model_output = self.model.predict(observations)

        # Stage 4: Interpret results
        interpretation = self.interpreter.update(model_output)

        # Stage 5: Generate signals (optional)
        if self.signal_generator:
            signals = self.signal_generator.update(interpretation)

        # Stage 6: Create report (optional)
        if self.report:
            report = self.report.update(...)

        return interpretation  # or full results dict
```

### 2. Factory Pattern

Encapsulates object creation logic.

```python
class PipelineFactory:
    def create_financial_pipeline(self, ticker, n_states, ...):
        # Create configs
        data_config = FinancialDataConfig(ticker=ticker, ...)
        model_config = HMMConfig(n_states=n_states, ...)
        interpreter_config = InterpreterConfiguration(...)

        # Create components
        data = component_factory.create_data_component(data_config)
        model = component_factory.create_model_component(model_config)
        interpreter = component_factory.create_interpreter_component(interpreter_config)

        # Wire together
        return Pipeline(data=data, model=model, interpreter=interpreter)
```

### 3. Strategy Pattern

Allows swapping algorithms without changing client code.

```python
# Different initialization strategies for HMM
config1 = HMMConfig(initialization_method='kmeans')
config2 = HMMConfig(initialization_method='random')
config3 = HMMConfig(initialization_method='custom', custom_emission_means=[...])

# Same model class, different behavior
hmm1 = HiddenMarkovModel(config=config1)
hmm2 = HiddenMarkovModel(config=config2)
hmm3 = HiddenMarkovModel(config=config3)
```

### 4. Template Method Pattern

Defines algorithm skeleton with customizable steps.

```python
class BaseInterpreter:
    def update(self, model_output):
        # Template method
        states = self._extract_states(model_output)
        regimes = self._map_states_to_regimes(states)  # Customizable
        profiles = self._characterize_regimes(regimes)  # Customizable
        return self._format_output(regimes, profiles)

    @abstractmethod
    def _map_states_to_regimes(self, states):
        """Subclass implements domain-specific mapping."""
        pass

    @abstractmethod
    def _characterize_regimes(self, regimes):
        """Subclass implements domain-specific characterization."""
        pass
```

### 5. Dependency Injection Pattern

Components receive dependencies through configuration.

```python
# Dependencies injected via config
data_component = FinancialDataLoader(
    config=FinancialDataConfig(
        ticker='AAPL',
        data_source='yfinance',  # Injected dependency
        session_pool=session_pool  # Injected dependency
    )
)
```

### 6. Facade Pattern

MarketEventStudy provides simple interface to complex subsystem.

```python
class MarketEventStudy:
    """Facade hiding pipeline complexity."""

    def run(self):
        # Internally uses:
        # - PipelineFactory
        # - TemporalController
        # - Multiple components
        # - Visualization
        # - Export utilities

        # User sees only:
        study.run()
```

---

## Backward Compatibility (v1.x → v2.0.0)

### What Changed in v2.0.0

1. **`AnalysisComponent` → `InterpreterComponent`**
   - Old `AnalysisComponent` interface DELETED
   - New `InterpreterComponent` interface introduced
   - Financial domain knowledge moved to `/interpreter/`

2. **`/financial/` module DELETED**
   - `RegimeType`, `RegimeProfile` moved to `/interpreter/regime_types.py`
   - Financial characterization logic moved to `/interpreter/financial.py`

3. **`FinancialAnalysis` now uses `InterpreterComponent`**
   - Name is confusing (should be "FinancialInterpreter") but kept for compatibility
   - Inherits from `InterpreterComponent`, not `AnalysisComponent`

### Migration Guide

**Old Code (v1.x):**
```python
from hidden_regime.financial import RegimeType, RegimeProfile
from hidden_regime.pipeline import AnalysisComponent

class MyAnalysis(AnalysisComponent):
    def update(self, **kwargs):
        ...
```

**New Code (v2.0.0):**
```python
from hidden_regime.interpreter import RegimeType, RegimeProfile
from hidden_regime.pipeline import InterpreterComponent

class MyAnalysis(InterpreterComponent):
    def update(self, model_output):
        ...
```

### Deprecation Warnings

v2.0.0 adds `FutureWarning` to direct component instantiation:

```python
# Shows FutureWarning in v2.0.0
from hidden_regime.models import HiddenMarkovModel
hmm = HiddenMarkovModel(n_states=3)
# Warning: "Direct instantiation deprecated. Use hr.create_financial_pipeline()"

# Recommended approach
import hidden_regime as hr
pipeline = hr.create_financial_pipeline('AAPL', n_states=3)
```

---

## Summary

Hidden Regime's architecture is built on four core principles:

1. **Pipeline is THE interface** - Always use factories
2. **MarketEventStudy for convenience** - High-level API for train/test analysis
3. **ALL financial knowledge in Interpreter** - Domain expertise centralized
4. **Model = Pure mathematics** - Domain-agnostic, reusable core

This separation enables:
- **Extensibility**: Add new domains (crypto, forex) without touching models
- **Testability**: Each layer can be tested independently
- **Reusability**: Same model works across different financial domains
- **Maintainability**: Clear boundaries make debugging easier
- **Temporal Safety**: Built-in V&V isolation prevents look-ahead bias

For usage examples, see:
- `README.md` - Quick start and feature overview
- Module READMEs (`hidden_regime/*/README.md`) - Detailed component docs
- `examples/` directory - Working code demonstrations
- `REFACTORING_PLAN.md` - v2.0.0 architectural changes

---

**Version:** 2.0.0
**Last Updated:** 2025-11-16
**Maintained by:** hidden-regime team
