# QuantConnect Integration: Market Regime Detection

This module integrates Hidden Regime's HMM-based market regime detection with QuantConnect's LEAN backtesting and live trading platform. It enables building sophisticated trading strategies that adapt to market regimes with zero lookahead bias.

**Table of Contents**
- [What is QuantConnect?](#what-is-quantconnect)
- [Quick Start](#quick-start)
- [How It Works: Architecture Overview](#how-it-works-architecture-overview)
- [The Data Flow: From Bars to Trades](#the-data-flow-from-bars-to-trades)
- [Core Components](#core-components)
- [Configuration Guide](#configuration-guide)
- [Signal Generation](#signal-generation)
- [Available Strategies](#available-strategies)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## What is QuantConnect?

QuantConnect is a cloud-based algorithmic trading platform that provides:

- **LEAN Engine**: A C# backtesting engine that simulates real market execution
- **Extensive Data**: Historical price data for stocks, crypto, forex, and options
- **Live Trading**: Real portfolio execution through broker integration
- **Algorithm Lifecycle**: An event-driven architecture for handling market data

### Algorithm Lifecycle

Every QuantConnect algorithm follows this pattern:

```
1. Initialize()        - Run once at start to configure algorithm
2. OnData(data) loop   - Called for each bar (daily, hourly, minute, etc.)
3. SetHoldings()       - Adjust positions based on signals
```

### Key QuantConnect Concepts

**Symbol**: Identifier for a security (e.g., "SPY" for SPDR S&P 500 ETF)

**TradeBar**: A single candlestick containing OHLCV data (Open, High, Low, Close, Volume)

**OnData(data)**: Called when new market data arrives. `data[symbol]` returns the latest TradeBar.

**SetHoldings(symbol, allocation)**: Sets target position:
- `1.0` = 100% long (all capital in this position)
- `0.5` = 50% long (half capital in position)
- `0.0` = No position (cash)
- `-0.5` = 50% short (borrowed position)

**Resolution**: Data frequency - `Resolution.Daily`, `Resolution.Hour`, `Resolution.Minute`

---

## Quick Start

Get a regime-based strategy running in 5 minutes:

```python
from AlgorithmImports import *
from hidden_regime.quantconnect import HiddenRegimeAlgorithm

class MyFirstStrategy(HiddenRegimeAlgorithm):
    """Buy in bull regime, hold cash in bear regime."""

    def Initialize(self):
        # Backtest period
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Add SPY to portfolio
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Initialize regime detection
        self.initialize_regime_detection(
            ticker="SPY",
            n_states=3,                    # 3-state HMM
            lookback_days=252,             # 1 year history
            retrain_frequency="weekly",    # Retrain weekly
            regime_allocations={
                "Bull": 1.0,      # 100% long
                "Bear": 0.0,      # Cash
                "Sideways": 0.5,  # 50% long
            }
        )

    def OnData(self, data):
        # Skip if no data for SPY
        if not data.ContainsKey(self.symbol):
            return

        # Feed new price bar to regime detector
        bar = data[self.symbol]
        self.on_tradebar("SPY", bar)

        # Wait for enough data
        if not self.regime_is_ready():
            return

        # Update regime detection
        self.update_regime()

        # Get target allocation for current regime
        allocation = self.get_regime_allocation("SPY")

        # Execute trade to reach target
        self.SetHoldings(self.symbol, allocation)

    def on_regime_change(self, old_regime, new_regime, confidence, ticker):
        """Called when regime transitions."""
        self.Log(f"Regime: {old_regime} → {new_regime} ({confidence:.0%})")
```

**Key methods:**
- `initialize_regime_detection()` - Setup regime detection
- `on_tradebar(ticker, bar)` - Feed new price data
- `regime_is_ready()` - Check if enough data collected
- `update_regime()` - Run HMM and detect current regime
- `get_regime_allocation(ticker)` - Get recommended position size
- `on_regime_change()` - Override to handle transitions

---

## How It Works: Architecture Overview

The system connects four major components:

```
QuantConnect LEAN Engine
    ↓
    │ OnData(data) provides TradeBar
    │
    ↓
QuantConnectDataAdapter
    │ Buffers bars → pandas DataFrame (OHLCV)
    │
    ↓
Hidden-Regime Pipeline
    │ Data → Observation → Model (HMM) → Interpreter
    │ Outputs: regime_name, state, confidence
    │
    ↓
RegimeSignalAdapter
    │ Regime → TradingSignal (direction, allocation)
    │
    ↓
HiddenRegimeAlgorithm.SetHoldings()
    │ Execute position size
    ↓
Broker Execution
    │ Real/simulated fills
    ↓
Portfolio Update
```

### Component Relationships

```
HiddenRegimeAlgorithm
├── Manages: QuantConnectDataAdapter (data buffering)
├── Manages: Hidden-Regime Pipeline (regime detection)
├── Manages: RegimeSignalAdapter (signal generation)
└── Calls: SetHoldings() (position execution)
```

### Data Flow

1. **Price Reception**: OnData() receives `TradeBar` with OHLCV
2. **Buffering**: `QuantConnectDataAdapter` accumulates bars in memory
3. **DataFrame**: Adapter converts buffer to pandas DataFrame
4. **Pipeline Injection**: DataFrame injected into hidden-regime pipeline
5. **Regime Detection**: HMM trains/infers regime from price patterns
6. **Interpretation**: Interpreter maps HMM state to regime name (Bull/Bear/etc.)
7. **Signal Generation**: Regime → TradingSignal (what to buy/sell/hold)
8. **Position Execution**: Signal → SetHoldings() call to broker

---

## The Data Flow: From Bars to Trades

Here's a detailed 6-step walkthrough:

### Step 1: Bar Buffering

When `OnData(data)` is called with a new TradeBar:

```python
def OnData(self, data):
    bar = data[self.symbol]  # TradeBar: {O, H, L, C, V}
    self.on_tradebar("SPY", bar)  # Buffer it
```

The adapter stores the bar in a rolling window (last 252 days by default).

### Step 2: DataFrame Creation

The adapter converts the buffer to a DataFrame:

```
Date        Open    High    Low     Close   Volume
2024-01-01  410.5   415.2   410.0   414.8   52.3M
2024-01-02  414.8   418.1   413.2   416.5   48.1M
...
```

### Step 3: Pipeline Injection

The adapter's DataFrame is injected into the hidden-regime pipeline:

```python
pipeline = hr.create_financial_pipeline(ticker="SPY", n_states=3)
pipeline.data._data = df  # Inject data
```

### Step 4: Regime Detection

The pipeline runs through:

- **Observation**: Transform Close prices → log returns
- **Model**: HMM trains on log returns, infers hidden states
- **Interpreter**: Maps states to financial regimes

Example:
```
HMM State 0: Low returns, high volatility → "Bear"
HMM State 1: High returns, low volatility → "Bull"
HMM State 2: Medium returns → "Sideways"
```

### Step 5: Signal Generation

The interpreter output is converted to a TradingSignal:

```python
signal = TradingSignal(
    direction=SignalDirection.LONG,
    strength=SignalStrength.STRONG,
    allocation=1.0,           # 100% long
    confidence=0.85,          # 85% confident
    regime_name="Bull",
    regime_state=1,
    timestamp=pd.Timestamp.now()
)
```

### Step 6: Position Execution

The signal determines position size:

```python
allocation = self.get_regime_allocation("SPY")  # Returns 1.0 for Bull
self.SetHoldings(self.symbol, allocation)       # Buy 100% SPY
```

---

## Core Components

### HiddenRegimeAlgorithm

The base class for all regime-based strategies. Extends QuantConnect's `QCAlgorithm`.

**Key Methods:**

```python
def initialize_regime_detection(
    ticker: str,              # Security symbol ("SPY")
    n_states: int = 3,        # Number of regimes to detect
    lookback_days: int = 252, # Historical data window
    retrain_frequency: str = "weekly",  # 'daily', 'weekly', 'monthly'
    regime_allocations: Dict[str, float] = None,  # Regime→position mapping
    min_confidence: float = 0.0          # Confidence threshold
)
```

Initializes regime detection for a ticker.

```python
def on_tradebar(ticker: str, bar: TradeBar) -> None
```

Called from `OnData()` to buffer new price data.

```python
def regime_is_ready(ticker: str = None) -> bool
```

Returns `True` once enough data is collected (default: 252 bars).

```python
def update_regime(ticker: str = None) -> None
```

Runs regime detection pipeline, updates `current_regime`, `regime_confidence`, `regime_state`.

```python
def get_regime_allocation(ticker: str) -> float
```

Returns recommended portfolio allocation for current regime.

```python
def on_regime_change(
    old_regime: str,
    new_regime: str,
    confidence: float,
    ticker: str
) -> None
```

Override this method to handle regime transitions (e.g., log events, adjust risk).

**State Management:**

```python
self.current_regime      # Regime name ("Bull", "Bear", "Sideways", "Crisis")
self.regime_confidence   # Confidence 0.0-1.0
self.regime_state        # Numeric HMM state (0, 1, 2, ...)
```

### QuantConnectDataAdapter

Converts QuantConnect price data to pandas DataFrames for hidden-regime processing.

**What it does:**
- Buffers incoming TradeBar data
- Maintains a rolling window (default: 252 days)
- Converts to DataFrame format: `{Date, Open, High, Low, Close, Volume}`
- Validates data (drops NaNs, handles gaps)

**Use case:**

```python
adapter = QuantConnectDataAdapter(lookback_days=252)
adapter.add_bar(timestamp, open_price, high, low, close, volume)
df = adapter.to_dataframe()  # Returns clean pandas DataFrame
```

### TradingSignal & RegimeSignalAdapter

`TradingSignal` encapsulates a complete trading signal from regime detection:

```python
@dataclass
class TradingSignal:
    direction: SignalDirection        # LONG, SHORT, NEUTRAL
    strength: SignalStrength          # STRONG, MODERATE, WEAK, NONE
    allocation: float                 # Portfolio position size (0-1 or negative)
    confidence: float                 # Regime confidence (0-1)
    regime_name: str                  # "Bull", "Bear", "Sideways", "Crisis"
    regime_state: int                 # HMM state ID
    timestamp: pd.Timestamp           # When signal generated
    metadata: Dict                    # Additional info
```

**Methods:**

```python
signal.is_actionable(min_confidence=0.6)        # Is confidence high enough?
signal.should_rebalance(current_allocation=0.5, threshold=0.1)  # Rebalance?
```

`RegimeSignalAdapter` generates signals from pipeline output:

```python
adapter = RegimeSignalAdapter(regime_allocations={"Bull": 1.0, "Bear": 0.0})
signal = adapter.generate_signal(
    regime_name="Bull",
    regime_state=1,
    confidence=0.85,
    timestamp=pd.Timestamp.now()
)
```

### Configuration Classes

**QuantConnectConfig** - QC-specific settings:

```python
from hidden_regime.quantconnect import QuantConnectConfig

config = QuantConnectConfig(
    lookback_days=252,          # Historical window
    retrain_frequency="weekly", # 'daily', 'weekly', 'monthly', 'never'
    warmup_days=252,            # Algorithm warm-up period
    use_cache=True,             # Cache trained models
    log_regime_changes=True,    # Log transitions
    min_confidence=0.0          # Minimum confidence for signals
)
```

**RegimeTradingConfig** - Regime-to-allocation mapping:

```python
from hidden_regime.quantconnect import RegimeTradingConfig

config = RegimeTradingConfig(
    regime_allocations={
        "Bull": 1.0,        # 100% long
        "Bear": 0.0,        # 0% (cash)
        "Sideways": 0.5,    # 50% long
        "Crisis": 0.0,      # 0% (cash)
    },
    rebalance_threshold=0.1,   # 10% change triggers rebalance
    use_risk_parity=False,     # Ignore for single-ticker
    max_leverage=1.0,          # Max leverage allowed
    cash_regime="Bear"         # "Bear" regime → move to cash
)

# Factory methods for common configurations
config = RegimeTradingConfig.create_conservative()  # Bull:0.6, Bear:0, Crisis:0
config = RegimeTradingConfig.create_aggressive()    # Bull:1.0, Bear:0.2, Crisis:0
config = RegimeTradingConfig.create_market_neutral() # Bull:0.5, Bear:-0.5 (long/short)
```

---

## Configuration Guide

### Setting Up Regime Allocations

The `regime_allocations` dict maps regime names to portfolio positions:

```python
regime_allocations = {
    "Bull": 1.0,      # 100% long - maximizes upside in bull markets
    "Bear": 0.0,      # 0% (cash) - avoids losses in bear markets
    "Sideways": 0.5,  # 50% long - medium risk in uncertain markets
    "Crisis": 0.0,    # 0% (cash) - defensive in crisis
}
```

### Choosing State Count (n_states)

- **3 states** (recommended): Bull, Bear, Sideways - simplest, most interpretable
- **4 states**: Bull, Bear, Sideways, Crisis - detect market crashes
- **2 states**: Up, Down - extreme simplicity
- **5 states**: Fine-grained regime structure - requires more data

Rule of thumb: Use `n_states = 3` unless you have specific needs.

### Lookback Window (lookback_days)

Determines how much historical data the HMM sees:

- **252 days** (1 year, default): Captures full market cycle, slow to adapt
- **126 days** (6 months): Faster adaptation, less stable
- **504 days** (2 years): Very stable, slow to detect regime change
- **63 days** (3 months): Very reactive, choppy

Use **252 days** for most cases.

### Retraining Frequency (retrain_frequency)

How often to retrain the HMM:

- **"weekly"** (default): Good balance of stability and adaptation
- **"daily"**: Responds quickly to new data, but may overfit
- **"monthly"**: Very stable, but slow to detect changes
- **"never"**: Train once, freeze parameters (not recommended)

Use **"weekly"** for most cases.

### Confidence Threshold (min_confidence)

Only trade when regime confidence exceeds this threshold:

```python
initialize_regime_detection(
    ticker="SPY",
    min_confidence=0.6  # Only trade when 60%+ confident
)

# In OnData, check confidence before trading
if self.regime_is_ready() and self.regime_confidence >= 0.6:
    self.update_regime()
    allocation = self.get_regime_allocation("SPY")
    self.SetHoldings(self.symbol, allocation)
```

---

## Signal Generation

### Understanding TradingSignal Structure

A TradingSignal contains:

```
direction    - LONG (1), NEUTRAL (0), or SHORT (-1)
strength     - STRONG (3), MODERATE (2), WEAK (1), NONE (0)
allocation   - Portfolio position size
confidence   - Regime confidence (0-1)
regime_name  - Regime label ("Bull", "Bear", etc.)
regime_state - HMM state number
timestamp    - When signal generated
metadata     - Additional context (dict)
```

### Signal Direction Rules

Based on regime allocations:

- **allocation > 0.1** → LONG (buy)
- **allocation < -0.1** → SHORT (sell short)
- **-0.1 ≤ allocation ≤ 0.1** → NEUTRAL (no position)

### Signal Strength Mapping

Derived from confidence:

- **confidence ≥ 0.9** → STRONG
- **0.7 ≤ confidence < 0.9** → MODERATE
- **0.5 ≤ confidence < 0.7** → WEAK
- **confidence < 0.5** → NONE

### Filtering by Confidence

Only act on high-confidence signals:

```python
def OnData(self, data):
    self.on_tradebar("SPY", data[self.symbol])

    if not self.regime_is_ready():
        return

    self.update_regime()

    # Only trade if confident
    signal = self.get_signal("SPY")  # Returns TradingSignal
    if signal.is_actionable(min_confidence=0.7):
        self.SetHoldings(self.symbol, signal.allocation)
```

---

## Available Strategies

Nine complete templates demonstrate different strategy patterns:

| File | Complexity | Strategy |
|------|-----------|----------|
| `basic_regime_switching.py` | Beginner | Single asset: Bull→100%, Bear→0%, Sideways→50% |
| `multi_asset_rotation.py` | Intermediate | Rotate among SPY, QQQ, TLT, GLD by regime |
| `crisis_detection.py` | Intermediate | Detect crisis regime, move to bonds |
| `dynamic_position_sizing.py` | Intermediate | Scale position size by confidence |
| `sector_rotation.py` | Advanced | Rotate among sectors (XLK, XLF, XLV, etc.) based on regime |
| `regime_deterioration_short.py` | Advanced | Short when regime deteriorates from Bull→Bear |
| `framework_example.py` | Advanced | Use QC Framework (alpha models, portfolio construction) |
| `optimized_multi_asset.py` | Advanced | Multi-asset with caching, batch updates, profiling |

### When to Use Each Template

**Basic Regime Switching**: Start here if new to regime trading. Simplest logic.

**Multi-Asset Rotation**: Multiple stocks/ETFs, rotate to best regime.

**Crisis Detection**: Defensive strategy, focus on downside protection.

**Dynamic Position Sizing**: Scale position size with confidence (size down in uncertain markets).

**Sector Rotation**: Rotate among industry sectors (tech, finance, healthcare, etc.).

**Regime Deterioration Short**: Aggressive strategy, short when Bull→Bear transition detected.

**Framework Example**: Use QC's alpha, portfolio construction, and risk management modules.

**Optimized Multi-Asset**: Large portfolio (10+ assets), enable caching and parallel updates.

---

## Performance Optimization

For single-asset strategies, performance is typically not an issue. For multi-asset portfolios, use these features:

### Caching Trained Models

Avoid retraining the same HMM repeatedly:

```python
from hidden_regime.quantconnect import HiddenRegimeAlgorithmOptimized

class MyStrategy(HiddenRegimeAlgorithmOptimized):
    def Initialize(self):
        # Enable model caching
        self.enable_caching(max_cache_size=100, retrain_frequency="weekly")
```

**Benefits:**
- First training takes 2-5 seconds per asset
- Cached lookups take < 0.01 seconds
- Huge speedup for multi-asset strategies

### Parallel Updates for Multiple Assets

Update regimes for multiple assets in parallel:

```python
from hidden_regime.quantconnect import HiddenRegimeAlgorithmOptimized

class MyStrategy(HiddenRegimeAlgorithmOptimized):
    def Initialize(self):
        # Enable parallel batch updates
        self.enable_batch_updates(max_workers=4, use_parallel=True)

        # Add many assets
        for ticker in ["SPY", "QQQ", "TLT", "GLD", "EEM", "DBC"]:
            self.initialize_regime_detection(ticker, n_states=3)
```

**Benefits:**
- 4 assets with 4 workers: ~4x speedup
- Automatic fallback to sequential if threading fails

### Performance Profiling

Identify bottlenecks:

```python
from hidden_regime.quantconnect import HiddenRegimeAlgorithmOptimized

class MyStrategy(HiddenRegimeAlgorithmOptimized):
    def Initialize(self):
        self.enable_profiling()  # Track timing of operations

    def OnEndOfAlgorithm(self):
        # Print timing report
        self.profiler.print_summary()
```

Output shows which operations consume most time (data loading, HMM training, signal generation, etc.).

---

## Best Practices

### 1. Always Call regime_is_ready() Before Accessing Regime

```python
def OnData(self, data):
    self.on_tradebar("SPY", data[self.symbol])

    # WRONG: Will crash if < 252 bars
    # allocation = self.get_regime_allocation("SPY")

    # RIGHT: Check readiness first
    if self.regime_is_ready():
        allocation = self.get_regime_allocation("SPY")
        self.SetHoldings(self.symbol, allocation)
```

### 2. Use Confidence Thresholds

Don't blindly follow every regime signal:

```python
if self.regime_is_ready() and self.regime_confidence >= 0.7:
    self.update_regime()
    # Only trade if confident
```

### 3. Handle Regime Changes Gracefully

Override `on_regime_change()` for important events:

```python
def on_regime_change(self, old_regime, new_regime, confidence, ticker):
    self.Log(f"ALERT: {ticker} {old_regime} → {new_regime}")

    # Cancel pending orders on regime transition
    self.Transactions.CancelOpenOrders(self.symbol)

    # Add slippage buffer for immediate rebalance
    self.SetHoldings(self.symbol, self.get_regime_allocation(ticker))
```

### 4. Use Appropriate Lookback Window

- **Short-term trading (daily)**: 126 days (6 months)
- **Medium-term (weekly signals)**: 252 days (1 year, default)
- **Long-term (monthly signals)**: 504 days (2 years)

### 5. Monitor Confidence Evolution

Low confidence often precedes regime changes:

```python
def OnData(self, data):
    self.on_tradebar("SPY", data[self.symbol])

    if not self.regime_is_ready():
        return

    self.update_regime()

    # Log confidence trend
    self.Log(f"Confidence: {self.regime_confidence:.1%}")

    # Hold back on trading if confidence < 0.5
    if self.regime_confidence < 0.5:
        self.SetHoldings(self.symbol, 0.25)  # Reduce to 25% as precaution
    else:
        allocation = self.get_regime_allocation("SPY")
        self.SetHoldings(self.symbol, allocation)
```

### 6. Avoid Overtraining

Don't retrain too frequently:

```python
# GOOD: Retrain weekly, stable
self.initialize_regime_detection(
    ticker="SPY",
    retrain_frequency="weekly"
)

# OKAY: Retrain daily if very responsive needed
self.initialize_regime_detection(
    ticker="SPY",
    retrain_frequency="daily"
)

# AVOID: Retrain every bar, overfits to noise
self.initialize_regime_detection(
    ticker="SPY",
    retrain_frequency="every_bar"  # Don't do this
)
```

### 7. Use Appropriate State Count

```python
# Most cases: 3 states (Bull, Bear, Sideways)
self.initialize_regime_detection(ticker="SPY", n_states=3)

# Crisis detection: 4 states
self.initialize_regime_detection(ticker="SPY", n_states=4)

# Avoid: Too many states overfit, too few miss structure
# n_states should be 2-5
```

---

## API Reference

### HiddenRegimeAlgorithm

**Class Methods:**

| Method | Purpose |
|--------|---------|
| `initialize_regime_detection(ticker, n_states=3, lookback_days=252, retrain_frequency="weekly", regime_allocations=None, min_confidence=0.0)` | Setup regime detection |
| `on_tradebar(ticker, bar)` | Buffer new price data |
| `regime_is_ready(ticker=None)` | Check if enough data collected |
| `update_regime(ticker=None)` | Run regime detection, update state |
| `get_regime_allocation(ticker)` | Get recommended position size |
| `on_regime_change(old_regime, new_regime, confidence, ticker)` | Override for transition handling |

**Properties:**

| Property | Type | Meaning |
|----------|------|---------|
| `current_regime` | str | Current regime name ("Bull", "Bear", "Sideways", "Crisis") |
| `regime_confidence` | float | Confidence in regime (0.0-1.0) |
| `regime_state` | int | Numeric HMM state (0, 1, 2, ...) |

### Configuration Classes

**QuantConnectConfig**

```python
QuantConnectConfig(
    lookback_days=252,              # int: Historical window
    retrain_frequency="weekly",     # str: 'daily', 'weekly', 'monthly', 'never'
    warmup_days=252,                # int: Warm-up period
    use_cache=True,                 # bool: Cache models
    log_regime_changes=True,        # bool: Log transitions
    min_confidence=0.0              # float: Confidence threshold
)
```

**RegimeTradingConfig**

```python
RegimeTradingConfig(
    regime_allocations={            # Dict[str, float]: Regime→allocation
        "Bull": 1.0,
        "Bear": 0.0,
        "Sideways": 0.5,
        "Crisis": 0.0
    },
    rebalance_threshold=0.1,        # float: Min change to rebalance
    use_risk_parity=False,          # bool: Volatility-weight positions
    max_leverage=1.0,               # float: Max allowed leverage
    cash_regime="Bear"              # str: Regime triggering cash move
)
```

### Signal Classes

**TradingSignal**

```python
@dataclass
class TradingSignal:
    direction: SignalDirection           # LONG, SHORT, NEUTRAL
    strength: SignalStrength             # STRONG, MODERATE, WEAK, NONE
    allocation: float                    # Position size
    confidence: float                    # Regime confidence
    regime_name: str                     # Regime label
    regime_state: int                    # HMM state
    timestamp: pd.Timestamp              # Signal time
    metadata: Dict                       # Extra info

    # Methods:
    is_actionable(min_confidence=0.0) -> bool
    should_rebalance(current_allocation, threshold) -> bool
```

**SignalDirection Enum:**
- `SignalDirection.LONG` (1) - Buy signal
- `SignalDirection.NEUTRAL` (0) - Hold
- `SignalDirection.SHORT` (-1) - Sell/short

**SignalStrength Enum:**
- `SignalStrength.STRONG` (3) - Very confident
- `SignalStrength.MODERATE` (2) - Moderately confident
- `SignalStrength.WEAK` (1) - Low confidence
- `SignalStrength.NONE` (0) - No signal

### Data Adapters

**QuantConnectDataAdapter**

```python
adapter = QuantConnectDataAdapter(lookback_days=252)
adapter.add_bar(timestamp, open, high, low, close, volume)
df = adapter.to_dataframe()  # Returns DataFrame
is_ready = adapter.is_ready(min_bars=252)
```

---

## Troubleshooting

### "regime_is_ready() returns False for too long"

**Symptom:** Algorithm runs for weeks, never triggers trading.

**Cause:** Less than 252 bars received.

**Solution:**
- Increase `SetWarmup()` period in Initialize to ensure data availability
- Reduce `lookback_days` if backtesting short periods

```python
def Initialize(self):
    self.SetWarmup(252)  # Collect 252 bars before OnData starts
    self.initialize_regime_detection(ticker="SPY", lookback_days=252)
```

### "Current regime keeps switching frequently"

**Symptom:** Regime changes every few bars (noisy).

**Cause:** Too few states, too short lookback, low confidence threshold.

**Solution:**
- Increase `lookback_days` to 252+ (1+ year)
- Reduce number of states if using > 4
- Increase `retrain_frequency` from daily to weekly
- Increase `min_confidence` threshold

```python
# Change from:
self.initialize_regime_detection(ticker="SPY", n_states=5, lookback_days=63, retrain_frequency="daily")

# To:
self.initialize_regime_detection(ticker="SPY", n_states=3, lookback_days=252, retrain_frequency="weekly")
```

### "TypeError: 'NoneType' object when accessing current_regime"

**Symptom:** `current_regime` is None, crashes when checking if "Bull".

**Cause:** Called before `update_regime()` or before regime is ready.

**Solution:** Always check `regime_is_ready()` first:

```python
def OnData(self, data):
    self.on_tradebar("SPY", data[self.symbol])

    if not self.regime_is_ready():
        return  # Exit early, skip regime access

    self.update_regime()  # Now safe to access current_regime

    if self.current_regime == "Bull":
        # Safe to access here
        pass
```

### "Performance is slow with 10+ assets"

**Symptom:** Algorithm slow to process each bar.

**Cause:** Training 10 HMMs every bar/week.

**Solution:** Enable caching and parallel processing:

```python
class MyStrategy(HiddenRegimeAlgorithmOptimized):
    def Initialize(self):
        # Enable performance features
        self.enable_caching(max_cache_size=50)
        self.enable_batch_updates(max_workers=4)

        # Now add many assets
        for ticker in tickers:
            self.initialize_regime_detection(ticker, n_states=3)
```

### "Regime seems too stable, barely ever changes"

**Symptom:** Stays in Bull or Bear for months/years.

**Cause:** Too long lookback window, or stable market conditions.

**Solution:**
- Reduce `lookback_days` to 126 (6 months)
- Increase `n_states` to 4-5 for finer granularity
- Check regime manually to confirm legitimacy

```python
self.initialize_regime_detection(
    ticker="SPY",
    n_states=4,           # More states
    lookback_days=126,    # Shorter window
    retrain_frequency="weekly"
)
```

### "Data validation errors with certain tickers"

**Symptom:** Missing data, NaN values, or data gaps.

**Cause:** Sparse data, weekend/holiday gaps, or delisted symbols.

**Solution:**
- Use liquid assets (SPY, QQQ, TLT, major sectors)
- Increase date range to span gaps
- Check ticker is valid on QuantConnect platform

```python
# Avoid sparse tickers
# GOOD: Highly liquid ETFs
self.AddEquity("SPY", Resolution.Daily)  # S&P 500
self.AddEquity("QQQ", Resolution.Daily)  # Nasdaq
self.AddEquity("TLT", Resolution.Daily)  # Bonds

# RISKY: Penny stocks, illiquid symbols
self.AddEquity("PENNY", Resolution.Daily)  # May have gaps
```

---

## Further Reading

- **Main Hidden-Regime Docs**: [https://github.com/hidden-regime/hidden-regime](https://github.com/hidden-regime/hidden-regime)
- **QuantConnect Docs**: [https://www.quantconnect.com/docs/](https://www.quantconnect.com/docs/)
- **Template Examples**: See `quantconnect_templates/` directory for 9 complete strategies
- **Hidden-Regime Architecture**: See `ARCHITECTURE.md` for pipeline details
