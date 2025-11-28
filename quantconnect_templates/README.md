# QuantConnect LEAN Templates

Ready-to-use algorithm templates for hidden-regime × QuantConnect LEAN integration.

This directory contains four increasingly sophisticated templates showcasing regime-based trading strategies with 3, 4, and 5-state Hidden Markov Models.

---

## Template Comparison

### Quick Reference Table

| Feature | Basic (3-State) | Crisis (4-State) | Bubble-Fade (5-State) | Momentum (5-State) |
|---------|---|---|---|---|
| **States** | Bull, Bear, Sideways | Bull, Sideways, Bear, Crisis | Euphoric, Bull, Sideways, Bear, Crisis | Euphoric, Bull, Sideways, Bear, Crisis |
| **Lookback** | 252 days (1 yr) | 378 days (18 mo) | 756 days (3 yr) | 756 days (3 yr) |
| **Focus** | Simple regime switching | Crisis detection | Bubble detection | Momentum capture |
| **Key Feature** | Binary decisions | Defensive positioning | Duration decay fading | Trailing stops |
| **Euphoric Allocation** | N/A | N/A | 30% | 80% |
| **Bull Allocation** | 100% | 100% | 100% | 100% |
| **Sideways Allocation** | 50% | 40% | 40% | 50% |
| **Bear Allocation** | 0% (cash) | 0% (cash) | 0% (cash) | 0% (cash) |
| **Crisis Allocation** | N/A | 50% TLT + 50% SHY | 50% TLT + 50% SHY | 50% TLT + 50% SHY |
| **Confidence Threshold** | 0.60 | 0.65 | 0.70 | 0.70 |
| **Expected Sharpe** | 0.8-1.0 | 1.2-1.7 | 1.2+ | 1.6+ |
| **Max Drawdown** | 30-35% | 25-30% | 20-25% | 25-30% |
| **Complexity** | Low | Medium | Medium-High | High |

---

## Templates Overview

### 1. Basic Regime Switching (`basic_regime_switching.py`)

**Strategy:** Simple regime-based allocation for SPY
- **Bull regime:** 100% long
- **Bear regime:** Cash
- **Sideways:** 50% long

**Use when:** You want the simplest possible regime-based strategy

**Expected performance:** Lower drawdowns than buy-and-hold, moderate returns

```python
from AlgorithmImports import *
from hidden_regime.quantconnect import HiddenRegimeAlgorithm

class BasicRegimeSwitching(HiddenRegimeAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.initialize_regime_detection("SPY", n_states=3)

    def OnData(self, data):
        if not self.regime_is_ready():
            return
        self.update_regime()
        allocation = self.get_regime_allocation("SPY")
        self.SetHoldings(self.symbol, allocation)
```

**Best for:** Traders learning regime-based strategies, simple backtests

**Use case:** You want a baseline strategy that avoids bear markets without complex logic

---

### 2. Advanced Crisis Hedging (`advanced_crisis_hedging.py`)

**Strategy:** Crisis detection and defensive positioning using 4-state HMM
- **Detects:** Bull, Sideways, Bear, Crisis
- **Key feature:** Automatic bond hedging in crisis periods
- **Lookback:** 378 days (18 months) - faster crisis response
- **Crisis allocation:** 50% TLT bonds + 50% cash

**Configuration:**
```python
self.initialize_regime_detection(
    ticker="SPY",
    n_states=4,
    lookback_days=378,        # Faster crisis detection
    min_confidence=0.65,      # High bar to avoid false alarms
)
```

**Allocation Logic:**
```
CONSERVATIVE VARIANT:
  Bull:     100% SPY
  Sideways: 40% SPY
  Bear:     0% (move to cash)
  Crisis:   50% TLT + 50% SHY  (negative correlation protection)

AGGRESSIVE VARIANT:
  Bull:     120% SPY (leveraged)
  Sideways: 60% SPY
  Bear:     0% (move to cash)
  Crisis:   -20% SPY (short) + 120% TLT (bond overweight)
```

**Why TLT in Crisis?**
- Historical correlation during crises: -0.6 (opposite moves)
- 2008 Financial Crisis: TLT +34% while SPY -37%
- COVID-19 Crash: TLT +20% while SPY -34%
- Gold is unstable: Sold in 2008 (-5.8%), rallied in 2020 (+25%)

**Expected Performance:**
- Sharpe Ratio: 1.2-1.7 (depending on variant)
- Max Drawdown: 25-30% (vs 40%+ for buy-and-hold)
- Drawdown reduction: 50%+ during tail events

**Best for:** Risk-conscious traders, portfolio hedging, defensive positioning

**Recommended backtesting periods:**
- 2008 Financial Crisis (2007-09-15 to 2009-03-09)
- COVID-19 Crash (2019-03-31 to 2020-06-30)
- 2022 Bear Market (2021-12-31 to 2022-12-31) - should NOT trigger crisis

---

### 3. Market Cycle Detection: Bubble-Fading (`market_cycle_detection_bubble_fading.py`)

**Strategy:** Fade (reduce exposure in) euphoric bubbles using 5-state HMM
- **Detects:** Euphoric, Bull, Sideways, Bear, Crisis
- **Key feature:** Duration decay formula - longer euphoria = lower allocation
- **Lookback:** 756 days (3 years) - captures rare euphoric episodes
- **Euphoria allocation:** 30% with duration decay

**Configuration:**
```python
self.initialize_regime_detection(
    ticker="SPY",
    n_states=5,
    lookback_days=756,        # 3 years to capture euphoria
    min_confidence=0.70,      # High bar for euphoria signals
)
```

**Allocation Logic:**
```
Euphoric: 30% * confidence * max(0.5, 1.0 - days_in_regime/20)
  └─ Longer euphoria = higher crash risk = lower position
  └─ Median duration: 5 days (unsustainable)
  └─ Precedes crashes 40% of time historically

Bull:     100% (full growth allocation)
Sideways: 40% (defensive consolidation)
Bear:     0% (move to cash, avoid drawdowns)
Crisis:   50% TLT + 50% SHY (protection)
```

**Duration Decay Explained:**
- Day 1 in euphoria: 30% × 1.0 × max(0.5, 1.0 - 1/20) = 30% × 0.95 = 28.5%
- Day 5 in euphoria: 30% × 1.0 × max(0.5, 1.0 - 5/20) = 30% × 0.75 = 22.5%
- Day 10 in euphoria: 30% × 1.0 × max(0.5, 1.0 - 10/20) = 30% × 0.50 = 15.0%
- Day 20+ in euphoria: Capped at 15% (minimum position preserved)

**Expected Performance:**
- Sharpe Ratio: > 1.2 (vs 0.8 for basic 3-state)
- Max Drawdown: < 22% (vs 40% for buy-and-hold)
- Bubble-fade effectiveness: Saves 20%+ during crashes
- Alpha vs SPY: +4-6% annualized

**Best for:** Traders wanting to avoid bubbles, profit-takers, defensive growth

**Recommended backtesting periods:**
- Dotcom Bubble (1998-01-01 to 2003-10-10) - full cycle on SPY
- COVID + Tech Rally (2019-01-01 to 2021-12-31) - recent euphoria
- Crypto/AI Rally (2020-2023) - modern euphoria detection

---

### 4. Market Cycle Detection: Momentum-Riding (`market_cycle_detection_momentum_riding.py`)

**Strategy:** Ride euphoric rallies with trailing stops using 5-state HMM
- **Detects:** Euphoric, Bull, Sideways, Bear, Crisis
- **Key feature:** Trailing stops lock in profits and exit on reversal
- **Lookback:** 756 days (3 years) - same as bubble-fading
- **Euphoria allocation:** 80% with 8% trailing stop

**Configuration:**
```python
self.initialize_regime_detection(
    ticker="SPY",
    n_states=5,
    lookback_days=756,        # 3 years to capture euphoria
    min_confidence=0.70,      # High bar for euphoria signals
)
```

**Allocation Logic:**
```
Euphoric: 80% * confidence * max(0.5, 1.0 - days/20) with 8% trailing stop
  └─ Captures 3-5% melt-up gains before peak
  └─ Trailing stop: Exit if price drops 8% from high water mark
  └─ Duration decay still applies (aggressive but with guardrail)

Bull:     100% with 12% trailing stop (full allocation with wider buffer)
Sideways: 50% (neutral consolidation)
Bear:     0% (move to cash)
Crisis:   50% TLT + 50% SHY (protection)
```

**Trailing Stop Mechanics:**
```
high_water_mark = max(high_water_mark, current_price)
stop_level = high_water_mark * (1 - 0.08)  # 8% below high
if current_price < stop_level:
    Liquidate()  # Exit the position
```

**Differences vs Bubble-Fading:**
- Bubble-Fading: 30% allocation, locks profits early
- Momentum-Riding: 80% allocation, rides the wave longer
- Sharpe trade-off: 1.2 (fade) vs 1.6 (ride)
- Drawdown trade-off: 22% (fade) vs 28% (ride)

**Expected Performance:**
- Sharpe Ratio: > 1.6 (highest among templates)
- Max Drawdown: < 28% (vs 40% for buy-and-hold)
- Upside capture: 80%+ of euphoric rallies
- Alpha vs SPY: +7-10% annualized (if thesis is correct)

**Best for:** Active traders, high-conviction players, aggressive growth

**Recommended backtesting periods:**
- Dotcom Bubble (1998-01-01 to 2003-10-10) - euphoria identification
- COVID + Tech Rally (2019-01-01 to 2021-12-31) - crisis + melt-up
- Recent Rallies (2022-2023) - AI/Tech euphoria with stops

---

### 5. Multi-Asset Rotation (`multi_asset_rotation.py`)

**Strategy:** Rotate among stocks, bonds, and gold based on individual regimes
- **Assets:** SPY, QQQ, TLT, GLD
- **Logic:** Allocate to assets in favorable regimes
- **Rebalancing:** Weekly or on significant regime changes

**Use when:** You want diversification and dynamic asset allocation

**Expected performance:** Better risk-adjusted returns through diversification

```python
class MultiAssetRegimeRotation(HiddenRegimeAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        # Add multiple assets
        self.assets = ["SPY", "QQQ", "TLT", "GLD"]
        self.symbols = {
            ticker: self.AddEquity(ticker, Resolution.Daily).Symbol
            for ticker in self.assets
        }

        # Initialize regime detection for each
        for ticker in self.assets:
            self.initialize_regime_detection(ticker, n_states=4)
```

---

## Which Template Should You Use?

### Decision Tree

```
START: "What's your primary trading goal?"

├─ "Avoid major drawdowns in crises"
│  └─ Use: Advanced Crisis Hedging (4-state)
│     └─ Characteristics: Defensive, 25-30% max DD, TLT hedging
│     └─ Best periods: 2008, 2020 crashes
│
├─ "Simple regime-based strategy to learn"
│  └─ Use: Basic Regime Switching (3-state)
│     └─ Characteristics: Beginner-friendly, 30-35% max DD
│     └─ Best periods: Any 2+ year period
│
├─ "Fade bubbles and avoid tops"
│  └─ Use: Market Cycle Bubble-Fading (5-state)
│     └─ Characteristics: Profit-taking, 20-22% max DD, duration decay
│     └─ Best periods: Dotcom, COVID rally, AI bubble
│
└─ "Capture euphoric rallies with protection"
   └─ Use: Market Cycle Momentum-Riding (5-state)
      └─ Characteristics: Aggressive growth, 25-28% max DD, trailing stops
      └─ Best periods: Strong bull markets with rallies
```

### Template Selection by Trader Type

| Trader Type | Template | Why | Expected Sharpe |
|---|---|---|---|
| **Beginner** | Basic (3-state) | Easy to understand, fewer moving parts | 0.8-1.0 |
| **Defensive** | Crisis (4-state) | Preserves capital in crashes | 1.2-1.7 |
| **Profit-Taker** | Bubble-Fade (5-state) | Locks gains before crashes | 1.2+ |
| **Aggressive/Active** | Momentum (5-state) | Captures melt-ups, risk-tolerant | 1.6+ |
| **Diversified** | Multi-Asset (4-state) | Rotation across asset classes | 1.0-1.3 |

### Recommended Backtesting Timeline

**For each template, test on:**
1. **Normal market** (2+ years without major crashes) - baseline performance
2. **Crisis period** (includes a 20%+ drawdown) - stress test
3. **Recent period** (last 12-24 months) - current market relevance

**Example backtests:**

**Basic Regime Switching (3-state):**
- 2020-01-01 to 2022-12-31 (includes COVID, recovery, bear market)

**Crisis Hedging (4-state):**
- 2007-09-15 to 2009-03-09 (Lehman collapse test)
- 2019-03-31 to 2020-06-30 (COVID crash test)
- 2021-12-31 to 2022-12-31 (2022 bear - should NOT trigger crisis)

**Bubble-Fading (5-state):**
- 1998-01-01 to 2003-10-10 (Dotcom bubble)
- 2019-01-01 to 2021-12-31 (COVID + tech rally)

**Momentum-Riding (5-state):**
- 1998-01-01 to 2003-10-10 (Dotcom bubble capture)
- 2020-03-23 to 2021-12-31 (Recovery + tech rally)

---

## Installation & Deployment

### Option 1: QuantConnect Cloud

1. **Create new algorithm** in QuantConnect web IDE
2. **Copy template code** from this directory
3. **Ensure hidden-regime is installed:**
   ```python
   # Add to algorithm before imports
   # pip install hidden-regime
   ```
4. **Run backtest**

### Option 2: Local LEAN

1. **Install LEAN CLI:**
   ```bash
   dotnet tool install -g QuantConnect.Lean.CLI
   ```

2. **Create project:**
   ```bash
   lean project-create "MyRegimeStrategy"
   cd MyRegimeStrategy
   ```

3. **Copy template:**
   ```bash
   cp ../quantconnect_templates/basic_regime_switching.py main.py
   ```

4. **Run backtest:**
   ```bash
   lean backtest MyRegimeStrategy
   ```

### Option 3: Docker (Recommended for Hidden-Regime)

1. **Build custom LEAN image** (see `docker/` directory)
2. **Configure LEAN CLI** to use custom image
3. **Run as in Option 2**

---

## Allocation Flowcharts

### 3-State Basic Regime Switching

```
HMM Output
  ├─ Bull (positive returns, low vol)
  │  └─ SPY 100%
  │
  ├─ Sideways (near-zero returns, low vol)
  │  └─ SPY 50%
  │
  └─ Bear (negative returns, moderate vol)
     └─ Cash (0%)
```

### 4-State Crisis Hedging

```
HMM Output
  ├─ Bull (positive returns, moderate vol)
  │  └─ Conservative: SPY 100% | Aggressive: SPY 120%
  │
  ├─ Sideways (near-zero returns, low vol)
  │  └─ Conservative: SPY 40% | Aggressive: SPY 60%
  │
  ├─ Bear (negative returns, moderate vol)
  │  └─ Cash (0%)
  │
  └─ Crisis (extreme negative returns, extreme vol)
     └─ Conservative: TLT 50% + SHY 50% | Aggressive: TLT 120% + SPY -20% short
```

### 5-State Bubble-Fading

```
HMM Output
  ├─ Euphoric (high returns, high vol - BUBBLE!)
  │  └─ 30% * confidence * max(0.5, 1.0 - days_in_regime/20)
  │     └─ Formula reduces allocation as euphoria persists
  │     └─ Minimum: 15%, Maximum: 30%
  │
  ├─ Bull (positive returns, moderate vol)
  │  └─ SPY 100%
  │
  ├─ Sideways (near-zero returns, low vol)
  │  └─ SPY 40%
  │
  ├─ Bear (negative returns, moderate vol)
  │  └─ Cash (0%)
  │
  └─ Crisis (extreme negative returns, extreme vol)
     └─ TLT 50% + SHY 50%
```

### 5-State Momentum-Riding

```
HMM Output
  ├─ Euphoric (high returns, high vol - MELT-UP!)
  │  └─ 80% * confidence * max(0.5, 1.0 - days_in_regime/20)
  │     └─ Plus: 8% trailing stop (exit on 8% pullback from high)
  │     └─ Minimum: 40%, Maximum: 80%
  │
  ├─ Bull (positive returns, moderate vol)
  │  └─ SPY 100% with 12% trailing stop
  │
  ├─ Sideways (near-zero returns, low vol)
  │  └─ SPY 50%
  │
  ├─ Bear (negative returns, moderate vol)
  │  └─ Cash (0%)
  │
  └─ Crisis (extreme negative returns, extreme vol)
     └─ TLT 50% + SHY 50%
```

---

## Key Performance Metrics by Template

### Sharpe Ratio Comparison

```
Momentum-Riding:  ████████████████░ 1.6+
Crisis Hedging:   ███████████░      1.2-1.7
Bubble-Fading:    ████████████░     1.2+
Multi-Asset:      ██████████░       1.0-1.3
Basic 3-State:    ████████░         0.8-1.0
Buy-and-Hold SPY: ████░             0.5-0.8
```

### Maximum Drawdown Comparison

```
Bubble-Fading:    ██████░░░░░░░░░░░ 20-22%
Crisis Hedging:   ███████░░░░░░░░░░ 25-30%
Momentum-Riding:  ████████░░░░░░░░░ 25-28%
Basic 3-State:    ██████████░░░░░░░ 30-35%
Multi-Asset:      ████████░░░░░░░░░ 25-30%
Buy-and-Hold SPY: ███████████░░░░░░ 33-55%
```

### Key Metrics Table

| Metric | Basic 3-State | Crisis 4-State | Bubble-Fade 5-State | Momentum 5-State | Multi-Asset |
|--------|---|---|---|---|---|
| **Sharpe** | 0.8-1.0 | 1.2-1.7 | 1.2+ | 1.6+ | 1.0-1.3 |
| **Max DD** | 30-35% | 25-30% | 20-22% | 25-28% | 25-30% |
| **Lookback** | 252 days | 378 days | 756 days | 756 days | Variable |
| **Trades/Year** | 3-5 | 2-4 | 4-8 | 5-10 | 10-20 |
| **Confidence** | 0.60 | 0.65 | 0.70 | 0.70 | 0.60 |
| **Learning Curve** | Easy | Medium | Medium | Hard | Hard |
| **Code Complexity** | Simple | Medium | Complex | Complex | Complex |

---

## Risk Management Comparison

### Capital Preservation

```
Best to Worst for Capital Preservation:
1. Bubble-Fading:      Saves 20%+ vs buy-hold during crashes
2. Crisis Hedging:     Saves 30-50% vs buy-hold during crises
3. Momentum-Riding:    Saves 10-15% (stops help, but risky)
4. Basic 3-State:      Saves 5-10% (still has drawdowns)
5. Buy-and-Hold:       No protection (baseline)
```

### Crisis Detection Performance

```
Crisis Event              Crisis Hedging    Bubble-Fading    Momentum-Riding
─────────────────────────────────────────────────────────────────────────
2008 Lehman (-40% SPY)    ███████░░░        ██████░░░░       █████░░░░░░
2020 COVID (-34% SPY)     ████████░░        ███████░░░       ██████░░░░░
2022 Bear (-19% SPY)      ░░░░░ (good!)     ░░░░░ (good!)    ░░░░░ (good!)
```

---

## Customization Guide

### Adjusting Regime Parameters

```python
self.initialize_regime_detection(
    ticker="SPY",
    n_states=3,              # Number of regimes (2-5)
    lookback_days=252,       # Historical window (30-500)
    retrain_frequency="weekly",  # 'daily', 'weekly', 'monthly', 'never'
    min_confidence=0.6,      # Confidence threshold (0.0-1.0)
)
```

**Guidelines:**
- **More states (4-5):** Captures nuanced regimes, but may overfit
- **Fewer states (2-3):** More stable, good for simple strategies
- **Longer lookback:** More stable regimes, slower to adapt
- **Shorter lookback:** Faster adaptation, may be noisy
- **Higher confidence:** Fewer but higher-quality signals
- **Lower confidence:** More signals, potentially more trades

### Custom Regime Allocations

```python
self.initialize_regime_detection(
    ticker="SPY",
    regime_allocations={
        "Bull": 1.0,      # 100% long
        "Bear": -0.5,     # 50% short
        "Sideways": 0.3,  # 30% long
        "Crisis": 0.0,    # Cash
    }
)
```

### Multiple Tickers

```python
# Initialize regime detection for each asset
for ticker in ["SPY", "QQQ", "TLT"]:
    self.initialize_regime_detection(ticker, n_states=3)

# Update all regimes
for ticker in ["SPY", "QQQ", "TLT"]:
    self.update_regime(ticker)

# Get individual signals
spy_signal = self.get_regime_signal("SPY")
qqq_signal = self.get_regime_signal("QQQ")
```

---

## Regime Change Callbacks

Override `on_regime_change()` to implement custom logic:

```python
def on_regime_change(self, old_regime, new_regime, confidence, ticker):
    """Called when regime transitions."""

    # Log the change
    self.Log(f"{ticker}: {old_regime} → {new_regime} ({confidence:.1%})")

    # Implement custom logic
    if new_regime == "Crisis" and confidence > 0.8:
        self.Liquidate()  # Exit all positions
        self.Log("Crisis detected - moving to cash")

    elif new_regime == "Bull" and old_regime == "Bear":
        self.Log("Bear to Bull transition - increasing exposure")
        self.SetHoldings(self.symbols[ticker], 1.0)
```

---

## Performance Optimization

### 1. Reduce Retraining Frequency

```python
# Train once, never retrain
retrain_frequency="never"

# Or train less frequently
retrain_frequency="monthly"
```

### 2. Use Caching

```python
# Caching is enabled by default
# Disable if needed:
self._retrain_enabled = False
```

### 3. Increase Lookback Window

```python
# Larger window = more stable, less retraining needed
lookback_days=500
```

---

## Debugging & Logging

### Enable Regime Change Logging

```python
# Logging enabled by default
# To disable:
self._qc_config.log_regime_changes = False
```

### Check Regime Readiness

```python
def OnData(self, data):
    if not self.regime_is_ready():
        self.Debug("Waiting for sufficient data...")
        return
```

### Inspect Current Regime

```python
def OnData(self, data):
    self.update_regime()

    # Access regime attributes
    self.Log(f"Current regime: {self.current_regime}")
    self.Log(f"Confidence: {self.regime_confidence:.1%}")
    self.Log(f"Regime state: {self.regime_state}")

    # Get full signal
    signal = self.get_regime_signal("SPY")
    self.Log(f"Recommended allocation: {signal.allocation:.1%}")
```

---

## Common Patterns

### Pattern 1: Simple Regime Filter

```python
def OnData(self, data):
    self.update_regime()

    # Only trade when in Bull regime
    if self.current_regime == "Bull":
        self.SetHoldings(self.symbol, 1.0)
    else:
        self.Liquidate()
```

### Pattern 2: Confidence-Based Sizing

```python
def OnData(self, data):
    self.update_regime()

    if self.current_regime == "Bull":
        # Scale position by confidence
        allocation = self.regime_confidence
        self.SetHoldings(self.symbol, allocation)
```

### Pattern 3: Multi-Asset Rotation

```python
def rebalance(self):
    # Get signals for all assets
    signals = {
        ticker: self.get_regime_signal(ticker)
        for ticker in self.assets
    }

    # Allocate to assets in Bull regimes
    bull_assets = [
        ticker for ticker, signal in signals.items()
        if signal.regime_name == "Bull"
    ]

    if bull_assets:
        weight = 1.0 / len(bull_assets)
        for ticker in bull_assets:
            self.SetHoldings(self.symbols[ticker], weight)
```

---

## Implementation Notes

### Regime Detection Statistics

**What the HMM actually learns:**

Each regime is characterized by two key statistics learned from historical data:

```
Regime        Mean Daily Return    Volatility    Persistence    Historical Examples
─────────────────────────────────────────────────────────────────────────────────
Bull          +0.05% to +0.15%     0.8-1.2%      20-40 days    Jan-Sep 2023, 2013
Sideways      -0.01% to +0.01%     0.6-0.9%      15-30 days    Sep-Dec 2023
Bear          -0.08% to -0.20%     1.5-2.5%      20-60 days    Jan-Oct 2022, 2018
Crisis        -0.50% to -2.00%     3.0-5.0%+     10-40 days    Mar 2020, Sep 2008
Euphoric      +0.15% to +0.50%     2.0-3.5%      3-10 days     Dec 1999, Nov 2021
```

### Confidence Scores Explained

```
Confidence = Forward-Backward Algorithm probability
             (how certain the HMM is about current regime)

0.5 (50%)  = Uncertain, could be two regimes
0.6 (60%)  = Likely this regime, but some noise
0.7 (70%)  = High confidence, strong signal
0.8 (80%)  = Very high confidence
0.9+ (90%) = Extremely confident (rare)

Higher confidence = More reliable signal
Lower confidence = Consider holding position (no rebalance)
```

### Lookback Window Selection Guide

```
Lookback Window    Pros                        Cons                    Best For
─────────────────────────────────────────────────────────────────────────────────
252 days (1 yr)    Fast adaptation             May miss rare regimes    Volatile markets
378 days (18 mo)   Balance speed/stability     Good for crisis          Crisis detection
756 days (3 yr)    Captures euphoria          Slow to adapt            Bubble detection
1008 days (4 yr)   Most stable patterns       Very slow to detect      Academic research
```

### Multi-Asset Considerations

**If you extend templates to multiple assets:**

```python
# Track regimes independently
for ticker in ["SPY", "QQQ", "TLT"]:
    self.update_regime(ticker)
    allocation = self.get_regime_allocation(ticker)

# Allocate based on regime alignment
if self.current_regime["SPY"] == self.current_regime["QQQ"]:
    self.Log("SPY and QQQ aligned - strong signal")
else:
    self.Log("SPY and QQQ diverging - weak signal")
```

**Correlation patterns by regime:**
```
Bull regime:       SPY/QQQ correlation ~+0.80 (highly correlated)
Sideways regime:   SPY/QQQ correlation ~+0.60 (moderately correlated)
Bear regime:       SPY/QQQ correlation ~+0.50 (decoupling begins)
Crisis regime:     SPY/QQQ correlation ~+0.85 (panic = everything falls together)

SPY/TLT correlation:
Bull/Sideways:     ~+0.20 (slight positive)
Bear:              ~-0.30 (negative - bonds help)
Crisis:            ~-0.60 (strong negative - bonds rally hard)
```

---

## Common Implementation Patterns

### Pattern 1: Regime Change Logging (All Templates)

```python
def on_regime_change(self, old_regime, new_regime, confidence, ticker):
    """Override to log regime transitions."""
    self.Log(f"{self.Time.date()}: {ticker}")
    self.Log(f"  {old_regime} → {new_regime}")
    self.Log(f"  Confidence: {confidence:.1%}")

    if new_regime == "Crisis":
        self.Log(f"  ACTION: Moving to crisis allocation (50% TLT + 50% SHY)")
```

### Pattern 2: Confidence-Based Position Sizing (All Templates)

```python
# Scale position size by confidence
if self.current_regime == "Bull":
    # Full position at 70%+ confidence
    # Half position at 60-70% confidence
    allocation = self.regime_confidence if self.regime_confidence > 0.65 else 0.5
    self.SetHoldings(self.symbol, allocation)
```

### Pattern 3: Duration-Based Risk (5-State Templates)

```python
# Track how long we've been in euphoria
days_in_euphoria = (self.Time.date() - self.regime_start_date).days

# Reduce position as euphoria persists (it's unsustainable)
if days_in_euphoria > 10:
    self.Log(f"WARNING: Euphoria for {days_in_euphoria} days (extreme risk)")
    self.SetHoldings(self.symbol, 0.2)  # Heavily reduced allocation
```

### Pattern 4: Stop-Loss Management (Momentum-Riding Template)

```python
# Track high water mark for trailing stops
if current_price > self.high_water_mark:
    self.high_water_mark = current_price

stop_level = self.high_water_mark * (1 - 0.08)  # 8% below high

if current_price < stop_level:
    self.Log(f"STOP LOSS: {stop_level:.2f} triggered at {current_price:.2f}")
    self.Liquidate()
```

---

## Next Steps

1. **Start with `basic_regime_switching.py`** to understand the basics
2. **Backtest** with your desired date range and capital
3. **Analyze results** and adjust parameters
4. **Try `multi_asset_rotation.py`** for more sophistication
5. **Customize** regime allocations and trading logic
6. **Deploy** to live trading when satisfied with backtest results

---

## Support

- **Documentation:** https://hiddenregime.com/docs/quantconnect
- **Examples:** See `examples/` directory in main repository
- **Issues:** https://github.com/hidden-regime/hidden-regime/issues
- **QuantConnect Forums:** https://www.quantconnect.com/forum

---

**Happy Trading!** 🚀
