# QuantConnect LEAN Templates

Ready-to-use algorithm templates for hidden-regime × QuantConnect LEAN integration.

## Quick Start

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

---

### 2. Multi-Asset Rotation (`multi_asset_rotation.py`)

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
