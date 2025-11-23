# Hidden-Regime × QuantConnect Quick Start Tutorial

**Get from zero to backtesting in 5 minutes!**

This tutorial will guide you through your first hidden-regime strategy on QuantConnect LEAN.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Your First Strategy (3 minutes)](#your-first-strategy)
4. [Understanding the Results](#understanding-the-results)
5. [Next Steps](#next-steps)

---

## Prerequisites

**What you'll need:**
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Python 3.8+ (for local development)
- Basic understanding of Python
- 5 minutes of your time ⏱️

**No prior experience required with:**
- ❌ Hidden Markov Models
- ❌ QuantConnect LEAN
- ❌ Algorithmic trading

---

## Installation

### Option 1: One-Command Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/hidden-regime.git
cd hidden-regime

# Run the setup script
bash scripts/setup_quantconnect.sh
```

**What it does:**
1. ✅ Installs LEAN CLI
2. ✅ Builds custom Docker image with hidden-regime
3. ✅ Configures QuantConnect environment
4. ✅ Creates example project

**Expected output:**
```
🚀 Setting up hidden-regime for QuantConnect LEAN...
✓ LEAN CLI installed
✓ Docker image built: quantconnect/lean:hidden-regime
✓ Environment configured
✓ Example project created: BasicRegimeStrategy

Setup complete! Ready to backtest.
```

### Option 2: Manual Setup

```bash
# 1. Install LEAN CLI
dotnet tool install -g QuantConnect.Lean.CLI

# 2. Build Docker image
docker build -t quantconnect/lean:hidden-regime -f docker/Dockerfile .

# 3. Configure LEAN
lean config set engine-image quantconnect/lean:hidden-regime

# 4. Create project
lean project-create "MyRegimeStrategy" --language python
```

---

## Your First Strategy

### Step 1: Create Your Strategy File (1 minute)

Create a file `main.py` in your project directory:

```python
from AlgorithmImports import *
from hidden_regime.quantconnect import HiddenRegimeAlgorithm

class BasicRegimeSwitching(HiddenRegimeAlgorithm):
    """
    Simple regime-based strategy:
    - Bull regime: 100% long SPY
    - Bear regime: 100% cash
    - Sideways: 50% long SPY
    """

    def Initialize(self):
        # Basic setup
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Add SPY
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Initialize regime detection
        self.initialize_regime_detection(
            ticker="SPY",
            n_states=3,  # Bull, Bear, Sideways
            lookback_days=252,  # 1 year of data
            regime_allocations={
                "Bull": 1.0,      # 100% long
                "Bear": 0.0,      # Cash
                "Sideways": 0.5   # 50% long
            }
        )

        self.Log("🚀 Regime detection initialized!")

    def OnData(self, data):
        # Skip if no data
        if not data.ContainsKey(self.symbol):
            return

        # Update regime
        result = self.update_regime()

        # Trade based on current regime
        if self.current_regime == "Bull":
            self.SetHoldings(self.symbol, 1.0)

        elif self.current_regime == "Bear":
            self.Liquidate(self.symbol)

        elif self.current_regime == "Sideways":
            self.SetHoldings(self.symbol, 0.5)

    def on_regime_change(self, old_regime, new_regime, confidence):
        """Called when regime changes."""
        self.Log(f"📊 Regime change: {old_regime} → {new_regime}")
        self.Log(f"   Confidence: {confidence:.1%}")
```

**That's it!** 30 lines of code for a complete regime-trading strategy.

### Step 2: Run Your Backtest (30 seconds)

```bash
# Navigate to your project
cd MyRegimeStrategy

# Run backtest
lean backtest .
```

**What happens:**
1. LEAN starts the backtest engine
2. Hidden-regime detects market regimes
3. Strategy trades based on detected regimes
4. Results are generated

**Expected output:**
```
20240101 00:00:00 🚀 Regime detection initialized!
20240115 00:00:00 📊 Regime change: None → Bull
20240115 00:00:00    Confidence: 82.4%
20240315 00:00:00 📊 Regime change: Bull → Sideways
20240315 00:00:00    Confidence: 71.2%
...

Backtest complete!
Results saved to: backtests/2024-11-17_MyRegimeStrategy
```

### Step 3: View Results (30 seconds)

```bash
# Open results
open backtests/latest/index.html  # macOS
xdg-open backtests/latest/index.html  # Linux
start backtests/latest/index.html  # Windows
```

---

## Understanding the Results

### Performance Metrics

You'll see metrics like:

```
Total Return: 42.3%
Sharpe Ratio: 1.45
Max Drawdown: -12.8%
Win Rate: 62.1%
```

**What do these mean?**

- **Total Return:** Your strategy returned 42.3% over the period
- **Sharpe Ratio:** Risk-adjusted return (>1 is good, >2 is excellent)
- **Max Drawdown:** Largest peak-to-trough decline (-12.8%)
- **Win Rate:** 62.1% of trades were profitable

### Regime Detection Insights

The logs show regime changes:
```
Bull (82.4% confidence) → Sideways (71.2% confidence)
```

**Interpretation:**
- **High confidence (>80%):** Strong regime signal - safe to take full position
- **Medium confidence (60-80%):** Moderate signal - consider partial position
- **Low confidence (<60%):** Weak signal - be cautious

### Equity Curve

The chart shows your portfolio value over time:
- **Smooth upward slope:** Good, consistent returns
- **Sharp drops:** Drawdown periods (check if in Bear regime)
- **Flat periods:** Sideways regime, reduced exposure

---

## Next Steps

### 🎯 Improve Your Strategy

**1. Optimize Parameters**

Try different lookback periods:
```python
# More sensitive (faster regime changes)
self.initialize_regime_detection(
    ticker="SPY",
    n_states=3,
    lookback_days=90  # 3 months
)

# More stable (slower regime changes)
self.initialize_regime_detection(
    ticker="SPY",
    n_states=3,
    lookback_days=504  # 2 years
)
```

**2. Add More Regimes**

Detect more nuanced market states:
```python
# 4-state model: Bull, Bear, Sideways, Crisis
self.initialize_regime_detection(
    ticker="SPY",
    n_states=4,
    regime_allocations={
        "Bull": 1.0,
        "Bear": 0.0,
        "Sideways": 0.5,
        "Crisis": -0.3  # Short or defensive
    }
)
```

**3. Use Multi-Asset Rotation**

Trade multiple assets based on their individual regimes:

```python
from hidden_regime.quantconnect import HiddenRegimeAlgorithm

class MultiAssetRotation(HiddenRegimeAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        # Add multiple assets
        self.assets = {
            'SPY': self.AddEquity('SPY', Resolution.Daily).Symbol,  # Stocks
            'TLT': self.AddEquity('TLT', Resolution.Daily).Symbol,  # Bonds
            'GLD': self.AddEquity('GLD', Resolution.Daily).Symbol,  # Gold
        }

        # Initialize regime detection for each
        for ticker in self.assets.keys():
            self.initialize_regime_detection(
                ticker=ticker,
                n_states=3,
                lookback_days=252
            )

    def OnData(self, data):
        # Update all regimes
        regime_results = {}
        for ticker in self.assets.keys():
            regime_results[ticker] = self.update_regime(ticker)

        # Allocate to best opportunities
        for ticker, result in regime_results.items():
            if result['regime'] == 'Bull':
                self.SetHoldings(self.assets[ticker], 1.0 / len(self.assets))
            else:
                self.Liquidate(self.assets[ticker])
```

**4. Enable Performance Optimizations**

Speed up your backtest by 60-70%:

```python
from hidden_regime.quantconnect.optimized_algorithm import HiddenRegimeAlgorithmOptimized

class FastStrategy(HiddenRegimeAlgorithmOptimized):
    def Initialize(self):
        # Basic setup...

        # Enable optimizations (3 lines!)
        self.enable_caching(max_cache_size=200, retrain_frequency='monthly')
        self.enable_batch_updates(max_workers=4)
        self.enable_profiling()

        # Rest of your strategy...
```

---

## Common Patterns

### Pattern 1: Confidence-Based Position Sizing

Adjust position size based on regime confidence:

```python
def OnData(self, data):
    result = self.update_regime()

    confidence = result['confidence']

    if self.current_regime == "Bull":
        # Higher confidence = larger position
        if confidence >= 0.80:
            self.SetHoldings(self.symbol, 1.00)  # Full position
        elif confidence >= 0.60:
            self.SetHoldings(self.symbol, 0.70)  # Partial position
        else:
            self.SetHoldings(self.symbol, 0.40)  # Small position
```

### Pattern 2: Defensive Positioning in Crisis

Go defensive when crisis detected:

```python
def on_regime_change(self, old_regime, new_regime, confidence):
    if new_regime == "Crisis":
        # Defensive allocation
        self.SetHoldings(self.tlt, 0.50)  # Bonds
        self.SetHoldings(self.gld, 0.30)  # Gold
        self.SetHoldings(self.shy, 0.20)  # Cash proxy
```

### Pattern 3: Regime-Based Stop Loss

Only use stops in uncertain regimes:

```python
def OnData(self, data):
    result = self.update_regime()

    if self.current_regime == "Sideways" and result['confidence'] < 0.6:
        # Use stop loss in uncertain sideways market
        self.SetHoldings(self.symbol, 0.5)
        self.SetStopLoss(self.symbol, 0.02)  # 2% stop
    else:
        # No stop loss in clear trends
        self.SetHoldings(self.symbol, 1.0)
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'hidden_regime'"

**Solution:**
```bash
# Rebuild Docker image with hidden-regime
docker build -t quantconnect/lean:hidden-regime -f docker/Dockerfile .

# Ensure LEAN uses correct image
lean config set engine-image quantconnect/lean:hidden-regime
```

### Issue: "Not enough data to initialize regime detection"

**Cause:** Insufficient historical data

**Solution:**
```python
# Add warmup period
def Initialize(self):
    self.SetWarmUp(timedelta(days=252))  # Warm up for 1 year

    # Your strategy code...
```

### Issue: Backtest is slow

**Solution 1:** Use optimized algorithm
```python
from hidden_regime.quantconnect.optimized_algorithm import HiddenRegimeAlgorithmOptimized

class MyStrategy(HiddenRegimeAlgorithmOptimized):
    def Initialize(self):
        self.enable_caching(retrain_frequency='monthly')  # Cache models
```

**Solution 2:** Reduce training frequency
```python
self.initialize_regime_detection(
    ticker="SPY",
    n_states=3,
    lookback_days=252,
    retrain_frequency='monthly'  # Train less often
)
```

### Issue: Too many regime changes (whipsaws)

**Cause:** Model too sensitive

**Solution:** Increase lookback period and confidence threshold
```python
self.initialize_regime_detection(
    ticker="SPY",
    n_states=3,
    lookback_days=504,  # Longer history = more stable
    confidence_threshold=0.7  # Higher threshold
)
```

---

## Quick Command Reference

```bash
# Create new project
lean project-create "ProjectName" --language python

# Run backtest
lean backtest ProjectName

# Run with specific date range
lean backtest ProjectName --start 20200101 --end 20231231

# View logs
cat ProjectName/backtests/latest/log.txt

# Clean up old backtests
lean backtest clean ProjectName

# Update LEAN engine
lean update

# Show configuration
lean config list
```

---

## Templates Available

Ready-made strategies you can use immediately:

| Template | Description | Complexity |
|----------|-------------|------------|
| `basic_regime_switching.py` | Simple 3-regime strategy | Beginner |
| `multi_asset_rotation.py` | 4-asset rotation | Intermediate |
| `crisis_detection.py` | 4-regime with crisis detection | Intermediate |
| `sector_rotation.py` | 8-sector rotation | Advanced |
| `dynamic_position_sizing.py` | Confidence-based sizing | Advanced |
| `optimized_multi_asset.py` | 6-asset with optimizations | Advanced |

**Copy a template:**
```bash
cp quantconnect_templates/basic_regime_switching.py MyProject/main.py
```

---

## Learning Resources

### Documentation
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Best Practices](BEST_PRACTICES.md) - Trading strategy best practices
- [Template Usage Guide](TEMPLATE_USAGE_GUIDE.md) - Detailed template documentation

### Examples
- `quantconnect_templates/` - 6 ready-to-use strategy templates
- `examples/` - Additional example notebooks

### Community
- GitHub Issues: Report bugs or request features
- Discussions: Ask questions and share strategies

---

## Success Checklist

After this tutorial, you should be able to:

- ✅ Set up hidden-regime with QuantConnect LEAN
- ✅ Create a basic regime-switching strategy
- ✅ Run backtests and interpret results
- ✅ Understand regime detection confidence
- ✅ Modify parameters to optimize performance
- ✅ Use templates as starting points

---

## What's Next?

**Ready for more?**

1. **Explore Templates** - Try the advanced templates
2. **Read Best Practices** - Learn professional trading strategies
3. **Optimize Performance** - Use caching and batch updates
4. **Go Live** - Deploy to live trading (paper or real)

---

## Congratulations! 🎉

You've completed your first hidden-regime backtest on QuantConnect!

**Time invested:** ~5 minutes
**Skills gained:** Regime detection, algorithmic trading, backtesting

**Next milestone:** Run your first multi-asset rotation strategy using `multi_asset_rotation.py` template.

---

**Need help?** Check the [FAQ](FAQ.md) or open an issue on GitHub.

**Happy trading!** 📈
