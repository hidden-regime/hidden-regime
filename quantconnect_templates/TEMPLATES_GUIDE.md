# Hidden-Regime Algorithm Templates

Complete guide to all available strategy templates for QuantConnect LEAN.

---

## Template Overview

| Template | Complexity | Use Case | Assets | States |
|----------|------------|----------|---------|--------|
| [Basic Regime Switching](#1-basic-regime-switching) | ⭐ Beginner | Simple regime-based allocation | SPY | 3 |
| [Multi-Asset Rotation](#2-multi-asset-rotation) | ⭐⭐ Intermediate | Diversified regime allocation | SPY, QQQ, TLT, GLD | 4 |
| [Crisis Detection](#3-crisis-detection) | ⭐⭐ Intermediate | Defensive positioning | SPY, TLT, GLD, SHY | 4 |
| [Sector Rotation](#4-sector-rotation) | ⭐⭐⭐ Advanced | Sector-based allocation | 8 Sector ETFs | 4 |
| [Dynamic Position Sizing](#5-dynamic-position-sizing) | ⭐⭐⭐ Advanced | Confidence-based sizing | SPY | 3 |
| [Framework Example](#6-framework-example) | ⭐⭐⭐ Advanced | Full QC Framework | QQQ components | 3 |

---

## 1. Basic Regime Switching

**File:** `basic_regime_switching.py`

### Overview
The simplest regime-based strategy. Perfect for beginners.

### Strategy
- Detect SPY regime (Bull, Bear, Sideways)
- Bull: 100% SPY
- Bear: 100% cash
- Sideways: 50% SPY

### Configuration
```python
n_states=3
lookback_days=252
retrain_frequency="weekly"
```

### Pros
✅ Simple to understand
✅ Low maintenance
✅ Good starting point

### Cons
❌ Single asset exposure
❌ Binary decisions
❌ No diversification

### Best For
- Learning regime detection
- Understanding basics
- Simple market timing

### Quick Start
```bash
./scripts/quick_backtest.sh basic_regime_switching
```

---

## 2. Multi-Asset Rotation

**File:** `multi_asset_rotation.py`

### Overview
Rotate among multiple asset classes based on individual regimes.

### Strategy
- Monitor 4 assets independently
- SPY (stocks), QQQ (tech), TLT (bonds), GLD (gold)
- Allocate to assets in Bull regimes
- Defensive allocation when no bulls

### Configuration
```python
n_states=4  # per asset
lookback_days=180
rebalance_frequency=5  # days
```

### Allocation Logic
- Bull assets: Equal weight among favorable
- No bull assets: 60% TLT, 40% GLD (defensive)
- Max 40% per asset (concentration limit)

### Pros
✅ Diversification across asset classes
✅ Independent regime detection
✅ Automatic rebalancing

### Cons
❌ More complexity
❌ Higher turnover
❌ More regime updates needed

### Best For
- Diversified portfolios
- Risk-averse strategies
- Multi-asset exposure

### Quick Start
```bash
./scripts/quick_backtest.sh multi_asset_rotation
```

---

## 3. Crisis Detection

**File:** `crisis_detection.py`

### Overview
Fast crisis detection with immediate defensive positioning.

### Strategy
- 4-state HMM for crisis identification
- 90-day lookback for faster adaptation
- Immediate flight to safety in crisis
- Defensive assets: TLT, GLD, SHY

### Crisis Triggers
- Regime = "Crisis" OR "Bear"
- Confidence ≥ 75%
- Triggers immediate defensive allocation

### Allocation by Regime

**Crisis Mode:**
- 50% TLT (long-term bonds)
- 30% GLD (gold)
- 20% SHY (short-term bonds)

**Bear (non-crisis):**
- 40% TLT
- 30% SHY
- 30% GLD

**Sideways:**
- 60% SPY
- 25% TLT
- 15% GLD

**Bull:**
- 100% SPY

### Pros
✅ Fast crisis detection (90-day window)
✅ Automatic defensive positioning
✅ Flight to quality assets

### Cons
❌ May react to false signals
❌ Shorter lookback = more noise
❌ Multiple assets needed

### Best For
- Risk management
- Protecting against crashes
- Conservative strategies

### Quick Start
```bash
./scripts/quick_backtest.sh crisis_detection
```

---

## 4. Sector Rotation

**File:** `sector_rotation.py`

### Overview
Rotate among sector ETFs based on market regime.

### Strategy
Two implementations:

**A) Market-Based Rotation:**
- Detect SPY market regime
- Rotate sectors that perform well in that regime
- Bull → Tech (XLK), Discretionary (XLY), Financials (XLF)
- Bear → Utilities (XLU), Staples (XLP)
- Crisis → Healthcare (XLV), Utilities (XLU)

**B) Individual Sector Regimes** (AdvancedSectorRotation):
- Detect regime for EACH sector independently
- Allocate to sectors in Bull regimes
- High-confidence sectors get more weight

### Sector ETFs
- XLK: Technology
- XLY: Consumer Discretionary
- XLF: Financials
- XLP: Consumer Staples
- XLU: Utilities
- XLV: Healthcare
- XLE: Energy
- XLI: Industrials

### Configuration
```python
# Market-based
n_states=4
max_sectors=4

# Individual sector regimes
n_states=3  # per sector
rebalance_frequency=7  # weekly
```

### Pros
✅ Sector diversification
✅ Regime-appropriate exposure
✅ Two implementation approaches

### Cons
❌ Requires multiple sector ETFs
❌ More complex rebalancing
❌ Higher data requirements

### Best For
- Sector rotation strategies
- Advanced diversification
- Market regime exploitation

### Quick Start
```bash
./scripts/quick_backtest.sh sector_rotation
```

---

## 5. Dynamic Position Sizing

**File:** `dynamic_position_sizing.py`

### Overview
Adjust position size based on regime confidence and risk limits.

### Strategy
Two implementations:

**A) Confidence-Based Sizing:**
- Bull + High Confidence (>80%): 100% allocation
- Bull + Medium Confidence (60-80%): 70%
- Bull + Low Confidence (<60%): 40%
- Sideways: 20-50% (scaled by confidence)
- Bear/Crisis: 0% (cash)

**B) Kelly Criterion Sizing:**
- Calculate optimal position size using Kelly formula
- Based on historical regime win rates
- Adjusts for regime-specific performance

### Risk Management
- Maximum drawdown: 15%
- Stop-loss: 5% trailing
- Position limits
- Risk tracking

### Configuration
```python
n_states=3
max_drawdown=0.15
stop_loss_pct=0.05
```

### Pros
✅ Intelligent position sizing
✅ Risk management built-in
✅ Adapts to confidence levels

### Cons
❌ More complex logic
❌ Requires historical data (Kelly)
❌ May underperform in strong trends

### Best For
- Risk-adjusted returns
- Portfolio optimization
- Advanced traders

### Quick Start
```bash
./scripts/quick_backtest.sh dynamic_position_sizing
```

---

## 6. Framework Example

**File:** `framework_example.py`

### Overview
Full QuantConnect Framework integration using all Framework components.

### Framework Components

**Universe Selection:**
- Manual universe (QQQ components)
- AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA

**Alpha Model:**
- HiddenRegimeAlphaModel
- Generates insights from regime detection
- 90-day lookback, 65% confidence threshold

**Portfolio Construction:**
- EqualWeightingPortfolioConstructionModel
- Daily rebalancing

**Execution:**
- ImmediateExecutionModel

**Risk Management:**
- MaximumDrawdownPercentPerSecurity (10%)

### Configuration
```python
n_states=3
lookback_days=90
confidence_threshold=0.65
insight_period_days=5
```

### Pros
✅ Uses full QC Framework
✅ Modular components
✅ Professional structure

### Cons
❌ Most complex template
❌ Framework learning curve
❌ More components to understand

### Best For
- QC Framework users
- Professional strategies
- Scalable algorithms

### Quick Start
```bash
./scripts/quick_backtest.sh framework_example
```

---

## Template Comparison

### Performance Characteristics

| Template | Expected Sharpe | Max Drawdown | Turnover | Complexity |
|----------|----------------|--------------|----------|------------|
| Basic Regime Switching | 0.8-1.2 | 15-25% | Low | Low |
| Multi-Asset Rotation | 1.0-1.5 | 12-20% | Medium | Medium |
| Crisis Detection | 0.9-1.3 | 10-15% | Medium | Medium |
| Sector Rotation | 1.1-1.6 | 15-22% | High | High |
| Dynamic Position Sizing | 1.2-1.7 | 10-18% | Low | High |
| Framework Example | 1.0-1.4 | 12-20% | Medium | High |

*Note: Performance estimates are illustrative. Actual results depend on market conditions and parameters.*

---

## Choosing the Right Template

### By Experience Level

**Beginner:**
1. Basic Regime Switching
2. Multi-Asset Rotation

**Intermediate:**
3. Crisis Detection
4. Sector Rotation (market-based version)

**Advanced:**
5. Dynamic Position Sizing
6. Sector Rotation (individual regimes)
7. Framework Example

### By Investment Goal

**Capital Preservation:**
- Crisis Detection
- Multi-Asset Rotation

**Growth:**
- Basic Regime Switching
- Sector Rotation

**Risk-Adjusted Returns:**
- Dynamic Position Sizing
- Framework Example

**Diversification:**
- Multi-Asset Rotation
- Sector Rotation

---

## Customization Guide

### Common Modifications

#### Adjust Regime Sensitivity

```python
# More responsive (shorter history)
lookback_days=90  # Instead of 252

# More stable (longer history)
lookback_days=500  # Instead of 252
```

#### Change Confidence Threshold

```python
# More signals (lower threshold)
min_confidence=0.50  # Instead of 0.65

# Fewer, higher-quality signals
min_confidence=0.80  # Instead of 0.65
```

#### Modify Retraining Schedule

```python
# More frequent updates
retrain_frequency="daily"  # Instead of "weekly"

# Less frequent (more stable)
retrain_frequency="monthly"  # Instead of "weekly"

# One-time training
retrain_frequency="never"  # Train once, use forever
```

#### Custom Regime Allocations

```python
regime_allocations={
    "Bull": 1.5,    # 150% long (leveraged)
    "Bear": -0.5,   # 50% short
    "Sideways": 0.5,  # 50% long
}
```

---

## Performance Optimization

### Reduce Computation Time

1. **Longer lookback** = Less frequent retraining
2. **Lower frequency** = Less processing
3. **Fewer states** = Faster HMM training

### Improve Strategy Performance

1. **Tune confidence threshold** for your risk tolerance
2. **Adjust regime allocations** based on backtest results
3. **Test different lookback periods** for your asset
4. **Combine multiple templates** for hybrid strategies

---

## Common Patterns

### Pattern 1: Regime Filter

Use regime detection as a filter for another strategy:

```python
def OnData(self, data):
    self.update_regime()

    # Only trade your strategy in Bull regimes
    if self.current_regime == "Bull" and self.regime_confidence > 0.7:
        # Your existing strategy logic here
        pass
    else:
        self.Liquidate()
```

### Pattern 2: Confidence Weighting

Scale any strategy by regime confidence:

```python
def OnData(self, data):
    self.update_regime()

    base_signal = your_signal_function()  # Your existing signal
    adjusted_signal = base_signal * self.regime_confidence

    self.SetHoldings(self.symbol, adjusted_signal)
```

### Pattern 3: Multi-Timeframe

Combine regime detection at different timeframes:

```python
# Long-term regime (monthly)
self.initialize_regime_detection("SPY_long", n_states=3, lookback_days=500)

# Short-term regime (daily)
self.initialize_regime_detection("SPY_short", n_states=3, lookback_days=90)

# Trade only when both agree
if long_term_regime == "Bull" and short_term_regime == "Bull":
    # Full exposure
    pass
```

---

## Troubleshooting

### No Trades Executing

**Cause:** Confidence threshold too high
**Solution:** Lower `min_confidence` to 0.50

**Cause:** Insufficient warm-up period
**Solution:** Ensure `SetWarmUp(timedelta(days=lookback_days))`

### Too Many Trades

**Cause:** Regime switching too frequently
**Solution:** Increase `lookback_days` or `min_confidence`

### Poor Performance

**Cause:** Wrong regime allocations for market
**Solution:** Backtest different allocation schemes

**Cause:** Overfitting to training period
**Solution:** Use longer lookback, test out-of-sample

---

## Next Steps

1. **Start simple** - Try basic_regime_switching first
2. **Understand the output** - Review logs and regime transitions
3. **Backtest thoroughly** - Test across different market conditions
4. **Customize** - Adjust parameters for your goals
5. **Combine** - Hybrid strategies using multiple templates

---

## Support

- **Documentation**: `/QC_ROADMAP.md`, `/QUANTCONNECT_INSTALLATION.md`
- **Examples**: All `.py` files in this directory
- **Issues**: https://github.com/hidden-regime/hidden-regime/issues

---

**Happy Trading!** 🚀

*Last updated: 2025-11-17 - Phase 3 Complete*
