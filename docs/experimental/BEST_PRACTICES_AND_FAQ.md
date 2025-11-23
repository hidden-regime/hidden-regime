# Hidden-Regime × QuantConnect: Best Practices & FAQ

**Complete guide to professional regime trading strategies**

---

## Table of Contents

**Part I: Best Practices**
1. [Strategy Design](#strategy-design)
2. [Parameter Selection](#parameter-selection)
3. [Risk Management](#risk-management)
4. [Performance Optimization](#performance-optimization)
5. [Production Deployment](#production-deployment)

**Part II: FAQ**
6. [General Questions](#general-questions)
7. [Technical Questions](#technical-questions)
8. [Performance Questions](#performance-questions)
9. [Troubleshooting](#troubleshooting)

---

# Part I: Best Practices

## Strategy Design

### ✅ DO: Start Simple

```python
# Good - Simple 3-regime strategy
class SimpleStrategy(HiddenRegimeAlgorithm):
    def Initialize(self):
        self.initialize_regime_detection(
            ticker="SPY",
            n_states=3,  # Bull, Bear, Sideways
            regime_allocations={
                "Bull": 1.0,
                "Bear": 0.0,
                "Sideways": 0.5
            }
        )
```

**Why:** Simple strategies are easier to understand, debug, and maintain. Add complexity only when simple approaches fail.

### ❌ DON'T: Over-complicate Initially

```python
# Bad - Too complex for initial testing
class OverComplexStrategy(HiddenRegimeAlgorithm):
    def Initialize(self):
        # 10 assets × 5 regimes × dynamic sizing = hard to debug
        for asset in self.universe:
            self.initialize_regime_detection(
                ticker=asset,
                n_states=5,
                use_dynamic_sizing=True,
                confidence_weighting=True,
                # ... 20 more parameters
            )
```

### ✅ DO: Use Confidence Thresholds

```python
def OnData(self, data):
    result = self.update_regime()

    # Only trade on high-confidence signals
    if result['confidence'] >= 0.70:
        if self.current_regime == "Bull":
            self.SetHoldings(self.symbol, 1.0)
    else:
        # Low confidence - reduce exposure
        self.SetHoldings(self.symbol, 0.3)
```

**Why:** High-confidence signals lead to better risk-adjusted returns. Low-confidence periods often indicate regime transitions.

### ✅ DO: Validate Regime Labels

```python
def on_regime_change(self, old_regime, new_regime, confidence):
    # Log regime characteristics
    self.Debug(f"New regime: {new_regime}")
    self.Debug(f"Recent volatility: {self.calculate_volatility():.2%}")
    self.Debug(f"Recent return: {self.calculate_return():.2%}")

    # Verify regime makes sense
    if new_regime == "Bull" and self.calculate_return() < 0:
        self.Debug("⚠️  Warning: Bull regime but negative returns")
```

**Why:** Regime labels are assigned automatically. Validation ensures they match your intuition about market conditions.

---

## Parameter Selection

### Lookback Period

**Guidelines:**

| Market Condition | Recommended Lookback | Rationale |
|-----------------|---------------------|-----------|
| Fast-moving (crypto, tech stocks) | 60-90 days | Capture recent regime quickly |
| Normal equities | 180-252 days | Balance stability and responsiveness |
| Slow-moving (bonds, commodities) | 252-504 days | Longer patterns matter more |

**Example:**
```python
# Crypto - fast regime changes
self.initialize_regime_detection(
    ticker="BTCUSD",
    lookback_days=60
)

# S&P 500 - balanced
self.initialize_regime_detection(
    ticker="SPY",
    lookback_days=252
)

# Bonds - slow regime changes
self.initialize_regime_detection(
    ticker="TLT",
    lookback_days=504
)
```

### Number of States

**Guidelines:**

| States | Use Case | Example Regimes |
|--------|----------|----------------|
| 2 | Binary decisions | Trend / No Trend |
| 3 | Classic regime detection | Bull / Bear / Sideways |
| 4 | Include crisis detection | Bull / Bear / Sideways / Crisis |
| 5+ | Specialized strategies | Multiple volatility levels |

**Recommendation:** Start with 3 states. Add more only if backtests show clear benefit.

### Retrain Frequency

**Impact on Performance:**

| Frequency | Training Overhead | Adaptability | Best For |
|-----------|------------------|--------------|----------|
| Daily | High (slow) | Very responsive | Day trading, volatile assets |
| Weekly | Medium | Balanced | Most strategies |
| Monthly | Low (fast) | Stable | Long-term strategies |

**With Caching:**
```python
self.enable_caching(
    max_cache_size=200,
    retrain_frequency='monthly'  # Good default with caching
)
```

**Why:** Monthly retraining with caching provides 99% cache hit rate while staying current.

---

## Risk Management

### Position Sizing

**✅ DO: Size by Confidence**

```python
def calculate_position_size(self, regime, confidence):
    """Dynamic position sizing based on regime and confidence."""

    base_allocation = {
        "Bull": 1.0,
        "Bear": 0.0,
        "Sideways": 0.5,
        "Crisis": 0.0
    }

    allocation = base_allocation.get(regime, 0.5)

    # Scale by confidence
    if confidence >= 0.80:
        multiplier = 1.0  # Full position
    elif confidence >= 0.60:
        multiplier = 0.7  # Partial position
    else:
        multiplier = 0.4  # Small position

    return allocation * multiplier
```

**✅ DO: Implement Drawdown Limits**

```python
def OnData(self, data):
    # Check current drawdown
    current_value = self.Portfolio.TotalPortfolioValue
    peak_value = max(self.peak_value, current_value)
    drawdown = (peak_value - current_value) / peak_value

    if drawdown > 0.15:  # 15% drawdown limit
        # Go defensive
        self.Liquidate()
        self.Debug(f"⚠️  Drawdown limit reached: {drawdown:.1%}")
        return

    # Normal trading logic
    result = self.update_regime()
    # ...
```

### Diversification

**✅ DO: Diversify Across Asset Classes**

```python
class DiversifiedStrategy(HiddenRegimeAlgorithm):
    def Initialize(self):
        # Multiple uncorrelated assets
        self.assets = {
            'SPY': 'US Stocks',
            'EFA': 'International Stocks',
            'TLT': 'Bonds',
            'GLD': 'Gold',
            'DBC': 'Commodities'
        }

        for ticker in self.assets.keys():
            self.AddEquity(ticker, Resolution.Daily)
            self.initialize_regime_detection(ticker=ticker, n_states=3)
```

**Why:** Different assets have different regime cycles. Diversification reduces strategy-specific risk.

### Stop Losses

**⚠️  CAUTION: Stops in Regime Strategies**

Regime strategies inherently manage risk through allocation changes. Additional stops can cause issues:

```python
# ❌ Problematic
def OnData(self, data):
    # Regime says "Bull" but stop loss triggers
    # Creates conflict between regime signal and stop
    self.SetHoldings(self.symbol, 1.0)
    self.SetStopLoss(self.symbol, 0.05)  # May trigger in normal volatility
```

**Better approach:**
```python
# ✅ Use regime confidence as "stop"
def OnData(self, data):
    result = self.update_regime()

    # Confidence drop acts as signal to reduce exposure
    if result['confidence'] < 0.5:  # Confidence "stop"
        self.Liquidate()  # Exit position
```

---

## Performance Optimization

### Use Caching for Multi-Asset Strategies

**Impact:** 60-70% faster backtests

```python
class OptimizedStrategy(HiddenRegimeAlgorithmOptimized):
    def Initialize(self):
        # Enable all optimizations
        self.enable_caching(
            max_cache_size=200,  # Cache 200 models
            retrain_frequency='monthly'  # Retrain monthly
        )

        self.enable_batch_updates(
            max_workers=4  # Parallel processing
        )

        self.enable_profiling()  # Track performance
```

**When to use:**
- ✅ Multi-asset strategies (4+ assets)
- ✅ Long backtests (2+ years)
- ✅ Higher resolution data (hourly, minute)

**When not needed:**
- ❌ Single-asset strategies
- ❌ Short backtests (<6 months)
- ❌ Already fast enough

### Batch Updates

**For 8+ assets, use batch processing:**

```python
def OnData(self, data):
    # Update all regimes in parallel
    regime_results = self.batch_update_regimes(self.assets.keys())

    # Process results
    for ticker, result in regime_results.items():
        self.process_regime_signal(ticker, result)
```

**Speedup:** 3.3x for 8 assets with 4 workers

---

## Production Deployment

### Pre-Deployment Checklist

**✅ Backtesting**
- [ ] Backtest on at least 3 years of data
- [ ] Test across different market conditions (bull, bear, crisis)
- [ ] Verify Sharpe ratio > 1.0
- [ ] Check maximum drawdown < 20%
- [ ] Review trade frequency (not too high/low)

**✅ Paper Trading**
- [ ] Run paper trading for 1-3 months
- [ ] Verify regime detection works in real-time
- [ ] Check execution costs and slippage
- [ ] Monitor cache hit rates
- [ ] Validate logging and monitoring

**✅ Risk Controls**
- [ ] Implement position size limits
- [ ] Set maximum drawdown limits
- [ ] Configure emergency liquidation conditions
- [ ] Test risk controls in paper trading

### Monitoring in Production

**Essential Metrics:**

```python
def OnEndOfDay(self):
    # Daily monitoring
    self.Log(f"Current Regime: {self.current_regime}")
    self.Log(f"Regime Confidence: {self.regime_confidence:.1%}")
    self.Log(f"Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
    self.Log(f"Today's Return: {self.today_return:.2%}")

    # Cache performance (if using optimizations)
    if hasattr(self, '_model_cache'):
        stats = self._model_cache.get_statistics()
        self.Log(f"Cache Hit Rate: {stats['hit_rate']:.1%}")
```

### Emergency Procedures

**Define clear liquidation triggers:**

```python
def check_emergency_conditions(self):
    """Check if emergency liquidation needed."""

    # Condition 1: Extreme drawdown
    if self.current_drawdown > 0.25:  # 25%
        self.emergency_liquidate("Extreme drawdown")
        return True

    # Condition 2: System errors
    if self.error_count > 10:
        self.emergency_liquidate("Excessive errors")
        return True

    # Condition 3: Confidence collapse
    if self.regime_confidence < 0.3 and self.current_regime == "Crisis":
        self.emergency_liquidate("Low confidence crisis")
        return True

    return False

def emergency_liquidate(self, reason):
    """Emergency liquidation with logging."""
    self.Log(f"🚨 EMERGENCY LIQUIDATION: {reason}")
    self.Liquidate()
    self.Quit(reason)
```

---

# Part II: Frequently Asked Questions

## General Questions

### Q1: What is regime detection?

**A:** Regime detection identifies distinct market states (regimes) using Hidden Markov Models. Each regime has different statistical properties (volatility, trend, correlation).

**Example regimes:**
- **Bull:** Low volatility, positive trend
- **Bear:** High volatility, negative trend
- **Sideways:** Low volatility, no clear trend
- **Crisis:** Extreme volatility, sharp negative trend

### Q2: How many states should I use?

**A:** Start with 3 states (Bull/Bear/Sideways). This is:
- ✅ Easy to understand
- ✅ Robust across markets
- ✅ Less prone to overfitting

Add a 4th state (Crisis) if you trade through volatile periods (2008, 2020, etc.).

### Q3: What's a good Sharpe ratio for regime strategies?

**A:** Target metrics:
- **Sharpe Ratio:** > 1.0 (good), > 1.5 (excellent)
- **vs Buy-and-Hold:** Should outperform in risk-adjusted terms
- **Max Drawdown:** < 15% (conservative), < 25% (aggressive)

### Q4: Can I use this for day trading?

**A:** Yes, but with modifications:
- Use minute or hour resolution
- Shorter lookback (20-60 bars)
- Higher retraining frequency
- More granular regimes (5-7 states)

**Not recommended for:** Complete beginners, very low capital (<$25k)

### Q5: Does this work with crypto?

**A:** Yes! Crypto often has clear regimes:

```python
self.AddCrypto("BTCUSD", Resolution.Hour)

self.initialize_regime_detection(
    ticker="BTCUSD",
    n_states=4,  # Bull/Bear/Sideways/Crash
    lookback_days=60  # Crypto moves fast
)
```

**Tip:** Crypto regimes change faster than equities. Use shorter lookbacks and higher retrain frequency.

---

## Technical Questions

### Q6: How is regime different from trend following?

| Aspect | Trend Following | Regime Detection |
|--------|----------------|------------------|
| **Method** | Price patterns (MA, MACD) | Statistical modeling (HMM) |
| **States** | Continuous | Discrete |
| **Predictions** | Direction | Market state |
| **Adaptability** | Fixed rules | Learns from data |

**Regime detection is superior when:**
- Markets have distinct behavioral phases
- You want probability-based decisions
- You need to identify "crisis" or "sideways" states

### Q7: What data does the model use?

**A:** By default, the model uses:
- **Close prices** (required)
- **Volume** (optional, improves accuracy)
- **Derived features:** Returns, volatility, momentum

**You can add custom features:**
```python
# In advanced usage
pipeline.add_custom_observations([
    'rsi', 'macd', 'bollinger_bands'
])
```

### Q8: How often should the model retrain?

**Guidelines:**

| Scenario | Retrain Frequency | Rationale |
|----------|------------------|-----------|
| Live trading | Daily | Stay current with market |
| Backtesting (with cache) | Monthly | 99% cache hit, fast |
| Very stable assets | Weekly | Balance speed and accuracy |
| Research/development | On-demand | Train when needed |

**Default recommendation:** Weekly for live, monthly for backtesting with cache.

### Q9: Can I backtest before 2000?

**A:** Yes, but be aware:
- Pre-2000 data may have survival bias
- Market structure was different
- Regime patterns may not apply to modern markets

**Recommendation:** Backtest from 2000+ for US equities, 2010+ for ETFs.

### Q10: How do I add custom indicators?

**A:** Create custom observations:

```python
# In advanced pipeline customization
from hidden_regime.observations import FinancialObservation

class CustomObservation(FinancialObservation):
    def transform(self, data):
        # Add your custom features
        data['custom_indicator'] = self.calculate_custom_metric(data)
        return data
```

Then use in your strategy. See `API_REFERENCE.md` for details.

---

## Performance Questions

### Q11: Why is my backtest slow?

**Common causes:**

1. **Too frequent retraining**
   ```python
   # ❌ Slow
   retrain_frequency='daily'  # Trains every day

   # ✅ Fast
   retrain_frequency='monthly'  # Trains once per month
   ```

2. **No caching**
   ```python
   # ✅ Add caching for 60-70% speedup
   from hidden_regime.quantconnect.optimized_algorithm import HiddenRegimeAlgorithmOptimized

   class FastStrategy(HiddenRegimeAlgorithmOptimized):
       def Initialize(self):
           self.enable_caching(retrain_frequency='monthly')
   ```

3. **Too many assets without batch processing**
   ```python
   # ✅ Enable batch updates for multi-asset
   self.enable_batch_updates(max_workers=4)
   ```

### Q12: What's the expected cache hit rate?

**Targets:**
- **First 10 iterations:** 0-20% (cold cache)
- **After warm-up:** 70-90%
- **With monthly retrain:** 95-99%

**Check your cache:**
```python
stats = self._model_cache.get_statistics()
self.Debug(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

If hit rate < 70% after warm-up, increase `retrain_frequency` to 'monthly'.

### Q13: How much memory does caching use?

**A:** Approximately:
- **Per cached model:** 1-5 MB
- **100 cached models:** 100-500 MB
- **Memory overhead:** ~22% (validated in tests)

**For typical strategies:**
- Single asset: <10 MB
- 4-6 assets: 20-100 MB
- 20+ assets: 200-500 MB

**Limit cache size if needed:**
```python
self.enable_caching(max_cache_size=50)  # Limit to 50 models
```

---

## Troubleshooting

### Q14: Error: "Not enough data to initialize regime detection"

**Cause:** Insufficient historical data for lookback period.

**Solutions:**

1. **Add warmup period:**
   ```python
   def Initialize(self):
       self.SetWarmUp(timedelta(days=252))  # Warm up for lookback period
   ```

2. **Reduce lookback:**
   ```python
   self.initialize_regime_detection(
       ticker="SPY",
       lookback_days=60  # Reduced from 252
   )
   ```

3. **Check data availability:**
   ```python
   history = self.History(self.symbol, 252, Resolution.Daily)
   self.Debug(f"Available bars: {len(history)}")
   ```

### Q15: Regime labels don't match intuition

**Example:** Model says "Bull" during market crash.

**Causes:**
1. **Lookback too long** - Model hasn't adapted yet
2. **Wrong number of states** - 3 states might miss "Crisis"
3. **Label assignment** - Labels are arbitrary (highest return state = "Bull")

**Solutions:**

1. **Shorter lookback for faster adaptation:**
   ```python
   lookback_days=90  # Faster adaptation
   ```

2. **Add more states:**
   ```python
   n_states=4  # Bull/Bear/Sideways/Crisis
   ```

3. **Validate regime characteristics:**
   ```python
   def on_regime_change(self, old_regime, new_regime, confidence):
       # Check if regime matches expected behavior
       recent_return = self.calculate_recent_return()
       recent_vol = self.calculate_volatility()

       self.Debug(f"Regime: {new_regime}")
       self.Debug(f"Return: {recent_return:.2%}, Vol: {recent_vol:.2%}")
   ```

### Q16: Too many regime changes (whipsaws)

**Symptoms:**
- Regime changes daily or weekly
- Poor performance due to transaction costs
- Unclear market signals

**Solutions:**

1. **Increase confidence threshold:**
   ```python
   if result['confidence'] >= 0.75:  # Higher threshold
       # Only trade on high confidence
   ```

2. **Longer lookback:**
   ```python
   lookback_days=504  # 2 years = more stable
   ```

3. **Add hysteresis:**
   ```python
   def on_regime_change(self, old_regime, new_regime, confidence):
       # Require strong signal to change
       if confidence < 0.80:
           # Stay in old regime
           self.current_regime = old_regime
           return
   ```

### Q17: Backtest results don't match paper trading

**Common causes:**

1. **Look-ahead bias in backtest** - Using future data
2. **Execution costs** - Slippage/commissions not modeled
3. **Different data** - Backtest vs live data mismatch

**Solutions:**

1. **Add realistic costs:**
   ```python
   self.SetSecurityInitializer(lambda x: x.SetFeeModel(ConstantFeeModel(1)))
   self.SetSecurityInitializer(lambda x: x.SetSlippageModel(ConstantSlippageModel(0.0001)))
   ```

2. **Use same data source:**
   ```python
   # Ensure backtest and paper trading use same data
   self.SetBrokerageModel(BrokerageName.InteractiveBrokers)
   ```

3. **Account for market impact:**
   ```python
   # For large positions, model price impact
   self.SetSecurityInitializer(lambda x: x.SetSlippageModel(VolumeShareSlippageModel()))
   ```

### Q18: ModuleNotFoundError: No module named 'hidden_regime'

**Cause:** Docker image doesn't include hidden-regime.

**Solution:**

```bash
# Rebuild Docker image
docker build -t quantconnect/lean:hidden-regime -f docker/Dockerfile .

# Ensure LEAN uses correct image
lean config set engine-image quantconnect/lean:hidden-regime

# Verify image
docker images | grep hidden-regime
```

---

## Quick Reference

### Essential Commands

```bash
# Setup
bash scripts/setup_quantconnect.sh

# Backtest
lean backtest MyStrategy

# Logs
cat MyStrategy/backtests/latest/log.txt

# Cleanup
rm -rf MyStrategy/backtests/*
```

### Key Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `n_states` | 3 | 2-7 | More states = more regimes |
| `lookback_days` | 252 | 60-1000 | Longer = more stable |
| `confidence_threshold` | 0.6 | 0.5-0.9 | Higher = fewer trades |
| `retrain_frequency` | weekly | daily/weekly/monthly | More frequent = slower |

### Performance Targets

| Metric | Good | Excellent |
|--------|------|-----------|
| Sharpe Ratio | >1.0 | >1.5 |
| Max Drawdown | <20% | <15% |
| Win Rate | >55% | >60% |
| Cache Hit Rate | >70% | >85% |

---

## Additional Resources

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Quick Start Tutorial](QUICKSTART_TUTORIAL.md)** - 5-minute setup guide
- **[Template Usage Guide](TEMPLATE_USAGE_GUIDE.md)** - Template documentation
- **[Testing Guide](TESTING_GUIDE.md)** - Testing your strategies

---

**Still have questions?** Open an issue on GitHub or check the documentation!
