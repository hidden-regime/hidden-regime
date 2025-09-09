# Hidden Regime: Quick Reference Guide

*Essential cheat sheet for retail traders using regime-based strategies*

---

## 🚀 Quick Start (5 Minutes)

```python
import hidden_regime as hr

# 1. Load your data
data = hr.load_stock_data('AAPL', '2023-01-01', '2023-12-31')

# 2. Detect regimes  
states, model = hr.detect_regimes(data['log_return'], return_model=True)

# 3. Get current regime
current_regime = states[-1]
current_probs = model.predict_proba(data['log_return'][-20:])[-1]

print(f"Current regime: {current_regime}")
print(f"Probabilities: {current_probs}")
```

---

## 📊 Model Selection Cheat Sheet

| Asset Class | States | Forgetting | Adaptation | Training Days |
|-------------|--------|------------|------------|---------------|
| **Equities** | 3 | 0.98 | 0.05 | 250+ |
| **Crypto** | 4 | 0.95 | 0.08 | 100+ |
| **FX** | 3 | 0.97 | 0.06 | 300+ |
| **Commodities** | 3 | 0.98 | 0.05 | 250+ |

---

## ⚖️ Position Sizing Rules

### Base Position by Regime
```python
positions = {
    # Equities (3-state)
    'bear': -0.3,      # Short/defensive
    'sideways': 0.3,   # Conservative long  
    'bull': 0.8,       # Strong long
    
    # Crypto (4-state)
    'crisis': 0.1,     # Small long
    'bear': 0.2,       # Small long
    'sideways': 0.5,   # Moderate
    'bull': 0.8        # Large position
}
```

### Confidence Adjustments
- **High (>80%)**: Full position
- **Medium (60-80%)**: 70% of position  
- **Low (<60%)**: 50% or cash

### Volatility Scaling
```python
vol_adjusted_position = base_position * min(target_vol / current_vol, 2.0)
```

---

## 🚨 Risk Management Limits

### Maximum Positions
- **Conservative**: 60% long, 20% short
- **Moderate**: 80% long, 30% short
- **Aggressive**: 90% long, 50% short

### Stop Losses
- **Equities**: 5% stop, 20% max drawdown
- **Crypto**: 15% stop, 40% max drawdown
- **FX**: 3% stop, 15% max drawdown

### Emergency Rules
```python
if drawdown > max_drawdown_limit:
    position = 0  # Go to cash
elif confidence < 0.6:
    position *= 0.5  # Reduce risk
```

---

## 📈 Performance Benchmarks

### Expected Returns (Regime Strategy vs Buy-Hold)

| Asset | Strategy Return | Buy-Hold Return | Max Drawdown | Sharpe |
|-------|-----------------|------------------|--------------|--------|
| **AAPL** | 15% | 55% | -16% | 0.13 |
| **DOGE** | 36% | 2,899% | -65% | 0.67 |
| **SPY** | 12% | 18% | -12% | 0.45 |

### When Regime Strategies Excel
- ✅ **Bear markets**: 20-40% outperformance
- ✅ **Volatile periods**: Superior risk-adjusted returns
- ✅ **Crisis events**: Excellent downside protection
- ❌ **Strong bull runs**: May underperform buy-hold

---

## 🔄 Rebalancing Schedule

### Model Updates
- **Equities**: Monthly
- **Crypto**: Weekly  
- **FX**: Bi-weekly

### Position Adjustments
- **Daily**: Check regime probabilities
- **Immediate**: When confidence drops below 60%
- **Emergency**: When drawdown exceeds limits

---

## 🎯 Regime Identification Quick Check

### Bear Regime Signals
- Negative mean returns
- High volatility
- Low regime confidence
- **Action**: Defensive/short positions

### Bull Regime Signals  
- Positive mean returns
- Moderate volatility
- High persistence
- **Action**: Large long positions

### Crisis Regime Signals
- Very negative returns
- Extreme volatility
- Short duration
- **Action**: Cash/hedged

### Sideways Regime Signals
- Near-zero returns
- Low volatility
- Long persistence
- **Action**: Range trading

---

## ⚠️ Common Mistakes to Avoid

### 1. Over-Optimization
```python
# ❌ Don't do this
model.fit(all_historical_data)  # Overfitted

# ✅ Do this instead  
model = walk_forward_validate(data, window=252)
```

### 2. Ignoring Transaction Costs
```python
# ✅ Always include costs
net_return = gross_return - (position_change * commission_rate)
```

### 3. Regime Switching
```python
# ✅ Add stability filters
if regime_confidence < 0.6:
    maintain_current_position()
```

### 4. Position Sizing Errors
```python
# ❌ Fixed position sizes
position = 0.8  # Always 80%

# ✅ Dynamic sizing
position = base_position * confidence * vol_adjustment
```

---

## 📱 Daily Trading Checklist

### Morning Routine (5 minutes)
1. ☐ Download latest price data
2. ☐ Update regime probabilities  
3. ☐ Check model confidence level
4. ☐ Calculate recommended position
5. ☐ Review current drawdown

### Position Decision Tree
```
High Confidence (>80%)?
├── Yes → Use full regime position
└── No → Confidence 60-80%?
    ├── Yes → Use 70% of position
    └── No → Use 50% or go to cash
```

### End of Day (2 minutes)
1. ☐ Record performance
2. ☐ Update rolling statistics
3. ☐ Check for regime transitions
4. ☐ Set alerts for tomorrow

---

## 🔧 Troubleshooting

### Model Not Working?
- ✅ Check sufficient training data (6+ months)
- ✅ Verify data quality (no missing values)
- ✅ Confirm regime persistence (avg 3+ days)
- ✅ Review parameter settings for asset class

### Too Many Regime Switches?
- ✅ Increase confidence threshold (0.6 → 0.7)
- ✅ Add regime stability penalty
- ✅ Use longer smoothing windows

### Poor Performance?
- ✅ Check transaction costs
- ✅ Verify position sizing logic
- ✅ Review stop-loss levels
- ✅ Consider regime correlation with market

### Low Confidence?
- ✅ Increase training window
- ✅ Check for structural breaks
- ✅ Consider multi-state models
- ✅ Add volatility filters

---

## 📞 Emergency Protocols

### Market Crash Scenario
1. **Immediate**: Reduce all positions by 50%
2. **Monitor**: Regime detection for crisis signals
3. **Wait**: For confidence > 70% before re-entering
4. **Review**: Model parameters post-crisis

### Flash Crash / Extreme Move
1. **Pause**: All automated trading
2. **Assess**: Data quality and outliers
3. **Verify**: Regime transition legitimacy
4. **Resume**: Only after manual review

### System Failure
1. **Backup**: Maintain manual position records
2. **Fallback**: Use simple moving average signals
3. **Recovery**: Restart with fresh calibration
4. **Audit**: Check all position changes

---

## 📚 Code Templates

### Basic Strategy Template
```python
class RegimeStrategy:
    def __init__(self, asset_class='equity'):
        self.model = self._setup_model(asset_class)
        self.position = 0
        self.confidence_threshold = 0.6
    
    def update(self, new_return):
        # Update model
        regime_info = self.model.add_observation(new_return)
        
        # Calculate position
        if regime_info['confidence'] > self.confidence_threshold:
            self.position = self._calculate_position(regime_info)
        else:
            self.position *= 0.5  # Reduce risk
            
        return self.position
```

### Risk Management Template
```python
class RiskManager:
    def __init__(self, max_drawdown=0.2, stop_loss=0.05):
        self.max_drawdown = max_drawdown
        self.stop_loss = stop_loss
        self.peak_capital = 0
        
    def check_limits(self, current_capital, current_position):
        # Calculate drawdown
        self.peak_capital = max(self.peak_capital, current_capital)
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        
        # Apply limits
        if drawdown > self.max_drawdown:
            return 0  # Emergency stop
        elif drawdown > self.max_drawdown * 0.8:
            return current_position * 0.5  # Reduce risk
        else:
            return current_position
```

---

## 🔗 Quick Links

- **Main Documentation**: `/docs/`
- **Case Studies**: `/examples/retail_use_cases/`
- **API Reference**: `/hidden_regime/__init__.py`
- **Troubleshooting**: `/docs/troubleshooting_guide.md`

---

*Keep this guide handy during trading sessions. Remember: Regime-based trading excels at risk management, not maximum returns!*