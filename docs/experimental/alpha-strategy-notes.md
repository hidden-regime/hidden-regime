# Alpha Strategy Notes: HMM-Based Regime Trading

## Honest Assessment of Current HMM Implementation

### What We Built
A **high-quality regime descriptor** - identifies current market state based on returns, volatility, and persistence patterns.

### What It's Good For
- **Risk overlay** - Reduce exposure in crisis/high-vol regimes
- **Conditional strategy execution** - Run different strategies per regime
- **Position sizing** - Scale positions based on regime confidence

### Limitations
- **Descriptive, not predictive** - Identifies regimes after they've established
- **Lag at transitions** - By the time it identifies "bull", first leg is done
- **Markov assumption** - Doesn't account for regime "aging"

### Typical Performance
- Sharpe 0.3-0.6 for naive regime-following strategies
- Main value is drawdown reduction, not return enhancement

---

## Alpha Opportunity: Regime Transition Anticipation

### Core Insight
Don't trade the **current regime** → Trade the **probability of regime change**

The HMM provides transition probabilities and duration statistics that can be combined with stress indicators to anticipate regime shifts before they're obvious.

---

## Strategy Components

### 1. Transition Probability Signal

From the HMM transition matrix:
```python
# Example transition matrix
#         → Bull  Sideways  Bear
# Bull      0.92    0.06    0.02
# Sideways  0.08    0.88    0.04
# Bear      0.05    0.10    0.85

current_regime = "Bull"
P_stay_bull = 0.92
P_exit_bull = 0.08  # Base transition probability
```

### 2. Duration Adjustment (Non-Markov Enhancement)

Standard HMMs are memoryless, but real regimes show duration dependence:

```python
days_in_regime = 40
median_bull_duration = 35  # From historical analysis

# Hazard rate increases as regime ages
duration_factor = days_in_regime / median_bull_duration
adjusted_exit_prob = P_exit_bull * min(duration_factor, 2.0)

# Result: 0.08 * 1.14 = 9.1% adjusted exit probability
```

### 3. Regime Stress Indicators

Early warning signs that current regime is weakening:

| Indicator | Bull Regime Stress | Bear Regime Stress |
|-----------|-------------------|-------------------|
| Volatility | 20-day vol > 60-day vol | Vol compression |
| Momentum | RSI divergence from price | Selling exhaustion |
| Breadth | Fewer stocks making highs | Washout readings |
| Credit | Spreads widening | Spreads tightening |
| Volume | Declining on rallies | Climactic selling |

### 4. Composite Transition Score

```python
transition_score = (
    0.30 * adjusted_exit_prob +      # HMM transition + duration
    0.30 * vol_stress_signal +        # Volatility regime stress
    0.20 * momentum_divergence +      # Technical divergence
    0.20 * credit_spread_change       # Cross-asset confirmation
)
```

---

## Position Sizing Logic

```python
if current_regime == "Bull":
    position = base_position * (1 - transition_score)
    # High transition risk → reduce exposure before the move

elif current_regime == "Bear":
    short_position = base_short * (1 - transition_score)
    # High transition risk → cover shorts before reversal
```

---

## Trading Rules

| Scenario | Transition Score | Action |
|----------|-----------------|--------|
| Bull regime | Low (< 0.15) | Full long position |
| Bull regime | Medium (0.15-0.35) | Reduce to 75%, tighten stops |
| Bull regime | High (> 0.35) | Reduce to 50%, add put hedges |
| Transition detected | N/A | Scale out over 3-5 days |
| Sideways regime | Rising bull prob | Accumulate long exposure |
| Bear regime | High (> 0.35) | Cover shorts, go neutral |

---

## Implementation Sketch

```python
class RegimeTransitionStrategy:
    def __init__(self, hmm_model):
        self.hmm = hmm_model
        self.regime_tracker = RegimeDurationTracker()

    def get_signal(self, current_data):
        # 1. Get HMM state and probabilities
        regime, probs = self.hmm.predict_proba(current_data)
        transition_prob = 1 - probs[regime]

        # 2. Adjust for duration (break Markov assumption)
        duration = self.regime_tracker.days_in_regime
        median_dur = self.regime_tracker.median_duration(regime)
        duration_factor = duration / median_dur if median_dur > 0 else 1.0

        # 3. Calculate stress indicators
        vol_stress = self.calc_vol_stress(current_data)
        momentum_div = self.calc_momentum_divergence(current_data)

        # 4. Composite signal
        exit_probability = (
            0.4 * transition_prob * duration_factor +
            0.3 * vol_stress +
            0.3 * momentum_div
        )

        return {
            'regime': regime,
            'confidence': probs[regime],
            'transition_risk': exit_probability,
            'position_scalar': 1 - min(exit_probability, 0.8),
            'days_in_regime': duration,
            'regime_age_percentile': duration / median_dur
        }

    def calc_vol_stress(self, data):
        """Rising short-term vol vs long-term = regime stress."""
        vol_20 = data['returns'].tail(20).std() * np.sqrt(252)
        vol_60 = data['returns'].tail(60).std() * np.sqrt(252)
        return max(0, (vol_20 / vol_60) - 1)  # 0 if vol contracting

    def calc_momentum_divergence(self, data):
        """Price making highs but RSI not confirming."""
        # Placeholder - would use actual RSI divergence logic
        pass
```

---

## Where Alpha Comes From

1. **Exit before the crowd** - HMM + duration + stress indicators provide 1-3 day lead time on regime transitions

2. **Avoid whipsaws** - Don't flip positions on every regime change; use probability thresholds to filter noise

3. **Asymmetric positioning** - Full position in high-confidence regimes, reduced in uncertain periods

4. **Cross-asset confirmation** - Credit spreads, vol surface, breadth provide independent confirmation

---

## Expected Performance

Realistic expectations for a well-implemented version:

| Metric | Naive Regime Following | Transition Anticipation |
|--------|----------------------|------------------------|
| Sharpe Ratio | 0.30 - 0.40 | 0.60 - 0.80 |
| Max Drawdown | -25% to -35% | -15% to -25% |
| Win Rate | 35-40% | 40-45% |
| Avg Win/Avg Loss | 1.5x | 1.8x |

**Note**: These are realistic targets, not guarantees. Actual performance depends on parameter tuning, transaction costs, and market conditions.

---

## Key Risks

1. **Overfitting** - Duration statistics from limited regime samples
2. **Non-stationarity** - 2008 transitions ≠ 2023 transitions
3. **Crowding** - If everyone uses similar signals, edge erodes
4. **Transaction costs** - Frequent position adjustments eat returns

---

## Next Steps to Explore

1. **Backtest duration effect** - Do regime transitions become more likely as regimes age? (Break Markov assumption empirically)

2. **Stress indicator research** - Which indicators provide earliest/most reliable warning?

3. **Cross-asset signals** - Credit spreads, VIX term structure, equity-bond correlation

4. **Position sizing optimization** - Kelly criterion or risk parity based on regime confidence

5. **Walk-forward validation** - Ensure no lookahead bias in transition prediction

---

## References for Further Research

- Hamilton (1989) - Original regime-switching model
- Ang & Bekaert (2002) - Regime switches and stock returns
- Guidolin & Timmermann (2007) - Asset allocation with regime shifts
- Kritzman et al. (2012) - Regime shifts and asset allocation

---

*Document created: 2024-01-XX*
*Status: Conceptual - not yet implemented or backtested*
