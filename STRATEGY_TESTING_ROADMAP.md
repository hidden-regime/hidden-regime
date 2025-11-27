# Strategy Testing Roadmap Development

## Executive Summary

This document outlines a **4-phase progression** for developing HMM-based quantitative trading strategies, advancing from simple univariate regime detection to sophisticated multi-timeframe, multi-asset systems targeting Sharpe ratios of 7-10+.

**Key Insight**: The roadmap progresses through complexity **multiplicatively**, where each phase builds validated components from the previous phase rather than replacing them.

**Realistic Expectations**:
- Sharpe 1-2 is achievable and respectable
- Sharpe 3+ is world-class
- Sharpe 7-10+ is possible but extremely rare in live trading
- Focus on robustness across market cycles, not peak Sharpe

---

## Phase 1: Univariate Foundation (Sharpe 0.5-1.0)

**Timeline**: 1-2 weeks | **Effort**: 20-30 hours | **Validation Period**: 6-12 months historical data

### Strategy Design

**Core Logic**: Simple trend-following based on regime detection
```
IF regime == BULLISH AND confidence >= 70% THEN
    position_size = 100%
    side = LONG
ELIF regime == BEARISH AND confidence >= 70% THEN
    position_size = 100%
    side = SHORT
ELSE
    position_size = 0%
    side = FLAT
END IF

# Position management
IF position is open:
    - Hard stop loss: -5% from entry
    - Profit target: None (ride the regime)
    - Hold minimum: 5 days
```

### Validation Criteria

**Must Pass All**:
- ✓ Sharpe ratio ≥ 0.50 on **3 out of 5 test assets**
  - SPY (broad market)
  - QQQ (technology)
  - IWM (small-cap)
  - BTC (cryptocurrency)
  - GLD (commodity)
- ✓ Temporal isolation verified (TemporalController shows no lookahead bias)
- ✓ Maximum drawdown ≤ 30%
- ✓ Win rate ≥ 40%
- ✓ Transaction costs included (5 bps slippage + 2 bps spread + 1 bp commission)
- ✓ Minimum 50 trades per asset over test period

**Why These Assets?** They have different regime characteristics:
- SPY: Broad market (strong trends)
- QQQ: Tech-heavy (high volatility)
- IWM: Smaller constituents (regime reversal frequency)
- BTC: Crypto correlation (different market factors)
- GLD: Non-correlated asset (proves generalizability)

### Implementation Checklist

1. **Data Preparation**
   - [ ] Download 2 years historical data (training + validation)
   - [ ] Verify no missing data or gaps
   - [ ] Calculate log returns (not raw prices)

2. **Model Training**
   - [ ] Initialize with 3-state HMM (BULLISH, SIDEWAYS, BEARISH)
   - [ ] Train on first 12 months (avoid forward bias)
   - [ ] Use KMeans initialization

3. **Backtesting**
   - [ ] Use TemporalController for V&V compliance
   - [ ] Step through time day-by-day
   - [ ] Record: entry/exit prices, confidence scores, regime transitions
   - [ ] Include actual transaction costs

4. **Validation**
   - [ ] Generate metrics: Sharpe, max DD, win rate, trade count
   - [ ] Compare against baseline (buy-and-hold)
   - [ ] Run Walk-Forward test (retrain monthly, test out-of-sample)

### Success Outcome

If Phase 1 passes:
- You have a validated baseline
- You understand the codebase
- You're ready for Phase 2

If Phase 1 fails:
- Debug regime detection quality (confusion matrix)
- Increase confidence threshold to 75-80%
- Try 4-state HMM (add CRISIS state)
- Revisit stop-loss placement

---

## Phase 2: Multivariate Enhancement (Sharpe 1.0-2.0)

**Timeline**: 2-3 weeks | **Effort**: 30-50 hours | **Build On**: Phase 1 infrastructure

### Strategy Evolution

**Add Features** to regime detection:
1. **Volume Change** (regime confirmation)
   - Rising volume in bull regime = strength
   - Falling volume in bull regime = warning sign

2. **Realized Volatility** (crisis detection)
   - Spike in volatility during transitions = crisis
   - Helps distinguish crisis from normal bear regimes

**Multivariate Observation Vector**:
```python
observation = [
    log_return,           # Primary signal
    volume_change,        # Confirmation
    realized_volatility,  # Risk measurement
]

# Train 4-state HMM (better than 3-state)
states = [CRISIS, BEARISH, SIDEWAYS, BULLISH]
```

### Position Sizing Evolution

**Phase 1**: Fixed 100%
```
position_size = 100% (always full)
```

**Phase 2**: Confidence-weighted
```
position_size = base_size * (confidence - 0.50) / 0.50
# At 50% confidence: 0% position
# At 70% confidence: 40% position
# At 90% confidence: 100% position
```

This lets you trade lighter positions in low-confidence regimes.

### Validation Criteria

**Must Show Improvement**:
- ✓ Sharpe ratio ≥ 1.00 (improvement of +0.50 from Phase 1)
- ✓ Each feature adds ≥ 0.30 Sharpe individually
  - Test: Train HMM with only log_return, note Sharpe
  - Then add volume_change, measure incremental Sharpe gain
  - Then add volatility, measure incremental Sharpe gain
- ✓ Walk-forward out-of-sample degradation < 40%
  - (In-sample Sharpe - OOS Sharpe) / In-sample Sharpe < 0.40
- ✓ Win rate improvement of +5 percentage points vs Phase 1
- ✓ Maximum drawdown ≤ 25% (improvement vs Phase 1)

### Key Technique: Feature Contribution Analysis

```python
def feature_contribution(asset):
    # Baseline (log returns only)
    s1 = backtest_with_features([log_return])
    sharpe_1 = s1.sharpe

    # Add volume change
    s2 = backtest_with_features([log_return, volume_change])
    sharpe_2 = s2.sharpe
    contribution_volume = sharpe_2 - sharpe_1

    # Add volatility
    s3 = backtest_with_features([log_return, volume_change, volatility])
    sharpe_3 = s3.sharpe
    contribution_volatility = sharpe_3 - sharpe_2

    # Only keep features that contribute > 0.30 Sharpe
    return {
        'volume_change': contribution_volume,
        'volatility': contribution_volatility,
    }
```

**Critical Rule**: If a feature doesn't add ≥ 0.30 Sharpe, **remove it**. More features don't mean better performance.

### Implementation Checklist

1. **Feature Engineering**
   - [ ] Calculate volume_change (volume[t] / volume_MA_20)
   - [ ] Calculate realized_volatility (std of log returns over 20 days)
   - [ ] Normalize features to mean 0, std 1

2. **Model Training**
   - [ ] Train 4-state HMM with multivariate observations
   - [ ] Use ObservationMode.MULTIVARIATE
   - [ ] Train on first 12 months

3. **Feature Validation**
   - [ ] Test feature contribution individually
   - [ ] Keep only features where contribution > 0.30 Sharpe

4. **Backtesting**
   - [ ] Use TemporalController (same as Phase 1)
   - [ ] Record feature values at each decision point

5. **Walk-Forward Validation**
   - [ ] Retrain model every 3 months
   - [ ] Test on 1 month out-of-sample
   - [ ] Measure degradation < 40%

### Success Outcome

If Phase 2 passes:
- You have feature engineering working
- You understand multivariate HMMs
- Sharpe improved by +0.5
- Ready for Phase 3

If Phase 2 fails:
- Features may be noisy (try wider lookback windows)
- Model may be overfit (reduce to 3 states)
- Volume data may be unreliable (use alternative like price range)

---

## Phase 3: Multi-Timeframe Filtering (Sharpe 2.0-4.0)

**Timeline**: 2-3 weeks | **Effort**: 30-40 hours | **Build On**: Phase 1-2 foundation

### The Multi-Timeframe Concept

**Problem**: Single timeframe regimes can be noisy. A daily rally might reverse within hours.

**Solution**: Train independent HMMs on multiple timeframes, only trade when they **align**.

```python
# Train three independent models
daily_regime = train_hmm(prices_daily, n_states=3)
weekly_regime = train_hmm(prices_weekly, n_states=3)
monthly_regime = train_hmm(prices_monthly, n_states=3)

# Calculate alignment score (0-1)
# alignment = fraction of timeframes agreeing
alignment = 1.0 if (daily == weekly == monthly == BULLISH) else 0.67 else 0.33 else 0.0

# Only trade high-conviction signals
IF alignment >= 0.70 AND daily_confidence >= 70% THEN
    position_size = 100%
ELSE
    position_size = 0%
END IF
```

### Why Multi-Timeframe Works

**Signal Filtering Effect**:
- Daily signals: ~500/year (noise)
- Daily + weekly alignment: ~200/year (filtered)
- Daily + weekly + monthly alignment: ~80/year (high conviction)

**Win Rate Improvement**:
- Daily-only win rate: 45%
- Aligned signals win rate: 60-75% (15-30 ppt improvement!)

**Expected Trade Reduction**: 60-70% fewer trades (saves on transaction costs)

### Position Sizing Evolution

```
Phase 1: Fixed 100%
Phase 2: Confidence-weighted
Phase 3: Alignment × Confidence weighted

position_size = base_size × alignment_score × (confidence - 0.50) / 0.50
# At alignment=0.67, confidence=70%: 0.67 × 0.40 × 40% = 10.7%
# At alignment=1.00, confidence=90%: 1.00 × 1.00 × 100% = 100%
```

### Validation Criteria

**Must Show Improvement**:
- ✓ Sharpe ratio ≥ 2.00 (improvement of +1.0 from Phase 2)
- ✓ Signal reduction ≥ 60%
  - Baseline (all signals): 200 trades/year
  - With alignment filter (≥0.70): ≤ 80 trades/year
- ✓ Win rate ≥ 55% (improvement of +10 ppts from Phase 1)
- ✓ Win rate (aligned signals) > Win rate (all signals) by ≥ 15 ppts
- ✓ Maximum drawdown ≤ 20% (further improvement)

### Critical Insight: The Signal Reduction Paradox

Fewer trades = Better performance

Why? Because:
1. Transaction costs drop dramatically
2. Win rate improves (only high-conviction setups)
3. Sharpe improves via both return and volatility reduction

**Expected Result**: 2-3x fewer trades, 3-5x better Sharpe.

### Implementation Checklist

1. **Data Preparation**
   - [ ] Resample daily data to weekly and monthly
   - [ ] Ensure date alignment (Friday close for weekly, month-end for monthly)

2. **Model Training**
   - [ ] Train 3 independent HMMs (daily, weekly, monthly)
   - [ ] Use same feature set from Phase 2 for each
   - [ ] Use MultiTimeframeRegime class

3. **Alignment Scoring**
   - [ ] Calculate alignment score each day
   - [ ] Record alignment history
   - [ ] Analyze: at what alignment threshold does win rate spike?

4. **Backtesting**
   - [ ] Test with alignment thresholds: 0.50, 0.67, 0.75, 1.00
   - [ ] Find optimal threshold (usually 0.67-0.75)
   - [ ] Measure signal reduction vs win rate improvement tradeoff

5. **Validation**
   - [ ] Walk-forward test (retrain monthly)
   - [ ] Ensure daily/weekly/monthly models stay synchronized
   - [ ] Verify out-of-sample alignment degrades < 30%

### Success Outcome

If Phase 3 passes:
- You have a high-conviction trading system
- Sharpe ≥ 2.0, often 2.5-3.0
- Win rate 55-65%
- Trading ~80-100 times/year (manageable)
- Ready for Phase 4

If Phase 3 fails:
- Alignment filtering may not work on your assets (check correlations)
- Try alignment thresholds of 0.60 or 0.75 instead of 0.70
- Consider only daily + weekly (drop monthly)

---

## Phase 4: Advanced Integration (Sharpe 5.0-10.0+)

**Timeline**: 4-6 weeks | **Effort**: 60-100 hours | **Build On**: All previous phases

### Integrated Multi-Strategy System

Phase 4 is not a single strategy, but a **framework combining multiple approaches**:

```
Core Strategy (60% of alpha):
├─ Phase 3 multi-timeframe + multivariate
├─ Momentum-based position sizing
└─ Base Kelly sizing

+ Mean Reversion Overlay (30% of alpha):
├─ Fade regime overshoots
├─ Trade regime transitions
└─ Duration-based adjustments

+ Dynamic Risk Management (10% of alpha):
├─ Kelly criterion position sizing
├─ Correlation-based hedging
├─ Drawdown circuit breakers
└─ Volatility targeting
```

### Component 1: Enhanced Trend-Following Core

**What to keep from Phase 3**:
- Multi-timeframe alignment filtering (the money maker)
- Multivariate regime detection
- Confidence weighting

**What to add**:
- Momentum confirmation (price trend within regime)
- Transition detection (entering a new regime)

```python
# Enhanced position sizing
momentum = (price[t] - price[t-20]) / price[t-20]

IF alignment >= 0.70 AND regime == BULLISH AND confidence >= 70% THEN
    # Boost position if momentum is strong
    position_size = base_size × alignment × (confidence - 0.50) / 0.50 × (1.0 + min(momentum, 0.05))
ELSE
    position_size = 0%
END IF
```

### Component 2: Mean Reversion Overlay

**When to use**: Add 30% to position sizing during regime transitions

```python
IF (previous_regime != BULLISH) AND (current_regime == BULLISH) THEN
    # Entering a bullish regime: add mean reversion position
    # Buy the dip that started the transition
    position_size += 0.30 × base_size
ELIF (regime == BULLISH) AND (rsi > 80) THEN
    # Overbought condition: scale back
    position_size *= 0.70
END IF
```

**Key Insight**: Regimes have "momentum" - they tend to last 1-3 months. The first week of a new regime is often a good entry point because:
- Others are still in the old regime
- Price can mean-revert within the new regime
- Alignment score is still valid

### Component 3: Dynamic Risk Management

#### Kelly Criterion Position Sizing

```python
# Based on historical win rate and win/loss ratios
win_rate = 0.65  # From Phase 3 validation
avg_win_pct = 2.5%
avg_loss_pct = 1.0%
odds = avg_win_pct / avg_loss_pct  # 2.5

kelly_fraction = (win_rate * odds - (1 - win_rate)) / odds
kelly_fraction = (0.65 × 2.5 - 0.35) / 2.5 = 0.49

# Use half-Kelly for safety (Kelly is optimal but max volatility)
position_size = base_size × kelly_fraction × 0.50
position_size = base_size × 0.245  # 24.5% of account per trade
```

#### Correlation-Based Hedging

```python
# If you trade multiple assets
correlation_matrix = calculate_correlations(assets)

# Reduce position if too correlated
IF correlation(SPY, QQQ) > 0.85 THEN
    # Both tech-exposed, reduce combined leverage
    position_SPY *= 0.70
    position_QQQ *= 0.70
END IF
```

#### Drawdown Circuit Breakers

```python
# Hard stops to prevent catastrophic losses
IF max_drawdown > -12% THEN
    position_size *= 0.50  # Cut position in half
END IF
IF max_drawdown > -18% THEN
    position_size = 0%     # Flatten everything, stop trading
END IF
```

### Validation Criteria (Extremely Strict)

**Must Show Exceptional Results**:
- ✓ Sharpe ratio ≥ 5.00 (target 7-10, but 5.0 = world-class)
- ✓ Maximum drawdown ≤ 15% (must protect capital)
- ✓ Win rate ≥ 65% (stricter than Phase 3)
- ✓ Profit factor (gross profit / gross loss) ≥ 2.5
- ✓ Monte Carlo simulation: 5th percentile Sharpe ≥ 3.0
  - Run 1000 random trade shuffles
  - 95% of permutations still have Sharpe ≥ 3.0
  - Indicates robustness to luck/randomness
- ✓ Works across market cycles (2008 crisis, 2015 correction, 2020 COVID, 2022 bear)
  - Sharpe > 1.0 in each period
  - Proves not overfitted to one regime
- ✓ Out-of-sample degradation < 30%
  - Walk-forward tested rigorously

### Implementation Checklist

1. **Integration Testing**
   - [ ] Combine Phase 3 core + mean reversion + risk management
   - [ ] Test each component contribution (remove, measure Sharpe change)
   - [ ] Ensure components don't conflict

2. **Parameter Optimization**
   - [ ] Kelly fraction (1/2-Kelly vs full Kelly)
   - [ ] Position cap (max 30% of account per trade?)
   - [ ] Drawdown thresholds for circuit breakers

3. **Robustness Testing**
   - [ ] Walk-forward validation (retrain quarterly)
   - [ ] Out-of-sample testing (held out data)
   - [ ] Multiple market cycles

4. **Monte Carlo Analysis**
   - [ ] Shuffle trade order 1000 times
   - [ ] Calculate Sharpe for each shuffle
   - [ ] Is strategy robust or dependent on lucky order?

5. **Market Cycle Testing**
   - [ ] 2007-2009 financial crisis
   - [ ] 2015 flash crash
   - [ ] 2020 COVID pandemic
   - [ ] 2022 bear market / rate hikes
   - [ ] Each should be Sharpe > 1.0

### Success Outcome

If Phase 4 passes:
- You have a professional-grade strategy
- Sharpe 5.0+ is world-class
- Ready for live trading (paper trade first!)
- Can be deployed on real capital

If Phase 4 fails:
- Sharpe < 5.0 is still good if > 3.0
- Mean reversion may not work (remove it)
- Risk management may be too strict (loosen circuit breakers)
- **This is OK** - Sharpe 3.0 in live trading is success

---

## Common Pitfalls & Solutions

### Pitfall 1: Lookahead Bias (THE WORST)

**What is it?** Using future information to make past decisions.

**Example**:
```python
# WRONG - Uses today's close to predict today's regime
IF close[today] > close[yesterday] THEN
    # Enter trade at open[today] - BUT YOU USED TODAY'S CLOSE!
    enter_trade()
```

**Solution**:
```python
# RIGHT - Use only data available at trade time
IF close[yesterday] > close[day_before] THEN
    # At open[today], enter trade using only yesterday's data
    enter_trade()

# BETTER - Use TemporalController
controller = create_temporal_controller(pipeline, data)
for date in trading_dates:
    # Only past data available at 'date'
    result = controller.update_as_of(date)
    audit = controller.get_access_audit()
    assert not audit.has_future_access()  # Verify no leakage
```

### Pitfall 2: Overfitting to Historical Regimes

**What is it?** Creating rules that work great on training data but fail on new data.

**Example**:
```python
# WRONG - Optimized parameters to specific regime
IF rsi > 83.7 AND confidence > 71.2% THEN  # Too precise!
    enter_trade()
```

**Solution**:
```python
# RIGHT - Use round thresholds with margin of safety
IF rsi > 80 AND confidence >= 70% THEN  # Nice round numbers
    enter_trade()

# BETTER - Test parameter sensitivity
for threshold in [75, 80, 85, 90]:
    for confidence in [60%, 65%, 70%, 75%]:
        sharpe = backtest(threshold, confidence)
        print(f"RSI {threshold}, Conf {confidence}: {sharpe}")
# Parameters should be robust across range
```

### Pitfall 3: Curve-Fitting in Mean Reversion

**What is it?** Over-optimizing reversion settings to specific market conditions.

**Solution**:
- Use simple heuristics (fade when RSI > 80)
- Avoid complex rules (fade when RSI > 73.2 AND MACD < -0.45)
- Test on 3+ market regimes before deployment

### Pitfall 4: Over-Trading in Sideways Markets

**What is it?** Taking too many signals in low-volatility periods when regimes are unclear.

**Example**:
```
Sideways market: BULLISH → SIDEWAYS → BULLISH → SIDEWAYS
Result: 5 trades/week at 45% win rate = slow bleed
```

**Solution**:
```python
# Require high confidence AND minimum holding period
IF confidence >= 70% AND time_since_entry >= 5 days THEN
    # Only allow entry if last trade is 5+ days old
    enter_trade()

# Skip trading entirely in SIDEWAYS regime
IF regime == SIDEWAYS THEN
    position_size = 0%
END IF
```

### Pitfall 5: Position Sizing Gone Wrong

**What is it?** Leverage blowup when correlations break down.

**Example**:
```
Assume SPY and QQQ correlation = 0.90
Size both at 50% → effective 100% leverage
Then correlation spikes to 0.99 → one move hits both
Result: Catastrophic loss
```

**Solution**:
```python
# Use Kelly criterion, not fixed leverage
kelly = 0.245  # From Phase 4 analysis
position_size = base_size × kelly

# Add correlation limits
max_beta = 1.5  # Max portfolio beta to market
IF portfolio_beta > max_beta THEN
    scale_all_positions = max_beta / portfolio_beta
END IF
```

### Pitfall 6: Ignoring Transaction Costs

**What is it?** Backtests show great returns, but live trading is much worse.

**Example**:
- Backtest assumes: 0 slippage, 0 spread, 0 commission
- Live reality: 5 bps slippage + 2 bps spread + 1 bp commission = 8 bps per round trip
- 100 trades/year × 8 bps = 80 bps drag on returns
- Turns Sharpe 2.0 into Sharpe 1.0!

**Solution**:
```python
# Model realistic costs
slippage_bps = 5      # Entry/exit price impact
spread_bps = 2        # Bid-ask spread
commission_bps = 1    # Broker commission

transaction_cost = (slippage_bps + spread_bps + commission_bps) / 10000
annual_trades = 100
annual_cost = transaction_cost × 2 × annual_trades  # 2 = round trip
# = 0.0008 × 200 = 0.16 = 1.6% annual drag

sharpe_after_costs = sharpe - (annual_cost / volatility)
```

### Pitfall 7: Insufficient Sample Size

**What is it?** Declaring success on too few trades.

**Example**:
- Backtest: 20 trades, 75% win rate, Sharpe 3.0
- Live: 200 trades, 50% win rate, Sharpe 0.5
- Why? Variance of small samples is huge

**Solution**:
- Phase 1: Minimum 50 trades per asset
- Phase 2: Minimum 100 trades per asset
- Phase 3: Minimum 80 total trades (across all assets)
- Phase 4: Minimum 200 total trades before declaring success

### Pitfall 8: Regime Label Confusion

**What is it?** Using raw HMM state numbers instead of interpreted labels.

**Example**:
```python
# WRONG - State 0 might be bull OR bear depending on training
IF state == 0 THEN
    go_long()
```

**Solution**:
```python
# RIGHT - Use interpreted regime labels
IF regime == RegimeType.BULLISH THEN
    go_long()
```

---

## Risk Management Evolution

| Phase | Position Sizing | Stop Loss | Max DD | Circuit Breaker |
|-------|-----------------|-----------|--------|-----------------|
| **1** | Fixed 100% | Hard -5% | 30% | None |
| **2** | Confidence-weighted | Hard -5% | 25% | None |
| **3** | Alignment × confidence | Hard -4% | 20% | Max -12% → 50% reduce |
| **4** | Kelly × alignment × confidence | Varies | 15% | -12% → 50%, -18% → flat |

---

## Expected Sharpe Progression

| Phase | Min Sharpe | Target Sharpe | Achievement Rate |
|-------|-----------|---------------|------------------|
| **1** | 0.50 | 0.75 | 80% (achievable) |
| **2** | 1.00 | 1.50 | 70% (moderately hard) |
| **3** | 2.00 | 2.50 | 60% (hard) |
| **4** | 5.00 | 7.00 | 30% (very hard) |

**Key**: If you achieve Phase 3 (Sharpe 2.0-2.5), you have an exceptional strategy. Phase 4 targets are aspirational.

---

## Implementation Timeline

- **Phase 1**: 1-2 weeks
- **Phase 2**: 2-3 weeks
- **Phase 3**: 2-3 weeks
- **Phase 4**: 4-6 weeks

**Total**: 9-14 weeks (3-4 months) for full development

**With iterations**: 4-6 months realistic timeline

---

## Key Success Principles

1. **Don't skip phases** - Each builds on the last
2. **Validate rigorously** - Pass/fail, not "pretty close"
3. **Use temporal isolation** - TemporalController every time
4. **Test multiple assets** - Luck on one asset ≠ robust system
5. **Include transaction costs** - Real money has friction
6. **Survive the worst** - Drawdown limits are non-negotiable
7. **Focus on win rate** - It's the path to high Sharpe

---

## Recommended Next Steps

1. **Implement Phase 1** immediately
   - Use `create_trading_pipeline()` factory
   - Backtest on 5 assets
   - Run TemporalController backtests

2. **Create example scripts** in `examples/strategy_testing/`
   - `01_univariate_trend_following.py`
   - `02_multivariate_enhanced.py`
   - `03_multi_timeframe_filtering.py`
   - `04_advanced_integrated.py`

3. **Document performance** clearly
   - Sharpe, max DD, win rate for each phase
   - Walk-forward results
   - Out-of-sample degradation

4. **Iterate carefully** - If a phase fails, diagnose before moving forward

---

## Final Thoughts

**Realistic expectations**:
- Phase 1 (Sharpe 0.5-1.0): Straightforward, 80% pass rate
- Phase 2 (Sharpe 1.0-2.0): Moderate difficulty, 70% pass rate
- Phase 3 (Sharpe 2.0-4.0): Hard, 60% pass rate
- Phase 4 (Sharpe 7.0-10.0+): Very hard, 30% pass rate

**If you achieve Sharpe 2-3 in live trading, you have a world-class strategy.**

Success is not the final Sharpe number. Success is a robust, validated system that works across market cycles and handles real-world friction (costs, slippage, commissions).

**Remember: Risk management first, always.**
