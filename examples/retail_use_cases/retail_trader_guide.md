# Hidden Regime: Complete Retail Trader Guide

*From Market Basics to Advanced Regime-Based Trading Strategies*

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding Market Regimes](#understanding-market-regimes)
3. [Choosing the Right Model](#choosing-the-right-model)
4. [Position Sizing & Risk Management](#position-sizing--risk-management)
5. [Asset Class Strategies](#asset-class-strategies)
6. [Common Pitfalls & How to Avoid Them](#common-pitfalls--how-to-avoid-them)
7. [Performance Analysis](#performance-analysis)
8. [Advanced Techniques](#advanced-techniques)

---

## Getting Started

### What is Regime-Based Trading?

Traditional technical analysis assumes markets behave consistently over time. **Regime-based trading** recognizes that markets operate in distinct behavioral phases (regimes) with different statistical properties:

- **Bear Regime**: Negative returns, high volatility, persistent downtrends
- **Sideways Regime**: Near-zero returns, low volatility, range-bound trading
- **Bull Regime**: Positive returns, moderate volatility, sustained uptrends
- **Crisis Regime**: Very negative returns, extreme volatility, panic selling

### Why Hidden Markov Models?

Hidden Markov Models (HMMs) are perfect for regime detection because:

1. **Regimes are "hidden"** - We observe returns but must infer the underlying market state
2. **Probabilistic inference** - Instead of hard classifications, we get probabilities
3. **Memory effects** - Current regime depends on previous regime (Markov property)
4. **Real-time adaptation** - Models update as new data arrives

### Quick Start Example

```python
import hidden_regime as hr

# Load your data
data = hr.load_stock_data('AAPL', '2023-01-01', '2023-12-31')

# Detect regimes
states, model = hr.detect_regimes(data['log_return'], return_model=True)

# Get current market state
current_state = model.predict(data['log_return'][-20:])[-1]
current_probs = model.predict_proba(data['log_return'][-20:])[-1]

print(f"Current regime: {current_state}")
print(f"Regime probabilities: {current_probs}")
```

---

## Understanding Market Regimes

### The Four Regime Framework

Based on our case studies, we recommend a **4-state regime framework** for most trading applications:

| Regime | Characteristics | Typical Duration | Trading Strategy |
|--------|-----------------|------------------|------------------|
| **Bear** | Daily return: -1.5%, Vol: 2.5% | 5-15 days | Defensive/Short |
| **Sideways** | Daily return: 0.1%, Vol: 1.5% | 10-30 days | Range trading |
| **Bull** | Daily return: 1.2%, Vol: 2.0% | 8-25 days | Long positions |
| **Crisis** | Daily return: -3.5%, Vol: 4.0% | 3-10 days | Cash/Hedged |

### Regime Identification in Practice

#### AAPL COVID Case Study Results:
- **Bear**: μ=-0.0080, σ=0.0090 (14.4% annual volatility)
- **Sideways**: μ=-0.0044, σ=0.0406 (64.3% annual volatility)  
- **Bull**: μ=0.0057, σ=0.0110 (17.5% annual volatility)

#### DOGE Crypto Case Study Results:
- **Crisis**: μ=-0.0080, σ=0.1632 (311.9% annual volatility)
- **Bear**: μ=-0.0054, σ=0.0367 (70.2% annual volatility)
- **Sideways**: μ=0.0004, σ=0.0167 (31.9% annual volatility)
- **Bull**: μ=0.0555, σ=0.0623 (119.0% annual volatility)

### Key Observations:

1. **Volatility Clustering**: Regimes show distinct volatility levels
2. **Asset Dependence**: Same regime types but different parameters across assets
3. **Time Persistence**: Regimes tend to persist for multiple days
4. **Transition Patterns**: Some transitions more likely than others

---

## Choosing the Right Model

### Model Selection by Asset Class

#### Traditional Equities (Stocks, ETFs)
```python
config = HMMConfig.for_standardized_regimes(
    regime_type='3_state',      # Bear, Sideways, Bull
    conservative=True           # Stable parameter estimation
)

online_config = OnlineHMMConfig(
    forgetting_factor=0.98,     # Slow adaptation
    adaptation_rate=0.05,       # Conservative learning
    min_observations_for_update=50
)
```

#### Cryptocurrency
```python
config = HMMConfig.for_standardized_regimes(
    regime_type='4_state',      # Include Crisis/Euphoria
    conservative=False          # Allow aggressive fitting
)

online_config = OnlineHMMConfig(
    forgetting_factor=0.95,     # Faster forgetting
    adaptation_rate=0.08,       # Higher learning rate
    min_observations_for_update=20
)
```

#### Commodities & FX
```python
config = HMMConfig.for_standardized_regimes(
    regime_type='3_state',      # Standard framework
    conservative=True           # Stable for macro factors
)

online_config = OnlineHMMConfig(
    forgetting_factor=0.97,     # Medium adaptation
    adaptation_rate=0.06,       # Moderate learning
    min_observations_for_update=30
)
```

### Training Data Requirements

| Asset Class | Minimum Training Days | Recommended Training | Retraining Frequency |
|-------------|----------------------|---------------------|---------------------|
| **Equities** | 200 days (1 year) | 500 days (2 years) | Monthly |
| **Crypto** | 100 days (3 months) | 300 days (1 year) | Weekly |
| **FX** | 300 days (1.5 years) | 750 days (3 years) | Bi-weekly |
| **Commodities** | 250 days (1 year) | 600 days (2.5 years) | Monthly |

---

## Position Sizing & Risk Management

### Confidence-Based Position Sizing

The core principle: **Position size should reflect both regime type and model confidence**.

```python
def calculate_position_size(regime_probs, regime_names, base_capital=100000):
    """
    Calculate position size based on regime probabilities
    
    Args:
        regime_probs: Array of regime probabilities
        regime_names: List of regime names 
        base_capital: Available capital
        
    Returns:
        Recommended position size (-1 to 1)
    """
    # Base position sizes by regime
    regime_positions = {
        'bear': -0.3,      # Short position
        'sideways': 0.2,   # Small long
        'bull': 0.8,       # Large long  
        'crisis': 0.0      # Cash
    }
    
    # Calculate expected position
    expected_position = 0
    for prob, regime in zip(regime_probs, regime_names):
        base_pos = regime_positions.get(regime.lower(), 0)
        expected_position += prob * base_pos
    
    # Confidence adjustment
    max_confidence = max(regime_probs)
    confidence_multiplier = 0.5 + 0.5 * max_confidence
    
    return expected_position * confidence_multiplier
```

### Risk Management Framework

#### 1. Maximum Position Limits
```python
# Conservative retail trader
max_position_equity = 0.6    # 60% max long position
max_position_crypto = 0.8    # 80% max for crypto risk tolerance
max_short_position = -0.3    # 30% max short

# Aggressive trader  
max_position_equity = 0.9    # 90% max long
max_position_crypto = 0.95   # 95% max for crypto
max_short_position = -0.5    # 50% max short
```

#### 2. Drawdown Protection
```python
# Stop trading if portfolio drops below thresholds
drawdown_limits = {
    'conservative': 0.15,    # 15% max drawdown
    'moderate': 0.25,        # 25% max drawdown  
    'aggressive': 0.40       # 40% max drawdown
}

def check_drawdown_limit(current_value, peak_value, risk_tolerance='moderate'):
    drawdown = (peak_value - current_value) / peak_value
    limit = drawdown_limits[risk_tolerance]
    
    if drawdown > limit:
        return "HALT_TRADING"  # Go to cash
    elif drawdown > limit * 0.8:
        return "REDUCE_RISK"   # Cut positions by 50%
    else:
        return "CONTINUE"      # Normal trading
```

#### 3. Volatility Scaling
```python
def volatility_adjusted_position(base_position, current_volatility, target_volatility=0.02):
    """
    Scale position size inversely with volatility
    
    Higher volatility = Smaller position
    Lower volatility = Larger position
    """
    vol_ratio = target_volatility / current_volatility
    vol_adjusted = base_position * min(vol_ratio, 2.0)  # Cap at 2x
    
    return np.clip(vol_adjusted, -0.9, 0.9)
```

---

## Asset Class Strategies

### Equity Trading Strategy

**Best for**: Blue-chip stocks, sector ETFs, broad market indexes

**Model Configuration**:
- 3-state regime model (Bear, Sideways, Bull)
- Conservative parameter estimation
- 250-day training window

**Position Sizing**:
```python
equity_positions = {
    'bear': -0.2,       # Small short or cash
    'sideways': 0.3,    # Conservative long
    'bull': 0.7         # Strong long position
}
```

**Risk Management**:
- Maximum 60% long position
- Maximum 20% short position  
- 20% maximum drawdown limit
- Monthly model retraining

**Expected Performance**: 
- Lower volatility than buy-and-hold
- Better downside protection during bear markets
- Modest outperformance during bull markets

### Cryptocurrency Trading Strategy

**Best for**: Major cryptocurrencies (BTC, ETH, DOGE)

**Model Configuration**:
- 4-state regime model (Crisis, Bear, Sideways, Bull/Euphoria)
- Aggressive parameter estimation
- 100-day training window

**Position Sizing**:
```python
crypto_positions = {
    'crisis': 0.1,      # Small position (crypto rarely goes to zero)
    'bear': 0.2,        # Small long
    'sideways': 0.5,    # Moderate position
    'bull': 0.8         # Large position (bubble risk management)
}
```

**Risk Management**:
- Maximum 90% long position
- No short positions (crypto volatility)
- 40% maximum drawdown limit
- Weekly model retraining
- Volatility-based position scaling

**Expected Performance**:
- Significant downside protection during crypto crashes
- Reduced exposure during euphoric bubbles
- Lower returns than HODL during bull runs
- Better risk-adjusted returns

### Forex Trading Strategy

**Best for**: Major currency pairs (EUR/USD, GBP/USD, USD/JPY)

**Model Configuration**:
- 3-state regime model
- Conservative parameter estimation  
- 500-day training window

**Position Sizing**:
```python
fx_positions = {
    'bear': -0.4,       # Moderate short
    'sideways': 0.0,    # Neutral (range trading)
    'bull': 0.4         # Moderate long
}
```

**Risk Management**:
- Maximum 50% position either direction
- 25% maximum drawdown limit
- Bi-weekly model retraining
- Focus on major economic event timing

### Commodities Trading Strategy

**Best for**: Gold, oil, agricultural commodities

**Model Configuration**:
- 3-state regime model
- Conservative parameter estimation
- 400-day training window

**Position Sizing**:
```python
commodity_positions = {
    'bear': -0.3,       # Short during downtrends
    'sideways': 0.1,    # Small long (inflation hedge)
    'bull': 0.6         # Strong long during bull runs
}
```

**Risk Management**:
- Maximum 60% long position
- Maximum 30% short position
- 30% maximum drawdown limit
- Monthly model retraining
- Macro factor integration

---

## Common Pitfalls & How to Avoid Them

### 1. Over-Optimization on Historical Data

**Problem**: Model performs great on training data but fails on new data

**Solution**:
```python
# Use walk-forward validation
def walk_forward_validation(returns, window_size=252, test_size=30):
    results = []
    
    for i in range(window_size, len(returns) - test_size, test_size):
        # Training window
        train_data = returns[i-window_size:i]
        
        # Test window  
        test_data = returns[i:i+test_size]
        
        # Train model
        model = HiddenMarkovModel()
        model.fit(train_data)
        
        # Test performance
        test_states = model.predict(test_data)
        results.append(evaluate_performance(test_data, test_states))
    
    return results
```

### 2. Ignoring Transaction Costs

**Problem**: Strategy looks profitable but costs eat away returns

**Solution**:
```python
def include_transaction_costs(returns, positions, cost_per_trade=0.001):
    """
    Adjust returns for transaction costs
    
    Args:
        returns: Asset returns
        positions: Position changes
        cost_per_trade: Cost per trade (0.1% = 0.001)
    """
    position_changes = np.abs(np.diff(positions, prepend=positions[0]))
    trading_costs = position_changes * cost_per_trade
    
    # Subtract costs from returns
    net_returns = returns - trading_costs
    
    return net_returns
```

### 3. Regime Instability

**Problem**: Model changes regimes too frequently (regime switching)

**Solution**:
```python
# Add regime persistence penalty
config = HMMConfig.for_standardized_regimes(
    regime_type='3_state',
    conservative=True,
    min_regime_duration=3,      # Minimum 3 days per regime
    transition_penalty=0.1      # Penalty for frequent switching
)

# Use regime confidence threshold
def stable_regime_prediction(regime_probs, min_confidence=0.6):
    max_prob = np.max(regime_probs)
    
    if max_prob < min_confidence:
        return "uncertain"  # Stay in cash during uncertainty
    else:
        return np.argmax(regime_probs)
```

### 4. Neglecting Market Microstructure

**Problem**: Strategies work on daily data but fail intraday

**Considerations**:
- **Bid-ask spreads**: Wider spreads = higher costs
- **Market impact**: Large orders move prices against you
- **Liquidity**: Some assets harder to trade during stress
- **After-hours trading**: Different regime behavior

### 5. Survivorship Bias

**Problem**: Only analyzing assets that survived, missing failures

**Solution**:
```python
# Include delisted/failed assets in backtests
# Test on multiple asset universes
# Use broad market ETFs to avoid single-stock risk
```

---

## Performance Analysis

### Key Metrics for Regime-Based Strategies

#### 1. Regime-Adjusted Sharpe Ratio
```python
def regime_adjusted_sharpe(returns, regimes, regime_names):
    """
    Calculate Sharpe ratio adjusted for regime performance
    """
    sharpe_by_regime = {}
    
    for regime_name in regime_names:
        regime_mask = regimes == regime_name
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) > 0:
            sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252)
            sharpe_by_regime[regime_name] = sharpe
    
    # Weighted average by regime frequency
    weights = [np.mean(regimes == name) for name in regime_names]
    weighted_sharpe = np.average(list(sharpe_by_regime.values()), weights=weights)
    
    return weighted_sharpe, sharpe_by_regime
```

#### 2. Regime Transition Analysis
```python
def analyze_regime_transitions(regimes):
    """
    Analyze regime switching patterns
    """
    transitions = {}
    total_transitions = 0
    
    for i in range(1, len(regimes)):
        current = regimes[i]
        previous = regimes[i-1]
        
        if current != previous:
            total_transitions += 1
            transition = f"{previous}_to_{current}"
            transitions[transition] = transitions.get(transition, 0) + 1
    
    # Calculate transition probabilities
    transition_probs = {k: v/total_transitions for k, v in transitions.items()}
    
    return {
        'total_transitions': total_transitions,
        'avg_regime_duration': len(regimes) / (total_transitions + 1),
        'transition_matrix': transition_probs
    }
```

#### 3. Drawdown by Regime
```python
def regime_specific_drawdowns(returns, regimes, regime_names):
    """
    Calculate maximum drawdown within each regime
    """
    regime_drawdowns = {}
    
    for regime_name in regime_names:
        regime_mask = regimes == regime_name  
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) > 0:
            cumulative = (1 + regime_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            
            regime_drawdowns[regime_name] = {
                'max_drawdown': drawdown.min(),
                'avg_drawdown': drawdown.mean(),
                'drawdown_days': len(drawdown[drawdown < -0.05])  # Days with >5% drawdown
            }
    
    return regime_drawdowns
```

### Benchmark Comparisons

#### Against Buy-and-Hold
```python
def compare_to_buy_hold(strategy_returns, asset_returns):
    """
    Compare regime strategy to simple buy-and-hold
    """
    strategy_total = (1 + strategy_returns).prod() - 1
    buy_hold_total = (1 + asset_returns).prod() - 1
    
    strategy_vol = strategy_returns.std() * np.sqrt(252)
    buy_hold_vol = asset_returns.std() * np.sqrt(252)
    
    strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    buy_hold_sharpe = asset_returns.mean() / asset_returns.std() * np.sqrt(252)
    
    return {
        'excess_return': strategy_total - buy_hold_total,
        'volatility_reduction': buy_hold_vol - strategy_vol,
        'sharpe_improvement': strategy_sharpe - buy_hold_sharpe,
        'win_rate': np.mean(strategy_returns > asset_returns)
    }
```

#### Risk-Adjusted Performance
```python
def risk_adjusted_metrics(returns):
    """
    Calculate comprehensive risk-adjusted performance metrics
    """
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    sortino = annual_return / (downside_returns.std() * np.sqrt(252))
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar = annual_return / abs(max_drawdown)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar
    }
```

---

## Advanced Techniques

### 1. Multi-Asset Regime Models

For portfolio-level regime detection:

```python
def multi_asset_regime_model(returns_matrix, asset_names):
    """
    Fit regime model to multiple assets simultaneously
    
    Args:
        returns_matrix: N x M matrix (N observations, M assets)
        asset_names: List of asset names
    """
    from sklearn.mixture import GaussianMixture
    
    # Fit mixture model to multi-dimensional returns
    n_regimes = 3
    gmm = GaussianMixture(n_components=n_regimes, random_state=42)
    regimes = gmm.fit_predict(returns_matrix)
    
    # Analyze regime characteristics
    regime_stats = {}
    for regime in range(n_regimes):
        regime_mask = regimes == regime
        regime_returns = returns_matrix[regime_mask]
        
        regime_stats[f'Regime_{regime}'] = {
            'mean_returns': regime_returns.mean(axis=0),
            'covariance': np.cov(regime_returns.T),
            'frequency': regime_mask.mean()
        }
    
    return regimes, regime_stats
```

### 2. Regime-Aware Portfolio Optimization

```python
def regime_aware_portfolio(returns_matrix, regimes, regime_names, target_return=0.10):
    """
    Optimize portfolio weights considering regime-specific covariances
    """
    from scipy.optimize import minimize
    
    n_assets = returns_matrix.shape[1]
    regime_weights = []
    
    for regime in np.unique(regimes):
        regime_mask = regimes == regime
        regime_returns = returns_matrix[regime_mask]
        
        if len(regime_returns) > n_assets:  # Sufficient data
            # Mean returns and covariance for this regime
            mu = regime_returns.mean(axis=0)
            cov = np.cov(regime_returns.T)
            
            # Optimize portfolio for this regime
            def objective(weights):
                portfolio_var = weights.T @ cov @ weights
                return portfolio_var
            
            def constraint(weights):
                return weights.T @ mu - target_return
            
            constraints = {'type': 'eq', 'fun': constraint}
            bounds = [(0, 1) for _ in range(n_assets)]
            
            result = minimize(objective, np.ones(n_assets)/n_assets, 
                            bounds=bounds, constraints=constraints)
            
            regime_weights.append(result.x)
        else:
            regime_weights.append(np.ones(n_assets)/n_assets)  # Equal weight fallback
    
    return regime_weights
```

### 3. Dynamic Regime Forecasting

```python
def forecast_regime_transitions(model, current_regime, horizon=5):
    """
    Forecast likely regime transitions over specified horizon
    """
    transition_matrix = model.transition_matrix_
    current_probs = np.zeros(model.n_states)
    current_probs[current_regime] = 1.0
    
    forecasts = [current_probs]
    
    for step in range(horizon):
        next_probs = forecasts[-1] @ transition_matrix
        forecasts.append(next_probs)
    
    return np.array(forecasts)
```

### 4. Regime Confidence Indicators

```python
def calculate_regime_confidence_indicators(model, recent_returns, window=20):
    """
    Calculate multiple confidence indicators for current regime
    """
    # Get recent state probabilities
    recent_probs = model.predict_proba(recent_returns[-window:])
    
    # 1. Maximum probability confidence
    max_prob_confidence = np.mean([np.max(probs) for probs in recent_probs])
    
    # 2. Entropy-based confidence (lower entropy = higher confidence)
    entropies = [-np.sum(probs * np.log(probs + 1e-10)) for probs in recent_probs]
    entropy_confidence = 1 - np.mean(entropies) / np.log(model.n_states)
    
    # 3. Regime stability (how often regime changes)
    predicted_states = model.predict(recent_returns[-window:])
    regime_changes = np.sum(predicted_states[1:] != predicted_states[:-1])
    stability_confidence = 1 - regime_changes / len(predicted_states)
    
    # 4. Model likelihood confidence
    recent_likelihood = model.score(recent_returns[-window:])
    baseline_likelihood = getattr(model, '_baseline_likelihood', recent_likelihood)
    likelihood_confidence = min(recent_likelihood / baseline_likelihood, 2.0) - 1
    
    return {
        'max_probability': max_prob_confidence,
        'entropy_confidence': entropy_confidence, 
        'stability_confidence': stability_confidence,
        'likelihood_confidence': likelihood_confidence,
        'composite_confidence': np.mean([
            max_prob_confidence, entropy_confidence, 
            stability_confidence, max(likelihood_confidence, 0)
        ])
    }
```

---

## Conclusion

This guide provides a comprehensive framework for implementing regime-based trading strategies across different asset classes. Key takeaways:

1. **Start Simple**: Begin with 3-state models on equities before advancing to crypto or multi-asset strategies

2. **Focus on Risk Management**: Regime-based trading excels at downside protection, not maximum returns

3. **Adapt to Asset Classes**: Different assets require different model configurations and risk parameters

4. **Monitor Model Health**: Use confidence indicators and walk-forward validation to ensure model reliability

5. **Combine with Fundamentals**: Regime models work best when combined with macro-economic awareness

6. **Practice Before Live Trading**: Thoroughly backtest and paper trade before risking real capital

The case studies (AAPL COVID and DOGE explosive growth) demonstrate the framework's versatility across different market conditions. Use this guide as your foundation for developing sophisticated, regime-aware trading systems.

---

**Next Steps**: 
- Start with the basic equity strategy on a stock you know well
- Gradually add complexity as you gain experience  
- Consider the advanced Bayesian techniques in Phase 3 for institutional-level applications

*Remember: Past performance does not guarantee future results. Always practice proper risk management and never risk more than you can afford to lose.*