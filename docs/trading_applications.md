# Trading Applications Guide

*Comprehensive guide to using HMMs for quantitative trading strategies*

---

## Table of Contents

1. [Overview](#overview)
2. [Regime-Based Trading Philosophy](#regime-based-trading-philosophy)
3. [Signal Generation](#signal-generation)
4. [Position Sizing](#position-sizing)
5. [Risk Management](#risk-management)
6. [Portfolio Management](#portfolio-management)
7. [Backtesting Framework](#backtesting-framework)
8. [Live Trading Integration](#live-trading-integration)
9. [Performance Metrics](#performance-metrics)
10. [Advanced Strategies](#advanced-strategies)
11. [Multi-Asset Applications](#multi-asset-applications)
12. [Implementation Examples](#implementation-examples)

---

## Overview

Hidden Markov Models provide a powerful framework for developing systematic trading strategies based on market regime detection. This guide covers practical applications from signal generation to portfolio management, with complete implementation examples.

### Core Advantages of HMM Trading

1. **Regime Awareness**: Strategies adapt to different market conditions
2. **Probabilistic Framework**: Uncertainty quantification for better risk management
3. **Real-Time Adaptation**: Online learning adjusts to changing market dynamics
4. **Multi-Asset Scalability**: Apply to individual assets or portfolios
5. **Risk-Adjusted Returns**: Improve Sharpe ratios through regime-based risk management

### Trading Strategy Categories

| Strategy Type | Description | Best Use Cases |
|---------------|-------------|----------------|
| **Trend Following** | Long bull regimes, short bear regimes | Strongly trending markets |
| **Mean Reversion** | Trade against regime transitions | Sideways/ranging markets |
| **Momentum** | Accelerate positions during regime persistence | Volatile trend markets |
| **Risk Parity** | Adjust risk based on regime volatility | Portfolio management |
| **Tactical Allocation** | Switch between asset classes by regime | Multi-asset portfolios |

---

## Regime-Based Trading Philosophy

### Market Regime Characteristics

Understanding regime properties is crucial for strategy design:

#### Bull Market Regime
- **Characteristics**: Positive mean returns, moderate volatility
- **Duration**: Typically 2-6 weeks for daily data
- **Trading Approach**: Maximize long exposure, momentum strategies
- **Risk Profile**: Moderate risk-taking appropriate

#### Bear Market Regime  
- **Characteristics**: Negative mean returns, high volatility
- **Duration**: Typically 1-3 weeks for daily data
- **Trading Approach**: Defensive positioning, short strategies
- **Risk Profile**: Reduce exposure, capital preservation

#### Sideways Market Regime
- **Characteristics**: Near-zero returns, low-to-moderate volatility
- **Duration**: Highly variable, often longest regime
- **Trading Approach**: Range trading, mean reversion
- **Risk Profile**: Neutral positioning, reduced size

### Regime Transition Dynamics

Understanding how markets transition between regimes:

```python
def analyze_regime_transitions(hmm_model, returns):
    """Analyze regime transition patterns"""
    states = hmm_model.predict(returns)
    transitions = []
    
    for t in range(1, len(states)):
        if states[t] != states[t-1]:
            transitions.append({
                'from_regime': states[t-1],
                'to_regime': states[t],
                'date': t,
                'return_during_transition': returns[t]
            })
    
    # Analyze transition patterns
    transition_matrix = hmm_model.transition_matrix_
    expected_durations = 1 / (1 - np.diag(transition_matrix))
    
    return {
        'transitions': transitions,
        'expected_durations': expected_durations,
        'transition_probabilities': transition_matrix,
        'regime_statistics': compute_regime_stats(states, returns)
    }
```

---

## Signal Generation

### Basic Regime Signals

#### 1. Direct Regime Signal
Use current regime as trading signal:

```python
class DirectRegimeStrategy:
    """Direct regime-based trading signals"""
    
    def __init__(self, hmm_model):
        self.hmm = hmm_model
        self.regime_positions = {0: -1.0, 1: 0.0, 2: 1.0}  # Bear, Sideways, Bull
        
    def generate_signal(self, returns):
        """Generate trading signal based on current regime"""
        regime_info = self.hmm.get_current_regime_info()
        
        base_position = self.regime_positions[regime_info['current_state']]
        confidence_adjustment = regime_info['confidence']
        
        # Adjust position by confidence
        signal_strength = base_position * confidence_adjustment
        
        return {
            'position': signal_strength,
            'regime': regime_info['current_state'],
            'confidence': confidence_adjustment,
            'rationale': f"Regime {regime_info['current_state']} with {confidence_adjustment:.2%} confidence"
        }
```

#### 2. Regime Transition Signal
Trade on regime changes:

```python
class RegimeTransitionStrategy:
    """Trade on regime transitions"""
    
    def __init__(self, hmm_model, lookback=5):
        self.hmm = hmm_model
        self.lookback = lookback
        self.previous_regime = None
        self.regime_history = deque(maxlen=lookback)
        
    def generate_signal(self, new_return):
        """Generate signal based on regime transitions"""
        # Update model and get regime
        self.hmm.add_observation(new_return)
        regime_info = self.hmm.get_current_regime_info()
        current_regime = regime_info['current_state']
        
        # Track regime history
        self.regime_history.append(current_regime)
        
        # Detect regime change
        if self.previous_regime is not None and current_regime != self.previous_regime:
            signal = self.regime_change_signal(
                from_regime=self.previous_regime,
                to_regime=current_regime,
                confidence=regime_info['confidence']
            )
        else:
            signal = {'position': 0.0, 'action': 'hold'}
            
        self.previous_regime = current_regime
        return signal
    
    def regime_change_signal(self, from_regime, to_regime, confidence):
        """Generate signal based on specific regime transition"""
        # Define transition signals
        transition_signals = {
            (0, 1): 0.3,   # Bear to Sideways: Small long
            (0, 2): 0.8,   # Bear to Bull: Strong long  
            (1, 0): -0.5,  # Sideways to Bear: Short
            (1, 2): 0.6,   # Sideways to Bull: Long
            (2, 0): -0.8,  # Bull to Bear: Strong short
            (2, 1): -0.3,  # Bull to Sideways: Reduce long
        }
        
        base_signal = transition_signals.get((from_regime, to_regime), 0.0)
        adjusted_signal = base_signal * confidence
        
        return {
            'position': adjusted_signal,
            'action': 'buy' if adjusted_signal > 0 else 'sell' if adjusted_signal < 0 else 'hold',
            'transition': f"{from_regime} -> {to_regime}",
            'confidence': confidence
        }
```

#### 3. Regime Momentum Signal
Trade on regime persistence:

```python
class RegimeMomentumStrategy:
    """Trade on regime persistence and strength"""
    
    def __init__(self, hmm_model, momentum_window=10):
        self.hmm = hmm_model
        self.momentum_window = momentum_window
        self.regime_probabilities_history = deque(maxlen=momentum_window)
        
    def generate_signal(self, new_return):
        """Generate signal based on regime momentum"""
        # Update model
        self.hmm.add_observation(new_return)
        regime_info = self.hmm.get_current_regime_info()
        
        # Track regime probability history
        self.regime_probabilities_history.append(regime_info['state_probabilities'])
        
        if len(self.regime_probabilities_history) < self.momentum_window:
            return {'position': 0.0, 'action': 'wait'}
        
        # Calculate regime momentum
        momentum = self.calculate_regime_momentum()
        position = self.momentum_to_position(momentum, regime_info)
        
        return {
            'position': position,
            'momentum': momentum,
            'regime': regime_info['current_state'],
            'confidence': regime_info['confidence']
        }
    
    def calculate_regime_momentum(self):
        """Calculate regime persistence momentum"""
        prob_array = np.array(list(self.regime_probabilities_history))
        
        # Calculate momentum as trend in regime probabilities
        momentum = {}
        for regime in range(prob_array.shape[1]):
            regime_probs = prob_array[:, regime]
            # Linear regression slope as momentum indicator
            x = np.arange(len(regime_probs))
            slope = np.polyfit(x, regime_probs, 1)[0]
            momentum[regime] = slope
            
        return momentum
    
    def momentum_to_position(self, momentum, regime_info):
        """Convert momentum to position size"""
        current_regime = regime_info['current_state']
        regime_momentum = momentum[current_regime]
        
        # Position based on regime type and momentum
        regime_multipliers = {0: -1.0, 1: 0.0, 2: 1.0}  # Bear, Sideways, Bull
        base_position = regime_multipliers[current_regime]
        
        # Momentum adjustment (stronger momentum = larger position)
        momentum_multiplier = 1.0 + np.tanh(regime_momentum * 100)  # Scale and bound
        
        return base_position * momentum_multiplier * regime_info['confidence']
```

### Advanced Signal Combinations

#### Multi-Signal Strategy
Combine multiple signal types:

```python
class MultiSignalHMMStrategy:
    """Combine multiple HMM-based signals"""
    
    def __init__(self, hmm_model):
        self.hmm = hmm_model
        
        # Individual strategy components
        self.direct_strategy = DirectRegimeStrategy(hmm_model)
        self.transition_strategy = RegimeTransitionStrategy(hmm_model)
        self.momentum_strategy = RegimeMomentumStrategy(hmm_model)
        
        # Signal weights
        self.signal_weights = {
            'direct': 0.4,
            'transition': 0.3,
            'momentum': 0.3
        }
    
    def generate_combined_signal(self, new_return):
        """Generate combined signal from multiple strategies"""
        # Get signals from each strategy
        signals = {
            'direct': self.direct_strategy.generate_signal([new_return]),
            'transition': self.transition_strategy.generate_signal(new_return),
            'momentum': self.momentum_strategy.generate_signal(new_return)
        }
        
        # Combine signals using weighted average
        combined_position = 0.0
        total_weight = 0.0
        
        for strategy_name, signal in signals.items():
            if 'position' in signal and signal['position'] != 0:
                weight = self.signal_weights[strategy_name]
                combined_position += signal['position'] * weight
                total_weight += weight
        
        # Normalize if any signals were active
        if total_weight > 0:
            combined_position /= total_weight
        
        return {
            'combined_position': combined_position,
            'individual_signals': signals,
            'signal_strength': abs(combined_position),
            'direction': 'long' if combined_position > 0 else 'short' if combined_position < 0 else 'neutral'
        }
```

---

## Position Sizing

### Risk-Adjusted Position Sizing

Position sizing based on regime characteristics and confidence:

```python
class HMMPositionSizer:
    """Position sizing based on HMM regime information"""
    
    def __init__(self, base_capital=100000, max_position_pct=1.0):
        self.base_capital = base_capital
        self.max_position_pct = max_position_pct
        self.regime_risk_multipliers = {
            0: 0.5,   # Bear market: Reduce exposure
            1: 0.7,   # Sideways: Moderate exposure  
            2: 1.0    # Bull market: Full exposure
        }
        
    def calculate_position_size(self, signal_strength, regime_info, current_volatility):
        """Calculate position size based on multiple factors"""
        
        # Base position from signal strength
        base_position = abs(signal_strength) * self.max_position_pct
        
        # Regime-based adjustment
        regime = regime_info['current_state']
        regime_multiplier = self.regime_risk_multipliers[regime]
        
        # Confidence adjustment
        confidence_multiplier = regime_info['confidence']
        
        # Volatility adjustment (inverse relationship)
        vol_target = 0.02  # 2% daily volatility target
        vol_adjustment = min(vol_target / max(current_volatility, 0.005), 2.0)
        
        # Combined position size
        final_position = (base_position * 
                         regime_multiplier * 
                         confidence_multiplier * 
                         vol_adjustment)
        
        # Apply maximum position limits
        final_position = min(final_position, self.max_position_pct)
        
        return {
            'position_size': final_position,
            'base_position': base_position,
            'regime_multiplier': regime_multiplier,
            'confidence_multiplier': confidence_multiplier,
            'vol_adjustment': vol_adjustment,
            'regime': regime,
            'rationale': self.get_sizing_rationale(regime, confidence_multiplier, vol_adjustment)
        }
    
    def get_sizing_rationale(self, regime, confidence_mult, vol_mult):
        """Provide human-readable rationale for position sizing"""
        regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        
        return (f"{regime_names[regime]} regime "
                f"(confidence: {confidence_mult:.2%}, "
                f"vol adjustment: {vol_mult:.2f})")
```

### Kelly Criterion with HMM

Optimal position sizing using Kelly criterion:

```python
class HMMKellyPositionSizer:
    """Kelly criterion position sizing for HMM strategies"""
    
    def __init__(self, lookback_window=252):
        self.lookback_window = lookback_window
        self.returns_history = deque(maxlen=lookback_window)
        
    def calculate_kelly_fraction(self, regime_info, historical_regime_performance):
        """Calculate Kelly fraction for each regime"""
        current_regime = regime_info['current_state']
        regime_stats = historical_regime_performance[current_regime]
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        
        win_prob = regime_stats['win_rate']
        avg_win = regime_stats['avg_winning_return']
        avg_loss = abs(regime_stats['avg_losing_return'])
        
        if avg_loss > 0 and win_prob > 0:
            # Kelly fraction calculation
            kelly_fraction = ((avg_win * win_prob) - (avg_loss * (1 - win_prob))) / avg_win
            
            # Apply safety factor (usually 0.25 to 0.5 of full Kelly)
            safe_kelly = max(0, min(kelly_fraction * 0.25, 0.5))
        else:
            safe_kelly = 0.0
            
        return {
            'kelly_fraction': kelly_fraction,
            'safe_kelly': safe_kelly,
            'win_prob': win_prob,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def analyze_regime_performance(self, returns, states):
        """Analyze historical performance by regime"""
        regime_performance = {}
        
        for regime in range(3):  # Assuming 3 states
            regime_returns = returns[states == regime]
            
            if len(regime_returns) > 0:
                winning_returns = regime_returns[regime_returns > 0]
                losing_returns = regime_returns[regime_returns < 0]
                
                regime_performance[regime] = {
                    'total_returns': len(regime_returns),
                    'win_rate': len(winning_returns) / len(regime_returns),
                    'avg_return': np.mean(regime_returns),
                    'avg_winning_return': np.mean(winning_returns) if len(winning_returns) > 0 else 0,
                    'avg_losing_return': np.mean(losing_returns) if len(losing_returns) > 0 else 0,
                    'volatility': np.std(regime_returns),
                    'sharpe_ratio': np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0
                }
            else:
                regime_performance[regime] = {
                    'total_returns': 0,
                    'win_rate': 0.5,
                    'avg_return': 0,
                    'avg_winning_return': 0,
                    'avg_losing_return': 0,
                    'volatility': 0.02,
                    'sharpe_ratio': 0
                }
                
        return regime_performance
```

---

## Risk Management

### Regime-Aware Risk Controls

Risk management that adapts to market regimes:

```python
class HMMRiskManager:
    """Comprehensive risk management using HMM regime information"""
    
    def __init__(self, max_portfolio_risk=0.02, regime_risk_budgets=None):
        self.max_portfolio_risk = max_portfolio_risk
        self.regime_risk_budgets = regime_risk_budgets or {
            0: 0.015,  # Bear: Lower risk budget
            1: 0.020,  # Sideways: Normal risk budget
            2: 0.025   # Bull: Higher risk budget
        }
        
        # Risk monitoring
        self.position_history = deque(maxlen=252)
        self.pnl_history = deque(maxlen=252)
        
    def check_risk_limits(self, proposed_position, regime_info, current_portfolio):
        """Check if proposed position meets risk limits"""
        current_regime = regime_info['current_state']
        regime_risk_budget = self.regime_risk_budgets[current_regime]
        
        # Calculate portfolio risk metrics
        risk_metrics = self.calculate_portfolio_risk(proposed_position, current_portfolio)
        
        # Risk limit checks
        risk_checks = {
            'portfolio_var_ok': risk_metrics['portfolio_var'] <= regime_risk_budget,
            'max_position_ok': abs(proposed_position) <= 1.0,
            'drawdown_ok': risk_metrics['current_drawdown'] <= 0.1,  # 10% max drawdown
            'regime_appropriate': self.is_regime_appropriate_position(proposed_position, regime_info)
        }
        
        # Overall risk approval
        all_checks_passed = all(risk_checks.values())
        
        if not all_checks_passed:
            # Calculate adjusted position that meets risk limits
            adjusted_position = self.adjust_position_for_risk(
                proposed_position, regime_info, risk_metrics
            )
        else:
            adjusted_position = proposed_position
            
        return {
            'approved_position': adjusted_position,
            'original_position': proposed_position,
            'risk_checks': risk_checks,
            'risk_metrics': risk_metrics,
            'risk_budget': regime_risk_budget,
            'position_adjusted': adjusted_position != proposed_position
        }
    
    def calculate_portfolio_risk(self, position, portfolio):
        """Calculate comprehensive portfolio risk metrics"""
        # Estimate portfolio volatility (simplified)
        if len(self.pnl_history) > 30:
            recent_returns = list(self.pnl_history)[-30:]
            portfolio_vol = np.std(recent_returns)
            current_drawdown = self.calculate_current_drawdown()
        else:
            portfolio_vol = 0.02  # Default assumption
            current_drawdown = 0.0
            
        # Value at Risk (95% confidence)
        portfolio_var = portfolio_vol * 1.645  # 95% VaR
        
        return {
            'portfolio_vol': portfolio_vol,
            'portfolio_var': portfolio_var,
            'current_drawdown': current_drawdown,
            'position_concentration': abs(position)
        }
    
    def is_regime_appropriate_position(self, position, regime_info):
        """Check if position direction matches regime"""
        regime = regime_info['current_state']
        confidence = regime_info['confidence']
        
        # Regime-position alignment rules
        if regime == 0:  # Bear
            return position <= 0.1 or confidence < 0.6  # Allow small long only with low confidence
        elif regime == 1:  # Sideways
            return abs(position) <= 0.5  # Moderate positions in sideways
        else:  # Bull
            return position >= -0.1 or confidence < 0.6  # Allow small short only with low confidence
    
    def adjust_position_for_risk(self, position, regime_info, risk_metrics):
        """Adjust position to meet risk constraints"""
        current_regime = regime_info['current_state']
        risk_budget = self.regime_risk_budgets[current_regime]
        
        # Scale position based on risk budget vs current risk
        if risk_metrics['portfolio_var'] > risk_budget:
            risk_scaling_factor = risk_budget / risk_metrics['portfolio_var']
            adjusted_position = position * risk_scaling_factor
        else:
            adjusted_position = position
            
        # Apply maximum position limits
        max_position_by_regime = {0: 0.5, 1: 0.7, 2: 1.0}
        max_allowed = max_position_by_regime[current_regime]
        
        adjusted_position = np.clip(adjusted_position, -max_allowed, max_allowed)
        
        return adjusted_position
    
    def calculate_current_drawdown(self):
        """Calculate current portfolio drawdown"""
        if len(self.pnl_history) < 2:
            return 0.0
            
        cumulative_pnl = np.cumsum(list(self.pnl_history))
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = (cumulative_pnl - running_max) / np.maximum(running_max, 1)  # Avoid division by zero
        
        return abs(drawdowns[-1]) if len(drawdowns) > 0 else 0.0
    
    def update_risk_monitoring(self, position, pnl):
        """Update risk monitoring with new position and P&L"""
        self.position_history.append(position)
        self.pnl_history.append(pnl)
```

### Dynamic Stop Loss System

Regime-aware stop loss management:

```python
class HMMStopLossManager:
    """Dynamic stop loss based on regime characteristics"""
    
    def __init__(self):
        self.regime_stop_multipliers = {
            0: 1.5,   # Bear: Tighter stops due to higher volatility
            1: 1.0,   # Sideways: Normal stops
            2: 1.2    # Bull: Slightly wider stops to avoid whipsaws
        }
        self.base_stop_loss = 0.02  # 2% base stop loss
        
    def calculate_stop_loss(self, entry_price, position_direction, regime_info, current_volatility):
        """Calculate dynamic stop loss based on regime and volatility"""
        current_regime = regime_info['current_state']
        confidence = regime_info['confidence']
        
        # Base stop loss adjusted for regime
        regime_multiplier = self.regime_stop_multipliers[current_regime]
        
        # Volatility adjustment
        vol_multiplier = max(0.5, min(current_volatility / 0.02, 3.0))  # Scale vs 2% base vol
        
        # Confidence adjustment (lower confidence = tighter stops)
        confidence_multiplier = 0.5 + 0.5 * confidence
        
        # Combined stop loss
        stop_distance = (self.base_stop_loss * 
                        regime_multiplier * 
                        vol_multiplier * 
                        confidence_multiplier)
        
        # Calculate stop price
        if position_direction > 0:  # Long position
            stop_price = entry_price * (1 - stop_distance)
        else:  # Short position
            stop_price = entry_price * (1 + stop_distance)
            
        return {
            'stop_price': stop_price,
            'stop_distance_pct': stop_distance,
            'regime_multiplier': regime_multiplier,
            'vol_multiplier': vol_multiplier,
            'confidence_multiplier': confidence_multiplier,
            'rationale': f"Regime {current_regime} stop with {confidence:.1%} confidence"
        }
    
    def update_trailing_stop(self, current_price, position, regime_info, trailing_stops):
        """Update trailing stop based on current market conditions"""
        if position == 0:
            return trailing_stops
            
        current_regime = regime_info['current_state']
        
        # Regime-based trailing parameters
        trailing_multipliers = {
            0: 0.8,   # Bear: Less aggressive trailing
            1: 1.0,   # Sideways: Normal trailing
            2: 1.2    # Bull: More aggressive trailing
        }
        
        trailing_mult = trailing_multipliers[current_regime]
        base_trailing_distance = 0.03 * trailing_mult  # 3% base trailing
        
        if position > 0:  # Long position
            # Update trailing stop if price moved favorably
            new_stop = current_price * (1 - base_trailing_distance)
            if 'long_stop' not in trailing_stops or new_stop > trailing_stops['long_stop']:
                trailing_stops['long_stop'] = new_stop
        else:  # Short position
            # Update trailing stop if price moved favorably  
            new_stop = current_price * (1 + base_trailing_distance)
            if 'short_stop' not in trailing_stops or new_stop < trailing_stops['short_stop']:
                trailing_stops['short_stop'] = new_stop
                
        return trailing_stops
```

---

## Portfolio Management

### Multi-Asset Regime-Based Allocation

Portfolio allocation using regime correlations:

```python
class HMMPortfolioManager:
    """Portfolio management using multi-asset HMM regime detection"""
    
    def __init__(self, assets, regime_correlation_window=100):
        self.assets = assets
        self.hmm_models = {asset: OnlineHMM(n_states=3) for asset in assets}
        self.regime_correlation_window = regime_correlation_window
        self.regime_history = {asset: deque(maxlen=regime_correlation_window) for asset in assets}
        
        # Asset-specific regime allocations
        self.regime_allocations = {
            asset: {0: -0.2, 1: 0.1, 2: 0.8} for asset in assets  # Bear, Sideways, Bull
        }
        
    def update_all_models(self, returns_dict):
        """Update all asset HMM models with new returns"""
        regime_info = {}
        
        for asset, return_val in returns_dict.items():
            if asset in self.hmm_models:
                # Update model
                self.hmm_models[asset].add_observation(return_val)
                
                # Get regime information
                info = self.hmm_models[asset].get_current_regime_info()
                regime_info[asset] = info
                
                # Track regime history
                self.regime_history[asset].append(info['current_state'])
        
        return regime_info
    
    def calculate_portfolio_allocation(self, regime_info, risk_budget=1.0):
        """Calculate optimal portfolio allocation based on current regimes"""
        
        # Base allocation from individual regimes
        base_allocations = {}
        total_confidence = 0
        
        for asset, info in regime_info.items():
            regime = info['current_state']
            confidence = info['confidence']
            
            base_allocation = self.regime_allocations[asset][regime]
            confidence_adjusted = base_allocation * confidence
            
            base_allocations[asset] = confidence_adjusted
            total_confidence += confidence
            
        # Regime correlation adjustment
        correlation_adjustment = self.calculate_regime_correlation_adjustment()
        
        # Apply correlation adjustment
        final_allocations = {}
        total_allocation = 0
        
        for asset in self.assets:
            if asset in base_allocations:
                correlation_mult = correlation_adjustment.get(asset, 1.0)
                adjusted_allocation = base_allocations[asset] * correlation_mult
                final_allocations[asset] = adjusted_allocation
                total_allocation += abs(adjusted_allocation)
        
        # Scale to risk budget
        if total_allocation > 0:
            scale_factor = risk_budget / max(total_allocation, risk_budget)
            final_allocations = {asset: alloc * scale_factor 
                               for asset, alloc in final_allocations.items()}
        
        return {
            'allocations': final_allocations,
            'base_allocations': base_allocations,
            'correlation_adjustments': correlation_adjustment,
            'total_gross_exposure': sum(abs(alloc) for alloc in final_allocations.values()),
            'regime_summary': {asset: info['regime_interpretation'] 
                             for asset, info in regime_info.items()}
        }
    
    def calculate_regime_correlation_adjustment(self):
        """Calculate correlation-based allocation adjustments"""
        if len(list(self.regime_history.values())[0]) < 30:  # Need enough history
            return {asset: 1.0 for asset in self.assets}
        
        # Create regime correlation matrix
        regime_sequences = {asset: list(history) for asset, history in self.regime_history.items()}
        
        correlations = {}
        for asset1 in self.assets:
            for asset2 in self.assets:
                if asset1 != asset2:
                    seq1 = regime_sequences[asset1]
                    seq2 = regime_sequences[asset2]
                    
                    # Calculate regime sequence correlation
                    correlation = np.corrcoef(seq1, seq2)[0, 1]
                    correlations[(asset1, asset2)] = correlation
        
        # Convert correlations to allocation adjustments
        adjustments = {}
        for asset in self.assets:
            # Average correlation with other assets
            asset_correlations = [correlations.get((asset, other), 0) 
                                for other in self.assets if other != asset]
            
            avg_correlation = np.mean(asset_correlations) if asset_correlations else 0
            
            # Reduce allocation for highly correlated assets
            correlation_penalty = max(0.5, 1.0 - abs(avg_correlation) * 0.5)
            adjustments[asset] = correlation_penalty
            
        return adjustments
    
    def rebalance_portfolio(self, current_positions, target_allocations, transaction_cost=0.001):
        """Calculate optimal rebalancing trades considering transaction costs"""
        
        trades = {}
        total_transaction_cost = 0
        
        for asset in self.assets:
            current_pos = current_positions.get(asset, 0)
            target_pos = target_allocations.get(asset, 0)
            
            position_diff = target_pos - current_pos
            
            # Only trade if difference is significant (considering transaction costs)
            min_trade_size = 0.05  # 5% minimum trade
            
            if abs(position_diff) > min_trade_size:
                trade_cost = abs(position_diff) * transaction_cost
                
                # Only trade if expected benefit exceeds transaction cost
                expected_benefit = self.estimate_rebalancing_benefit(asset, position_diff)
                
                if expected_benefit > trade_cost * 2:  # 2x cost hurdle
                    trades[asset] = {
                        'current_position': current_pos,
                        'target_position': target_pos,
                        'trade_size': position_diff,
                        'transaction_cost': trade_cost,
                        'expected_benefit': expected_benefit
                    }
                    total_transaction_cost += trade_cost
        
        return {
            'trades': trades,
            'total_transaction_cost': total_transaction_cost,
            'net_expected_benefit': sum(trade['expected_benefit'] for trade in trades.values()) - total_transaction_cost
        }
    
    def estimate_rebalancing_benefit(self, asset, position_change):
        """Estimate expected benefit from rebalancing trade"""
        # Simplified benefit estimation based on regime momentum
        if asset in self.hmm_models:
            regime_info = self.hmm_models[asset].get_current_regime_info()
            confidence = regime_info['confidence']
            
            # Higher confidence regimes provide better rebalancing opportunities
            expected_return = 0.001 * confidence * np.sign(position_change)  # Simplified
            return abs(position_change) * expected_return
        
        return 0.0
```

---

## Backtesting Framework

### Comprehensive HMM Backtesting

Rigorous backtesting framework for HMM strategies:

```python
class HMMBacktester:
    """Comprehensive backtesting framework for HMM trading strategies"""
    
    def __init__(self, initial_capital=100000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Performance tracking
        self.portfolio_history = []
        self.trade_history = []
        self.regime_history = []
        self.metrics_history = []
        
    def backtest_strategy(self, strategy, returns_data, prices_data, regime_model):
        """Run comprehensive backtest of HMM strategy"""
        
        # Initialize
        capital = self.initial_capital
        position = 0.0
        portfolio_values = [capital]
        
        # Walk through data
        for i in range(1, len(returns_data)):
            current_return = returns_data[i]
            current_price = prices_data[i]
            
            # Update regime model (for online HMM)
            if hasattr(regime_model, 'add_observation'):
                regime_model.add_observation(current_return)
            
            # Generate trading signal
            signal_info = strategy.generate_signal(current_return, regime_model, i)
            
            # Calculate P&L from existing position
            if position != 0:
                pnl = position * capital * current_return
                capital += pnl
            else:
                pnl = 0
            
            # Execute new position if signal changed
            target_position = signal_info.get('position', 0)
            
            if abs(target_position - position) > 0.01:  # 1% threshold
                # Calculate transaction cost
                trade_size = abs(target_position - position)
                trans_cost = trade_size * capital * self.transaction_cost
                capital -= trans_cost
                
                # Record trade
                self.trade_history.append({
                    'date_index': i,
                    'old_position': position,
                    'new_position': target_position,
                    'trade_size': target_position - position,
                    'price': current_price,
                    'transaction_cost': trans_cost,
                    'reason': signal_info.get('rationale', 'Signal change')
                })
                
                position = target_position
            
            # Record daily metrics
            portfolio_value = capital
            portfolio_values.append(portfolio_value)
            
            self.portfolio_history.append({
                'date_index': i,
                'portfolio_value': portfolio_value,
                'position': position,
                'capital': capital,
                'pnl': pnl,
                'return': current_return
            })
            
            # Record regime information
            if hasattr(regime_model, 'get_current_regime_info'):
                regime_info = regime_model.get_current_regime_info()
                self.regime_history.append({
                    'date_index': i,
                    'regime': regime_info['current_state'],
                    'regime_interpretation': regime_info['regime_interpretation'],
                    'confidence': regime_info['confidence'],
                    'state_probabilities': regime_info['state_probabilities'].copy()
                })
        
        # Calculate comprehensive performance metrics
        performance_metrics = self.calculate_performance_metrics(
            portfolio_values, returns_data
        )
        
        return {
            'performance_metrics': performance_metrics,
            'portfolio_history': self.portfolio_history,
            'trade_history': self.trade_history,
            'regime_history': self.regime_history
        }
    
    def calculate_performance_metrics(self, portfolio_values, benchmark_returns):
        """Calculate comprehensive performance metrics"""
        
        # Portfolio returns
        portfolio_values = np.array(portfolio_values)
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Benchmark comparison (buy and hold)
        benchmark_total_return = np.prod(1 + benchmark_returns) - 1
        benchmark_annualized = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
        benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252)
        
        alpha = annualized_return - benchmark_annualized
        
        # Beta calculation
        if len(portfolio_returns) == len(benchmark_returns):
            beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            information_ratio = alpha / np.std(portfolio_returns - benchmark_returns) * np.sqrt(252)
        else:
            beta = 0
            information_ratio = 0
        
        # Trade analysis
        num_trades = len(self.trade_history)
        avg_trade_cost = np.mean([trade['transaction_cost'] for trade in self.trade_history]) if num_trades > 0 else 0
        total_transaction_costs = sum([trade['transaction_cost'] for trade in self.trade_history])
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'num_trades': num_trades,
            'avg_trade_cost': avg_trade_cost,
            'total_transaction_costs': total_transaction_costs,
            'benchmark_return': benchmark_annualized,
            'excess_return': annualized_return - benchmark_annualized
        }
    
    def analyze_regime_performance(self):
        """Analyze performance by market regime"""
        if not self.portfolio_history or not self.regime_history:
            return {}
        
        # Create regime performance analysis
        regime_performance = {0: [], 1: [], 2: []}  # Bear, Sideways, Bull
        
        for i, (portfolio, regime) in enumerate(zip(self.portfolio_history[1:], self.regime_history[1:])):
            if i > 0:
                daily_return = (portfolio['portfolio_value'] - self.portfolio_history[i]['portfolio_value']) / self.portfolio_history[i]['portfolio_value']
                regime_performance[regime['regime']].append(daily_return)
        
        # Calculate metrics by regime
        regime_metrics = {}
        regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        
        for regime, returns in regime_performance.items():
            if len(returns) > 0:
                regime_metrics[regime_names[regime]] = {
                    'count': len(returns),
                    'mean_return': np.mean(returns),
                    'volatility': np.std(returns),
                    'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
                    'win_rate': len([r for r in returns if r > 0]) / len(returns),
                    'avg_win': np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0,
                    'avg_loss': np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
                }
        
        return regime_metrics

# Example usage
def run_hmm_backtest_example():
    """Example of running HMM backtest"""
    
    # Load data (example)
    returns = np.random.normal(0.001, 0.02, 1000)  # 1000 days
    prices = 100 * np.cumprod(1 + returns)
    
    # Initialize models
    hmm_model = OnlineHMM(n_states=3)
    hmm_model.fit(returns[:200])  # Initialize with first 200 days
    
    # Create strategy
    strategy = DirectRegimeStrategy(hmm_model)
    
    # Run backtest
    backtester = HMMBacktester()
    results = backtester.backtest_strategy(
        strategy=strategy,
        returns_data=returns,
        prices_data=prices,
        regime_model=hmm_model
    )
    
    # Display results
    perf = results['performance_metrics']
    print(f"Total Return: {perf['total_return']:.2%}")
    print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
    print(f"Number of Trades: {perf['num_trades']}")
    
    # Analyze regime performance
    regime_perf = backtester.analyze_regime_performance()
    for regime, metrics in regime_perf.items():
        print(f"\n{regime} Regime Performance:")
        print(f"  Mean Return: {metrics['mean_return']:.4f}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

if __name__ == "__main__":
    run_hmm_backtest_example()
```

---

## Live Trading Integration

### Production Trading System

Production-ready HMM trading system:

```python
class ProductionHMMTradingSystem:
    """Production-ready HMM trading system"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize models and strategies
        self.hmm_model = OnlineHMM(
            n_states=config['n_states'],
            online_config=OnlineHMMConfig(**config['online_hmm_config'])
        )
        
        self.strategy = MultiSignalHMMStrategy(self.hmm_model)
        self.position_sizer = HMMPositionSizer(config['capital'])
        self.risk_manager = HMMRiskManager(config['max_risk'])
        
        # Trading state
        self.current_position = 0.0
        self.portfolio_value = config['capital']
        self.last_price = None
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Safety mechanisms
        self.circuit_breaker = CircuitBreaker(
            max_daily_loss=config.get('max_daily_loss', 0.05),
            max_position_change=config.get('max_position_change', 0.5)
        )
        
    def initialize_system(self, historical_data):
        """Initialize system with historical data"""
        try:
            print("Initializing HMM trading system...")
            
            # Train initial model
            self.hmm_model.fit(historical_data['returns'], verbose=True)
            
            # Validate model
            validation_metrics = self.validate_model(historical_data)
            
            if not validation_metrics['is_valid']:
                raise Exception(f"Model validation failed: {validation_metrics['errors']}")
            
            print(f" System initialized successfully")
            print(f"   Model accuracy: {validation_metrics['regime_accuracy']:.2%}")
            print(f"   Confidence: {validation_metrics['avg_confidence']:.2%}")
            
            return True
            
        except Exception as e:
            print(f" System initialization failed: {e}")
            return False
    
    def process_market_data(self, market_data):
        """Process new market data and generate trading decisions"""
        try:
            # Extract data
            current_price = market_data['price']
            timestamp = market_data['timestamp']
            
            # Calculate return
            if self.last_price is not None:
                log_return = np.log(current_price / self.last_price)
                
                # Update model
                self.hmm_model.add_observation(log_return)
                
                # Generate signal
                signal_info = self.strategy.generate_combined_signal(log_return)
                
                # Calculate position size
                regime_info = self.hmm_model.get_current_regime_info()
                position_info = self.position_sizer.calculate_position_size(
                    signal_strength=signal_info['combined_position'],
                    regime_info=regime_info,
                    current_volatility=self.estimate_current_volatility()
                )
                
                # Risk management check
                risk_check = self.risk_manager.check_risk_limits(
                    proposed_position=position_info['position_size'],
                    regime_info=regime_info,
                    current_portfolio={'value': self.portfolio_value, 'position': self.current_position}
                )
                
                # Circuit breaker check
                circuit_check = self.circuit_breaker.check_limits(
                    proposed_position=risk_check['approved_position'],
                    current_position=self.current_position,
                    current_pnl=self.calculate_current_pnl(current_price)
                )
                
                # Execute trade if approved
                if circuit_check['approved']:
                    trade_result = self.execute_trade(
                        target_position=circuit_check['final_position'],
                        current_price=current_price,
                        timestamp=timestamp
                    )
                    
                    # Update portfolio
                    self.update_portfolio(current_price, log_return)
                    
                    return {
                        'status': 'success',
                        'trade_executed': trade_result['trade_executed'],
                        'new_position': self.current_position,
                        'regime': regime_info['regime_interpretation'],
                        'confidence': regime_info['confidence'],
                        'signal_strength': signal_info['signal_strength']
                    }
                else:
                    return {
                        'status': 'blocked',
                        'reason': circuit_check['block_reason'],
                        'regime': regime_info['regime_interpretation']
                    }
            
            self.last_price = current_price
            return {'status': 'initialized'}
            
        except Exception as e:
            print(f" Error processing market data: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def execute_trade(self, target_position, current_price, timestamp):
        """Execute trading order"""
        position_change = target_position - self.current_position
        
        if abs(position_change) < 0.01:  # 1% minimum trade size
            return {'trade_executed': False, 'reason': 'insufficient_change'}
        
        # Calculate order details
        order_size = abs(position_change)
        order_side = 'BUY' if position_change > 0 else 'SELL'
        
        try:
            # Execute order (integrate with your broker API)
            order_result = self.broker_api.place_order(
                symbol=self.config['symbol'],
                side=order_side,
                size=order_size,
                price=current_price,
                order_type='MARKET'
            )
            
            if order_result['status'] == 'FILLED':
                # Update position
                self.current_position = target_position
                
                # Record trade
                self.performance_tracker.record_trade({
                    'timestamp': timestamp,
                    'side': order_side,
                    'size': order_size,
                    'price': current_price,
                    'new_position': self.current_position,
                    'order_id': order_result['order_id']
                })
                
                print(f" Trade executed: {order_side} {order_size:.3f} at {current_price:.2f}")
                print(f"   New position: {self.current_position:.3f}")
                
                return {'trade_executed': True, 'order_result': order_result}
            else:
                return {'trade_executed': False, 'reason': 'order_failed', 'details': order_result}
                
        except Exception as e:
            print(f" Trade execution failed: {e}")
            return {'trade_executed': False, 'reason': 'execution_error', 'error': str(e)}
    
    def validate_model(self, historical_data):
        """Validate model performance on historical data"""
        # Out-of-sample validation
        train_size = int(0.8 * len(historical_data['returns']))
        train_data = historical_data['returns'][:train_size]
        test_data = historical_data['returns'][train_size:]
        
        # Train on subset
        validation_model = OnlineHMM(n_states=self.config['n_states'])
        validation_model.fit(train_data)
        
        # Test regime detection accuracy
        test_regimes = validation_model.predict(test_data)
        test_probs = validation_model.predict_proba(test_data)
        
        # Calculate validation metrics
        avg_confidence = np.mean(np.max(test_probs, axis=1))
        regime_stability = self.calculate_regime_stability(test_regimes)
        
        # Validation criteria
        is_valid = (
            avg_confidence >= 0.6 and          # Minimum 60% average confidence
            regime_stability >= 0.3 and       # Minimum 30% stability
            len(np.unique(test_regimes)) >= 2  # At least 2 regimes detected
        )
        
        return {
            'is_valid': is_valid,
            'regime_accuracy': regime_stability,
            'avg_confidence': avg_confidence,
            'num_regimes_detected': len(np.unique(test_regimes)),
            'errors': [] if is_valid else ['Low confidence or stability']
        }
    
    def calculate_regime_stability(self, regimes):
        """Calculate regime stability metric"""
        if len(regimes) < 2:
            return 0.0
            
        # Count regime changes
        changes = np.sum(regimes[1:] != regimes[:-1])
        stability = 1.0 - (changes / len(regimes))
        return stability
    
    def estimate_current_volatility(self):
        """Estimate current market volatility"""
        # Simple volatility estimation (could be enhanced)
        return 0.02  # Default 2% daily volatility
    
    def calculate_current_pnl(self, current_price):
        """Calculate current unrealized P&L"""
        if self.last_price is None or self.current_position == 0:
            return 0.0
        
        price_change = (current_price - self.last_price) / self.last_price
        return self.current_position * self.portfolio_value * price_change
    
    def update_portfolio(self, current_price, log_return):
        """Update portfolio value and metrics"""
        # Calculate P&L
        if self.current_position != 0:
            pnl = self.current_position * self.portfolio_value * log_return
            self.portfolio_value += pnl
        
        # Update performance tracking
        self.performance_tracker.update_daily_metrics(
            portfolio_value=self.portfolio_value,
            position=self.current_position,
            price=current_price,
            log_return=log_return
        )

class CircuitBreaker:
    """Safety circuit breaker for trading system"""
    
    def __init__(self, max_daily_loss=0.05, max_position_change=0.5):
        self.max_daily_loss = max_daily_loss
        self.max_position_change = max_position_change
        self.daily_pnl = 0.0
        self.daily_reset_time = None
        
    def check_limits(self, proposed_position, current_position, current_pnl):
        """Check if proposed trade meets safety limits"""
        
        # Reset daily tracking
        self.reset_daily_tracking()
        
        # Check daily loss limit
        if abs(current_pnl) > self.max_daily_loss:
            return {
                'approved': False,
                'block_reason': f'Daily loss limit exceeded: {current_pnl:.2%}',
                'final_position': current_position  # Maintain current position
            }
        
        # Check position change limit
        position_change = abs(proposed_position - current_position)
        if position_change > self.max_position_change:
            # Scale down position change
            scaled_change = self.max_position_change * np.sign(proposed_position - current_position)
            final_position = current_position + scaled_change
            
            return {
                'approved': True,
                'block_reason': f'Position change scaled down from {position_change:.2f} to {self.max_position_change:.2f}',
                'final_position': final_position
            }
        
        return {
            'approved': True,
            'block_reason': None,
            'final_position': proposed_position
        }
    
    def reset_daily_tracking(self):
        """Reset daily tracking at start of new trading day"""
        from datetime import datetime
        
        current_time = datetime.now()
        if self.daily_reset_time is None or current_time.date() != self.daily_reset_time.date():
            self.daily_pnl = 0.0
            self.daily_reset_time = current_time

# Example configuration and usage
def create_production_config():
    """Create production configuration"""
    return {
        'symbol': 'SPY',
        'n_states': 3,
        'capital': 100000,
        'max_risk': 0.02,
        'max_daily_loss': 0.03,
        'max_position_change': 0.3,
        'online_hmm_config': {
            'forgetting_factor': 0.98,
            'adaptation_rate': 0.05,
            'parameter_smoothing': True,
            'enable_change_detection': True
        }
    }

def run_production_system():
    """Run production trading system"""
    
    # Create system
    config = create_production_config()
    trading_system = ProductionHMMTradingSystem(config)
    
    # Initialize with historical data
    historical_data = {
        'returns': np.random.normal(0.001, 0.02, 500),  # Replace with real data
        'prices': None  # Replace with real price data
    }
    
    if trading_system.initialize_system(historical_data):
        print(" Production trading system is running...")
        
        # In production, this would be connected to live market data
        # Example simulation:
        for day in range(10):
            market_data = {
                'price': 100 + day + np.random.normal(0, 1),
                'timestamp': f"2024-01-{day+1:02d}",
            }
            
            result = trading_system.process_market_data(market_data)
            print(f"Day {day+1}: {result}")
    
    else:
        print(" Failed to initialize trading system")

if __name__ == "__main__":
    run_production_system()
```

---

## Performance Metrics

### Comprehensive Performance Analysis

Advanced performance metrics specifically designed for regime-based strategies:

```python
class HMMPerformanceAnalyzer:
    """Comprehensive performance analysis for HMM trading strategies"""
    
    def __init__(self):
        self.metrics = {}
        
    def analyze_strategy_performance(self, portfolio_returns, benchmark_returns, 
                                   regime_history, trade_history):
        """Comprehensive performance analysis"""
        
        # Basic performance metrics
        basic_metrics = self.calculate_basic_metrics(portfolio_returns, benchmark_returns)
        
        # Regime-specific metrics
        regime_metrics = self.calculate_regime_metrics(portfolio_returns, regime_history)
        
        # Trading efficiency metrics
        trading_metrics = self.calculate_trading_metrics(trade_history, portfolio_returns)
        
        # Risk-adjusted metrics
        risk_metrics = self.calculate_risk_metrics(portfolio_returns, benchmark_returns)
        
        # Regime transition analysis
        transition_metrics = self.analyze_regime_transitions(regime_history, portfolio_returns)
        
        return {
            'basic_metrics': basic_metrics,
            'regime_metrics': regime_metrics,
            'trading_metrics': trading_metrics,
            'risk_metrics': risk_metrics,
            'transition_metrics': transition_metrics,
            'overall_score': self.calculate_overall_score(basic_metrics, regime_metrics, risk_metrics)
        }
    
    def calculate_basic_metrics(self, portfolio_returns, benchmark_returns):
        """Calculate basic performance metrics"""
        
        # Convert to numpy arrays
        port_ret = np.array(portfolio_returns)
        bench_ret = np.array(benchmark_returns)
        
        # Returns
        total_return = np.prod(1 + port_ret) - 1
        annualized_return = (1 + total_return) ** (252 / len(port_ret)) - 1
        
        benchmark_total = np.prod(1 + bench_ret) - 1
        benchmark_annualized = (1 + benchmark_total) ** (252 / len(bench_ret)) - 1
        
        # Volatility and risk-adjusted returns
        volatility = np.std(port_ret) * np.sqrt(252)
        benchmark_vol = np.std(bench_ret) * np.sqrt(252)
        
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        benchmark_sharpe = benchmark_annualized / benchmark_vol if benchmark_vol > 0 else 0
        
        # Excess returns
        excess_returns = port_ret - bench_ret
        alpha = annualized_return - benchmark_annualized
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'benchmark_return': benchmark_annualized,
            'benchmark_sharpe': benchmark_sharpe,
            'excess_return': alpha,
            'tracking_error': np.std(excess_returns) * np.sqrt(252)
        }
    
    def calculate_regime_metrics(self, portfolio_returns, regime_history):
        """Calculate performance metrics by regime"""
        
        regime_performance = {}
        regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        
        for regime_num in range(3):
            # Extract returns during this regime
            regime_mask = np.array([r['regime'] for r in regime_history]) == regime_num
            regime_returns = np.array(portfolio_returns)[regime_mask[1:]]  # Align with returns
            
            if len(regime_returns) > 0:
                # Calculate regime-specific metrics
                regime_total_return = np.prod(1 + regime_returns) - 1
                regime_mean_return = np.mean(regime_returns)
                regime_volatility = np.std(regime_returns)
                regime_sharpe = regime_mean_return / regime_volatility * np.sqrt(252) if regime_volatility > 0 else 0
                
                # Hit rate and profit metrics
                positive_days = np.sum(regime_returns > 0)
                hit_rate = positive_days / len(regime_returns)
                
                avg_win = np.mean(regime_returns[regime_returns > 0]) if positive_days > 0 else 0
                avg_loss = np.mean(regime_returns[regime_returns < 0]) if positive_days < len(regime_returns) else 0
                
                # Regime duration analysis
                regime_durations = self.calculate_regime_durations(regime_history, regime_num)
                
                regime_performance[regime_names[regime_num]] = {
                    'total_return': regime_total_return,
                    'mean_daily_return': regime_mean_return,
                    'volatility': regime_volatility,
                    'sharpe_ratio': regime_sharpe,
                    'hit_rate': hit_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
                    'num_periods': len(regime_returns),
                    'avg_duration': np.mean(regime_durations) if regime_durations else 0,
                    'max_duration': np.max(regime_durations) if regime_durations else 0
                }
            else:
                regime_performance[regime_names[regime_num]] = {
                    'total_return': 0, 'mean_daily_return': 0, 'volatility': 0,
                    'sharpe_ratio': 0, 'hit_rate': 0, 'avg_win': 0, 'avg_loss': 0,
                    'profit_factor': 0, 'num_periods': 0, 'avg_duration': 0, 'max_duration': 0
                }
        
        return regime_performance
    
    def calculate_regime_durations(self, regime_history, target_regime):
        """Calculate duration statistics for specific regime"""
        durations = []
        current_duration = 0
        
        for regime_info in regime_history:
            if regime_info['regime'] == target_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Add final duration if regime continues to end
        if current_duration > 0:
            durations.append(current_duration)
            
        return durations
    
    def calculate_trading_metrics(self, trade_history, portfolio_returns):
        """Calculate trading efficiency metrics"""
        
        if not trade_history:
            return {
                'num_trades': 0,
                'avg_trade_return': 0,
                'win_rate': 0,
                'avg_winning_trade': 0,
                'avg_losing_trade': 0,
                'profit_factor': 0,
                'total_transaction_costs': 0,
                'cost_drag': 0
            }
        
        # Trade analysis
        num_trades = len(trade_history)
        total_transaction_costs = sum([trade['transaction_cost'] for trade in trade_history])
        
        # Estimate trade returns (simplified - would need more sophisticated calculation)
        trade_returns = []
        for i in range(len(trade_history) - 1):
            # Simple approximation of trade return
            trade_period_returns = portfolio_returns[trade_history[i]['date_index']:trade_history[i+1]['date_index']]
            if len(trade_period_returns) > 0:
                trade_return = np.prod(1 + trade_period_returns) - 1
                trade_returns.append(trade_return)
        
        if trade_returns:
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]
            
            win_rate = len(winning_trades) / len(trade_returns)
            avg_winning_trade = np.mean(winning_trades) if winning_trades else 0
            avg_losing_trade = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(avg_winning_trade / avg_losing_trade) if avg_losing_trade != 0 else float('inf')
            avg_trade_return = np.mean(trade_returns)
        else:
            win_rate = 0
            avg_winning_trade = 0
            avg_losing_trade = 0
            profit_factor = 0
            avg_trade_return = 0
        
        # Cost drag calculation
        total_portfolio_return = np.prod(1 + portfolio_returns) - 1
        cost_drag = total_transaction_costs / (total_portfolio_return + 1) if total_portfolio_return > -1 else 0
        
        return {
            'num_trades': num_trades,
            'avg_trade_return': avg_trade_return,
            'win_rate': win_rate,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'profit_factor': profit_factor,
            'total_transaction_costs': total_transaction_costs,
            'cost_drag': cost_drag,
            'trades_per_year': num_trades * 252 / len(portfolio_returns)
        }
    
    def calculate_risk_metrics(self, portfolio_returns, benchmark_returns):
        """Calculate comprehensive risk metrics"""
        
        port_ret = np.array(portfolio_returns)
        bench_ret = np.array(benchmark_returns)
        
        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + port_ret)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        
        max_drawdown = np.min(drawdowns)
        avg_drawdown = np.mean(drawdowns[drawdowns < 0]) if np.any(drawdowns < 0) else 0
        
        # Drawdown duration
        in_drawdown = drawdowns < -0.01  # 1% threshold
        drawdown_periods = self.calculate_consecutive_periods(in_drawdown)
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = np.max(drawdown_periods) if drawdown_periods else 0
        
        # Tail risk metrics
        var_95 = np.percentile(port_ret, 5)  # 5% VaR
        cvar_95 = np.mean(port_ret[port_ret <= var_95])  # Conditional VaR
        
        # Skewness and kurtosis
        skewness = self.calculate_skewness(port_ret)
        kurtosis = self.calculate_kurtosis(port_ret)
        
        # Beta and correlation with benchmark
        if len(port_ret) == len(bench_ret):
            beta = np.cov(port_ret, bench_ret)[0, 1] / np.var(bench_ret)
            correlation = np.corrcoef(port_ret, bench_ret)[0, 1]
        else:
            beta = 0
            correlation = 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'beta': beta,
            'correlation_with_benchmark': correlation,
            'downside_deviation': np.std(port_ret[port_ret < 0]) * np.sqrt(252) if np.any(port_ret < 0) else 0
        }
    
    def analyze_regime_transitions(self, regime_history, portfolio_returns):
        """Analyze performance around regime transitions"""
        
        transitions = []
        for i in range(1, len(regime_history)):
            if regime_history[i]['regime'] != regime_history[i-1]['regime']:
                transitions.append({
                    'date_index': i,
                    'from_regime': regime_history[i-1]['regime'],
                    'to_regime': regime_history[i]['regime'],
                    'confidence_before': regime_history[i-1]['confidence'],
                    'confidence_after': regime_history[i]['confidence']
                })
        
        if not transitions:
            return {'num_transitions': 0}
        
        # Analyze returns around transitions
        transition_performance = {}
        
        for window in [1, 3, 5, 10]:  # Days before/after transition
            before_returns = []
            after_returns = []
            
            for transition in transitions:
                idx = transition['date_index']
                
                # Before transition
                start_before = max(0, idx - window)
                before_rets = portfolio_returns[start_before:idx]
                if len(before_rets) == window:
                    before_returns.extend(before_rets)
                
                # After transition
                end_after = min(len(portfolio_returns), idx + window)
                after_rets = portfolio_returns[idx:end_after]
                if len(after_rets) == window:
                    after_returns.extend(after_rets)
            
            transition_performance[f'{window}d'] = {
                'before_avg': np.mean(before_returns) if before_returns else 0,
                'after_avg': np.mean(after_returns) if after_returns else 0,
                'before_vol': np.std(before_returns) if before_returns else 0,
                'after_vol': np.std(after_returns) if after_returns else 0,
                'improvement': (np.mean(after_returns) - np.mean(before_returns)) if (before_returns and after_returns) else 0
            }
        
        return {
            'num_transitions': len(transitions),
            'transitions_per_year': len(transitions) * 252 / len(regime_history),
            'transition_performance': transition_performance,
            'avg_confidence_change': np.mean([t['confidence_after'] - t['confidence_before'] for t in transitions])
        }
    
    def calculate_consecutive_periods(self, boolean_array):
        """Calculate lengths of consecutive True periods"""
        periods = []
        current_period = 0
        
        for value in boolean_array:
            if value:
                current_period += 1
            else:
                if current_period > 0:
                    periods.append(current_period)
                    current_period = 0
        
        if current_period > 0:
            periods.append(current_period)
            
        return periods
    
    def calculate_skewness(self, returns):
        """Calculate skewness of returns"""
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0
        return np.mean(((returns - mean_ret) / std_ret) ** 3)
    
    def calculate_kurtosis(self, returns):
        """Calculate excess kurtosis of returns"""
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0
        return np.mean(((returns - mean_ret) / std_ret) ** 4) - 3
    
    def calculate_overall_score(self, basic_metrics, regime_metrics, risk_metrics):
        """Calculate overall strategy score"""
        
        # Weighted scoring components
        components = {
            'returns': basic_metrics['annualized_return'] * 100,  # Weight returns highly
            'risk_adjusted': basic_metrics['sharpe_ratio'] * 50,  # Risk-adjusted performance
            'alpha': basic_metrics['alpha'] * 75,  # Excess return over benchmark
            'max_drawdown': (1 + risk_metrics['max_drawdown']) * 30,  # Lower drawdown is better
            'regime_consistency': self.calculate_regime_consistency_score(regime_metrics) * 40
        }
        
        # Calculate weighted score
        total_score = sum(components.values())
        
        return {
            'overall_score': total_score,
            'score_components': components,
            'rating': self.get_rating(total_score)
        }
    
    def calculate_regime_consistency_score(self, regime_metrics):
        """Calculate consistency score across regimes"""
        regime_sharpes = [metrics['sharpe_ratio'] for metrics in regime_metrics.values()]
        
        # Score based on positive Sharpe ratios across regimes
        positive_sharpes = sum(1 for sharpe in regime_sharpes if sharpe > 0)
        consistency_score = positive_sharpes / len(regime_sharpes) if regime_sharpes else 0
        
        return consistency_score
    
    def get_rating(self, score):
        """Convert numerical score to rating"""
        if score >= 80:
            return 'Excellent'
        elif score >= 60:
            return 'Very Good'
        elif score >= 40:
            return 'Good'
        elif score >= 20:
            return 'Fair'
        else:
            return 'Poor'

# Example usage
def analyze_hmm_strategy_performance():
    """Example of comprehensive performance analysis"""
    
    # Example data (replace with actual strategy results)
    portfolio_returns = np.random.normal(0.0008, 0.015, 1000)  # Daily returns
    benchmark_returns = np.random.normal(0.0005, 0.018, 1000)   # Market returns
    
    # Mock regime history
    regime_history = []
    current_regime = 1
    for i in range(1000):
        if np.random.random() < 0.05:  # 5% chance of regime change
            current_regime = (current_regime + 1) % 3
        regime_history.append({
            'regime': current_regime,
            'confidence': 0.6 + 0.3 * np.random.random()
        })
    
    # Mock trade history
    trade_history = []
    for i in range(0, 1000, 50):  # Trade every 50 days
        trade_history.append({
            'date_index': i,
            'transaction_cost': 100 + 50 * np.random.random()
        })
    
    # Analyze performance
    analyzer = HMMPerformanceAnalyzer()
    results = analyzer.analyze_strategy_performance(
        portfolio_returns, benchmark_returns, regime_history, trade_history
    )
    
    # Display results
    print(" HMM Strategy Performance Analysis")
    print("=" * 50)
    
    basic = results['basic_metrics']
    print(f"Total Return: {basic['total_return']:.2%}")
    print(f"Annualized Return: {basic['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {basic['sharpe_ratio']:.2f}")
    print(f"Alpha: {basic['alpha']:.2%}")
    print(f"Information Ratio: {basic['information_ratio']:.2f}")
    
    print(f"\n Regime Performance:")
    for regime, metrics in results['regime_metrics'].items():
        print(f"{regime:>9}: Return={metrics['mean_daily_return']*252:.2%}, "
              f"Sharpe={metrics['sharpe_ratio']:.2f}, "
              f"Hit Rate={metrics['hit_rate']:.1%}")
    
    risk = results['risk_metrics']
    print(f"\n[WARNING]  Risk Metrics:")
    print(f"Max Drawdown: {risk['max_drawdown']:.2%}")
    print(f"VaR (95%): {risk['var_95']:.2%}")
    print(f"Beta: {risk['beta']:.2f}")
    
    trade = results['trading_metrics']
    print(f"\n💼 Trading Metrics:")
    print(f"Number of Trades: {trade['num_trades']}")
    print(f"Win Rate: {trade['win_rate']:.1%}")
    print(f"Profit Factor: {trade['profit_factor']:.2f}")
    
    score = results['overall_score']
    print(f"\n🏆 Overall Score: {score['overall_score']:.1f} ({score['rating']})")

if __name__ == "__main__":
    analyze_hmm_strategy_performance()
```

This comprehensive Trading Applications Guide provides everything needed to implement, test, and deploy HMM-based trading strategies. It covers the full spectrum from basic signal generation to production-ready systems with sophisticated risk management and performance analysis.

---

*For additional information, see the [Online HMM Documentation](online_hmm.md), [Configuration Guide](configuration_guide.md), and [Mathematical Foundations](mathematical_foundations.md).*