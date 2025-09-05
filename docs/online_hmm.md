# Online Hidden Markov Models for Real-Time Market Regime Detection

*Comprehensive guide to streaming HMM capabilities for real-time financial applications*

---

## Table of Contents

1. [Overview](#overview)
2. [Online Learning vs Batch Learning](#online-learning-vs-batch-learning)
3. [Theoretical Foundation](#theoretical-foundation)
4. [Architecture and Design](#architecture-and-design)
5. [API Reference](#api-reference)
6. [Configuration Guide](#configuration-guide)
7. [Real-Time Processing](#real-time-processing)
8. [Performance Optimization](#performance-optimization)
9. [Use Cases and Applications](#use-cases-and-applications)
10. [Implementation Examples](#implementation-examples)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The Online Hidden Markov Model (OnlineHMM) extends the traditional batch HMM approach to enable **incremental learning** from streaming market data. This allows for real-time regime detection without the need to retrain the entire model when new observations arrive.

### Key Benefits

- **✅ Real-Time Processing**: Process observations as they arrive (< 10ms per observation)
- **✅ Temporal Consistency**: Maintain stable regime classifications (70%+ consistency)
- **✅ Memory Efficiency**: Bounded memory usage with exponential forgetting
- **✅ Adaptive Learning**: Automatically adapt to changing market conditions
- **✅ Production Ready**: Designed for high-frequency trading and live applications

### Core Innovation

Traditional HMMs require complete retraining when new data arrives, causing:
- **Historical Revision**: Past regime labels change dramatically
- **Computational Inefficiency**: O(T²) complexity for T observations
- **Parameter Instability**: Coefficients fluctuate excessively with new data

Online HMMs solve these problems through:
- **Incremental Updates**: O(1) complexity per new observation
- **Exponential Forgetting**: Gradual decay of old information
- **Parameter Smoothing**: Stable coefficient evolution
- **Sufficient Statistics**: Memory-efficient state tracking

---

## Online Learning vs Batch Learning

### Traditional Batch HMM

```
New Data Arrives → Complete Retraining → Updated Model
     ↓                      ↓                ↓
- All historical data  - Baum-Welch EM    - All parameters change
- Quadratic complexity - Full forward-    - Historical regimes 
- Memory intensive       backward pass      may be revised
```

**Problems:**
- Processing time scales with dataset size
- Parameter instability with new observations
- Historical regime labels are not stable
- Not suitable for real-time applications

### Online HMM Approach

```
New Observation → Incremental Update → Stable Evolution
      ↓                    ↓                ↓
- Single observation  - Update sufficient  - Smooth parameter
- Constant complexity   statistics          changes
- Memory efficient    - Forward step only  - Temporal consistency
```

**Advantages:**
- Constant O(1) processing time per observation
- Stable parameter evolution with smoothing
- Historical regime stability (70%+ consistency)
- Real-time processing capabilities

### Comparison Table

| Aspect | Batch HMM | Online HMM |
|--------|-----------|------------|
| **Processing Time** | O(N²T) | O(N²) |
| **Memory Usage** | O(T) | O(1) |
| **Parameter Stability** | Low | High |
| **Historical Consistency** | Poor | Good (70%+) |
| **Real-time Suitability** | No | Yes |
| **Adaptation Speed** | Slow | Fast |
| **Production Ready** | Limited | Yes |

---

## Theoretical Foundation

### Exponential Forgetting Framework

The online HMM uses **exponential forgetting** to weight recent observations more heavily than older ones.

#### Forgetting Factor (λ)
- **λ = 0.98**: Recent observations have weight 1.0, previous day has weight 0.98, etc.
- **λ = 0.95**: More aggressive forgetting, faster adaptation
- **λ = 0.99**: Conservative forgetting, slower adaptation

#### Effective Sample Size
The effective number of observations contributing to parameter estimates:
```
N_eff = 1 / (1 - λ) ≈ 50 observations for λ = 0.98
```

### Sufficient Statistics with Forgetting

Instead of storing all historical observations, we maintain **sufficient statistics** that are updated incrementally.

#### State Occupation Statistics
```python
# Update rule for state probabilities
gamma_sum[i] = forgetting_factor * gamma_sum[i] + current_gamma[i]
```

#### Transition Count Statistics
```python
# Update rule for transition counts
xi_sum[i,j] = forgetting_factor * xi_sum[i,j] + current_xi[i,j]
```

#### Observation Statistics
```python
# Mean calculation
obs_sum[i] = forgetting_factor * obs_sum[i] + current_gamma[i] * observation
obs_count[i] = forgetting_factor * obs_count[i] + current_gamma[i]
mean[i] = obs_sum[i] / obs_count[i]

# Variance calculation  
obs_sq_sum[i] = forgetting_factor * obs_sq_sum[i] + current_gamma[i] * observation**2
variance[i] = (obs_sq_sum[i] / obs_count[i]) - mean[i]**2
```

### Parameter Smoothing

To prevent excessive parameter volatility, new parameters are blended with old ones:

```python
# Parameter smoothing
smoothing_weight = 0.8
new_param = (1 - adaptation_rate) * old_param + adaptation_rate * updated_param
final_param = smoothing_weight * old_param + (1 - smoothing_weight) * new_param
```

### Recursive State Probability Update

For each new observation, state probabilities are updated using a **prediction-correction** framework:

#### Prediction Step
```python
# Predict next state probabilities
predicted_probs = current_probs @ transition_matrix
```

#### Correction Step  
```python
# Correct using new observation likelihood
likelihoods = [gaussian_pdf(observation, mean[i], std[i]) for i in range(n_states)]
corrected_probs = predicted_probs * likelihoods
corrected_probs /= corrected_probs.sum()  # Normalize
```

---

## Architecture and Design

### Core Components

```
OnlineHMM
├── Base HMM (inheritance from HiddenMarkovModel)
├── OnlineHMMConfig (configuration management)
├── SufficientStatistics (memory-efficient tracking)
├── ChangeDetection (structural break monitoring)
└── StreamingInterface (real-time data processing)
```

### Data Flow Architecture

```
Market Data Stream
        ↓
    Validation & Preprocessing
        ↓
    Online HMM Processing
        ├── Forward Step (prediction)
        ├── Parameter Update (correction)
        ├── Sufficient Statistics Update
        └── Change Detection
        ↓
    Regime Information Output
        ├── Current regime probabilities
        ├── Confidence measures
        ├── Transition predictions
        └── Trading signals
```

### Memory Management

The online HMM maintains bounded memory through:

1. **Rolling Window**: Keep only recent N observations in memory
2. **Sufficient Statistics**: Store aggregated statistics instead of raw data
3. **Exponential Decay**: Gradually forget old information
4. **Garbage Collection**: Periodically clean up unused data structures

---

## API Reference

### Core Classes

#### `OnlineHMM`

The main class for online HMM regime detection.

```python
from hidden_regime.models import OnlineHMM, OnlineHMMConfig, HMMConfig

# Initialize online HMM
online_config = OnlineHMMConfig(forgetting_factor=0.98, adaptation_rate=0.05)
hmm_config = HMMConfig(n_states=3, initialization_method='kmeans')

online_hmm = OnlineHMM(
    n_states=3,
    config=hmm_config,
    online_config=online_config
)
```

**Constructor Parameters:**
- `n_states` (int): Number of hidden regimes
- `config` (HMMConfig): Base HMM configuration
- `online_config` (OnlineHMMConfig): Online learning parameters

#### Key Methods

##### `fit(returns, verbose=False)`
Initialize the model with historical data (same as batch HMM).

##### `add_observation(new_return)`
Process a single new observation and update the model incrementally.

```python
# Process new market data
today_return = 0.015
online_hmm.add_observation(today_return)

# Get current regime information
regime_info = online_hmm.get_current_regime_info()
print(f"Current regime: {regime_info['regime_interpretation']}")
print(f"Confidence: {regime_info['confidence']:.2%}")
```

**Parameters:**
- `new_return` (float): New log return observation

**Returns:** None (updates internal state)

##### `get_current_regime_info()`
Get comprehensive information about the current market regime.

**Returns:**
```python
{
    'current_state': int,                    # Most likely current state (0, 1, 2)
    'state_probabilities': np.ndarray,       # Probabilities for all states
    'confidence': float,                     # Confidence in current state
    'regime_interpretation': str,            # Human-readable regime ("Bull Market", etc.)
    'days_in_regime': int,                  # Estimated days in current regime
    'expected_duration': float,             # Expected total regime duration
    'transition_probability': float,         # Probability of regime change soon
    'regime_strength': str                  # "Strong", "Moderate", "Weak"
}
```

##### `predict_regime_transition(horizon=5)`
Predict regime transition probabilities over the next few periods.

```python
# Predict transitions over next 5 days
transition_forecast = online_hmm.predict_regime_transition(horizon=5)
print(f"Probability of regime change in next 5 days: {transition_forecast['change_probability']:.2%}")
```

**Parameters:**
- `horizon` (int): Number of periods to forecast

**Returns:**
```python
{
    'horizon': int,                         # Forecast horizon
    'current_regime_prob': np.ndarray,      # Probability of staying in current regime
    'change_probability': float,            # Overall probability of any regime change
    'most_likely_transition': str,          # Most likely new regime if change occurs
    'transition_probabilities': dict        # Detailed transition probabilities
}
```

##### `get_sufficient_statistics()`
Access current sufficient statistics for analysis and debugging.

##### `reset_adaptation()`
Reset adaptation mechanisms while keeping learned parameters.

#### `OnlineHMMConfig`

Configuration class for online learning parameters.

```python
config = OnlineHMMConfig(
    # Forgetting parameters
    forgetting_factor=0.98,                 # Memory decay rate (0.95-0.99)
    adaptation_rate=0.05,                   # Learning speed (0.01-0.1)
    
    # Stability mechanisms  
    min_observations_for_update=10,         # Min obs before updates
    parameter_smoothing=True,               # Enable parameter smoothing
    smoothing_weight=0.8,                  # Weight for previous parameters
    
    # Memory management
    rolling_window_size=1000,              # Max observations in memory
    sufficient_stats_decay=0.99,           # Decay rate for statistics
    
    # Change detection
    enable_change_detection=True,          # Monitor structural breaks
    change_detection_threshold=3.0,        # Std devs for change detection
    change_detection_window=50,            # Window size for monitoring
    
    # Convergence
    convergence_tolerance=1e-4,            # Parameter convergence tolerance
    max_adaptation_iterations=5            # Max iterations per observation
)
```

#### `SufficientStatistics`

Internal class for tracking model statistics efficiently.

**Key Attributes:**
- `gamma_sum`: State occupation counts
- `xi_sum`: State transition counts  
- `obs_sum`: Weighted observation sums
- `obs_sq_sum`: Weighted squared observation sums
- `obs_count`: Weighted observation counts

---

## Configuration Guide

### Parameter Tuning by Use Case

#### High-Frequency Trading (Minutes/Seconds)
```python
config = OnlineHMMConfig(
    forgetting_factor=0.95,        # Faster adaptation to recent changes
    adaptation_rate=0.1,           # Higher learning rate
    min_observations_for_update=5, # Update more frequently  
    smoothing_weight=0.6,          # Less smoothing for faster response
    rolling_window_size=500        # Smaller memory footprint
)
```

#### Daily Trading
```python
config = OnlineHMMConfig(
    forgetting_factor=0.98,        # Balanced adaptation
    adaptation_rate=0.05,          # Moderate learning rate
    min_observations_for_update=10,# Standard update frequency
    smoothing_weight=0.8,          # Good stability
    rolling_window_size=1000       # Standard memory size
)
```

#### Long-Term Analysis (Weekly/Monthly)
```python
config = OnlineHMMConfig(
    forgetting_factor=0.99,        # Slow adaptation, long memory
    adaptation_rate=0.02,          # Conservative learning rate
    min_observations_for_update=20,# Less frequent updates
    smoothing_weight=0.9,          # High stability
    rolling_window_size=2000       # Larger memory for patterns
)
```

#### Volatile/Crisis Markets
```python
config = OnlineHMMConfig(
    forgetting_factor=0.96,        # Moderate adaptation
    adaptation_rate=0.08,          # Higher learning for rapid changes
    enable_change_detection=True,  # Monitor for structural breaks
    change_detection_threshold=2.5,# Lower threshold for sensitivity
    parameter_smoothing=True       # Maintain stability despite volatility
)
```

### Parameter Guidelines

#### Forgetting Factor (λ)
- **0.99**: Very conservative, 100-day effective memory
- **0.98**: Balanced, 50-day effective memory (recommended)
- **0.97**: Moderate, 33-day effective memory
- **0.95**: Aggressive, 20-day effective memory
- **0.90**: Very aggressive, 10-day effective memory

#### Adaptation Rate (α)
- **0.01**: Very conservative updates
- **0.05**: Balanced updates (recommended)
- **0.1**: Moderate updates
- **0.2**: Aggressive updates (use with caution)

#### Smoothing Weight (β)
- **0.9**: Maximum stability, slow adaptation
- **0.8**: High stability (recommended)
- **0.7**: Balanced stability and adaptation
- **0.6**: Lower stability, faster adaptation
- **0.5**: Minimum stability, maximum adaptation

### Configuration Validation

The system automatically validates configuration parameters:

```python
try:
    config = OnlineHMMConfig(forgetting_factor=0.5)  # Too low!
except ValueError as e:
    print(f"Configuration error: {e}")
    # Output: "forgetting_factor must be between 0.9 and 0.999"
```

---

## Real-Time Processing

### Data Ingestion Patterns

#### Single Observation Processing
```python
# Process observations one at a time
for new_return in live_data_stream:
    start_time = time.time()
    
    online_hmm.add_observation(new_return)
    regime_info = online_hmm.get_current_regime_info()
    
    processing_time = time.time() - start_time
    
    # Make trading decisions
    position = calculate_position_size(regime_info)
    execute_trade(position)
    
    print(f"Processed in {processing_time*1000:.2f}ms")
```

#### Micro-Batch Processing
```python
# Process small batches for efficiency
batch_size = 5
batch = []

for new_return in live_data_stream:
    batch.append(new_return)
    
    if len(batch) >= batch_size:
        # Process batch
        for return_val in batch:
            online_hmm.add_observation(return_val)
        
        # Get final regime info
        regime_info = online_hmm.get_current_regime_info()
        
        # Clear batch
        batch = []
```

### Integration with Live Data Feeds

#### WebSocket Integration
```python
import websocket
import json

def on_message(ws, message):
    """Handle incoming market data"""
    data = json.loads(message)
    
    if 'price' in data:
        # Calculate return
        new_return = np.log(data['price'] / last_price)
        
        # Process with online HMM
        online_hmm.add_observation(new_return)
        regime_info = online_hmm.get_current_regime_info()
        
        # Update trading logic
        update_trading_strategy(regime_info)
        
        # Update last price
        last_price = data['price']

# Connect to data feed
ws = websocket.WebSocketApp("ws://market-data-feed.com", on_message=on_message)
ws.run_forever()
```

#### REST API Integration
```python
import requests
import time

def poll_market_data():
    """Poll market data and process with online HMM"""
    while True:
        try:
            # Get latest data
            response = requests.get("https://api.market-data.com/latest")
            data = response.json()
            
            # Calculate return
            current_price = data['price']
            if hasattr(poll_market_data, 'last_price'):
                new_return = np.log(current_price / poll_market_data.last_price)
                
                # Process with online HMM
                online_hmm.add_observation(new_return)
                regime_info = online_hmm.get_current_regime_info()
                
                # Log regime information
                print(f"Current regime: {regime_info['regime_interpretation']} "
                      f"(confidence: {regime_info['confidence']:.2%})")
            
            poll_market_data.last_price = current_price
            time.sleep(1)  # Poll every second
            
        except Exception as e:
            print(f"Error polling data: {e}")
            time.sleep(5)  # Wait before retry

# Start polling
poll_market_data()
```

---

## Performance Optimization

### Processing Speed Optimization

#### Target Performance Metrics
- **Processing Time**: < 10ms per observation
- **Memory Usage**: < 100MB for standard configuration
- **Throughput**: > 1000 observations per second
- **Latency**: < 5ms for regime classification

#### Speed Optimization Techniques

##### 1. Efficient Matrix Operations
```python
# Use numpy vectorized operations
state_probs = np.dot(previous_probs, transition_matrix)

# Avoid Python loops where possible
emission_probs = np.array([gaussian_pdf(obs, means[i], stds[i]) for i in range(n_states)])
```

##### 2. Precomputed Constants
```python
class OptimizedOnlineHMM:
    def __init__(self, ...):
        # Precompute frequently used constants
        self.log_2pi = np.log(2 * np.pi)
        self.sqrt_2pi = np.sqrt(2 * np.pi)
        self.inv_stds = 1.0 / self.emission_stds  # Precompute inverse
    
    def fast_gaussian_pdf(self, x, mean, inv_std):
        """Fast Gaussian PDF computation"""
        diff = x - mean
        return inv_std / self.sqrt_2pi * np.exp(-0.5 * (diff * inv_std)**2)
```

##### 3. Memory Pool Allocation
```python
class MemoryEfficientOnlineHMM:
    def __init__(self, ...):
        # Pre-allocate arrays to avoid repeated allocation
        self.temp_probs = np.zeros(self.n_states)
        self.emission_cache = np.zeros(self.n_states)
        self.transition_cache = np.zeros(self.n_states)
    
    def add_observation(self, obs):
        # Reuse pre-allocated arrays
        self.compute_emissions(obs, out=self.emission_cache)
        self.update_probabilities(self.emission_cache, out=self.temp_probs)
```

### Memory Optimization

#### Memory Usage Monitoring
```python
def monitor_memory_usage(online_hmm):
    """Monitor memory usage of online HMM"""
    import psutil
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Get memory usage
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Get model-specific memory
    model_memory = {
        'sufficient_stats': sys.getsizeof(online_hmm.sufficient_statistics),
        'rolling_window': sys.getsizeof(online_hmm.rolling_window),
        'parameter_history': sys.getsizeof(online_hmm.parameter_history),
    }
    
    return {
        'total_memory_mb': memory_mb,
        'model_components': model_memory
    }
```

#### Memory Cleanup Strategies
```python
def cleanup_memory(online_hmm, max_memory_mb=100):
    """Clean up memory if usage exceeds threshold"""
    memory_info = monitor_memory_usage(online_hmm)
    
    if memory_info['total_memory_mb'] > max_memory_mb:
        # Reduce rolling window size
        online_hmm.rolling_window = online_hmm.rolling_window[-500:]
        
        # Clear parameter history
        online_hmm.parameter_history = online_hmm.parameter_history[-100:]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print(f"Memory cleaned up. Usage reduced to {monitor_memory_usage(online_hmm)['total_memory_mb']:.1f}MB")
```

---

## Use Cases and Applications

### 1. Real-Time Trading Systems

#### Intraday Regime Detection
```python
class IntradayTradingBot:
    def __init__(self):
        self.online_hmm = OnlineHMM(
            n_states=3,
            online_config=OnlineHMMConfig(forgetting_factor=0.96, adaptation_rate=0.08)
        )
        self.position = 0.0
        self.last_regime = None
    
    def process_tick(self, price_data):
        """Process each market tick"""
        # Calculate return
        if hasattr(self, 'last_price'):
            return_val = np.log(price_data['price'] / self.last_price)
            
            # Update HMM
            self.online_hmm.add_observation(return_val)
            regime_info = self.online_hmm.get_current_regime_info()
            
            # Check for regime change
            current_regime = regime_info['current_state']
            if current_regime != self.last_regime and regime_info['confidence'] > 0.8:
                self.handle_regime_change(regime_info)
            
            self.last_regime = current_regime
        
        self.last_price = price_data['price']
    
    def handle_regime_change(self, regime_info):
        """Handle regime transitions"""
        regime = regime_info['regime_interpretation']
        
        if 'Bull' in regime:
            self.position = min(self.position + 0.5, 1.0)  # Increase long position
        elif 'Bear' in regime:
            self.position = max(self.position - 0.5, -1.0)  # Increase short position
        else:  # Sideways
            self.position *= 0.5  # Reduce position size
        
        print(f"Regime change to {regime}. New position: {self.position:.2f}")
```

### 2. Portfolio Risk Management

#### Dynamic Risk Adjustment
```python
class DynamicRiskManager:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.online_hmm = OnlineHMM(n_states=3)
        self.risk_multipliers = {'Bull': 1.0, 'Sideways': 0.7, 'Bear': 0.3}
    
    def update_risk_limits(self, market_return):
        """Adjust portfolio risk based on current regime"""
        # Update regime detection
        self.online_hmm.add_observation(market_return)
        regime_info = self.online_hmm.get_current_regime_info()
        
        # Get current regime
        regime = regime_info['regime_interpretation'].split()[0]  # Extract 'Bull', 'Bear', etc.
        confidence = regime_info['confidence']
        
        # Calculate risk multiplier
        base_multiplier = self.risk_multipliers.get(regime, 0.5)
        confidence_adjustment = 0.5 + 0.5 * confidence  # Scale between 0.5 and 1.0
        
        risk_multiplier = base_multiplier * confidence_adjustment
        
        # Apply to portfolio
        self.portfolio.set_risk_multiplier(risk_multiplier)
        
        return {
            'regime': regime,
            'confidence': confidence,
            'risk_multiplier': risk_multiplier,
            'new_position_limit': self.portfolio.max_position * risk_multiplier
        }
```

### 3. Multi-Asset Regime Correlation

#### Cross-Asset Regime Analysis
```python
class MultiAssetRegimeTracker:
    def __init__(self, assets):
        self.assets = assets
        self.hmm_models = {
            asset: OnlineHMM(n_states=3) for asset in assets
        }
        self.regime_correlation_window = deque(maxlen=100)
    
    def update_all_regimes(self, returns_dict):
        """Update regime detection for all assets"""
        current_regimes = {}
        
        for asset, return_val in returns_dict.items():
            if asset in self.hmm_models:
                self.hmm_models[asset].add_observation(return_val)
                regime_info = self.hmm_models[asset].get_current_regime_info()
                current_regimes[asset] = regime_info
        
        # Track regime correlation
        self.regime_correlation_window.append(current_regimes)
        
        return current_regimes
    
    def analyze_regime_correlation(self):
        """Analyze correlation between asset regimes"""
        if len(self.regime_correlation_window) < 30:
            return None
        
        # Extract regime sequences
        regime_sequences = {asset: [] for asset in self.assets}
        
        for regime_snapshot in self.regime_correlation_window:
            for asset, regime_info in regime_snapshot.items():
                regime_sequences[asset].append(regime_info['current_state'])
        
        # Calculate correlation matrix
        import pandas as pd
        
        df = pd.DataFrame(regime_sequences)
        correlation_matrix = df.corr()
        
        return {
            'correlation_matrix': correlation_matrix,
            'avg_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
            'current_regimes': {asset: info['regime_interpretation'] 
                              for asset, info in list(self.regime_correlation_window)[-1].items()}
        }
```

---

## Implementation Examples

### Complete Trading System Example

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import time

class CompleteOnlineHMMTradingSystem:
    """Complete example of online HMM trading system"""
    
    def __init__(self, initial_capital=100000):
        # Initialize online HMM
        self.online_hmm = OnlineHMM(
            n_states=3,
            online_config=OnlineHMMConfig(
                forgetting_factor=0.98,
                adaptation_rate=0.05,
                enable_change_detection=True
            )
        )
        
        # Trading state
        self.capital = initial_capital
        self.position = 0.0
        self.last_price = None
        self.portfolio_value = initial_capital
        
        # Performance tracking
        self.trade_history = []
        self.regime_history = []
        self.performance_metrics = deque(maxlen=252)  # Track daily performance
        
        # Risk management
        self.max_position_size = 1.0
        self.stop_loss_pct = 0.02
        self.position_size_by_regime = {'Bull': 0.8, 'Sideways': 0.3, 'Bear': -0.4}
    
    def initialize_model(self, historical_returns):
        """Initialize the model with historical data"""
        print(f"Initializing model with {len(historical_returns)} historical observations...")
        
        # Fit initial model
        self.online_hmm.fit(historical_returns, verbose=True)
        
        print("✅ Model initialization complete")
        return self.online_hmm.analyze_regimes(historical_returns)
    
    def process_market_data(self, price, timestamp):
        """Process new market data point"""
        if self.last_price is None:
            self.last_price = price
            return None
        
        # Calculate return
        log_return = np.log(price / self.last_price)
        
        # Update online HMM
        start_time = time.time()
        self.online_hmm.add_observation(log_return)
        processing_time = time.time() - start_time
        
        # Get regime information
        regime_info = self.online_hmm.get_current_regime_info()
        
        # Make trading decision
        trading_signal = self.generate_trading_signal(regime_info, log_return, price)
        
        # Execute trades if needed
        if trading_signal['action'] != 'hold':
            self.execute_trade(trading_signal, price, timestamp)
        
        # Update portfolio value
        self.update_portfolio_value(price)
        
        # Store performance data
        performance_data = {
            'timestamp': timestamp,
            'price': price,
            'log_return': log_return,
            'regime': regime_info['regime_interpretation'],
            'confidence': regime_info['confidence'],
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'processing_time_ms': processing_time * 1000
        }
        
        self.regime_history.append(performance_data)
        self.performance_metrics.append(performance_data)
        self.last_price = price
        
        return performance_data
    
    def generate_trading_signal(self, regime_info, log_return, price):
        """Generate trading signals based on regime"""
        regime_type = regime_info['regime_interpretation'].split()[0]  # Bull, Bear, Sideways
        confidence = regime_info['confidence']
        
        # Base position size from regime
        target_position = self.position_size_by_regime.get(regime_type, 0.0)
        
        # Adjust for confidence
        confidence_multiplier = max(0.5, confidence)  # Minimum 50% confidence
        adjusted_target = target_position * confidence_multiplier
        
        # Current position
        current_position = self.position
        
        # Calculate required action
        position_diff = adjusted_target - current_position
        
        # Trading thresholds
        min_trade_size = 0.1
        
        if abs(position_diff) < min_trade_size:
            return {'action': 'hold', 'size': 0, 'reason': 'insufficient_signal'}
        
        # Stop loss check
        if self.check_stop_loss(price):
            return {'action': 'close', 'size': -current_position, 'reason': 'stop_loss'}
        
        # Generate signal
        if position_diff > 0:
            return {
                'action': 'buy',
                'size': position_diff,
                'target_position': adjusted_target,
                'regime': regime_type,
                'confidence': confidence,
                'reason': 'regime_signal'
            }
        else:
            return {
                'action': 'sell',
                'size': abs(position_diff),
                'target_position': adjusted_target,
                'regime': regime_type,
                'confidence': confidence,
                'reason': 'regime_signal'
            }
    
    def execute_trade(self, signal, price, timestamp):
        """Execute trading signal"""
        trade_size = signal['size']
        
        if signal['action'] in ['buy', 'sell']:
            # Calculate cost (assuming 0.1% transaction cost)
            transaction_cost = abs(trade_size) * price * 0.001
            
            # Update position
            if signal['action'] == 'buy':
                self.position += trade_size
            else:
                self.position -= trade_size
            
            # Deduct transaction cost
            self.capital -= transaction_cost
            
            # Record trade
            trade_record = {
                'timestamp': timestamp,
                'action': signal['action'],
                'size': trade_size,
                'price': price,
                'new_position': self.position,
                'cost': transaction_cost,
                'regime': signal.get('regime', 'Unknown'),
                'confidence': signal.get('confidence', 0.0),
                'reason': signal.get('reason', 'Unknown')
            }
            
            self.trade_history.append(trade_record)
            
            print(f"{timestamp}: {signal['action'].upper()} {trade_size:.3f} at ${price:.2f} "
                  f"({signal['regime']}, conf: {signal.get('confidence', 0):.2%}) "
                  f"-> Position: {self.position:.3f}")
        
        elif signal['action'] == 'close':
            # Close position
            self.position = 0.0
            print(f"{timestamp}: STOP LOSS - Position closed at ${price:.2f}")
    
    def check_stop_loss(self, current_price):
        """Check if stop loss should be triggered"""
        if self.position == 0:
            return False
        
        # Calculate unrealized P&L percentage
        price_change = (current_price - self.last_price) / self.last_price
        position_pnl = self.position * price_change
        
        # Trigger stop loss if loss exceeds threshold
        if position_pnl < -self.stop_loss_pct:
            return True
        
        return False
    
    def update_portfolio_value(self, current_price):
        """Update portfolio value"""
        # Cash + position value
        position_value = self.position * current_price
        self.portfolio_value = self.capital + position_value
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        if len(self.performance_metrics) < 2:
            return None
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(list(self.performance_metrics))
        
        # Calculate returns
        portfolio_returns = df['portfolio_value'].pct_change().dropna()
        
        # Performance metrics
        total_return = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0] - 1)
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1
        
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        rolling_max = df['portfolio_value'].expanding().max()
        drawdown = (df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Regime analysis
        regime_performance = df.groupby('regime').agg({
            'log_return': ['count', 'mean', 'std'],
            'confidence': 'mean'
        }).round(4)
        
        # Trading analysis
        total_trades = len(self.trade_history)
        avg_processing_time = df['processing_time_ms'].mean()
        
        return {
            'performance_metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_portfolio_value': df['portfolio_value'].iloc[-1]
            },
            'regime_performance': regime_performance,
            'trading_statistics': {
                'total_trades': total_trades,
                'avg_processing_time_ms': avg_processing_time,
                'current_position': self.position,
                'cash_balance': self.capital
            },
            'recent_regime_info': self.online_hmm.get_current_regime_info()
        }

# Example usage
def run_online_hmm_trading_example():
    """Run complete online HMM trading example"""
    
    # Initialize trading system
    trading_system = CompleteOnlineHMMTradingSystem(initial_capital=100000)
    
    # Load historical data for initialization
    # (In practice, load from your data source)
    np.random.seed(42)
    historical_returns = np.random.normal(0.001, 0.02, 500)  # 500 days of history
    
    # Initialize model
    initial_analysis = trading_system.initialize_model(historical_returns)
    print("Initial regime analysis:")
    for state, stats in initial_analysis['regime_statistics'].items():
        print(f"  State {state}: {stats['interpretation']}")
    
    # Simulate live trading
    print("\n" + "="*60)
    print("STARTING LIVE TRADING SIMULATION")
    print("="*60)
    
    current_price = 100.0
    timestamp = datetime.now()
    
    # Simulate 100 days of trading
    for day in range(100):
        # Simulate intraday prices (multiple ticks per day)
        for tick in range(10):  # 10 ticks per day
            # Random price movement
            price_change = np.random.normal(0.0001, 0.005)  # Small intraday movements
            current_price *= (1 + price_change)
            current_timestamp = timestamp + timedelta(days=day, hours=tick)
            
            # Process market data
            performance_data = trading_system.process_market_data(current_price, current_timestamp)
            
            # Print updates every 50 ticks
            if (day * 10 + tick) % 50 == 0 and performance_data:
                print(f"Day {day+1}, Tick {tick+1}: {performance_data['regime']} "
                      f"(conf: {performance_data['confidence']:.2%}) "
                      f"Portfolio: ${performance_data['portfolio_value']:,.0f} "
                      f"Position: {performance_data['position']:.3f}")
    
    # Final performance summary
    print("\n" + "="*60)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*60)
    
    summary = trading_system.get_performance_summary()
    
    if summary:
        perf = summary['performance_metrics']
        print(f"Total Return: {perf['total_return']:.2%}")
        print(f"Annualized Return: {perf['annualized_return']:.2%}")
        print(f"Volatility: {perf['volatility']:.2%}")
        print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
        print(f"Final Portfolio Value: ${perf['final_portfolio_value']:,.0f}")
        
        trading_stats = summary['trading_statistics']
        print(f"\nTrading Statistics:")
        print(f"Total Trades: {trading_stats['total_trades']}")
        print(f"Avg Processing Time: {trading_stats['avg_processing_time_ms']:.2f}ms")
        print(f"Final Position: {trading_stats['current_position']:.3f}")
        
        current_regime = summary['recent_regime_info']
        print(f"\nCurrent Market Regime: {current_regime['regime_interpretation']}")
        print(f"Confidence: {current_regime['confidence']:.2%}")
        
        print("\nRegime Performance:")
        print(summary['regime_performance'])

if __name__ == "__main__":
    run_online_hmm_trading_example()
```

---

## Best Practices

### 1. Model Initialization

#### Historical Data Requirements
- **Minimum**: 100 observations for stable initialization
- **Recommended**: 200+ observations (6-12 months of daily data)
- **Quality**: Clean, validated data with minimal gaps

#### Initialization Strategy
```python
# Step 1: Load sufficient historical data
historical_data = load_clean_historical_data(ticker, lookback_days=300)

# Step 2: Use batch HMM for initial parameters
batch_hmm = HiddenMarkovModel(n_states=3)
batch_hmm.fit(historical_data.returns, verbose=True)

# Step 3: Initialize online HMM with batch parameters
online_hmm = OnlineHMM(n_states=3)
online_hmm.initialize_from_batch_hmm(batch_hmm)

# Step 4: Validate initialization
initial_analysis = online_hmm.analyze_regimes(historical_data.returns[-100:])
print("Initialization validation:", initial_analysis['regime_statistics'])
```

### 2. Parameter Tuning

#### Systematic Parameter Search
```python
def optimize_online_hmm_parameters(validation_data, parameter_grid):
    """Optimize online HMM parameters using validation data"""
    best_score = -np.inf
    best_params = None
    
    for params in parameter_grid:
        config = OnlineHMMConfig(**params)
        
        # Test configuration
        hmm = OnlineHMM(n_states=3, online_config=config)
        hmm.fit(validation_data[:200])  # Initialize with first 200 obs
        
        # Test on remaining data
        log_likelihoods = []
        for obs in validation_data[200:]:
            hmm.add_observation(obs)
            ll = hmm.score([obs])  # Score current observation
            log_likelihoods.append(ll)
        
        avg_score = np.mean(log_likelihoods)
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
    
    return best_params, best_score

# Parameter grid for optimization
param_grid = [
    {'forgetting_factor': 0.96, 'adaptation_rate': 0.08, 'smoothing_weight': 0.7},
    {'forgetting_factor': 0.98, 'adaptation_rate': 0.05, 'smoothing_weight': 0.8},
    {'forgetting_factor': 0.99, 'adaptation_rate': 0.03, 'smoothing_weight': 0.9},
]

best_params, score = optimize_online_hmm_parameters(validation_returns, param_grid)
print(f"Best parameters: {best_params} (score: {score:.4f})")
```

### 3. Production Deployment

#### Error Handling and Monitoring
```python
class RobustOnlineHMM:
    """Production-ready online HMM with comprehensive error handling"""
    
    def __init__(self, config):
        self.hmm = OnlineHMM(config=config)
        self.error_count = 0
        self.max_errors = 10
        self.backup_hmm = None
        
    def safe_add_observation(self, observation):
        """Safely add observation with error handling"""
        try:
            # Validate observation
            if not np.isfinite(observation):
                raise ValueError(f"Invalid observation: {observation}")
            
            if abs(observation) > 0.5:  # 50% daily return threshold
                print(f"Warning: Extreme observation {observation:.4f}")
            
            # Process observation
            self.hmm.add_observation(observation)
            self.error_count = 0  # Reset error count on success
            
            return self.hmm.get_current_regime_info()
            
        except Exception as e:
            self.error_count += 1
            print(f"Error processing observation: {e}")
            
            if self.error_count >= self.max_errors:
                print("Max errors reached. Reinitializing model...")
                self.reinitialize_model()
            
            # Return previous regime info or default
            return self.get_safe_regime_info()
    
    def reinitialize_model(self):
        """Reinitialize model after errors"""
        try:
            # Use backup model if available
            if self.backup_hmm is not None:
                self.hmm = self.backup_hmm.copy()
                print("Restored from backup model")
            else:
                # Reinitialize with default parameters
                self.hmm = OnlineHMM(n_states=3)
                print("Reinitialized with default parameters")
            
            self.error_count = 0
            
        except Exception as e:
            print(f"Failed to reinitialize: {e}")
    
    def create_backup(self):
        """Create backup of current model"""
        self.backup_hmm = self.hmm.copy()
    
    def get_safe_regime_info(self):
        """Get regime info with fallback"""
        try:
            return self.hmm.get_current_regime_info()
        except:
            # Return safe default
            return {
                'current_state': 1,  # Default to sideways
                'confidence': 0.33,
                'regime_interpretation': 'Sideways Market (Default)',
                'state_probabilities': np.array([0.33, 0.34, 0.33])
            }
```

### 4. Performance Monitoring

#### Real-Time Performance Tracking
```python
class PerformanceMonitor:
    """Monitor online HMM performance in real-time"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.confidence_scores = deque(maxlen=window_size)
        self.regime_changes = deque(maxlen=window_size)
        self.last_regime = None
        
    def record_processing(self, processing_time, regime_info):
        """Record processing metrics"""
        self.processing_times.append(processing_time)
        self.confidence_scores.append(regime_info['confidence'])
        
        # Track regime changes
        current_regime = regime_info['current_state']
        regime_changed = (self.last_regime is not None and 
                         current_regime != self.last_regime)
        self.regime_changes.append(regime_changed)
        self.last_regime = current_regime
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if len(self.processing_times) == 0:
            return None
        
        return {
            'avg_processing_time_ms': np.mean(self.processing_times) * 1000,
            'max_processing_time_ms': np.max(self.processing_times) * 1000,
            'avg_confidence': np.mean(self.confidence_scores),
            'min_confidence': np.min(self.confidence_scores),
            'regime_change_rate': np.mean(self.regime_changes),
            'samples_count': len(self.processing_times)
        }
    
    def check_performance_alerts(self):
        """Check for performance issues"""
        stats = self.get_performance_stats()
        alerts = []
        
        if stats is None:
            return alerts
        
        # Processing time alerts
        if stats['avg_processing_time_ms'] > 10:
            alerts.append(f"High processing time: {stats['avg_processing_time_ms']:.2f}ms")
        
        if stats['max_processing_time_ms'] > 50:
            alerts.append(f"Maximum processing time exceeded: {stats['max_processing_time_ms']:.2f}ms")
        
        # Confidence alerts
        if stats['avg_confidence'] < 0.6:
            alerts.append(f"Low confidence: {stats['avg_confidence']:.2%}")
        
        # Regime stability alerts
        if stats['regime_change_rate'] > 0.1:  # More than 10% regime changes
            alerts.append(f"High regime instability: {stats['regime_change_rate']:.2%} change rate")
        
        return alerts
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Not Adapting to New Market Conditions

**Symptoms:**
- Regime predictions lag behind obvious market changes
- Low confidence scores persist
- Model seems "stuck" in one regime

**Solutions:**
```python
# Increase adaptation rate
config.adaptation_rate = 0.08  # from default 0.05

# Reduce smoothing weight for faster adaptation
config.smoothing_weight = 0.7  # from default 0.8

# Reduce forgetting factor for shorter memory
config.forgetting_factor = 0.96  # from default 0.98

# Enable change detection
config.enable_change_detection = True
config.change_detection_threshold = 2.5
```

#### 2. Excessive Regime Switching

**Symptoms:**
- Regime changes multiple times per day
- Very short regime durations
- High trading frequency

**Solutions:**
```python
# Increase smoothing for stability
config.smoothing_weight = 0.9

# Reduce adaptation rate
config.adaptation_rate = 0.03

# Increase minimum observations before updates
config.min_observations_for_update = 20

# Add regime persistence constraint
config.min_regime_duration = 5
```

#### 3. Memory Usage Growing Over Time

**Symptoms:**
- Gradually increasing memory consumption
- System slowdown over long periods
- Memory warnings or errors

**Solutions:**
```python
# Reduce rolling window size
config.rolling_window_size = 500  # from default 1000

# Enable periodic cleanup
def periodic_cleanup(online_hmm, frequency=1000):
    if online_hmm.observation_count % frequency == 0:
        online_hmm.cleanup_memory()
        print(f"Memory cleanup at observation {online_hmm.observation_count}")

# Monitor memory usage
import psutil

def check_memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    if memory_mb > 200:  # Alert if over 200MB
        print(f"High memory usage: {memory_mb:.1f}MB")
```

#### 4. Poor Performance on Volatile Data

**Symptoms:**
- Low confidence during high volatility periods
- Frequent false regime changes
- Poor regime detection accuracy

**Solutions:**
```python
# Use volatility-adjusted configuration
def create_volatility_adaptive_config(recent_volatility):
    if recent_volatility > 0.03:  # High volatility
        return OnlineHMMConfig(
            forgetting_factor=0.97,         # Moderate memory
            adaptation_rate=0.06,           # Moderate adaptation
            smoothing_weight=0.85,          # More smoothing
            change_detection_threshold=3.5  # Higher threshold
        )
    elif recent_volatility < 0.015:  # Low volatility
        return OnlineHMMConfig(
            forgetting_factor=0.99,         # Long memory
            adaptation_rate=0.03,           # Slow adaptation
            smoothing_weight=0.9,           # High smoothing
            change_detection_threshold=2.0  # Lower threshold
        )
    else:  # Normal volatility
        return OnlineHMMConfig()  # Default settings

# Example usage
rolling_volatility = np.std(recent_returns[-20:])
adaptive_config = create_volatility_adaptive_config(rolling_volatility)
```

---

This comprehensive Online HMM documentation provides everything needed to understand, implement, and deploy streaming HMM systems for real-time market regime detection. The combination of theoretical foundation, practical examples, and troubleshooting guidance makes it suitable for both academic understanding and production implementation.

---

*For additional information, see the [Mathematical Foundations](mathematical_foundations.md), [Trading Applications Guide](trading_applications.md), and [Configuration Guide](configuration_guide.md).*