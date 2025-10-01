# State Standardization in Hidden Regime

## Overview

The `StateStandardizer` is a post-processing utility that bridges the gap between raw HMM state numbers and economically meaningful regime labels. It transforms abstract mathematical states (0, 1, 2, ...) into interpretable financial regimes (Bear, Bull, Sideways, Crisis, Euphoric).

## When to Use StateStandardizer

### ✅ Production Financial Applications

**Use StateStandardizer when you need:**
- **Trading Systems**: Regime-based position sizing and strategy selection
- **Risk Management**: Regime-specific risk models and stress testing
- **Client Reporting**: Human-readable regime labels for dashboards and reports
- **Portfolio Management**: Regime-aware asset allocation
- **Research Communication**: Presenting findings to stakeholders

**Example**: A wealth management firm needs to explain market conditions to clients:
```python
# Instead of: "Currently in State 2 with 73% confidence"
# Provide: "Currently in Bull regime with 73% confidence"
```

### ✅ Standardization and Comparison

**Use StateStandardizer when you need:**
- **Model Consistency**: Same regime labels across different time periods
- **Cross-Asset Analysis**: Comparing regime patterns between assets
- **Backtesting**: Consistent regime definitions for historical analysis
- **Model Validation**: Economic validation of detected patterns

### ❌ Research and Exploration

**Skip StateStandardizer when you:**
- **Discover New Patterns**: Want to find regime structures without preconceptions
- **Academic Research**: Studying pure statistical properties of markets
- **Model Development**: Prototyping and experimenting with different approaches
- **Performance Critical**: Need maximum computational speed

### ❌ Non-Financial Applications

**Skip StateStandardizer for:**
- **Speech Recognition**: States represent phonemes, not market conditions
- **Medical Applications**: States represent symptoms or disease stages
- **Manufacturing**: States represent machine conditions or quality levels
- **General Sequence Modeling**: Domain-agnostic pattern recognition

## How It Works

### Core Algorithm

1. **Extract Emission Parameters**: Get `[mean_return, volatility]` for each state
2. **Sort by Performance**: Rank states from lowest to highest mean return
3. **Map to Economic Labels**: Assign regime names based on ranking

```python
# Example with 5 states
emission_params = np.array([
    [-0.002, 0.025],  # State 0: Bear
    [-0.008, 0.035],  # State 1: Crisis (lowest return)
    [0.0001, 0.015],  # State 2: Sideways  
    [0.001, 0.018],   # State 3: Bull
    [0.005, 0.022]    # State 4: Euphoric (highest return)
])

# Sorting by mean return: [1, 0, 2, 3, 4]
# Result mapping: {1: 'Crisis', 0: 'Bear', 2: 'Sideways', 3: 'Bull', 4: 'Euphoric'}
```

### Regime Configurations

| States | Configuration | Use Case |
|--------|---------------|----------|
| 3-state | `['Bear', 'Sideways', 'Bull']` | Simple trend identification |
| 4-state | `['Crisis', 'Bear', 'Sideways', 'Bull']` | Include extreme negative events |
| 5-state | `['Crisis', 'Bear', 'Sideways', 'Bull', 'Euphoric']` | Full regime spectrum |

## Usage Examples

### Basic Usage

```python
from hidden_regime.models.state_standardizer import StateStandardizer
import numpy as np

# Create standardizer
standardizer = StateStandardizer(regime_type='3_state')

# Get emission parameters from trained HMM
emission_params = model.emission_params_  # Shape: [n_states, 2]

# Create state mapping
state_mapping = standardizer.standardize_states(emission_params)
print(state_mapping)  # {0: 'Bear', 1: 'Sideways', 2: 'Bull'}

# Convert state sequence to regime names
states = np.array([2, 2, 1, 0, 0, 1, 2])
regimes = [state_mapping[s] for s in states]
print(regimes)  # ['Bull', 'Bull', 'Sideways', 'Bear', 'Bear', 'Sideways', 'Bull']
```

### Complete Workflow

```python
from hidden_regime.models.base_hmm import HiddenMarkovModel
from hidden_regime.models.config import HMMConfig
from hidden_regime.models.state_standardizer import StateStandardizer
from hidden_regime.data.loader import DataLoader

# 1. Load and prepare data
loader = DataLoader()
data = loader.load('AAPL', '2020-01-01', '2023-12-31')
returns = data['log_return'].values

# 2. Train HMM model
config = HMMConfig(n_states=3, max_iterations=100)
model = HiddenMarkovModel(config)
model.fit(returns)

# 3. Get predictions
states = model.predict(returns)

# 4. Create standardizer and convert to regime names
standardizer = StateStandardizer(regime_type='3_state') 
state_mapping = standardizer.standardize_states(model.emission_params_)
regime_names = [state_mapping[s] for s in states]

# 5. Analyze results
import pandas as pd
results = pd.DataFrame({
    'date': data['date'],
    'return': returns,
    'state': states,
    'regime': regime_names
})

print(results.groupby('regime')['return'].agg(['count', 'mean', 'std']))
```

### Integration with HMM Training

```python
# Option 1: Manual standardization (more control)
config = HMMConfig(n_states=3)
model = HiddenMarkovModel(config)
model.fit(returns)

standardizer = StateStandardizer(regime_type='3_state')
state_mapping = standardizer.standardize_states(model.emission_params_)

# Option 2: Built-in standardization (automatic)
config = HMMConfig.for_standardized_regimes(regime_type='3_state')
model = HiddenMarkovModel(config)
model.fit(returns)

# Access the internal state mapping (if available)
if hasattr(model, '_state_mapping'):
    state_mapping = model._state_mapping
```

## Advanced Features

### Economic Validation

```python
# Validate that detected regimes make economic sense
confidence = standardizer.validate_interpretation(
    states=states,
    returns=returns,
    emission_params=model.emission_params_
)
print(f"Economic validation confidence: {confidence:.2%}")
```

### Regime Characteristics Analysis

```python
# Get detailed regime interpretations
for state_idx in range(model.n_states):
    interpretation = standardizer.get_regime_interpretation_enhanced(
        state_idx, model.emission_params_, model.config
    )
    print(f"State {state_idx}: {interpretation}")
```

### Automatic State Selection

```python
# Let StateStandardizer choose optimal number of states
standardizer = StateStandardizer(regime_type='auto')
best_config, score, details = standardizer.select_optimal_configuration(
    returns, max_states=6
)
print(f"Optimal configuration: {best_config.n_states} states (score: {score:.3f})")
```

## Performance Considerations

### Computational Overhead

```python
import time

# Raw states (fast)
start = time.time()
states = model.predict(returns)
raw_time = time.time() - start

# With standardization (slower)
start = time.time()
standardizer = StateStandardizer(regime_type='3_state')
state_mapping = standardizer.standardize_states(model.emission_params_)
regime_names = [state_mapping[s] for s in states]
standardized_time = time.time() - start

print(f"Raw prediction: {raw_time:.4f}s")
print(f"With standardization: {standardized_time:.4f}s")
print(f"Overhead: {(standardized_time/raw_time - 1)*100:.1f}%")
```

### Memory Usage

StateStandardizer adds minimal memory overhead:
- State mapping dictionary: ~1KB for typical configurations
- Configuration objects: ~2-5KB
- No additional data storage required

## Common Patterns

### Batch Processing

```python
def process_multiple_assets(tickers, standardizer):
    """Process multiple assets with consistent regime labeling."""
    results = {}
    
    for ticker in tickers:
        data = loader.load(ticker, '2020-01-01', '2023-12-31')
        model = HiddenMarkovModel(config)
        model.fit(data['log_return'].values)
        
        states = model.predict(data['log_return'].values)
        state_mapping = standardizer.standardize_states(model.emission_params_)
        regime_names = [state_mapping[s] for s in states]
        
        results[ticker] = {
            'dates': data['date'],
            'regimes': regime_names,
            'states': states
        }
    
    return results
```

### Real-time Application

```python
class RealtimeRegimeDetector:
    def __init__(self, regime_type='3_state'):
        self.standardizer = StateStandardizer(regime_type)
        self.model = None
        self.state_mapping = None
    
    def update_model(self, new_data):
        """Retrain model with new data."""
        self.model.fit(new_data)
        self.state_mapping = self.standardizer.standardize_states(
            self.model.emission_params_
        )
    
    def get_current_regime(self, recent_returns):
        """Get current regime name."""
        state = self.model.predict_next_state(recent_returns)
        return self.state_mapping[state]
```

## Best Practices

### 1. Choose Appropriate Regime Type
- **3-state**: Simple trend following strategies
- **4-state**: Include crisis detection
- **5-state**: Detailed regime analysis, research

### 2. Validate Economic Meaning
```python
# Always check that regimes make economic sense
confidence = standardizer.validate_interpretation(states, returns)
if confidence < 0.7:
    warnings.warn("Low economic validation confidence")
```

### 3. Handle Edge Cases
```python
# Check for state mapping availability
if hasattr(model, '_state_mapping') and model._state_mapping:
    # Use built-in mapping
    regime_names = [model._state_mapping[s] for s in states]
else:
    # Create manual mapping
    standardizer = StateStandardizer(regime_type='3_state')
    state_mapping = standardizer.standardize_states(model.emission_params_)
    regime_names = [state_mapping[s] for s in states]
```

### 4. Document Regime Definitions
```python
# Always document which regime configuration was used
metadata = {
    'regime_type': '3_state',
    'regime_names': ['Bear', 'Sideways', 'Bull'],
    'creation_date': '2024-01-15',
    'validation_confidence': 0.82
}
```

## Conclusion

StateStandardizer bridges the gap between mathematical models and financial intuition. Use it when human interpretability matters, skip it when pure mathematical modeling is sufficient. The key is matching the tool to your specific use case and requirements.