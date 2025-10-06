# Configuration and Tuning Guide

*Complete guide to configuring and optimizing HMM models for different market conditions and use cases*

---

## Table of Contents

1. [Overview](#overview)
2. [HMM Configuration](#hmm-configuration)
3. [Online Learning Configuration](#online-learning-configuration)
4. [Data-Specific Settings](#data-specific-settings)
5. [Performance Tuning](#performance-tuning)
6. [Configuration Validation](#configuration-validation)
7. [Environment-Specific Configurations](#environment-specific-configurations)
8. [Automated Parameter Optimization](#automated-parameter-optimization)
9. [Best Practices](#best-practices)
10. [Troubleshooting Configuration Issues](#troubleshooting-configuration-issues)

---

## Overview

Proper configuration is critical for HMM model performance. This guide provides comprehensive guidance on tuning parameters for different market conditions, data frequencies, and use cases.

### Configuration Hierarchy

```
Configuration System
├── HMMConfig (Base model parameters)
│   ├── Model Structure (n_states, initialization)
│   ├── Training Parameters (iterations, tolerance)
│   └── Numerical Stability (regularization, bounds)
│
├── OnlineHMMConfig (Online learning parameters)
│   ├── Memory Management (forgetting_factor, window_size)
│   ├── Adaptation Control (adaptation_rate, smoothing)
│   └── Change Detection (thresholds, monitoring)
│
├── DataConfig (Data processing parameters)
│   ├── Quality Control (validation, outlier_detection)
│   ├── Preprocessing (return_calculation, normalization)
│   └── Feature Engineering (technical_indicators, lag_features)
│
└── ValidationConfig (Model validation parameters)
    ├── Cross-Validation (folds, walk_forward)
    ├── Performance Metrics (sharpe_threshold, stability_threshold)
    └── Model Selection (criteria, comparison_methods)
```

### Key Configuration Principles

1. **Start Simple**: Begin with default configurations and adjust incrementally
2. **Data-Driven**: Let data characteristics guide configuration choices
3. **Use Case Specific**: Tailor configurations to specific trading objectives
4. **Validate Changes**: Always validate configuration changes with proper testing
5. **Monitor Performance**: Continuously monitor and adjust in production

---

## HMM Configuration

### Basic HMM Parameters

The `HMMConfig` class controls core model behavior:

```python
from hidden_regime.models import HMMConfig

# Default configuration
default_config = HMMConfig()

# Custom configuration
custom_config = HMMConfig(
    n_states=3,                    # Number of market regimes
    max_iterations=100,            # Maximum EM iterations
    tolerance=1e-6,                # Convergence tolerance
    initialization_method='kmeans', # Parameter initialization
    random_seed=42,                # Reproducible results
    regularization=1e-6,           # Numerical regularization
    min_variance=1e-8,             # Minimum allowed variance
    early_stopping=True,           # Stop if converged early
    verbose_training=False         # Training output control
)
```

### Number of States (n_states)

**Guidelines for State Selection:**

#### 2 States (Bull/Bear)
```python
config_2_states = HMMConfig(
    n_states=2,
    max_iterations=150,  # May need more iterations
    tolerance=1e-7       # Tighter tolerance for stability
)

# Best for:
# - Highly trending markets
# - Simple trend-following strategies
# - Limited historical data (<100 observations)
# - High-frequency data with clear directional bias
```

#### 3 States (Bull/Sideways/Bear)
```python
config_3_states = HMMConfig(
    n_states=3,
    max_iterations=100,
    tolerance=1e-6,
    initialization_method='kmeans'  # Recommended for 3 states
)

# Best for:
# - Most general market conditions
# - Daily trading strategies
# - Balanced mix of trending and ranging periods
# - Standard use case (recommended default)
```

#### 4+ States (Complex Regime Structure)
```python
config_4_states = HMMConfig(
    n_states=4,
    max_iterations=200,           # More states need more iterations
    tolerance=1e-7,               # Tighter convergence
    regularization=1e-5,          # Higher regularization
    initialization_method='kmeans', # Critical for 4+ states
    early_stopping=True           # Prevent overtraining
)

# Best for:
# - Large datasets (1000+ observations)
# - Complex market dynamics
# - Multi-asset portfolio analysis
# - Long-term historical analysis
# 
# Warning: Risk of overfitting with insufficient data
```

### Initialization Methods

#### K-Means Initialization (Recommended)
```python
config = HMMConfig(
    initialization_method='kmeans',
    random_seed=42,  # Ensure reproducible clustering
    n_init_trials=10 # Multiple clustering attempts
)

# Advantages:
# - Data-driven parameter initialization
# - Better convergence properties
# - More stable results
# - Faster training (fewer iterations needed)

# Use when:
# - You have sufficient data (100+ observations)
# - Data has clear regime structure
# - Want consistent, reproducible results
```

#### Random Initialization
```python
config = HMMConfig(
    initialization_method='random',
    random_seed=42,
    n_init_trials=5,    # Try multiple random starts
    regularization=1e-5  # Higher regularization for stability
)

# Advantages:
# - Faster initialization
# - Good for exploration of parameter space
# - May find different local optima

# Use when:
# - Data doesn't cluster well
# - Exploratory analysis
# - Ensemble approaches (multiple random starts)
```

#### Custom Initialization
```python
# Define custom initial parameters
custom_initial_params = {
    'initial_probs': np.array([0.33, 0.34, 0.33]),
    'transition_matrix': np.array([
        [0.8, 0.15, 0.05],  # Bear regime persistence
        [0.1, 0.7, 0.2],    # Sideways transitions
        [0.05, 0.15, 0.8]   # Bull regime persistence
    ]),
    'emission_params': [
        {'mean': -0.01, 'std': 0.025},  # Bear: negative, high vol
        {'mean': 0.0, 'std': 0.015},    # Sideways: neutral, low vol
        {'mean': 0.01, 'std': 0.020}    # Bull: positive, moderate vol
    ]
}

config = HMMConfig(
    initialization_method='custom',
    custom_initial_params=custom_initial_params
)

# Use when:
# - You have domain knowledge about market regimes
# - Want to incorporate prior beliefs
# - Need specific regime characteristics
```

### Training Parameters

#### Convergence Control
```python
# Conservative (High Quality, Slower)
conservative_config = HMMConfig(
    max_iterations=200,
    tolerance=1e-8,
    early_stopping=False,  # Always run full iterations
    regularization=1e-7
)

# Balanced (Recommended)
balanced_config = HMMConfig(
    max_iterations=100,
    tolerance=1e-6,
    early_stopping=True,
    regularization=1e-6
)

# Fast (Lower Quality, Faster)
fast_config = HMMConfig(
    max_iterations=50,
    tolerance=1e-5,
    early_stopping=True,
    regularization=1e-5
)
```

#### Numerical Stability
```python
# High Stability Configuration
stable_config = HMMConfig(
    regularization=1e-5,      # Higher regularization
    min_variance=1e-7,        # Prevent degenerate variances
    log_likelihood_threshold=-1e10,  # Catch extreme values
    max_condition_number=1e12 # Prevent ill-conditioned matrices
)

# Use for:
# - Noisy data
# - Small datasets
# - Numerical precision issues
```

---

## Online Learning Configuration

### Core Online Learning Parameters

The `OnlineHMMConfig` class controls incremental learning behavior:

```python
from hidden_regime.models import OnlineHMMConfig

# Comprehensive configuration
online_config = OnlineHMMConfig(
    # Memory and Forgetting
    forgetting_factor=0.98,           # How much to remember
    adaptation_rate=0.05,             # How fast to adapt
    
    # Stability Mechanisms
    parameter_smoothing=True,         # Smooth parameter changes
    smoothing_weight=0.8,            # Weight for smoothing
    min_observations_for_update=10,   # Minimum data before updates
    
    # Memory Management
    rolling_window_size=1000,        # Observations to keep in memory
    sufficient_stats_decay=0.99,     # Decay rate for statistics
    
    # Change Detection
    enable_change_detection=True,     # Monitor structural breaks
    change_detection_threshold=3.0,   # Standard deviations
    change_detection_window=50,      # Window for monitoring
    
    # Convergence
    convergence_tolerance=1e-4,       # Parameter convergence
    max_adaptation_iterations=5       # Max iterations per update
)
```

### Forgetting Factor Tuning

The forgetting factor (λ) controls how quickly old information is discounted:

#### Conservative Memory (λ = 0.99)
```python
conservative_online = OnlineHMMConfig(
    forgetting_factor=0.99,    # Very long memory (~100 days)
    adaptation_rate=0.02,      # Slow adaptation
    smoothing_weight=0.9       # High smoothing
)

# Characteristics:
# - Effective sample size: ~100 observations
# - Very stable parameters
# - Slow adaptation to regime changes
# - Good for: Long-term analysis, stable markets
```

#### Balanced Memory (λ = 0.98)
```python
balanced_online = OnlineHMMConfig(
    forgetting_factor=0.98,    # Medium memory (~50 days)
    adaptation_rate=0.05,      # Moderate adaptation
    smoothing_weight=0.8       # Balanced smoothing
)

# Characteristics:
# - Effective sample size: ~50 observations
# - Good stability/adaptation balance
# - Recommended for most applications
# - Good for: Daily trading, general use
```

#### Aggressive Memory (λ = 0.95)
```python
aggressive_online = OnlineHMMConfig(
    forgetting_factor=0.95,    # Short memory (~20 days)
    adaptation_rate=0.1,       # Fast adaptation
    smoothing_weight=0.6       # Lower smoothing
)

# Characteristics:
# - Effective sample size: ~20 observations
# - Fast adaptation to changes
# - Higher parameter volatility
# - Good for: High-frequency trading, volatile markets
```

### Market Condition Specific Configurations

#### Crisis/High Volatility Markets
```python
crisis_config = OnlineHMMConfig(
    forgetting_factor=0.96,           # Shorter memory for rapid changes
    adaptation_rate=0.08,             # Faster adaptation
    parameter_smoothing=True,         # Keep smoothing for stability
    smoothing_weight=0.85,           # Higher smoothing to counter volatility
    
    # Enhanced change detection
    enable_change_detection=True,
    change_detection_threshold=2.5,   # Lower threshold (more sensitive)
    change_detection_window=30,      # Shorter window
    
    # More frequent updates
    min_observations_for_update=5
)
```

#### Trending Markets
```python
trending_config = OnlineHMMConfig(
    forgetting_factor=0.99,           # Longer memory for trends
    adaptation_rate=0.03,             # Slower adaptation
    parameter_smoothing=True,
    smoothing_weight=0.9,            # High smoothing for trend stability
    
    # Conservative change detection
    enable_change_detection=True,
    change_detection_threshold=3.5,   # Higher threshold
    change_detection_window=75,      # Longer window
    
    min_observations_for_update=15   # Less frequent updates
)
```

#### Range-Bound/Sideways Markets
```python
sideways_config = OnlineHMMConfig(
    forgetting_factor=0.97,           # Moderate memory
    adaptation_rate=0.06,             # Moderate adaptation
    parameter_smoothing=True,
    smoothing_weight=0.75,           # Moderate smoothing
    
    # Standard change detection
    enable_change_detection=True,
    change_detection_threshold=3.0,
    change_detection_window=50,
    
    min_observations_for_update=8
)
```

### Frequency-Specific Configurations

#### High-Frequency (Minutes/Seconds)
```python
hf_config = OnlineHMMConfig(
    # Fast adaptation for rapid changes
    forgetting_factor=0.95,
    adaptation_rate=0.12,
    
    # Minimal smoothing for responsiveness
    parameter_smoothing=True,
    smoothing_weight=0.6,
    
    # Efficient memory management
    rolling_window_size=500,          # Smaller window
    min_observations_for_update=3,    # Frequent updates
    
    # Aggressive change detection
    enable_change_detection=True,
    change_detection_threshold=2.0,
    change_detection_window=20
)
```

#### Daily Trading
```python
daily_config = OnlineHMMConfig(
    # Balanced parameters (recommended default)
    forgetting_factor=0.98,
    adaptation_rate=0.05,
    
    parameter_smoothing=True,
    smoothing_weight=0.8,
    
    rolling_window_size=1000,
    min_observations_for_update=10,
    
    enable_change_detection=True,
    change_detection_threshold=3.0,
    change_detection_window=50
)
```

#### Weekly/Monthly Analysis
```python
weekly_config = OnlineHMMConfig(
    # Conservative parameters for long-term analysis
    forgetting_factor=0.995,          # Very long memory
    adaptation_rate=0.02,             # Very slow adaptation
    
    parameter_smoothing=True,
    smoothing_weight=0.95,           # Maximum smoothing
    
    rolling_window_size=2000,        # Large window
    min_observations_for_update=25,   # Infrequent updates
    
    enable_change_detection=True,
    change_detection_threshold=4.0,   # Conservative threshold
    change_detection_window=100      # Long monitoring window
)
```

---

## Data-Specific Settings

### Data Quality and Preprocessing

#### High-Quality Data Configuration
```python
high_quality_config = HMMConfig(
    # Can use tighter tolerances with good data
    tolerance=1e-7,
    regularization=1e-7,
    min_variance=1e-9,
    
    # Fewer iterations needed
    max_iterations=75,
    early_stopping=True
)

# Pair with:
online_config = OnlineHMMConfig(
    # Less smoothing needed with good data
    parameter_smoothing=True,
    smoothing_weight=0.7,
    
    # Can adapt more aggressively
    adaptation_rate=0.08
)
```

#### Noisy/Low-Quality Data Configuration
```python
noisy_data_config = HMMConfig(
    # Higher regularization for stability
    regularization=1e-4,
    min_variance=1e-6,
    
    # More iterations may be needed
    max_iterations=150,
    tolerance=1e-5,
    
    # Prevent extreme values
    log_likelihood_threshold=-1e8
)

# Pair with:
noisy_online_config = OnlineHMMConfig(
    # More smoothing for noise reduction
    parameter_smoothing=True,
    smoothing_weight=0.9,
    
    # Slower adaptation to avoid noise
    adaptation_rate=0.03,
    forgetting_factor=0.99,
    
    # More observations before updates
    min_observations_for_update=20
)
```

### Asset-Specific Configurations

#### Equity Markets
```python
equity_config = HMMConfig(
    n_states=3,                      # Standard bull/sideways/bear
    initialization_method='kmeans',
    max_iterations=100,
    tolerance=1e-6
)

equity_online_config = OnlineHMMConfig(
    forgetting_factor=0.98,          # ~50 day memory (quarterly earnings impact)
    adaptation_rate=0.05,
    parameter_smoothing=True,
    smoothing_weight=0.8
)
```

#### Cryptocurrency Markets
```python
crypto_config = HMMConfig(
    n_states=4,                      # More volatile, need extra state
    initialization_method='kmeans',
    max_iterations=150,              # May need more iterations
    regularization=1e-5,             # Higher for stability
    tolerance=1e-6
)

crypto_online_config = OnlineHMMConfig(
    forgetting_factor=0.96,          # Shorter memory for rapid changes
    adaptation_rate=0.08,            # Faster adaptation
    parameter_smoothing=True,
    smoothing_weight=0.85,          # High smoothing for volatility
    
    # Aggressive change detection
    enable_change_detection=True,
    change_detection_threshold=2.5,
    change_detection_window=30
)
```

#### Foreign Exchange (Forex)
```python
forex_config = HMMConfig(
    n_states=3,
    initialization_method='kmeans',
    max_iterations=100,
    tolerance=1e-6,
    regularization=1e-6
)

forex_online_config = OnlineHMMConfig(
    forgetting_factor=0.99,          # Longer memory for central bank policies
    adaptation_rate=0.04,            # Moderate adaptation
    parameter_smoothing=True,
    smoothing_weight=0.8,
    
    # Standard change detection
    enable_change_detection=True,
    change_detection_threshold=3.0,
    change_detection_window=60      # Longer for policy impacts
)
```

#### Commodities
```python
commodities_config = HMMConfig(
    n_states=3,
    initialization_method='kmeans',
    max_iterations=120,              # May need more for seasonality
    tolerance=1e-6,
    regularization=1e-6
)

commodities_online_config = OnlineHMMConfig(
    forgetting_factor=0.985,         # Medium-long memory for seasonal patterns
    adaptation_rate=0.04,
    parameter_smoothing=True,
    smoothing_weight=0.82,
    
    # Moderate change detection
    enable_change_detection=True,
    change_detection_threshold=3.2,
    change_detection_window=60
)
```

---

## Performance Tuning

### Speed Optimization

#### High-Throughput Configuration
```python
fast_config = HMMConfig(
    n_states=3,                      # Keep states reasonable
    max_iterations=50,               # Reduce iterations
    tolerance=1e-5,                  # Looser tolerance
    early_stopping=True,             # Stop early when possible
    initialization_method='random'   # Faster than k-means
)

fast_online_config = OnlineHMMConfig(
    # Minimal processing per observation
    rolling_window_size=500,         # Smaller memory footprint
    min_observations_for_update=5,   # More frequent but lighter updates
    max_adaptation_iterations=3,     # Fewer iterations per update
    
    # Disable expensive features
    enable_change_detection=False,   # Skip change detection
    parameter_smoothing=True,        # Keep for stability
    smoothing_weight=0.7
)
```

#### Memory-Efficient Configuration
```python
memory_config = OnlineHMMConfig(
    rolling_window_size=300,         # Smaller window
    sufficient_stats_decay=0.98,     # Faster decay = less memory
    
    # Process in smaller batches
    min_observations_for_update=10,
    max_adaptation_iterations=3,
    
    # Simplified processing
    enable_change_detection=False
)

# Memory monitoring
def monitor_memory_usage(online_hmm):
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    if memory_mb > 100:  # 100MB threshold
        # Trigger memory cleanup
        online_hmm.cleanup_memory()
        print(f"Memory cleaned up: {memory_mb:.1f}MB")
```

### Accuracy Optimization

#### High-Accuracy Configuration
```python
accurate_config = HMMConfig(
    n_states=3,
    max_iterations=200,              # More iterations
    tolerance=1e-8,                  # Tighter convergence
    initialization_method='kmeans',  # Better initialization
    n_init_trials=10,               # Multiple attempts
    
    # Higher numerical precision
    regularization=1e-7,
    min_variance=1e-9
)

accurate_online_config = OnlineHMMConfig(
    # Conservative online learning
    forgetting_factor=0.99,          # Long memory
    adaptation_rate=0.03,            # Slow adaptation
    
    # Maximum stability
    parameter_smoothing=True,
    smoothing_weight=0.9,
    
    # More data before updates
    min_observations_for_update=20,
    
    # Enhanced monitoring
    enable_change_detection=True,
    change_detection_threshold=3.5,
    change_detection_window=75
)
```

#### Ensemble Configuration
```python
def create_ensemble_configs(base_config, n_models=5):
    """Create ensemble of slightly different configurations"""
    ensemble_configs = []
    
    for i in range(n_models):
        # Vary key parameters slightly
        config = OnlineHMMConfig(
            forgetting_factor=base_config.forgetting_factor + (i-2)*0.005,
            adaptation_rate=base_config.adaptation_rate + (i-2)*0.01,
            smoothing_weight=base_config.smoothing_weight + (i-2)*0.02,
            random_seed=base_config.random_seed + i
        )
        ensemble_configs.append(config)
    
    return ensemble_configs

# Usage
base_config = OnlineHMMConfig()
ensemble_configs = create_ensemble_configs(base_config)

# Train ensemble of models
ensemble_models = []
for config in ensemble_configs:
    model = OnlineHMM(n_states=3, online_config=config)
    model.fit(training_data)
    ensemble_models.append(model)

# Combine predictions (example)
def ensemble_prediction(models, new_observation):
    predictions = []
    for model in models:
        model.add_observation(new_observation)
        regime_info = model.get_current_regime_info()
        predictions.append(regime_info['state_probabilities'])
    
    # Average probabilities
    avg_probabilities = np.mean(predictions, axis=0)
    return {
        'state_probabilities': avg_probabilities,
        'current_state': np.argmax(avg_probabilities),
        'confidence': np.max(avg_probabilities)
    }
```

---

## Configuration Validation

### Automated Configuration Testing

```python
class ConfigurationValidator:
    """Validate HMM configurations before deployment"""
    
    def __init__(self, validation_data):
        self.validation_data = validation_data
        
    def validate_config(self, hmm_config, online_config=None):
        """Comprehensive configuration validation"""
        
        results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Test basic HMM configuration
        hmm_results = self.test_hmm_config(hmm_config)
        results.update(hmm_results)
        
        # Test online configuration if provided
        if online_config:
            online_results = self.test_online_config(online_config)
            results.update(online_results)
        
        # Integration tests
        if results['is_valid']:
            integration_results = self.test_integration(hmm_config, online_config)
            results.update(integration_results)
        
        return results
    
    def test_hmm_config(self, config):
        """Test HMM configuration"""
        results = {'warnings': [], 'errors': []}
        
        try:
            # Test model training
            hmm = HiddenMarkovModel(config=config)
            hmm.fit(self.validation_data, verbose=False)
            
            # Validate results
            if not hmm.is_fitted:
                results['errors'].append("Model failed to fit data")
            
            # Check for convergence
            if hasattr(hmm, 'training_history_'):
                final_ll = hmm.training_history_['final_log_likelihood']
                if np.isnan(final_ll) or np.isinf(final_ll):
                    results['errors'].append("Invalid final log-likelihood")
            
            # Parameter validation
            if hmm.is_fitted:
                # Check transition matrix
                if not np.allclose(hmm.transition_matrix_.sum(axis=1), 1.0):
                    results['errors'].append("Invalid transition matrix")
                
                # Check emission parameters
                for i, (mean, std) in enumerate(hmm.emission_params_):
                    if std <= 0:
                        results['errors'].append(f"Invalid variance for state {i}")
                    if abs(mean) > 0.5:  # 50% daily return threshold
                        results['warnings'].append(f"Extreme mean return for state {i}: {mean:.4f}")
        
        except Exception as e:
            results['errors'].append(f"Training failed: {str(e)}")
        
        results['is_valid'] = len(results['errors']) == 0
        return results
    
    def test_online_config(self, online_config):
        """Test online configuration"""
        results = {'warnings': [], 'errors': []}
        
        # Parameter range validation
        if not (0.9 <= online_config.forgetting_factor <= 0.999):
            results['errors'].append(f"Invalid forgetting_factor: {online_config.forgetting_factor}")
        
        if not (0.001 <= online_config.adaptation_rate <= 0.2):
            results['errors'].append(f"Invalid adaptation_rate: {online_config.adaptation_rate}")
        
        if online_config.rolling_window_size < 50:
            results['warnings'].append("Very small rolling_window_size may cause instability")
        
        # Consistency checks
        if online_config.adaptation_rate > 0.1 and online_config.smoothing_weight > 0.9:
            results['warnings'].append("High adaptation rate with high smoothing may conflict")
        
        if online_config.forgetting_factor < 0.95 and online_config.min_observations_for_update > 20:
            results['warnings'].append("Short memory with infrequent updates may miss changes")
        
        results['is_valid'] = len(results['errors']) == 0
        return results
    
    def test_integration(self, hmm_config, online_config):
        """Test integrated configuration"""
        results = {'warnings': [], 'errors': [], 'performance_metrics': {}}
        
        try:
            # Create and test online HMM
            online_hmm = OnlineHMM(
                n_states=hmm_config.n_states,
                config=hmm_config,
                online_config=online_config
            )
            
            # Initialize with first part of data
            init_size = len(self.validation_data) // 2
            online_hmm.fit(self.validation_data[:init_size])
            
            # Test online updates
            processing_times = []
            confidences = []
            
            for obs in self.validation_data[init_size:init_size+50]:  # Test 50 observations
                start_time = time.time()
                online_hmm.add_observation(obs)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                regime_info = online_hmm.get_current_regime_info()
                confidences.append(regime_info['confidence'])
            
            # Performance metrics
            avg_processing_time = np.mean(processing_times)
            avg_confidence = np.mean(confidences)
            
            results['performance_metrics'] = {
                'avg_processing_time_ms': avg_processing_time * 1000,
                'avg_confidence': avg_confidence,
                'max_processing_time_ms': np.max(processing_times) * 1000
            }
            
            # Performance warnings
            if avg_processing_time > 0.01:  # 10ms threshold
                results['warnings'].append(f"Slow processing: {avg_processing_time*1000:.1f}ms average")
            
            if avg_confidence < 0.6:
                results['warnings'].append(f"Low average confidence: {avg_confidence:.2%}")
                
        except Exception as e:
            results['errors'].append(f"Integration test failed: {str(e)}")
        
        results['is_valid'] = len(results['errors']) == 0
        return results

# Usage example
def validate_configuration_example():
    """Example of configuration validation"""
    
    # Load validation data
    validation_data = np.random.normal(0.001, 0.02, 500)  # Replace with real data
    
    validator = ConfigurationValidator(validation_data)
    
    # Test configuration
    hmm_config = HMMConfig(n_states=3, initialization_method='kmeans')
    online_config = OnlineHMMConfig(forgetting_factor=0.98, adaptation_rate=0.05)
    
    results = validator.validate_config(hmm_config, online_config)
    
    print("Configuration Validation Results:")
    print(f"Valid: {results['is_valid']}")
    
    if results['errors']:
        print("Errors:")
        for error in results['errors']:
            print(f"   {error}")
    
    if results['warnings']:
        print("Warnings:")
        for warning in results['warnings']:
            print(f"  [WARNING]  {warning}")
    
    if 'performance_metrics' in results:
        perf = results['performance_metrics']
        print("Performance:")
        print(f"  Processing time: {perf['avg_processing_time_ms']:.2f}ms")
        print(f"  Confidence: {perf['avg_confidence']:.2%}")

if __name__ == "__main__":
    validate_configuration_example()
```

---

## Environment-Specific Configurations

### Development Environment
```python
dev_config = HMMConfig(
    n_states=3,
    max_iterations=50,               # Faster for development
    tolerance=1e-5,                  # Looser for speed
    verbose_training=True,           # Want to see progress
    random_seed=42                   # Reproducible results
)

dev_online_config = OnlineHMMConfig(
    rolling_window_size=200,         # Smaller for faster testing
    enable_change_detection=False,   # Disable for simplicity
    min_observations_for_update=5    # More frequent updates for testing
)
```

### Testing Environment
```python
test_config = HMMConfig(
    n_states=3,
    max_iterations=25,               # Very fast for unit tests
    tolerance=1e-4,
    early_stopping=True,
    random_seed=42,                  # Must be deterministic
    verbose_training=False           # Quiet for automated tests
)

test_online_config = OnlineHMMConfig(
    rolling_window_size=100,         # Minimal for speed
    forgetting_factor=0.95,          # Fast convergence
    adaptation_rate=0.1,
    parameter_smoothing=False,       # Disable for predictability
    enable_change_detection=False    # Disable for consistency
)
```

### Staging Environment
```python
staging_config = HMMConfig(
    n_states=3,
    max_iterations=75,               # Moderate thoroughness
    tolerance=1e-6,
    initialization_method='kmeans',
    random_seed=42,
    verbose_training=False
)

staging_online_config = OnlineHMMConfig(
    # Production-like but slightly faster
    forgetting_factor=0.98,
    adaptation_rate=0.06,            # Slightly faster adaptation
    rolling_window_size=750,         # Smaller than production
    parameter_smoothing=True,
    smoothing_weight=0.8,
    enable_change_detection=True
)
```

### Production Environment
```python
production_config = HMMConfig(
    n_states=3,
    max_iterations=100,              # Full thoroughness
    tolerance=1e-6,
    initialization_method='kmeans',
    n_init_trials=5,                # Multiple attempts for stability
    early_stopping=True,
    verbose_training=False,          # Quiet for production
    
    # High stability
    regularization=1e-6,
    min_variance=1e-8,
    log_likelihood_threshold=-1e10
)

production_online_config = OnlineHMMConfig(
    # Optimized for stability and performance
    forgetting_factor=0.98,
    adaptation_rate=0.05,
    
    parameter_smoothing=True,
    smoothing_weight=0.8,
    min_observations_for_update=10,
    
    rolling_window_size=1000,
    sufficient_stats_decay=0.99,
    
    # Full monitoring
    enable_change_detection=True,
    change_detection_threshold=3.0,
    change_detection_window=50,
    
    convergence_tolerance=1e-4,
    max_adaptation_iterations=5
)
```

---

## Automated Parameter Optimization

### Grid Search Optimization
```python
class HMMParameterOptimizer:
    """Automated parameter optimization for HMM configurations"""
    
    def __init__(self, training_data, validation_data):
        self.training_data = training_data
        self.validation_data = validation_data
        
    def optimize_hmm_config(self, param_grid, metric='sharpe_ratio'):
        """Optimize HMM configuration parameters"""
        
        best_score = -np.inf
        best_params = None
        results = []
        
        for params in self.generate_param_combinations(param_grid):
            try:
                # Create configuration
                config = HMMConfig(**params)
                
                # Train and evaluate
                score, metrics = self.evaluate_config(config, metric)
                
                results.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
                print(f"Params: {params}, Score: {score:.4f}")
                
            except Exception as e:
                print(f"Failed with params {params}: {e}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def generate_param_combinations(self, param_grid):
        """Generate all combinations of parameters"""
        from itertools import product
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        for combination in product(*values):
            yield dict(zip(keys, combination))
    
    def evaluate_config(self, config, metric):
        """Evaluate configuration performance"""
        
        # Train model
        hmm = HiddenMarkovModel(config=config)
        hmm.fit(self.training_data, verbose=False)
        
        # Generate predictions on validation data
        states = hmm.predict(self.validation_data)
        probs = hmm.predict_proba(self.validation_data)
        
        # Calculate regime-based returns (simplified)
        regime_returns = self.calculate_regime_returns(states, self.validation_data)
        
        # Calculate metrics
        metrics = {
            'total_return': np.sum(regime_returns),
            'volatility': np.std(regime_returns),
            'sharpe_ratio': np.mean(regime_returns) / np.std(regime_returns) * np.sqrt(252) if np.std(regime_returns) > 0 else 0,
            'avg_confidence': np.mean(np.max(probs, axis=1)),
            'regime_stability': self.calculate_regime_stability(states)
        }
        
        return metrics[metric], metrics
    
    def calculate_regime_returns(self, states, returns):
        """Calculate returns based on regime predictions (simplified)"""
        # Simple strategy: long in bull (state 2), short in bear (state 0), neutral in sideways (state 1)
        position_map = {0: -1, 1: 0, 2: 1}  # Bear, Sideways, Bull
        
        positions = [position_map.get(state, 0) for state in states]
        regime_returns = [pos * ret for pos, ret in zip(positions, returns)]
        
        return np.array(regime_returns)
    
    def calculate_regime_stability(self, states):
        """Calculate regime stability metric"""
        changes = np.sum(states[1:] != states[:-1])
        return 1.0 - (changes / len(states))

# Usage example
def optimize_parameters_example():
    """Example of parameter optimization"""
    
    # Generate sample data
    training_data = np.random.normal(0.001, 0.02, 800)
    validation_data = np.random.normal(0.001, 0.02, 200)
    
    optimizer = HMMParameterOptimizer(training_data, validation_data)
    
    # Define parameter grid
    param_grid = {
        'n_states': [2, 3, 4],
        'max_iterations': [50, 100, 150],
        'tolerance': [1e-5, 1e-6, 1e-7],
        'initialization_method': ['random', 'kmeans'],
        'regularization': [1e-7, 1e-6, 1e-5]
    }
    
    # Optimize (warning: this will take time!)
    results = optimizer.optimize_hmm_config(param_grid, metric='sharpe_ratio')
    
    print("Optimization Results:")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Best Score: {results['best_score']:.4f}")
    
    return results

# Bayesian optimization (more efficient)
def bayesian_optimize_config():
    """Bayesian optimization for parameter tuning"""
    
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        
        # Define search space
        dimensions = [
            Real(0.95, 0.999, name='forgetting_factor'),
            Real(0.01, 0.15, name='adaptation_rate'),
            Real(0.5, 0.95, name='smoothing_weight'),
            Integer(5, 50, name='min_observations_for_update')
        ]
        
        def objective(params):
            """Objective function to minimize (negative score)"""
            forgetting_factor, adaptation_rate, smoothing_weight, min_obs = params
            
            try:
                config = OnlineHMMConfig(
                    forgetting_factor=forgetting_factor,
                    adaptation_rate=adaptation_rate,
                    smoothing_weight=smoothing_weight,
                    min_observations_for_update=min_obs
                )
                
                # Evaluate configuration (simplified)
                online_hmm = OnlineHMM(n_states=3, online_config=config)
                
                # Simple evaluation metric
                score = np.random.random()  # Replace with actual evaluation
                
                return -score  # Minimize negative score = maximize score
                
            except Exception as e:
                return 1000  # High penalty for invalid configurations
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=50,  # Number of evaluations
            random_state=42
        )
        
        print("Bayesian Optimization Results:")
        print(f"Best parameters: {result.x}")
        print(f"Best score: {-result.fun}")
        
        return result
        
    except ImportError:
        print("skopt not available. Install with: pip install scikit-optimize")
        return None

if __name__ == "__main__":
    # optimize_parameters_example()  # Warning: slow!
    bayesian_optimize_config()
```

---

## Best Practices

### Configuration Management

#### 1. Version Control Your Configurations
```python
# config_v1.py
CONFIG_VERSION = "1.0"
PRODUCTION_CONFIG = {
    'hmm_config': HMMConfig(
        n_states=3,
        max_iterations=100,
        tolerance=1e-6
    ),
    'online_config': OnlineHMMConfig(
        forgetting_factor=0.98,
        adaptation_rate=0.05
    )
}

# Always include version and metadata
METADATA = {
    'version': CONFIG_VERSION,
    'created': '2024-01-15',
    'description': 'Initial production configuration',
    'validation_data': 'SPY_2023_data.csv'
}
```

#### 2. Configuration Factory Pattern
```python
class ConfigurationFactory:
    """Factory for creating standard configurations"""
    
    @staticmethod
    def create_config(config_type, asset_class='equity', frequency='daily'):
        """Create configuration based on use case"""
        
        config_map = {
            ('development', 'equity', 'daily'): {
                'hmm': HMMConfig(n_states=3, max_iterations=50),
                'online': OnlineHMMConfig(forgetting_factor=0.97)
            },
            ('production', 'equity', 'daily'): {
                'hmm': HMMConfig(n_states=3, max_iterations=100),
                'online': OnlineHMMConfig(forgetting_factor=0.98)
            },
            ('production', 'crypto', 'hourly'): {
                'hmm': HMMConfig(n_states=4, max_iterations=150),
                'online': OnlineHMMConfig(forgetting_factor=0.95)
            }
        }
        
        key = (config_type, asset_class, frequency)
        if key in config_map:
            return config_map[key]
        else:
            raise ValueError(f"No configuration for {key}")

# Usage
configs = ConfigurationFactory.create_config('production', 'equity', 'daily')
hmm_config = configs['hmm']
online_config = configs['online']
```

#### 3. Configuration Validation Pipeline
```python
def configuration_pipeline(config, validation_data):
    """Complete configuration validation pipeline"""
    
    # Step 1: Parameter validation
    validator = ConfigurationValidator(validation_data)
    validation_results = validator.validate_config(config['hmm'], config['online'])
    
    if not validation_results['is_valid']:
        raise ValueError(f"Configuration invalid: {validation_results['errors']}")
    
    # Step 2: Performance testing
    performance_results = test_configuration_performance(config, validation_data)
    
    # Step 3: Memory and speed testing
    efficiency_results = test_configuration_efficiency(config)
    
    # Step 4: Generate report
    report = {
        'validation': validation_results,
        'performance': performance_results,
        'efficiency': efficiency_results,
        'overall_score': calculate_overall_config_score(validation_results, performance_results, efficiency_results),
        'recommendations': generate_configuration_recommendations(validation_results)
    }
    
    return report
```

### Configuration Documentation

#### Document Configuration Decisions
```python
class ConfigurationDocumentation:
    """Document configuration choices and rationale"""
    
    def __init__(self):
        self.config_history = []
        
    def document_config_change(self, old_config, new_config, rationale, performance_impact=None):
        """Document configuration changes"""
        
        change_record = {
            'timestamp': datetime.now(),
            'old_config': self.serialize_config(old_config),
            'new_config': self.serialize_config(new_config),
            'rationale': rationale,
            'performance_impact': performance_impact,
            'changed_parameters': self.find_config_differences(old_config, new_config)
        }
        
        self.config_history.append(change_record)
        
    def serialize_config(self, config):
        """Serialize configuration for storage"""
        if isinstance(config, dict):
            return {k: self.serialize_config(v) for k, v in config.items()}
        elif hasattr(config, '__dict__'):
            return config.__dict__.copy()
        else:
            return config
            
    def find_config_differences(self, old_config, new_config):
        """Find differences between configurations"""
        differences = []
        
        # Simple implementation - could be enhanced
        old_dict = self.serialize_config(old_config)
        new_dict = self.serialize_config(new_config)
        
        for key, new_value in new_dict.items():
            if key in old_dict and old_dict[key] != new_value:
                differences.append({
                    'parameter': key,
                    'old_value': old_dict[key],
                    'new_value': new_value
                })
        
        return differences
```

---

## Troubleshooting Configuration Issues

### Common Configuration Problems

#### Problem: Model Not Converging
```python
def fix_convergence_issues(config):
    """Fix common convergence problems"""
    
    # Increase iterations
    config.max_iterations = min(config.max_iterations * 2, 500)
    
    # Loosen tolerance
    if config.tolerance < 1e-4:
        config.tolerance = 1e-4
    
    # Increase regularization
    config.regularization = max(config.regularization * 10, 1e-5)
    
    # Try different initialization
    if config.initialization_method == 'random':
        config.initialization_method = 'kmeans'
        config.n_init_trials = 5
    
    return config
```

#### Problem: Poor Online Performance
```python
def fix_online_performance_issues(online_config, symptoms):
    """Fix online performance issues based on symptoms"""
    
    if 'slow_adaptation' in symptoms:
        # Increase adaptation speed
        online_config.adaptation_rate = min(online_config.adaptation_rate * 1.5, 0.15)
        online_config.forgetting_factor = max(online_config.forgetting_factor - 0.01, 0.90)
        
    if 'unstable_parameters' in symptoms:
        # Increase stability
        online_config.parameter_smoothing = True
        online_config.smoothing_weight = min(online_config.smoothing_weight + 0.1, 0.95)
        online_config.min_observations_for_update = max(online_config.min_observations_for_update * 2, 20)
        
    if 'high_memory_usage' in symptoms:
        # Reduce memory usage
        online_config.rolling_window_size = max(online_config.rolling_window_size // 2, 100)
        online_config.sufficient_stats_decay = max(online_config.sufficient_stats_decay - 0.01, 0.95)
        
    if 'slow_processing' in symptoms:
        # Optimize for speed
        online_config.max_adaptation_iterations = min(online_config.max_adaptation_iterations, 3)
        online_config.enable_change_detection = False
    
    return online_config
```

### Diagnostic Tools

#### Configuration Health Check
```python
def configuration_health_check(hmm_model, online_config, recent_data):
    """Perform health check on current configuration"""
    
    health_report = {
        'overall_health': 'Good',
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check model performance
    if hasattr(hmm_model, 'get_current_regime_info'):
        regime_info = hmm_model.get_current_regime_info()
        
        if regime_info['confidence'] < 0.5:
            health_report['issues'].append('Low confidence in regime detection')
            health_report['recommendations'].append('Consider increasing smoothing_weight')
        
        if regime_info['confidence'] < 0.3:
            health_report['overall_health'] = 'Poor'
    
    # Check parameter stability
    if hasattr(hmm_model, 'get_parameter_evolution'):
        evolution = hmm_model.get_parameter_evolution()
        
        for state in range(hmm_model.n_states):
            if len(evolution['means'][state]) > 50:
                recent_means = evolution['means'][state][-50:]
                mean_volatility = np.std(recent_means)
                
                if mean_volatility > 0.01:  # 1% threshold
                    health_report['warnings'].append(f'High parameter volatility for state {state}')
                    health_report['recommendations'].append('Consider increasing smoothing_weight')
    
    # Check processing performance
    if hasattr(hmm_model, 'processing_times'):
        avg_time = np.mean(hmm_model.processing_times[-100:])  # Last 100 observations
        
        if avg_time > 0.01:  # 10ms threshold
            health_report['warnings'].append(f'Slow processing: {avg_time*1000:.1f}ms average')
            health_report['recommendations'].append('Consider reducing rolling_window_size')
    
    # Overall health assessment
    if health_report['issues']:
        health_report['overall_health'] = 'Poor' if len(health_report['issues']) > 2 else 'Fair'
    elif health_report['warnings']:
        health_report['overall_health'] = 'Fair' if len(health_report['warnings']) > 3 else 'Good'
    
    return health_report

# Usage
def run_health_check_example():
    # Assume we have a trained model
    health = configuration_health_check(trained_model, online_config, recent_data)
    
    print(f"Configuration Health: {health['overall_health']}")
    
    if health['issues']:
        print("Issues:")
        for issue in health['issues']:
            print(f"   {issue}")
    
    if health['warnings']:
        print("Warnings:")
        for warning in health['warnings']:
            print(f"  [WARNING]  {warning}")
    
    if health['recommendations']:
        print("Recommendations:")
        for rec in health['recommendations']:
            print(f"  Note: {rec}")
```

This comprehensive Configuration and Tuning Guide provides everything needed to properly configure HMM models for different use cases, optimize performance, and troubleshoot common issues. The guide emphasizes data-driven configuration choices and provides practical tools for validation and optimization.

---

*For additional information, see the [Online HMM Documentation](online_hmm.md), [Trading Applications Guide](trading_applications.md), and [Mathematical Foundations](mathematical_foundations.md).*