# Config Module

The config module provides a comprehensive configuration system for all pipeline components with validation, serialization, and factory patterns for common use cases.

## Overview

The configuration system enables type-safe, validated component configuration with easy serialization:

```
Configuration → Validation → Component Creation
      ↓             ↓              ↓
   Dataclass    Type Checking   Initialized
   Parameters   Value Ranges     Component
```

**Key Features:**
- Immutable dataclass configurations (frozen)
- Automatic validation on creation
- JSON serialization/deserialization
- Factory methods for common patterns
- Component creation interface
- Configuration comparison and hashing

## Core Concepts

### BaseConfig

Abstract base class providing common functionality for all configurations.

```python
from hidden_regime.config import BaseConfig

# All config classes inherit from BaseConfig
class MyConfig(BaseConfig):
    param1: int = 10
    param2: str = "default"

    def validate(self) -> None:
        """Validate parameters."""
        super().validate()
        if self.param1 < 0:
            raise ValueError("param1 must be non-negative")

    def create_component(self) -> Any:
        """Create component from configuration."""
        return MyComponent(self)
```

**Inherited Functionality:**
- `to_dict()` / `from_dict()` - Dictionary conversion
- `to_json()` / `from_json()` - JSON serialization
- `save()` / `load()` - File persistence
- `copy()` - Create modified copies
- `validate()` - Parameter validation
- `create_component()` - Component instantiation

## Configuration Classes

### Data Configurations

#### FinancialDataConfig

Configure financial data loading from yfinance.

```python
from hidden_regime.config import FinancialDataConfig

config = FinancialDataConfig(
    ticker='AAPL',
    source='yfinance',
    start_date='2023-01-01',
    end_date='2024-01-01',
    frequency='days'
)

# Validate
config.validate()  # Raises ConfigurationError if invalid

# Create component
data_loader = config.create_component()

# Generate cache key
cache_key = config.get_cache_key()
```

**Parameters:**
- `ticker` (str): Stock symbol (e.g., 'AAPL', 'SPY')
- `source` (str): Data source ('yfinance', 'alpha_vantage', 'quandl', 'csv')
- `start_date` (str): Start date (YYYY-MM-DD)
- `end_date` (str): End date (YYYY-MM-DD)
- `num_samples` (int): Number of samples (alternative to date range)
- `frequency` (str): Data frequency ('days', 'hours', 'minutes')

### Observation Configurations

#### FinancialObservationConfig

Configure feature generation from price data.

```python
from hidden_regime.config import FinancialObservationConfig

config = FinancialObservationConfig(
    generators=['log_return', 'volatility', 'rsi'],
    volatility_window=20,
    rsi_period=14,
    normalize_features=False,
    handle_missing=True
)

# Create observation generator
obs_generator = config.create_component()
```

**Factory Methods:**
```python
# Default financial features
config = FinancialObservationConfig.create_default_financial()
# Includes: log_return, volatility

# Comprehensive features
config = FinancialObservationConfig.create_comprehensive()
# Includes: log_return, volatility, rsi, macd, bollinger_bands
```

**Parameters:**
- `generators` (list): Feature generators to use
- `volatility_window` (int): Window for volatility calculation
- `rsi_period` (int): Period for RSI calculation
- `ma_window` (int): Moving average window
- `bollinger_window` (int): Bollinger Bands window
- `bollinger_std` (float): Bollinger Bands standard deviations
- `normalize_features` (bool): Scale features to [0, 1]
- `handle_missing` (bool): Handle missing values
- `fill_method` (str): Method for filling missing values

### Model Configurations

#### HMMConfig

Comprehensive configuration for Hidden Markov Models.

```python
from hidden_regime.config import HMMConfig

config = HMMConfig(
    # Core parameters
    n_states=3,
    observed_signal='log_return',
    max_iterations=100,
    tolerance=1e-6,

    # Initialization
    initialization_method='kmeans',  # 'kmeans', 'random', 'custom'
    random_seed=42,

    # Stability
    min_regime_duration=2,
    min_variance=1e-8,
    regularization=1e-6
)

# Create HMM model
hmm_model = config.create_component()
```

**Factory Methods:**
```python
# Conservative (stable, slow-adapting)
config = HMMConfig.create_conservative()

# Aggressive (fast-adapting, responsive)
config = HMMConfig.create_aggressive()

# Balanced (general-purpose)
config = HMMConfig.create_balanced()
```

**Core Parameters:**
- `n_states` (int): Number of hidden states (2-10)
- `observed_signal` (str): Signal to model ('log_return')
- `max_iterations` (int): Maximum training iterations
- `tolerance` (float): Convergence tolerance
- `initialization_method` (str): Initialization strategy
- `random_seed` (int): Random seed for reproducibility

**Stability Parameters:**
- `min_regime_duration` (int): Minimum regime duration
- `min_variance` (float): Minimum allowed variance
- `regularization` (float): Regularization strength
- `check_convergence_every` (int): Convergence check frequency
- `early_stopping` (bool): Enable early stopping

**Online Learning Parameters:**
- `forgetting_factor` (float): Memory decay (0.8-1.0)
- `adaptation_rate` (float): Learning rate (0.001-0.5)
- `min_observations_for_update` (int): Minimum observations for update
- `parameter_smoothing` (bool): Smooth parameter updates
- `smoothing_weight` (float): Smoothing weight (0.1-1.0)
- `rolling_window_size` (int): Rolling window size
- `sufficient_stats_decay` (float): Statistics decay rate

**Change Detection Parameters:**
- `enable_change_detection` (bool): Enable change detection
- `change_detection_threshold` (float): Threshold for changes
- `change_detection_window` (int): Window for detection
- `convergence_tolerance` (float): Convergence tolerance
- `max_adaptation_iterations` (int): Maximum adaptation iterations

### Analysis Configurations

#### FinancialAnalysisConfig

Configure financial analysis and regime interpretation.

```python
from hidden_regime.config import FinancialAnalysisConfig

config = FinancialAnalysisConfig(
    # Core analysis
    compute_regime_stats=True,
    performance_metrics=True,

    # Technical indicators
    indicator_comparisons=True,
    indicators_to_compare=['RSI', 'MACD', 'SMA_Crossover'],

    # Regime interpretation
    n_states=3,
    regime_labels=['Bear', 'Sideways', 'Bull'],

    # Performance calculation
    risk_free_rate=0.02,
    trading_days_per_year=252
)

# Create analyzer
analyzer = config.create_component()
```

**Factory Methods:**
```python
# Comprehensive analysis
config = FinancialAnalysisConfig.create_comprehensive_financial()

# Basic analysis
config = FinancialAnalysisConfig.create_basic()

# Trading-focused
config = FinancialAnalysisConfig.create_trading_focused()
```

**Parameters:**
- `compute_regime_stats` (bool): Calculate regime statistics
- `performance_metrics` (bool): Compute performance metrics
- `indicator_comparisons` (bool): Compare with indicators
- `indicators_to_compare` (list): Indicators to compare
- `n_states` (int): Number of regimes
- `regime_labels` (list): Human-readable labels
- `risk_free_rate` (float): Annual risk-free rate
- `trading_days_per_year` (int): Trading days per year

### Report Configurations

#### ReportConfig

Configure report generation and formatting.

```python
from hidden_regime.config import ReportConfig

config = ReportConfig(
    # Output settings
    output_dir='./reports',
    output_format='markdown',  # markdown, html, pdf, json

    # Visualization
    show_plots=False,
    save_plots=True,
    plot_format='png',
    plot_dpi=300,

    # Content sections
    include_summary=True,
    include_regime_analysis=True,
    include_performance_metrics=True,
    include_risk_analysis=True,
    include_trading_signals=False,
    include_data_quality=True,

    # Styling
    title='Market Regime Analysis',
    template_style='professional'  # professional, academic, minimal
)

# Create report generator
report_gen = config.create_component()
```

**Factory Methods:**
```python
# Minimal report
config = ReportConfig.create_minimal()

# Comprehensive report
config = ReportConfig.create_comprehensive()

# Presentation-ready
config = ReportConfig.create_presentation()
```

## Configuration Operations

### Validation

All configurations validate on creation:

```python
from hidden_regime.config import HMMConfig
from hidden_regime.utils.exceptions import ConfigurationError

try:
    # This will raise ConfigurationError
    config = HMMConfig(n_states=1)  # Too few states
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Serialization

#### JSON Serialization

```python
from hidden_regime.config import FinancialDataConfig

config = FinancialDataConfig(ticker='AAPL')

# To JSON string
json_str = config.to_json()

# From JSON string
loaded_config = FinancialDataConfig.from_json(json_str)

assert config == loaded_config
```

#### Dictionary Conversion

```python
# To dictionary
config_dict = config.to_dict()
# {'ticker': 'AAPL', 'source': 'yfinance', ...}

# From dictionary
config = FinancialDataConfig.from_dict(config_dict)
```

#### File Persistence

```python
# Save to file
config.save('./configs/data_config.json')

# Load from file
loaded_config = FinancialDataConfig.load('./configs/data_config.json')
```

### Copying and Modification

```python
# Create modified copy
original_config = HMMConfig(n_states=3, max_iterations=100)

modified_config = original_config.copy(
    n_states=4,
    max_iterations=200
)

# Original unchanged (immutable)
assert original_config.n_states == 3
assert modified_config.n_states == 4
```

### Comparison

```python
config1 = HMMConfig(n_states=3)
config2 = HMMConfig(n_states=3)
config3 = HMMConfig(n_states=4)

# Equality comparison
assert config1 == config2
assert config1 != config3

# Hashing (for caching)
config_hash = hash(config1)
cache = {config1: result}  # Use as dictionary key
```

## Usage Examples

### Example 1: Basic Configuration

```python
from hidden_regime.config import (
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    FinancialAnalysisConfig
)

# Configure each component
data_config = FinancialDataConfig(
    ticker='SPY',
    start_date='2023-01-01',
    end_date='2024-01-01'
)

obs_config = FinancialObservationConfig(
    generators=['log_return', 'volatility']
)

model_config = HMMConfig(
    n_states=3,
    max_iterations=100
)

analysis_config = FinancialAnalysisConfig(
    compute_regime_stats=True,
    indicator_comparisons=False
)

# Create components
data_loader = data_config.create_component()
obs_generator = obs_config.create_component()
model = model_config.create_component()
analyzer = analysis_config.create_component()
```

### Example 2: Factory Method Configuration

```python
from hidden_regime.config import (
    FinancialObservationConfig,
    HMMConfig,
    FinancialAnalysisConfig,
    ReportConfig
)

# Use factory methods for common patterns
obs_config = FinancialObservationConfig.create_comprehensive()

model_config = HMMConfig.create_balanced()

analysis_config = FinancialAnalysisConfig.create_comprehensive_financial()

report_config = ReportConfig.create_comprehensive()
```

### Example 3: Save and Load Configuration

```python
from hidden_regime.config import HMMConfig

# Create configuration
config = HMMConfig(
    n_states=4,
    max_iterations=200,
    initialization_method='kmeans'
)

# Save to file
config.save('./configs/my_hmm_config.json')

# Later: Load from file
loaded_config = HMMConfig.load('./configs/my_hmm_config.json')

# Use loaded configuration
model = loaded_config.create_component()
```

### Example 4: Configuration Pipeline

```python
from hidden_regime.config import (
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    FinancialAnalysisConfig,
    ReportConfig
)

# Define all configurations
configs = {
    'data': FinancialDataConfig(ticker='AAPL'),
    'observation': FinancialObservationConfig.create_default_financial(),
    'model': HMMConfig.create_balanced(),
    'analysis': FinancialAnalysisConfig.create_basic(),
    'report': ReportConfig.create_minimal()
}

# Save configuration set
import json
with open('./configs/pipeline_config.json', 'w') as f:
    json.dump(
        {k: v.to_dict() for k, v in configs.items()},
        f,
        indent=2,
        default=str
    )

# Later: Load and create pipeline
with open('./configs/pipeline_config.json', 'r') as f:
    config_data = json.load(f)

# Recreate configs
data_config = FinancialDataConfig.from_dict(config_data['data'])
obs_config = FinancialObservationConfig.from_dict(config_data['observation'])
# ... etc
```

### Example 5: Configuration Modification

```python
from hidden_regime.config import HMMConfig

# Start with balanced configuration
base_config = HMMConfig.create_balanced()

# Create variants for experimentation
conservative = base_config.copy(
    forgetting_factor=0.99,
    adaptation_rate=0.01
)

aggressive = base_config.copy(
    forgetting_factor=0.95,
    adaptation_rate=0.15,
    enable_change_detection=True
)

# Run experiments with different configs
for name, config in [('conservative', conservative), ('aggressive', aggressive)]:
    model = config.create_component()
    # Run analysis...
```

## Best Practices

### 1. Use Factory Methods

```python
# Better: Use factory method
config = HMMConfig.create_balanced()

# Rather than: Manual specification
config = HMMConfig(
    n_states=3,
    max_iterations=100,
    tolerance=1e-6,
    forgetting_factor=0.98,
    # ... many more parameters
)
```

### 2. Validate Early

```python
# Validation happens automatically on creation
try:
    config = HMMConfig(n_states=1)  # Invalid
except ConfigurationError:
    # Handle error
    pass

# Explicit validation after modification
config_dict = config.to_dict()
config_dict['n_states'] = 50  # Too many
try:
    invalid_config = HMMConfig.from_dict(config_dict)
except ConfigurationError:
    # Handle error
    pass
```

### 3. Save Configurations

```python
# Save successful configurations
config = HMMConfig.create_balanced()
config.save('./configs/successful_config.json')

# Document what worked
metadata = {
    'config': config.to_dict(),
    'performance': {'sharpe': 1.25, 'return': 0.15},
    'notes': 'Best performing configuration on SPY 2023'
}
```

### 4. Version Configurations

```python
# Include version information
config_with_version = {
    'version': '1.0.0',
    'created': '2024-01-01',
    'config': config.to_dict(),
    'description': 'Production HMM configuration'
}
```

### 5. Use Type Hints

```python
from typing import Dict
from hidden_regime.config import HMMConfig

def train_model(config: HMMConfig) -> Any:
    """Type hints ensure correct configuration type."""
    model = config.create_component()
    # Training logic...
    return model

# IDE will catch type errors
config = HMMConfig.create_balanced()
train_model(config)  # ✓ Correct

wrong_config = FinancialDataConfig(ticker='AAPL')
train_model(wrong_config)  # ✗ Type error
```

## Configuration Patterns

### Pattern 1: Configuration Inheritance

```python
# Base configuration for production
PRODUCTION_BASE = HMMConfig(
    n_states=3,
    max_iterations=100,
    random_seed=42
)

# Variants for specific assets
spy_config = PRODUCTION_BASE.copy(n_states=3)
aapl_config = PRODUCTION_BASE.copy(n_states=4)
btc_config = PRODUCTION_BASE.copy(
    n_states=5,
    enable_change_detection=True
)
```

### Pattern 2: Configuration Registry

```python
from typing import Dict
from hidden_regime.config import HMMConfig

# Registry of known configurations
CONFIG_REGISTRY: Dict[str, HMMConfig] = {
    'conservative': HMMConfig.create_conservative(),
    'balanced': HMMConfig.create_balanced(),
    'aggressive': HMMConfig.create_aggressive(),
}

def get_config(name: str) -> HMMConfig:
    """Get configuration by name."""
    if name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown configuration: {name}")
    return CONFIG_REGISTRY[name]

# Usage
config = get_config('balanced')
```

### Pattern 3: Environment-Specific Configs

```python
import os

def get_environment_config() -> HMMConfig:
    """Get configuration based on environment."""
    env = os.getenv('ENV', 'development')

    if env == 'production':
        return HMMConfig.create_conservative()
    elif env == 'staging':
        return HMMConfig.create_balanced()
    else:  # development
        return HMMConfig.create_aggressive()
```

## Module Structure

```
config/
├── __init__.py            # Public API
├── base.py                # BaseConfig, MutableBaseConfig
├── data.py                # DataConfig, FinancialDataConfig
├── observation.py         # ObservationConfig, FinancialObservationConfig
├── model.py               # ModelConfig, HMMConfig
├── analysis.py            # AnalysisConfig, FinancialAnalysisConfig
├── report.py              # ReportConfig
├── case_study.py          # CaseStudyConfig
└── simulation.py          # SimulationConfig
```

## Related Modules

- **[pipeline](../pipeline/README.md)**: Pipeline component integration
- **[factories](../factories/README.md)**: High-level pipeline creation
- **[data](../data/README.md)**: Data loaders (created from configs)
- **[models](../models/README.md)**: Models (created from configs)

## Key Concepts

### Immutability

Configurations use frozen dataclasses for safety:

```python
config = HMMConfig(n_states=3)

# This raises an error
config.n_states = 4  # FrozenInstanceError

# Instead, create a new configuration
new_config = config.copy(n_states=4)
```

### Validation

Validation ensures configurations are correct:

```python
# Automatic validation on creation
config = HMMConfig(n_states=3)  # ✓ Valid

config = HMMConfig(n_states=1)  # ✗ Raises ConfigurationError
```

### Component Creation

Configs create their corresponding components:

```python
# Configuration knows how to create its component
config = HMMConfig(n_states=3)
model = config.create_component()
# Returns HiddenMarkovModel instance

# This pattern enables:
# 1. Separation of configuration from implementation
# 2. Easy serialization/deserialization
# 3. Configuration-driven component selection
```

---

For complete examples using configurations, see all example scripts in `examples/` directory in the project root.