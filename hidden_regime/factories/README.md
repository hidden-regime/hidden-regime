# Factories Module

The factories module provides convenient factory functions for creating pipelines and components from configuration objects, following the Factory Method pattern for extensibility and consistency.

## Overview

Rather than manually constructing pipelines and components, factories provide:

- **Pre-configured pipelines** for common use cases
- **Consistent construction** from configuration objects
- **Extensibility** for new component types
- **Simplified API** with sensible defaults

## Factory Types

### PipelineFactory

Creates complete pipeline systems from configuration objects.

```python
from hidden_regime.factories import pipeline_factory

# Create custom pipeline
pipeline = pipeline_factory.create_pipeline(
    data_config=data_cfg,
    observation_config=obs_cfg,
    model_config=model_cfg,
    analysis_config=analysis_cfg,
    report_config=report_cfg  # optional
)
```

### ComponentFactory

Creates individual pipeline components from configurations.

```python
from hidden_regime.factories import component_factory

# Create individual components
data_component = component_factory.create_data_component(data_config)
model_component = component_factory.create_model_component(model_config)
```

## Pre-Configured Pipelines

### Simple Regime Pipeline

Quick start for basic regime detection.

```python
import hidden_regime as hr

pipeline = hr.create_simple_regime_pipeline(
    ticker='AAPL',
    n_states=3
)

result = pipeline.update()
```

**Use Cases:**
- Quick exploration
- Initial regime analysis
- Educational demonstrations

**What's Included:**
- Data loading (yfinance)
- Log return observations
- 3-state HMM
- Basic regime analysis
- No report generation

### Financial Analysis Pipeline

Comprehensive financial analysis with full reporting.

```python
import hidden_regime as hr

pipeline = hr.create_financial_pipeline(
    ticker='SPY',
    n_states=3,
    start_date='2023-01-01',
    end_date='2024-01-01',
    include_report=True
)

result = pipeline.update()
```

**Configuration Options:**
```python
pipeline = hr.create_financial_pipeline(
    ticker='AAPL',
    n_states=4,
    start_date='2023-01-01',
    end_date='2024-01-01',
    include_report=True,
    # Model parameters
    tolerance=1e-6,
    max_iterations=100,
    random_seed=42,
    # Additional overrides
    data_config_overrides={'use_ohlc_average': True},
    model_config_overrides={'initialization_method': 'kmeans'}
)
```

**Use Cases:**
- Production analysis
- Comprehensive reporting
- Performance tracking
- Multi-state regime detection

### Trading Pipeline

Optimized for trading strategy development.

```python
import hidden_regime as hr

pipeline = hr.create_trading_pipeline(
    ticker='QQQ',
    n_states=4,
    risk_adjustment=True
)

result = pipeline.update()
```

**Features:**
- Risk-adjusted position sizing
- Trading signal generation
- Performance metrics
- Regime-specific strategies

**Use Cases:**
- Strategy backtesting
- Risk management
- Trading system development

### Research Pipeline

Comprehensive analysis for research applications.

```python
import hidden_regime as hr

pipeline = hr.create_research_pipeline(
    ticker='BTC-USD',
    n_states=3,
    comprehensive_analysis=True
)

result = pipeline.update()
```

**Features:**
- Extensive statistical analysis
- Technical indicator comparison
- Regime characterization
- Detailed reporting

**Use Cases:**
- Academic research
- Market analysis
- Strategy development
- Performance comparison

## Custom Pipeline Creation

### Using Configuration Objects

```python
from hidden_regime.config import (
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    FinancialAnalysisConfig,
    ReportConfig
)
from hidden_regime.factories import pipeline_factory

# Configure each component
data_config = FinancialDataConfig(
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    use_ohlc_average=True
)

obs_config = FinancialObservationConfig.create_default_financial()

model_config = HMMConfig(
    n_states=4,
    max_iterations=100,
    tolerance=1e-6,
    initialization_method='kmeans',
    random_seed=42
)

analysis_config = FinancialAnalysisConfig.create_comprehensive_financial()

report_config = ReportConfig(
    output_directory='./reports',
    generate_plots=True
)

# Create pipeline
pipeline = pipeline_factory.create_pipeline(
    data_config=data_config,
    observation_config=obs_config,
    model_config=model_config,
    analysis_config=analysis_config,
    report_config=report_config
)
```

### Configuration Factory Methods

Most config classes provide factory methods for common scenarios:

```python
from hidden_regime.config import HMMConfig, FinancialAnalysisConfig

# Market-optimized HMM
hmm_config = HMMConfig.for_market_data(conservative=True)

# Comprehensive financial analysis
analysis_config = FinancialAnalysisConfig.create_comprehensive_financial()

# Balanced HMM settings
balanced_hmm = HMMConfig.create_balanced()
```

## Component Factory Usage

For custom pipeline construction:

```python
from hidden_regime.factories import component_factory
from hidden_regime.config import (
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig
)
from hidden_regime.pipeline import Pipeline

# Create individual components
data = component_factory.create_data_component(
    FinancialDataConfig(ticker='AAPL')
)

observations = component_factory.create_observation_component(
    FinancialObservationConfig.create_default_financial()
)

model = component_factory.create_model_component(
    HMMConfig(n_states=3)
)

# Manual pipeline assembly (not recommended unless necessary)
from hidden_regime.analysis import FinancialAnalysis
from hidden_regime.config import FinancialAnalysisConfig

analysis = FinancialAnalysis(FinancialAnalysisConfig())

pipeline = Pipeline(
    data=data,
    observation=observations,
    model=model,
    analysis=analysis
)
```

## Factory Patterns

### Registration Pattern

Factories use component type registration for extensibility:

```python
# Internal implementation
class ComponentFactory:
    def __init__(self):
        self._data_types = {'financial': FinancialDataLoader}
        self._model_types = {'hmm': HiddenMarkovModel}
        # ... other registrations

    def create_data_component(self, config):
        component_type = config.source
        component_class = self._data_types.get(component_type)
        return component_class(config)
```

This allows adding new component types without modifying factory code.

### Configuration-Based Construction

All components are created from configuration objects:

```python
# Configuration defines behavior
config = HMMConfig(
    n_states=3,
    max_iterations=100,
    tolerance=1e-6
)

# Factory creates component
model = component_factory.create_model_component(config)

# Result: HiddenMarkovModel with specified configuration
```

## Best Practices

### 1. Use Pre-Configured Pipelines

```python
# Good: Use factory for standard use cases
pipeline = hr.create_financial_pipeline('AAPL', n_states=3)

# Avoid: Manual construction unless customizing
```

### 2. Override Only What's Needed

```python
# Override specific parameters
pipeline = hr.create_financial_pipeline(
    'SPY',
    n_states=4,
    tolerance=1e-8,  # Custom tolerance
    # Keep other defaults
)
```

### 3. Use Configuration Factory Methods

```python
# Good: Use config factory methods
config = HMMConfig.for_market_data(conservative=True)

# Avoid: Manual configuration unless necessary
config = HMMConfig(
    n_states=3,
    max_iterations=100,
    tolerance=1e-6,
    # ... many parameters
)
```

### 4. Validate Configurations

```python
from hidden_regime.config import HMMConfig

config = HMMConfig(n_states=10)  # May be too many states

# Validate before creating pipeline
if config.n_states > 5:
    print("Warning: Many states may lead to overfitting")
```

## Factory Function Reference

### Top-Level Functions

Available via `import hidden_regime as hr`:

```python
# Pipeline creation
hr.create_pipeline()                    # Custom from configs
hr.create_financial_pipeline()          # Financial analysis
hr.create_simple_regime_pipeline()      # Quick start
hr.create_trading_pipeline()            # Trading focus
hr.create_research_pipeline()           # Research focus

# Temporal control
hr.create_temporal_controller()         # Backtesting controller
```

### PipelineFactory Methods

```python
from hidden_regime.factories import pipeline_factory

pipeline_factory.create_pipeline(...)
pipeline_factory.create_financial_pipeline(...)
# ... etc
```

### ComponentFactory Methods

```python
from hidden_regime.factories import component_factory

component_factory.create_data_component(config)
component_factory.create_observation_component(config)
component_factory.create_model_component(config)
component_factory.create_analysis_component(config)
component_factory.create_report_component(config)
```

## Module Structure

```
factories/
├── __init__.py           # Public API exports
├── components.py         # ComponentFactory class
└── pipeline.py           # PipelineFactory class
```

## Examples

### Example 1: Quick Start

```python
import hidden_regime as hr

# Simplest possible usage
pipeline = hr.create_simple_regime_pipeline('AAPL')
result = pipeline.update()
print(result)
```

### Example 2: Custom Analysis Period

```python
import hidden_regime as hr

pipeline = hr.create_financial_pipeline(
    ticker='SPY',
    n_states=3,
    start_date='2020-01-01',
    end_date='2023-12-31'
)

result = pipeline.update()
```

### Example 3: Multiple Assets

```python
import hidden_regime as hr

tickers = ['AAPL', 'GOOGL', 'MSFT']

for ticker in tickers:
    pipeline = hr.create_financial_pipeline(ticker, n_states=3)
    result = pipeline.update()
    print(f"{ticker}: {result}")
```

### Example 4: Custom HMM Configuration

```python
import hidden_regime as hr
from hidden_regime.config import HMMConfig

# Custom HMM settings
hmm_config = HMMConfig(
    n_states=4,
    max_iterations=200,
    tolerance=1e-8,
    initialization_method='kmeans',
    random_seed=42
)

pipeline = hr.create_financial_pipeline(
    'AAPL',
    model_config_overrides=hmm_config.__dict__
)

result = pipeline.update()
```

## Related Modules

- **[pipeline](../pipeline/README.md)**: Pipeline architecture and components
- **[config](../config/README.md)**: Configuration dataclasses
- **[data](../data/README.md)**: Data components
- **[models](../models/README.md)**: Model components
- **[analysis](../analysis/README.md)**: Analysis components

---

For complete examples using factories, see `examples/` directory in the project root.