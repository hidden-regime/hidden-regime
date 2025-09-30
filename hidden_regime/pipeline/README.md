# Pipeline Architecture

The pipeline module provides the core architecture for building composable, reusable regime detection workflows with rigorous temporal data isolation for backtesting.

## Overview

The Hidden Regime pipeline implements a standardized **Data → Observation → Model → Analysis → Report** flow where each component follows a common interface, enabling flexible composition and systematic backtesting.

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
│    Data     │───▶│ Observation  │───▶│    Model    │───▶│   Analysis   │───▶│   Report   │
│  Component  │    │  Component   │    │  Component  │    │  Component   │    │ Component  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └────────────┘
      ↓                   ↓                   ↓                   ↓                   ↓
  Raw Data          Observations         HMM States       Regime Stats         Markdown
  Validation        Features             Probabilities    Performance          Plots
```

## Key Components

### Pipeline

Main orchestrator that coordinates component execution.

```python
from hidden_regime import Pipeline

pipeline = Pipeline(
    data=data_component,
    observation=observation_component,
    model=model_component,
    analysis=analysis_component,
    report=report_component  # optional
)

# Execute complete flow
result = pipeline.update()
```

### Component Interfaces

All pipeline components implement the `PipelineComponent` base interface:

```python
class PipelineComponent(ABC):
    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Process input and return output"""
        pass

    @abstractmethod
    def plot(self, **kwargs) -> plt.Figure:
        """Generate visualization"""
        pass
```

#### DataComponent

Loads and manages raw financial data.

```python
class DataComponent(PipelineComponent):
    @abstractmethod
    def get_all_data(self) -> pd.DataFrame:
        """Get complete dataset"""
        pass

    @abstractmethod
    def update(self, current_date: Optional[str] = None) -> pd.DataFrame:
        """Update data up to current_date"""
        pass
```

**Implementations:**
- `FinancialDataLoader` - Loads stock data from yfinance

#### ObservationComponent

Transforms raw data into model inputs (features).

```python
class ObservationComponent(PipelineComponent):
    @abstractmethod
    def update(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate observations from raw data"""
        pass
```

**Implementations:**
- `FinancialObservationGenerator` - Generates log returns and financial features

#### ModelComponent

Trains models and makes predictions.

```python
class ModelComponent(PipelineComponent):
    @abstractmethod
    def fit(self, observations: pd.DataFrame) -> None:
        """Train model on observations"""
        pass

    @abstractmethod
    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions"""
        pass
```

**Implementations:**
- `HiddenMarkovModel` - HMM-based regime detection

#### AnalysisComponent

Interprets model outputs and computes metrics.

```python
class AnalysisComponent(PipelineComponent):
    @abstractmethod
    def update(self, **kwargs) -> str:
        """Analyze results and return summary"""
        pass
```

**Implementations:**
- `FinancialAnalysis` - Regime statistics and performance metrics

#### ReportComponent (Optional)

Generates reports from pipeline results.

```python
class ReportComponent(PipelineComponent):
    @abstractmethod
    def update(self, **kwargs) -> str:
        """Generate report"""
        pass
```

**Implementations:**
- `MarkdownReportGenerator` - Generates markdown reports

## Temporal Data Isolation

### TemporalController

Provides rigorous backtesting by preventing future data leakage. This is critical for validating trading strategies.

```python
from hidden_regime.pipeline import TemporalController

# Create pipeline with full dataset
pipeline = create_financial_pipeline('AAPL')
full_data = pipeline.data.get_all_data()

# Create temporal controller
controller = TemporalController(pipeline, full_data)

# Step through time without data leakage
results = controller.step_through_time(
    start_date='2023-01-01',
    end_date='2024-01-01',
    step_days=1  # Daily updates
)

# Analyze results
for date, result in results.items():
    print(f"{date}: {result['current_regime']}")
```

**Key Features:**
- **No Future Data Access**: Each step only sees data up to that point in time
- **Isolated Updates**: Pipeline state resets at each step
- **Performance Tracking**: Tracks model evolution over time
- **Data Collection**: Records decisions and metrics at each step

### TemporalDataStub

Internal class that wraps temporally filtered data to prevent leakage.

```python
# Created automatically by TemporalController
stub = TemporalDataStub(filtered_data)

# Can only access pre-filtered data
data = stub.get_all_data()  # Returns only data up to filter date
data = stub.update('2025-01-01')  # Still returns filtered data (ignores future date)
```

## Usage Examples

### Basic Pipeline Creation

```python
import hidden_regime as hr

# Using factory (recommended)
pipeline = hr.create_financial_pipeline(
    ticker='SPY',
    n_states=3,
    start_date='2023-01-01',
    end_date='2024-01-01'
)

result = pipeline.update()
print(result)
```

### Manual Pipeline Construction

```python
from hidden_regime.config import (
    FinancialDataConfig,
    FinancialObservationConfig,
    HMMConfig,
    FinancialAnalysisConfig
)
from hidden_regime import create_pipeline

# Configure each component
data_config = FinancialDataConfig(ticker='AAPL')
obs_config = FinancialObservationConfig.create_default_financial()
model_config = HMMConfig.create_balanced()
analysis_config = FinancialAnalysisConfig.create_comprehensive_financial()

# Create pipeline
pipeline = create_pipeline(
    data_config=data_config,
    observation_config=obs_config,
    model_config=model_config,
    analysis_config=analysis_config
)

result = pipeline.update()
```

### Custom Components

```python
from hidden_regime.pipeline import PipelineComponent, Pipeline
import pandas as pd

class CustomAnalysis(AnalysisComponent):
    def update(self, data, observations, model_output):
        # Custom analysis logic
        return "Custom analysis results"

    def plot(self, **kwargs):
        # Custom visualization
        pass

# Use custom component
pipeline = Pipeline(
    data=data_component,
    observation=obs_component,
    model=model_component,
    analysis=CustomAnalysis(),  # Your custom component
    report=None
)
```

### Temporal Backtesting

```python
from hidden_regime.pipeline import TemporalController

# Setup
pipeline = hr.create_trading_pipeline('QQQ', n_states=4)
full_data = pipeline.data.get_all_data()

# Create controller
controller = TemporalController(pipeline, full_data)

# Backtest strategy
results = controller.step_through_time(
    start_date='2023-01-01',
    end_date='2024-01-01',
    step_days=7  # Weekly updates
)

# Analyze performance over time
regime_changes = []
for date, result in results.items():
    if 'regime_changed' in result and result['regime_changed']:
        regime_changes.append(date)

print(f"Detected {len(regime_changes)} regime changes")
```

### Component State Management

```python
# Access component outputs
pipeline.update()

# Get intermediate results
data = pipeline.component_outputs['data']
observations = pipeline.component_outputs['observations']
model_output = pipeline.component_outputs['model_output']
analysis = pipeline.component_outputs['analysis']

# Inspect pipeline state
print(f"Last update: {pipeline.last_update}")
print(f"Update count: {pipeline.update_count}")
```

## Component Communication

The pipeline passes data between components using keyword arguments:

```
Pipeline.update(current_date=None)
  ↓
data = data_component.update(current_date=current_date)
  ↓
observations = observation_component.update(data=data)
  ↓
model_component.fit(observations=observations)
model_output = model_component.predict(observations=observations)
  ↓
analysis = analysis_component.update(
    data=data,
    observations=observations,
    model_output=model_output
)
  ↓
report = report_component.update(
    data=data,
    observations=observations,
    model_output=model_output,
    analysis=analysis
)
```

## Best Practices

### 1. Use Factories for Standard Pipelines

```python
# Recommended: Use factory functions
pipeline = hr.create_financial_pipeline('AAPL', n_states=3)

# Rather than: Manual construction (unless customizing)
```

### 2. Enable Temporal Isolation for Backtesting

```python
# Always use TemporalController for strategy validation
controller = TemporalController(pipeline, full_data)
results = controller.step_through_time(start, end)

# Never: Train on full dataset then test on subset
# This causes look-ahead bias!
```

### 3. Store Component Outputs

```python
result = pipeline.update()

# Access intermediate results for debugging
if pipeline.component_outputs['model_output']['confidence'] < 0.5:
    print("Low confidence - investigate")
```

### 4. Implement Custom Components Carefully

```python
class MyComponent(PipelineComponent):
    def update(self, **kwargs):
        # Validate required inputs
        if 'data' not in kwargs:
            raise ValueError("Data required")

        # Your logic here
        return result

    def plot(self, **kwargs):
        # Always implement plot() even if simple
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Not implemented')
        return fig
```

## Module Structure

```
pipeline/
├── __init__.py         # Public API exports
├── core.py             # Pipeline class
├── interfaces.py       # Component interfaces
└── temporal.py         # TemporalController and TemporalDataStub
```

## Related Modules

- **[factories](../factories/README.md)**: Factory functions for creating pre-configured pipelines
- **[data](../data/README.md)**: Data component implementations
- **[observations](../observations/README.md)**: Observation component implementations
- **[models](../models/README.md)**: Model component implementations
- **[analysis](../analysis/README.md)**: Analysis component implementations
- **[reports](../reports/README.md)**: Report component implementations

## Key Advantages

1. **Modularity**: Swap components without changing pipeline logic
2. **Testability**: Test each component in isolation
3. **Reusability**: Create libraries of reusable components
4. **Temporal Safety**: Built-in prevention of look-ahead bias
5. **Consistency**: All components follow same interface pattern

---

For complete examples, see `examples/` directory in the project root.