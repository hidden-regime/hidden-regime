# QuantConnect LEAN Integration Roadmap

**Project:** Hidden-Regime × QuantConnect LEAN Integration
**Goal:** 5-minute backtest turnaround from design to running algorithm
**Target:** Top-performing algorithm on QuantConnect

---

## Executive Summary

Transform hidden-regime into a seamless QuantConnect LEAN integration achieving **5-minute backtest turnaround** from design to running algorithm. This positions hidden-regime as a competitive advantage for QuantConnect algorithms targeting top performance.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  QuantConnect LEAN Container                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         Custom Python Algorithm (User Code)            │  │
│  │                                                         │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │      HiddenRegimeAlgorithm (Adapter)             │  │  │
│  │  │  - QCAlgorithm base class                        │  │  │
│  │  │  - Regime detection on Initialize/OnData         │  │  │
│  │  │  - Signal generation → Trading logic             │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                         ↓                               │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │      Hidden-Regime Core                          │  │  │
│  │  │  - Pipeline architecture                         │  │  │
│  │  │  - HMM regime detection                          │  │  │
│  │  │  - Financial analysis                            │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  QuantConnect Data → hidden-regime → Trading Signals → LEAN  │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Integration Components

### 1.1 QuantConnect Adapter Package (`hidden_regime/quantconnect/`)

**Structure:**
```
hidden_regime/quantconnect/
├── __init__.py
├── algorithm.py          # HiddenRegimeAlgorithm base class
├── data_adapter.py       # QC data → hidden-regime format
├── signal_adapter.py     # Regime signals → QC trades
├── indicators.py         # Custom QC indicators using HMM
├── alpha_model.py        # QCAlgorithm Alpha model integration
└── universe_selection.py # Regime-based universe filtering
```

**Key Classes:**

```python
# algorithm.py
class HiddenRegimeAlgorithm(QCAlgorithm):
    """Base algorithm with integrated regime detection."""

    def __init__(self):
        self.regime_pipeline = None
        self.current_regime = None
        self.regime_confidence = 0.0

    def initialize_regime_detection(self, ticker, n_states=3, **kwargs):
        """Setup regime pipeline - called in Initialize()."""
        pass

    def update_regime(self):
        """Update regime state - called in OnData()."""
        pass

    def on_regime_change(self, old_regime, new_regime, confidence):
        """Override this method for regime change logic."""
        pass
```

### 1.2 Data Pipeline Integration

**Challenge:** QuantConnect provides real-time bars, hidden-regime expects historical DataFrames

**Solution:**
```python
class QuantConnectDataAdapter:
    """Converts QC RollingWindow to pandas DataFrame."""

    def __init__(self, window_size=252):  # 1 year of daily data
        self.price_window = RollingWindow[TradeBar](window_size)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert rolling window to DataFrame for hidden-regime."""
        return pd.DataFrame({
            'Date': [bar.Time for bar in self.price_window],
            'Close': [bar.Close for bar in self.price_window],
            'Open': [bar.Open for bar in self.price_window],
            'High': [bar.High for bar in self.price_window],
            'Low': [bar.Low for bar in self.price_window],
            'Volume': [bar.Volume for bar in self.price_window]
        }).set_index('Date').sort_index()
```

---

## Phase 2: Docker Infrastructure

### 2.1 Custom Dockerfile

```dockerfile
# Dockerfile.hidden-regime
FROM quantconnect/lean:latest

# Install hidden-regime and dependencies
RUN pip install --no-cache-dir \
    hidden-regime \
    numpy>=2.3.2 \
    pandas>=2.3.2 \
    scipy>=1.7.0 \
    scikit-learn>=1.0.0

# Pre-compile Python modules for faster startup
RUN python -c "import hidden_regime; hidden_regime.create_simple_regime_pipeline('SPY')"

# Set working directory
WORKDIR /Lean/Algorithm.Python

# Label for identification
LABEL description="QuantConnect LEAN with Hidden-Regime integration"
LABEL version="1.0.0"
```

### 2.2 Build Script

```bash
#!/bin/bash
# build_lean_hidden_regime.sh

# Build custom LEAN image with hidden-regime
docker build -t quantconnect/lean:hidden-regime -f Dockerfile.hidden-regime .

# Configure LEAN CLI to use custom image
lean config set engine-image quantconnect/lean:hidden-regime

echo "✓ Custom LEAN image ready with hidden-regime"
```

---

## Phase 3: Template Algorithms

### 3.1 Template 1: Basic Regime Switching

```python
# templates/basic_regime_switching.py
from AlgorithmImports import *
from hidden_regime.quantconnect import HiddenRegimeAlgorithm

class BasicRegimeSwitching(HiddenRegimeAlgorithm):
    """
    Simple regime-based strategy:
    - Bull regime: 100% long
    - Bear regime: 100% cash
    - Sideways: 50% long
    """

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Add SPY
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Initialize regime detection (3-state HMM)
        self.initialize_regime_detection(
            ticker="SPY",
            n_states=3,
            lookback_days=252
        )

    def OnData(self, data):
        if not data.ContainsKey(self.symbol):
            return

        # Update regime
        self.update_regime()

        # Trading logic based on regime
        if self.current_regime == "Bull":
            self.SetHoldings(self.symbol, 1.0)
        elif self.current_regime == "Bear":
            self.Liquidate(self.symbol)
        else:  # Sideways
            self.SetHoldings(self.symbol, 0.5)

    def on_regime_change(self, old_regime, new_regime, confidence):
        self.Log(f"Regime change: {old_regime} → {new_regime} (confidence: {confidence:.1%})")
```

### 3.2 Template 2: Multi-Asset Regime Rotation

```python
# templates/multi_asset_rotation.py
class MultiAssetRegimeRotation(HiddenRegimeAlgorithm):
    """
    Rotate between assets based on individual regime strength:
    - Allocate to assets in Bull regimes
    - Reduce exposure to Bear regimes
    - Bonds during high volatility regimes
    """

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        # Asset universe
        self.assets = {
            "SPY": self.AddEquity("SPY", Resolution.Daily).Symbol,  # Stocks
            "QQQ": self.AddEquity("QQQ", Resolution.Daily).Symbol,  # Tech
            "TLT": self.AddEquity("TLT", Resolution.Daily).Symbol,  # Bonds
            "GLD": self.AddEquity("GLD", Resolution.Daily).Symbol,  # Gold
        }

        # Initialize regime detection for each asset
        self.regimes = {}
        for ticker in self.assets.keys():
            self.initialize_regime_detection(
                ticker=ticker,
                n_states=4,
                lookback_days=180
            )

    def OnData(self, data):
        # Update all regimes
        for ticker in self.assets.keys():
            self.update_regime(ticker)

        # Calculate allocation based on regime scores
        allocations = self.calculate_regime_allocations()

        # Rebalance portfolio
        for ticker, weight in allocations.items():
            self.SetHoldings(self.assets[ticker], weight)
```

### 3.3 Template 3: Alpha Model Integration

```python
# templates/regime_alpha_model.py
from hidden_regime.quantconnect.alpha_model import HiddenRegimeAlphaModel

class RegimeAlphaStrategy(QCAlgorithm):
    """
    Uses regime detection as an Alpha model in QC Framework.
    """

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        # Framework setup
        self.SetUniverseSelection(ManualUniverseSelectionModel([
            Symbol.Create("SPY", SecurityType.Equity, Market.USA)
        ]))

        # Use Hidden-Regime as Alpha Model
        self.SetAlpha(HiddenRegimeAlphaModel(
            n_states=3,
            lookback_days=252,
            confidence_threshold=0.7
        ))

        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.05))
```

---

## Phase 4: Quick Start Workflow (< 5 Minutes)

### 4.1 One-Command Setup

```bash
# setup_quantconnect.sh
#!/bin/bash

echo "🚀 Setting up hidden-regime for QuantConnect LEAN..."

# 1. Install LEAN CLI
dotnet tool install -g QuantConnect.Lean.CLI

# 2. Build custom Docker image
docker build -t quantconnect/lean:hidden-regime -f docker/Dockerfile.hidden-regime .

# 3. Configure LEAN
lean config set engine-image quantconnect/lean:hidden-regime

# 4. Create project from template
lean project-create "MyRegimeStrategy" --language python

# 5. Copy template algorithm
cp templates/basic_regime_switching.py MyRegimeStrategy/main.py

echo "✓ Setup complete! Ready to backtest."
echo ""
echo "Quick start commands:"
echo "  cd MyRegimeStrategy"
echo "  lean backtest MyRegimeStrategy"
```

### 4.2 Configuration File (`regime_config.json`)

```json
{
  "regime_detection": {
    "n_states": 3,
    "lookback_days": 252,
    "initialization_method": "kmeans",
    "confidence_threshold": 0.65
  },
  "trading_rules": {
    "bull_allocation": 1.0,
    "bear_allocation": 0.0,
    "sideways_allocation": 0.5,
    "rebalance_threshold": 0.1
  },
  "risk_management": {
    "max_drawdown": 0.15,
    "position_sizing": "equal_weight",
    "stop_loss": null
  }
}
```

### 4.3 CLI Helper

```python
# cli/quick_backtest.py
import click
import json

@click.command()
@click.option('--ticker', default='SPY', help='Asset to trade')
@click.option('--states', default=3, help='Number of regime states')
@click.option('--start', default='2020-01-01', help='Start date')
@click.option('--end', default='2024-01-01', help='End date')
def quick_backtest(ticker, states, start, end):
    """
    Generate and run a regime strategy in under 5 minutes.

    Example:
        python quick_backtest.py --ticker SPY --states 3
    """
    # Generate algorithm from template
    template = generate_algorithm(ticker, states, start, end)

    # Write to project
    with open('main.py', 'w') as f:
        f.write(template)

    # Run backtest
    click.echo("🚀 Running backtest...")
    os.system("lean backtest .")

    click.echo("✓ Backtest complete!")
```

---

## Phase 5: Performance Optimizations

### 5.1 Caching Strategy

```python
class CachedRegimeDetection:
    """Cache regime computations to avoid re-training HMM every bar."""

    def __init__(self, retrain_frequency='weekly'):
        self.last_train_date = None
        self.cached_model = None
        self.retrain_frequency = retrain_frequency

    def should_retrain(self, current_date):
        if self.last_train_date is None:
            return True
        if self.retrain_frequency == 'daily':
            return True
        elif self.retrain_frequency == 'weekly':
            return (current_date - self.last_train_date).days >= 7
        elif self.retrain_frequency == 'monthly':
            return (current_date - self.last_train_date).days >= 30
        return False
```

### 5.2 Warm-Up Period

```python
def Initialize(self):
    # Set warm-up period for regime detection
    self.SetWarmUp(timedelta(days=252))  # 1 year for HMM training

    # During warm-up, just collect data
    self.is_warming_up = True
```

---

## Phase 6: Testing & Validation

### 6.1 Integration Tests

```python
# tests/test_quantconnect_integration.py
def test_basic_regime_algorithm():
    """Test basic regime switching algorithm."""
    algorithm = BasicRegimeSwitching()
    algorithm.SetStartDate(2020, 1, 1)
    algorithm.SetEndDate(2020, 12, 31)

    # Run backtest
    results = algorithm.Run()

    assert results.TotalPerformance.PortfolioStatistics.SharpeRatio > 0
    assert results.TotalPerformance.PortfolioStatistics.CompoundingAnnualReturn > 0
```

### 6.2 Benchmark Comparison

Compare hidden-regime strategies against:
- Buy & Hold SPY
- Equal-weight multi-asset
- Traditional momentum strategies

---

## Phase 7: Documentation & Examples

### 7.1 Quick Start Guide

```markdown
# Hidden-Regime × QuantConnect LEAN Quick Start

## 5-Minute Backtest Workflow

1. **Install & Setup** (1 min)
   ```bash
   bash setup_quantconnect.sh
   ```

2. **Create Strategy** (2 min)
   ```bash
   lean project-create "MyStrategy"
   cd MyStrategy
   cp ../templates/basic_regime_switching.py main.py
   ```

3. **Configure** (1 min)
   Edit `regime_config.json` with your parameters

4. **Run Backtest** (1 min)
   ```bash
   lean backtest .
   ```

5. **Done!** View results in `backtests/` directory
```

---

## Implementation Timeline

| **Phase** | **Deliverable** | **Effort** | **Status** |
|-----------|----------------|------------|------------|
| 1 | QuantConnect adapter package | 3-4 days | 🚧 In Progress |
| 2 | Docker integration | 1 day | ⏳ Pending |
| 3 | Template algorithms (3 templates) | 2 days | ⏳ Pending |
| 4 | Quick-start tooling & CLI | 2 days | ⏳ Pending |
| 5 | Performance optimizations | 2 days | ⏳ Pending |
| 6 | Testing & validation | 2 days | ⏳ Pending |
| 7 | Documentation | 1-2 days | ⏳ Pending |
| **Total** | **Complete LEAN integration** | **~2 weeks** | |

---

## Success Metrics

1. **Time to Backtest**: < 5 minutes from idea to running algorithm
2. **Performance**: Regime strategies outperform buy-and-hold baseline
3. **Flexibility**: Support 3+ strategy templates out-of-box
4. **Reliability**: 95%+ test coverage for adapter code
5. **Adoption**: Path to top QuantConnect algorithm performance

---

## Key Design Decisions

### Data Flow

1. **QuantConnect → Hidden-Regime**: Use rolling windows converted to DataFrames
2. **Hidden-Regime → QuantConnect**: Regime signals as trading instructions
3. **State Management**: Cache trained models, retrain periodically
4. **Warm-Up**: Use QC warm-up period to collect initial data for HMM training

### Architecture Principles

1. **Minimal Invasiveness**: Don't modify core hidden-regime, use adapters
2. **Idiomatic QC Code**: Follow QCAlgorithm patterns and conventions
3. **Performance First**: Cache aggressively, minimize recomputation
4. **Extensibility**: Support both basic usage and advanced QC Framework

### Integration Points

1. **Algorithm Base Class**: `HiddenRegimeAlgorithm` extends `QCAlgorithm`
2. **Alpha Model**: `HiddenRegimeAlphaModel` for QC Framework users
3. **Custom Indicators**: Native QC indicators wrapping HMM logic
4. **Universe Selection**: Regime-based security filtering

---

## Phase 1 Detailed Implementation Plan

### File Structure

```
hidden_regime/quantconnect/
├── __init__.py              # Package exports
├── algorithm.py             # HiddenRegimeAlgorithm base class
├── data_adapter.py          # QuantConnectDataAdapter
├── signal_adapter.py        # RegimeSignalAdapter
├── indicators.py            # RegimeIndicator, RegimeConfidenceIndicator
├── alpha_model.py           # HiddenRegimeAlphaModel
├── universe_selection.py    # RegimeBasedUniverseSelection
└── config.py                # QC-specific configuration
```

### Implementation Order

1. **data_adapter.py** - Convert QC data to hidden-regime format
2. **algorithm.py** - Base algorithm class with regime integration
3. **signal_adapter.py** - Regime → trading signal conversion
4. **indicators.py** - QC custom indicators
5. **alpha_model.py** - QC Framework alpha model
6. **universe_selection.py** - Universe filtering by regime

---

## Next Steps

- [x] Create roadmap document
- [ ] Create package structure
- [ ] Implement data adapter
- [ ] Implement base algorithm class
- [ ] Create first working example
- [ ] Test with local LEAN installation

---

**Project Start Date:** 2025-11-17
**Target Completion:** 2 weeks from start
**Current Phase:** Phase 1 - Core Integration Components
