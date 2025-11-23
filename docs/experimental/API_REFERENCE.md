# Hidden-Regime × QuantConnect API Reference

Complete API documentation for hidden-regime QuantConnect LEAN integration.

**Version:** 1.0.0
**Last Updated:** 2025-11-17

---

## Table of Contents

1. [Core Classes](#core-classes)
2. [Data Adapters](#data-adapters)
3. [Signal Adapters](#signal-adapters)
4. [Performance Components](#performance-components)
5. [Configuration](#configuration)
6. [Indicators](#indicators)
7. [Alpha Models](#alpha-models)

---

## Core Classes

### HiddenRegimeAlgorithm

Base algorithm class that extends `QCAlgorithm` with regime detection capabilities.

**Location:** `hidden_regime.quantconnect.algorithm`

#### Class Definition

```python
class HiddenRegimeAlgorithm(QCAlgorithm):
    """
    Base algorithm class with integrated Hidden Markov Model regime detection.

    Extends QuantConnect's QCAlgorithm to provide seamless regime detection
    and trading signal generation.
    """
```

#### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `current_regime` | str | Current detected regime (e.g., "Bull", "Bear") |
| `regime_confidence` | float | Confidence in current regime (0.0-1.0) |
| `regime_history` | list | Historical regime states |
| `pipelines` | dict | Regime detection pipelines by ticker |
| `data_adapters` | dict | Data adapters by ticker |

#### Methods

##### `initialize_regime_detection()`

Initialize regime detection for a ticker.

**Signature:**
```python
def initialize_regime_detection(
    self,
    ticker: str,
    n_states: int = 3,
    lookback_days: int = 252,
    regime_allocations: Dict[str, float] = None,
    **kwargs
) -> None:
```

**Parameters:**
- `ticker` (str): Asset ticker symbol
- `n_states` (int, default=3): Number of regime states
- `lookback_days` (int, default=252): Historical data window
- `regime_allocations` (dict, optional): Allocation map by regime
- `**kwargs`: Additional configuration options

**Example:**
```python
def Initialize(self):
    self.initialize_regime_detection(
        ticker="SPY",
        n_states=3,
        lookback_days=252,
        regime_allocations={
            "Bull": 1.0,
            "Bear": 0.0,
            "Sideways": 0.5
        }
    )
```

##### `update_regime()`

Update regime detection with latest data.

**Signature:**
```python
def update_regime(self, ticker: str = None) -> Dict[str, Any]:
```

**Parameters:**
- `ticker` (str, optional): Ticker to update. If None, updates all.

**Returns:**
- dict: Regime update result containing:
  - `regime` (str): Current regime
  - `confidence` (float): Confidence level
  - `probabilities` (array): State probabilities

**Example:**
```python
def OnData(self, data):
    result = self.update_regime("SPY")
    self.Log(f"Regime: {result['regime']}, Confidence: {result['confidence']:.2%}")
```

##### `on_regime_change()`

Callback triggered when regime changes.

**Signature:**
```python
def on_regime_change(
    self,
    old_regime: str,
    new_regime: str,
    confidence: float,
    ticker: str = None
) -> None:
```

**Parameters:**
- `old_regime` (str): Previous regime
- `new_regime` (str): New regime
- `confidence` (float): Confidence in new regime
- `ticker` (str, optional): Ticker that changed

**Example:**
```python
def on_regime_change(self, old_regime, new_regime, confidence, ticker=None):
    self.Log(f"Regime changed: {old_regime} → {new_regime} ({confidence:.1%})")

    if new_regime == "Crisis":
        self.Liquidate()  # Go to cash in crisis
```

---

### HiddenRegimeAlgorithmOptimized

Optimized version with caching, profiling, and batch updates.

**Location:** `hidden_regime.quantconnect.optimized_algorithm`

#### Class Definition

```python
class HiddenRegimeAlgorithmOptimized(HiddenRegimeAlgorithm):
    """
    Optimized algorithm with performance enhancements.

    Provides 60-70% performance improvement through:
    - Model caching (70-90% hit rate)
    - Batch parallel updates
    - Performance profiling
    """
```

#### Additional Methods

##### `enable_caching()`

Enable model caching for performance.

**Signature:**
```python
def enable_caching(
    self,
    max_cache_size: int = 100,
    retrain_frequency: str = "weekly"
) -> None:
```

**Parameters:**
- `max_cache_size` (int, default=100): Maximum cached models
- `retrain_frequency` (str, default="weekly"): Retraining frequency
  - Options: "daily", "weekly", "monthly"

**Example:**
```python
def Initialize(self):
    self.enable_caching(max_cache_size=200, retrain_frequency="monthly")
```

##### `enable_batch_updates()`

Enable parallel batch updates for multi-asset strategies.

**Signature:**
```python
def enable_batch_updates(
    self,
    max_workers: int = 4,
    use_parallel: bool = True
) -> None:
```

**Parameters:**
- `max_workers` (int, default=4): Number of parallel workers
- `use_parallel` (bool, default=True): Enable parallel processing

**Example:**
```python
def Initialize(self):
    self.enable_batch_updates(max_workers=4, use_parallel=True)
```

##### `enable_profiling()`

Enable performance profiling.

**Signature:**
```python
def enable_profiling(self) -> None:
```

**Example:**
```python
def Initialize(self):
    self.enable_profiling()

def OnEndOfAlgorithm(self):
    stats = self.get_profiling_stats()
    self.Log(f"Performance stats: {stats}")
```

---

## Data Adapters

### QuantConnectDataAdapter

Converts QuantConnect TradeBar data to pandas DataFrames.

**Location:** `hidden_regime.quantconnect.data_adapter`

#### Class Definition

```python
class QuantConnectDataAdapter:
    """
    Adapter for converting QC TradeBar objects to pandas DataFrames.

    Maintains a rolling window of bars and provides efficient conversion
    to the format required by hidden-regime.
    """
```

#### Constructor

```python
def __init__(self, window_size: int = 252):
    """
    Args:
        window_size: Number of bars to maintain in rolling window
    """
```

#### Methods

##### `add_bar()`

Add a TradeBar to the rolling window.

**Signature:**
```python
def add_bar(self, bar: TradeBar) -> None:
```

**Parameters:**
- `bar` (TradeBar): QuantConnect TradeBar object

**Example:**
```python
def OnData(self, data):
    if data.ContainsKey(self.symbol):
        self.data_adapter.add_bar(data[self.symbol])
```

##### `to_dataframe()`

Convert rolling window to pandas DataFrame.

**Signature:**
```python
def to_dataframe(self) -> pd.DataFrame:
```

**Returns:**
- DataFrame with columns: Open, High, Low, Close, Volume
- Index: DatetimeIndex

**Example:**
```python
df = self.data_adapter.to_dataframe()
# df is ready for regime detection pipeline
```

---

### RollingWindowDataAdapter

Converts QC RollingWindow to DataFrame.

**Location:** `hidden_regime.quantconnect.data_adapter`

#### Methods

##### `from_rolling_window()`

Convert RollingWindow to DataFrame.

**Signature:**
```python
def from_rolling_window(self, window: RollingWindow[TradeBar]) -> pd.DataFrame:
```

**Parameters:**
- `window` (RollingWindow[TradeBar]): QC RollingWindow object

**Returns:**
- DataFrame with OHLCV data

**Example:**
```python
adapter = RollingWindowDataAdapter(window_size=252)
df = adapter.from_rolling_window(self.price_window)
```

---

### HistoryDataAdapter

Converts QC History API results to DataFrame.

**Location:** `hidden_regime.quantconnect.data_adapter`

#### Methods

##### `from_history()`

Convert History API result to DataFrame.

**Signature:**
```python
def from_history(self, history_df: pd.DataFrame) -> pd.DataFrame:
```

**Parameters:**
- `history_df` (DataFrame): Result from self.History()

**Returns:**
- Cleaned DataFrame ready for analysis

**Example:**
```python
history = self.History(self.symbol, 252, Resolution.Daily)
adapter = HistoryDataAdapter()
df = adapter.from_history(history)
```

---

## Signal Adapters

### TradingSignal

Dataclass representing a trading signal.

**Location:** `hidden_regime.quantconnect.signal_adapter`

#### Definition

```python
@dataclass
class TradingSignal:
    """Trading signal with direction and allocation."""
    direction: str          # 'long', 'short', 'neutral', 'cash'
    strength: float = 1.0   # Signal strength (0.0-1.0)
    allocation: float = 1.0 # Position allocation (0.0-1.0+)
    confidence: float = 1.0 # Confidence level (0.0-1.0)
```

**Example:**
```python
signal = TradingSignal(
    direction='long',
    strength=0.8,
    allocation=1.0,
    confidence=0.9
)
```

---

### RegimeSignalAdapter

Converts regime states to trading signals.

**Location:** `hidden_regime.quantconnect.signal_adapter`

#### Constructor

```python
def __init__(
    self,
    regime_allocations: Dict[str, float],
    confidence_threshold: float = 0.6,
    use_dynamic_sizing: bool = False
):
    """
    Args:
        regime_allocations: Map of regime -> allocation
        confidence_threshold: Minimum confidence for full allocation
        use_dynamic_sizing: Adjust allocation based on confidence
    """
```

#### Methods

##### `regime_to_signal()`

Convert regime to trading signal.

**Signature:**
```python
def regime_to_signal(
    self,
    regime: str,
    confidence: float
) -> TradingSignal:
```

**Parameters:**
- `regime` (str): Current regime state
- `confidence` (float): Regime confidence

**Returns:**
- TradingSignal object

**Example:**
```python
adapter = RegimeSignalAdapter(
    regime_allocations={'Bull': 1.0, 'Bear': 0.0, 'Sideways': 0.5}
)

signal = adapter.regime_to_signal('Bull', confidence=0.85)
# signal.direction = 'long', signal.allocation = 1.0
```

---

### MultiAssetSignalAdapter

Portfolio allocation across multiple assets.

**Location:** `hidden_regime.quantconnect.signal_adapter`

#### Constructor

```python
def __init__(
    self,
    assets: List[str],
    allocation_method: str = 'equal_weight',
    rebalance_threshold: float = 0.05,
    defensive_assets: List[str] = None
):
    """
    Args:
        assets: List of asset tickers
        allocation_method: 'equal_weight', 'confidence_weighted',
                          'regime_score', 'risk_parity'
        rebalance_threshold: Minimum change to trigger rebalance
        defensive_assets: Assets to favor in crisis
    """
```

#### Methods

##### `calculate_allocations()`

Calculate portfolio allocations.

**Signature:**
```python
def calculate_allocations(
    self,
    regime_signals: Dict[str, TradingSignal],
    volatilities: Dict[str, float] = None
) -> Dict[str, float]:
```

**Parameters:**
- `regime_signals` (dict): Signal for each asset
- `volatilities` (dict, optional): Asset volatilities for risk parity

**Returns:**
- dict: Asset -> allocation (sums to ~1.0)

**Example:**
```python
adapter = MultiAssetSignalAdapter(
    assets=['SPY', 'QQQ', 'TLT', 'GLD'],
    allocation_method='confidence_weighted'
)

regime_signals = {
    'SPY': TradingSignal('long', allocation=1.0, confidence=0.9),
    'QQQ': TradingSignal('long', allocation=1.0, confidence=0.7),
    'TLT': TradingSignal('long', allocation=1.0, confidence=0.6),
    'GLD': TradingSignal('cash', allocation=0.0, confidence=0.5)
}

allocations = adapter.calculate_allocations(regime_signals)
# {'SPY': 0.45, 'QQQ': 0.35, 'TLT': 0.20, 'GLD': 0.0}
```

---

## Performance Components

### RegimeModelCache

LRU cache for trained regime models.

**Location:** `hidden_regime.quantconnect.performance.caching`

#### Constructor

```python
def __init__(self, max_cache_size: int = 100):
    """
    Args:
        max_cache_size: Maximum number of cached models
    """
```

#### Methods

##### `get()`

Retrieve cached model.

**Signature:**
```python
def get(
    self,
    ticker: str,
    n_states: int,
    data: pd.DataFrame,
    config: Any
) -> Optional[Any]:
```

**Returns:**
- Cached model or None if miss

##### `set()`

Store model in cache.

**Signature:**
```python
def set(
    self,
    ticker: str,
    n_states: int,
    data: pd.DataFrame,
    config: Any,
    model: Any
) -> None:
```

##### `get_statistics()`

Get cache statistics.

**Signature:**
```python
def get_statistics(self) -> Dict[str, Any]:
```

**Returns:**
```python
{
    'hit_count': int,
    'miss_count': int,
    'hit_rate': float,
    'cache_size': int
}
```

**Example:**
```python
cache = RegimeModelCache(max_cache_size=100)

# Try to get cached model
model = cache.get('SPY', 3, data, config)

if model is None:
    # Cache miss - train new model
    model = train_model(data)
    cache.set('SPY', 3, data, config, model)

# Check cache performance
stats = cache.get_statistics()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

---

### PerformanceProfiler

Performance timing and profiling.

**Location:** `hidden_regime.quantconnect.performance.profiling`

#### Methods

##### `enable()`/`disable()`

Enable or disable profiling.

**Signature:**
```python
def enable(self) -> None:
def disable(self) -> None:
```

##### `time_operation()`

Time an operation (decorator or context manager).

**Signature:**
```python
@profiler.time_operation('operation_name')
def my_function():
    pass

# Or as context manager
with profiler.time_operation('operation_name'):
    # code to time
    pass
```

##### `get_statistics()`

Get timing statistics.

**Signature:**
```python
def get_statistics(
    self,
    operation_name: Optional[str] = None
) -> Dict[str, Any]:
```

**Returns:**
```python
{
    'mean_time': float,
    'median_time': float,
    'min_time': float,
    'max_time': float,
    'std_dev': float,
    'call_count': int
}
```

**Example:**
```python
profiler = PerformanceProfiler()
profiler.enable()

@profiler.time_operation('regime_update')
def update_regime():
    # expensive operation
    pass

# Run many times
for _ in range(100):
    update_regime()

# Get stats
stats = profiler.get_statistics('regime_update')
print(f"Average time: {stats['mean_time']:.4f}s")
```

---

### BatchRegimeUpdater

Parallel batch processing for multi-asset strategies.

**Location:** `hidden_regime.quantconnect.performance.batch_updates`

#### Constructor

```python
def __init__(
    self,
    max_workers: int = 4,
    use_parallel: bool = True
):
    """
    Args:
        max_workers: Number of parallel workers
        use_parallel: Enable parallel processing
    """
```

#### Methods

##### `batch_update()`

Update multiple assets in parallel.

**Signature:**
```python
def batch_update(
    self,
    assets: List[str],
    data_dict: Dict[str, pd.DataFrame],
    pipeline_dict: Dict[str, Any],
    update_func: Callable,
    continue_on_error: bool = False
) -> Dict[str, Any]:
```

**Parameters:**
- `assets`: List of assets to update
- `data_dict`: Data for each asset
- `pipeline_dict`: Pipeline for each asset
- `update_func`: Function to call for each asset
- `continue_on_error`: Continue if some assets fail

**Returns:**
- dict: Results for each asset

**Example:**
```python
updater = BatchRegimeUpdater(max_workers=4)

def update_asset_regime(asset, data, pipeline):
    result = pipeline.run(data)
    return result

results = updater.batch_update(
    assets=['SPY', 'QQQ', 'TLT', 'GLD'],
    data_dict=data_by_asset,
    pipeline_dict=pipelines,
    update_func=update_asset_regime
)
```

---

## Configuration

### QuantConnectConfig

Configuration dataclass for QC integration.

**Location:** `hidden_regime.quantconnect.config`

#### Definition

```python
@dataclass
class QuantConnectConfig:
    """Configuration for QuantConnect integration."""

    # Data settings
    lookback_days: int = 252
    warmup_days: int = 252

    # Regime detection
    n_states: int = 3
    retrain_frequency: str = "weekly"

    # Performance
    use_cache: bool = True
    cache_size: int = 100
    batch_updates: bool = True
    max_workers: int = 4
    enable_profiling: bool = False

    # Trading
    rebalance_threshold: float = 0.05
    confidence_threshold: float = 0.6
```

**Example:**
```python
config = QuantConnectConfig(
    lookback_days=180,
    n_states=4,
    retrain_frequency="monthly",
    max_workers=8
)
```

---

## Indicators

### RegimeIndicator

Custom QC indicator for current regime.

**Location:** `hidden_regime.quantconnect.indicators`

**Usage:**
```python
def Initialize(self):
    self.regime_indicator = RegimeIndicator(
        "SPY_Regime",
        n_states=3,
        lookback=252
    )

    self.RegisterIndicator(self.symbol, self.regime_indicator, Resolution.Daily)

def OnData(self, data):
    if self.regime_indicator.IsReady:
        regime = self.regime_indicator.Current.Value
        self.Log(f"Regime: {regime}")
```

---

### RegimeConfidenceIndicator

Indicator for regime confidence level.

**Location:** `hidden_regime.quantconnect.indicators`

**Usage:**
```python
self.confidence_indicator = RegimeConfidenceIndicator(
    "SPY_Confidence",
    n_states=3,
    lookback=252
)

# Use in trading logic
if self.confidence_indicator.Current.Value > 0.8:
    # High confidence - take full position
    self.SetHoldings(self.symbol, 1.0)
```

---

## Alpha Models

### HiddenRegimeAlphaModel

QC Framework Alpha Model using regime detection.

**Location:** `hidden_regime.quantconnect.alpha_model`

#### Constructor

```python
def __init__(
    self,
    n_states: int = 3,
    lookback_days: int = 252,
    confidence_threshold: float = 0.7,
    insight_duration: timedelta = timedelta(days=1)
):
```

**Usage:**
```python
def Initialize(self):
    # Framework setup
    self.SetAlpha(HiddenRegimeAlphaModel(
        n_states=3,
        lookback_days=252,
        confidence_threshold=0.7
    ))

    self.SetPortfolioConstruction(
        EqualWeightingPortfolioConstructionModel()
    )

    self.SetExecution(ImmediateExecutionModel())
```

---

## Complete Example

```python
from AlgorithmImports import *
from hidden_regime.quantconnect.optimized_algorithm import HiddenRegimeAlgorithmOptimized

class MyRegimeStrategy(HiddenRegimeAlgorithmOptimized):
    """Optimized multi-asset regime strategy."""

    def Initialize(self):
        # Basic setup
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        # Add securities
        self.assets = {
            'SPY': self.AddEquity('SPY', Resolution.Daily).Symbol,
            'QQQ': self.AddEquity('QQQ', Resolution.Daily).Symbol,
            'TLT': self.AddEquity('TLT', Resolution.Daily).Symbol,
            'GLD': self.AddEquity('GLD', Resolution.Daily).Symbol
        }

        # Enable optimizations
        self.enable_caching(max_cache_size=200, retrain_frequency='monthly')
        self.enable_batch_updates(max_workers=4)
        self.enable_profiling()

        # Initialize regime detection for each asset
        for ticker in self.assets.keys():
            self.initialize_regime_detection(
                ticker=ticker,
                n_states=3,
                lookback_days=252,
                regime_allocations={
                    'Bull': 1.0,
                    'Bear': 0.0,
                    'Sideways': 0.5
                }
            )

    def OnData(self, data):
        # Update regimes for all assets
        regime_results = {}
        for ticker in self.assets.keys():
            regime_results[ticker] = self.update_regime(ticker)

        # Calculate portfolio allocation
        allocations = self.calculate_portfolio_allocations(regime_results)

        # Execute trades
        for ticker, allocation in allocations.items():
            self.SetHoldings(self.assets[ticker], allocation)

    def on_regime_change(self, old_regime, new_regime, confidence, ticker=None):
        self.Log(f"{ticker}: {old_regime} → {new_regime} ({confidence:.1%})")

    def OnEndOfAlgorithm(self):
        # Print performance statistics
        stats = self.get_profiling_stats()
        self.Log(f"Performance statistics: {stats}")
```

---

## See Also

- [Quick Start Tutorial](QUICKSTART_TUTORIAL.md)
- [Best Practices](BEST_PRACTICES.md)
- [Template Usage Guide](TEMPLATE_USAGE_GUIDE.md)
- [Testing Guide](TESTING_GUIDE.md)
