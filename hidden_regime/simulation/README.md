# Simulation Module

The simulation module provides a complete trading simulation framework for backtesting regime-based strategies with realistic capital tracking, risk management, and performance analytics.

## Overview

The simulation framework enables rigorous strategy backtesting with:

```
Signal Generation → Risk Management → Trade Execution → Performance Analytics
        ↓                  ↓                  ↓                  ↓
   HMM Regimes        Position Limits     Capital Tracking     Sharpe Ratio
   Indicators         Stop Losses         Trade Journal        Drawdowns
```

## Core Components

### TradingSimulationEngine

Main engine for executing trades and tracking capital.

```python
from hidden_regime.simulation import TradingSimulationEngine

# Create engine
engine = TradingSimulationEngine(
    initial_capital=100000,
    commission=0.001,  # 0.1% per trade
    slippage=0.0005    # 0.05% slippage
)

# Run simulation
results = engine.run(
    price_data=ohlcv_data,
    signals=trading_signals
)

print(f"Final Capital: ${results.final_capital:,.2f}")
print(f"Total Return: {results.total_return:.1%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
```

**Features:**
- Real capital tracking (not just returns)
- Trade-by-trade execution
- Commission and slippage modeling
- Long/short position support
- Position sizing

### Signal Generators

Generate trading signals from various inputs.

#### HMMSignalGenerator

Regime-based signal generation.

```python
from hidden_regime.simulation import HMMSignalGenerator

generator = HMMSignalGenerator(
    bullish_regimes=[2, 3],      # Bull and strong bull
    bearish_regimes=[0, 1],       # Crisis and bear
    neutral_regime=2,             # Sideways
    position_sizes={
        0: -0.5,  # Crisis: 50% short
        1: -0.2,  # Bear: 20% short
        2:  0.2,  # Sideways: 20% long
        3:  1.0   # Bull: 100% long
    }
)

# Generate signals from regimes
signals = generator.generate_signals(
    regimes=hmm_states,
    dates=dates
)
```

#### BuyHoldSignalGenerator

Benchmark buy-and-hold strategy.

```python
from hidden_regime.simulation import BuyHoldSignalGenerator

generator = BuyHoldSignalGenerator(position_size=1.0)

signals = generator.generate_signals(dates=dates)
# Always 100% long
```

#### TechnicalIndicatorSignalGenerator

Signals from technical indicators.

```python
from hidden_regime.simulation.signal_generators import TechnicalIndicatorSignalGenerator

generator = TechnicalIndicatorSignalGenerator(
    indicator='RSI',
    buy_threshold=30,   # Oversold
    sell_threshold=70,  # Overbought
    position_size=1.0
)

signals = generator.generate_signals(
    indicator_data=rsi_values,
    dates=dates
)
```

### Risk Management

#### RiskManager

Enforce position limits and risk controls.

```python
from hidden_regime.simulation import RiskManager

risk_mgr = RiskManager(
    max_position_size=1.0,        # 100% max long
    max_leverage=2.0,              # 2x leverage max
    max_daily_loss=0.02,          # 2% max daily loss
    max_drawdown=0.20             # 20% max drawdown
)

# Apply risk limits to signal
adjusted_signal = risk_mgr.apply_limits(
    signal=raw_signal,
    current_position=0.5,
    current_drawdown=0.10
)
```

#### StopLossManager

Implement stop-loss and take-profit levels.

```python
from hidden_regime.simulation import StopLossManager

stop_mgr = StopLossManager(
    stop_loss_pct=0.05,          # 5% stop loss
    take_profit_pct=0.15,        # 15% take profit
    trailing_stop=True,          # Use trailing stop
    trailing_stop_pct=0.03       # 3% trailing
)

# Check stop conditions
should_exit = stop_mgr.check_stops(
    entry_price=100.0,
    current_price=95.5,
    highest_price=105.0
)
```

### Performance Analytics

#### PerformanceAnalyzer

Compute comprehensive performance metrics.

```python
from hidden_regime.simulation import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

metrics = analyzer.analyze(
    returns=strategy_returns,
    benchmark_returns=market_returns,
    risk_free_rate=0.02
)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
print(f"Win Rate: {metrics['win_rate']:.1%}")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
```

**Computed Metrics:**
- Return metrics (total, annualized, CAGR)
- Risk metrics (volatility, downside deviation, VaR)
- Risk-adjusted (Sharpe, Sortino, Calmar)
- Trade metrics (win rate, profit factor, avg win/loss)
- Drawdown analysis

#### TradeJournal

Track individual trades for analysis.

```python
from hidden_regime.simulation import TradeJournal

journal = TradeJournal()

# Record trade
journal.record_trade(
    entry_date='2023-01-15',
    exit_date='2023-01-20',
    entry_price=100.0,
    exit_price=105.0,
    shares=100,
    regime='Bull',
    signal_source='HMM'
)

# Analyze trades
trade_stats = journal.get_statistics()
print(f"Total Trades: {trade_stats['total_trades']}")
print(f"Win Rate: {trade_stats['win_rate']:.1%}")
print(f"Avg Win: {trade_stats['avg_win']:.2f}")
print(f"Avg Loss: {trade_stats['avg_loss']:.2f}")

# Filter trades by regime
bull_trades = journal.filter_by_regime('Bull')
```

## Simulation Configuration

### SimulationConfig

```python
from hidden_regime.config import SimulationConfig

config = SimulationConfig(
    # Capital settings
    initial_capital=100000,
    commission=0.001,         # 0.1% commission
    slippage=0.0005,          # 0.05% slippage

    # Risk management
    max_position_size=1.0,
    max_leverage=2.0,
    stop_loss_pct=0.05,
    take_profit_pct=0.15,

    # Signal strategies
    signal_strategies=['HMM', 'BuyHold'],

    # Technical indicators for comparison
    technical_indicators=['RSI', 'MACD'],

    # Analysis
    risk_free_rate=0.02,
    benchmark='SPY'
)
```

## Complete Simulation Workflow

### Step 1: Generate Signals

```python
from hidden_regime.simulation import HMMSignalGenerator

# Create signal generator
generator = HMMSignalGenerator(
    bullish_regimes=[2],
    bearish_regimes=[0],
    position_sizes={0: -0.5, 1: 0.0, 2: 1.0}
)

# Generate signals
signals = generator.generate_signals(
    regimes=hmm_states,
    dates=dates
)
```

### Step 2: Apply Risk Management

```python
from hidden_regime.simulation import RiskManager

# Create risk manager
risk_mgr = RiskManager(
    max_position_size=1.0,
    max_daily_loss=0.02
)

# Apply to signals
adjusted_signals = []
for signal in signals:
    adj_signal = risk_mgr.apply_limits(signal, ...)
    adjusted_signals.append(adj_signal)
```

### Step 3: Run Simulation

```python
from hidden_regime.simulation import TradingSimulationEngine

# Create engine
engine = TradingSimulationEngine(
    initial_capital=100000,
    commission=0.001
)

# Run simulation
results = engine.run(
    price_data=ohlcv_data,
    signals=adjusted_signals
)
```

### Step 4: Analyze Performance

```python
from hidden_regime.simulation import PerformanceAnalyzer

# Analyze results
analyzer = PerformanceAnalyzer()
metrics = analyzer.analyze(
    returns=results.returns,
    benchmark_returns=spy_returns
)

# View metrics
for name, value in metrics.items():
    print(f"{name}: {value}")
```

## Usage Examples

### Example 1: Basic Simulation

```python
from hidden_regime.simulation import (
    TradingSimulationEngine,
    HMMSignalGenerator
)

# Generate signals
generator = HMMSignalGenerator(bullish_regimes=[2])
signals = generator.generate_signals(regimes=hmm_states, dates=dates)

# Run simulation
engine = TradingSimulationEngine(initial_capital=100000)
results = engine.run(price_data=data, signals=signals)

print(f"Return: {results.total_return:.1%}")
print(f"Sharpe: {results.sharpe_ratio:.2f}")
```

### Example 2: Compare Strategies

```python
from hidden_regime.simulation import (
    HMMSignalGenerator,
    BuyHoldSignalGenerator,
    TradingSimulationEngine
)

# HMM strategy
hmm_gen = HMMSignalGenerator(bullish_regimes=[2])
hmm_signals = hmm_gen.generate_signals(regimes=hmm_states, dates=dates)

# Buy-hold strategy
bh_gen = BuyHoldSignalGenerator()
bh_signals = bh_gen.generate_signals(dates=dates)

# Simulate both
engine = TradingSimulationEngine(initial_capital=100000)

hmm_results = engine.run(data, hmm_signals)
bh_results = engine.run(data, bh_signals)

# Compare
print(f"HMM Sharpe: {hmm_results.sharpe_ratio:.2f}")
print(f"Buy-Hold Sharpe: {bh_results.sharpe_ratio:.2f}")
```

### Example 3: Risk-Managed Strategy

```python
from hidden_regime.simulation import (
    HMMSignalGenerator,
    RiskManager,
    StopLossManager,
    TradingSimulationEngine
)

# Generate signals
generator = HMMSignalGenerator(bullish_regimes=[2, 3])
signals = generator.generate_signals(regimes=hmm_states, dates=dates)

# Apply risk management
risk_mgr = RiskManager(max_position_size=0.8, max_daily_loss=0.02)
stop_mgr = StopLossManager(stop_loss_pct=0.05, trailing_stop=True)

# Process each signal
adjusted_signals = []
for signal, price in zip(signals, prices):
    # Apply risk limits
    adj_signal = risk_mgr.apply_limits(signal, ...)

    # Check stops
    if stop_mgr.check_stops(entry_price, price, ...):
        adj_signal = 0.0  # Exit position

    adjusted_signals.append(adj_signal)

# Run simulation
engine = TradingSimulationEngine(initial_capital=100000)
results = engine.run(data, adjusted_signals)
```

### Example 4: Comprehensive Backtest

```python
from hidden_regime.simulation.simulation_orchestrator import SimulationOrchestrator
from hidden_regime.config import SimulationConfig

# Configure simulation
config = SimulationConfig(
    initial_capital=100000,
    signal_strategies=['HMM', 'BuyHold'],
    technical_indicators=['RSI', 'MACD'],
    max_position_size=1.0,
    stop_loss_pct=0.05
)

# Create orchestrator
orchestrator = SimulationOrchestrator(config)

# Run comprehensive simulation
results = orchestrator.run_simulation(
    price_data=ohlcv_data,
    regime_data=regime_dataframe,
    symbol='AAPL'
)

# Access results
print(results.summary_statistics)
print(results.strategy_comparison)
```

## Best Practices

### 1. Use Realistic Transaction Costs

```python
# Include commission and slippage
engine = TradingSimulationEngine(
    initial_capital=100000,
    commission=0.001,   # 0.1% typical for retail
    slippage=0.0005    # 0.05% for liquid stocks
)
```

### 2. Implement Risk Management

```python
# Always include risk controls
risk_mgr = RiskManager(
    max_position_size=1.0,
    max_daily_loss=0.02,
    max_drawdown=0.20
)
```

### 3. Compare to Benchmark

```python
# Always compare to buy-and-hold
analyzer = PerformanceAnalyzer()
metrics = analyzer.analyze(
    returns=strategy_returns,
    benchmark_returns=spy_returns  # Include benchmark!
)
```

### 4. Keep Trade Journal

```python
# Track trades for analysis
journal = TradeJournal()

# Record each trade
for trade in trades:
    journal.record_trade(...)

# Analyze patterns
stats = journal.get_statistics()
regime_performance = journal.analyze_by_regime()
```

### 5. Use Temporal Isolation

```python
# For rigorous backtesting, use TemporalController
from hidden_regime.pipeline import TemporalController

controller = TemporalController(pipeline, full_data)
results = controller.step_through_time(start, end)
```

## Module Structure

```
simulation/
├── __init__.py                    # Public API
├── trading_engine.py              # TradingSimulationEngine
├── signal_generators.py           # Signal generation classes
├── risk_management.py             # RiskManager, StopLossManager
├── performance_analytics.py       # PerformanceAnalyzer, TradeJournal
└── simulation_orchestrator.py     # SimulationOrchestrator
```

## Related Modules

- **[analysis](../analysis/README.md)**: Analysis results (input to simulation)
- **[models](../models/README.md)**: HMM outputs (regime signals)
- **[pipeline](../pipeline/README.md)**: Temporal isolation for backtesting
- **[visualization](../visualization/README.md)**: Plotting simulation results

## Key Concepts

### Capital-Based vs. Return-Based

Hidden Regime uses **capital-based simulation**:

```python
# Capital-based (used here)
initial_capital = 100000
position_size = 0.5  # 50% of capital
shares = (initial_capital * position_size) / price
pnl = shares * (exit_price - entry_price)

# vs. Return-based (simpler but less realistic)
portfolio_return = position_size * asset_return
```

**Advantages:**
- More realistic
- Handles varying position sizes
- Tracks actual P&L
- Models real constraints

### Signal Timing

Proper timing is critical:

```python
# Correct timing
signal_generated_at_close_day_t = 0.8  # 80% long
position_entered_at_open_day_t_plus_1 = 0.8

# This prevents look-ahead bias
```

### Transaction Costs

Always include realistic costs:

| Cost Type | Typical Range | Impact |
|-----------|---------------|--------|
| **Commission** | 0.05-0.2% | Reduces returns |
| **Slippage** | 0.01-0.1% | Execution variance |
| **Spread** | 0.01-0.05% | Bid-ask cost |

---

For complete examples using simulation, see `examples/` directory in the project root.