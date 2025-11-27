"""
Local backtesting engine for regime-based trading strategies.

This module provides a standalone backtesting framework that:
- Pulls historical data from QuantConnect API (requires PRO account)
- Detects market regimes locally using Hidden Markov Models
- Simulates portfolio rebalancing based on detected regimes
- Generates comprehensive performance metrics

Key classes:
- LocalBacktester: Main backtesting engine

Example:
    from hidden_regime.local import LocalBacktester

    backtester = LocalBacktester(ticker="SPY")
    results = backtester.run(
        start_date="2020-01-01",
        end_date="2023-12-31",
        n_states=3,
        allocations={"Bull": 1.0, "Bear": 0.0, "Sideways": 0.5}
    )
    backtester.export_results()
"""

from hidden_regime.local.backtester import LocalBacktester

__all__ = ["LocalBacktester"]
