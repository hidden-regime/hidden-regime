"""
Signal generation module.

This package contains the signal generation system that transforms regime interpretation
into actionable trading signals.

Key components:
- TradingSignal: Unified output format for trading signals
- Strategy: Base class for trading strategies
- SignalGenerator: Orchestrates strategy execution
"""

from hidden_regime.signals.types import TradingSignal

__all__ = ["TradingSignal"]
