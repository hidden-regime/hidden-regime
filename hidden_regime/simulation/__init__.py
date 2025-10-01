"""
Trading simulation engine for backtesting strategies with real capital tracking.

This module provides comprehensive trading simulation capabilities including:
- Capital-based position tracking instead of percentage returns
- Risk management with stop-losses and position limits
- Signal-based trading with proper timing (sell at close, buy at open)
- Comprehensive performance analytics and trade journaling
"""

from .performance_analytics import PerformanceAnalyzer, TradeJournal
from .risk_management import RiskManager, StopLossManager
from .signal_generators import (
    BuyHoldSignalGenerator,
    HMMSignalGenerator,
    SignalGenerator,
)
from .trading_engine import TradingSimulationEngine

__all__ = [
    "TradingSimulationEngine",
    "SignalGenerator",
    "HMMSignalGenerator",
    "BuyHoldSignalGenerator",
    "RiskManager",
    "StopLossManager",
    "PerformanceAnalyzer",
    "TradeJournal",
]
