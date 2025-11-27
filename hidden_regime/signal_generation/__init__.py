"""Signal Generation component for Hidden Regime.

The Signal Generator creates trading signals from Interpreter outputs.
It implements trading logic and position sizing strategies.

Key Principle: ALL trading logic belongs in the Signal Generator.
- Position signals (long/short/neutral)
- Strategy implementation
- Position sizing
- Entry/exit rules

The Interpreter provides regime interpretation. The Signal Generator provides trading logic.

New Architecture (v1.1+):
- SignalGenerator: Core generator that returns List[TradingSignal]
- StrategyBasedSignalGeneratorComponent: Pipeline-compatible wrapper
- Strategy objects: Encapsulate trading logic (RegimeFollowing, Contrarian, decorators)

Legacy Architecture (v1.0):
- BaseSignalGenerator / FinancialSignalGenerator: Configuration-based approach
"""

from hidden_regime.signal_generation.base import BaseSignalGenerator
from hidden_regime.signal_generation.financial import FinancialSignalGenerator
from hidden_regime.signal_generation.signal_generator import SignalGenerator
from hidden_regime.signal_generation.signal_generator_component import (
    StrategyBasedSignalGeneratorComponent,
)

__all__ = [
    # Legacy architecture
    "BaseSignalGenerator",
    "FinancialSignalGenerator",
    # New architecture
    "SignalGenerator",
    "StrategyBasedSignalGeneratorComponent",
]
