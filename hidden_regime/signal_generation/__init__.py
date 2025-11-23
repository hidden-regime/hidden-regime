"""Signal Generation component for Hidden Regime.

The Signal Generator creates trading signals from Interpreter outputs.
It implements trading logic and position sizing strategies.

Key Principle: ALL trading logic belongs in the Signal Generator.
- Position signals (long/short/neutral)
- Strategy implementation
- Position sizing
- Entry/exit rules

The Interpreter provides regime interpretation. The Signal Generator provides trading logic.
"""

from hidden_regime.signal_generation.base import BaseSignalGenerator
from hidden_regime.signal_generation.financial import FinancialSignalGenerator

__all__ = ["BaseSignalGenerator", "FinancialSignalGenerator"]
