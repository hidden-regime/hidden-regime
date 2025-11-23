"""Interpreter component for Hidden Regime.

The Interpreter component adds domain knowledge to model outputs.
It maps model state indices to semantic regime labels and characteristics.

Key Principle: ALL financial domain knowledge belongs in the Interpreter.
- State indices (0,1,2) → regime labels (Bear, Bull, Sideways)
- Regime characteristics (returns, volatility, duration)
- Consistent colors and naming for visualization

This module now contains ALL financial domain knowledge:
- RegimeType enum and RegimeProfile dataclass
- FinancialInterpreter with comprehensive regime characterization
- All financial metrics and trading statistics
"""

from hidden_regime.interpreter.base import BaseInterpreter
from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.interpreter.regime_types import (
    REGIME_TYPE_COLORS,
    RegimeProfile,
    RegimeType,
)

__all__ = [
    "BaseInterpreter",
    "FinancialInterpreter",
    "RegimeType",
    "RegimeProfile",
    "REGIME_TYPE_COLORS",
]
