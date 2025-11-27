"""Interpreter component for Hidden Regime.

The Interpreter component adds domain knowledge to model outputs.
It maps model state indices to semantic regime labels and characteristics.

Key Principle: ALL financial domain knowledge belongs in the Interpreter.
- State indices (0,1,2) → regime labels (Bear, Bull, Sideways)
- Regime characteristics (returns, volatility, duration)
- Trading semantics (bias, position direction, confidence thresholds)
- Consistent colors and naming for visualization

This module contains:
- RegimeType enum (backward compatibility and color lookups)
- RegimeLabel dataclass (new: single source of truth)
- RegimeCharacteristics, TradingSemantics (components of RegimeLabel)
- FinancialInterpreter with comprehensive regime characterization
- RegimeLabelInterpreter (wraps FinancialInterpreter to produce RegimeLabel objects)
"""

from hidden_regime.interpreter.base import BaseInterpreter
from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.interpreter.regime_label_builder import RegimeLabelBuilder
from hidden_regime.interpreter.regime_label_interpreter import RegimeLabelInterpreter
from hidden_regime.interpreter.regime_types import (
    REGIME_TYPE_COLORS,
    RegimeCharacteristics,
    RegimeLabel,
    RegimeType,
    TradingSemantics,
)

__all__ = [
    "BaseInterpreter",
    "FinancialInterpreter",
    "RegimeLabelInterpreter",
    "RegimeType",
    "RegimeLabel",
    "RegimeCharacteristics",
    "TradingSemantics",
    "RegimeLabelBuilder",
    "REGIME_TYPE_COLORS",
]
