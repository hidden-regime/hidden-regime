"""
Trading signal types and definitions.

This module defines the unified output format for trading signals generated from
regime interpretation and strategy execution.

TradingSignal is the contract between:
- Interpreter (produces RegimeLabel)
- Strategy (consumes RegimeLabel, produces TradingSignal)
- QuantConnect (consumes TradingSignal)
- User code (reads TradingSignal)
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Literal

from hidden_regime.interpreter.regime_types import RegimeLabel


@dataclass
class TradingSignal:
    """
    Complete trading signal output from strategy execution.

    This is the unified format for trading decisions. By standardizing the output,
    we enable:
    - Easy A/B testing of different strategies
    - Clear audit trail (which regime + which strategy = which signal)
    - Type-safe integration with QuantConnect and other systems
    - No column naming errors or ambiguity

    Immutable to prevent accidental modification after creation.
    """

    # Trading direction
    direction: Literal["long", "short", "neutral", "cash"]

    # Position sizing
    position_size: float  # 0-1 (or higher for leverage)

    # Confidence in this signal
    confidence: float  # 0-1

    # Regime that generated this signal (for transparency and audit)
    regime_label: RegimeLabel

    # Strategy configuration that was used (for audit trail)
    strategy_name: str

    # Risk management constraints applied
    risk_management_applied: Dict[str, Any] = field(default_factory=dict)

    # Is this signal actionable?
    valid: bool = True

    # Additional metadata (timestamp, index, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_direction(self, new_direction: Literal["long", "short", "neutral", "cash"]) -> "TradingSignal":
        """Create a new signal with adjusted direction."""
        return replace(self, direction=new_direction)

    def with_adjusted_position(self, new_position_size: float) -> "TradingSignal":
        """Create a new signal with adjusted position size."""
        return replace(self, position_size=new_position_size)

    def with_confidence(self, new_confidence: float) -> "TradingSignal":
        """Create a new signal with adjusted confidence."""
        return replace(self, confidence=new_confidence)

    def with_risk_constraint(self, constraint_name: str, constraint_value: Any) -> "TradingSignal":
        """Create a new signal with an additional risk management constraint."""
        updated_constraints = self.risk_management_applied.copy()
        updated_constraints[constraint_name] = constraint_value
        return replace(self, risk_management_applied=updated_constraints)

    def mark_invalid(self, reason: str = "Unknown") -> "TradingSignal":
        """Mark this signal as invalid with a reason."""
        updated_metadata = self.metadata.copy()
        updated_metadata["invalid_reason"] = reason
        return replace(self, valid=False, metadata=updated_metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for DataFrame conversion."""
        return {
            "direction": self.direction,
            "position_size": self.position_size,
            "confidence": self.confidence,
            "regime_name": self.regime_label.name,
            "regime_type": self.regime_label.regime_type.value,
            "strategy_name": self.strategy_name,
            "signal_valid": self.valid,
            "risk_management": str(self.risk_management_applied),
        }
