"""
New SignalGenerator implementation using Strategy objects.

This module provides the refactored SignalGenerator that:
1. Accepts Strategy objects (not configuration strings)
2. Consumes RegimeLabel objects from the Interpreter
3. Produces List[TradingSignal] objects (typed, structured output)
4. Supports optional DataFrame conversion for backward compatibility

This replaces the old BaseSignalGenerator/FinancialSignalGenerator architecture.
"""

from typing import List, Optional

import pandas as pd

from hidden_regime.interpreter.regime_types import RegimeLabel
from hidden_regime.signals.types import TradingSignal
from hidden_regime.strategies.base import RiskManagementConfig, Strategy


class SignalGenerator:
    """
    Generates trading signals using Strategy objects.

    This is a thin orchestrator that:
    1. Receives RegimeLabel objects from Interpreter
    2. Delegates to Strategy.get_signal_for_regime()
    3. Applies risk management constraints
    4. Returns List[TradingSignal]

    The actual trading logic lives in Strategy implementations (RegimeFollowing, etc.)
    and their decorators (ConfidenceWeighted, VolatilityAdjusted, etc.).

    Example:
        from hidden_regime.strategies import RegimeFollowingStrategy
        strategy = RegimeFollowingStrategy(
            long_confidence_threshold=0.65,
            short_confidence_threshold=0.65
        )
        generator = SignalGenerator(strategy)
        signals = generator.update(interpreter_output)
    """

    def __init__(self, strategy: Strategy):
        """Initialize signal generator with a strategy.

        Args:
            strategy: Strategy object that generates signals from regimes
        """
        self.strategy = strategy

    def update(self, interpreter_output: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals from interpreter output.

        Args:
            interpreter_output: DataFrame with 'regime_label' column containing RegimeLabel objects.
                Can also contain optional columns like 'timestamp', 'state', etc.

        Returns:
            List[TradingSignal] objects (one per row of input)

        Raises:
            ValueError: If 'regime_label' column is missing or contains None values
        """
        if interpreter_output.empty:
            return []

        # Validate required column
        if "regime_label" not in interpreter_output.columns:
            raise ValueError(
                "interpreter_output must contain 'regime_label' column with RegimeLabel objects"
            )

        signals = []
        for idx, row in interpreter_output.iterrows():
            regime = row["regime_label"]

            if regime is None:
                raise ValueError(f"RegimeLabel cannot be None at index {idx}")

            # Get signal from strategy (includes composition/decoration)
            signal = self.strategy.get_signal_for_regime(regime)

            # Apply risk management constraints
            final_signal = self._apply_risk_constraints(signal)

            # Add metadata (timestamp if available)
            if "timestamp" in row and pd.notna(row["timestamp"]):
                final_signal.metadata["timestamp"] = row["timestamp"]

            signals.append(final_signal)

        return signals

    def _apply_risk_constraints(self, signal: TradingSignal) -> TradingSignal:
        """
        Apply hard risk management constraints to the signal.

        Applies limits from strategy.risk_management:
        - prevent_shorts: Convert shorts to neutral
        - prevent_longs: Convert longs to neutral
        - max_position_size: Cap position size
        - kelly_criterion: Apply Kelly sizing (if enabled)

        Args:
            signal: Raw TradingSignal from strategy

        Returns:
            Modified TradingSignal with constraints applied
        """
        constraints_applied = signal.risk_management_applied.copy()

        # Enforce direction restrictions
        if self.strategy.risk_management.prevent_shorts and signal.direction == "short":
            signal = signal.with_direction("neutral").with_adjusted_position(0.0)
            constraints_applied["short_prevented"] = True

        if self.strategy.risk_management.prevent_longs and signal.direction == "long":
            signal = signal.with_direction("neutral").with_adjusted_position(0.0)
            constraints_applied["long_prevented"] = True

        # Enforce position size limit
        max_size = self.strategy.risk_management.max_position_size
        if signal.position_size > max_size:
            signal = signal.with_adjusted_position(max_size)
            constraints_applied["max_position_capped"] = True

        # Apply Kelly criterion if enabled
        if self.strategy.risk_management.kelly_criterion:
            kelly_size = self._calculate_kelly_position(signal)
            signal = signal.with_adjusted_position(kelly_size)
            constraints_applied["kelly_applied"] = True

        signal.risk_management_applied.update(constraints_applied)
        return signal

    def _calculate_kelly_position(self, signal: TradingSignal) -> float:
        """
        Calculate Kelly criterion position size.

        Simplified Kelly formula (requires win_rate and avg_win/loss from regime):
        f* = (p * b - q) / b
        where p = win rate, q = 1 - p, b = avg_win / avg_loss

        Falls back to signal position if stats unavailable.

        Args:
            signal: TradingSignal with regime label

        Returns:
            Position size based on Kelly criterion, capped at max
        """
        regime = signal.regime_label
        chars = regime.characteristics

        # Check if we have win rate data
        if chars.win_rate is None or chars.win_rate == 0:
            return signal.position_size

        # Simplified Kelly using available characteristics
        # win_rate is [0,1], use it directly
        p = chars.win_rate
        q = 1 - p

        # Estimate b (average win / average loss) from Sharpe ratio
        # Sharpe ~ win_rate - loss_rate, so use as proxy
        b = max(1.0, chars.sharpe_ratio) if chars.sharpe_ratio else 1.0

        # Kelly formula
        kelly_frac = (p * b - q) / b if b > 0 else 0
        kelly_frac = max(0, min(kelly_frac, 1.0))  # Clip to [0, 1]

        # Apply Kelly fraction (e.g., 0.25 for quarter Kelly)
        kelly_size = kelly_frac * self.strategy.risk_management.kelly_fraction
        kelly_size = min(kelly_size, self.strategy.risk_management.max_position_size)

        return kelly_size

    def to_dataframe(self, signals: List[TradingSignal]) -> pd.DataFrame:
        """
        Convert List[TradingSignal] to DataFrame for backward compatibility.

        Args:
            signals: List of TradingSignal objects

        Returns:
            DataFrame with columns: direction, position_size, confidence, regime_name, etc.
        """
        data = [signal.to_dict() for signal in signals]
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        return f"SignalGenerator(strategy={self.strategy.name})"
