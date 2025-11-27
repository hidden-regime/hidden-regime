"""
Concrete strategy implementations.

This module contains:
1. Core strategies (leaf nodes):
   - RegimeFollowingStrategy: Trade in direction of regime
   - ContrarianStrategy: Fade/short the regime

2. Wrapper/Decorator strategies (composable modifiers):
   - ConfidenceWeightedStrategy: Scale position by regime confidence
   - VolatilityAdjustedStrategy: Reduce position in high volatility
   - MultiTimeframeAlignmentStrategy: Only trade when timeframes align

All can be composed using the decorator pattern:
  ConfidenceWeighted(MultiTimeframeAlignment(RegimeFollowing()))
"""

from typing import Optional

from hidden_regime.interpreter.regime_types import RegimeLabel
from hidden_regime.signals.types import TradingSignal
from hidden_regime.strategies.base import RiskManagementConfig, Strategy


# ============================================================================
# CORE STRATEGIES (Leaf Nodes)
# ============================================================================


class RegimeFollowingStrategy(Strategy):
    """
    Trade in the direction of the detected regime.

    This is the fundamental "follow the trend" strategy:
    - BULLISH → long signal
    - BEARISH → short signal (if allowed by risk management)
    - SIDEWAYS → neutral signal
    - CRISIS → reduced short signal
    - MIXED → neutral signal
    """

    def __init__(
        self,
        long_confidence_threshold: float = 0.65,
        short_confidence_threshold: float = 0.65,
        risk_management: Optional[RiskManagementConfig] = None,
    ):
        """Initialize regime following strategy.

        Args:
            long_confidence_threshold: Minimum regime confidence for long signals
            short_confidence_threshold: Minimum regime confidence for short signals
            risk_management: RiskManagementConfig with hard constraints
        """
        super().__init__(
            name="RegimeFollowing",
            description="Trade in direction of detected regime",
            risk_management=risk_management,
        )
        self.long_confidence_threshold = long_confidence_threshold
        self.short_confidence_threshold = short_confidence_threshold

    def get_signal_for_regime(self, regime: RegimeLabel) -> TradingSignal:
        """Generate signal based on regime type."""
        # Check if regime confidence meets threshold
        if regime.regime_strength < max(
            self.long_confidence_threshold, self.short_confidence_threshold
        ):
            return TradingSignal(
                direction="neutral",
                position_size=0.0,
                confidence=regime.regime_strength,
                regime_label=regime,
                strategy_name=self.name,
                valid=False,
                metadata={"reason": "regime_confidence_too_low"},
            )

        # Determine direction based on regime type
        if regime.name == "BULLISH":
            direction = "long"
            position_size = 1.0
            confidence = regime.regime_strength
        elif regime.name == "BEARISH":
            direction = "short"
            position_size = 1.0
            confidence = regime.regime_strength
        elif regime.name == "CRISIS":
            # Crisis: reduced short signal
            direction = "short"
            position_size = 0.5
            confidence = regime.regime_strength * 0.75
        elif regime.name == "SIDEWAYS":
            direction = "neutral"
            position_size = 0.0
            confidence = regime.regime_strength
        else:  # MIXED
            direction = "neutral"
            position_size = 0.0
            confidence = regime.regime_strength * 0.5

        # Check threshold for this direction
        if direction == "long" and regime.regime_strength < self.long_confidence_threshold:
            direction = "neutral"
            position_size = 0.0
        elif direction == "short" and regime.regime_strength < self.short_confidence_threshold:
            direction = "neutral"
            position_size = 0.0

        return TradingSignal(
            direction=direction,
            position_size=position_size,
            confidence=confidence,
            regime_label=regime,
            strategy_name=self.name,
            valid=direction != "neutral",
        )


class ContrarianStrategy(Strategy):
    """
    Fade/short the regime (contrarian strategy).

    Opposite of regime following:
    - BULLISH → short signal
    - BEARISH → long signal
    - SIDEWAYS → neutral signal
    - CRISIS → reduced long signal
    - MIXED → neutral signal
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        risk_management: Optional[RiskManagementConfig] = None,
    ):
        """Initialize contrarian strategy.

        Args:
            confidence_threshold: Minimum regime confidence to generate signal
            risk_management: RiskManagementConfig with hard constraints
        """
        super().__init__(
            name="Contrarian",
            description="Fade/short the detected regime",
            risk_management=risk_management,
        )
        self.confidence_threshold = confidence_threshold

    def get_signal_for_regime(self, regime: RegimeLabel) -> TradingSignal:
        """Generate contrarian signal (opposite of regime)."""
        # Check if regime confidence meets threshold
        if regime.regime_strength < self.confidence_threshold:
            return TradingSignal(
                direction="neutral",
                position_size=0.0,
                confidence=regime.regime_strength,
                regime_label=regime,
                strategy_name=self.name,
                valid=False,
            )

        # Invert the regime direction
        if regime.name == "BULLISH":
            direction = "short"
            position_size = 1.0
            confidence = regime.regime_strength
        elif regime.name == "BEARISH":
            direction = "long"
            position_size = 1.0
            confidence = regime.regime_strength
        elif regime.name == "CRISIS":
            # Crisis: reduced long signal (fade the fear)
            direction = "long"
            position_size = 0.5
            confidence = regime.regime_strength * 0.75
        else:  # SIDEWAYS or MIXED
            direction = "neutral"
            position_size = 0.0
            confidence = regime.regime_strength

        return TradingSignal(
            direction=direction,
            position_size=position_size,
            confidence=confidence,
            regime_label=regime,
            strategy_name=self.name,
            valid=direction != "neutral",
        )


# ============================================================================
# DECORATOR/WRAPPER STRATEGIES (Composable Modifiers)
# ============================================================================


class ConfidenceWeightedStrategy(Strategy):
    """
    Wraps another strategy and scales position by regime confidence.

    This is a decorator that modifies the base strategy's signal:
    - Position size scaled by regime confidence (confidence * base_position)
    - Confidence weighted accordingly
    - All other aspects unchanged

    Example:
        base_strategy = RegimeFollowingStrategy()
        weighted = ConfidenceWeightedStrategy(base_strategy)
        signal = weighted.get_signal_for_regime(regime_label)
    """

    def __init__(
        self,
        wrapped_strategy: Strategy,
        position_scale: float = 1.0,
        risk_management: Optional[RiskManagementConfig] = None,
    ):
        """Initialize confidence weighted wrapper.

        Args:
            wrapped_strategy: Strategy to wrap
            position_scale: Scaling factor for position sizing
            risk_management: RiskManagementConfig with hard constraints
        """
        super().__init__(
            name=f"ConfidenceWeighted({wrapped_strategy.name})",
            description=f"Scales position of {wrapped_strategy.name} by regime confidence",
            risk_management=risk_management or wrapped_strategy.risk_management,
        )
        self.wrapped_strategy = wrapped_strategy
        self.position_scale = position_scale

    def get_signal_for_regime(self, regime: RegimeLabel) -> TradingSignal:
        """Get base signal and scale position by confidence."""
        base_signal = self.wrapped_strategy.get_signal_for_regime(regime)

        # Scale position by regime confidence
        scaled_position = base_signal.position_size * regime.regime_strength * self.position_scale
        scaled_position = min(scaled_position, self.risk_management.max_position_size)

        return base_signal.with_adjusted_position(scaled_position)


class VolatilityAdjustedStrategy(Strategy):
    """
    Wraps another strategy and reduces position in high volatility.

    This decorator reduces position sizing when volatility is elevated:
    - adjusted_position = base_position / (1 + volatility * factor)
    - Useful for risk management in uncertain markets

    Example:
        base_strategy = RegimeFollowingStrategy()
        vol_adjusted = VolatilityAdjustedStrategy(base_strategy, volatility_threshold=0.25)
        signal = vol_adjusted.get_signal_for_regime(regime_label)
    """

    def __init__(
        self,
        wrapped_strategy: Strategy,
        volatility_threshold: float = 0.25,
        volatility_factor: float = 10.0,
        risk_management: Optional[RiskManagementConfig] = None,
    ):
        """Initialize volatility adjusted wrapper.

        Args:
            wrapped_strategy: Strategy to wrap
            volatility_threshold: Reference volatility level
            volatility_factor: Scaling factor for volatility adjustment
            risk_management: RiskManagementConfig with hard constraints
        """
        super().__init__(
            name=f"VolatilityAdjusted({wrapped_strategy.name})",
            description=f"Reduces position of {wrapped_strategy.name} in high volatility",
            risk_management=risk_management or wrapped_strategy.risk_management,
        )
        self.wrapped_strategy = wrapped_strategy
        self.volatility_threshold = volatility_threshold
        self.volatility_factor = volatility_factor

    def get_signal_for_regime(self, regime: RegimeLabel) -> TradingSignal:
        """Get base signal and adjust for volatility."""
        base_signal = self.wrapped_strategy.get_signal_for_regime(regime)

        # Volatility adjustment: reduce position when volatility is high
        volatility = regime.characteristics.annualized_volatility
        vol_adjustment = 1.0 / (1.0 + max(0, volatility - self.volatility_threshold) * self.volatility_factor)

        adjusted_position = base_signal.position_size * vol_adjustment
        adjusted_position = min(adjusted_position, self.risk_management.max_position_size)

        return base_signal.with_adjusted_position(adjusted_position).with_risk_constraint(
            "volatility_adjustment", f"{vol_adjustment:.3f}"
        )


class MultiTimeframeAlignmentStrategy(Strategy):
    """
    Wraps another strategy and only trades when timeframes align.

    This decorator filters signals based on multi-timeframe alignment.
    Only generates valid signals when alignment score is high enough.
    Critical for "Sharpe 10+" high-conviction strategies.

    Note: Requires 'timeframe_alignment' field in regime metadata.

    Example:
        base_strategy = RegimeFollowingStrategy()
        mtf_aligned = MultiTimeframeAlignmentStrategy(
            base_strategy,
            alignment_threshold=0.7
        )
        signal = mtf_aligned.get_signal_for_regime(regime_label)
    """

    def __init__(
        self,
        wrapped_strategy: Strategy,
        alignment_threshold: float = 0.7,
        risk_management: Optional[RiskManagementConfig] = None,
    ):
        """Initialize multi-timeframe alignment wrapper.

        Args:
            wrapped_strategy: Strategy to wrap
            alignment_threshold: Minimum alignment score (0-1) to trade
            risk_management: RiskManagementConfig with hard constraints
        """
        super().__init__(
            name=f"MultiTimeframeAlignment({wrapped_strategy.name})",
            description=f"Trades {wrapped_strategy.name} only when timeframes align",
            risk_management=risk_management or wrapped_strategy.risk_management,
        )
        self.wrapped_strategy = wrapped_strategy
        self.alignment_threshold = alignment_threshold

    def get_signal_for_regime(self, regime: RegimeLabel) -> TradingSignal:
        """Get base signal and filter by timeframe alignment."""
        # Check if alignment data is available
        alignment_score = regime.metadata.get("timeframe_alignment", None) if hasattr(regime, "metadata") else None

        if alignment_score is None or alignment_score < self.alignment_threshold:
            base_signal = self.wrapped_strategy.get_signal_for_regime(regime)
            return base_signal.mark_invalid(
                f"Timeframe alignment {alignment_score or 'N/A'} < {self.alignment_threshold}"
            )

        # Alignment is good, return base signal
        return self.wrapped_strategy.get_signal_for_regime(regime)
