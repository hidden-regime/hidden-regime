"""
Builder for creating RegimeLabel objects from interpreter output.

This module handles the conversion of raw interpreter data into structured RegimeLabel objects.
It encapsulates the logic for:
1. Computing regime characteristics (returns, volatility, drawdown, etc.)
2. Determining trading semantics (bias, position sign, thresholds)
3. Creating immutable RegimeLabel objects

This enables the Interpreter to produce RegimeLabel objects that serve as the
single source of truth for regime semantics.
"""

from typing import Dict, Optional

import numpy as np

from hidden_regime.interpreter.regime_types import (
    RegimeCharacteristics,
    RegimeLabel,
    RegimeType,
    TradingSemantics,
    REGIME_TYPE_COLORS,
)


class RegimeLabelBuilder:
    """
    Builds RegimeLabel objects from regime characteristics and statistics.

    Encapsulates the logic for creating RegimeLabel objects with proper
    characteristics and trading semantics based on financial metrics.
    """

    @staticmethod
    def create_regime_label(
        name: str,
        regime_type: RegimeType,
        characteristics: RegimeCharacteristics,
        regime_strength: float,
    ) -> RegimeLabel:
        """
        Create a RegimeLabel with properly inferred trading semantics.

        Args:
            name: Regime name (BULLISH, BEARISH, SIDEWAYS, CRISIS, MIXED)
            regime_type: RegimeType enum for color lookup
            characteristics: RegimeCharacteristics with financial metrics
            regime_strength: 0-1 confidence in classification

        Returns:
            Complete RegimeLabel object with semantics inferred from characteristics
        """
        # Infer trading semantics from regime type and characteristics
        trading_semantics = RegimeLabelBuilder._infer_trading_semantics(
            name, characteristics
        )

        # Get color for this regime type
        color = REGIME_TYPE_COLORS.get(regime_type, "#9970ab")  # Default to purple

        return RegimeLabel(
            name=name,
            regime_type=regime_type,
            color=color,
            characteristics=characteristics,
            trading_semantics=trading_semantics,
            regime_strength=regime_strength,
        )

    @staticmethod
    def _infer_trading_semantics(
        regime_name: str, characteristics: RegimeCharacteristics
    ) -> TradingSemantics:
        """
        Infer trading semantics from regime name and characteristics.

        Determines:
        - bias: positive/negative/neutral
        - typical_position_sign: +1 for long, -1 for short, 0 for flat
        - confidence_threshold: when to act on this regime
        - position_sizing_bias: how much to size relative to normal

        Args:
            regime_name: Name of the regime (BULLISH, BEARISH, etc.)
            characteristics: Financial characteristics

        Returns:
            TradingSemantics object with inferred values
        """
        mean_return = characteristics.mean_daily_return
        volatility = characteristics.annualized_volatility

        if regime_name == "BULLISH":
            bias = "positive"
            position_sign = 1  # Long
            # Be more aggressive in low-volatility bull markets
            confidence_threshold = 0.60 if volatility < 0.20 else 0.70
            position_sizing_bias = 1.2 if mean_return > 0.15 else 1.0

        elif regime_name == "BEARISH":
            bias = "negative"
            position_sign = -1  # Short
            # Be cautious in high-volatility bear markets
            confidence_threshold = 0.70 if volatility > 0.25 else 0.65
            position_sizing_bias = 1.1 if volatility > 0.25 else 0.9

        elif regime_name == "SIDEWAYS":
            bias = "neutral"
            position_sign = 0  # Flat
            # Sideways is risky, need high confidence
            confidence_threshold = 0.75
            position_sizing_bias = 0.5  # Reduce size in unclear regime

        elif regime_name == "CRISIS":
            bias = "negative"
            position_sign = -1  # Short/hedge
            # Crisis demands very high confidence before acting
            confidence_threshold = 0.80
            position_sizing_bias = 0.6  # Reduce exposure in crisis

        else:  # MIXED
            bias = "neutral"
            position_sign = 0  # Flat
            # Mixed regime is very risky
            confidence_threshold = 0.85
            position_sizing_bias = 0.3  # Minimal position

        return TradingSemantics(
            bias=bias,
            typical_position_sign=position_sign,
            confidence_threshold=confidence_threshold,
            position_sizing_bias=position_sizing_bias,
        )

    @staticmethod
    def build_from_metrics(
        regime_name: str,
        regime_type: RegimeType,
        regime_strength: float,
        mean_daily_return: float,
        annualized_return: float,
        daily_volatility: float,
        annualized_volatility: float,
        win_rate: float,
        max_drawdown: float,
        return_skewness: float,
        return_kurtosis: float,
        sharpe_ratio: float,
        persistence_days: float,
        transition_volatility: float,
        transition_probs: Optional[Dict[str, float]] = None,
        state_id: Optional[int] = None,
    ) -> RegimeLabel:
        """
        Build a RegimeLabel from individual metrics.

        This is a convenience method that builds characteristics and creates a label.

        Args:
            regime_name: Name of the regime
            regime_type: RegimeType enum
            regime_strength: 0-1 classification confidence
            mean_daily_return: Average daily return
            annualized_return: Annualized return
            daily_volatility: Daily standard deviation
            annualized_volatility: Annualized volatility
            win_rate: Percentage of positive days
            max_drawdown: Maximum drawdown
            return_skewness: Distribution skewness
            return_kurtosis: Distribution kurtosis
            sharpe_ratio: Risk-adjusted return
            persistence_days: Average regime duration
            transition_volatility: Volatility during transitions
            transition_probs: Probabilities to other regimes (optional)
            state_id: HMM state index (optional)

        Returns:
            Complete RegimeLabel object
        """
        characteristics = RegimeCharacteristics(
            mean_daily_return=mean_daily_return,
            annualized_return=annualized_return,
            daily_volatility=daily_volatility,
            annualized_volatility=annualized_volatility,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            return_skewness=return_skewness,
            return_kurtosis=return_kurtosis,
            sharpe_ratio=sharpe_ratio,
            persistence_days=persistence_days,
            regime_strength=regime_strength,
            transition_volatility=transition_volatility,
            transition_probs=transition_probs or {},
            state_id=state_id,
        )

        return RegimeLabelBuilder.create_regime_label(
            name=regime_name,
            regime_type=regime_type,
            characteristics=characteristics,
            regime_strength=regime_strength,
        )
