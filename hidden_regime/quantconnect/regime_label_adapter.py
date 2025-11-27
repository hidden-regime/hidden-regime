"""Adapter for using RegimeLabel objects with QuantConnect algorithms.

This module provides adapters that convert RegimeLabel objects from the new
architecture into QuantConnect-compatible trading signals, eliminating the need
for re-interpretation in the algorithm layer.

Key Advantage: The single source of truth (RegimeLabel) flows directly from
Interpreter → Strategy → SignalGenerator → QuantConnect, with no redundant
interpretation in QuantConnect code.
"""

from typing import Optional

import pandas as pd

from hidden_regime.interpreter.regime_types import RegimeLabel
from hidden_regime.signals.types import TradingSignal as HiddenRegimeTradingSignal


class RegimeLabelQuantConnectAdapter:
    """
    Adapts RegimeLabel objects to QuantConnect trading signals.

    This adapter bridges the new hidden-regime architecture (RegimeLabel + Strategy)
    with QuantConnect's algorithm requirements, eliminating manual interpretation.

    Example Usage:
        from hidden_regime.quantconnect import RegimeLabelQuantConnectAdapter

        adapter = RegimeLabelQuantConnectAdapter(
            regime_allocations={
                "BULLISH": 1.0,
                "BEARISH": -0.5,
                "SIDEWAYS": 0.0,
                "CRISIS": -1.0,
            }
        )

        # From pipeline signals
        signal = adapter.adapt_hidden_regime_signal(trading_signal_obj, regime_label)

        # Or from direct regime label
        signal = adapter.adapt_regime_label(regime_label)
    """

    def __init__(
        self,
        regime_allocations: Optional[dict] = None,
        min_confidence: float = 0.0,
    ):
        """Initialize the adapter.

        Args:
            regime_allocations: Dict mapping regime names to allocations.
                Default: BULLISH→1.0, BEARISH→-0.5, SIDEWAYS→0.0, CRISIS→-1.0
            min_confidence: Minimum regime confidence to generate actionable signal.
                Signals below this threshold will be converted to neutral.
        """
        self.regime_allocations = regime_allocations or {
            "BULLISH": 1.0,
            "BEARISH": -0.5,
            "SIDEWAYS": 0.0,
            "CRISIS": -1.0,
            "MIXED": 0.0,
        }
        self.min_confidence = min_confidence

    def adapt_regime_label(
        self, regime_label: RegimeLabel, timestamp: Optional[pd.Timestamp] = None
    ) -> dict:
        """
        Adapt a RegimeLabel directly to a QuantConnect-compatible signal.

        This is the primary use case when you have RegimeLabel objects from
        the Interpreter and want direct QuantConnect integration.

        Args:
            regime_label: RegimeLabel object from interpreter
            timestamp: Optional timestamp for the signal

        Returns:
            Dictionary with QuantConnect-compatible signal data:
            - direction: 'long', 'short', or 'neutral'
            - allocation: Float representing position size
            - confidence: Float 0-1 representing regime confidence
            - regime_name: Name of the regime (from RegimeLabel)
            - regime_type: Type of regime (BULLISH, BEARISH, etc.)
            - characteristics: Dict of financial characteristics
            - trading_semantics: Dict of trading semantics
        """
        # Get allocation based on regime type
        allocation = self.regime_allocations.get(
            regime_label.name, self.regime_allocations.get("MIXED", 0.0)
        )

        # Determine direction
        if allocation > 0.1:
            direction = "long"
        elif allocation < -0.1:
            direction = "short"
        else:
            direction = "neutral"

        # Check confidence threshold
        if regime_label.regime_strength < self.min_confidence:
            direction = "neutral"
            allocation = 0.0

        return {
            "direction": direction,
            "allocation": allocation,
            "confidence": regime_label.regime_strength,
            "regime_name": regime_label.name,
            "regime_type": regime_label.regime_type.value,
            "color": regime_label.color,
            "characteristics": {
                "mean_daily_return": regime_label.characteristics.mean_daily_return,
                "annualized_return": regime_label.characteristics.annualized_return,
                "annualized_volatility": regime_label.characteristics.annualized_volatility,
                "sharpe_ratio": regime_label.characteristics.sharpe_ratio,
                "max_drawdown": regime_label.characteristics.max_drawdown,
                "win_rate": regime_label.characteristics.win_rate,
                "persistence_days": regime_label.characteristics.persistence_days,
            },
            "trading_semantics": {
                "bias": regime_label.trading_semantics.bias,
                "typical_position_sign": regime_label.trading_semantics.typical_position_sign,
                "confidence_threshold": regime_label.trading_semantics.confidence_threshold,
                "position_sizing_bias": regime_label.trading_semantics.position_sizing_bias,
            },
        }

    def adapt_hidden_regime_signal(
        self,
        signal: HiddenRegimeTradingSignal,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> dict:
        """
        Adapt a hidden-regime TradingSignal to QuantConnect format.

        This is useful when you have TradingSignal objects from the SignalGenerator
        and want to integrate them with QuantConnect while preserving all metadata.

        Args:
            signal: TradingSignal from SignalGenerator
            timestamp: Optional timestamp (can come from signal metadata)

        Returns:
            Dictionary with QuantConnect-compatible signal data including full
            audit trail and risk management information
        """
        # Map signal direction to allocation
        direction_to_allocation = {
            "long": 1.0,
            "short": -1.0,
            "neutral": 0.0,
        }

        base_allocation = direction_to_allocation.get(signal.direction, 0.0)
        final_allocation = base_allocation * signal.position_size

        return {
            "direction": signal.direction,
            "allocation": final_allocation,
            "position_size": signal.position_size,
            "confidence": signal.confidence,
            "regime_name": signal.regime_label.name if signal.regime_label else "UNKNOWN",
            "regime_type": signal.regime_label.regime_type.value if signal.regime_label else "unknown",
            "strategy": signal.strategy_name,
            "valid": signal.valid,
            "risk_management_applied": signal.risk_management_applied,
            "characteristics": self._extract_characteristics(signal.regime_label),
            "trading_semantics": self._extract_trading_semantics(signal.regime_label),
            "timestamp": timestamp or signal.metadata.get("timestamp"),
        }

    def _extract_characteristics(self, regime_label: Optional[RegimeLabel]) -> dict:
        """Extract characteristics from regime label."""
        if not regime_label or not regime_label.characteristics:
            return {}

        chars = regime_label.characteristics
        return {
            "mean_daily_return": chars.mean_daily_return,
            "annualized_return": chars.annualized_return,
            "annualized_volatility": chars.annualized_volatility,
            "sharpe_ratio": chars.sharpe_ratio,
            "max_drawdown": chars.max_drawdown,
            "win_rate": chars.win_rate,
            "persistence_days": chars.persistence_days,
        }

    def _extract_trading_semantics(self, regime_label: Optional[RegimeLabel]) -> dict:
        """Extract trading semantics from regime label."""
        if not regime_label or not regime_label.trading_semantics:
            return {}

        semantics = regime_label.trading_semantics
        return {
            "bias": semantics.bias,
            "typical_position_sign": semantics.typical_position_sign,
            "confidence_threshold": semantics.confidence_threshold,
            "position_sizing_bias": semantics.position_sizing_bias,
        }

    def from_pipeline_dataframe(self, signals_df: pd.DataFrame) -> list:
        """
        Convert a DataFrame of signals from the pipeline to QuantConnect format.

        This processes the output of SignalGenerator (via StrategyBasedSignalGeneratorComponent)
        into a list of QuantConnect-compatible signals.

        Args:
            signals_df: DataFrame from signal generator component with columns:
                - direction, position_size, confidence, regime_label, strategy_name, etc.

        Returns:
            List of dicts with QuantConnect-compatible signal data
        """
        signals = []

        for idx, row in signals_df.iterrows():
            # Try to get RegimeLabel object
            regime_label = row.get("regime_label")

            # If we have a RegimeLabel object
            if regime_label and isinstance(regime_label, RegimeLabel):
                signal_data = self.adapt_regime_label(regime_label, timestamp=idx)
                signal_data.update({
                    "direction": row.get("direction", "neutral"),
                    "position_size": row.get("position_size", 0.0),
                    "strategy": row.get("strategy_name", "unknown"),
                })
                signals.append(signal_data)
            else:
                # Fallback: try to extract what we can from DataFrame
                signals.append({
                    "direction": row.get("direction", "neutral"),
                    "position_size": row.get("position_size", 0.0),
                    "confidence": row.get("confidence", 0.0),
                    "regime_name": row.get("regime_name", "UNKNOWN"),
                    "strategy": row.get("strategy_name", "unknown"),
                    "timestamp": idx,
                })

        return signals
