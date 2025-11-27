"""
Signal adapters for converting regime detections to trading signals.

This module converts hidden-regime's regime detections and confidence scores
into actionable trading signals for QuantConnect algorithms.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import pandas as pd


class SignalDirection(Enum):
    """Trading signal direction."""

    LONG = 1
    NEUTRAL = 0
    SHORT = -1


class SignalStrength(Enum):
    """Signal strength levels."""

    STRONG = 3
    MODERATE = 2
    WEAK = 1
    NONE = 0


@dataclass
class TradingSignal:
    """
    Trading signal generated from regime detection.

    Attributes:
        direction: Signal direction (long, short, neutral)
        strength: Signal strength (strong, moderate, weak, none)
        allocation: Recommended portfolio allocation (0.0 to 1.0 or negative)
        confidence: Regime detection confidence (0.0 to 1.0)
        regime_name: Current regime name
        regime_state: Numeric regime state
        timestamp: Signal timestamp
        metadata: Additional signal information
    """

    direction: SignalDirection
    strength: SignalStrength
    allocation: float
    confidence: float
    regime_name: str
    regime_state: int
    timestamp: pd.Timestamp
    metadata: Dict = None

    def __post_init__(self) -> None:
        """Validate signal attributes."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

        if not -2.0 <= self.allocation <= 2.0:
            raise ValueError(f"Allocation must be -2 to 2, got {self.allocation}")

        if self.metadata is None:
            self.metadata = {}

    def is_actionable(self, min_confidence: float = 0.0) -> bool:
        """
        Check if signal is strong enough to act on.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            True if signal confidence meets threshold
        """
        return self.confidence >= min_confidence

    def should_rebalance(
        self, current_allocation: float, threshold: float = 0.1
    ) -> bool:
        """
        Determine if rebalancing is needed.

        Args:
            current_allocation: Current portfolio allocation
            threshold: Minimum change to trigger rebalance

        Returns:
            True if allocation change exceeds threshold
        """
        change = abs(self.allocation - current_allocation)
        return change >= threshold


class RegimeSignalAdapter:
    """
    Converts regime detection results to trading signals.

    This adapter takes hidden-regime's pipeline results and converts them
    into TradingSignal objects that can be used by QuantConnect algorithms.
    """

    def __init__(
        self,
        regime_allocations: Optional[Dict[str, float]] = None,
        min_confidence: float = 0.0,
    ):
        """
        Initialize signal adapter.

        Args:
            regime_allocations: Dict mapping regime names to allocations
            min_confidence: Minimum confidence for signals
        """
        self.regime_allocations = regime_allocations or {
            "Bull": 1.0,
            "Bear": 0.0,
            "Sideways": 0.5,
            "Crisis": 0.0,
        }
        self.min_confidence = min_confidence
        self._last_signal: Optional[TradingSignal] = None

    def generate_signal(
        self,
        regime_name: str,
        regime_state: int,
        confidence: float,
        timestamp: pd.Timestamp,
        **metadata,
    ) -> TradingSignal:
        """
        Generate trading signal from regime detection.

        Args:
            regime_name: Current regime name (e.g., "Bull", "Bear", "Bull-1", etc.)
            regime_state: Numeric regime state
            confidence: Detection confidence (0.0 to 1.0)
            timestamp: Signal timestamp
            **metadata: Additional signal metadata (regime_type can infer allocation)

        Returns:
            TradingSignal object
        """
        # Get allocation for this regime
        # First try exact match
        allocation = self.regime_allocations.get(regime_name)

        if allocation is None:
            # Try fuzzy matching based on regime_type if available
            regime_type = metadata.get("regime_type", "").lower()

            # Map regime types to allocations
            type_to_allocation = {
                "bullish": self.regime_allocations.get("Bull", 1.0),
                "bearish": self.regime_allocations.get("Bear", 0.0),
                "sideways": self.regime_allocations.get("Sideways", 0.5),
                "crisis": self.regime_allocations.get("Crisis", 0.0),
                "high": self.regime_allocations.get("Bull", 1.0),  # Alternative labels
                "low": self.regime_allocations.get("Bear", 0.0),
                "medium": self.regime_allocations.get("Sideways", 0.5),
            }

            allocation = type_to_allocation.get(regime_type)

            # If still not found, use regime_type to determine direction
            if allocation is None:
                # Last resort: infer from regime_type
                if "bullish" in regime_type or "high" in regime_type:
                    allocation = self.regime_allocations.get("Bull", 1.0)
                elif "bearish" in regime_type or "crisis" in regime_type:
                    allocation = self.regime_allocations.get("Bear", 0.0)
                elif "sideways" in regime_type or "medium" in regime_type:
                    allocation = self.regime_allocations.get("Sideways", 0.5)
                else:
                    allocation = 0.0  # Default to cash if unclear

        # Determine direction
        if allocation > 0.1:
            direction = SignalDirection.LONG
        elif allocation < -0.1:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        # Determine strength based on confidence
        if confidence >= 0.8:
            strength = SignalStrength.STRONG
        elif confidence >= 0.6:
            strength = SignalStrength.MODERATE
        elif confidence >= 0.4:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NONE

        signal = TradingSignal(
            direction=direction,
            strength=strength,
            allocation=allocation,
            confidence=confidence,
            regime_name=regime_name,
            regime_state=regime_state,
            timestamp=timestamp,
            metadata=metadata,
        )

        self._last_signal = signal
        return signal

    def from_pipeline_result(self, result_df: pd.DataFrame) -> TradingSignal:
        """
        Generate signal from hidden-regime pipeline result.

        Args:
            result_df: DataFrame from pipeline.update()

        Returns:
            TradingSignal for the latest data point

        Raises:
            ValueError: If result DataFrame is empty or missing columns
        """
        if result_df is None or result_df.empty:
            raise ValueError("Pipeline result is empty")

        # Get latest row
        latest = result_df.iloc[-1]

        # Extract regime information
        # The interpreter outputs 'regime_label' not 'regime_name'
        regime_name = latest.get("regime_label", latest.get("regime_name", "Unknown"))

        # Use 'regime_state' if available, otherwise fall back to 'predicted_state' or 'state'
        if "regime_state" in latest:
            regime_state = int(latest.get("regime_state", -1))
        elif "state" in latest:
            regime_state = int(latest.get("state", -1))
        else:
            regime_state = int(latest.get("predicted_state", -1))

        # The interpreter outputs 'regime_confidence' not 'confidence'
        confidence = float(latest.get("regime_confidence", latest.get("confidence", 0.0)))
        timestamp = result_df.index[-1]

        # Extract additional metadata
        metadata = {}
        optional_fields = [
            "mean_return",
            "volatility",
            "win_rate",
            "avg_regime_duration",
            "regime_strength",
            "regime_type",
            "days_in_regime",
            "expected_return",
            "expected_volatility",
            "max_drawdown",
        ]
        for field in optional_fields:
            if field in latest:
                metadata[field] = latest[field]

        return self.generate_signal(
            regime_name=regime_name,
            regime_state=regime_state,
            confidence=confidence,
            timestamp=timestamp,
            **metadata,
        )

    def has_regime_changed(self, new_signal: TradingSignal) -> bool:
        """
        Check if regime has changed since last signal.

        Args:
            new_signal: New trading signal

        Returns:
            True if regime changed
        """
        if self._last_signal is None:
            return True

        return self._last_signal.regime_name != new_signal.regime_name

    def get_allocation_change(self, new_signal: TradingSignal) -> float:
        """
        Calculate allocation change from last signal.

        Args:
            new_signal: New trading signal

        Returns:
            Allocation change (positive or negative)
        """
        if self._last_signal is None:
            return new_signal.allocation

        return new_signal.allocation - self._last_signal.allocation

    @property
    def last_signal(self) -> Optional[TradingSignal]:
        """Get the last generated signal."""
        return self._last_signal


class MultiAssetSignalAdapter:
    """
    Signal adapter for multi-asset regime-based strategies.

    Manages signals for multiple assets and generates portfolio allocations
    based on individual asset regimes.
    """

    def __init__(
        self,
        equal_weight: bool = True,
        concentration_limit: float = 0.5,
    ):
        """
        Initialize multi-asset signal adapter.

        Args:
            equal_weight: Whether to equally weight assets in same regime
            concentration_limit: Maximum allocation to single asset
        """
        self.equal_weight = equal_weight
        self.concentration_limit = concentration_limit
        self._asset_signals: Dict[str, TradingSignal] = {}

    def add_asset_signal(self, ticker: str, signal: TradingSignal) -> None:
        """
        Add or update signal for an asset.

        Args:
            ticker: Asset ticker symbol
            signal: Trading signal for the asset
        """
        self._asset_signals[ticker] = signal

    def calculate_portfolio_allocations(
        self, preferred_regimes: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Calculate portfolio allocations across all assets.

        Args:
            preferred_regimes: List of regime names to prefer (e.g., ["Bull"])

        Returns:
            Dict mapping tickers to portfolio allocations
        """
        if not self._asset_signals:
            return {}

        preferred_regimes = preferred_regimes or ["Bull", "Sideways"]

        # Filter assets in preferred regimes
        eligible_assets = {
            ticker: signal
            for ticker, signal in self._asset_signals.items()
            if signal.regime_name in preferred_regimes
        }

        if not eligible_assets:
            # No assets in preferred regimes - go to cash
            return {ticker: 0.0 for ticker in self._asset_signals.keys()}

        # Calculate allocations
        allocations = {}

        if self.equal_weight:
            # Equal weight among eligible assets
            weight = min(1.0 / len(eligible_assets), self.concentration_limit)

            for ticker in self._asset_signals.keys():
                if ticker in eligible_assets:
                    allocations[ticker] = weight
                else:
                    allocations[ticker] = 0.0
        else:
            # Weight by confidence
            total_confidence = sum(s.confidence for s in eligible_assets.values())

            for ticker in self._asset_signals.keys():
                if ticker in eligible_assets:
                    signal = eligible_assets[ticker]
                    weight = signal.confidence / total_confidence
                    allocations[ticker] = min(weight, self.concentration_limit)
                else:
                    allocations[ticker] = 0.0

        # Normalize to ensure sum <= 1.0
        total_allocation = sum(allocations.values())
        if total_allocation > 1.0:
            allocations = {k: v / total_allocation for k, v in allocations.items()}

        return allocations

    def get_asset_signal(self, ticker: str) -> Optional[TradingSignal]:
        """
        Get signal for specific asset.

        Args:
            ticker: Asset ticker symbol

        Returns:
            TradingSignal or None if not available
        """
        return self._asset_signals.get(ticker)

    def get_regime_summary(self) -> Dict[str, int]:
        """
        Get count of assets in each regime.

        Returns:
            Dict mapping regime names to asset counts
        """
        summary = {}
        for signal in self._asset_signals.values():
            regime = signal.regime_name
            summary[regime] = summary.get(regime, 0) + 1
        return summary

    def clear(self) -> None:
        """Clear all asset signals."""
        self._asset_signals.clear()
