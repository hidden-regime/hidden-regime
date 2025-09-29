"""
Financial signal generation based on regime characteristics.

Replaces naive state number assumptions with intelligent signal generation
based on actual financial regime characteristics and market behavior.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from enum import Enum

from .regime_characterizer import FinancialRegimeCharacterizer, RegimeProfile, RegimeType
from ..simulation.signal_generators import SignalGenerator, SignalType
from ..utils.exceptions import AnalysisError


class FinancialSignalGenerator(SignalGenerator):
    """
    Intelligent signal generator that uses regime financial characteristics.

    Replaces naive "state 0 = bear, state N-1 = bull" with data-driven
    signal generation based on actual regime behavior analysis.
    """

    def __init__(self,
                 strategy_type: Literal['regime_following', 'regime_contrarian', 'confidence_weighted'] = 'regime_following',
                 min_confidence: float = 0.3,
                 position_scaling: bool = True):
        """
        Initialize financial signal generator.

        Args:
            strategy_type: How to interpret regime signals
            min_confidence: Minimum regime confidence for non-zero signals
            position_scaling: Whether to scale position size by regime strength
        """
        super().__init__(f'financial_{strategy_type}')
        self.strategy_type = strategy_type
        self.min_confidence = min_confidence
        self.position_scaling = position_scaling
        self.regime_characterizer = FinancialRegimeCharacterizer()
        self.regime_profiles: Optional[Dict[int, RegimeProfile]] = None

    def generate_signals(self,
                        price_data: pd.DataFrame,
                        additional_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate trading signals based on regime financial characteristics.

        Args:
            price_data: DataFrame with OHLC price data and returns
            additional_data: DataFrame with regime predictions and confidence

        Returns:
            Series with trading signals scaled by regime characteristics
        """
        if not self.validate_data(price_data):
            raise ValueError("Invalid price data for financial signal generation")

        if additional_data is None or 'predicted_state' not in additional_data.columns:
            raise ValueError("Financial signal generator requires regime predictions in additional_data")

        # Characterize regimes if not done already
        if self.regime_profiles is None:
            self.regime_profiles = self.regime_characterizer.characterize_regimes(
                additional_data, price_data
            )

        # Generate signals based on regime characteristics
        signals = self._generate_characteristic_based_signals(
            price_data, additional_data, self.regime_profiles
        )

        return signals

    def _generate_characteristic_based_signals(self,
                                             price_data: pd.DataFrame,
                                             regime_data: pd.DataFrame,
                                             regime_profiles: Dict[int, RegimeProfile]) -> pd.Series:
        """Generate signals based on actual regime financial characteristics."""

        signals = pd.Series(SignalType.HOLD.value, index=price_data.index)

        # Align data
        aligned_data = price_data.join(regime_data[['predicted_state', 'confidence']], how='inner')

        for i, (idx, row) in enumerate(aligned_data.iterrows()):
            regime_state = int(row['predicted_state'])
            confidence = row.get('confidence', 1.0)

            if regime_state not in regime_profiles:
                continue

            profile = regime_profiles[regime_state]

            # Generate base signal from regime characteristics
            base_signal = self._get_base_signal_from_regime(profile)

            # Apply confidence and strategy type
            final_signal = self._apply_strategy_and_confidence(
                base_signal, profile, confidence
            )

            signals.iloc[i] = final_signal

        return signals

    def _get_base_signal_from_regime(self, profile: RegimeProfile) -> float:
        """Get base signal strength from regime financial characteristics."""

        if profile.regime_type == RegimeType.BULLISH:
            # Strong positive signal based on regime strength
            # Scale by annualized return and regime strength
            signal_strength = min(1.0, profile.annualized_return / 0.20)  # Normalize by 20% return
            return signal_strength * profile.regime_strength

        elif profile.regime_type == RegimeType.BEARISH:
            # Strong negative signal based on regime characteristics
            signal_strength = min(1.0, abs(profile.annualized_return) / 0.15)  # Normalize by 15% loss
            return -signal_strength * profile.regime_strength

        elif profile.regime_type == RegimeType.CRISIS:
            # Crisis: strong defensive signal (go to cash)
            return -0.5 * profile.regime_strength

        elif profile.regime_type == RegimeType.SIDEWAYS:
            # Sideways: minimal signal (hold current position)
            return 0.0

        else:  # MIXED
            # Mixed regime: conservative signal based on returns
            if profile.mean_daily_return > 0:
                return 0.2 * profile.regime_strength
            else:
                return -0.2 * profile.regime_strength

    def _apply_strategy_and_confidence(self,
                                     base_signal: float,
                                     profile: RegimeProfile,
                                     confidence: float) -> float:
        """Apply strategy type and confidence weighting to base signal."""

        # Apply minimum confidence threshold
        if confidence < self.min_confidence:
            return 0.0

        # Apply strategy type
        if self.strategy_type == 'regime_following':
            strategy_signal = base_signal

        elif self.strategy_type == 'regime_contrarian':
            # Contrarian: opposite of regime signal, but scaled down
            strategy_signal = -base_signal * 0.5

        elif self.strategy_type == 'confidence_weighted':
            # Weight signal by confidence and regime characteristics
            regime_confidence_weight = (confidence + profile.confidence_score) / 2
            strategy_signal = base_signal * regime_confidence_weight

        else:
            strategy_signal = base_signal

        # Apply position scaling if enabled
        if self.position_scaling:
            # Scale by confidence
            final_signal = strategy_signal * confidence
        else:
            # Binary signals based on sign
            final_signal = np.sign(strategy_signal) if abs(strategy_signal) > 0.1 else 0.0

        # Ensure signal is in valid range [-1, 1]
        return np.clip(final_signal, -1.0, 1.0)

    def get_regime_trading_summary(self) -> str:
        """Get summary of how each regime translates to trading signals."""

        if self.regime_profiles is None:
            return "No regime profiles available. Run generate_signals first."

        summary_lines = [
            f"Financial Signal Generation Summary ({self.strategy_type}):",
            "=" * 60
        ]

        for state_id, profile in self.regime_profiles.items():
            base_signal = self._get_base_signal_from_regime(profile)

            if base_signal > 0.5:
                signal_desc = "Strong Long"
            elif base_signal > 0.1:
                signal_desc = "Moderate Long"
            elif base_signal < -0.5:
                signal_desc = "Strong Short"
            elif base_signal < -0.1:
                signal_desc = "Moderate Short"
            else:
                signal_desc = "Hold/Neutral"

            summary_lines.extend([
                f"\nState {state_id} ({profile.regime_type.value}):",
                f"  Financial Signal: {signal_desc} ({base_signal:.2f})",
                f"  Return: {profile.annualized_return:.1%}",
                f"  Volatility: {profile.annualized_volatility:.1%}",
                f"  Regime Strength: {profile.regime_strength:.2f}",
                f"  Confidence: {profile.confidence_score:.2f}"
            ])

        return "\n".join(summary_lines)


class AdaptiveSignalGenerator(SignalGenerator):
    """
    Adaptive signal generator that adjusts strategy based on regime characteristics.

    Automatically selects the most appropriate strategy for each regime type.
    """

    def __init__(self, min_confidence: float = 0.4):
        super().__init__('adaptive_financial')
        self.min_confidence = min_confidence
        self.regime_characterizer = FinancialRegimeCharacterizer()
        self.regime_profiles: Optional[Dict[int, RegimeProfile]] = None

    def generate_signals(self,
                        price_data: pd.DataFrame,
                        additional_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """Generate adaptive signals based on regime-specific strategies."""

        if not self.validate_data(price_data):
            raise ValueError("Invalid price data for adaptive signal generation")

        if additional_data is None or 'predicted_state' not in additional_data.columns:
            raise ValueError("Adaptive signal generator requires regime predictions")

        # Characterize regimes
        if self.regime_profiles is None:
            self.regime_profiles = self.regime_characterizer.characterize_regimes(
                additional_data, price_data
            )

        # Generate adaptive signals
        signals = self._generate_adaptive_signals(
            price_data, additional_data, self.regime_profiles
        )

        return signals

    def _generate_adaptive_signals(self,
                                  price_data: pd.DataFrame,
                                  regime_data: pd.DataFrame,
                                  regime_profiles: Dict[int, RegimeProfile]) -> pd.Series:
        """Generate signals with regime-specific adaptive strategies."""

        signals = pd.Series(0.0, index=price_data.index)
        aligned_data = price_data.join(regime_data[['predicted_state', 'confidence']], how='inner')

        for i, (idx, row) in enumerate(aligned_data.iterrows()):
            regime_state = int(row['predicted_state'])
            confidence = row.get('confidence', 1.0)

            if regime_state not in regime_profiles or confidence < self.min_confidence:
                continue

            profile = regime_profiles[regime_state]

            # Select strategy based on regime type
            if profile.regime_type == RegimeType.BULLISH:
                # Trend following for bull markets
                signal = self._trend_following_signal(profile, confidence)

            elif profile.regime_type == RegimeType.BEARISH:
                # Defensive strategy for bear markets
                signal = self._defensive_signal(profile, confidence)

            elif profile.regime_type == RegimeType.SIDEWAYS:
                # Mean reversion for sideways markets
                signal = self._mean_reversion_signal(profile, confidence, i, aligned_data)

            elif profile.regime_type == RegimeType.CRISIS:
                # Risk-off for crisis periods
                signal = self._crisis_signal(profile, confidence)

            else:  # MIXED
                # Conservative approach for unclear regimes
                signal = self._conservative_signal(profile, confidence)

            signals.iloc[i] = signal

        return signals

    def _trend_following_signal(self, profile: RegimeProfile, confidence: float) -> float:
        """Generate trend-following signal for bullish regimes."""
        base_strength = min(1.0, profile.annualized_return / 0.25)
        return base_strength * confidence * profile.regime_strength

    def _defensive_signal(self, profile: RegimeProfile, confidence: float) -> float:
        """Generate defensive signal for bearish regimes."""
        # Strong defensive signal based on downside risk
        downside_strength = min(1.0, abs(profile.annualized_return) / 0.20)
        return -downside_strength * confidence * profile.regime_strength

    def _mean_reversion_signal(self,
                              profile: RegimeProfile,
                              confidence: float,
                              current_idx: int,
                              data: pd.DataFrame) -> float:
        """Generate mean reversion signal for sideways regimes."""
        # For sideways markets, use recent price action for mean reversion
        if current_idx < 5:
            return 0.0

        recent_returns = data['log_return'].iloc[max(0, current_idx-4):current_idx+1]
        if len(recent_returns) < 3:
            return 0.0

        # Simple mean reversion: if recent returns are strongly positive, signal negative
        recent_avg = recent_returns.mean()
        signal_strength = -np.tanh(recent_avg * 50)  # Scale and bound signal

        return signal_strength * confidence * 0.5  # Conservative for sideways

    def _crisis_signal(self, profile: RegimeProfile, confidence: float) -> float:
        """Generate crisis response signal."""
        # In crisis, go defensive immediately
        return -0.8 * confidence

    def _conservative_signal(self, profile: RegimeProfile, confidence: float) -> float:
        """Generate conservative signal for mixed regimes."""
        # Very conservative positioning
        if profile.mean_daily_return > 0:
            return 0.2 * confidence
        else:
            return -0.2 * confidence