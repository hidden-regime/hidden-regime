"""
Financial regime characterization for intelligent trading signal generation.

This module analyzes the actual financial characteristics of detected regimes
to determine their market behavior, replacing naive state number assumptions
with data-driven regime interpretation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

from ..utils.exceptions import AnalysisError


class RegimeType(Enum):
    """Financial regime types based on actual market behavior."""

    BULLISH = "bullish"  # Strong positive returns, moderate volatility
    BEARISH = "bearish"  # Negative returns, often high volatility
    SIDEWAYS = "sideways"  # Low returns, typically low volatility
    CRISIS = "crisis"  # Extreme volatility, negative returns
    MIXED = "mixed"  # Unclear financial characteristics


@dataclass
class RegimeProfile:
    """
    Financial profile of a detected regime.

    Contains actual financial characteristics rather than arbitrary state labels.
    """

    state_id: int
    regime_type: RegimeType
    mean_daily_return: float
    daily_volatility: float
    annualized_return: float
    annualized_volatility: float
    persistence_days: float
    regime_strength: float  # How distinct this regime is from others
    confidence_score: float  # Statistical confidence in classification

    # Trading characteristics
    win_rate: float  # Percentage of positive return days
    max_drawdown: float  # Maximum drawdown during this regime
    return_skewness: float  # Distribution shape
    return_kurtosis: float  # Tail risk

    # Regime transition behavior
    avg_duration: float  # Average time spent in this regime
    transition_volatility: float  # Volatility around regime changes


class FinancialRegimeCharacterizer:
    """
    Analyzes detected regime states to understand their financial characteristics.

    Replaces naive "state 0 = bear, state N-1 = bull" assumptions with actual
    analysis of regime behavior in terms of returns, volatility, and persistence.
    """

    def __init__(
        self,
        min_regime_days: int = 5,
        return_column: str = "log_return",
        price_column: str = "close",
    ):
        """
        Initialize financial regime characterizer.

        Args:
            min_regime_days: Minimum days in regime for reliable characterization
            return_column: Column name for daily returns
            price_column: Column name for price data
        """
        self.min_regime_days = min_regime_days
        self.return_column = return_column
        self.price_column = price_column

    def characterize_regimes(
        self, regime_data: pd.DataFrame, price_data: pd.DataFrame
    ) -> Dict[int, RegimeProfile]:
        """
        Analyze the financial characteristics of each detected regime.

        Args:
            regime_data: DataFrame with regime predictions and confidence
            price_data: DataFrame with price and return data

        Returns:
            Dictionary mapping state IDs to RegimeProfile objects
        """
        if "predicted_state" not in regime_data.columns:
            raise AnalysisError("regime_data must contain 'predicted_state' column")

        if self.return_column not in price_data.columns:
            raise AnalysisError(
                f"price_data must contain '{self.return_column}' column"
            )

        # Align data
        aligned_data = price_data.join(
            regime_data[["predicted_state", "confidence"]], how="inner"
        )

        if len(aligned_data) == 0:
            raise AnalysisError("No aligned data between price and regime predictions")

        # Get unique states
        states = sorted(aligned_data["predicted_state"].unique())
        n_states = len(states)

        regime_profiles = {}

        for state in states:
            # Extract data for this regime
            state_data = aligned_data[aligned_data["predicted_state"] == state]

            if len(state_data) < self.min_regime_days:
                print(
                    f"Warning: State {state} has only {len(state_data)} days, may be unreliable"
                )

            # Calculate financial characteristics
            profile = self._analyze_regime_state(state, state_data, aligned_data)
            regime_profiles[state] = profile

        # Calculate relative regime strengths
        self._calculate_regime_strengths(regime_profiles)

        return regime_profiles

    def _analyze_regime_state(
        self, state_id: int, state_data: pd.DataFrame, full_data: pd.DataFrame
    ) -> RegimeProfile:
        """Analyze financial characteristics of a single regime state."""

        returns = state_data[self.return_column].dropna()

        if len(returns) == 0:
            raise AnalysisError(f"No valid returns for state {state_id}")

        # Basic return statistics
        mean_daily_return = returns.mean()
        daily_volatility = returns.std()
        annualized_return = mean_daily_return * 252
        annualized_volatility = daily_volatility * np.sqrt(252)

        # Trading characteristics
        win_rate = (returns > 0).mean()
        return_skewness = returns.skew()
        return_kurtosis = returns.kurtosis()

        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(state_data[self.price_column])

        # Regime persistence analysis
        persistence_days = self._calculate_persistence(state_data, full_data, state_id)
        avg_duration = self._calculate_average_duration(full_data, state_id)
        transition_volatility = self._calculate_transition_volatility(
            full_data, state_id
        )

        # Classify regime type
        regime_type = self._classify_regime_type(
            mean_daily_return, daily_volatility, win_rate, max_drawdown
        )

        # Calculate confidence in classification
        confidence_score = self._calculate_classification_confidence(
            returns, regime_type
        )

        return RegimeProfile(
            state_id=state_id,
            regime_type=regime_type,
            mean_daily_return=mean_daily_return,
            daily_volatility=daily_volatility,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            persistence_days=persistence_days,
            regime_strength=0.0,  # Will be calculated later
            confidence_score=confidence_score,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            return_skewness=return_skewness,
            return_kurtosis=return_kurtosis,
            avg_duration=avg_duration,
            transition_volatility=transition_volatility,
        )

    def _classify_regime_type(
        self,
        mean_return: float,
        volatility: float,
        win_rate: float,
        max_drawdown: float,
    ) -> RegimeType:
        """
        Classify regime based on financial characteristics.

        Uses empirical thresholds based on typical market behavior.
        """
        # Convert to annualized values for easier interpretation
        ann_return = mean_return * 252
        ann_volatility = volatility * np.sqrt(252)

        # Crisis detection (extreme volatility)
        if ann_volatility > 0.50 or abs(max_drawdown) > 0.30:
            return RegimeType.CRISIS

        # Bullish regime: positive returns with reasonable volatility
        if ann_return > 0.10 and win_rate > 0.55 and ann_volatility < 0.40:
            return RegimeType.BULLISH

        # Bearish regime: negative returns
        if ann_return < -0.05 and win_rate < 0.45:
            return RegimeType.BEARISH

        # Sideways: low absolute returns
        if abs(ann_return) < 0.05 and ann_volatility < 0.25:
            return RegimeType.SIDEWAYS

        # Mixed: doesn't fit clear patterns
        return RegimeType.MIXED

    def _calculate_persistence(
        self, state_data: pd.DataFrame, full_data: pd.DataFrame, state_id: int
    ) -> float:
        """Calculate average persistence (days before switching) for this regime."""

        # Find regime transitions
        regime_series = full_data["predicted_state"]
        transitions = []
        current_regime = regime_series.iloc[0]
        start_idx = 0

        for i in range(1, len(regime_series)):
            if regime_series.iloc[i] != current_regime:
                if current_regime == state_id:
                    duration = i - start_idx
                    transitions.append(duration)
                current_regime = regime_series.iloc[i]
                start_idx = i

        # Handle final regime
        if current_regime == state_id:
            duration = len(regime_series) - start_idx
            transitions.append(duration)

        return np.mean(transitions) if transitions else 1.0

    def _calculate_average_duration(
        self, full_data: pd.DataFrame, state_id: int
    ) -> float:
        """Calculate average duration of regime episodes."""
        return self._calculate_persistence(
            full_data[full_data["predicted_state"] == state_id], full_data, state_id
        )

    def _calculate_transition_volatility(
        self, full_data: pd.DataFrame, state_id: int
    ) -> float:
        """Calculate volatility around regime transitions."""

        regime_series = full_data["predicted_state"]
        returns = full_data[self.return_column]

        # Find transitions into this regime
        transition_returns = []

        for i in range(1, len(regime_series)):
            if (
                regime_series.iloc[i] == state_id
                and regime_series.iloc[i - 1] != state_id
            ):
                # Transition into this regime - look at surrounding volatility
                start_idx = max(0, i - 2)
                end_idx = min(len(returns), i + 3)
                transition_period_returns = returns.iloc[start_idx:end_idx]
                transition_returns.extend(transition_period_returns.tolist())

        if len(transition_returns) > 1:
            return np.std(transition_returns)
        else:
            return returns.std()  # Fallback to overall volatility

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown during this regime."""
        if len(prices) == 0:
            return 0.0

        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_classification_confidence(
        self, returns: pd.Series, regime_type: RegimeType
    ) -> float:
        """
        Calculate confidence in regime classification.

        Higher confidence when regime characteristics are clear and consistent.
        """
        if len(returns) < 5:
            return 0.0

        # Base confidence on consistency of returns with regime type
        ann_return = returns.mean() * 252
        win_rate = (returns > 0).mean()

        if regime_type == RegimeType.BULLISH:
            # Bullish should have positive returns and high win rate
            return_confidence = max(0, min(1, (ann_return + 0.1) / 0.3))
            win_confidence = max(0, min(1, (win_rate - 0.4) / 0.3))
            return (return_confidence + win_confidence) / 2

        elif regime_type == RegimeType.BEARISH:
            # Bearish should have negative returns and low win rate
            return_confidence = max(0, min(1, (-ann_return) / 0.2))
            win_confidence = max(0, min(1, (0.6 - win_rate) / 0.3))
            return (return_confidence + win_confidence) / 2

        elif regime_type == RegimeType.SIDEWAYS:
            # Sideways should have low absolute returns
            return_confidence = max(0, min(1, 1 - abs(ann_return) / 0.1))
            return return_confidence

        elif regime_type == RegimeType.CRISIS:
            # Crisis should have high volatility
            volatility = returns.std() * np.sqrt(252)
            vol_confidence = max(0, min(1, (volatility - 0.3) / 0.4))
            return vol_confidence

        else:  # MIXED
            return 0.5  # Medium confidence for mixed regimes

    def _calculate_regime_strengths(
        self, regime_profiles: Dict[int, RegimeProfile]
    ) -> None:
        """
        Calculate relative strength of each regime compared to others.

        Strength indicates how distinct this regime is from the baseline.
        """
        if len(regime_profiles) < 2:
            for profile in regime_profiles.values():
                profile.regime_strength = 1.0
            return

        # Calculate overall market statistics
        all_returns = []
        all_volatilities = []

        for profile in regime_profiles.values():
            all_returns.append(profile.mean_daily_return)
            all_volatilities.append(profile.daily_volatility)

        baseline_return = np.mean(all_returns)
        baseline_volatility = np.mean(all_volatilities)

        # Calculate strength as deviation from baseline
        for profile in regime_profiles.values():
            return_deviation = abs(profile.mean_daily_return - baseline_return)
            vol_deviation = abs(profile.daily_volatility - baseline_volatility)

            # Normalize by baseline values
            return_strength = return_deviation / (abs(baseline_return) + 0.001)
            vol_strength = vol_deviation / (baseline_volatility + 0.001)

            # Combine return and volatility strength
            profile.regime_strength = min(1.0, (return_strength + vol_strength) / 2)

    def get_regime_summary(self, regime_profiles: Dict[int, RegimeProfile]) -> str:
        """Generate human-readable summary of regime characteristics."""

        if not regime_profiles:
            return "No regimes characterized."

        summary_lines = ["Regime Characterization Summary:", "=" * 40]

        for state_id, profile in regime_profiles.items():
            summary_lines.extend(
                [
                    f"\nState {state_id}: {profile.regime_type.value.upper()}",
                    f"  Annual Return: {profile.annualized_return:.1%}",
                    f"  Annual Volatility: {profile.annualized_volatility:.1%}",
                    f"  Win Rate: {profile.win_rate:.1%}",
                    f"  Max Drawdown: {profile.max_drawdown:.1%}",
                    f"  Avg Duration: {profile.avg_duration:.1f} days",
                    f"  Strength: {profile.regime_strength:.2f}",
                    f"  Confidence: {profile.confidence_score:.2f}",
                ]
            )

        return "\n".join(summary_lines)

    def suggest_trading_approach(
        self, regime_profiles: Dict[int, RegimeProfile]
    ) -> Dict[int, str]:
        """Suggest appropriate trading approach for each regime."""

        suggestions = {}

        for state_id, profile in regime_profiles.items():
            if profile.regime_type == RegimeType.BULLISH:
                if profile.confidence_score > 0.7:
                    suggestions[state_id] = "Strong long bias - increase position size"
                else:
                    suggestions[state_id] = (
                        "Moderate long bias - standard position size"
                    )

            elif profile.regime_type == RegimeType.BEARISH:
                if profile.confidence_score > 0.7:
                    suggestions[state_id] = "Strong short bias - defensive positioning"
                else:
                    suggestions[state_id] = "Moderate short bias - reduced exposure"

            elif profile.regime_type == RegimeType.SIDEWAYS:
                suggestions[state_id] = "Range trading - mean reversion strategies"

            elif profile.regime_type == RegimeType.CRISIS:
                suggestions[state_id] = "Risk-off - minimize exposure, cash preference"

            else:  # MIXED
                suggestions[state_id] = "Unclear signals - conservative positioning"

        return suggestions
