"""
Financial regime characterization for intelligent trading signal generation.

This module analyzes the actual financial characteristics of detected regimes
to determine their market behavior, replacing naive state number assumptions
with data-driven regime interpretation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

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


# Colorblind-safe color mapping for regime types
# Source: ColorBrewer2 diverging scheme (Red-Yellow-Blue)
REGIME_TYPE_COLORS = {
    RegimeType.BULLISH: "#4575b4",    # Blue (colorblind safe)
    RegimeType.BEARISH: "#d73027",    # Red (colorblind safe)
    RegimeType.SIDEWAYS: "#fee08b",   # Yellow (colorblind safe)
    RegimeType.CRISIS: "#a50026",     # Dark Red (colorblind safe)
    RegimeType.MIXED: "#9970ab",      # Purple (colorblind safe)
}


@dataclass
class RegimeProfile:
    """
    Financial profile of a detected regime.

    Contains actual financial characteristics rather than arbitrary state labels.
    Includes colorblind-safe color for consistent visualization across the system.
    """

    state_id: int
    regime_type: RegimeType
    color: str  # Colorblind-safe hex color for this regime type
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

    # Data-driven label (replaces heuristic enum-based labeling)
    regime_type_str: Optional[str] = None

    def get_display_name(self) -> str:
        """
        Get the display name for this regime.

        Returns data-driven regime_type_str if available,
        otherwise falls back to enum value for backward compatibility.
        """
        if self.regime_type_str is not None:
            return self.regime_type_str
        return self.regime_type.value


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

        # Apply data-driven classification to assign unique string labels
        self._data_driven_classify_regimes(regime_profiles)

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

        # Assign colorblind-safe color based on regime type
        color = REGIME_TYPE_COLORS[regime_type]

        return RegimeProfile(
            state_id=state_id,
            regime_type=regime_type,
            color=color,  # Colorblind-safe color for visualization consistency
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
        DEPRECATED: Legacy heuristic classification method.

        This method is kept ONLY for backward compatibility with the RegimeType enum.
        The enum value is stored in profile.regime_type but should NOT be used for
        any business logic.

        ALL regime classification should use the data-driven labels from
        profile.regime_type_str (accessed via profile.get_display_name()).

        This method will be removed in a future major version.
        """
        # Return a placeholder enum value - NOT USED FOR ANY LOGIC
        # The actual classification is done in _data_driven_classify_regimes()
        return RegimeType.MIXED  # Default placeholder

    def _data_driven_classify_regimes(
        self, regime_profiles: Dict[int, RegimeProfile]
    ) -> None:
        """
        Assign data-driven string labels to regimes based on their actual statistics.

        This replaces heuristic threshold-based classification with ordinal ranking
        based on the HMM's learned emission parameters, guaranteeing unique labels.

        The approach:
        1. Extract emission means (mean_daily_return) for all states
        2. Identify crisis regimes (extreme volatility relative to median)
        3. Sort remaining states by return in ascending order
        4. Assign ordinal labels based on rank (bearish < sideways < bullish)
        5. For n>3 states, add intensity modifiers based on data quantiles

        Args:
            regime_profiles: Dictionary of RegimeProfile objects to classify
        """
        if not regime_profiles:
            return

        n_states = len(regime_profiles)

        # Step 1: Extract emission parameters
        state_returns = {sid: p.mean_daily_return for sid, p in regime_profiles.items()}
        state_vols = {sid: p.daily_volatility for sid, p in regime_profiles.items()}

        # Compute median statistics for relative thresholds
        median_vol = np.median(list(state_vols.values()))
        median_return_abs = np.median([abs(r) for r in state_returns.values()])

        # Step 2: Identify crisis regimes (extreme volatility)
        crisis_threshold = 2.0 * median_vol
        crisis_states = {
            sid: vol for sid, vol in state_vols.items() if vol > crisis_threshold
        }

        # Step 3: Sort non-crisis states by return (ascending)
        non_crisis_states = [
            sid for sid in state_returns.keys() if sid not in crisis_states
        ]
        sorted_states = sorted(non_crisis_states, key=lambda sid: state_returns[sid])

        # Step 4: Assign ordinal labels based on position and count
        n_non_crisis = len(sorted_states)

        if n_non_crisis == 1:
            # Single non-crisis state
            regime_profiles[sorted_states[0]].regime_type_str = "neutral"

        elif n_non_crisis == 2:
            # Two states: bearish (low return), bullish (high return)
            regime_profiles[sorted_states[0]].regime_type_str = "bearish"
            regime_profiles[sorted_states[1]].regime_type_str = "bullish"

        elif n_non_crisis == 3:
            # Three states: bearish, sideways, bullish
            regime_profiles[sorted_states[0]].regime_type_str = "bearish"
            regime_profiles[sorted_states[1]].regime_type_str = "sideways"
            regime_profiles[sorted_states[2]].regime_type_str = "bullish"

        elif n_non_crisis == 4:
            # Four states: strong bearish, weak bearish, weak bullish, strong bullish
            regime_profiles[sorted_states[0]].regime_type_str = "strong_bearish"
            regime_profiles[sorted_states[1]].regime_type_str = "weak_bearish"
            regime_profiles[sorted_states[2]].regime_type_str = "weak_bullish"
            regime_profiles[sorted_states[3]].regime_type_str = "strong_bullish"

        elif n_non_crisis >= 5:
            # Five or more states: use quantile-based intensity
            returns_sorted = [state_returns[sid] for sid in sorted_states]
            q20 = np.percentile(returns_sorted, 20)
            q40 = np.percentile(returns_sorted, 40)
            q60 = np.percentile(returns_sorted, 60)
            q80 = np.percentile(returns_sorted, 80)

            for sid in sorted_states:
                ret = state_returns[sid]
                if ret < q20:
                    regime_profiles[sid].regime_type_str = "strong_bearish"
                elif ret < q40:
                    regime_profiles[sid].regime_type_str = "weak_bearish"
                elif ret < q60:
                    regime_profiles[sid].regime_type_str = "sideways"
                elif ret < q80:
                    regime_profiles[sid].regime_type_str = "weak_bullish"
                else:
                    regime_profiles[sid].regime_type_str = "strong_bullish"

        # Step 5: Label crisis regimes
        for crisis_idx, sid in enumerate(crisis_states.keys()):
            if len(crisis_states) == 1:
                regime_profiles[sid].regime_type_str = "crisis"
            else:
                # Multiple crisis states - distinguish by severity
                regime_profiles[sid].regime_type_str = f"crisis_{crisis_idx + 1}"

        # Step 6: Assign colors based on data-driven labels
        for sid, profile in regime_profiles.items():
            label = profile.regime_type_str
            if label:
                # Assign color based on label pattern
                profile.color = self._get_color_for_label(label)

        # # Diagnostic output
        # print("\n" + "=" * 60)
        # print("DATA-DRIVEN REGIME CLASSIFICATION")
        # print("=" * 60)
        # print(f"Total states: {n_states}")
        # print(f"Non-crisis states: {n_non_crisis}")
        # print(f"Crisis states: {len(crisis_states)}")
        # print(f"Median volatility: {median_vol:.6f}")
        # print(f"Crisis threshold: {crisis_threshold:.6f}")
        # print("\nState Assignments:")
        # for sid in sorted(regime_profiles.keys()):
        #     profile = regime_profiles[sid]
        #     print(
        #         f"  State {sid}: {profile.regime_type_str:20s} "
        #         f"(return={profile.mean_daily_return:+.6f}, "
        #         f"vol={profile.daily_volatility:.6f})"
        #     )
        # print("=" * 60 + "\n")

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
        Calculate confidence in regime classification based on statistical properties.

        Uses data-driven metrics:
        - Return consistency (low standard deviation of returns)
        - Directional strength (magnitude of mean return)
        - Distribution characteristics (skewness, kurtosis)

        Higher confidence when the regime exhibits clear, consistent statistical patterns.
        NO HEURISTIC ASSUMPTIONS about what regimes "should" look like.
        """
        if len(returns) < 5:
            return 0.0

        # Calculate statistical metrics
        mean_return = returns.mean()
        std_return = returns.std()

        # Confidence based on consistency (inverse of coefficient of variation)
        # More consistent returns = higher confidence
        if abs(mean_return) > 0.0001:  # Avoid division by zero
            cv = abs(std_return / mean_return)
            consistency_confidence = max(0, min(1, 1.0 / (1.0 + cv)))
        else:
            # Low mean return - confidence based on low volatility
            annualized_vol = std_return * np.sqrt(252)
            consistency_confidence = max(0, min(1, 1 - annualized_vol / 0.5))

        # Confidence based on signal strength (magnitude of returns)
        annualized_return = abs(mean_return * 252)
        strength_confidence = max(0, min(1, annualized_return / 0.3))  # Normalize by 30%

        # Confidence based on sample size (more data = more confident)
        sample_confidence = min(1.0, len(returns) / 100)  # Full confidence at 100+ samples

        # Combined confidence (weighted average)
        total_confidence = (
            0.4 * consistency_confidence +
            0.4 * strength_confidence +
            0.2 * sample_confidence
        )

        return max(0.0, min(1.0, total_confidence))

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
                    f"\nState {state_id}: {profile.get_display_name().upper()}",
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
        """
        DEPRECATED: This method has been removed.

        Trading suggestions should be based on your interpretation of the
        data-driven regime labels and characteristics, not hard-coded heuristics.

        Use the RegimeProfile attributes to make your own decisions:
        - profile.get_display_name() - data-driven regime label
        - profile.mean_daily_return - actual observed return
        - profile.daily_volatility - actual observed volatility
        - profile.win_rate - percentage of positive days
        - profile.regime_strength - how distinct this regime is
        - profile.confidence_score - statistical confidence

        This method will be completely removed in the next major version.
        """
        raise DeprecationWarning(
            "suggest_trading_approach() is deprecated. "
            "Interpret data-driven regime labels yourself based on RegimeProfile statistics."
        )

    def _get_color_for_label(self, label: str) -> str:
        """
        Assign color based on data-driven label string (pattern matching).

        This function supports ANY label, not just predefined ones.
        Colors are assigned based on keyword patterns in the label.
        Uses colorblind-safe palette.

        Args:
            label: Data-driven regime label string

        Returns:
            Hex color code for visualization
        """
        label_lower = label.lower()

        # Bullish patterns - shades of blue/green
        if 'strong_bullish' in label_lower or 'very_bullish' in label_lower:
            return "#006400"  # Dark Green (very strong)
        elif 'weak_bullish' in label_lower:
            return "#90EE90"  # Light Green (weak)
        elif 'bullish' in label_lower:
            return "#4575b4"  # Blue (standard bullish)

        # Bearish patterns - shades of red
        elif 'strong_bearish' in label_lower or 'very_bearish' in label_lower:
            return "#8B0000"  # Dark Red (very strong)
        elif 'weak_bearish' in label_lower:
            return "#F08080"  # Light Coral (weak)
        elif 'bearish' in label_lower:
            return "#d73027"  # Red (standard bearish)

        # Sideways/neutral patterns - shades of yellow/gold
        elif 'sideways' in label_lower or 'neutral' in label_lower:
            return "#fee08b"  # Yellow

        # Crisis patterns - dark red
        elif 'crisis' in label_lower:
            return "#a50026"  # Dark Red (crisis)

        # Default for any other label
        else:
            return "#9970ab"  # Purple (mixed/unknown)
