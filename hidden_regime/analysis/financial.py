"""
Financial analysis component for pipeline architecture.

Provides FinancialAnalysis that implements AnalysisComponent interface for
interpreting HMM regime predictions in financial context with domain knowledge.
"""

import warnings
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config.analysis import FinancialAnalysisConfig
from ..pipeline.interfaces import AnalysisComponent
from ..utils.exceptions import ValidationError
from ..utils.state_mapping import (
    apply_regime_mapping_to_analysis,
    get_regime_characteristics,
    map_states_to_financial_regimes,
)
from .performance import RegimePerformanceAnalyzer

# Try to import technical indicators
try:
    from .indicator_comparison import IndicatorPerformanceComparator
    from .technical_indicators import TechnicalIndicatorAnalyzer

    INDICATORS_AVAILABLE = True
except ImportError as e:
    INDICATORS_AVAILABLE = False
    warnings.warn(
        f"Technical indicators not available - indicator comparisons disabled: {e}"
    )


class FinancialAnalysis(AnalysisComponent):
    """
    Financial analysis component for interpreting HMM regime predictions.

    Implements AnalysisComponent interface to provide financial domain knowledge
    interpretation of regime states including performance metrics and trading insights.
    """

    def __init__(self, config: FinancialAnalysisConfig):
        """
        Initialize financial analysis with configuration.

        Args:
            config: FinancialAnalysisConfig with analysis parameters
        """
        self.config = config
        self._last_model_output = None
        self._last_analysis = None
        self._last_raw_data = None

        # Get regime labels
        self.regime_labels = config.get_default_regime_labels()

        # Cache for computed statistics
        self._regime_stats_cache = {}

        # Lazy import to avoid circular dependency
        # FinancialRegimeCharacterizer will be initialized on first use
        self.regime_characterizer = None
        self._regime_profiles = {}  # Cache for regime profiles

        # Initialize indicator calculator if available
        if INDICATORS_AVAILABLE and self.config.indicator_comparisons:
            self.indicator_analyzer = TechnicalIndicatorAnalyzer()
            self.indicator_comparator = IndicatorPerformanceComparator()
            self._indicators_cache = {}
        else:
            self.indicator_analyzer = None
            self.indicator_comparator = None
            self._indicators_cache = {}

        # Initialize performance analyzer
        self.performance_analyzer = RegimePerformanceAnalyzer()

    def update(
        self,
        model_output: pd.DataFrame,
        raw_data: Optional[pd.DataFrame] = None,
        model_component: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Interpret model output and add domain knowledge.

        Args:
            model_output: Raw model predictions with predicted_state and confidence
            raw_data: Optional raw OHLCV data for indicator calculations
            model_component: Optional model component to get emission parameters for data-driven interpretation

        Returns:
            DataFrame with interpreted analysis results
        """
        if model_output.empty:
            raise ValidationError("Model output cannot be empty")

        # Validate required columns
        required_cols = ["predicted_state", "confidence"]
        missing_cols = [col for col in required_cols if col not in model_output.columns]
        if missing_cols:
            raise ValidationError(
                f"Required columns missing from model output: {missing_cols}"
            )

        # Store references for plotting and indicator calculations
        self._last_model_output = model_output.copy()
        self._last_raw_data = raw_data.copy() if raw_data is not None else None

        # Start with model output
        analysis = model_output.copy()

        # Add regime interpretations using data-driven state mapping
        analysis = self._add_regime_interpretations(analysis, model_component)

        # Add regime statistics if requested
        if self.config.calculate_regime_statistics:
            analysis = self._add_regime_statistics(analysis)

        # Add duration analysis if requested
        if self.config.include_duration_analysis:
            analysis = self._add_duration_analysis(analysis)

        # Add return analysis if requested
        if self.config.include_return_analysis:
            analysis = self._add_return_analysis(analysis)

        # Add volatility analysis if requested
        if self.config.include_volatility_analysis:
            analysis = self._add_volatility_analysis(analysis)

        # Add indicator comparisons if requested and available
        if (
            self.config.include_indicator_performance
            and self.config.indicator_comparisons
            and raw_data is not None
        ):
            analysis = self._add_indicator_comparisons(analysis, raw_data)

        # Add trading signals if requested
        if self.config.include_trading_signals:
            analysis = self._add_trading_signals(analysis)

        # Store for plotting
        self._last_analysis = analysis.copy()

        return analysis

    def _add_regime_interpretations(
        self, analysis: pd.DataFrame, model_component: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Add regime name interpretations using FinancialRegimeCharacterizer.

        This method uses comprehensive financial analysis to characterize regimes
        based on actual returns, volatility, win rates, and other market metrics.
        """
        # Check for user-provided regime label override
        force_regime_labels = getattr(self.config, "force_regime_labels", None)
        acknowledge_override = getattr(self.config, "acknowledge_override", False)

        if force_regime_labels is not None and acknowledge_override:
            # User override path: Apply custom labels without validation
            warnings.warn(
                "Using user-provided regime labels. Data-driven validation is bypassed. "
                "Ensure labels match actual market behavior for meaningful analysis.",
                UserWarning,
            )

            # Create direct mapping from state index to user labels
            state_mapping = {i: label for i, label in enumerate(force_regime_labels)}
            analysis["regime_name"] = analysis["predicted_state"].map(state_mapping)
            analysis["regime_type"] = analysis["regime_name"]

            # Store the mapping for future reference
            self._current_state_mapping = state_mapping

            # Add generic characteristics since we can't validate user labels
            for _, row in analysis.iterrows():
                regime_name = row["regime_name"]
                # Use neutral characteristics for user-defined regimes
                analysis.loc[row.name, "expected_return"] = 0.0
                analysis.loc[row.name, "expected_volatility"] = 0.015
                analysis.loc[row.name, "expected_duration"] = 10.0

            return analysis

        # DATA-DRIVEN PATH: Use FinancialRegimeCharacterizer for comprehensive analysis
        # Fall back to simple mapping if raw data is not available
        if self._last_raw_data is None or "log_return" not in self._last_raw_data.columns:
            warnings.warn(
                "Raw data with 'log_return' column not provided. "
                "Falling back to simple emission means-based regime mapping. "
                "For full regime characterization, provide raw_data to update() method.",
                UserWarning,
            )

            # Fall back to simple state mapping
            if model_component is None or not hasattr(
                model_component, "emission_means_"
            ):
                raise ValidationError(
                    "Model component with emission_means_ is required for regime interpretation "
                    "when raw data is not provided."
                )

            emission_means = model_component.emission_means_
            if len(emission_means) != self.config.n_states:
                raise ValidationError(
                    f"Emission means length ({len(emission_means)}) does not match "
                    f"configured n_states ({self.config.n_states})"
                )

            # Apply simple state mapping
            analysis = apply_regime_mapping_to_analysis(
                analysis, emission_means, self.config.n_states
            )
            self._current_state_mapping = map_states_to_financial_regimes(
                emission_means, self.config.n_states
            )

            return analysis

        try:
            # Lazy import to avoid circular dependency
            if self.regime_characterizer is None:
                from ..financial.regime_characterizer import (
                    FinancialRegimeCharacterizer,
                    RegimeType,
                )

                self.regime_characterizer = FinancialRegimeCharacterizer(
                    min_regime_days=5,
                    return_column="log_return",
                    price_column="close",
                )
                # Store RegimeType for later use
                self._RegimeType = RegimeType

            # Use FinancialRegimeCharacterizer to analyze regimes
            regime_profiles = self.regime_characterizer.characterize_regimes(
                regime_data=analysis, price_data=self._last_raw_data
            )

            # Store regime profiles for later use
            self._regime_profiles = regime_profiles

            # Map regime characteristics to analysis DataFrame
            for state_id, profile in regime_profiles.items():
                state_mask = analysis["predicted_state"] == state_id

                # Add regime type and name from characterization
                # Use data-driven label if available, otherwise fall back to enum
                display_name = profile.get_display_name()
                analysis.loc[state_mask, "regime_type"] = display_name
                analysis.loc[state_mask, "regime_name"] = display_name

                # Add financial characteristics from profile
                analysis.loc[state_mask, "expected_return"] = profile.mean_daily_return
                analysis.loc[
                    state_mask, "expected_volatility"
                ] = profile.daily_volatility
                analysis.loc[state_mask, "expected_duration"] = profile.avg_duration
                analysis.loc[state_mask, "win_rate"] = profile.win_rate
                analysis.loc[state_mask, "max_drawdown"] = profile.max_drawdown
                analysis.loc[state_mask, "regime_strength"] = profile.regime_strength
                analysis.loc[
                    state_mask, "characterization_confidence"
                ] = profile.confidence_score

            # Store state mapping for reference (use data-driven labels)
            self._current_state_mapping = {
                state_id: profile.get_display_name()
                for state_id, profile in regime_profiles.items()
            }

            return analysis

        except Exception as e:
            raise ValidationError(
                f"Regime characterization failed: {e}. "
                f"This may indicate insufficient data, inappropriate model configuration, "
                f"or data that doesn't exhibit clear regime structure. "
                f"Consider: (1) more training data, (2) different n_states, or (3) checking data quality."
            ) from e


    @staticmethod
    def assess_data_for_regime_detection(
        raw_data: pd.DataFrame, observed_signal: str = "log_return"
    ) -> Dict[str, Any]:
        """
        Assess raw data to determine optimal regime detection approach.

        Args:
            raw_data: Raw financial data with price and return information
            observed_signal: Column name for returns (default: 'log_return')

        Returns:
            Dictionary with data assessment and recommendations
        """
        if observed_signal not in raw_data.columns:
            raise ValidationError(
                f"Observed signal '{observed_signal}' not found in data columns: {list(raw_data.columns)}"
            )

        returns = raw_data[observed_signal].dropna()
        if len(returns) < 10:
            raise ValidationError(
                f"Insufficient data: only {len(returns)} valid returns available. Need at least 10 observations."
            )

        # Convert to percentage space for analysis
        returns_pct = np.exp(returns) - 1

        # Basic statistics
        mean_return = returns_pct.mean()
        std_return = returns_pct.std()
        min_return = returns_pct.min()
        max_return = returns_pct.max()

        # Trend analysis
        positive_returns = returns_pct > 0.001  # > 0.1% daily
        negative_returns = returns_pct < -0.001  # < -0.1% daily
        neutral_returns = abs(returns_pct) <= 0.001  # ±0.1% daily

        pct_positive = positive_returns.mean()
        pct_negative = negative_returns.mean()
        pct_neutral = neutral_returns.mean()

        # Volatility clustering (simple measure)
        volatility = returns_pct.rolling(5).std().dropna()
        high_vol_periods = volatility > volatility.quantile(0.75)
        low_vol_periods = volatility < volatility.quantile(0.25)

        # Regime switching indicators
        return_spread = max_return - min_return
        consecutive_trends = FinancialAnalysis._detect_consecutive_trends(returns_pct)

        # Determine regime feasibility
        assessment = {
            "data_summary": {
                "n_observations": len(returns),
                "mean_daily_return": f"{mean_return:.3%}",
                "daily_volatility": f"{std_return:.3%}",
                "return_range": f"{min_return:.3%} to {max_return:.3%}",
                "return_spread": f"{return_spread:.3%}",
            },
            "return_distribution": {
                "pct_positive_days": f"{pct_positive:.1%}",
                "pct_negative_days": f"{pct_negative:.1%}",
                "pct_neutral_days": f"{pct_neutral:.1%}",
            },
            "regime_indicators": {
                "has_distinct_volatility_regimes": high_vol_periods.sum()
                > len(volatility) * 0.1,
                "max_consecutive_up_days": consecutive_trends["max_up"],
                "max_consecutive_down_days": consecutive_trends["max_down"],
                "return_spread_adequate": return_spread > 0.02,  # >2% daily spread
            },
        }

        # Generate recommendations
        recommendations = FinancialAnalysis._generate_regime_recommendations(
            mean_return, pct_positive, pct_negative, return_spread, consecutive_trends
        )

        assessment["recommendations"] = recommendations

        return assessment

    @staticmethod
    def _detect_consecutive_trends(returns_pct: pd.Series) -> Dict[str, int]:
        """Detect consecutive trend periods in returns."""
        up_days = returns_pct > 0.001
        down_days = returns_pct < -0.001

        # Find consecutive runs
        up_runs = []
        down_runs = []
        current_up = 0
        current_down = 0

        for up, down in zip(up_days, down_days):
            if up:
                current_up += 1
                if current_down > 0:
                    down_runs.append(current_down)
                    current_down = 0
            elif down:
                current_down += 1
                if current_up > 0:
                    up_runs.append(current_up)
                    current_up = 0
            else:
                if current_up > 0:
                    up_runs.append(current_up)
                    current_up = 0
                if current_down > 0:
                    down_runs.append(current_down)
                    current_down = 0

        # Add final runs
        if current_up > 0:
            up_runs.append(current_up)
        if current_down > 0:
            down_runs.append(current_down)

        return {
            "max_up": max(up_runs) if up_runs else 0,
            "max_down": max(down_runs) if down_runs else 0,
            "avg_up": np.mean(up_runs) if up_runs else 0,
            "avg_down": np.mean(down_runs) if down_runs else 0,
        }

    @staticmethod
    def _generate_regime_recommendations(
        mean_return: float,
        pct_positive: float,
        pct_negative: float,
        return_spread: float,
        consecutive_trends: Dict[str, int],
    ) -> Dict[str, Any]:
        """Generate regime detection recommendations based on data characteristics."""

        # Determine if data exhibits regime-switching behavior
        is_monotonic_up = (
            pct_positive > 0.8 and mean_return > 0.005
        )  # >80% up days, >0.5% daily mean
        is_monotonic_down = (
            pct_negative > 0.8 and mean_return < -0.005
        )  # >80% down days, <-0.5% daily mean
        is_sideways = (
            abs(mean_return) < 0.001 and pct_positive < 0.6 and pct_negative < 0.6
        )  # Neutral with mixed days

        long_trends = (
            consecutive_trends["max_up"] > 10 or consecutive_trends["max_down"] > 10
        )
        small_spread = return_spread < 0.015  # <1.5% daily spread

        recommendations = {
            "suitable_for_regime_detection": True,
            "recommended_n_states": 3,
            "regime_detection_approach": "standard",
            "warnings": [],
            "rationale": [],
        }

        # Analyze suitability
        if is_monotonic_up:
            recommendations["suitable_for_regime_detection"] = False
            recommendations["warnings"].append(
                "Data shows monotonic upward trend (>80% positive days). "
                "Regime detection may create artificial distinctions."
            )
            recommendations["recommended_n_states"] = 2
            recommendations["regime_detection_approach"] = "trend_following"

        elif is_monotonic_down:
            recommendations["suitable_for_regime_detection"] = False
            recommendations["warnings"].append(
                "Data shows monotonic downward trend (>80% negative days). "
                "Regime detection may create artificial distinctions."
            )
            recommendations["recommended_n_states"] = 2
            recommendations["regime_detection_approach"] = "trend_following"

        elif small_spread:
            recommendations["warnings"].append(
                f"Small return spread ({return_spread:.2%}) may not support distinct regimes. "
                "Consider using 2 states or more training data."
            )
            recommendations["recommended_n_states"] = 2

        elif long_trends:
            recommendations["warnings"].append(
                "Long consecutive trend periods detected. Data may be more trending than regime-switching."
            )

        # Positive indicators for regime detection
        if is_sideways:
            recommendations["rationale"].append(
                "Mixed positive/negative days with neutral mean suggests regime-switching behavior."
            )

        if return_spread > 0.03:  # >3% spread
            recommendations["rationale"].append(
                "Large return spread supports distinct regime identification."
            )
            recommendations["recommended_n_states"] = min(
                4, recommendations["recommended_n_states"] + 1
            )

        if 0.3 < pct_positive < 0.7:  # Balanced positive/negative days
            recommendations["rationale"].append(
                "Balanced distribution of positive/negative days supports regime detection."
            )

        # Final recommendation
        if not recommendations["rationale"]:
            recommendations["rationale"].append(
                "Data characteristics are marginal for regime detection."
            )

        return recommendations

    def _add_regime_statistics(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """Add basic regime statistics."""
        # Calculate days in current regime
        analysis["days_in_regime"] = self._calculate_days_in_regime(
            analysis["predicted_state"]
        )

        # Add expected regime characteristics (simplified)
        regime_characteristics = self._get_regime_characteristics()

        for char, values in regime_characteristics.items():
            analysis[f"expected_{char}"] = analysis["predicted_state"].map(
                {
                    i: values[i] if i < len(values) else 0.0
                    for i in range(self.config.n_states)
                }
            )

        return analysis

    def _calculate_days_in_regime(self, states: pd.Series) -> pd.Series:
        """Calculate number of consecutive days in current regime."""
        days_in_regime = []
        current_count = 1

        for i in range(len(states)):
            if i == 0:
                days_in_regime.append(current_count)
            elif states.iloc[i] == states.iloc[i - 1]:
                current_count += 1
                days_in_regime.append(current_count)
            else:
                current_count = 1
                days_in_regime.append(current_count)

        return pd.Series(days_in_regime, index=states.index)

    def _get_regime_characteristics(self) -> Dict[str, List[float]]:
        """Get expected regime characteristics from stored regime profiles."""
        # Use stored regime profiles from FinancialRegimeCharacterizer if available
        if self._regime_profiles:
            try:
                characteristics = {
                    "return": [],
                    "volatility": [],
                    "duration": [],
                }

                # Get characteristics in state ID order
                for state_idx in range(self.config.n_states):
                    if state_idx in self._regime_profiles:
                        profile = self._regime_profiles[state_idx]
                        characteristics["return"].append(profile.mean_daily_return)
                        characteristics["volatility"].append(profile.daily_volatility)
                        characteristics["duration"].append(profile.avg_duration)
                    else:
                        # Fallback for missing states
                        characteristics["return"].append(0.0)
                        characteristics["volatility"].append(0.015)
                        characteristics["duration"].append(10.0)

                return characteristics

            except Exception as e:
                warnings.warn(
                    f"Failed to retrieve regime characteristics from profiles: {e}"
                )

        # Fallback to default characteristics based on financial knowledge
        if self.config.n_states == 3:
            return {
                "return": [-0.002, 0.0001, 0.001],  # Bear, Sideways, Bull daily returns
                "volatility": [0.025, 0.012, 0.018],  # Expected daily volatility
                "duration": [8.0, 15.0, 12.0],  # Expected regime duration in days
            }
        elif self.config.n_states == 4:
            return {
                "return": [
                    -0.005,
                    -0.002,
                    0.0001,
                    0.001,
                ],  # Crisis, Bear, Sideways, Bull
                "volatility": [0.040, 0.025, 0.012, 0.018],
                "duration": [5.0, 8.0, 15.0, 12.0],
            }
        else:
            # Default to neutral characteristics
            return {
                "return": [0.0] * self.config.n_states,
                "volatility": [0.015] * self.config.n_states,
                "duration": [10.0] * self.config.n_states,
            }

    def _calculate_regime_duration(self, state_idx: int) -> float:
        """Calculate average duration for a specific regime state."""
        if self._last_analysis is None:
            return 10.0

        # Find regime episodes for this state
        state_mask = self._last_analysis["predicted_state"] == state_idx

        if not state_mask.any():
            return 10.0

        # Calculate consecutive periods in this state
        state_series = (self._last_analysis["predicted_state"] == state_idx).astype(int)
        transitions = state_series.diff().fillna(0)

        # Find start and end of episodes
        episode_starts = transitions == 1
        episode_ends = transitions == -1

        durations = []
        current_start = None

        for i, is_start in enumerate(episode_starts):
            if is_start:
                current_start = i
            elif episode_ends.iloc[i] and current_start is not None:
                duration = i - current_start
                durations.append(duration)
                current_start = None

        # Handle ongoing episode at end
        if current_start is not None:
            duration = len(state_series) - current_start
            durations.append(duration)

        return np.mean(durations) if durations else 10.0

    def _add_duration_analysis(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """Add regime duration analysis."""
        # Calculate regime transitions
        transitions = (
            analysis["predicted_state"] != analysis["predicted_state"].shift(1)
        ).cumsum()
        analysis["regime_episode"] = transitions

        # Ensure required columns exist for duration analysis
        if "expected_duration" not in analysis.columns:
            # Add expected duration based on regime characteristics
            regime_characteristics = self._get_regime_characteristics()
            if "duration" in regime_characteristics:
                analysis["expected_duration"] = analysis["predicted_state"].map(
                    {
                        i: (
                            regime_characteristics["duration"][i]
                            if i < len(regime_characteristics["duration"])
                            else 10.0
                        )
                        for i in range(self.config.n_states)
                    }
                )
            else:
                # Fallback to default duration
                analysis["expected_duration"] = 10.0

        if "days_in_regime" not in analysis.columns:
            # Calculate days in regime if not already present
            analysis["days_in_regime"] = self._calculate_days_in_regime(
                analysis["predicted_state"]
            )

        # Calculate expected remaining duration
        analysis["expected_remaining_duration"] = analysis.apply(
            lambda row: max(1, row["expected_duration"] - row["days_in_regime"]), axis=1
        )

        return analysis

    def _add_return_analysis(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """Add return-based analysis using actual price data."""
        # Add rolling returns if we have sufficient data and raw data available
        if (
            len(analysis) >= self.config.return_window
            and self._last_raw_data is not None
        ):
            raw_data = self._last_raw_data

            # Calculate rolling cumulative returns
            if "log_return" in raw_data.columns:
                # Rolling return over specified window
                rolling_log_returns = (
                    raw_data["log_return"]
                    .rolling(window=self.config.return_window)
                    .sum()
                )
                analysis["rolling_return"] = (
                    np.exp(rolling_log_returns) - 1
                )  # Convert to percentage

                # Calculate regime-specific expected returns based on actual emission means
                if hasattr(self, "_current_state_mapping"):
                    # Use actual emission means for expected returns
                    expected_returns = {}
                    if hasattr(self, "_last_model_output") and hasattr(
                        self, "_current_state_mapping"
                    ):
                        # Try to get emission means from stored model component reference
                        for (
                            state_idx,
                            regime_name,
                        ) in self._current_state_mapping.items():
                            # Use actual daily log return as expected return
                            expected_returns[state_idx] = analysis[
                                analysis["predicted_state"] == state_idx
                            ]["rolling_return"].median()

                    # Map expected returns to analysis
                    analysis["expected_rolling_return"] = analysis[
                        "predicted_state"
                    ].map(expected_returns)

                    # Calculate return vs expected (actual performance relative to regime expectation)
                    analysis["return_vs_expected"] = (
                        analysis["rolling_return"] - analysis["expected_rolling_return"]
                    )
                else:
                    # Fallback to simple return comparison
                    analysis["return_vs_expected"] = (
                        analysis["rolling_return"]
                        - analysis["rolling_return"].rolling(window=50).mean()
                    )
            else:
                # Fallback if no log_return available
                analysis["rolling_return"] = 0.0
                analysis["return_vs_expected"] = 0.0
        else:
            # Not enough data for rolling calculations
            analysis["rolling_return"] = 0.0
            analysis["return_vs_expected"] = 0.0

        return analysis

    def _add_volatility_analysis(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """Add volatility analysis using actual price data."""
        # Add rolling volatility if we have sufficient data and raw data available
        if (
            len(analysis) >= self.config.volatility_window
            and self._last_raw_data is not None
        ):
            raw_data = self._last_raw_data

            # Calculate rolling volatility using log returns
            if "log_return" in raw_data.columns:
                # Standard rolling volatility (standard deviation of log returns)
                rolling_vol = (
                    raw_data["log_return"]
                    .rolling(window=self.config.volatility_window)
                    .std()
                )
                analysis["rolling_volatility"] = rolling_vol * np.sqrt(
                    252
                )  # Annualized volatility

                # Calculate regime-specific expected volatility based on actual data
                if hasattr(self, "_current_state_mapping"):
                    # Calculate actual volatility by regime from historical data
                    expected_volatilities = {}
                    for state_idx, regime_name in self._current_state_mapping.items():
                        state_mask = analysis["predicted_state"] == state_idx
                        if state_mask.any():
                            # Get log returns for this regime's periods
                            regime_dates = analysis[state_mask].index
                            regime_returns = raw_data.loc[regime_dates, "log_return"]

                            # Calculate regime-specific volatility
                            regime_vol = regime_returns.std() * np.sqrt(
                                252
                            )  # Annualized
                            expected_volatilities[state_idx] = regime_vol

                    # Map expected volatilities to analysis
                    analysis["expected_volatility"] = analysis["predicted_state"].map(
                        expected_volatilities
                    )

                    # Calculate volatility vs expected (current vol relative to regime average)
                    analysis["volatility_vs_expected"] = (
                        analysis["rolling_volatility"] - analysis["expected_volatility"]
                    )

                    # Add volatility regime classification
                    analysis["volatility_regime"] = analysis.apply(
                        lambda row: self._classify_volatility_regime(
                            row["rolling_volatility"],
                            row.get("expected_volatility", 0.15),
                        ),
                        axis=1,
                    )
                else:
                    # Fallback to simple volatility comparison
                    long_term_vol = raw_data["log_return"].std() * np.sqrt(252)
                    analysis["expected_volatility"] = long_term_vol
                    analysis["volatility_vs_expected"] = (
                        analysis["rolling_volatility"] - long_term_vol
                    )
                    analysis["volatility_regime"] = "Normal"
            else:
                # Fallback if no log_return available
                analysis["rolling_volatility"] = 0.0
                analysis["volatility_vs_expected"] = 0.0
                analysis["expected_volatility"] = 0.0
                analysis["volatility_regime"] = "Unknown"
        else:
            # Not enough data for rolling calculations
            analysis["rolling_volatility"] = 0.0
            analysis["volatility_vs_expected"] = 0.0
            analysis["expected_volatility"] = 0.0
            analysis["volatility_regime"] = "Unknown"

        return analysis

    def _classify_volatility_regime(
        self, current_vol: float, expected_vol: float
    ) -> str:
        """Classify current volatility relative to expected."""
        if current_vol > expected_vol * 1.5:
            return "High Volatility"
        elif current_vol > expected_vol * 1.2:
            return "Elevated Volatility"
        elif current_vol < expected_vol * 0.8:
            return "Low Volatility"
        else:
            return "Normal Volatility"

    def _add_indicator_comparisons(
        self, analysis: pd.DataFrame, raw_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add technical indicator comparisons."""
        if not INDICATORS_AVAILABLE or not self.indicator_analyzer:
            return analysis

        try:
            # Calculate indicators for the specified comparisons
            for indicator_name in self.config.indicator_comparisons:
                indicator_values = self._calculate_single_indicator(
                    indicator_name, raw_data
                )

                if indicator_values is not None and len(indicator_values) == len(
                    analysis
                ):
                    # Add indicator values
                    analysis[f"{indicator_name}_value"] = indicator_values

                    # Add indicator signals (simplified)
                    analysis[f"{indicator_name}_signal"] = self._get_indicator_signal(
                        indicator_name, indicator_values
                    )

                    # Add regime vs indicator agreement
                    analysis[f"{indicator_name}_agreement"] = (
                        self._calculate_regime_indicator_agreement(
                            analysis["predicted_state"],
                            analysis[f"{indicator_name}_signal"],
                        )
                    )

            # Add overall indicator consensus if multiple indicators
            if len(self.config.indicator_comparisons) > 1:
                signal_cols = [
                    f"{ind}_signal"
                    for ind in self.config.indicator_comparisons
                    if f"{ind}_signal" in analysis.columns
                ]
                if signal_cols:
                    analysis["indicator_consensus"] = analysis[signal_cols].mean(axis=1)
                    analysis["regime_consensus_agreement"] = (
                        self._calculate_regime_consensus_agreement(
                            analysis["predicted_state"], analysis["indicator_consensus"]
                        )
                    )

        except Exception as e:
            warnings.warn(f"Indicator comparison failed: {e}")

        return analysis

    def _calculate_single_indicator(
        self, indicator_name: str, raw_data: pd.DataFrame
    ) -> Optional[pd.Series]:
        """Calculate a single technical indicator."""
        try:
            # Simple implementations for common indicators
            if indicator_name.lower() == "rsi":
                if "close" in raw_data.columns:
                    return self._calculate_rsi(raw_data["close"])
            elif indicator_name.lower() == "macd":
                if "close" in raw_data.columns:
                    return self._calculate_macd_signal(raw_data["close"])
            elif indicator_name.lower() == "bollinger_bands":
                if "close" in raw_data.columns:
                    return self._calculate_bollinger_position(raw_data["close"])
            elif indicator_name.lower() == "moving_average":
                if "close" in raw_data.columns:
                    return self._calculate_ma_signal(raw_data["close"])

            # Fallback: try to use full indicators calculator if available
            if hasattr(self.indicator_analyzer, "calculate_all_indicators"):
                indicators = self.indicator_analyzer.calculate_all_indicators(raw_data)
                if indicator_name in indicators.columns:
                    return indicators[indicator_name]

        except Exception as e:
            warnings.warn(f"Failed to calculate {indicator_name}: {e}")

        return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd_signal(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.Series:
        """Calculate MACD signal line."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line  # MACD histogram

    def _calculate_bollinger_position(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> pd.Series:
        """Calculate position within Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return (prices - lower_band) / (upper_band - lower_band)

    def _calculate_ma_signal(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate moving average signal."""
        ma = prices.rolling(window=period).mean()
        return (prices - ma) / ma  # Relative position to MA

    def _get_indicator_signal(
        self, indicator_name: str, values: pd.Series
    ) -> pd.Series:
        """Convert indicator values to standardized signals (-1 to 1)."""
        if indicator_name.lower() == "rsi":
            # RSI: >70 overbought (sell), <30 oversold (buy)
            signals = np.where(values > 70, -1, np.where(values < 30, 1, 0))
            return pd.Series(signals, index=values.index)
        elif indicator_name.lower() == "macd":
            # MACD: positive = bullish, negative = bearish
            return np.sign(values)
        elif indicator_name.lower() == "bollinger_bands":
            # Bollinger position: >1 above upper band, <0 below lower band
            signals = np.where(values > 1, -1, np.where(values < 0, 1, 0))
            return pd.Series(signals, index=values.index)
        elif indicator_name.lower() == "moving_average":
            # MA signal: positive = above MA (bullish), negative = below MA (bearish)
            return np.sign(values)
        else:
            # Default: standardize to -1, 1 range
            return np.sign(values - values.median())

    def _calculate_regime_indicator_agreement(
        self, regime_states: pd.Series, indicator_signals: pd.Series
    ) -> pd.Series:
        """Calculate agreement between regime and indicator signals."""
        # Convert regime states to signals (simplified)
        if self.config.n_states == 3:
            # 0=Bear(-1), 1=Sideways(0), 2=Bull(+1)
            regime_signals = regime_states.map({0: -1, 1: 0, 2: 1})
        elif self.config.n_states == 4:
            # 0=Crisis(-1), 1=Bear(-1), 2=Sideways(0), 3=Bull(+1)
            regime_signals = regime_states.map({0: -1, 1: -1, 2: 0, 3: 1})
        else:
            # Default mapping
            regime_signals = regime_states - (self.config.n_states // 2)

        # Calculate agreement as correlation
        agreement = []
        window = 20  # Rolling window for agreement calculation

        for i in range(len(regime_signals)):
            if i < window:
                agreement.append(0.0)  # Not enough data
            else:
                regime_window = regime_signals.iloc[i - window : i]
                indicator_window = indicator_signals.iloc[i - window : i]

                # Calculate correlation
                try:
                    corr = regime_window.corr(indicator_window)
                    agreement.append(corr if not pd.isna(corr) else 0.0)
                except:
                    agreement.append(0.0)

        return pd.Series(agreement, index=regime_signals.index)

    def _calculate_regime_consensus_agreement(
        self, regime_states: pd.Series, consensus: pd.Series
    ) -> pd.Series:
        """Calculate agreement between regime and indicator consensus."""
        # Similar to single indicator but using consensus signal
        if self.config.n_states == 3:
            regime_signals = regime_states.map({0: -1, 1: 0, 2: 1})
        elif self.config.n_states == 4:
            regime_signals = regime_states.map({0: -1, 1: -1, 2: 0, 3: 1})
        else:
            regime_signals = regime_states - (self.config.n_states // 2)

        # Calculate rolling correlation
        agreement = []
        window = 20

        for i in range(len(regime_signals)):
            if i < window:
                agreement.append(0.0)
            else:
                regime_window = regime_signals.iloc[i - window : i]
                consensus_window = consensus.iloc[i - window : i]

                try:
                    corr = regime_window.corr(consensus_window)
                    agreement.append(corr if not pd.isna(corr) else 0.0)
                except:
                    agreement.append(0.0)

        return pd.Series(agreement, index=regime_signals.index)

    def _add_trading_signals(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """Add trading signals based on regime analysis using RegimeType enum."""
        # Simple position sizing based on regime and confidence
        position_signals = []

        for _, row in analysis.iterrows():
            regime_type = row["regime_type"]
            confidence = row["confidence"]

            # Use regime type values for consistent matching
            if regime_type == "crisis":
                signal = -0.5 * confidence  # Defensive position
            elif regime_type == "bearish":
                signal = -0.3 * confidence  # Short position
            elif regime_type == "sideways":
                signal = 0.1 * confidence  # Minimal position
            elif regime_type == "bullish":
                signal = 0.8 * confidence  # Long position
            elif regime_type == "mixed":
                signal = 0.0  # No clear signal for mixed regimes
            else:
                # Legacy support for old regime names
                if "Bear" in str(regime_type):
                    signal = -0.3 * confidence
                elif "Bull" in str(regime_type):
                    signal = 0.8 * confidence
                elif "Crisis" in str(regime_type):
                    signal = -0.5 * confidence
                elif "Sideways" in str(regime_type):
                    signal = 0.1 * confidence
                else:
                    signal = 0.0

            # Apply risk adjustment if enabled
            if self.config.risk_adjustment:
                expected_vol = row.get("expected_volatility", 0.015)
                risk_factor = min(
                    1.0, 0.015 / max(expected_vol, 0.005)
                )  # Scale by volatility
                signal *= risk_factor

            position_signals.append(signal)

        analysis["position_signal"] = position_signals
        analysis["signal_strength"] = analysis["position_signal"].abs()

        return analysis

    def plot(self, ax=None, **kwargs) -> plt.Figure:
        """
        Generate visualization for analysis results.

        Args:
            ax: Optional matplotlib axes to plot into for pipeline integration
            **kwargs: Additional plotting arguments

        Returns:
            matplotlib Figure with analysis visualizations
        """
        if self._last_analysis is None:
            if ax is not None:
                ax.text(
                    0.5,
                    0.5,
                    "No analysis results yet",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                return ax.figure
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(
                    0.5,
                    0.5,
                    "No analysis results yet",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                return fig

        # If ax is provided, create compact plot for pipeline integration
        if ax is not None:
            return self._plot_compact(ax, **kwargs)

        # Otherwise, create full standalone plot
        return self._plot_full(**kwargs)

    def _plot_compact(self, ax, **kwargs):
        """Create compact plot for pipeline integration."""
        analysis = self._last_analysis

        # Color map for regimes
        regime_colors = {
            "crisis": "red",
            "bearish": "orange",
            "sideways": "gray",
            "bullish": "green",
            "mixed": "purple",
            # Legacy support
            "Crisis": "red",
            "Bear": "orange",
            "Sideways": "gray",
            "Bull": "green",
            "Euphoric": "purple",
        }

        # Plot regime sequence as colored bars
        x_vals = range(len(analysis))
        for i, (idx, row) in enumerate(analysis.iterrows()):
            regime_type = row.get("regime_type", f"Regime_{row['predicted_state']}")
            color = regime_colors.get(regime_type, "blue")
            alpha = row["confidence"] * 0.7 + 0.3  # Scale alpha by confidence

            ax.bar(i, 1, color=color, alpha=alpha, width=1, edgecolor="none")

        ax.set_title("Analysis - Regime Sequence")
        ax.set_ylabel("Regime")
        ax.set_ylim(0, 1)

        # Add compact legend
        unique_regimes = analysis["regime_type"].unique()
        legend_elements = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=regime_colors.get(regime, "blue"),
                alpha=0.7,
                label=regime,
            )
            for regime in unique_regimes
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        return ax.figure

    def _plot_full(self, **kwargs):
        """Create full standalone plot with subplots."""
        analysis = self._last_analysis

        # Create subplots
        n_plots = 3
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))

        # Plot 1: Regime sequence with confidence
        ax1 = axes[0]

        # Color map for regimes
        regime_colors = {
            "crisis": "red",
            "bearish": "orange",
            "sideways": "gray",
            "bullish": "green",
            "mixed": "purple",
            # Legacy support
            "Crisis": "red",
            "Bear": "orange",
            "Sideways": "gray",
            "Bull": "green",
            "Euphoric": "purple",
        }

        # Plot regime states as colored background
        for i, (idx, row) in enumerate(analysis.iterrows()):
            regime_type = row.get("regime_type", f"Regime_{row['predicted_state']}")
            color = regime_colors.get(regime_type, "blue")
            alpha = row["confidence"] * 0.7 + 0.3  # Scale alpha by confidence

            ax1.bar(i, 1, color=color, alpha=alpha, width=1, edgecolor="none")

        ax1.set_title("Regime Sequence (colored by type, opacity by confidence)")
        ax1.set_ylabel("Regime")
        ax1.set_ylim(0, 1)

        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, label=regime)
            for regime, color in regime_colors.items()
            if regime in analysis["regime_type"].values
        ]
        ax1.legend(handles=legend_elements, loc="upper right")

        # Plot 2: Confidence and days in regime
        ax2 = axes[1]
        ax2_twin = ax2.twinx()

        x_vals = range(len(analysis))
        line1 = ax2.plot(
            x_vals, analysis["confidence"], "b-", linewidth=2, label="Confidence"
        )

        if "days_in_regime" in analysis.columns:
            line2 = ax2_twin.plot(
                x_vals,
                analysis["days_in_regime"],
                "r--",
                linewidth=2,
                label="Days in Regime",
            )
            ax2_twin.set_ylabel("Days in Regime", color="red")
            ax2_twin.tick_params(axis="y", labelcolor="red")

        ax2.set_title("Confidence and Regime Duration")
        ax2.set_ylabel("Confidence", color="blue")
        ax2.set_xlabel("Time")
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = (
            ax2_twin.get_legend_handles_labels()
            if "days_in_regime" in analysis.columns
            else ([], [])
        )
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # Plot 3: Trading signals (if available)
        ax3 = axes[2]
        if "position_signal" in analysis.columns:
            signals = analysis["position_signal"]

            # Color signals by regime type
            colors = [
                regime_colors.get(regime_type, "blue")
                for regime_type in analysis["regime_type"]
            ]

            bars = ax3.bar(
                x_vals,
                signals,
                color=colors,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax3.set_title("Trading Position Signals")
            ax3.set_ylabel("Position Size")
            ax3.set_xlabel("Time")
            ax3.grid(True, alpha=0.3)

            # Add horizontal lines for reference
            ax3.axhline(
                y=0.5, color="green", linestyle="--", alpha=0.5, label="Strong Bull"
            )
            ax3.axhline(
                y=-0.5, color="red", linestyle="--", alpha=0.5, label="Strong Bear"
            )
            ax3.legend()
        else:
            ax3.text(
                0.5,
                0.5,
                "No trading signals generated",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title("Trading Position Signals")

        plt.tight_layout()
        return fig

    def get_current_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime state."""
        if self._last_analysis is None or len(self._last_analysis) == 0:
            return {"status": "No analysis available"}

        current = self._last_analysis.iloc[-1]

        summary = {
            "current_state": int(current["predicted_state"]),
            "regime_name": current.get(
                "regime_name", f"State_{current['predicted_state']}"
            ),
            "regime_type": current.get("regime_type", "Unknown"),
            "confidence": float(current["confidence"]),
            "days_in_regime": int(current.get("days_in_regime", 0)),
        }

        # Add expected characteristics if available
        for key in ["expected_return", "expected_volatility", "expected_duration"]:
            if key in current:
                summary[key] = float(current[key])

        # Add trading signal if available
        if "position_signal" in current:
            summary["position_signal"] = float(current["position_signal"])
            summary["signal_strength"] = float(current.get("signal_strength", 0))

        return summary

    def get_comprehensive_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance analysis using RegimePerformanceAnalyzer.

        Returns:
            Dictionary with detailed performance metrics and analysis
        """
        if self._last_analysis is None:
            return {"status": "No analysis available"}

        try:
            # Get comprehensive performance analysis
            performance_metrics = self.performance_analyzer.analyze_regime_performance(
                analysis_results=self._last_analysis, raw_data=self._last_raw_data
            )

            return performance_metrics

        except Exception as e:
            return {"error": f"Performance analysis failed: {str(e)}"}

    def get_regime_transition_matrix(self) -> Optional[Dict[str, Any]]:
        """Get regime transition matrix and statistics."""
        if self._last_analysis is None:
            return None

        try:
            performance_metrics = self.get_comprehensive_performance_metrics()
            return performance_metrics.get("transition_analysis", {})
        except Exception:
            return None

    def get_regime_duration_statistics(self) -> Optional[Dict[str, Any]]:
        """Get regime duration statistics."""
        if self._last_analysis is None:
            return None

        try:
            performance_metrics = self.get_comprehensive_performance_metrics()
            return performance_metrics.get("duration_analysis", {})
        except Exception:
            return None

    def get_regime_profiles(self) -> Optional[Dict[int, Any]]:
        """
        Get regime profiles for visualization color consistency.

        Returns:
            Dictionary mapping state IDs to RegimeProfile objects, or None if not available
        """
        return self._regime_profiles if hasattr(self, '_regime_profiles') else None
