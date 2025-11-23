"""
Indicator performance comparison framework for regime analysis.

Provides comprehensive comparison between HMM regime detection and technical indicators
including performance metrics, agreement analysis, and predictive accuracy evaluation.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..utils.exceptions import ValidationError


class IndicatorPerformanceComparator:
    """
    Framework for comparing HMM regime detection with technical indicators.

    Provides comprehensive analysis of agreement, predictive accuracy, and
    performance metrics between regime states and technical indicator signals.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize indicator performance comparator.

        Args:
            config: Optional configuration for analysis parameters
        """
        self.config = config or FinancialAnalysisConfig()
        self._cache = {}

        # Performance thresholds
        self.agreement_thresholds = {
            "excellent": 0.8,
            "good": 0.6,
            "moderate": 0.4,
            "poor": 0.2,
        }

        # Indicator calculation parameters
        self.indicator_params = {
            "rsi": {"period": 14, "overbought": 70, "oversold": 30},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger_bands": {"period": 20, "std_dev": 2},
            "moving_average": {"period": 20},
            "stochastic": {"k_period": 14, "d_period": 3},
            "williams_r": {"period": 14},
        }

    def compare_regime_vs_indicators(
        self,
        analysis_results: pd.DataFrame,
        raw_data: Optional[pd.DataFrame] = None,
        indicators: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison between regime detection and technical indicators.

        Args:
            analysis_results: Results from regime analysis with predicted states
            raw_data: Optional OHLCV data for indicator calculations
            indicators: List of indicator names to compare (default: all available)

        Returns:
            Dictionary with comprehensive comparison results
        """
        if analysis_results.empty:
            raise ValidationError("Analysis results cannot be empty")

        if "predicted_state" not in analysis_results.columns:
            raise ValidationError(
                "Analysis results must contain 'predicted_state' column"
            )

        # Use default indicators if none specified
        if indicators is None:
            indicators = ["rsi", "macd", "bollinger_bands", "moving_average"]

        comparison_results = {
            "summary": self._generate_comparison_summary(analysis_results, indicators),
            "indicator_analysis": {},
            "cross_indicator_analysis": {},
            "predictive_performance": {},
            "regime_indicator_matrix": {},
            "statistical_tests": {},
        }

        # Calculate indicators if raw data is available
        calculated_indicators = {}
        if raw_data is not None and "close" in raw_data.columns:
            for indicator in indicators:
                try:
                    indicator_data = self._calculate_indicator(indicator, raw_data)
                    if indicator_data is not None:
                        calculated_indicators[indicator] = indicator_data
                except Exception as e:
                    warnings.warn(f"Failed to calculate {indicator}: {e}")

        # Individual indicator analysis
        for indicator in indicators:
            if indicator in calculated_indicators:
                comparison_results["indicator_analysis"][indicator] = (
                    self._analyze_single_indicator(
                        analysis_results["predicted_state"],
                        calculated_indicators[indicator],
                        indicator,
                        analysis_results.get(
                            "confidence", pd.Series([1.0] * len(analysis_results))
                        ),
                    )
                )

        # Cross-indicator analysis
        if len(calculated_indicators) > 1:
            comparison_results["cross_indicator_analysis"] = (
                self._analyze_cross_indicators(
                    calculated_indicators, analysis_results["predicted_state"]
                )
            )

        # Predictive performance analysis
        if len(calculated_indicators) > 0:
            comparison_results["predictive_performance"] = (
                self._analyze_predictive_performance(
                    analysis_results, calculated_indicators, raw_data
                )
            )

        # Regime-indicator agreement matrix
        comparison_results["regime_indicator_matrix"] = (
            self._build_regime_indicator_matrix(analysis_results, calculated_indicators)
        )

        # Statistical significance tests
        comparison_results["statistical_tests"] = self._perform_statistical_tests(
            analysis_results, calculated_indicators
        )

        return comparison_results

    def _generate_comparison_summary(
        self, analysis_results: pd.DataFrame, indicators: List[str]
    ) -> Dict[str, Any]:
        """Generate high-level comparison summary."""
        regime_states = analysis_results["predicted_state"].unique()
        n_states = len(regime_states)

        return {
            "total_observations": len(analysis_results),
            "date_range": {
                "start": (
                    analysis_results.index[0]
                    if hasattr(analysis_results.index, "to_pydatetime")
                    else 0
                ),
                "end": (
                    analysis_results.index[-1]
                    if hasattr(analysis_results.index, "to_pydatetime")
                    else len(analysis_results) - 1
                ),
            },
            "regime_summary": {
                "n_states": n_states,
                "state_distribution": analysis_results["predicted_state"]
                .value_counts()
                .to_dict(),
                "avg_confidence": analysis_results.get(
                    "confidence", pd.Series([1.0] * len(analysis_results))
                ).mean(),
            },
            "indicators_requested": indicators,
            "analysis_timestamp": pd.Timestamp.now(),
        }

    def _calculate_indicator(
        self, indicator_name: str, raw_data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Calculate technical indicator from raw price data."""
        if "close" not in raw_data.columns:
            return None

        prices = raw_data["close"]
        params = self.indicator_params.get(indicator_name, {})

        try:
            if indicator_name == "rsi":
                return self._calculate_rsi_detailed(prices, params.get("period", 14))
            elif indicator_name == "macd":
                return self._calculate_macd_detailed(prices, params)
            elif indicator_name == "bollinger_bands":
                return self._calculate_bollinger_detailed(prices, params)
            elif indicator_name == "moving_average":
                return self._calculate_ma_detailed(prices, params.get("period", 20))
            elif indicator_name == "stochastic":
                return self._calculate_stochastic_detailed(raw_data, params)
            elif indicator_name == "williams_r":
                return self._calculate_williams_r_detailed(
                    raw_data, params.get("period", 14)
                )
            else:
                warnings.warn(f"Unknown indicator: {indicator_name}")
                return None
        except Exception as e:
            warnings.warn(f"Error calculating {indicator_name}: {e}")
            return None

    def _calculate_rsi_detailed(
        self, prices: pd.Series, period: int = 14
    ) -> pd.DataFrame:
        """Calculate detailed RSI with signals and zones."""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signals = np.where(
            rsi > 70, -1, np.where(rsi < 30, 1, 0)
        )  # -1=sell, 1=buy, 0=hold

        # Zone classification
        zones = np.where(
            rsi > 80,
            "extreme_overbought",
            np.where(
                rsi > 70,
                "overbought",
                np.where(
                    rsi > 50,
                    "bullish",
                    np.where(
                        rsi > 30,
                        "bearish",
                        np.where(rsi > 20, "oversold", "extreme_oversold"),
                    ),
                ),
            ),
        )

        return pd.DataFrame(
            {
                "value": rsi,
                "signal": signals,
                "zone": zones,
                "strength": np.abs(rsi - 50) / 50,  # Signal strength (0-1)
            },
            index=prices.index,
        )

    def _calculate_macd_detailed(
        self, prices: pd.Series, params: Dict[str, int]
    ) -> pd.DataFrame:
        """Calculate detailed MACD with signals and momentum."""
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal_period = params.get("signal", 9)

        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line

        # Generate signals
        signals = np.where(histogram > 0, 1, -1)  # Above/below signal line

        # Momentum classification
        momentum = np.where(
            histogram > histogram.shift(1),
            "increasing",
            np.where(histogram < histogram.shift(1), "decreasing", "stable"),
        )

        return pd.DataFrame(
            {
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
                "signal": signals,
                "momentum": momentum,
                "strength": np.abs(histogram) / np.abs(histogram).rolling(20).max(),
            },
            index=prices.index,
        )

    def _calculate_bollinger_detailed(
        self, prices: pd.Series, params: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate detailed Bollinger Bands with position and squeeze indicators."""
        period = int(params.get("period", 20))
        std_dev = params.get("std_dev", 2)

        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        # Position within bands
        position = (prices - lower_band) / (upper_band - lower_band)

        # Generate signals
        signals = np.where(position > 1, -1, np.where(position < 0, 1, 0))

        # Squeeze detection (narrow bands)
        band_width = (upper_band - lower_band) / sma
        squeeze = band_width < band_width.rolling(20).quantile(0.2)

        return pd.DataFrame(
            {
                "upper_band": upper_band,
                "middle_band": sma,
                "lower_band": lower_band,
                "position": position,
                "signal": signals,
                "squeeze": squeeze,
                "band_width": band_width,
            },
            index=prices.index,
        )

    def _calculate_ma_detailed(
        self, prices: pd.Series, period: int = 20
    ) -> pd.DataFrame:
        """Calculate detailed moving average with trend and strength."""
        ma = prices.rolling(window=period).mean()
        position = (prices - ma) / ma

        # Trend direction
        ma_slope = ma.diff() / ma
        trend = np.where(
            ma_slope > 0.001,
            "uptrend",
            np.where(ma_slope < -0.001, "downtrend", "sideways"),
        )

        # Generate signals
        signals = np.where(position > 0.02, 1, np.where(position < -0.02, -1, 0))

        return pd.DataFrame(
            {
                "ma": ma,
                "position": position,
                "signal": signals,
                "trend": trend,
                "slope": ma_slope,
                "strength": np.abs(position),
            },
            index=prices.index,
        )

    def _calculate_stochastic_detailed(
        self, raw_data: pd.DataFrame, params: Dict[str, int]
    ) -> pd.DataFrame:
        """Calculate detailed Stochastic oscillator."""
        k_period = params.get("k_period", 14)
        d_period = params.get("d_period", 3)

        if not all(col in raw_data.columns for col in ["high", "low", "close"]):
            return None

        high = raw_data["high"]
        low = raw_data["low"]
        close = raw_data["close"]

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()

        # Generate signals
        signals = np.where(k_percent > 80, -1, np.where(k_percent < 20, 1, 0))

        return pd.DataFrame(
            {
                "k_percent": k_percent,
                "d_percent": d_percent,
                "signal": signals,
                "momentum": np.where(k_percent > d_percent, "bullish", "bearish"),
            },
            index=close.index,
        )

    def _calculate_williams_r_detailed(
        self, raw_data: pd.DataFrame, period: int = 14
    ) -> pd.DataFrame:
        """Calculate detailed Williams %R."""
        if not all(col in raw_data.columns for col in ["high", "low", "close"]):
            return None

        high = raw_data["high"]
        low = raw_data["low"]
        close = raw_data["close"]

        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)

        # Generate signals
        signals = np.where(williams_r > -20, -1, np.where(williams_r < -80, 1, 0))

        return pd.DataFrame(
            {
                "williams_r": williams_r,
                "signal": signals,
                "zone": np.where(
                    williams_r > -20,
                    "overbought",
                    np.where(williams_r < -80, "oversold", "neutral"),
                ),
            },
            index=close.index,
        )

    def _analyze_single_indicator(
        self,
        regime_states: pd.Series,
        indicator_data: pd.DataFrame,
        indicator_name: str,
        confidence: pd.Series,
    ) -> Dict[str, Any]:
        """Analyze agreement between regime states and single indicator."""
        if "signal" not in indicator_data.columns:
            return {"error": f"No signal column in {indicator_name} data"}

        # Align data
        common_index = regime_states.index.intersection(indicator_data.index)
        regime_aligned = regime_states.loc[common_index]
        indicator_aligned = indicator_data.loc[common_index, "signal"]
        confidence_aligned = confidence.loc[common_index]

        if len(common_index) == 0:
            return {"error": "No overlapping data between regime and indicator"}

        # Convert regime states to signals (simplified mapping)
        n_states = len(regime_aligned.unique())
        if n_states == 3:
            regime_signals = regime_aligned.map(
                {0: -1, 1: 0, 2: 1}
            )  # Bear, Sideways, Bull
        elif n_states == 4:
            regime_signals = regime_aligned.map(
                {0: -1, 1: -1, 2: 0, 3: 1}
            )  # Crisis, Bear, Sideways, Bull
        else:
            # Default mapping
            regime_signals = regime_aligned - (n_states // 2)

        # Calculate agreement metrics
        agreement_analysis = {
            "correlation": self._calculate_correlation_metrics(
                regime_signals, indicator_aligned
            ),
            "classification_metrics": self._calculate_classification_metrics(
                regime_signals, indicator_aligned
            ),
            "confidence_weighted_agreement": self._calculate_confidence_weighted_agreement(
                regime_signals, indicator_aligned, confidence_aligned
            ),
            "temporal_analysis": self._analyze_temporal_agreement(
                regime_signals, indicator_aligned
            ),
            "signal_strength_analysis": self._analyze_signal_strength(
                regime_signals,
                indicator_data.get("strength", pd.Series([1.0] * len(indicator_data))),
            ),
        }

        return {
            "indicator_name": indicator_name,
            "data_points": len(common_index),
            "agreement_analysis": agreement_analysis,
            "performance_rating": self._rate_indicator_performance(agreement_analysis),
            "summary_statistics": self._generate_indicator_summary_stats(
                indicator_data
            ),
        }

    def _calculate_correlation_metrics(
        self, regime_signals: pd.Series, indicator_signals: pd.Series
    ) -> Dict[str, float]:
        """Calculate correlation-based agreement metrics."""
        try:
            pearson_corr = regime_signals.corr(indicator_signals)
            spearman_corr = regime_signals.corr(indicator_signals, method="spearman")

            # Rolling correlations
            rolling_corr = regime_signals.rolling(20).corr(indicator_signals)

            return {
                "pearson_correlation": (
                    float(pearson_corr) if not pd.isna(pearson_corr) else 0.0
                ),
                "spearman_correlation": (
                    float(spearman_corr) if not pd.isna(spearman_corr) else 0.0
                ),
                "rolling_correlation_mean": (
                    float(rolling_corr.mean()) if not rolling_corr.isna().all() else 0.0
                ),
                "rolling_correlation_std": (
                    float(rolling_corr.std()) if not rolling_corr.isna().all() else 0.0
                ),
                "correlation_stability": (
                    float(1.0 - rolling_corr.std())
                    if not rolling_corr.isna().all()
                    else 0.0
                ),
            }
        except Exception:
            return {
                "pearson_correlation": 0.0,
                "spearman_correlation": 0.0,
                "rolling_correlation_mean": 0.0,
                "rolling_correlation_std": 0.0,
                "correlation_stability": 0.0,
            }

    def _calculate_classification_metrics(
        self, regime_signals: pd.Series, indicator_signals: pd.Series
    ) -> Dict[str, float]:
        """Calculate classification-based agreement metrics."""
        try:
            # Convert to same scale for comparison
            regime_classes = np.sign(regime_signals)
            indicator_classes = np.sign(indicator_signals)

            accuracy = accuracy_score(regime_classes, indicator_classes)
            precision = precision_score(
                regime_classes, indicator_classes, average="weighted", zero_division=0
            )
            recall = recall_score(
                regime_classes, indicator_classes, average="weighted", zero_division=0
            )
            f1 = f1_score(
                regime_classes, indicator_classes, average="weighted", zero_division=0
            )

            # Direction agreement (ignoring neutral signals)
            non_neutral_mask = (regime_classes != 0) & (indicator_classes != 0)
            if non_neutral_mask.sum() > 0:
                direction_agreement = (
                    regime_classes[non_neutral_mask]
                    == indicator_classes[non_neutral_mask]
                ).mean()
            else:
                direction_agreement = 0.0

            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "direction_agreement": float(direction_agreement),
            }
        except Exception:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "direction_agreement": 0.0,
            }

    def _calculate_confidence_weighted_agreement(
        self,
        regime_signals: pd.Series,
        indicator_signals: pd.Series,
        confidence: pd.Series,
    ) -> Dict[str, float]:
        """Calculate agreement weighted by regime confidence."""
        try:
            # Weight agreement by confidence
            agreement_raw = (
                np.sign(regime_signals) == np.sign(indicator_signals)
            ).astype(float)

            # Weighted metrics
            weighted_agreement = (agreement_raw * confidence).sum() / confidence.sum()

            # High confidence agreement (confidence > 0.7)
            high_conf_mask = confidence > 0.7
            if high_conf_mask.sum() > 0:
                high_conf_agreement = agreement_raw[high_conf_mask].mean()
            else:
                high_conf_agreement = 0.0

            # Low confidence agreement (confidence < 0.5)
            low_conf_mask = confidence < 0.5
            if low_conf_mask.sum() > 0:
                low_conf_agreement = agreement_raw[low_conf_mask].mean()
            else:
                low_conf_agreement = 0.0

            return {
                "weighted_agreement": float(weighted_agreement),
                "high_confidence_agreement": float(high_conf_agreement),
                "low_confidence_agreement": float(low_conf_agreement),
                "confidence_correlation": (
                    float(confidence.corr(agreement_raw))
                    if not confidence.corr(agreement_raw)
                    != confidence.corr(agreement_raw)
                    else 0.0
                ),
            }
        except Exception:
            return {
                "weighted_agreement": 0.0,
                "high_confidence_agreement": 0.0,
                "low_confidence_agreement": 0.0,
                "confidence_correlation": 0.0,
            }

    def _analyze_temporal_agreement(
        self, regime_signals: pd.Series, indicator_signals: pd.Series
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in agreement."""
        try:
            agreement = (np.sign(regime_signals) == np.sign(indicator_signals)).astype(
                float
            )

            # Rolling agreement
            rolling_agreement = agreement.rolling(20).mean()

            # Trend in agreement
            if len(agreement) > 1:
                agreement_trend = np.polyfit(range(len(agreement)), agreement, 1)[0]
            else:
                agreement_trend = 0.0

            # Volatility of agreement
            agreement_volatility = rolling_agreement.std()

            return {
                "rolling_agreement_mean": (
                    float(rolling_agreement.mean())
                    if not rolling_agreement.isna().all()
                    else 0.0
                ),
                "rolling_agreement_std": (
                    float(rolling_agreement.std())
                    if not rolling_agreement.isna().all()
                    else 0.0
                ),
                "agreement_trend": float(agreement_trend),
                "agreement_volatility": (
                    float(agreement_volatility)
                    if not pd.isna(agreement_volatility)
                    else 0.0
                ),
                "periods_analyzed": len(agreement),
            }
        except Exception:
            return {
                "rolling_agreement_mean": 0.0,
                "rolling_agreement_std": 0.0,
                "agreement_trend": 0.0,
                "agreement_volatility": 0.0,
                "periods_analyzed": 0,
            }

    def _analyze_signal_strength(
        self, regime_signals: pd.Series, indicator_strength: pd.Series
    ) -> Dict[str, float]:
        """Analyze relationship between signal strength and regime agreement."""
        try:
            agreement = (np.sign(regime_signals) == np.sign(regime_signals)).astype(
                float
            )

            # Correlation between strength and agreement
            strength_agreement_corr = indicator_strength.corr(agreement)

            # Strong signal agreement (top quartile strength)
            strong_threshold = indicator_strength.quantile(0.75)
            strong_mask = indicator_strength > strong_threshold
            if strong_mask.sum() > 0:
                strong_signal_agreement = agreement[strong_mask].mean()
            else:
                strong_signal_agreement = 0.0

            return {
                "strength_agreement_correlation": (
                    float(strength_agreement_corr)
                    if not pd.isna(strength_agreement_corr)
                    else 0.0
                ),
                "strong_signal_agreement": float(strong_signal_agreement),
                "avg_signal_strength": float(indicator_strength.mean()),
                "signal_strength_stability": (
                    float(1.0 - indicator_strength.std())
                    if not pd.isna(indicator_strength.std())
                    else 0.0
                ),
            }
        except Exception:
            return {
                "strength_agreement_correlation": 0.0,
                "strong_signal_agreement": 0.0,
                "avg_signal_strength": 0.0,
                "signal_strength_stability": 0.0,
            }

    def _rate_indicator_performance(
        self, agreement_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rate indicator performance based on agreement analysis."""
        try:
            # Extract key metrics
            correlation = agreement_analysis["correlation"]["pearson_correlation"]
            accuracy = agreement_analysis["classification_metrics"]["accuracy"]
            weighted_agreement = agreement_analysis["confidence_weighted_agreement"][
                "weighted_agreement"
            ]

            # Composite score
            composite_score = (
                abs(correlation) * 0.4 + accuracy * 0.3 + weighted_agreement * 0.3
            )

            # Rating categories
            if composite_score >= self.agreement_thresholds["excellent"]:
                rating = "excellent"
            elif composite_score >= self.agreement_thresholds["good"]:
                rating = "good"
            elif composite_score >= self.agreement_thresholds["moderate"]:
                rating = "moderate"
            else:
                rating = "poor"

            return {
                "composite_score": float(composite_score),
                "rating": rating,
                "correlation_component": float(abs(correlation)),
                "accuracy_component": float(accuracy),
                "weighted_agreement_component": float(weighted_agreement),
            }
        except Exception:
            return {
                "composite_score": 0.0,
                "rating": "poor",
                "correlation_component": 0.0,
                "accuracy_component": 0.0,
                "weighted_agreement_component": 0.0,
            }

    def _generate_indicator_summary_stats(
        self, indicator_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate summary statistics for indicator data."""
        try:
            stats = {}

            if "value" in indicator_data.columns:
                values = indicator_data["value"].dropna()
                stats["value_stats"] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median": float(values.median()),
                }

            if "signal" in indicator_data.columns:
                signals = indicator_data["signal"].dropna()
                signal_counts = signals.value_counts()
                stats["signal_distribution"] = {
                    "buy_signals": int(signal_counts.get(1, 0)),
                    "sell_signals": int(signal_counts.get(-1, 0)),
                    "hold_signals": int(signal_counts.get(0, 0)),
                    "signal_frequency": float((signals != 0).mean()),
                }

            return stats
        except Exception:
            return {}

    def _analyze_cross_indicators(
        self, indicators: Dict[str, pd.DataFrame], regime_states: pd.Series
    ) -> Dict[str, Any]:
        """Analyze relationships between multiple indicators."""
        if len(indicators) < 2:
            return {}

        try:
            # Extract signals from all indicators
            signal_data = {}
            for name, data in indicators.items():
                if "signal" in data.columns:
                    signal_data[name] = data["signal"]

            if len(signal_data) < 2:
                return {}

            # Create signal correlation matrix
            signal_df = pd.DataFrame(signal_data)
            correlation_matrix = signal_df.corr()

            # Consensus analysis
            consensus_signal = signal_df.mean(axis=1)

            # Individual vs consensus agreement
            individual_vs_consensus = {}
            for name in signal_data.keys():
                individual_vs_consensus[name] = signal_data[name].corr(consensus_signal)

            return {
                "signal_correlation_matrix": correlation_matrix.to_dict(),
                "consensus_analysis": {
                    "consensus_regime_correlation": regime_states.corr(
                        consensus_signal
                    ),
                    "individual_vs_consensus": individual_vs_consensus,
                    "consensus_volatility": float(consensus_signal.std()),
                },
                "indicator_diversity": float(
                    correlation_matrix.values[
                        np.triu_indices_from(correlation_matrix.values, k=1)
                    ].std()
                ),
            }
        except Exception:
            return {}

    def _analyze_predictive_performance(
        self,
        analysis_results: pd.DataFrame,
        indicators: Dict[str, pd.DataFrame],
        raw_data: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Analyze predictive performance of regimes vs indicators."""
        if raw_data is None or "close" not in raw_data.columns:
            return {"error": "No price data available for predictive analysis"}

        try:
            # Calculate forward returns
            returns = raw_data["close"].pct_change()
            forward_returns = {}

            for horizon in [1, 3, 5, 10]:  # Days ahead
                forward_returns[f"{horizon}d"] = returns.shift(-horizon)

            predictive_results = {}

            # Analyze regime predictive power
            predictive_results["regime_predictive_power"] = (
                self._analyze_regime_predictive_power(
                    analysis_results["predicted_state"], forward_returns
                )
            )

            # Analyze indicator predictive power
            predictive_results["indicator_predictive_power"] = {}
            for name, data in indicators.items():
                if "signal" in data.columns:
                    predictive_results["indicator_predictive_power"][name] = (
                        self._analyze_indicator_predictive_power(
                            data["signal"], forward_returns, name
                        )
                    )

            # Compare predictive performance
            predictive_results["comparative_analysis"] = (
                self._compare_predictive_performance(
                    analysis_results["predicted_state"], indicators, forward_returns
                )
            )

            return predictive_results
        except Exception as e:
            return {"error": f"Predictive analysis failed: {str(e)}"}

    def _analyze_regime_predictive_power(
        self, regime_states: pd.Series, forward_returns: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """Analyze predictive power of regime states."""
        results = {}

        for horizon, returns in forward_returns.items():
            if returns.dropna().empty:
                continue

            try:
                # Align data
                common_index = regime_states.index.intersection(returns.index)
                regime_aligned = regime_states.loc[common_index]
                returns_aligned = returns.loc[common_index]

                # Remove NaN values
                valid_mask = ~returns_aligned.isna()
                regime_aligned = regime_aligned[valid_mask]
                returns_aligned = returns_aligned[valid_mask]

                if len(regime_aligned) == 0:
                    continue

                # Calculate returns by regime
                regime_returns = {}
                for state in regime_aligned.unique():
                    state_returns = returns_aligned[regime_aligned == state]
                    if len(state_returns) > 0:
                        regime_returns[f"regime_{state}"] = {
                            "mean_return": float(state_returns.mean()),
                            "std_return": float(state_returns.std()),
                            "sharpe_ratio": (
                                float(state_returns.mean() / state_returns.std())
                                if state_returns.std() > 0
                                else 0.0
                            ),
                            "positive_return_rate": float((state_returns > 0).mean()),
                            "observations": len(state_returns),
                        }

                # Statistical tests
                if len(regime_aligned.unique()) > 1:
                    groups = [
                        returns_aligned[regime_aligned == state]
                        for state in regime_aligned.unique()
                    ]
                    groups = [
                        group.dropna() for group in groups if len(group.dropna()) > 0
                    ]

                    if len(groups) > 1:
                        try:
                            f_stat, p_value = stats.f_oneway(*groups)
                            results[horizon] = {
                                "regime_returns": regime_returns,
                                "anova_f_stat": float(f_stat),
                                "anova_p_value": float(p_value),
                                "significant_difference": p_value < 0.05,
                            }
                        except:
                            results[horizon] = {"regime_returns": regime_returns}
                    else:
                        results[horizon] = {"regime_returns": regime_returns}
                else:
                    results[horizon] = {"regime_returns": regime_returns}
            except Exception:
                continue

        return results

    def _analyze_indicator_predictive_power(
        self,
        indicator_signals: pd.Series,
        forward_returns: Dict[str, pd.Series],
        indicator_name: str,
    ) -> Dict[str, Any]:
        """Analyze predictive power of indicator signals."""
        results = {}

        for horizon, returns in forward_returns.items():
            if returns.dropna().empty:
                continue

            try:
                # Align data
                common_index = indicator_signals.index.intersection(returns.index)
                signals_aligned = indicator_signals.loc[common_index]
                returns_aligned = returns.loc[common_index]

                # Remove NaN values
                valid_mask = ~returns_aligned.isna()
                signals_aligned = signals_aligned[valid_mask]
                returns_aligned = returns_aligned[valid_mask]

                if len(signals_aligned) == 0:
                    continue

                # Calculate returns by signal
                signal_returns = {}
                for signal in signals_aligned.unique():
                    signal_name = (
                        "buy" if signal == 1 else "sell" if signal == -1 else "hold"
                    )
                    signal_data = returns_aligned[signals_aligned == signal]
                    if len(signal_data) > 0:
                        signal_returns[signal_name] = {
                            "mean_return": float(signal_data.mean()),
                            "std_return": float(signal_data.std()),
                            "sharpe_ratio": (
                                float(signal_data.mean() / signal_data.std())
                                if signal_data.std() > 0
                                else 0.0
                            ),
                            "positive_return_rate": float((signal_data > 0).mean()),
                            "observations": len(signal_data),
                        }

                # Direction accuracy
                direction_accuracy = 0.0
                if len(signals_aligned) > 0:
                    correct_directions = (
                        ((signals_aligned == 1) & (returns_aligned > 0))
                        | ((signals_aligned == -1) & (returns_aligned < 0))
                    ).sum()
                    total_directional_signals = (signals_aligned != 0).sum()
                    if total_directional_signals > 0:
                        direction_accuracy = float(
                            correct_directions / total_directional_signals
                        )

                results[horizon] = {
                    "signal_returns": signal_returns,
                    "direction_accuracy": direction_accuracy,
                    "signal_correlation": (
                        float(signals_aligned.corr(returns_aligned))
                        if len(signals_aligned) > 1
                        else 0.0
                    ),
                }
            except Exception:
                continue

        return results

    def _compare_predictive_performance(
        self,
        regime_states: pd.Series,
        indicators: Dict[str, pd.DataFrame],
        forward_returns: Dict[str, pd.Series],
    ) -> Dict[str, Any]:
        """Compare predictive performance between regimes and indicators."""
        try:
            comparison_results = {}

            for horizon, returns in forward_returns.items():
                if returns.dropna().empty:
                    continue

                horizon_results = {}

                # Get regime correlation
                common_index = regime_states.index.intersection(returns.index)
                regime_aligned = regime_states.loc[common_index]
                returns_aligned = returns.loc[common_index]

                valid_mask = ~returns_aligned.isna()
                regime_aligned = regime_aligned[valid_mask]
                returns_aligned = returns_aligned[valid_mask]

                if len(regime_aligned) > 1:
                    regime_correlation = abs(regime_aligned.corr(returns_aligned))
                    horizon_results["regime_correlation"] = (
                        float(regime_correlation)
                        if not pd.isna(regime_correlation)
                        else 0.0
                    )

                # Get indicator correlations
                indicator_correlations = {}
                for name, data in indicators.items():
                    if "signal" in data.columns:
                        signal_aligned = data["signal"].loc[common_index][valid_mask]
                        if len(signal_aligned) > 1:
                            correlation = abs(signal_aligned.corr(returns_aligned))
                            indicator_correlations[name] = (
                                float(correlation) if not pd.isna(correlation) else 0.0
                            )

                horizon_results["indicator_correlations"] = indicator_correlations

                # Best performer
                all_correlations = {
                    "regime": horizon_results.get("regime_correlation", 0.0)
                }
                all_correlations.update(indicator_correlations)

                if all_correlations:
                    best_performer = max(
                        all_correlations.keys(), key=lambda k: all_correlations[k]
                    )
                    horizon_results["best_performer"] = {
                        "name": best_performer,
                        "correlation": all_correlations[best_performer],
                    }

                comparison_results[horizon] = horizon_results

            return comparison_results
        except Exception:
            return {}

    def _build_regime_indicator_matrix(
        self, analysis_results: pd.DataFrame, indicators: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Build agreement matrix between regimes and indicators."""
        if not indicators:
            return {}

        try:
            regime_states = analysis_results["predicted_state"]
            matrix_data = {}

            for indicator_name, indicator_data in indicators.items():
                if "signal" in indicator_data.columns:
                    # Align data
                    common_index = regime_states.index.intersection(
                        indicator_data.index
                    )
                    regime_aligned = regime_states.loc[common_index]
                    signal_aligned = indicator_data.loc[common_index, "signal"]

                    # Create contingency table
                    contingency = pd.crosstab(
                        regime_aligned, signal_aligned, normalize="index"
                    )
                    matrix_data[indicator_name] = contingency.to_dict()

            return {
                "contingency_matrices": matrix_data,
                "summary": {
                    "total_indicators": len(indicators),
                    "indicators_with_signals": len(matrix_data),
                    "regime_states": list(regime_states.unique()),
                },
            }
        except Exception:
            return {}

    def _perform_statistical_tests(
        self, analysis_results: pd.DataFrame, indicators: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        if not indicators:
            return {}

        try:
            test_results = {}
            regime_states = analysis_results["predicted_state"]

            for indicator_name, indicator_data in indicators.items():
                if "signal" in indicator_data.columns:
                    # Align data
                    common_index = regime_states.index.intersection(
                        indicator_data.index
                    )
                    regime_aligned = regime_states.loc[common_index]
                    signal_aligned = indicator_data.loc[common_index, "signal"]

                    if len(common_index) > 10:  # Minimum sample size
                        try:
                            # Chi-square test for independence
                            contingency = pd.crosstab(regime_aligned, signal_aligned)
                            chi2, p_value, dof, expected = stats.chi2_contingency(
                                contingency
                            )

                            test_results[indicator_name] = {
                                "chi2_statistic": float(chi2),
                                "chi2_p_value": float(p_value),
                                "degrees_of_freedom": int(dof),
                                "significant_association": p_value < 0.05,
                                "sample_size": len(common_index),
                            }
                        except Exception:
                            test_results[indicator_name] = {
                                "error": "Statistical test failed",
                                "sample_size": len(common_index),
                            }

            return test_results
        except Exception:
            return {}

    def plot_comparison_results(
        self, comparison_results: Dict[str, Any], **kwargs
    ) -> plt.Figure:
        """Generate comprehensive visualization of comparison results."""
        if not comparison_results or "indicator_analysis" not in comparison_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No comparison results available",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return fig

        indicator_analysis = comparison_results["indicator_analysis"]
        n_indicators = len(indicator_analysis)

        if n_indicators == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No indicators analyzed",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return fig

        # Create subplots
        fig = plt.figure(figsize=(16, 12))

        # Plot 1: Performance ratings
        ax1 = plt.subplot(3, 2, 1)
        indicators = list(indicator_analysis.keys())
        scores = [
            indicator_analysis[ind]["agreement_analysis"]["correlation"][
                "pearson_correlation"
            ]
            for ind in indicators
        ]

        bars = ax1.bar(indicators, scores, alpha=0.7)
        ax1.set_title("Indicator-Regime Correlation")
        ax1.set_ylabel("Pearson Correlation")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # Color bars by performance
        for bar, score in zip(bars, scores):
            if abs(score) >= 0.6:
                bar.set_color("green")
            elif abs(score) >= 0.4:
                bar.set_color("orange")
            else:
                bar.set_color("red")

        # Plot 2: Accuracy comparison
        ax2 = plt.subplot(3, 2, 2)
        accuracies = [
            indicator_analysis[ind]["agreement_analysis"]["classification_metrics"][
                "accuracy"
            ]
            for ind in indicators
        ]

        ax2.bar(indicators, accuracies, alpha=0.7, color="skyblue")
        ax2.set_title("Classification Accuracy")
        ax2.set_ylabel("Accuracy")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Correlation matrix (if cross-indicator analysis available)
        ax3 = plt.subplot(3, 2, 3)
        if (
            "cross_indicator_analysis" in comparison_results
            and "signal_correlation_matrix"
            in comparison_results["cross_indicator_analysis"]
        ):
            corr_matrix = pd.DataFrame(
                comparison_results["cross_indicator_analysis"][
                    "signal_correlation_matrix"
                ]
            )
            im = ax3.imshow(corr_matrix.values, cmap="RdBu", vmin=-1, vmax=1)
            ax3.set_xticks(range(len(corr_matrix.columns)))
            ax3.set_yticks(range(len(corr_matrix.index)))
            ax3.set_xticklabels(corr_matrix.columns, rotation=45)
            ax3.set_yticklabels(corr_matrix.index)
            ax3.set_title("Indicator Signal Correlations")
            plt.colorbar(im, ax=ax3)
        else:
            ax3.text(
                0.5,
                0.5,
                "No cross-indicator data",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title("Indicator Signal Correlations")

        # Plot 4: Performance ratings distribution
        ax4 = plt.subplot(3, 2, 4)
        ratings = [
            indicator_analysis[ind]["performance_rating"]["rating"]
            for ind in indicators
        ]
        rating_counts = pd.Series(ratings).value_counts()

        ax4.pie(
            rating_counts.values,
            labels=rating_counts.index,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax4.set_title("Performance Rating Distribution")

        # Plot 5: Composite scores
        ax5 = plt.subplot(3, 2, 5)
        composite_scores = [
            indicator_analysis[ind]["performance_rating"]["composite_score"]
            for ind in indicators
        ]

        ax5.bar(indicators, composite_scores, alpha=0.7, color="lightgreen")
        ax5.set_title("Composite Performance Scores")
        ax5.set_ylabel("Composite Score")
        ax5.tick_params(axis="x", rotation=45)
        ax5.grid(True, alpha=0.3)

        # Plot 6: Data summary
        ax6 = plt.subplot(3, 2, 6)
        data_points = [indicator_analysis[ind]["data_points"] for ind in indicators]

        ax6.bar(indicators, data_points, alpha=0.7, color="lightcoral")
        ax6.set_title("Data Points per Indicator")
        ax6.set_ylabel("Number of Observations")
        ax6.tick_params(axis="x", rotation=45)
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
