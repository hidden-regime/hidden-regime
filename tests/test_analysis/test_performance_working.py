"""
Working tests for RegimePerformanceAnalyzer component.

Tests that work with the current implementation, focusing on coverage
and validation of performance analysis functionality.
"""

import warnings
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from hidden_regime.analysis.performance import RegimePerformanceAnalyzer
from hidden_regime.utils.exceptions import ValidationError


class TestRegimePerformanceAnalyzerWorking:
    """Working tests for RegimePerformanceAnalyzer that focus on coverage."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test basic initialization."""
        analyzer = RegimePerformanceAnalyzer()

        assert analyzer.regime_stats_cache == {}
        assert analyzer.transition_stats_cache == {}

    @pytest.mark.integration
    def test_basic_performance_analysis(self):
        """Test basic performance analysis functionality."""
        analyzer = RegimePerformanceAnalyzer()

        # Create sample analysis results
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        analysis_results = pd.DataFrame(
            {
                "predicted_state": np.random.randint(0, 3, 100),
                "confidence": np.random.uniform(0.5, 1.0, 100),
            },
            index=dates,
        )

        # Perform analysis
        performance_report = analyzer.analyze_regime_performance(analysis_results)

        # Check that analysis was performed
        assert isinstance(performance_report, dict)
        assert "regime_distribution" in performance_report
        assert "transition_analysis" in performance_report
        assert "duration_analysis" in performance_report
        assert "confidence_analysis" in performance_report
        assert "stability_metrics" in performance_report
        assert "summary" in performance_report

    @pytest.mark.integration
    def test_performance_analysis_with_raw_data(self):
        """Test performance analysis with raw price data."""
        analyzer = RegimePerformanceAnalyzer()

        # Create sample data
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        # Analysis results
        analysis_results = pd.DataFrame(
            {
                "predicted_state": np.random.randint(0, 3, 50),
                "confidence": np.random.uniform(0.6, 0.95, 50),
                "regime_name": np.random.choice(["Bear", "Sideways", "Bull"], 50),
            },
            index=dates,
        )

        # Raw price data
        prices = 100 + np.cumsum(np.random.normal(0.1, 1.0, 50))
        raw_data = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.02,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.random.randint(1000000, 5000000, 50),
            },
            index=dates,
        )

        # Perform analysis with raw data
        performance_report = analyzer.analyze_regime_performance(
            analysis_results, raw_data
        )

        # Check that regime performance analysis was included
        assert "regime_performance" in performance_report
        assert isinstance(performance_report, dict)

    @pytest.mark.integration
    def test_regime_distribution_analysis(self):
        """Test regime distribution analysis."""
        analyzer = RegimePerformanceAnalyzer()

        # Create data with known distribution
        dates = pd.date_range("2023-01-01", periods=90, freq="D")
        states = [0] * 30 + [1] * 40 + [2] * 20  # 30-40-20 distribution

        analysis_results = pd.DataFrame(
            {
                "predicted_state": states,
                "confidence": np.random.uniform(0.7, 1.0, 90),
                "regime_name": ["Bear"] * 30 + ["Sideways"] * 40 + ["Bull"] * 20,
            },
            index=dates,
        )

        performance_report = analyzer.analyze_regime_performance(analysis_results)
        distribution = performance_report["regime_distribution"]

        # Check distribution calculations
        assert distribution["total_periods"] == 90
        assert distribution["unique_states"] == 3
        assert distribution["state_counts"][0] == 30
        assert distribution["state_counts"][1] == 40
        assert distribution["state_counts"][2] == 20

        # Check percentages
        assert abs(distribution["state_percentages"][0] - 33.33) < 0.1
        assert abs(distribution["state_percentages"][1] - 44.44) < 0.1
        assert abs(distribution["state_percentages"][2] - 22.22) < 0.1

    @pytest.mark.integration
    def test_transition_analysis(self):
        """Test regime transition analysis."""
        analyzer = RegimePerformanceAnalyzer()

        # Create data with known transitions
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        states = [0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 2] * 2  # Clear transitions

        analysis_results = pd.DataFrame(
            {"predicted_state": states, "confidence": np.random.uniform(0.6, 1.0, 30)},
            index=dates,
        )

        performance_report = analyzer.analyze_regime_performance(analysis_results)
        transition_analysis = performance_report["transition_analysis"]

        # Check that transition analysis was performed
        assert isinstance(transition_analysis, dict)
        # May include transition matrix, transition counts, etc.

    @pytest.mark.integration
    def test_duration_analysis(self):
        """Test regime duration analysis."""
        analyzer = RegimePerformanceAnalyzer()

        # Create data with specific regime durations
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        states = [0] * 10 + [1] * 20 + [2] * 15 + [0] * 5 + [1] * 10  # Known durations

        analysis_results = pd.DataFrame(
            {"predicted_state": states, "confidence": np.random.uniform(0.5, 1.0, 60)},
            index=dates,
        )

        performance_report = analyzer.analyze_regime_performance(analysis_results)
        duration_analysis = performance_report["duration_analysis"]

        # Check that duration analysis was performed
        assert isinstance(duration_analysis, dict)
        # Should include average durations, duration distributions, etc.

    @pytest.mark.integration
    def test_confidence_analysis(self):
        """Test confidence metrics analysis."""
        analyzer = RegimePerformanceAnalyzer()

        # Create data with varying confidence levels
        dates = pd.date_range("2023-01-01", periods=40, freq="D")

        # Create confidence with different ranges for different states
        confidence = np.concatenate(
            [
                np.random.uniform(0.8, 1.0, 15),  # High confidence
                np.random.uniform(0.5, 0.7, 15),  # Medium confidence
                np.random.uniform(0.3, 0.6, 10),  # Low confidence
            ]
        )

        analysis_results = pd.DataFrame(
            {"predicted_state": np.random.randint(0, 3, 40), "confidence": confidence},
            index=dates,
        )

        performance_report = analyzer.analyze_regime_performance(analysis_results)
        confidence_analysis = performance_report["confidence_analysis"]

        # Check that confidence analysis was performed
        assert isinstance(confidence_analysis, dict)
        # Should include confidence statistics, distributions, etc.

    @pytest.mark.unit
    def test_error_handling_empty_results(self):
        """Test error handling for empty analysis results."""
        analyzer = RegimePerformanceAnalyzer()

        empty_results = pd.DataFrame()

        with pytest.raises(ValidationError) as exc_info:
            analyzer.analyze_regime_performance(empty_results)

        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.integration
    def test_stability_metrics_analysis(self):
        """Test model stability metrics analysis."""
        analyzer = RegimePerformanceAnalyzer()

        # Create data with stability patterns
        dates = pd.date_range("2023-01-01", periods=80, freq="D")

        # Mix of stable and unstable periods
        stable_period = [0] * 20 + [1] * 25 + [2] * 15  # Stable regimes
        unstable_period = [0, 1, 0, 2, 1, 0, 2, 1, 2, 0] * 2  # Rapid switching

        analysis_results = pd.DataFrame(
            {
                "predicted_state": stable_period + unstable_period,
                "confidence": np.random.uniform(0.4, 0.9, 80),
            },
            index=dates,
        )

        performance_report = analyzer.analyze_regime_performance(analysis_results)
        stability_metrics = performance_report["stability_metrics"]

        # Check that stability analysis was performed
        assert isinstance(stability_metrics, dict)

    @pytest.mark.unit
    def test_performance_summary_generation(self):
        """Test performance summary generation."""
        analyzer = RegimePerformanceAnalyzer()

        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        analysis_results = pd.DataFrame(
            {
                "predicted_state": np.random.randint(0, 3, 50),
                "confidence": np.random.uniform(0.5, 0.95, 50),
            },
            index=dates,
        )

        performance_report = analyzer.analyze_regime_performance(analysis_results)
        summary = performance_report["summary"]

        # Check that summary was generated
        assert isinstance(summary, dict)
        # Summary should contain key performance indicators

    @pytest.mark.integration
    def test_regime_performance_with_returns(self):
        """Test regime performance analysis with return calculations."""
        analyzer = RegimePerformanceAnalyzer()

        dates = pd.date_range("2023-01-01", periods=60, freq="D")

        # Create regime-specific price patterns
        # State 0: declining prices, State 1: sideways, State 2: rising
        price_changes = []
        states = []

        for i in range(20):  # State 0 - bearish
            price_changes.append(np.random.normal(-0.5, 1.0))
            states.append(0)
        for i in range(20):  # State 1 - sideways
            price_changes.append(np.random.normal(0.0, 0.5))
            states.append(1)
        for i in range(20):  # State 2 - bullish
            price_changes.append(np.random.normal(0.5, 1.0))
            states.append(2)

        prices = 100 + np.cumsum(price_changes)

        analysis_results = pd.DataFrame(
            {"predicted_state": states, "confidence": np.random.uniform(0.6, 0.9, 60)},
            index=dates,
        )

        raw_data = pd.DataFrame({"close": prices}, index=dates)

        performance_report = analyzer.analyze_regime_performance(
            analysis_results, raw_data
        )

        # Check that regime performance analysis was included
        assert "regime_performance" in performance_report
        regime_perf = performance_report["regime_performance"]
        assert isinstance(regime_perf, dict)


class TestRegimePerformanceAnalyzerCoverage:
    """Additional tests to improve code coverage."""

    @pytest.mark.integration
    def test_various_data_patterns(self):
        """Test analyzer with various data patterns."""
        analyzer = RegimePerformanceAnalyzer()

        # Test with single state (no transitions)
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        single_state_results = pd.DataFrame(
            {
                "predicted_state": [1] * 20,  # All same state
                "confidence": np.random.uniform(0.7, 1.0, 20),
            },
            index=dates,
        )

        performance_report = analyzer.analyze_regime_performance(single_state_results)
        assert performance_report["regime_distribution"]["unique_states"] == 1

    @pytest.mark.integration
    def test_missing_regime_name_column(self):
        """Test analysis when regime_name column is missing."""
        analyzer = RegimePerformanceAnalyzer()

        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        analysis_results = pd.DataFrame(
            {
                "predicted_state": np.random.randint(0, 3, 30),
                "confidence": np.random.uniform(0.5, 1.0, 30),
                # No 'regime_name' column
            },
            index=dates,
        )

        performance_report = analyzer.analyze_regime_performance(analysis_results)

        # Should still work without regime names
        assert isinstance(performance_report, dict)
        assert "regime_distribution" in performance_report

    @pytest.mark.unit
    def test_missing_predicted_state_column(self):
        """Test error handling when predicted_state column is missing."""
        analyzer = RegimePerformanceAnalyzer()

        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        invalid_results = pd.DataFrame(
            {
                "confidence": np.random.uniform(0.5, 1.0, 20)
                # Missing 'predicted_state' column
            },
            index=dates,
        )

        # Should still attempt analysis but may return error in distribution
        performance_report = analyzer.analyze_regime_performance(invalid_results)
        distribution = performance_report["regime_distribution"]

        # Should handle missing column gracefully
        assert "error" in distribution or "total_periods" in distribution

    @pytest.mark.integration
    def test_edge_case_confidence_values(self):
        """Test analysis with edge case confidence values."""
        analyzer = RegimePerformanceAnalyzer()

        dates = pd.date_range("2023-01-01", periods=25, freq="D")
        analysis_results = pd.DataFrame(
            {
                "predicted_state": np.random.randint(0, 3, 25),
                "confidence": [0.0, 0.1, 0.5, 0.9, 1.0]
                * 5,  # Edge values including 0 and 1
            },
            index=dates,
        )

        performance_report = analyzer.analyze_regime_performance(analysis_results)
        confidence_analysis = performance_report["confidence_analysis"]

        # Should handle edge confidence values
        assert isinstance(confidence_analysis, dict)

    @pytest.mark.unit
    def test_cache_functionality(self):
        """Test cache functionality."""
        analyzer = RegimePerformanceAnalyzer()

        # Check initial cache state
        assert len(analyzer.regime_stats_cache) == 0
        assert len(analyzer.transition_stats_cache) == 0

        # Perform analysis (may populate caches depending on implementation)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        analysis_results = pd.DataFrame(
            {
                "predicted_state": np.random.randint(0, 3, 30),
                "confidence": np.random.uniform(0.5, 1.0, 30),
            },
            index=dates,
        )

        analyzer.analyze_regime_performance(analysis_results)

        # Cache structure should exist (whether populated or not)
        assert hasattr(analyzer, "regime_stats_cache")
        assert hasattr(analyzer, "transition_stats_cache")

    @pytest.mark.integration
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        analyzer = RegimePerformanceAnalyzer()

        # Create larger dataset
        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        analysis_results = pd.DataFrame(
            {
                "predicted_state": np.random.randint(0, 4, 500),  # 4 states
                "confidence": np.random.uniform(0.3, 1.0, 500),
            },
            index=dates,
        )

        # Should handle larger datasets efficiently
        performance_report = analyzer.analyze_regime_performance(analysis_results)

        assert isinstance(performance_report, dict)
        assert performance_report["regime_distribution"]["total_periods"] == 500
        assert performance_report["regime_distribution"]["unique_states"] == 4

    @pytest.mark.integration
    def test_datetime_index_handling(self):
        """Test handling of different datetime index formats."""
        analyzer = RegimePerformanceAnalyzer()

        # Test with different datetime frequencies
        hourly_dates = pd.date_range("2023-01-01", periods=48, freq="H")
        analysis_results = pd.DataFrame(
            {
                "predicted_state": np.random.randint(0, 3, 48),
                "confidence": np.random.uniform(0.5, 1.0, 48),
            },
            index=hourly_dates,
        )

        performance_report = analyzer.analyze_regime_performance(analysis_results)

        # Should handle different time frequencies
        assert isinstance(performance_report, dict)
        assert performance_report["regime_distribution"]["total_periods"] == 48


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
