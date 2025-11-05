"""
Unit tests for MCP tools with mocked pipelines.

Tests tool functionality without network dependencies by mocking
the create_financial_pipeline function and pipeline outputs.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastmcp.exceptions import ToolError

from hidden_regime_mcp.tools import (
    detect_regime,
    get_regime_statistics,
    get_transition_probabilities,
    calculate_price_performance,
    determine_regime_status,
    analyze_regime_stability,
    generate_regime_interpretation,
)


class TestPricePerformanceCalculation:
    """Test price performance calculation utility."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range(end="2024-12-31", periods=365, freq="D")
        # Simulate price movement: start at 100, random walk
        prices = 100 * np.exp(np.cumsum(np.random.randn(365) * 0.01))
        return pd.DataFrame({"close": prices}, index=dates)

    @pytest.mark.unit
    def test_calculate_price_performance_basic(self, sample_price_data):
        """Test basic price performance calculation."""
        result = calculate_price_performance(sample_price_data)

        assert "ytd_return" in result
        assert "30d_return" in result
        assert "7d_return" in result
        assert "current_price" in result
        assert "price_trend" in result
        assert isinstance(result["current_price"], float)

    @pytest.mark.unit
    def test_calculate_price_performance_empty_dataframe(self):
        """Test error handling for empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            calculate_price_performance(empty_df)

    @pytest.mark.unit
    def test_calculate_price_performance_price_trend_up(self):
        """Test price trend detection for upward movement."""
        dates = pd.date_range(end="2024-12-31", periods=30, freq="D")
        # Steady upward trend
        prices = np.linspace(100, 110, 30)
        data = pd.DataFrame({"close": prices}, index=dates)

        result = calculate_price_performance(data)
        assert result["price_trend"] == "up"

    @pytest.mark.unit
    def test_calculate_price_performance_price_trend_down(self):
        """Test price trend detection for downward movement."""
        dates = pd.date_range(end="2024-12-31", periods=30, freq="D")
        # Steady downward trend
        prices = np.linspace(100, 90, 30)
        data = pd.DataFrame({"close": prices}, index=dates)

        result = calculate_price_performance(data)
        assert result["price_trend"] == "down"


class TestRegimeStatusDetermination:
    """Test regime status determination utility."""

    @pytest.mark.unit
    def test_regime_status_early(self):
        """Test early regime status (< 25% complete)."""
        status = determine_regime_status(days_in_regime=5, expected_duration=30)
        assert status == "early"

    @pytest.mark.unit
    def test_regime_status_mid(self):
        """Test mid regime status (25-60% complete)."""
        status = determine_regime_status(days_in_regime=15, expected_duration=30)
        assert status == "mid"

    @pytest.mark.unit
    def test_regime_status_mature(self):
        """Test mature regime status (60-100% complete)."""
        status = determine_regime_status(days_in_regime=25, expected_duration=30)
        assert status == "mature"

    @pytest.mark.unit
    def test_regime_status_overdue(self):
        """Test overdue regime status (> 100% complete)."""
        status = determine_regime_status(days_in_regime=35, expected_duration=30)
        assert status == "overdue"

    @pytest.mark.unit
    def test_regime_status_unknown_invalid_duration(self):
        """Test unknown status for invalid expected duration."""
        status = determine_regime_status(days_in_regime=10, expected_duration=0)
        assert status == "unknown"


class TestRegimeStabilityAnalysis:
    """Test regime stability analysis utility."""

    @pytest.fixture
    def sample_stable_regime_data(self):
        """Create sample data for stable regime."""
        dates = pd.date_range(end="2024-12-31", periods=60, freq="D")
        return pd.DataFrame(
            {
                "regime_name": ["Bull"] * 60,
                "regime_episode": [1] * 60,
                "confidence": np.random.uniform(0.7, 0.9, 60),
            },
            index=dates,
        )

    @pytest.fixture
    def sample_volatile_regime_data(self):
        """Create sample data for volatile regime transitions."""
        dates = pd.date_range(end="2024-12-31", periods=60, freq="D")
        # Very frequent regime changes (more than 3 unique regimes in last 30 days)
        regimes = ["Bull", "Sideways", "Bear", "Bull", "Sideways", "Bear"] * 10
        episodes = list(range(1, 61))
        return pd.DataFrame(
            {"regime_name": regimes, "regime_episode": episodes, "confidence": 0.8},
            index=dates,
        )

    @pytest.mark.unit
    def test_analyze_regime_stability_stable(self, sample_stable_regime_data):
        """Test stability analysis for stable regime."""
        result = analyze_regime_stability(sample_stable_regime_data)

        assert result["regime_stability"] == "stable"
        assert result["recent_transitions"] == 0
        assert result["previous_regime"] == "none"

    @pytest.mark.unit
    def test_analyze_regime_stability_moderate(self, sample_volatile_regime_data):
        """Test stability analysis for moderate regime transitions."""
        result = analyze_regime_stability(sample_volatile_regime_data)

        # With 3 unique regimes, regime_changes = 2, which is "moderate"
        assert result["regime_stability"] == "moderate"
        assert result["recent_transitions"] == 2

    @pytest.mark.unit
    def test_analyze_regime_stability_empty_dataframe(self):
        """Test error handling for empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            analyze_regime_stability(empty_df)


class TestRegimeInterpretation:
    """Test regime interpretation generation."""

    @pytest.mark.unit
    def test_generate_regime_interpretation_basic(self):
        """Test basic interpretation generation."""
        price_perf = {
            "ytd_return": 0.15,
            "ytd_return_pct": "+15.0%",
            "30d_return": 0.03,
            "30d_return_pct": "+3.0%",
        }

        result = generate_regime_interpretation(
            regime_name="bull", mean_return=0.002, volatility=0.015, price_perf=price_perf
        )

        assert "interpretation" in result
        assert "explanation" in result
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0

    @pytest.mark.unit
    def test_generate_regime_interpretation_high_volatility(self):
        """Test interpretation for high volatility regime."""
        price_perf = {
            "ytd_return": 0.10,
            "ytd_return_pct": "+10.0%",
            "30d_return": 0.02,
            "30d_return_pct": "+2.0%",
        }

        result = generate_regime_interpretation(
            regime_name="bull", mean_return=0.003, volatility=0.06, price_perf=price_perf
        )

        assert "high volatility" in result["interpretation"].lower()

    @pytest.mark.unit
    def test_generate_regime_interpretation_bearish_contradicts_ytd(self):
        """Test interpretation when bearish regime contradicts YTD performance."""
        price_perf = {
            "ytd_return": 0.15,
            "ytd_return_pct": "+15.0%",
            "30d_return": -0.05,
            "30d_return_pct": "-5.0%",
        }

        result = generate_regime_interpretation(
            regime_name="bear", mean_return=-0.002, volatility=0.025, price_perf=price_perf
        )

        # Should mention the contradiction
        assert "up" in result["explanation"].lower() or "ytd" in result["explanation"].lower()


class TestDetectRegimeUnit:
    """Unit tests for detect_regime with mocked pipeline."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline with all required components."""
        pipeline = MagicMock()

        # Mock data_output (price data)
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
        pipeline.data_output = pd.DataFrame({"close": prices}, index=dates)

        # Mock analysis_output (regime analysis)
        regimes = ["Bull"] * 200 + ["Sideways"] * 100 + ["Bear"] * (len(dates) - 300)
        episodes = [1] * 200 + [2] * 100 + [3] * (len(dates) - 300)
        pipeline.analysis_output = pd.DataFrame(
            {
                "regime_name": regimes,
                "confidence": np.random.uniform(0.7, 0.9, len(dates)),
                "expected_return": np.where(
                    np.array(regimes) == "Bull", 0.002, np.where(np.array(regimes) == "Bear", -0.001, 0.0)
                ),
                "expected_volatility": 0.015,
                "days_in_regime": range(1, len(dates) + 1),
                "expected_duration": 25.0,
                "win_rate": 0.65,
                "regime_episode": episodes,
                "predicted_state": np.where(
                    np.array(regimes) == "Bull", 2, np.where(np.array(regimes) == "Bear", 0, 1)
                ),
            },
            index=dates,
        )

        pipeline.update.return_value = "Success"
        return pipeline

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_detect_regime_success(self, mock_pipeline):
        """Test detect_regime with successful pipeline execution."""
        with patch("hidden_regime_mcp.tools.create_financial_pipeline", return_value=mock_pipeline):
            result = await detect_regime(ticker="SPY", n_states=3)

            assert result["ticker"] == "SPY"
            assert result["current_regime"] in ["Bull", "Bear", "Sideways"]
            assert 0.0 <= result["confidence"] <= 1.0
            assert isinstance(result["mean_return"], float)
            assert isinstance(result["volatility"], float)
            assert "days_in_regime" in result
            assert "expected_duration" in result
            assert "price_performance" in result
            assert "interpretation" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_detect_regime_with_dates(self, mock_pipeline):
        """Test detect_regime with custom date range."""
        with patch("hidden_regime_mcp.tools.create_financial_pipeline", return_value=mock_pipeline):
            result = await detect_regime(
                ticker="SPY", n_states=3, start_date="2024-01-01", end_date="2024-06-30"
            )

            assert result["ticker"] == "SPY"
            assert result["n_states"] == 3
            assert "2024" in result["analysis_period"]["start"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_detect_regime_empty_analysis_output(self):
        """Test detect_regime with empty analysis output."""
        empty_pipeline = MagicMock()
        empty_pipeline.analysis_output = pd.DataFrame()
        empty_pipeline.update.return_value = "Success"

        with patch("hidden_regime_mcp.tools.create_financial_pipeline", return_value=empty_pipeline):
            with pytest.raises(ToolError):  # Will raise error about empty data
                await detect_regime(ticker="SPY", n_states=3)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_detect_regime_pipeline_error(self):
        """Test detect_regime with pipeline error."""
        with patch(
            "hidden_regime_mcp.tools.create_financial_pipeline",
            side_effect=Exception("No data found"),
        ):
            with pytest.raises(ToolError):  # Will raise error about data loading
                await detect_regime(ticker="INVALID", n_states=3)


class TestGetRegimeStatisticsUnit:
    """Unit tests for get_regime_statistics with mocked pipeline."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline with regime statistics."""
        pipeline = MagicMock()

        # Mock data_output
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        pipeline.data_output = pd.DataFrame({"close": np.random.randn(len(dates)) + 100}, index=dates)

        # Mock analysis_output with multiple regimes
        n_days = len(dates)
        n_bull = n_days // 3
        n_sideways = n_days // 3
        n_bear = n_days - n_bull - n_sideways

        regimes = ["Bull"] * n_bull + ["Sideways"] * n_sideways + ["Bear"] * n_bear
        pipeline.analysis_output = pd.DataFrame(
            {
                "regime_name": regimes,
                "expected_return": np.where(
                    np.array(regimes) == "Bull",
                    0.003,
                    np.where(np.array(regimes) == "Bear", -0.002, 0.0),
                ),
                "expected_volatility": np.where(
                    np.array(regimes) == "Bull",
                    0.015,
                    np.where(np.array(regimes) == "Bear", 0.025, 0.010),
                ),
                "expected_duration": 25.0,
                "win_rate": np.where(
                    np.array(regimes) == "Bull", 0.70, np.where(np.array(regimes) == "Bear", 0.35, 0.50)
                ),
            },
            index=dates,
        )

        pipeline.update.return_value = "Success"
        return pipeline

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_regime_statistics_success(self, mock_pipeline):
        """Test get_regime_statistics with successful execution."""
        with patch("hidden_regime_mcp.tools.create_financial_pipeline", return_value=mock_pipeline):
            result = await get_regime_statistics(ticker="SPY", n_states=3)

            assert result["ticker"] == "SPY"
            assert result["n_states"] == 3
            assert "regimes" in result
            assert len(result["regimes"]) == 3

            # Check regime structure
            for regime_name, stats in result["regimes"].items():
                assert "mean_return" in stats
                assert "volatility" in stats
                assert "duration_days" in stats
                assert "win_rate" in stats
                assert "observations" in stats
                assert 0.0 <= stats["win_rate"] <= 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_regime_statistics_regime_characteristics(self, mock_pipeline):
        """Test that regime statistics differentiate regimes."""
        with patch("hidden_regime_mcp.tools.create_financial_pipeline", return_value=mock_pipeline):
            result = await get_regime_statistics(ticker="SPY", n_states=3)

            regimes = result["regimes"]
            # Bull should have higher mean return than Bear
            if "Bull" in regimes and "Bear" in regimes:
                assert regimes["Bull"]["mean_return"] > regimes["Bear"]["mean_return"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_regime_statistics_empty_output(self):
        """Test get_regime_statistics with empty output."""
        empty_pipeline = MagicMock()
        empty_pipeline.analysis_output = pd.DataFrame()
        empty_pipeline.update.return_value = "Success"

        with patch("hidden_regime_mcp.tools.create_financial_pipeline", return_value=empty_pipeline):
            with pytest.raises(ToolError):  # Will raise error about empty data
                await get_regime_statistics(ticker="SPY", n_states=3)


class TestGetTransitionProbabilitiesUnit:
    """Unit tests for get_transition_probabilities with mocked pipeline."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline with transition matrix."""
        pipeline = MagicMock()

        # Mock data_output
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        pipeline.data_output = pd.DataFrame({"close": np.random.randn(len(dates)) + 100}, index=dates)

        # Mock model with transition matrix
        pipeline.model = MagicMock()
        pipeline.model.transition_matrix_ = np.array(
            [[0.85, 0.10, 0.05], [0.10, 0.75, 0.15], [0.20, 0.20, 0.60]]
        )

        # Mock analysis_output
        states = np.random.choice([0, 1, 2], size=len(dates), p=[0.4, 0.35, 0.25])
        regimes = np.where(states == 0, "Bear", np.where(states == 1, "Sideways", "Bull"))

        pipeline.analysis_output = pd.DataFrame(
            {"regime_name": regimes, "predicted_state": states}, index=dates
        )

        pipeline.update.return_value = "Success"
        return pipeline

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transition_probabilities_success(self, mock_pipeline):
        """Test get_transition_probabilities with successful execution."""
        with patch("hidden_regime_mcp.tools.create_financial_pipeline", return_value=mock_pipeline):
            result = await get_transition_probabilities(ticker="SPY", n_states=3)

            assert result["ticker"] == "SPY"
            assert result["n_states"] == 3
            assert "transition_matrix" in result
            assert "expected_durations" in result
            assert "steady_state" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transition_probabilities_matrix_structure(self, mock_pipeline):
        """Test transition matrix structure."""
        with patch("hidden_regime_mcp.tools.create_financial_pipeline", return_value=mock_pipeline):
            result = await get_transition_probabilities(ticker="SPY", n_states=3)

            matrix = result["transition_matrix"]
            assert len(matrix) == 3

            # Check each row sums to approximately 1.0
            for from_regime, transitions in matrix.items():
                total_prob = sum(transitions.values())
                assert 0.95 <= total_prob <= 1.05, f"Row {from_regime} sums to {total_prob}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transition_probabilities_durations(self, mock_pipeline):
        """Test expected duration calculations."""
        with patch("hidden_regime_mcp.tools.create_financial_pipeline", return_value=mock_pipeline):
            result = await get_transition_probabilities(ticker="SPY", n_states=3)

            durations = result["expected_durations"]
            assert len(durations) == 3

            # All durations should be positive
            for regime, duration in durations.items():
                assert duration > 0 or duration == float("inf")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transition_probabilities_steady_state(self, mock_pipeline):
        """Test steady state probabilities."""
        with patch("hidden_regime_mcp.tools.create_financial_pipeline", return_value=mock_pipeline):
            result = await get_transition_probabilities(ticker="SPY", n_states=3)

            steady_state = result["steady_state"]
            assert len(steady_state) == 3

            # Probabilities should sum to approximately 1.0
            total_prob = sum(steady_state.values())
            assert 0.95 <= total_prob <= 1.05

            # All probabilities between 0 and 1
            for regime, prob in steady_state.items():
                assert 0.0 <= prob <= 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transition_probabilities_empty_output(self):
        """Test get_transition_probabilities with empty output."""
        empty_pipeline = MagicMock()
        empty_pipeline.analysis_output = pd.DataFrame()
        empty_pipeline.update.return_value = "Success"

        with patch("hidden_regime_mcp.tools.create_financial_pipeline", return_value=empty_pipeline):
            with pytest.raises(ToolError):  # Will raise error about empty data
                await get_transition_probabilities(ticker="SPY", n_states=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
