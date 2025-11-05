"""
Tests for Hidden Regime MCP tools.

Tests the core MCP tools: detect_regime, get_regime_statistics, and
get_transition_probabilities.
"""

import pytest
from fastmcp.exceptions import ToolError

from hidden_regime_mcp.tools import (
    detect_regime,
    get_regime_statistics,
    get_transition_probabilities,
    validate_ticker,
    validate_n_states,
    validate_date,
)


class TestValidation:
    """Test validation functions."""

    @pytest.mark.unit
    def test_validate_ticker_valid(self):
        """Test valid ticker symbols."""
        validate_ticker("SPY")
        validate_ticker("AAPL")
        validate_ticker("BRK.B")  # With dot
        validate_ticker("BRK-B")  # With dash

    @pytest.mark.unit
    def test_validate_ticker_invalid(self):
        """Test invalid ticker symbols."""
        with pytest.raises(ToolError, match="Ticker symbol is required"):
            validate_ticker("")

        with pytest.raises(ToolError, match="Invalid ticker symbol"):
            validate_ticker("SP Y")  # Space

        with pytest.raises(ToolError, match="Invalid ticker symbol"):
            validate_ticker("SPY@")  # Special char

        with pytest.raises(ToolError, match="Ticker symbol too long"):
            validate_ticker("VERYLONGTICKER")

    @pytest.mark.unit
    def test_validate_n_states_valid(self):
        """Test valid n_states values."""
        validate_n_states(2)
        validate_n_states(3)
        validate_n_states(4)
        validate_n_states(5)

    @pytest.mark.unit
    def test_validate_n_states_invalid(self):
        """Test invalid n_states values."""
        with pytest.raises(ToolError, match="n_states must be between 2 and 5"):
            validate_n_states(1)

        with pytest.raises(ToolError, match="n_states must be between 2 and 5"):
            validate_n_states(6)

    @pytest.mark.unit
    def test_validate_date_valid(self):
        """Test valid date formats."""
        validate_date("2024-01-01", "start_date")
        validate_date("2024-12-31", "end_date")
        validate_date(None, "date")  # None is valid

    @pytest.mark.unit
    def test_validate_date_invalid(self):
        """Test invalid date formats."""
        with pytest.raises(ToolError, match="must be in YYYY-MM-DD format"):
            validate_date("01/01/2024", "start_date")

        with pytest.raises(ToolError, match="must be in YYYY-MM-DD format"):
            validate_date("2024-1-1", "start_date")

        with pytest.raises(ToolError, match="must be in YYYY-MM-DD format"):
            validate_date("not-a-date", "start_date")


@pytest.mark.asyncio
@pytest.mark.network
class TestDetectRegime:
    """Test detect_regime tool."""

    @pytest.mark.e2e
    async def test_detect_regime_basic(self):
        """Test basic regime detection."""
        result = await detect_regime(ticker="SPY", n_states=3)

        assert result["ticker"] == "SPY"
        assert result["current_regime"] in ["bull", "bear", "sideways"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["mean_return"], float)
        assert isinstance(result["volatility"], float)
        assert result["n_states"] == 3
        assert "analysis_period" in result

    @pytest.mark.e2e
    async def test_detect_regime_with_dates(self):
        """Test regime detection with custom date range."""
        result = await detect_regime(
            ticker="SPY",
            n_states=3,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert result["ticker"] == "SPY"
        assert result["n_states"] == 3
        assert "2024" in result["analysis_period"]["start"]
        assert "2024" in result["analysis_period"]["end"]

    @pytest.mark.e2e
    async def test_detect_regime_invalid_ticker(self):
        """Test regime detection with invalid ticker."""
        with pytest.raises(ToolError):
            await detect_regime(ticker="INVALIDTICKER123456789", n_states=3)

    @pytest.mark.e2e
    async def test_detect_regime_invalid_n_states(self):
        """Test regime detection with invalid n_states."""
        with pytest.raises(ToolError, match="n_states must be between 2 and 5"):
            await detect_regime(ticker="SPY", n_states=10)


@pytest.mark.asyncio
@pytest.mark.network
class TestGetRegimeStatistics:
    """Test get_regime_statistics tool."""

    @pytest.mark.e2e
    async def test_get_regime_statistics_basic(self):
        """Test basic regime statistics."""
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

    @pytest.mark.e2e
    async def test_get_regime_statistics_with_dates(self):
        """Test regime statistics with custom date range."""
        result = await get_regime_statistics(
            ticker="SPY",
            n_states=3,
            start_date="2024-01-01",
            end_date="2024-06-30",
        )

        assert result["ticker"] == "SPY"
        assert "2024" in result["analysis_period"]["start"]
        assert result["analysis_period"]["total_days"] > 0


@pytest.mark.asyncio
@pytest.mark.network
class TestGetTransitionProbabilities:
    """Test get_transition_probabilities tool."""

    @pytest.mark.e2e
    async def test_get_transition_probabilities_basic(self):
        """Test basic transition probabilities."""
        result = await get_transition_probabilities(ticker="SPY", n_states=3)

        assert result["ticker"] == "SPY"
        assert result["n_states"] == 3
        assert "transition_matrix" in result
        assert "expected_durations" in result
        assert "steady_state" in result

        # Check transition matrix structure
        matrix = result["transition_matrix"]
        assert len(matrix) == 3

        for from_regime, transitions in matrix.items():
            # Each row should have 3 transitions
            assert len(transitions) == 3

            # Probabilities should sum to ~1.0
            total_prob = sum(transitions.values())
            assert 0.95 <= total_prob <= 1.05

    @pytest.mark.e2e
    async def test_transition_probabilities_expected_durations(self):
        """Test expected duration calculations."""
        result = await get_transition_probabilities(ticker="SPY", n_states=3)

        durations = result["expected_durations"]
        assert len(durations) == 3

        for regime, duration in durations.items():
            assert duration > 0  # Should be positive
            assert duration < 1000  # Reasonable upper bound

    @pytest.mark.e2e
    async def test_transition_probabilities_steady_state(self):
        """Test steady state probabilities."""
        result = await get_transition_probabilities(ticker="SPY", n_states=3)

        steady_state = result["steady_state"]
        assert len(steady_state) == 3

        # Probabilities should sum to ~1.0
        total_prob = sum(steady_state.values())
        assert 0.95 <= total_prob <= 1.05

        # Each probability should be between 0 and 1
        for regime, prob in steady_state.items():
            assert 0.0 <= prob <= 1.0
