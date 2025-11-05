"""
Tests for MCP tool error handling and edge cases.

Tests validation logic, error messages, and graceful failure modes.
"""

import pytest
from fastmcp.exceptions import ToolError

from hidden_regime_mcp.tools import (
    validate_ticker,
    validate_n_states,
    validate_date,
    validate_date_range,
    handle_pipeline_error,
    detect_regime,
    get_regime_statistics,
    get_transition_probabilities,
)


class TestValidationFunctions:
    """Test input validation functions."""

    @pytest.mark.unit
    def test_validate_ticker_valid(self):
        """Test valid ticker symbols pass validation."""
        validate_ticker("SPY")
        validate_ticker("AAPL")
        validate_ticker("NVDA")
        # Should not raise

    @pytest.mark.unit
    def test_validate_ticker_empty(self):
        """Test empty ticker raises error."""
        with pytest.raises(ToolError, match="Ticker symbol is required"):
            validate_ticker("")

    @pytest.mark.unit
    def test_validate_ticker_none(self):
        """Test None ticker raises error."""
        with pytest.raises(ToolError, match="Ticker symbol is required"):
            validate_ticker(None)

    @pytest.mark.unit
    def test_validate_ticker_invalid_characters(self):
        """Test ticker with invalid characters raises error."""
        with pytest.raises(ToolError, match="Must be alphanumeric"):
            validate_ticker("SPY!")
        with pytest.raises(ToolError, match="Must be alphanumeric"):
            validate_ticker("AAPL@")

    @pytest.mark.unit
    def test_validate_ticker_too_long(self):
        """Test ticker exceeding max length raises error."""
        with pytest.raises(ToolError, match="Max 10 characters"):
            validate_ticker("TOOLONGTICKER")

    @pytest.mark.unit
    def test_validate_n_states_valid(self):
        """Test valid n_states pass validation."""
        validate_n_states(2)
        validate_n_states(3)
        validate_n_states(5)
        # Should not raise

    @pytest.mark.unit
    def test_validate_n_states_too_low(self):
        """Test n_states below minimum raises error."""
        with pytest.raises(ToolError, match="must be between 2 and 5"):
            validate_n_states(1)

    @pytest.mark.unit
    def test_validate_n_states_too_high(self):
        """Test n_states above maximum raises error."""
        with pytest.raises(ToolError, match="must be between 2 and 5"):
            validate_n_states(6)

    @pytest.mark.unit
    def test_validate_date_valid(self):
        """Test valid date formats pass validation."""
        validate_date("2024-01-01", "test_date")
        validate_date("2023-12-31", "test_date")
        validate_date(None, "test_date")  # None is allowed
        # Should not raise

    @pytest.mark.unit
    def test_validate_date_invalid_format(self):
        """Test invalid date format raises error."""
        with pytest.raises(ToolError, match="must be in YYYY-MM-DD format"):
            validate_date("01-01-2024", "test_date")
        with pytest.raises(ToolError, match="must be in YYYY-MM-DD format"):
            validate_date("2024/01/01", "test_date")

    @pytest.mark.unit
    def test_validate_date_future(self):
        """Test future date raises error."""
        with pytest.raises(ToolError, match="cannot be in the future"):
            validate_date("2030-01-01", "test_date")

    @pytest.mark.unit
    def test_validate_date_range_valid(self):
        """Test valid date ranges pass validation."""
        validate_date_range("2024-01-01", "2024-12-31")
        validate_date_range("2024-01-01", "2024-01-01")  # Same date is ok
        validate_date_range(None, "2024-12-31")  # None is allowed
        validate_date_range("2024-01-01", None)  # None is allowed
        # Should not raise

    @pytest.mark.unit
    def test_validate_date_range_invalid(self):
        """Test invalid date range raises error."""
        with pytest.raises(ToolError, match="must be before or equal to"):
            validate_date_range("2024-12-31", "2024-01-01")


class TestErrorHandling:
    """Test standardized error handling."""

    @pytest.mark.unit
    def test_handle_pipeline_error_no_data(self):
        """Test 'No data' error produces helpful message."""
        with pytest.raises(ToolError, match="Unable to load data"):
            handle_pipeline_error("INVALID", Exception("No data found"), "Test operation")

    @pytest.mark.unit
    def test_handle_pipeline_error_could_not_load(self):
        """Test 'Could not load' error produces helpful message."""
        with pytest.raises(ToolError, match="Unable to load data"):
            handle_pipeline_error("INVALID", Exception("Could not load ticker"), "Test operation")

    @pytest.mark.unit
    def test_handle_pipeline_error_insufficient_data(self):
        """Test insufficient data error produces helpful message."""
        with pytest.raises(ToolError, match="Insufficient data"):
            handle_pipeline_error("SPY", Exception("insufficient observations"), "Test operation")

    @pytest.mark.unit
    def test_handle_pipeline_error_generic(self):
        """Test generic error produces operation-specific message."""
        with pytest.raises(ToolError, match="Test operation failed for SPY: some error"):
            handle_pipeline_error("SPY", Exception("some error"), "Test operation")


@pytest.mark.network
class TestDetectRegimeEdgeCases:
    """Test detect_regime edge cases requiring network access."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_detect_regime_invalid_ticker(self):
        """Test detect_regime with invalid ticker raises error."""
        with pytest.raises(ToolError, match="Unable to load data"):
            await detect_regime(ticker="INVALIDTICKER12345", n_states=3)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_detect_regime_future_date(self):
        """Test detect_regime with future date raises error."""
        with pytest.raises(ToolError, match="cannot be in the future"):
            await detect_regime(ticker="SPY", end_date="2030-01-01")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_detect_regime_invalid_date_range(self):
        """Test detect_regime with start > end raises error."""
        with pytest.raises(ToolError, match="must be before or equal to"):
            await detect_regime(ticker="SPY", start_date="2024-12-31", end_date="2024-01-01")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_detect_regime_invalid_n_states(self):
        """Test detect_regime with invalid n_states raises error."""
        with pytest.raises(ToolError, match="must be between 2 and 5"):
            await detect_regime(ticker="SPY", n_states=10)


@pytest.mark.network
class TestGetRegimeStatisticsEdgeCases:
    """Test get_regime_statistics edge cases requiring network access."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_get_regime_statistics_invalid_ticker(self):
        """Test get_regime_statistics with invalid ticker raises error."""
        with pytest.raises(ToolError, match="Unable to load data"):
            await get_regime_statistics(ticker="INVALIDTICKER12345", n_states=3)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_get_regime_statistics_future_date(self):
        """Test get_regime_statistics with future date raises error."""
        with pytest.raises(ToolError, match="cannot be in the future"):
            await get_regime_statistics(ticker="SPY", end_date="2030-01-01")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_get_regime_statistics_invalid_date_range(self):
        """Test get_regime_statistics with start > end raises error."""
        with pytest.raises(ToolError, match="must be before or equal to"):
            await get_regime_statistics(
                ticker="SPY", start_date="2024-12-31", end_date="2024-01-01"
            )


@pytest.mark.network
class TestGetTransitionProbabilitiesEdgeCases:
    """Test get_transition_probabilities edge cases requiring network access."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_get_transition_probabilities_invalid_ticker(self):
        """Test get_transition_probabilities with invalid ticker raises error."""
        with pytest.raises(ToolError, match="Unable to load data"):
            await get_transition_probabilities(ticker="INVALIDTICKER12345", n_states=3)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_get_transition_probabilities_invalid_n_states(self):
        """Test get_transition_probabilities with invalid n_states raises error."""
        with pytest.raises(ToolError, match="must be between 2 and 5"):
            await get_transition_probabilities(ticker="SPY", n_states=10)
