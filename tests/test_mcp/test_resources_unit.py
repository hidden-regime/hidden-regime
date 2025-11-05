"""
Unit tests for MCP resources with mocked tools.

Tests resource functionality without network dependencies by mocking
the underlying detect_regime and get_transition_probabilities tools.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock
from fastmcp.exceptions import ToolError

from hidden_regime_mcp.resources import (
    get_current_regime_resource,
    get_transitions_resource,
)


class TestCurrentRegimeResource:
    """Unit tests for get_current_regime_resource with mocked tools."""

    @pytest.fixture
    def mock_detect_regime_response(self):
        """Create mock response for detect_regime."""
        return {
            "ticker": "SPY",
            "current_regime": "bull",
            "confidence": 0.85,
            "mean_return": 0.002,
            "volatility": 0.015,
            "last_updated": "2024-12-31",
            "n_states": 3,
            "analysis_period": {"start": "2024-01-01", "end": "2024-12-31"},
            "days_in_regime": 25,
            "expected_duration": 30.5,
            "percent_complete": 82.0,
            "regime_status": "mature",
            "days_until_expected_transition": 5.5,
            "price_performance": {
                "ytd_return": 0.187,
                "ytd_return_pct": "+18.7%",
                "30d_return": 0.034,
                "30d_return_pct": "+3.4%",
                "7d_return": 0.012,
                "7d_return_pct": "+1.2%",
                "current_price": 450.25,
                "price_trend": "up",
            },
            "current_price": 450.25,
            "price_trend": "up",
            "regime_stability": "stable",
            "recent_transitions": 1,
            "previous_regime": "sideways",
            "last_transition_date": "2024-12-06",
            "interpretation": "Low volatility phase with strong positive returns",
            "explanation": "The bull regime is characterized by low volatility...",
        }

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_current_regime_resource_success(self, mock_detect_regime_response):
        """Test current regime resource with successful tool call."""
        with patch(
            "hidden_regime_mcp.resources.detect_regime",
            new_callable=AsyncMock,
            return_value=mock_detect_regime_response,
        ):
            result = await get_current_regime_resource("SPY")

            # Should return valid JSON
            data = json.loads(result)

            assert data["ticker"] == "SPY"
            assert data["current_regime"] == "bull"
            assert data["confidence"] == 0.85
            assert data["mean_return"] == 0.002
            assert data["volatility"] == 0.015
            assert "last_updated" in data

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_current_regime_resource_lowercase_ticker(self, mock_detect_regime_response):
        """Test resource converts lowercase ticker to uppercase."""
        with patch(
            "hidden_regime_mcp.resources.detect_regime",
            new_callable=AsyncMock,
            return_value=mock_detect_regime_response,
        ) as mock_detect:
            result = await get_current_regime_resource("spy")

            # Should call detect_regime with uppercase
            mock_detect.assert_called_once_with(ticker="SPY", n_states=3)

            data = json.loads(result)
            assert data["ticker"] == "SPY"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_current_regime_resource_includes_all_fields(self, mock_detect_regime_response):
        """Test resource includes all expected fields."""
        with patch(
            "hidden_regime_mcp.resources.detect_regime",
            new_callable=AsyncMock,
            return_value=mock_detect_regime_response,
        ):
            result = await get_current_regime_resource("SPY")
            data = json.loads(result)

            # Check basic fields
            assert "ticker" in data
            assert "current_regime" in data
            assert "confidence" in data
            assert "mean_return" in data
            assert "volatility" in data

            # Check temporal context
            assert "days_in_regime" in data
            assert "expected_duration" in data
            assert "percent_complete" in data

            # Check price context
            assert "price_performance" in data
            assert "current_price" in data

            # Check stability metrics
            assert "regime_stability" in data
            assert "recent_transitions" in data

            # Check interpretation
            assert "interpretation" in data
            assert "explanation" in data

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_current_regime_resource_tool_error(self):
        """Test resource handles ToolError gracefully."""
        with patch(
            "hidden_regime_mcp.resources.detect_regime",
            new_callable=AsyncMock,
            side_effect=ToolError("Unable to load data for INVALID"),
        ):
            result = await get_current_regime_resource("INVALID")

            # Should return error JSON, not raise exception
            data = json.loads(result)

            assert "error" in data
            assert "ticker" in data
            assert data["ticker"] == "INVALID"
            assert "Unable to load data" in data["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_current_regime_resource_unexpected_error(self):
        """Test resource handles unexpected errors gracefully."""
        with patch(
            "hidden_regime_mcp.resources.detect_regime",
            new_callable=AsyncMock,
            side_effect=Exception("Unexpected failure"),
        ):
            result = await get_current_regime_resource("SPY")

            # Should return error JSON
            data = json.loads(result)

            assert "error" in data
            assert "Unexpected error" in data["error"]
            assert data["ticker"] == "SPY"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_current_regime_resource_valid_json(self, mock_detect_regime_response):
        """Test that resource returns valid, parseable JSON."""
        with patch(
            "hidden_regime_mcp.resources.detect_regime",
            new_callable=AsyncMock,
            return_value=mock_detect_regime_response,
        ):
            result = await get_current_regime_resource("SPY")

            # Should not raise json.JSONDecodeError
            data = json.loads(result)
            assert isinstance(data, dict)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_current_regime_resource_confidence_range(self, mock_detect_regime_response):
        """Test that confidence is within valid range."""
        with patch(
            "hidden_regime_mcp.resources.detect_regime",
            new_callable=AsyncMock,
            return_value=mock_detect_regime_response,
        ):
            result = await get_current_regime_resource("SPY")
            data = json.loads(result)

            assert 0.0 <= data["confidence"] <= 1.0


class TestTransitionsResource:
    """Unit tests for get_transitions_resource with mocked tools."""

    @pytest.fixture
    def mock_transition_response(self):
        """Create mock response for get_transition_probabilities."""
        return {
            "ticker": "SPY",
            "transition_matrix": {
                "bull": {"bull": 0.85, "bear": 0.05, "sideways": 0.10},
                "bear": {"bull": 0.10, "bear": 0.75, "sideways": 0.15},
                "sideways": {"bull": 0.25, "bear": 0.15, "sideways": 0.60},
            },
            "expected_durations": {"bull": 20.0, "bear": 10.0, "sideways": 6.67},
            "steady_state": {"bull": 0.45, "bear": 0.25, "sideways": 0.30},
            "n_states": 3,
        }

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transitions_resource_success(self, mock_transition_response):
        """Test transitions resource with successful tool call."""
        with patch(
            "hidden_regime_mcp.resources.get_transition_probabilities",
            new_callable=AsyncMock,
            return_value=mock_transition_response,
        ):
            result = await get_transitions_resource("SPY")

            # Should return valid JSON
            data = json.loads(result)

            assert data["ticker"] == "SPY"
            assert "transition_matrix" in data
            assert "expected_durations" in data
            assert "steady_state" in data
            assert data["n_states"] == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transitions_resource_lowercase_ticker(self, mock_transition_response):
        """Test resource converts lowercase ticker to uppercase."""
        with patch(
            "hidden_regime_mcp.resources.get_transition_probabilities",
            new_callable=AsyncMock,
            return_value=mock_transition_response,
        ) as mock_trans:
            result = await get_transitions_resource("spy")

            # Should call get_transition_probabilities with uppercase
            mock_trans.assert_called_once_with(ticker="SPY", n_states=3)

            data = json.loads(result)
            assert data["ticker"] == "SPY"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transitions_resource_matrix_structure(self, mock_transition_response):
        """Test transition matrix structure in response."""
        with patch(
            "hidden_regime_mcp.resources.get_transition_probabilities",
            new_callable=AsyncMock,
            return_value=mock_transition_response,
        ):
            result = await get_transitions_resource("SPY")
            data = json.loads(result)

            matrix = data["transition_matrix"]

            # Check matrix is square
            assert len(matrix) == 3

            # Check each row has correct transitions
            for from_regime, to_probs in matrix.items():
                assert len(to_probs) == 3

                # Probabilities should sum to approximately 1.0
                total_prob = sum(to_probs.values())
                assert 0.95 <= total_prob <= 1.05

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transitions_resource_durations(self, mock_transition_response):
        """Test expected durations in response."""
        with patch(
            "hidden_regime_mcp.resources.get_transition_probabilities",
            new_callable=AsyncMock,
            return_value=mock_transition_response,
        ):
            result = await get_transitions_resource("SPY")
            data = json.loads(result)

            durations = data["expected_durations"]
            assert len(durations) == 3

            # All durations should be positive
            for regime, duration in durations.items():
                assert duration > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transitions_resource_steady_state(self, mock_transition_response):
        """Test steady state probabilities in response."""
        with patch(
            "hidden_regime_mcp.resources.get_transition_probabilities",
            new_callable=AsyncMock,
            return_value=mock_transition_response,
        ):
            result = await get_transitions_resource("SPY")
            data = json.loads(result)

            steady_state = data["steady_state"]
            assert len(steady_state) == 3

            # Probabilities should sum to approximately 1.0
            total_prob = sum(steady_state.values())
            assert 0.95 <= total_prob <= 1.05

            # All probabilities between 0 and 1
            for regime, prob in steady_state.items():
                assert 0.0 <= prob <= 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transitions_resource_tool_error(self):
        """Test resource handles ToolError gracefully."""
        with patch(
            "hidden_regime_mcp.resources.get_transition_probabilities",
            new_callable=AsyncMock,
            side_effect=ToolError("Unable to load data for INVALID"),
        ):
            result = await get_transitions_resource("INVALID")

            # Should return error JSON, not raise exception
            data = json.loads(result)

            assert "error" in data
            assert "ticker" in data
            assert data["ticker"] == "INVALID"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transitions_resource_unexpected_error(self):
        """Test resource handles unexpected errors gracefully."""
        with patch(
            "hidden_regime_mcp.resources.get_transition_probabilities",
            new_callable=AsyncMock,
            side_effect=Exception("Unexpected failure"),
        ):
            result = await get_transitions_resource("SPY")

            # Should return error JSON
            data = json.loads(result)

            assert "error" in data
            assert "Unexpected error" in data["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_transitions_resource_valid_json(self, mock_transition_response):
        """Test that resource returns valid, parseable JSON."""
        with patch(
            "hidden_regime_mcp.resources.get_transition_probabilities",
            new_callable=AsyncMock,
            return_value=mock_transition_response,
        ):
            result = await get_transitions_resource("SPY")

            # Should not raise json.JSONDecodeError
            data = json.loads(result)
            assert isinstance(data, dict)


class TestResourceErrorRecovery:
    """Test resource error handling and recovery."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_current_regime_resource_empty_ticker(self):
        """Test current regime resource with empty ticker."""
        with patch(
            "hidden_regime_mcp.resources.detect_regime",
            new_callable=AsyncMock,
            side_effect=ToolError("Ticker symbol is required"),
        ):
            result = await get_current_regime_resource("")

            # Should return valid JSON with error
            data = json.loads(result)
            assert "error" in data or "ticker" in data

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_transitions_resource_empty_ticker(self):
        """Test transitions resource with empty ticker."""
        with patch(
            "hidden_regime_mcp.resources.get_transition_probabilities",
            new_callable=AsyncMock,
            side_effect=ToolError("Ticker symbol is required"),
        ):
            result = await get_transitions_resource("")

            # Should return valid JSON with error
            data = json.loads(result)
            assert "error" in data or "ticker" in data

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_current_regime_resource_json_serializable(self):
        """Test that current regime resource output is JSON serializable."""
        mock_response = {
            "ticker": "SPY",
            "current_regime": "bull",
            "confidence": 0.85,
            "mean_return": 0.002,
            "volatility": 0.015,
            "last_updated": "2024-12-31",
            "n_states": 3,
            "analysis_period": {"start": "2024-01-01", "end": "2024-12-31"},
            "days_in_regime": 25,
            "expected_duration": 30.5,
            "percent_complete": 82.0,
            "regime_status": "mature",
            "days_until_expected_transition": 5.5,
            "price_performance": {"ytd_return": 0.187},
            "current_price": 450.25,
            "price_trend": "up",
            "regime_stability": "stable",
            "recent_transitions": 1,
            "previous_regime": "sideways",
            "last_transition_date": "2024-12-06",
            "interpretation": "Test",
            "explanation": "Test explanation",
        }

        with patch(
            "hidden_regime_mcp.resources.detect_regime",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await get_current_regime_resource("SPY")

            # Should be valid JSON
            data = json.loads(result)
            # Should be re-serializable
            json.dumps(data)  # Should not raise

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_transitions_resource_json_serializable(self):
        """Test that transitions resource output is JSON serializable."""
        mock_response = {
            "ticker": "SPY",
            "transition_matrix": {
                "bull": {"bull": 0.85, "bear": 0.05, "sideways": 0.10},
            },
            "expected_durations": {"bull": 20.0},
            "steady_state": {"bull": 0.45},
            "n_states": 3,
        }

        with patch(
            "hidden_regime_mcp.resources.get_transition_probabilities",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await get_transitions_resource("SPY")

            # Should be valid JSON
            data = json.loads(result)
            # Should be re-serializable
            json.dumps(data)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
