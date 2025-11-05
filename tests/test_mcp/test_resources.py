"""
Tests for MCP resources.

Tests resource URI handlers for regime detection data.
"""

import json
import pytest

from hidden_regime_mcp.resources import (
    get_current_regime_resource,
    get_transitions_resource,
)


@pytest.mark.network
class TestCurrentRegimeResource:
    """Test regime://{ticker}/current resource."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_current_regime_resource_success(self):
        """Test current regime resource returns valid JSON for valid ticker."""
        result = await get_current_regime_resource("SPY")

        # Should return valid JSON
        data = json.loads(result)

        # Check structure
        assert "ticker" in data
        assert "current_regime" in data
        assert "confidence" in data
        assert "mean_return" in data
        assert "volatility" in data
        assert "last_updated" in data

        # Check values
        assert data["ticker"] == "SPY"
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["mean_return"], float)
        assert isinstance(data["volatility"], float)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_current_regime_resource_lowercase_ticker(self):
        """Test resource handles lowercase ticker by converting to uppercase."""
        result = await get_current_regime_resource("spy")

        data = json.loads(result)
        assert data["ticker"] == "SPY"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_current_regime_resource_invalid_ticker(self):
        """Test current regime resource returns error JSON for invalid ticker."""
        result = await get_current_regime_resource("INVALIDTICKER12345")

        # Should return error JSON, not raise exception
        data = json.loads(result)

        # Check error structure
        assert "error" in data
        assert "ticker" in data
        assert data["ticker"] == "INVALIDTICKER12345"
        assert "Unable to load data" in data["error"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_current_regime_resource_includes_temporal_context(self):
        """Test current regime resource includes enhanced temporal context fields."""
        result = await get_current_regime_resource("SPY")

        data = json.loads(result)

        # Check temporal context fields
        assert "days_in_regime" in data
        assert "expected_duration" in data
        assert "percent_complete" in data
        assert "regime_status" in data

        # Check price context fields
        assert "price_performance" in data
        assert "current_price" in data
        assert "price_trend" in data

        # Check stability fields
        assert "regime_stability" in data
        assert "recent_transitions" in data

        # Check interpretation fields
        assert "interpretation" in data
        assert "explanation" in data

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_current_regime_resource_interpretation_quality(self):
        """Test resource includes meaningful interpretation."""
        result = await get_current_regime_resource("NVDA")

        data = json.loads(result)

        if "error" not in data:  # Only check if request succeeded
            interpretation = data.get("interpretation", "")
            explanation = data.get("explanation", "")

            # Interpretation should be concise
            assert len(interpretation) > 0
            assert len(interpretation) < 200

            # Explanation should be detailed
            assert len(explanation) > 0
            assert "volatility" in explanation.lower() or "return" in explanation.lower()


@pytest.mark.network
class TestTransitionsResource:
    """Test regime://{ticker}/transitions resource."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_transitions_resource_success(self):
        """Test transitions resource returns valid JSON for valid ticker."""
        result = await get_transitions_resource("SPY")

        # Should return valid JSON
        data = json.loads(result)

        # Check structure
        assert "ticker" in data
        assert "transition_matrix" in data
        assert "expected_durations" in data
        assert "steady_state" in data
        assert "n_states" in data

        # Check values
        assert data["ticker"] == "SPY"
        assert isinstance(data["transition_matrix"], dict)
        assert isinstance(data["expected_durations"], dict)
        assert isinstance(data["steady_state"], dict)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_transitions_resource_lowercase_ticker(self):
        """Test resource handles lowercase ticker by converting to uppercase."""
        result = await get_transitions_resource("spy")

        data = json.loads(result)
        assert data["ticker"] == "SPY"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_transitions_resource_invalid_ticker(self):
        """Test transitions resource returns error JSON for invalid ticker."""
        result = await get_transitions_resource("INVALIDTICKER12345")

        # Should return error JSON, not raise exception
        data = json.loads(result)

        # Check error structure
        assert "error" in data
        assert "ticker" in data
        assert data["ticker"] == "INVALIDTICKER12345"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_transitions_resource_matrix_structure(self):
        """Test transition matrix has valid probability structure."""
        result = await get_transitions_resource("SPY")

        data = json.loads(result)

        if "error" not in data:  # Only check if request succeeded
            matrix = data["transition_matrix"]

            # Check matrix is square
            regimes = list(matrix.keys())
            assert len(regimes) >= 2  # At least 2 states

            for from_regime in regimes:
                to_probs = matrix[from_regime]

                # Each row should have transitions to all states
                assert len(to_probs) == len(regimes)

                # Probabilities should sum to approximately 1.0
                total_prob = sum(to_probs.values())
                assert 0.99 <= total_prob <= 1.01, f"Probabilities sum to {total_prob}"

                # All individual probabilities should be between 0 and 1
                for prob in to_probs.values():
                    assert 0 <= prob <= 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_transitions_resource_durations_positive(self):
        """Test expected durations are positive."""
        result = await get_transitions_resource("SPY")

        data = json.loads(result)

        if "error" not in data:  # Only check if request succeeded
            durations = data["expected_durations"]

            for regime, duration in durations.items():
                # Duration should be positive (or inf for absorbing states)
                assert duration > 0 or duration == float("inf")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_transitions_resource_steady_state_sums_to_one(self):
        """Test steady state probabilities sum to 1."""
        result = await get_transitions_resource("SPY")

        data = json.loads(result)

        if "error" not in data:  # Only check if request succeeded
            steady_state = data["steady_state"]

            # Steady state probabilities should sum to 1
            total_prob = sum(steady_state.values())
            assert 0.99 <= total_prob <= 1.01, f"Steady state sums to {total_prob}"

            # All individual probabilities should be between 0 and 1
            for prob in steady_state.values():
                assert 0 <= prob <= 1


class TestResourceErrorRecovery:
    """Test resource error handling and recovery."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_current_regime_resource_graceful_failure(self):
        """Test current regime resource doesn't crash on errors."""
        # Should not raise exception even with invalid input
        result = await get_current_regime_resource("")

        # Should return valid JSON with error
        data = json.loads(result)
        assert "error" in data or "ticker" in data

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transitions_resource_graceful_failure(self):
        """Test transitions resource doesn't crash on errors."""
        # Should not raise exception even with invalid input
        result = await get_transitions_resource("")

        # Should return valid JSON with error
        data = json.loads(result)
        assert "error" in data or "ticker" in data
