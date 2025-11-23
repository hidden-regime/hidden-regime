"""
Unit tests for formatting utilities.

Tests formatting functions for strategy names, regime types, percentages, and currency.
"""

import pytest

from hidden_regime.utils.formatting import (
    format_strategy_name,
    format_strategy_names_dict,
    format_regime_type,
    format_percentage,
    format_currency,
)


class TestFormatStrategyName:
    """Tests for format_strategy_name function."""

    @pytest.mark.unit
    def test_ta_strategy_formatting(self):
        """Test formatting of technical indicator strategies."""
        assert format_strategy_name("ta_rsi") == "Ta Rsi"
        assert format_strategy_name("ta_macd") == "Ta Macd"
        assert format_strategy_name("ta_bollinger_bands") == "Ta Bollinger Bands"
        assert format_strategy_name("ta_moving_average") == "Ta Moving Average"

    @pytest.mark.unit
    def test_hmm_strategy_formatting(self):
        """Test formatting of HMM strategies."""
        assert format_strategy_name("hmm_regime_following") == "Hmm Regime Following"
        assert format_strategy_name("hmm_regime_detection") == "Hmm Regime Detection"
        assert format_strategy_name("hmm_bull_bear") == "Hmm Bull Bear"

    @pytest.mark.unit
    def test_generic_strategy_formatting(self):
        """Test formatting of generic strategies without prefix."""
        assert format_strategy_name("buy_hold") == "Buy Hold"
        assert format_strategy_name("momentum") == "Momentum"
        assert format_strategy_name("mean_reversion") == "Mean Reversion"

    @pytest.mark.unit
    def test_single_word_strategy(self):
        """Test formatting of single-word strategies."""
        assert format_strategy_name("ta_rsi") == "Ta Rsi"
        assert format_strategy_name("hmm") == "Hmm"
        assert format_strategy_name("momentum") == "Momentum"

    @pytest.mark.unit
    def test_empty_string(self):
        """Test formatting of empty string."""
        assert format_strategy_name("") == ""


class TestFormatStrategyNamesDict:
    """Tests for format_strategy_names_dict function."""

    @pytest.mark.unit
    def test_empty_dict(self):
        """Test formatting empty dictionary."""
        assert format_strategy_names_dict({}) == {}

    @pytest.mark.unit
    def test_single_strategy(self):
        """Test formatting dictionary with single strategy."""
        input_dict = {"ta_rsi": {"returns": 0.15, "sharpe": 1.5}}
        expected = {"Ta Rsi": {"returns": 0.15, "sharpe": 1.5}}
        assert format_strategy_names_dict(input_dict) == expected

    @pytest.mark.unit
    def test_multiple_strategies(self):
        """Test formatting dictionary with multiple strategies."""
        input_dict = {
            "ta_rsi": {"returns": 0.15},
            "hmm_regime_following": {"returns": 0.20},
            "buy_hold": {"returns": 0.10},
        }
        expected = {
            "Ta Rsi": {"returns": 0.15},
            "Hmm Regime Following": {"returns": 0.20},
            "Buy Hold": {"returns": 0.10},
        }
        assert format_strategy_names_dict(input_dict) == expected

    @pytest.mark.unit
    def test_preserves_data_structure(self):
        """Test that data structure is preserved."""
        input_dict = {
            "ta_macd": {
                "metrics": {"sharpe": 1.2, "sortino": 1.5},
                "trades": [1, 2, 3],
            }
        }
        result = format_strategy_names_dict(input_dict)
        assert "Ta Macd" in result
        assert result["Ta Macd"]["metrics"] == {"sharpe": 1.2, "sortino": 1.5}
        assert result["Ta Macd"]["trades"] == [1, 2, 3]


class TestFormatRegimeType:
    """Tests for format_regime_type function."""

    @pytest.mark.unit
    def test_uppercase_regime(self):
        """Test formatting uppercase regime types."""
        assert format_regime_type("BULLISH") == "Bullish"
        assert format_regime_type("BEARISH") == "Bearish"
        assert format_regime_type("SIDEWAYS") == "Sideways"
        assert format_regime_type("CRISIS") == "Crisis"

    @pytest.mark.unit
    def test_lowercase_regime(self):
        """Test formatting lowercase regime types."""
        assert format_regime_type("bullish") == "Bullish"
        assert format_regime_type("bearish") == "Bearish"
        assert format_regime_type("sideways") == "Sideways"

    @pytest.mark.unit
    def test_mixed_case_regime(self):
        """Test formatting mixed case regime types."""
        assert format_regime_type("BuLLiSh") == "Bullish"
        assert format_regime_type("BeArIsH") == "Bearish"

    @pytest.mark.unit
    def test_empty_string(self):
        """Test formatting empty string."""
        assert format_regime_type("") == ""


class TestFormatPercentage:
    """Tests for format_percentage function."""

    @pytest.mark.unit
    def test_default_decimals(self):
        """Test formatting with default 2 decimal places."""
        assert format_percentage(0.0523) == "5.23%"
        assert format_percentage(0.15) == "15.00%"
        assert format_percentage(1.0) == "100.00%"
        assert format_percentage(0.0) == "0.00%"

    @pytest.mark.unit
    def test_custom_decimals(self):
        """Test formatting with custom decimal places."""
        assert format_percentage(0.0523, decimals=1) == "5.2%"
        assert format_percentage(0.0523, decimals=3) == "5.230%"
        assert format_percentage(0.0523, decimals=0) == "5%"

    @pytest.mark.unit
    def test_negative_values(self):
        """Test formatting negative percentages."""
        assert format_percentage(-0.0523) == "-5.23%"
        assert format_percentage(-0.15, decimals=1) == "-15.0%"

    @pytest.mark.unit
    def test_small_values(self):
        """Test formatting very small percentages."""
        assert format_percentage(0.0001) == "0.01%"
        assert format_percentage(0.000001, decimals=4) == "0.0001%"

    @pytest.mark.unit
    def test_large_values(self):
        """Test formatting large percentages."""
        assert format_percentage(5.234) == "523.40%"
        assert format_percentage(10.0) == "1000.00%"


class TestFormatCurrency:
    """Tests for format_currency function."""

    @pytest.mark.unit
    def test_default_decimals(self):
        """Test formatting with default 0 decimal places."""
        assert format_currency(123456) == "$123,456"
        assert format_currency(1000) == "$1,000"
        assert format_currency(0) == "$0"

    @pytest.mark.unit
    def test_with_decimals(self):
        """Test formatting with decimal places."""
        assert format_currency(123456.789, decimals=2) == "$123,456.79"
        assert format_currency(1000.5, decimals=2) == "$1,000.50"
        assert format_currency(99.99, decimals=2) == "$99.99"

    @pytest.mark.unit
    def test_negative_values(self):
        """Test formatting negative currency values."""
        assert format_currency(-123456) == "$-123,456"
        assert format_currency(-1000.5, decimals=2) == "$-1,000.50"

    @pytest.mark.unit
    def test_small_values(self):
        """Test formatting small currency values."""
        assert format_currency(0.99, decimals=2) == "$0.99"
        assert format_currency(5, decimals=0) == "$5"

    @pytest.mark.unit
    def test_large_values(self):
        """Test formatting large currency values."""
        assert format_currency(1000000) == "$1,000,000"
        assert format_currency(1234567.89, decimals=2) == "$1,234,567.89"
        assert format_currency(999999999, decimals=0) == "$999,999,999"

    @pytest.mark.unit
    def test_rounding(self):
        """Test that values are properly rounded."""
        assert format_currency(123.456, decimals=2) == "$123.46"
        assert format_currency(123.454, decimals=2) == "$123.45"
        assert format_currency(123.5, decimals=0) == "$124"
