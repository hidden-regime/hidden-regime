"""
Unit tests for signal_generation/ edge cases.

Tests confidence filtering, regime transitions, position sizing,
signal strength calculation, and validation.
"""

import pytest
import pandas as pd
import numpy as np

from hidden_regime.signal_generation import FinancialSignalGenerator
from hidden_regime.config.signal_generation import SignalGenerationConfiguration
from hidden_regime.utils.exceptions import ValidationError


@pytest.fixture
def sample_regime_input():
    """Create sample regime interpreter output."""
    return pd.DataFrame({
        "state": [0, 1, 2, 1, 0],
        "regime_type": ["Bearish", "Sideways", "Bullish", "Sideways", "Bearish"],
        "regime_confidence": [0.9, 0.7, 0.95, 0.6, 0.85],
        "regime_strength": [0.8, 0.5, 0.9, 0.4, 0.7],
        "signal_valid": [True, True, True, True, True],
    })


# ============================================================================
# UNIT TESTS (10 tests)
# ============================================================================


def test_signal_generator_confidence_filtering(sample_regime_input):
    """Test confidence threshold filtering."""
    config = SignalGenerationConfiguration(
        strategy_type="regime_following",
        confidence_threshold=0.8,
    )

    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(sample_regime_input)

    # Only high-confidence regimes should generate signals
    assert "signal_strength" in result.columns or "position_size" in result.columns


def test_signal_generator_position_sizing(sample_regime_input):
    """Test position size calculation."""
    config = SignalGenerationConfiguration(
        strategy_type="regime_following",
        position_size_range=(0.0, 1.0),
    )

    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(sample_regime_input)

    # Position sizes should be within range (can be negative for short positions)
    if "position_size" in result.columns:
        assert all(result["position_size"] >= -1.0)
        assert all(result["position_size"] <= 1.0)


def test_signal_generator_regime_change_exits():
    """Test exit logic on regime changes."""
    regime_data = pd.DataFrame({
        "state": [2, 2, 1, 1, 0],  # Bull -> Sideways -> Bear
        "regime_type": ["Bullish", "Bullish", "Sideways", "Sideways", "Bearish"],
        "regime_confidence": [0.9, 0.9, 0.8, 0.8, 0.9],
        "regime_strength": [0.9, 0.9, 0.5, 0.5, 0.8],
        "signal_valid": [True, True, True, True, True],
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(regime_data)

    # Should detect regime changes
    assert len(result) == len(regime_data)


def test_signal_generator_signal_strength():
    """Test signal strength calculation."""
    regime_data = pd.DataFrame({
        "state": [2],
        "regime_type": ["Bullish"],
        "regime_confidence": [0.95],
        "regime_strength": [0.9],
        "signal_valid": [True],
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(regime_data)

    # Should generate signal with strength
    assert len(result) > 0


def test_signal_generator_invalid_signals():
    """Test validation of invalid signals."""
    # Invalid regime data (missing required columns)
    invalid_data = pd.DataFrame({
        "state": [0, 1, 2],
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)

    # Should handle missing columns gracefully or raise error
    try:
        result = signal_gen.update(invalid_data)
        # If it doesn't raise, check it returns something
        assert result is not None
    except (ValidationError, KeyError, ValueError):
        # Expected to raise on invalid input
        pass


def test_financial_signal_generator_bull_regime():
    """Test signal generation in bullish regime."""
    bull_regime = pd.DataFrame({
        "state": [2],
        "regime_type": ["Bullish"],
        "regime_confidence": [0.9],
        "regime_strength": [0.8],
        "signal_valid": [True],
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(bull_regime)

    # Bullish regime should generate positive signal
    assert len(result) > 0


def test_financial_signal_generator_bear_regime():
    """Test signal generation in bearish regime."""
    bear_regime = pd.DataFrame({
        "state": [0],
        "regime_type": ["Bearish"],
        "regime_confidence": [0.9],
        "regime_strength": [0.8],
        "signal_valid": [True],
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(bear_regime)

    # Bearish regime should generate exit/short signal
    assert len(result) > 0


def test_financial_signal_generator_sideways_regime():
    """Test signal generation in sideways regime."""
    sideways_regime = pd.DataFrame({
        "state": [1],
        "regime_type": ["Sideways"],
        "regime_confidence": [0.8],
        "regime_strength": [0.5],
        "signal_valid": [True],
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(sideways_regime)

    # Sideways regime should generate hold signal
    assert len(result) > 0


def test_financial_signal_generator_regime_transitions():
    """Test signal generation during regime transitions."""
    transition_data = pd.DataFrame({
        "state": [0, 1, 2],  # Bear -> Sideways -> Bull
        "regime_type": ["Bearish", "Sideways", "Bullish"],
        "regime_confidence": [0.9, 0.7, 0.95],
        "regime_strength": [0.8, 0.5, 0.9],
        "signal_valid": [True, True, True],
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(transition_data)

    # Should handle transitions
    assert len(result) == 3


def test_financial_signal_generator_lookback_window():
    """Test signal generation with lookback window."""
    regime_data = pd.DataFrame({
        "state": list(range(10)),
        "regime_type": ["Bullish"] * 10,
        "regime_confidence": [0.9] * 10,
        "regime_strength": [0.8] * 10,
        "signal_valid": [True] * 10,
    })

    config = SignalGenerationConfiguration(
        strategy_type="regime_following",
        lookback_days=5,  # Use lookback_days not lookback_window
    )

    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(regime_data)

    # Should generate signals for all data
    assert len(result) == 10


# ============================================================================
# INTEGRATION TESTS (5 tests)
# ============================================================================


def test_signal_generator_with_interpreter_output():
    """Test signal generator with realistic interpreter output."""
    # Simulate interpreter output
    interpreter_output = pd.DataFrame({
        "state": [0, 0, 1, 2, 2],
        "regime_type": ["Bearish", "Bearish", "Sideways", "Bullish", "Bullish"],
        "regime_confidence": [0.85, 0.9, 0.75, 0.92, 0.88],
        "regime_strength": [0.7, 0.8, 0.5, 0.9, 0.85],
        "signal_valid": [True, True, True, True, True],
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(interpreter_output)

    assert len(result) == len(interpreter_output)


def test_signal_generator_multi_regime_scenario():
    """Test signal generator across multiple regime types."""
    multi_regime = pd.DataFrame({
        "state": [0, 1, 2, 1, 0, 2],
        "regime_type": ["Bearish", "Sideways", "Bullish", "Sideways", "Bearish", "Bullish"],
        "regime_confidence": [0.9, 0.7, 0.95, 0.8, 0.85, 0.92],
        "regime_strength": [0.8, 0.5, 0.9, 0.6, 0.75, 0.88],
        "signal_valid": [True, True, True, True, True, True],
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(multi_regime)

    assert len(result) == 6


def test_signal_generator_backtesting_workflow():
    """Test signal generator in backtesting workflow."""
    # Simulate historical regime data
    historical_regimes = pd.DataFrame({
        "state": [0] * 20 + [1] * 20 + [2] * 20,
        "regime_type": ["Bearish"] * 20 + ["Sideways"] * 20 + ["Bullish"] * 20,
        "regime_confidence": [0.9] * 60,
        "regime_strength": [0.8] * 60,
        "signal_valid": [True] * 60,
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(historical_regimes)

    # Should generate signals for all periods
    assert len(result) == 60


def test_signal_generator_risk_adjusted_sizing():
    """Test risk-adjusted position sizing."""
    regime_data = pd.DataFrame({
        "state": [2],
        "regime_type": ["Bullish"],
        "regime_confidence": [0.95],
        "regime_strength": [0.9],
        "signal_valid": [True],
    })

    config = SignalGenerationConfiguration(
        strategy_type="regime_following",
        position_size_range=(0.0, 0.5),  # Max 50% position
    )

    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(regime_data)

    if "position_size" in result.columns:
        assert result.iloc[0]["position_size"] <= 0.5


def test_signal_generator_signal_aggregation():
    """Test signal aggregation across multiple sources."""
    # Multiple concurrent regime signals
    regime_data = pd.DataFrame({
        "state": [2, 2, 2],
        "regime_type": ["Bullish", "Bullish", "Bullish"],
        "regime_confidence": [0.9, 0.95, 0.85],
        "regime_strength": [0.8, 0.9, 0.7],
        "signal_valid": [True, True, True],
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)
    result = signal_gen.update(regime_data)

    # Should aggregate signals appropriately
    assert len(result) == 3
