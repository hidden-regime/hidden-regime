"""
Unit tests for simulation/signal_generators.py

Tests signal generation including buy-hold, HMM-based, and technical indicator signals.
"""

import pytest
import pandas as pd
import numpy as np

from hidden_regime.simulation.signal_generators import (
    SignalType,
    SignalGenerator,
    BuyHoldSignalGenerator,
    HMMSignalGenerator,
    TechnicalIndicatorSignalGenerator,
)


@pytest.fixture
def sample_price_data():
    """Create sample OHLC price data."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            "open": np.linspace(100, 110, 20),
            "high": np.linspace(102, 112, 20),
            "low": np.linspace(98, 108, 20),
            "close": np.linspace(100, 110, 20),
        },
        index=dates,
    )


@pytest.fixture
def sample_regime_data():
    """Create sample regime prediction data."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            "predicted_state": [0, 0, 1, 1, 2, 2, 2, 1, 1, 0] * 2,
            "confidence": [0.9, 0.85, 0.95, 0.9, 0.92, 0.88, 0.91, 0.85, 0.9, 0.87] * 2,
        },
        index=dates,
    )


# ============================================================================
# UNIT TESTS (7 tests)
# ============================================================================


def test_signal_type_enum():
    """Test SignalType enum values."""
    assert SignalType.BUY.value == 1
    assert SignalType.SELL.value == -1
    assert SignalType.HOLD.value == 0


def test_create_buy_signal():
    """Test creating buy signal."""
    signal = SignalType.BUY
    assert signal.value == 1


def test_create_sell_signal():
    """Test creating sell signal."""
    signal = SignalType.SELL
    assert signal.value == -1


def test_buy_hold_generator_basic(sample_price_data):
    """Test buy-and-hold signal generator."""
    generator = BuyHoldSignalGenerator()
    signals = generator.generate_signals(sample_price_data)

    # First signal should be BUY
    assert signals.iloc[0] == SignalType.BUY.value
    # Rest should be HOLD
    assert all(signals.iloc[1:] == SignalType.HOLD.value)
    assert len(signals) == len(sample_price_data)


def test_signal_validation(sample_price_data):
    """Test signal data validation."""
    generator = BuyHoldSignalGenerator()
    assert generator.validate_data(sample_price_data) is True

    # Invalid data (missing required columns)
    invalid_data = pd.DataFrame({"close": [100, 101, 102]})
    assert generator.validate_data(invalid_data) is False


def test_hmm_signal_generator_regime_following(sample_price_data, sample_regime_data):
    """Test HMM regime-following signal generation."""
    generator = HMMSignalGenerator(strategy_type="regime_following")
    signals = generator.generate_signals(sample_price_data, sample_regime_data)

    assert len(signals) == len(sample_price_data)
    # Should have mix of buy/sell/hold signals
    assert SignalType.BUY.value in signals.values
    assert SignalType.SELL.value in signals.values


def test_hmm_signal_generator_confidence_weighted(sample_price_data, sample_regime_data):
    """Test HMM confidence-weighted signal generation."""
    generator = HMMSignalGenerator(strategy_type="confidence_weighted")
    signals = generator.generate_signals(sample_price_data, sample_regime_data)

    assert len(signals) == len(sample_price_data)
    # Signals should exist
    assert signals is not None


# ============================================================================
# INTEGRATION TESTS (3 tests)
# ============================================================================


def test_signal_generators_with_regime_interpreter():
    """Test signal generators with regime interpreter output."""
    # Create realistic regime data
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    price_data = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        },
        index=dates,
    )

    regime_data = pd.DataFrame(
        {
            "predicted_state": [0, 0, 1, 1, 2, 2, 2, 2, 1, 0],
            "confidence": [0.9] * 10,
        },
        index=dates,
    )

    generator = HMMSignalGenerator(strategy_type="regime_following")
    signals = generator.generate_signals(price_data, regime_data)

    # Should generate signals based on regime transitions
    assert len(signals) == 10
    assert isinstance(signals, pd.Series)


def test_signal_generators_multi_strategy():
    """Test multiple signal generators with same data."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    price_data = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        },
        index=dates,
    )

    # Buy-and-hold strategy
    bh_generator = BuyHoldSignalGenerator()
    bh_signals = bh_generator.generate_signals(price_data)

    assert len(bh_signals) == 10
    assert bh_signals.iloc[0] == SignalType.BUY.value


def test_signal_generators_edge_cases():
    """Test signal generators with edge cases."""
    # Empty data
    empty_data = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": []},
    )

    generator = BuyHoldSignalGenerator()
    signals = generator.generate_signals(empty_data)

    # Should handle empty data gracefully
    assert len(signals) == 0


def test_technical_indicator_signal_generator():
    """Test technical indicator signal generator."""
    dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
    price_data = pd.DataFrame(
        {
            "open": np.random.uniform(95, 105, 30),
            "high": np.random.uniform(100, 110, 30),
            "low": np.random.uniform(90, 100, 30),
            "close": np.random.uniform(95, 105, 30),
        },
        index=dates,
    )

    # SMA crossover signal generator
    generator = TechnicalIndicatorSignalGenerator(
        indicator_name="sma", indicator_params={"period": 10}
    )

    signals = generator.generate_signals(price_data)

    assert len(signals) == len(price_data)
    assert isinstance(signals, pd.Series)
