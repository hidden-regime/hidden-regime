"""
Signal Generation Tests - Domain Logic.

Tests verify that trading signals are generated correctly from regime interpretations.
Critical for business value - signals drive actual trading decisions.

Tests cover:
1. Signal direction (long/short/neutral)
2. Signal strength/sizing
3. Risk management (stops, limits)
4. Strategy implementations (following, contrarian, confidence-weighted)
"""

import pytest
import numpy as np
import pandas as pd

from hidden_regime.signal_generation.financial import FinancialSignalGenerator
from hidden_regime.config.signal_generation import SignalGenerationConfiguration


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_config():
    """Basic signal generation configuration."""
    return SignalGenerationConfiguration(
        strategy_type="regime_following",
        position_size_range=(0.0, 1.0)  # Correct parameter name
    )


@pytest.fixture
def bullish_regime_input():
    """Interpreter output indicating bullish regime."""
    return pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['bullish'],
        'regime_confidence': [0.8],
        'regime_strength': [0.75],
        'signal_valid': [True]
    })


@pytest.fixture
def bearish_regime_input():
    """Interpreter output indicating bearish regime."""
    return pd.DataFrame({
        'state': [1],  # REQUIRED by signal generator
        'regime_type': ['bearish'],
        'regime_confidence': [0.85],
        'regime_strength': [0.7],
        'signal_valid': [True]
    })


@pytest.fixture
def neutral_regime_input():
    """Interpreter output indicating neutral/sideways regime."""
    return pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['sideways'],
        'regime_confidence': [0.6],
        'regime_strength': [0.5],
        'signal_valid': [True]
    })


@pytest.fixture
def crisis_regime_input():
    """Interpreter output indicating crisis regime."""
    return pd.DataFrame({
        'state': [1],  # REQUIRED by signal generator
        'regime_type': ['crisis'],
        'regime_confidence': [0.9],
        'regime_strength': [0.8],
        'signal_valid': [True]
    })


# ============================================================================
# Basic Signal Direction Tests
# ============================================================================


@pytest.mark.unit
def test_bullish_regime_generates_long_signal(basic_config, bullish_regime_input):
    """Test that bullish regime generates long (positive) signal."""
    generator = FinancialSignalGenerator(basic_config)

    result = generator.update(bullish_regime_input)

    signal = result.iloc[0]['position_size']

    assert signal > 0, f"Bullish regime should generate positive signal, got {signal}"
    assert signal <= 1.0, "Signal should not exceed max position"


@pytest.mark.unit
def test_bearish_regime_generates_short_signal(basic_config, bearish_regime_input):
    """Test that bearish regime generates short (negative) signal."""
    generator = FinancialSignalGenerator(basic_config)

    result = generator.update(bearish_regime_input)

    signal = result.iloc[0]['position_size']

    assert signal < 0, f"Bearish regime should generate negative signal, got {signal}"
    assert signal >= -1.0, "Signal should not exceed max short position"


@pytest.mark.unit
def test_neutral_regime_generates_no_signal(basic_config, neutral_regime_input):
    """Test that neutral/sideways regime generates zero or near-zero signal."""
    generator = FinancialSignalGenerator(basic_config)

    result = generator.update(neutral_regime_input)

    signal = result.iloc[0]['position_size']

    assert abs(signal) <= 0.1, \
        f"Neutral regime should generate near-zero signal, got {signal}"


@pytest.mark.unit
def test_crisis_regime_defensive_signal(basic_config, crisis_regime_input):
    """Test that crisis regime generates defensive (short or neutral) signal."""
    generator = FinancialSignalGenerator(basic_config)

    result = generator.update(crisis_regime_input)

    signal = result.iloc[0]['position_size']

    # Crisis should be defensive: short or flat
    assert signal <= 0, f"Crisis regime should be defensive (<=0), got {signal}"


# ============================================================================
# Strategy Implementation Tests
# ============================================================================


@pytest.mark.unit
def test_regime_following_strategy(bullish_regime_input, bearish_regime_input):
    """Test regime-following strategy implementation."""
    config = SignalGenerationConfiguration(strategy_type="regime_following")
    generator = FinancialSignalGenerator(config)

    # Bullish should be long
    result_bull = generator.update(bullish_regime_input)
    assert result_bull.iloc[0]['position_size'] > 0, "Should follow bullish regime"

    # Bearish should be short
    result_bear = generator.update(bearish_regime_input)
    assert result_bear.iloc[0]['position_size'] < 0, "Should follow bearish regime"


@pytest.mark.unit
def test_regime_contrarian_strategy(bullish_regime_input, bearish_regime_input):
    """Test regime-contrarian strategy (fade the regime)."""
    config = SignalGenerationConfiguration(strategy_type="regime_contrarian")
    generator = FinancialSignalGenerator(config)

    # Bullish should generate SHORT (contrarian)
    result_bull = generator.update(bullish_regime_input)
    assert result_bull.iloc[0]['position_size'] < 0, \
        "Contrarian should short bullish regime"

    # Bearish should generate LONG (contrarian)
    result_bear = generator.update(bearish_regime_input)
    assert result_bear.iloc[0]['position_size'] > 0, \
        "Contrarian should long bearish regime"


@pytest.mark.unit
def test_confidence_weighted_strategy():
    """Test that confidence-weighted strategy scales by confidence."""
    config = SignalGenerationConfiguration(strategy_type="confidence_weighted")
    generator = FinancialSignalGenerator(config)

    # High confidence bullish
    high_conf = pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['bullish'],
        'regime_confidence': [0.9],
        'regime_strength': [0.9],
        'signal_valid': [True]
    })

    # Low confidence bullish
    low_conf = pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['bullish'],
        'regime_confidence': [0.6],
        'regime_strength': [0.6],
        'signal_valid': [True]
    })

    signal_high = generator.update(high_conf).iloc[0]['position_size']
    signal_low = generator.update(low_conf).iloc[0]['position_size']

    # High confidence should have larger position
    assert abs(signal_high) > abs(signal_low), \
        f"High confidence ({signal_high}) should exceed low confidence ({signal_low})"


# ============================================================================
# Position Sizing Tests
# ============================================================================


@pytest.mark.unit
def test_position_sizing_fixed():
    """Test fixed position sizing."""
    config = SignalGenerationConfiguration(
        strategy_type="regime_following",
        position_size_range=(1.0, 1.0)  # Fixed sizing via equal min/max
    )
    generator = FinancialSignalGenerator(config)

    regime_input = pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['bullish'],
        'regime_confidence': [0.8],
        'regime_strength': [0.7],
        'signal_valid': [True]
    })

    result = generator.update(regime_input)
    signal = result.iloc[0]['position_size']

    # Fixed sizing should give full position
    assert abs(signal) == 1.0, f"Fixed sizing should give 1.0, got {abs(signal)}"


@pytest.mark.unit
def test_position_sizing_confidence_based():
    """Test confidence-based position sizing."""
    config = SignalGenerationConfiguration(
        strategy_type="confidence_weighted",  # Use confidence-weighted strategy
        position_size_range=(0.0, 1.0)
    )
    generator = FinancialSignalGenerator(config)

    # Different confidence levels
    high_conf = pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['bullish'],
        'regime_confidence': [0.95],
        'regime_strength': [0.95],
        'signal_valid': [True]
    })

    medium_conf = pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['bullish'],
        'regime_confidence': [0.7],
        'regime_strength': [0.7],
        'signal_valid': [True]
    })

    signal_high = abs(generator.update(high_conf).iloc[0]['position_size'])
    signal_med = abs(generator.update(medium_conf).iloc[0]['position_size'])

    # Higher confidence should yield larger position
    assert signal_high > signal_med, \
        f"High conf ({signal_high}) should exceed medium conf ({signal_med})"


@pytest.mark.unit
def test_max_position_size_respected():
    """Test that signals never exceed max position size."""
    config = SignalGenerationConfiguration(
        strategy_type="regime_following",
        position_size_range=(0.0, 0.5)  # Limit to 50%
    )
    generator = FinancialSignalGenerator(config)

    regime_input = pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['bullish'],
        'regime_confidence': [1.0],  # Maximum confidence
        'regime_strength': [1.0],
        'signal_valid': [True]
    })

    result = generator.update(regime_input)
    signal = abs(result.iloc[0]['position_size'])

    assert signal <= 0.5, f"Signal ({signal}) should not exceed max position size (0.5)"


# ============================================================================
# Risk Management Tests
# ============================================================================


@pytest.mark.unit
def test_signal_generation_with_invalid_flag():
    """Test that invalid signals are zeroed out."""
    config = SignalGenerationConfiguration(strategy_type="regime_following")
    generator = FinancialSignalGenerator(config)

    invalid_input = pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['bullish'],
        'regime_confidence': [0.9],
        'regime_strength': [0.9],
        'signal_valid': [False]  # Invalid signal
    })

    result = generator.update(invalid_input)
    signal = result.iloc[0]['position_size']

    assert signal == 0.0, f"Invalid signal should be zero, got {signal}"


@pytest.mark.unit
def test_signal_strength_quantification():
    """Test that signal strength is quantified correctly."""
    config = SignalGenerationConfiguration(strategy_type="confidence_weighted")
    generator = FinancialSignalGenerator(config)

    regime_input = pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['bullish'],
        'regime_confidence': [0.85],
        'regime_strength': [0.75],
        'signal_valid': [True]
    })

    result = generator.update(regime_input)

    # Should have signal strength metric
    if 'signal_strength' in result.columns:
        strength = result.iloc[0]['signal_strength']
        assert 0 <= strength <= 1.0, "Signal strength should be in [0, 1]"


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.unit
def test_missing_regime_type_handling():
    """Test graceful handling when regime_type is missing."""
    config = SignalGenerationConfiguration(strategy_type="regime_following")
    generator = FinancialSignalGenerator(config)

    incomplete_input = pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_confidence': [0.8],
        'regime_strength': [0.8],
        'signal_valid': [True]
        # Missing regime_type
    })

    result = generator.update(incomplete_input)
    signal = result.iloc[0]['position_size']

    # Should default to neutral (zero signal)
    assert signal == 0.0, f"Missing regime_type should give zero signal, got {signal}"


@pytest.mark.unit
def test_unknown_regime_type():
    """Test handling of unrecognized regime types."""
    config = SignalGenerationConfiguration(strategy_type="regime_following")
    generator = FinancialSignalGenerator(config)

    unknown_regime = pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['unknown_regime_xyz'],
        'regime_confidence': [0.8],
        'regime_strength': [0.8],
        'signal_valid': [True]
    })

    result = generator.update(unknown_regime)
    signal = result.iloc[0]['position_size']

    # Should default to neutral for unknown regime
    assert abs(signal) <= 0.1, \
        f"Unknown regime should give near-zero signal, got {signal}"


@pytest.mark.unit
def test_zero_confidence_handling():
    """Test handling of zero or very low confidence."""
    config = SignalGenerationConfiguration(
        strategy_type="confidence_weighted"
    )
    generator = FinancialSignalGenerator(config)

    zero_conf = pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['bullish'],
        'regime_confidence': [0.0],
        'regime_strength': [0.0],
        'signal_valid': [True]
    })

    result = generator.update(zero_conf)
    signal = result.iloc[0]['position_size']

    # Zero confidence should give zero or minimal signal
    assert abs(signal) < 0.2, \
        f"Zero confidence should give minimal signal, got {signal}"


@pytest.mark.unit
def test_signal_bounds_are_respected():
    """Test that signals are always in valid range [-1, 1] or within max_position."""
    config = SignalGenerationConfiguration(
        strategy_type="confidence_weighted",
        position_size_range=(0.0, 1.0)
    )
    generator = FinancialSignalGenerator(config)

    # Try extreme values
    extreme_input = pd.DataFrame({
        'state': [0],  # REQUIRED by signal generator
        'regime_type': ['bullish'],
        'regime_confidence': [2.0],  # Invalid: > 1.0
        'regime_strength': [5.0],    # Invalid: > 1.0
        'signal_valid': [True]
    })

    result = generator.update(extreme_input)
    signal = result.iloc[0]['position_size']

    # Should be bounded
    assert -1.0 <= signal <= 1.0, \
        f"Signal ({signal}) should be bounded to [-1, 1]"


@pytest.mark.unit
def test_multiple_regime_time_series():
    """Test signal generation across time series with regime changes."""
    config = SignalGenerationConfiguration(strategy_type="regime_following")
    generator = FinancialSignalGenerator(config)

    # Time series transitioning from bullish to bearish
    time_series = pd.DataFrame({
        'state': [0, 0, 1, 1, 1],  # REQUIRED by signal generator
        'regime_type': ['bullish', 'bullish', 'sideways', 'bearish', 'bearish'],
        'regime_confidence': [0.8, 0.7, 0.6, 0.75, 0.85],
        'regime_strength': [0.8, 0.7, 0.6, 0.75, 0.85],
        'signal_valid': [True, True, True, True, True]
    })

    result = generator.update(time_series)
    signals = result['position_size'].values

    # First two should be positive (bullish)
    assert signals[0] > 0 and signals[1] > 0, "Bullish periods should be long"

    # Middle should be near zero (sideways)
    assert abs(signals[2]) < 0.2, "Sideways should be neutral"

    # Last two should be negative (bearish)
    assert signals[3] < 0 and signals[4] < 0, "Bearish periods should be short"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
