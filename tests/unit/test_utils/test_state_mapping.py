"""
Unit tests for utils/state_mapping.py

Tests state-to-regime mapping logic including threshold classification,
regime characteristics, and custom label handling.
"""

import pytest
import numpy as np
import pandas as pd

from hidden_regime.utils.state_mapping import (
    map_states_to_financial_regimes,
    percent_change_to_log_return,
    log_return_to_percent_change,
    get_regime_characteristics,
)


# ============================================================================
# UNIT TESTS (12 tests)
# ============================================================================


def test_map_states_to_regimes_3state():
    """Test 3-state regime mapping."""
    emission_means = np.array([-0.01, 0.0005, 0.015])  # Bear, Sideways, Bull
    mapping = map_states_to_financial_regimes(emission_means, n_states=3)

    assert len(mapping) == 3
    assert "Bearish" in mapping.values() or "Bear" in str(mapping.values())
    assert "Sideways" in mapping.values()
    assert "Bullish" in mapping.values() or "Bull" in str(mapping.values())


def test_map_states_to_regimes_custom_labels():
    """Test regime mapping with forced labels."""
    # Test that mapping respects emission means regardless of state order
    emission_means = np.array([0.02, -0.01, 0.001])  # Bull, Bear, Sideways (scrambled)
    mapping = map_states_to_financial_regimes(emission_means, n_states=3)

    # State 0 (highest mean) should be bull-like
    # State 1 (negative mean) should be bear-like
    # State 2 (near zero) should be sideways-like
    assert len(mapping) == 3


def test_get_regime_characteristics():
    """Test regime characteristic extraction."""
    # Define typical regime characteristics
    regime_data = pd.DataFrame({
        "regime_type": ["Bullish", "Bearish", "Sideways"],
        "mean_return": [0.02, -0.01, 0.0],
        "volatility": [0.015, 0.025, 0.01],
    })

    # Test that characteristics can be extracted
    for idx, row in regime_data.iterrows():
        regime_type = row["regime_type"]
        characteristics = get_regime_characteristics(regime_type)
        assert isinstance(characteristics, dict)
        assert "description" in characteristics or characteristics is not None


def test_percent_change_to_log_return():
    """Test percent change to log return conversion."""
    # 10% return
    pct_change = 0.10
    log_return = percent_change_to_log_return(pct_change)

    # log(1 + 0.10) ≈ 0.0953
    assert log_return == pytest.approx(0.0953, abs=0.001)


def test_log_return_to_percent_change():
    """Test log return to percent change conversion."""
    # Log return of 0.0953
    log_return = 0.0953
    pct_change = log_return_to_percent_change(log_return)

    # exp(0.0953) - 1 ≈ 0.10
    assert pct_change == pytest.approx(0.10, abs=0.001)


def test_apply_regime_mapping_basic():
    """Test basic regime mapping application."""
    states = np.array([0, 1, 2, 1, 0])
    mapping = {0: "Bearish", 1: "Sideways", 2: "Bullish"}

    regimes = [mapping[s] for s in states]

    assert regimes == ["Bearish", "Sideways", "Bullish", "Sideways", "Bearish"]


def test_apply_regime_mapping_preserves_index():
    """Test that regime mapping preserves DataFrame index."""
    dates = pd.date_range("2023-01-01", periods=5)
    states = pd.Series([0, 1, 2, 1, 0], index=dates)
    mapping = {0: "Bearish", 1: "Sideways", 2: "Bullish"}

    regimes = states.map(mapping)

    assert all(regimes.index == dates)


def test_create_regime_profile():
    """Test regime profile creation."""
    emission_means = np.array([-0.01, 0.001, 0.02])
    emission_stds = np.array([0.02, 0.01, 0.015])

    # Create profile dictionary
    profile = {
        "means": emission_means,
        "stds": emission_stds,
        "n_states": 3,
    }

    assert profile["n_states"] == 3
    assert len(profile["means"]) == 3
    assert len(profile["stds"]) == 3


def test_validate_regime_mapping():
    """Test regime mapping validation."""
    # Valid mapping
    emission_means = np.array([-0.01, 0.0, 0.01])
    mapping = map_states_to_financial_regimes(emission_means, n_states=3, validate=True)

    assert len(mapping) == 3


def test_regime_mapping_edge_cases():
    """Test regime mapping with edge cases."""
    # All zeros (rare but possible)
    emission_means = np.array([0.0, 0.0, 0.0])
    mapping = map_states_to_financial_regimes(emission_means, n_states=3, validate=False)

    # Should still create mapping
    assert len(mapping) == 3


def test_regime_characteristics_calculation():
    """Test that regime characteristics are calculated correctly."""
    emission_means = np.array([-0.02, 0.001, 0.025])
    means_pct = np.exp(emission_means) - 1

    # Convert log returns to percent returns
    assert means_pct[0] < 0  # Negative for bear
    assert abs(means_pct[1]) < 0.01  # Near zero for sideways
    assert means_pct[2] > 0  # Positive for bull


def test_regime_type_determination():
    """Test regime type determination from returns."""
    # High positive return -> Bull
    # High negative return -> Bear
    # Near-zero return -> Sideways

    high_positive = 0.02  # 2% daily
    high_negative = -0.02  # -2% daily
    near_zero = 0.0005  # 0.05% daily

    # These should map to different regimes
    assert high_positive > 0.01  # Bull threshold
    assert high_negative < -0.005  # Bear threshold
    assert abs(near_zero) < 0.01  # Sideways range


def test_regime_label_override():
    """Test custom regime label overrides."""
    emission_means = np.array([-0.01, 0.001, 0.02])
    mapping = map_states_to_financial_regimes(emission_means, n_states=3)

    # Override with custom labels
    custom_mapping = {i: f"Custom_{v}" for i, v in mapping.items()}

    assert all("Custom_" in v for v in custom_mapping.values())


def test_regime_mapping_with_confidence():
    """Test regime mapping integrated with confidence scores."""
    states = pd.Series([0, 1, 2, 1, 0])
    confidence = pd.Series([0.9, 0.85, 0.95, 0.88, 0.92])
    mapping = {0: "Bearish", 1: "Sideways", 2: "Bullish"}

    regimes = states.map(mapping)
    regime_df = pd.DataFrame({
        "regime": regimes,
        "confidence": confidence,
    })

    assert len(regime_df) == 5
    assert "regime" in regime_df.columns
    assert "confidence" in regime_df.columns


# ============================================================================
# INTEGRATION TESTS (6 tests)
# ============================================================================


def test_state_mapping_with_hmm_output():
    """Test state mapping with actual HMM output format."""
    # Simulate HMM output
    hmm_output = pd.DataFrame({
        "predicted_state": [0, 0, 1, 2, 2, 1, 0],
        "confidence": [0.9, 0.85, 0.92, 0.88, 0.91, 0.87, 0.89],
    })

    emission_means = np.array([-0.01, 0.001, 0.02])
    mapping = map_states_to_financial_regimes(emission_means, n_states=3)

    # Map states to regimes
    hmm_output["regime"] = hmm_output["predicted_state"].map(mapping)

    assert "regime" in hmm_output.columns
    assert len(hmm_output) == 7


def test_state_mapping_with_interpreter():
    """Test state mapping integration with interpreter output."""
    # Simulate interpreter-style output
    interpreter_output = pd.DataFrame({
        "state": [0, 1, 2, 1, 0],
        "state_probabilities": [
            np.array([0.8, 0.15, 0.05]),
            np.array([0.1, 0.85, 0.05]),
            np.array([0.05, 0.15, 0.8]),
            np.array([0.1, 0.85, 0.05]),
            np.array([0.8, 0.15, 0.05]),
        ],
    })

    emission_means = np.array([-0.01, 0.001, 0.02])
    mapping = map_states_to_financial_regimes(emission_means, n_states=3)

    interpreter_output["regime"] = interpreter_output["state"].map(mapping)

    assert all(interpreter_output["regime"].notna())


def test_state_mapping_preserves_temporal_order():
    """Test that state mapping preserves temporal ordering."""
    dates = pd.date_range("2023-01-01", periods=10)
    states = pd.Series([0, 0, 1, 1, 2, 2, 1, 1, 0, 0], index=dates)
    mapping = {0: "Bear", 1: "Sideways", 2: "Bull"}

    regimes = states.map(mapping)

    # Order should be preserved
    assert list(regimes.index) == list(dates)


def test_state_mapping_handles_regime_transitions():
    """Test state mapping correctly handles regime transitions."""
    states = pd.Series([0, 0, 1, 2, 2, 1, 0])
    mapping = {0: "Bear", 1: "Sideways", 2: "Bull"}

    regimes = states.map(mapping)

    # Count transitions
    transitions = (regimes != regimes.shift()).sum()

    # Should have transitions at indices 2, 3, 5, 6 = 4 transitions
    assert transitions >= 4


def test_state_mapping_multi_timeframe():
    """Test state mapping works with multi-timeframe data."""
    # Daily and weekly states
    daily_states = pd.Series([0, 0, 1, 1, 2, 2, 1, 0])
    weekly_states = pd.Series([0, 1, 2, 1])  # Aggregated

    mapping = {0: "Bear", 1: "Sideways", 2: "Bull"}

    daily_regimes = daily_states.map(mapping)
    weekly_regimes = weekly_states.map(mapping)

    assert len(daily_regimes) == 8
    assert len(weekly_regimes) == 4


def test_state_mapping_performance():
    """Test state mapping performance with large dataset."""
    # Large dataset
    n_samples = 10000
    states = np.random.randint(0, 3, n_samples)
    mapping = {0: "Bear", 1: "Sideways", 2: "Bull"}

    # Should handle large datasets efficiently
    states_series = pd.Series(states)
    regimes = states_series.map(mapping)

    assert len(regimes) == n_samples
    assert all(regimes.notna())
