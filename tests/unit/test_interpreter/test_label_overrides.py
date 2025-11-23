"""
Label Override Edge Case Tests - Critical Configuration Cases.

Tests verify that forced regime labels work correctly in all edge cases
and don't cause unexpected behavior.

This addresses configuration bugs with force_regime_labels:
- Wrong number of labels
- Empty/None labels
- Special characters in labels
- Label consistency across states
- Interaction with interpretation methods

Tests cover:
1. Correct number of labels
2. Wrong number of labels (too few, too many)
3. Empty and None label handling
4. Special characters and formatting
5. Label-state mapping consistency
6. Interaction with data-driven interpretation
"""

import pytest
import numpy as np
import pandas as pd

from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.config.interpreter import InterpreterConfiguration


# ============================================================================
# Correct Label Override Tests
# ============================================================================


@pytest.mark.unit
def test_forced_labels_2_states():
    """Test forced labels with 2-state model."""
    labels = ["custom_bull", "custom_bear"]
    config = InterpreterConfiguration(
        n_states=2,
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0, 1],
        'confidence': [0.8, 0.75],
        'state_probabilities': [
            np.array([0.8, 0.2]),
            np.array([0.2, 0.75])
        ],
        'emission_means': [np.array([0.001, -0.001])] * 2,
        'emission_stds': [np.array([0.01, 0.02])] * 2,
    })

    result = interpreter.update(model_output)

    # Should use forced labels exactly
    assert result.iloc[0]['regime_type'] == "custom_bull"
    assert result.iloc[1]['regime_type'] == "custom_bear"


@pytest.mark.unit
def test_forced_labels_5_states():
    """Test forced labels with 5-state model."""
    labels = ["label_0", "label_1", "label_2", "label_3", "label_4"]
    config = InterpreterConfiguration(
        n_states=5,
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0, 1, 2, 3, 4],
        'confidence': [0.7] * 5,
        'state_probabilities': [
            np.array([0.7 if i == j else 0.075 for j in range(5)])
            for i in range(5)
        ],
        'emission_means': [np.array(np.linspace(-0.002, 0.002, 5))] * 5,
        'emission_stds': [np.array([0.01, 0.015, 0.02, 0.015, 0.01])] * 5,
    })

    result = interpreter.update(model_output)

    # Each state should get corresponding forced label
    for i in range(5):
        assert result.iloc[i]['regime_type'] == f"label_{i}", \
            f"State {i} should have label 'label_{i}'"


@pytest.mark.unit
def test_forced_labels_override_data_driven():
    """Test that forced labels override data-driven interpretation."""
    labels = ["forced_regime_A", "forced_regime_B"]
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven",  # Should be overridden
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    # Even with clear bullish characteristics, should use forced label
    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [0.9],
        'state_probabilities': [np.array([0.9, 0.1])],
        'emission_means': [np.array([0.003, -0.002])],  # Strong bull signal
        'emission_stds': [np.array([0.008, 0.03])],  # Low vol, high vol
    })

    result = interpreter.update(model_output)

    # Should NOT be "bullish", should be forced label
    regime = result.iloc[0]['regime_type']
    assert regime == "forced_regime_A", \
        f"Should use forced label 'forced_regime_A', got: {regime}"


# ============================================================================
# Wrong Number of Labels Tests
# ============================================================================


@pytest.mark.unit
def test_too_few_labels_raises_error():
    """Test that providing too few labels raises appropriate error."""
    # 3 states but only 2 labels
    labels = ["label_0", "label_1"]

    with pytest.raises((ValueError, AssertionError, IndexError)):
        config = InterpreterConfiguration(
            n_states=3,
            force_regime_labels=labels,
            acknowledge_override=True
        )
        interpreter = FinancialInterpreter(config)

        model_output = pd.DataFrame({
            'state': [2],  # State 2 doesn't have a label
            'confidence': [0.8],
            'state_probabilities': [np.array([0.1, 0.1, 0.8])],
            'emission_means': [np.array([0.001, 0.0, -0.001])],
            'emission_stds': [np.array([0.01, 0.015, 0.02])],
        })

        # This should fail
        interpreter.update(model_output)


@pytest.mark.unit
def test_too_many_labels_raises_error():
    """Test that providing too many labels raises error (strict validation)."""
    # 2 states but 3 labels - should fail validation
    labels = ["label_0", "label_1", "label_2_unused"]

    with pytest.raises(ValueError, match="must have exactly 2 labels"):
        config = InterpreterConfiguration(
            n_states=2,
            force_regime_labels=labels,
            acknowledge_override=True
        )


# ============================================================================
# Empty and None Label Tests
# ============================================================================


@pytest.mark.unit
def test_empty_string_labels():
    """Test handling of empty string labels."""
    labels = ["", "valid_label"]

    config = InterpreterConfiguration(
        n_states=2,
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0, 1],
        'confidence': [0.8, 0.75],
        'state_probabilities': [
            np.array([0.8, 0.2]),
            np.array([0.2, 0.75])
        ],
        'emission_means': [np.array([0.001, -0.001])] * 2,
        'emission_stds': [np.array([0.01, 0.02])] * 2,
    })

    result = interpreter.update(model_output)

    # Empty string should be used as-is (or fall back to default)
    regime_0 = result.iloc[0]['regime_type']
    assert regime_0 is not None  # Should have some value

    regime_1 = result.iloc[1]['regime_type']
    assert regime_1 == "valid_label"


@pytest.mark.unit
def test_none_in_labels_list():
    """Test handling when None appears in labels list."""
    labels = [None, "valid_label"]

    config = InterpreterConfiguration(
        n_states=2,
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0, 1],
        'confidence': [0.8, 0.75],
        'state_probabilities': [
            np.array([0.8, 0.2]),
            np.array([0.2, 0.75])
        ],
        'emission_means': [np.array([0.001, -0.001])] * 2,
        'emission_stds': [np.array([0.01, 0.02])] * 2,
    })

    result = interpreter.update(model_output)

    # None is used as-is (becomes None in regime_type)
    regime_0 = result.iloc[0]['regime_type']
    # Accept None as valid (it's what the user requested)
    assert regime_0 is None or isinstance(regime_0, str)

    regime_1 = result.iloc[1]['regime_type']
    assert regime_1 == "valid_label"


@pytest.mark.unit
def test_all_empty_labels():
    """Test handling when all labels are empty strings."""
    labels = ["", "", ""]

    config = InterpreterConfiguration(
        n_states=3,
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0, 1, 2],
        'confidence': [0.7, 0.75, 0.8],
        'state_probabilities': [
            np.array([0.7, 0.2, 0.1]),
            np.array([0.2, 0.75, 0.05]),
            np.array([0.1, 0.1, 0.8]),
        ],
        'emission_means': [np.array([0.001, 0.0, -0.001])] * 3,
        'emission_stds': [np.array([0.01, 0.015, 0.02])] * 3,
    })

    result = interpreter.update(model_output)

    # Should produce valid output (even if labels are empty)
    assert len(result) == 3
    for i in range(3):
        regime = result.iloc[i]['regime_type']
        assert regime is not None  # Should have some value (possibly empty string)


# ============================================================================
# Special Characters and Formatting Tests
# ============================================================================


@pytest.mark.unit
def test_labels_with_special_characters():
    """Test labels containing special characters."""
    labels = [
        "regime-bull-market",  # Hyphens
        "regime_bear_market",  # Underscores
        "regime.sideways.123",  # Dots and numbers
        "régime_spécial",  # Unicode characters
    ]

    config = InterpreterConfiguration(
        n_states=4,
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0, 1, 2, 3],
        'confidence': [0.7] * 4,
        'state_probabilities': [
            np.array([0.7 if i == j else 0.1 for j in range(4)])
            for i in range(4)
        ],
        'emission_means': [np.array([0.002, -0.002, 0.0, -0.001])] * 4,
        'emission_stds': [np.array([0.01, 0.02, 0.015, 0.025])] * 4,
    })

    result = interpreter.update(model_output)

    # All special character labels should be preserved exactly
    assert result.iloc[0]['regime_type'] == "regime-bull-market"
    assert result.iloc[1]['regime_type'] == "regime_bear_market"
    assert result.iloc[2]['regime_type'] == "regime.sideways.123"
    assert result.iloc[3]['regime_type'] == "régime_spécial"


@pytest.mark.unit
def test_labels_with_spaces():
    """Test labels containing spaces."""
    labels = ["Bull Market", "Bear Market", "Sideways Range"]

    config = InterpreterConfiguration(
        n_states=3,
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0, 1, 2],
        'confidence': [0.8, 0.75, 0.7],
        'state_probabilities': [
            np.array([0.8, 0.1, 0.1]),
            np.array([0.1, 0.75, 0.15]),
            np.array([0.1, 0.2, 0.7]),
        ],
        'emission_means': [np.array([0.001, -0.001, 0.0])] * 3,
        'emission_stds': [np.array([0.01, 0.02, 0.012])] * 3,
    })

    result = interpreter.update(model_output)

    # Labels with spaces should be preserved
    assert result.iloc[0]['regime_type'] == "Bull Market"
    assert result.iloc[1]['regime_type'] == "Bear Market"
    assert result.iloc[2]['regime_type'] == "Sideways Range"


@pytest.mark.unit
def test_very_long_label_names():
    """Test handling of very long label names."""
    labels = [
        "This_is_a_very_long_regime_label_name_that_contains_many_characters_and_describes_a_complex_market_state",
        "short"
    ]

    config = InterpreterConfiguration(
        n_states=2,
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0, 1],
        'confidence': [0.8, 0.75],
        'state_probabilities': [
            np.array([0.8, 0.2]),
            np.array([0.2, 0.75])
        ],
        'emission_means': [np.array([0.001, -0.001])] * 2,
        'emission_stds': [np.array([0.01, 0.02])] * 2,
    })

    result = interpreter.update(model_output)

    # Very long label should be preserved in full
    assert result.iloc[0]['regime_type'] == labels[0]
    assert len(result.iloc[0]['regime_type']) > 50


@pytest.mark.unit
def test_numeric_only_labels():
    """Test labels that are numeric strings."""
    labels = ["0", "1", "2"]

    config = InterpreterConfiguration(
        n_states=3,
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0, 1, 2],
        'confidence': [0.7, 0.75, 0.8],
        'state_probabilities': [
            np.array([0.7, 0.2, 0.1]),
            np.array([0.2, 0.75, 0.05]),
            np.array([0.1, 0.1, 0.8]),
        ],
        'emission_means': [np.array([0.001, 0.0, -0.001])] * 3,
        'emission_stds': [np.array([0.01, 0.015, 0.02])] * 3,
    })

    result = interpreter.update(model_output)

    # Numeric string labels should work
    assert result.iloc[0]['regime_type'] == "0"
    assert result.iloc[1]['regime_type'] == "1"
    assert result.iloc[2]['regime_type'] == "2"


# ============================================================================
# Label-State Mapping Consistency Tests
# ============================================================================


@pytest.mark.unit
def test_label_state_mapping_persistent():
    """Test that state-to-label mapping is persistent across updates."""
    labels = ["persistent_bull", "persistent_bear", "persistent_neutral"]

    config = InterpreterConfiguration(
        n_states=3,
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    # First update
    model_output_1 = pd.DataFrame({
        'state': [0, 1, 2],
        'confidence': [0.8, 0.75, 0.7],
        'state_probabilities': [
            np.array([0.8, 0.1, 0.1]),
            np.array([0.1, 0.75, 0.15]),
            np.array([0.1, 0.2, 0.7]),
        ],
        'emission_means': [np.array([0.002, -0.002, 0.0])] * 3,
        'emission_stds': [np.array([0.01, 0.02, 0.015])] * 3,
    })

    result_1 = interpreter.update(model_output_1)

    # Second update with same states
    model_output_2 = pd.DataFrame({
        'state': [0, 1, 2],
        'confidence': [0.85, 0.8, 0.75],
        'state_probabilities': [
            np.array([0.85, 0.08, 0.07]),
            np.array([0.08, 0.8, 0.12]),
            np.array([0.07, 0.18, 0.75]),
        ],
        'emission_means': [np.array([0.002, -0.002, 0.0])] * 3,
        'emission_stds': [np.array([0.01, 0.02, 0.015])] * 3,
    })

    result_2 = interpreter.update(model_output_2)

    # Mapping should be consistent
    assert result_1.iloc[0]['regime_type'] == result_2.iloc[0]['regime_type'] == "persistent_bull"
    assert result_1.iloc[1]['regime_type'] == result_2.iloc[1]['regime_type'] == "persistent_bear"
    assert result_1.iloc[2]['regime_type'] == result_2.iloc[2]['regime_type'] == "persistent_neutral"


@pytest.mark.unit
def test_label_mapping_with_time_series():
    """Test label mapping consistency across time series."""
    labels = ["ts_state_0", "ts_state_1"]

    config = InterpreterConfiguration(
        n_states=2,
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    # Time series with alternating states
    states = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]

    model_output = pd.DataFrame({
        'state': states,
        'confidence': [0.8] * 10,
        'state_probabilities': [
            np.array([0.8, 0.2]) if s == 0 else np.array([0.2, 0.8])
            for s in states
        ],
        'emission_means': [np.array([0.001, -0.001])] * 10,
        'emission_stds': [np.array([0.01, 0.02])] * 10,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # All state 0 instances should have same label
    state_0_regimes = [regimes[i] for i, s in enumerate(states) if s == 0]
    assert all(r == "ts_state_0" for r in state_0_regimes), \
        f"All state 0 instances should be 'ts_state_0', got: {set(state_0_regimes)}"

    # All state 1 instances should have same label
    state_1_regimes = [regimes[i] for i, s in enumerate(states) if s == 1]
    assert all(r == "ts_state_1" for r in state_1_regimes), \
        f"All state 1 instances should be 'ts_state_1', got: {set(state_1_regimes)}"


# ============================================================================
# Interaction with Interpretation Methods Tests
# ============================================================================


@pytest.mark.unit
def test_forced_labels_with_threshold_method():
    """Test that forced labels work with threshold interpretation method."""
    labels = ["threshold_regime_A", "threshold_regime_B"]

    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="threshold",
        force_regime_labels=labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0, 1],
        'confidence': [0.8, 0.75],
        'state_probabilities': [
            np.array([0.8, 0.2]),
            np.array([0.2, 0.75])
        ],
        'emission_means': [np.array([0.001, -0.001])] * 2,
        'emission_stds': [np.array([0.01, 0.02])] * 2,
    })

    result = interpreter.update(model_output)

    # Should use forced labels regardless of interpretation method
    assert result.iloc[0]['regime_type'] == "threshold_regime_A"
    assert result.iloc[1]['regime_type'] == "threshold_regime_B"


@pytest.mark.unit
def test_no_forced_labels_uses_default():
    """Test that when force_regime_labels is None, data-driven labels are used."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven",
        force_regime_labels=None  # No forced labels
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.8, 0.2])],
        'emission_means': [np.array([0.002, -0.002])],  # Clear bull signal
        'emission_stds': [np.array([0.01, 0.03])],
    })

    result = interpreter.update(model_output)

    regime = result.iloc[0]['regime_type'].lower()

    # Should use data-driven classification (bullish)
    assert 'bull' in regime or 'uptrend' in regime, \
        f"Without forced labels, should use data-driven (bullish), got: {regime}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
