"""
Regime Boundary Condition Tests - Critical Edge Cases.

Tests verify correct behavior at exact threshold values and boundary conditions
where regime classification might be ambiguous or unstable.

This addresses critical interpretation bugs that occur at decision boundaries:
- Exact zero returns (bull vs bear threshold)
- Equal state probabilities (classification tie-breaking)
- Exact volatility thresholds (normal vs volatile regime)
- Confidence boundaries (valid vs invalid signals)
- Numerical precision issues

Tests cover:
1. Zero return boundary (bull/bear/sideways)
2. Equal probability states
3. Exact threshold values
4. Numerical precision edge cases
5. Tie-breaking consistency
"""

import pytest
import numpy as np
import pandas as pd

from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.config.interpreter import InterpreterConfiguration


# ============================================================================
# Zero Return Boundary Tests
# ============================================================================


@pytest.mark.unit
def test_exactly_zero_return():
    """Test regime classification when emission mean is exactly 0.0."""
    config = InterpreterConfiguration(
        n_states=3,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # State 0 has exactly zero return
    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.8, 0.1, 0.1])],
        'emission_means': [np.array([0.0, 0.001, -0.001])],  # Exactly zero
        'emission_stds': [np.array([0.01, 0.01, 0.02])],
    })

    result = interpreter.update(model_output)
    regime_type = result.iloc[0]['regime_type'].lower()

    # Should classify as sideways or neutral, NOT bullish or bearish
    assert 'sideways' in regime_type or 'neutral' in regime_type or 'range' in regime_type, \
        f"Exactly zero return should be sideways/neutral, got: {regime_type}"


@pytest.mark.unit
def test_tiny_positive_return():
    """Test regime classification with very small positive return (near zero).

    Very small returns may be treated as sideways (noise threshold) or bullish
    depending on the interpreter's classification logic.
    """
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # State 0 has tiny positive return (0.0001% daily = 3.65% annual)
    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.8, 0.2])],
        'emission_means': [np.array([0.00001, -0.001])],  # Very small positive
        'emission_stds': [np.array([0.01, 0.02])],
    })

    result = interpreter.update(model_output)
    regime_type = result.iloc[0]['regime_type'].lower()

    # May classify as sideways (noise threshold) or bullish
    is_valid = ('bull' in regime_type or 'uptrend' in regime_type or
                'sideways' in regime_type or 'neutral' in regime_type or 'range' in regime_type)
    assert is_valid, \
        f"Tiny positive return should be bullish or sideways, got: {regime_type}"


@pytest.mark.unit
def test_tiny_negative_return():
    """Test regime classification with very small negative return (near zero).

    Very small returns may be treated as sideways (noise threshold) or bearish
    depending on the interpreter's classification logic.
    """
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # State 0 has tiny negative return
    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.8, 0.2])],
        'emission_means': [np.array([-0.00001, 0.001])],  # Very small negative
        'emission_stds': [np.array([0.01, 0.01])],
    })

    result = interpreter.update(model_output)
    regime_type = result.iloc[0]['regime_type'].lower()

    # May classify as sideways (noise threshold) or bearish
    is_valid = ('bear' in regime_type or 'downtrend' in regime_type or
                'sideways' in regime_type or 'neutral' in regime_type or 'range' in regime_type)
    assert is_valid, \
        f"Tiny negative return should be bearish or sideways, got: {regime_type}"


@pytest.mark.unit
def test_multiple_zero_return_states():
    """Test behavior when multiple states have exactly zero return."""
    config = InterpreterConfiguration(
        n_states=3,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # States 0 and 1 both have zero return, differ only in volatility
    model_output = pd.DataFrame({
        'state': [0, 1, 2],
        'confidence': [0.7, 0.75, 0.8],
        'state_probabilities': [
            np.array([0.7, 0.2, 0.1]),
            np.array([0.2, 0.75, 0.05]),
            np.array([0.1, 0.1, 0.8]),
        ],
        'emission_means': [np.array([0.0, 0.0, 0.002])] * 3,  # Two zeros
        'emission_stds': [np.array([0.008, 0.025, 0.012])] * 3,  # Different vols
    })

    result = interpreter.update(model_output)

    # States 0 and 1 should both be sideways/neutral (or possibly volatile for high vol)
    regime_0 = result.iloc[0]['regime_type'].lower()
    regime_1 = result.iloc[1]['regime_type'].lower()

    # State 0 (low vol, zero return): sideways
    assert 'sideways' in regime_0 or 'neutral' in regime_0 or 'range' in regime_0, \
        f"State 0 (zero return, low vol) should be sideways, got: {regime_0}"

    # State 1 (high vol, zero return): might be volatile or sideways
    is_volatile_or_sideways = ('volatile' in regime_1 or 'sideways' in regime_1 or
                               'neutral' in regime_1 or 'range' in regime_1)
    assert is_volatile_or_sideways, \
        f"State 1 (zero return, high vol) should be volatile or sideways, got: {regime_1}"


# ============================================================================
# Equal Probability Tests
# ============================================================================


@pytest.mark.unit
def test_exactly_equal_probabilities_two_states():
    """Test tie-breaking when two states have exactly equal probability."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Exactly 50/50 probability
    model_output = pd.DataFrame({
        'state': [0],  # State 0 is chosen by max (or first encountered)
        'confidence': [0.5],  # Exactly 50%
        'state_probabilities': [np.array([0.5, 0.5])],  # Perfect tie
        'emission_means': [np.array([0.001, -0.001])],
        'emission_stds': [np.array([0.01, 0.02])],
    })

    result = interpreter.update(model_output)

    # Should still produce valid output (no crash)
    assert 'regime_type' in result.columns
    regime_type = result.iloc[0]['regime_type']
    assert regime_type is not None and len(regime_type) > 0

    # Confidence should reflect the tie
    confidence = result.iloc[0]['regime_confidence']
    assert confidence == 0.5, f"Confidence should be 0.5 for perfect tie, got {confidence}"


@pytest.mark.unit
def test_exactly_equal_probabilities_three_states():
    """Test tie-breaking when three states have exactly equal probability."""
    config = InterpreterConfiguration(
        n_states=3,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Exactly 1/3 probability each
    model_output = pd.DataFrame({
        'state': [0],  # Some state is chosen
        'confidence': [0.333333],  # Low confidence
        'state_probabilities': [np.array([1/3, 1/3, 1/3])],  # Perfect three-way tie
        'emission_means': [np.array([0.001, 0.0, -0.001])],
        'emission_stds': [np.array([0.01, 0.015, 0.02])],
    })

    result = interpreter.update(model_output)

    # Should produce valid output
    assert 'regime_type' in result.columns

    # Confidence should be very low
    confidence = result.iloc[0]['regime_confidence']
    assert confidence < 0.4, f"Confidence should be low for three-way tie, got {confidence}"


@pytest.mark.unit
def test_near_equal_probabilities():
    """Test behavior with nearly equal probabilities (numerical precision)."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Very close but not exactly equal (0.500001 vs 0.499999)
    prob_diff = 1e-6
    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [0.5 + prob_diff/2],
        'state_probabilities': [np.array([0.5 + prob_diff/2, 0.5 - prob_diff/2])],
        'emission_means': [np.array([0.001, -0.001])],
        'emission_stds': [np.array([0.01, 0.02])],
    })

    result = interpreter.update(model_output)

    # Should handle gracefully without numerical issues
    confidence = result.iloc[0]['regime_confidence']
    assert 0.499 <= confidence <= 0.501, \
        f"Confidence should be near 0.5 for near-tie, got {confidence}"


# ============================================================================
# Volatility Threshold Tests
# ============================================================================


@pytest.mark.unit
def test_exact_volatility_threshold():
    """Test behavior at exact volatility thresholds for crisis classification."""
    config = InterpreterConfiguration(
        n_states=3,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Common volatility thresholds:
    # Normal: ~1-2% daily (15-30% annual)
    # Elevated: ~3-4% daily (45-60% annual)
    # Crisis: ~5%+ daily (75%+ annual)

    # Test at exact 5% daily volatility boundary
    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.8, 0.1, 0.1])],
        'emission_means': [np.array([-0.002, 0.0, 0.001])],  # Negative return
        'emission_stds': [np.array([0.05, 0.015, 0.01])],  # Exactly 5% vol
    })

    result = interpreter.update(model_output)
    regime_type = result.iloc[0]['regime_type'].lower()

    # With high vol and negative return, should be crisis or bearish
    is_crisis_or_bear = 'crisis' in regime_type or 'bear' in regime_type or 'volatile' in regime_type
    assert is_crisis_or_bear, \
        f"High vol (5%) + negative return should be crisis/bear/volatile, got: {regime_type}"


@pytest.mark.unit
def test_volatility_just_below_threshold():
    """Test volatility just below crisis threshold."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Just below crisis threshold (4.9% daily vol)
    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.8, 0.2])],
        'emission_means': [np.array([-0.002, 0.001])],
        'emission_stds': [np.array([0.049, 0.01])],  # Just below 5%
    })

    result = interpreter.update(model_output)
    regime_type = result.iloc[0]['regime_type'].lower()

    # Should still be bearish or volatile (high vol, negative return)
    is_bear_or_volatile = 'bear' in regime_type or 'volatile' in regime_type or 'crisis' in regime_type
    assert is_bear_or_volatile, \
        f"High vol (4.9%) + negative return should be bear/volatile, got: {regime_type}"


@pytest.mark.unit
def test_volatility_just_above_threshold():
    """Test volatility just above crisis threshold."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Just above crisis threshold (5.1% daily vol)
    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.8, 0.2])],
        'emission_means': [np.array([-0.002, 0.001])],
        'emission_stds': [np.array([0.051, 0.01])],  # Just above 5%
    })

    result = interpreter.update(model_output)
    regime_type = result.iloc[0]['regime_type'].lower()

    # Should be crisis or volatile (very high vol)
    is_crisis_or_volatile = 'crisis' in regime_type or 'volatile' in regime_type or 'bear' in regime_type
    assert is_crisis_or_volatile, \
        f"Very high vol (5.1%) + negative return should be crisis/volatile, got: {regime_type}"


# ============================================================================
# Numerical Precision Tests
# ============================================================================


@pytest.mark.unit
def test_very_small_numerical_values():
    """Test handling of very small numerical values (near machine epsilon)."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Very small values (near floating point precision limits)
    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.8, 0.2])],
        'emission_means': [np.array([1e-10, -1e-10])],  # Extremely small
        'emission_stds': [np.array([1e-8, 1e-7])],
    })

    result = interpreter.update(model_output)

    # Should handle without numerical errors
    assert 'regime_type' in result.columns
    regime_type = result.iloc[0]['regime_type']
    assert regime_type is not None


@pytest.mark.unit
def test_probability_normalization_edge_case():
    """Test behavior when probabilities don't sum to exactly 1.0 (numerical precision)."""
    config = InterpreterConfiguration(
        n_states=3,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Probabilities that don't sum to exactly 1.0 (0.999999)
    probs = np.array([0.7, 0.2, 0.1])
    probs = probs * 0.999999  # Slight normalization error

    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [probs[0]],
        'state_probabilities': [probs],
        'emission_means': [np.array([0.001, 0.0, -0.001])],
        'emission_stds': [np.array([0.01, 0.015, 0.02])],
    })

    result = interpreter.update(model_output)

    # Should handle gracefully
    confidence = result.iloc[0]['regime_confidence']
    assert 0.6 <= confidence <= 0.8, \
        f"Should handle un-normalized probabilities, got confidence={confidence}"


@pytest.mark.unit
def test_extreme_confidence_values():
    """Test behavior with extreme confidence values (0.0, 1.0)."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Test minimum confidence (0.5 for 2 states)
    model_output_min = pd.DataFrame({
        'state': [0],
        'confidence': [0.5],
        'state_probabilities': [np.array([0.5, 0.5])],
        'emission_means': [np.array([0.001, -0.001])],
        'emission_stds': [np.array([0.01, 0.02])],
    })

    result_min = interpreter.update(model_output_min)
    assert result_min.iloc[0]['regime_confidence'] == 0.5

    # Test maximum confidence (1.0)
    model_output_max = pd.DataFrame({
        'state': [0],
        'confidence': [1.0],
        'state_probabilities': [np.array([1.0, 0.0])],
        'emission_means': [np.array([0.001, -0.001])],
        'emission_stds': [np.array([0.01, 0.02])],
    })

    result_max = interpreter.update(model_output_max)
    assert result_max.iloc[0]['regime_confidence'] == 1.0


# ============================================================================
# Threshold Consistency Tests
# ============================================================================


@pytest.mark.unit
def test_consistent_classification_at_threshold():
    """Test that regime classification is consistent at exact threshold values."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Run same exact input multiple times
    model_output = pd.DataFrame({
        'state': [0],
        'confidence': [0.7],
        'state_probabilities': [np.array([0.7, 0.3])],
        'emission_means': [np.array([0.0, -0.001])],  # Exact zero
        'emission_stds': [np.array([0.02, 0.015])],
    })

    # Run 5 times
    results = [interpreter.update(model_output.copy()) for _ in range(5)]

    # All should give identical regime classification
    regime_types = [r.iloc[0]['regime_type'] for r in results]
    assert all(rt == regime_types[0] for rt in regime_types), \
        f"Same input should give same output, got: {regime_types}"


@pytest.mark.unit
def test_symmetry_in_bull_bear_classification():
    """Test that bull/bear classification is symmetric around zero.

    Uses larger returns to ensure they're above any noise threshold.
    Uses separate interpreter instances to avoid state contamination.
    """
    # Positive return (larger to avoid noise threshold)
    config_bull = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter_bull = FinancialInterpreter(config_bull)

    model_output_bull = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.8, 0.2])],
        'emission_means': [np.array([0.002, -0.002])],  # 0.2% daily = ~50% annual
        'emission_stds': [np.array([0.01, 0.01])],
    })

    result_bull = interpreter_bull.update(model_output_bull)
    regime_bull = result_bull.iloc[0]['regime_type'].lower()

    # Negative return (exactly opposite) - use separate interpreter
    config_bear = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter_bear = FinancialInterpreter(config_bear)

    model_output_bear = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.8, 0.2])],
        'emission_means': [np.array([-0.002, 0.002])],  # -0.2% daily
        'emission_stds': [np.array([0.01, 0.01])],
    })

    result_bear = interpreter_bear.update(model_output_bear)
    regime_bear = result_bear.iloc[0]['regime_type'].lower()

    # Should be opposite classifications (with larger returns)
    assert 'bull' in regime_bull or 'uptrend' in regime_bull, \
        f"Positive return should be bullish, got: {regime_bull}"
    assert 'bear' in regime_bear or 'downtrend' in regime_bear, \
        f"Negative return should be bearish, got: {regime_bear}"


@pytest.mark.unit
def test_infinitesimal_differences_in_probabilities():
    """Test that infinitesimally small probability differences don't cause instability."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Two scenarios with infinitesimal difference
    epsilon = 1e-15  # Machine epsilon order

    model_output_1 = pd.DataFrame({
        'state': [0],
        'confidence': [0.7],
        'state_probabilities': [np.array([0.7, 0.3])],
        'emission_means': [np.array([0.001, -0.001])],
        'emission_stds': [np.array([0.01, 0.02])],
    })

    model_output_2 = pd.DataFrame({
        'state': [0],
        'confidence': [0.7 + epsilon],
        'state_probabilities': [np.array([0.7 + epsilon, 0.3 - epsilon])],
        'emission_means': [np.array([0.001, -0.001])],
        'emission_stds': [np.array([0.01, 0.02])],
    })

    result_1 = interpreter.update(model_output_1)
    result_2 = interpreter.update(model_output_2)

    # Should give same regime (not affected by machine epsilon differences)
    regime_1 = result_1.iloc[0]['regime_type']
    regime_2 = result_2.iloc[0]['regime_type']
    assert regime_1 == regime_2, \
        f"Infinitesimal differences should not affect classification: {regime_1} vs {regime_2}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
