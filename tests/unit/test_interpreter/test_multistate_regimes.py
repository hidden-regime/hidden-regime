"""
Multi-State Regime Classification Tests - Critical Edge Cases.

Tests verify that regime interpretation works correctly with complex multi-state
HMM models (4-7 states), which is a critical gap in current coverage.

This addresses the user's pain point: "continuing problems of regime interpretation"
specifically for models with more than 2-3 states.

Tests cover:
1. 4-state model classification
2. 5-state model with extreme regimes
3. 7-state model granularity
4. Label consistency across varying state counts
5. State ordering and ranking
6. Ambiguous regime resolution
"""

import pytest
import numpy as np
import pandas as pd

from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.config.interpreter import InterpreterConfiguration


# ============================================================================
# 4-State Model Tests
# ============================================================================


@pytest.mark.unit
def test_4_state_model_bull_bear_sideways_crisis():
    """Test 4-state model: bullish, bearish, sideways, crisis."""
    config = InterpreterConfiguration(
        n_states=4,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # State 0: Strong Bull (high return, low vol)
    # State 1: Bear (negative return, medium vol)
    # State 2: Sideways (zero return, low vol)
    # State 3: Crisis (very negative return, very high vol)
    model_output = pd.DataFrame({
        'state': [0, 1, 2, 3],
        'confidence': [0.8, 0.75, 0.7, 0.85],
        'state_probabilities': [
            np.array([0.8, 0.1, 0.05, 0.05]),
            np.array([0.1, 0.75, 0.1, 0.05]),
            np.array([0.1, 0.2, 0.7, 0.0]),
            np.array([0.05, 0.1, 0.0, 0.85]),
        ],
        'emission_means': [
            np.array([0.002, -0.001, 0.0, -0.004]),  # Returns
        ] * 4,
        'emission_stds': [
            np.array([0.01, 0.02, 0.008, 0.06]),  # Volatilities
        ] * 4,
    })

    result = interpreter.update(model_output)

    # Verify each state gets correct classification
    regimes = result['regime_type'].values

    # State 0: Should be bullish
    assert 'bull' in regimes[0].lower() or 'uptrend' in regimes[0].lower(), \
        f"State 0 (high return, low vol) should be bullish, got: {regimes[0]}"

    # State 1: Should be bearish
    assert 'bear' in regimes[1].lower() or 'downtrend' in regimes[1].lower(), \
        f"State 1 (negative return) should be bearish, got: {regimes[1]}"

    # State 2: Should be sideways/neutral
    assert 'sideways' in regimes[2].lower() or 'neutral' in regimes[2].lower() or 'range' in regimes[2].lower(), \
        f"State 2 (zero return, low vol) should be sideways, got: {regimes[2]}"

    # State 3: Should be crisis
    assert 'crisis' in regimes[3].lower() or 'volatile' in regimes[3].lower(), \
        f"State 3 (very negative, very high vol) should be crisis, got: {regimes[3]}"


@pytest.mark.unit
def test_4_state_model_all_bullish_gradations():
    """Test 4-state model with gradations of bullish regimes."""
    config = InterpreterConfiguration(
        n_states=4,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # All bullish, but different strengths
    # State 0: Weak bull (low return, medium vol)
    # State 1: Moderate bull (medium return, low vol)
    # State 2: Strong bull (high return, low vol)
    # State 3: Aggressive bull (very high return, medium vol)
    model_output = pd.DataFrame({
        'state': [0, 1, 2, 3],
        'confidence': [0.7, 0.75, 0.8, 0.75],
        'state_probabilities': [
            np.array([0.7, 0.2, 0.05, 0.05]),
            np.array([0.1, 0.75, 0.1, 0.05]),
            np.array([0.05, 0.1, 0.8, 0.05]),
            np.array([0.05, 0.1, 0.1, 0.75]),
        ],
        'emission_means': [
            np.array([0.0003, 0.0008, 0.0015, 0.0025]),  # Increasing returns
        ] * 4,
        'emission_stds': [
            np.array([0.012, 0.008, 0.007, 0.013]),  # Volatilities
        ] * 4,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # All should be classified as some form of bullish
    for i, regime in enumerate(regimes):
        assert 'bull' in regime.lower() or 'uptrend' in regime.lower(), \
            f"State {i} with positive return should be bullish, got: {regime}"


@pytest.mark.unit
def test_4_state_confidence_with_similar_states():
    """Test that confidence is lower when multiple states have similar characteristics."""
    config = InterpreterConfiguration(
        n_states=4,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # States 0 and 1 are very similar (both mildly bullish)
    # States 2 and 3 are very different
    model_output = pd.DataFrame({
        'state': [0, 1, 2, 3],
        'confidence': [0.6, 0.55, 0.9, 0.85],  # Lower confidence for similar states
        'state_probabilities': [
            np.array([0.6, 0.35, 0.03, 0.02]),  # State 0 vs 1 ambiguous
            np.array([0.4, 0.55, 0.03, 0.02]),  # State 1 vs 0 ambiguous
            np.array([0.05, 0.03, 0.9, 0.02]),  # State 2 clear
            np.array([0.02, 0.03, 0.1, 0.85]),  # State 3 clear
        ],
        'emission_means': [
            np.array([0.0005, 0.0006, -0.003, 0.0]),  # States 0/1 similar
        ] * 4,
        'emission_stds': [
            np.array([0.011, 0.012, 0.045, 0.008]),
        ] * 4,
    })

    result = interpreter.update(model_output)

    # Confidence should reflect state probability
    confidences = result['regime_confidence'].values

    # States 0 and 1 should have lower confidence (ambiguous)
    assert confidences[0] < 0.7, f"Ambiguous state should have low confidence, got {confidences[0]}"
    assert confidences[1] < 0.7, f"Ambiguous state should have low confidence, got {confidences[1]}"

    # States 2 and 3 should have higher confidence (clear)
    assert confidences[2] > 0.8, f"Clear state should have high confidence, got {confidences[2]}"
    assert confidences[3] > 0.8, f"Clear state should have high confidence, got {confidences[3]}"


# ============================================================================
# 5-State Model Tests
# ============================================================================


@pytest.mark.unit
def test_5_state_model_extreme_regimes():
    """Test 5-state model with extreme market conditions."""
    config = InterpreterConfiguration(
        n_states=5,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # State 0: Crash (extreme negative, extreme vol)
    # State 1: Crisis (very negative, very high vol)
    # State 2: Sideways (neutral)
    # State 3: Bull (positive, low vol)
    # State 4: Bubble (extreme positive, increasing vol)
    model_output = pd.DataFrame({
        'state': [0, 1, 2, 3, 4],
        'confidence': [0.9, 0.85, 0.7, 0.8, 0.75],
        'state_probabilities': [
            np.array([0.9, 0.05, 0.02, 0.02, 0.01]),
            np.array([0.1, 0.85, 0.03, 0.01, 0.01]),
            np.array([0.05, 0.05, 0.7, 0.1, 0.1]),
            np.array([0.02, 0.03, 0.1, 0.8, 0.05]),
            np.array([0.01, 0.02, 0.05, 0.17, 0.75]),
        ],
        'emission_means': [
            np.array([-0.008, -0.004, 0.0, 0.0012, 0.0035]),  # Returns
        ] * 5,
        'emission_stds': [
            np.array([0.09, 0.055, 0.01, 0.009, 0.025]),  # Volatilities
        ] * 5,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # State 0: Extreme crash
    assert 'crisis' in regimes[0].lower() or 'crash' in regimes[0].lower() or 'bear' in regimes[0].lower(), \
        f"State 0 (extreme negative, extreme vol) should be crisis/crash, got: {regimes[0]}"

    # State 1: Crisis
    assert 'crisis' in regimes[1].lower() or 'volatile' in regimes[1].lower() or 'bear' in regimes[1].lower(), \
        f"State 1 (very negative, very high vol) should be crisis, got: {regimes[1]}"

    # State 2: Sideways
    assert 'sideways' in regimes[2].lower() or 'neutral' in regimes[2].lower() or 'range' in regimes[2].lower(), \
        f"State 2 (neutral) should be sideways, got: {regimes[2]}"

    # State 3: Bull
    assert 'bull' in regimes[3].lower() or 'uptrend' in regimes[3].lower(), \
        f"State 3 (positive, low vol) should be bullish, got: {regimes[3]}"

    # State 4: Bubble (might be classified as volatile bull or just bull)
    # Accept either bullish or volatile classification
    is_bull_or_volatile = ('bull' in regimes[4].lower() or 'uptrend' in regimes[4].lower() or
                          'volatile' in regimes[4].lower())
    assert is_bull_or_volatile, \
        f"State 4 (extreme positive) should be bullish or volatile, got: {regimes[4]}"


@pytest.mark.unit
def test_5_state_model_state_ordering():
    """Test that state ordering doesn't affect regime classification."""
    config = InterpreterConfiguration(
        n_states=5,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Same regimes but in different order
    # Verify that regime classification is based on characteristics, not state index
    emissions_original = np.array([-0.003, -0.001, 0.0, 0.001, 0.003])
    emissions_shuffled = np.array([0.001, -0.003, 0.003, 0.0, -0.001])

    model_output_shuffled = pd.DataFrame({
        'state': [0, 1, 2, 3, 4],
        'confidence': [0.8] * 5,
        'state_probabilities': [
            np.array([0.8, 0.05, 0.05, 0.05, 0.05]),
            np.array([0.05, 0.8, 0.05, 0.05, 0.05]),
            np.array([0.05, 0.05, 0.8, 0.05, 0.05]),
            np.array([0.05, 0.05, 0.05, 0.8, 0.05]),
            np.array([0.05, 0.05, 0.05, 0.05, 0.8]),
        ],
        'emission_means': [emissions_shuffled] * 5,
        'emission_stds': [np.array([0.02, 0.03, 0.025, 0.01, 0.015])] * 5,
    })

    result = interpreter.update(model_output_shuffled)

    regimes = result['regime_type'].values

    # State 0 (return=0.001): Should be bullish
    assert 'bull' in regimes[0].lower() or 'uptrend' in regimes[0].lower(), \
        f"Positive return state should be bullish regardless of index, got: {regimes[0]}"

    # State 1 (return=-0.003, vol=0.03): Should be bearish or crisis (high vol)
    is_bearish_or_crisis = ('bear' in regimes[1].lower() or 'downtrend' in regimes[1].lower() or
                            'crisis' in regimes[1].lower())
    assert is_bearish_or_crisis, \
        f"Negative return state should be bearish or crisis regardless of index, got: {regimes[1]}"

    # State 2 (return=0.003): Should be bullish
    assert 'bull' in regimes[2].lower() or 'uptrend' in regimes[2].lower(), \
        f"Positive return state should be bullish regardless of index, got: {regimes[2]}"

    # State 3 (return=0.0): Should be sideways/neutral
    assert 'sideways' in regimes[3].lower() or 'neutral' in regimes[3].lower() or 'range' in regimes[3].lower(), \
        f"Zero return state should be sideways regardless of index, got: {regimes[3]}"


# ============================================================================
# 7-State Model Tests
# ============================================================================


@pytest.mark.unit
def test_7_state_model_granular_classification():
    """Test 7-state model with very granular regime classification."""
    config = InterpreterConfiguration(
        n_states=7,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Very granular model:
    # State 0: Strong Bear
    # State 1: Weak Bear
    # State 2: Sideways Low Vol
    # State 3: Sideways High Vol
    # State 4: Weak Bull
    # State 5: Strong Bull
    # State 6: Crisis
    model_output = pd.DataFrame({
        'state': [0, 1, 2, 3, 4, 5, 6],
        'confidence': [0.8, 0.75, 0.7, 0.65, 0.75, 0.8, 0.85],
        'state_probabilities': [
            np.array([0.8, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02]),
            np.array([0.1, 0.75, 0.05, 0.03, 0.03, 0.02, 0.02]),
            np.array([0.02, 0.05, 0.7, 0.13, 0.05, 0.03, 0.02]),
            np.array([0.02, 0.03, 0.2, 0.65, 0.05, 0.03, 0.02]),
            np.array([0.02, 0.03, 0.05, 0.05, 0.75, 0.08, 0.02]),
            np.array([0.02, 0.02, 0.03, 0.03, 0.1, 0.8, 0.0]),
            np.array([0.05, 0.05, 0.02, 0.02, 0.01, 0.0, 0.85]),
        ],
        'emission_means': [
            np.array([-0.0025, -0.0008, 0.0, 0.0, 0.0008, 0.0025, -0.004]),
        ] * 7,
        'emission_stds': [
            np.array([0.02, 0.015, 0.008, 0.022, 0.012, 0.009, 0.055]),
        ] * 7,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # States 0 and 1: Bearish
    assert 'bear' in regimes[0].lower() or 'downtrend' in regimes[0].lower(), \
        f"State 0 (strong bear) should be bearish, got: {regimes[0]}"
    assert 'bear' in regimes[1].lower() or 'downtrend' in regimes[1].lower(), \
        f"State 1 (weak bear) should be bearish, got: {regimes[1]}"

    # States 2 and 3: Sideways (might be classified differently based on vol)
    is_sideways_2 = ('sideways' in regimes[2].lower() or 'neutral' in regimes[2].lower() or
                     'range' in regimes[2].lower())
    assert is_sideways_2, f"State 2 (sideways low vol) should be sideways, got: {regimes[2]}"

    # State 3 might be volatile or sideways
    is_sideways_or_volatile_3 = ('sideways' in regimes[3].lower() or 'neutral' in regimes[3].lower() or
                                  'volatile' in regimes[3].lower() or 'range' in regimes[3].lower())
    assert is_sideways_or_volatile_3, \
        f"State 3 (sideways high vol) should be sideways or volatile, got: {regimes[3]}"

    # States 4 and 5: Bullish
    assert 'bull' in regimes[4].lower() or 'uptrend' in regimes[4].lower(), \
        f"State 4 (weak bull) should be bullish, got: {regimes[4]}"
    assert 'bull' in regimes[5].lower() or 'uptrend' in regimes[5].lower(), \
        f"State 5 (strong bull) should be bullish, got: {regimes[5]}"

    # State 6: Crisis
    assert 'crisis' in regimes[6].lower() or 'volatile' in regimes[6].lower(), \
        f"State 6 (crisis) should be crisis/volatile, got: {regimes[6]}"


@pytest.mark.unit
def test_7_state_model_label_consistency():
    """Test that regime labels are consistently applied across all 7 states."""
    config = InterpreterConfiguration(
        n_states=7,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Create a time series where we visit each state multiple times
    # Each state should get the same label every time
    states = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
    confidences = [0.8] * 14

    model_output = pd.DataFrame({
        'state': states,
        'confidence': confidences,
        'state_probabilities': [
            np.array([0.8, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02]) if s == 0 else
            np.array([0.03, 0.8, 0.03, 0.03, 0.03, 0.03, 0.02]) if s == 1 else
            np.array([0.03, 0.03, 0.8, 0.03, 0.03, 0.03, 0.02]) if s == 2 else
            np.array([0.03, 0.03, 0.03, 0.8, 0.03, 0.03, 0.02]) if s == 3 else
            np.array([0.03, 0.03, 0.03, 0.03, 0.8, 0.03, 0.02]) if s == 4 else
            np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.8, 0.02]) if s == 5 else
            np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88])
            for s in states
        ],
        'emission_means': [
            np.array([-0.002, -0.0008, 0.0, 0.0001, 0.0008, 0.002, -0.003])
        ] * 14,
        'emission_stds': [
            np.array([0.02, 0.015, 0.01, 0.018, 0.012, 0.01, 0.045])
        ] * 14,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # Each state should have consistent label
    for state_idx in range(7):
        # Find all occurrences of this state
        state_regimes = [regimes[i] for i, s in enumerate(states) if s == state_idx]

        # All should be the same
        assert all(r == state_regimes[0] for r in state_regimes), \
            f"State {state_idx} has inconsistent labels: {state_regimes}"


# ============================================================================
# Label Consistency Tests (Across Different State Counts)
# ============================================================================


@pytest.mark.unit
def test_consistent_classification_across_state_counts():
    """Test that similar market conditions get similar labels regardless of n_states."""
    # Create similar bullish conditions in 2-state, 3-state, 4-state models
    # Should all classify as bullish

    bullish_emission_mean = 0.0015
    bullish_emission_std = 0.01

    for n_states in [2, 3, 4, 5]:
        config = InterpreterConfiguration(
            n_states=n_states,
            interpretation_method="data_driven"
        )
        interpreter = FinancialInterpreter(config)

        # Create emission parameters where state 0 is always bullish
        emission_means = np.zeros(n_states)
        emission_means[0] = bullish_emission_mean
        # Make other states different
        for i in range(1, n_states):
            emission_means[i] = -0.001 * i

        emission_stds = np.full(n_states, 0.015)
        emission_stds[0] = bullish_emission_std

        model_output = pd.DataFrame({
            'state': [0],
            'confidence': [0.8],
            'state_probabilities': [np.array([0.8] + [0.2/(n_states-1)]*(n_states-1))],
            'emission_means': [emission_means],
            'emission_stds': [emission_stds],
        })

        result = interpreter.update(model_output)
        regime = result.iloc[0]['regime_type'].lower()

        assert 'bull' in regime or 'uptrend' in regime, \
            f"Bullish conditions should be classified as bullish in {n_states}-state model, got: {regime}"


@pytest.mark.unit
def test_forced_labels_work_with_7_states():
    """Test that forced regime labels work correctly with 7-state model."""
    custom_labels = [
        "custom_crisis",
        "custom_strong_bear",
        "custom_weak_bear",
        "custom_neutral",
        "custom_weak_bull",
        "custom_strong_bull",
        "custom_bubble"
    ]

    config = InterpreterConfiguration(
        n_states=7,
        force_regime_labels=custom_labels,
        acknowledge_override=True
    )
    interpreter = FinancialInterpreter(config)

    # Test state 3 (should get "custom_neutral")
    model_output = pd.DataFrame({
        'state': [3],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.02, 0.02, 0.02, 0.8, 0.05, 0.05, 0.04])],
        'emission_means': [np.array([-0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003])],
        'emission_stds': [np.array([0.03, 0.02, 0.015, 0.01, 0.012, 0.015, 0.025])],
    })

    result = interpreter.update(model_output)
    regime_type = result.iloc[0]['regime_type']

    assert regime_type == "custom_neutral", \
        f"Should use forced label 'custom_neutral' for state 3, got: {regime_type}"


# ============================================================================
# Edge Cases for Multi-State Models
# ============================================================================


@pytest.mark.unit
def test_all_states_have_labels():
    """Test that every state in multi-state model gets a valid regime label."""
    for n_states in [4, 5, 6, 7]:
        config = InterpreterConfiguration(
            n_states=n_states,
            interpretation_method="data_driven"
        )
        interpreter = FinancialInterpreter(config)

        # Create model output for all states
        emission_means = np.linspace(-0.003, 0.003, n_states)
        emission_stds = np.linspace(0.01, 0.03, n_states)

        model_output = pd.DataFrame({
            'state': list(range(n_states)),
            'confidence': [0.7 + i*0.02 for i in range(n_states)],
            'state_probabilities': [
                np.array([0.7 if j == i else 0.3/(n_states-1) for j in range(n_states)])
                for i in range(n_states)
            ],
            'emission_means': [emission_means] * n_states,
            'emission_stds': [emission_stds] * n_states,
        })

        result = interpreter.update(model_output)

        # Every state should have a regime type
        assert len(result) == n_states, f"Should have {n_states} results"

        for i in range(n_states):
            regime_type = result.iloc[i]['regime_type']
            assert regime_type is not None, f"State {i} in {n_states}-state model has None regime"
            assert isinstance(regime_type, str), f"State {i} regime should be string"
            assert len(regime_type) > 0, f"State {i} regime should not be empty"


@pytest.mark.unit
def test_multistate_with_missing_emission_params():
    """Test graceful handling when emission params missing in multi-state model."""
    config = InterpreterConfiguration(
        n_states=5,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Model output without emission parameters
    model_output = pd.DataFrame({
        'state': [0, 1, 2, 3, 4],
        'confidence': [0.8, 0.75, 0.7, 0.75, 0.8],
        'state_probabilities': [
            np.array([0.8, 0.05, 0.05, 0.05, 0.05]),
            np.array([0.05, 0.75, 0.1, 0.05, 0.05]),
            np.array([0.05, 0.1, 0.7, 0.1, 0.05]),
            np.array([0.05, 0.05, 0.1, 0.75, 0.05]),
            np.array([0.05, 0.05, 0.05, 0.05, 0.8]),
        ],
        # Missing emission_means and emission_stds
    })

    result = interpreter.update(model_output)

    # Should still produce valid output with default labels
    assert len(result) == 5
    for i in range(5):
        regime_type = result.iloc[i]['regime_type']
        assert regime_type is not None and len(regime_type) > 0, \
            f"State {i} should have valid default label"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
