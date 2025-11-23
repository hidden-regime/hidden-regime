"""
Regime Classification Tests - Domain Logic.

Tests verify that regime interpretation correctly maps HMM states to
meaningful financial regimes (bull, bear, volatile, sideways, crisis).

This is critical for business value delivery - users need accurate
regime labels to make trading decisions.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.config.interpreter import InterpreterConfiguration


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_config():
    """Basic interpreter configuration."""
    return InterpreterConfiguration(
        n_states=3,
        interpretation_method="data_driven"
    )


@pytest.fixture
def model_output_bullish():
    """Model output clearly indicating bullish regime."""
    return pd.DataFrame({
        'state': [0],  # REQUIRED: Current state index
        'confidence': [0.8],  # REQUIRED: Max probability
        'state_probabilities': [np.array([0.8, 0.15, 0.05])],
        'emission_means': [np.array([0.001, 0.0, -0.001])],  # Daily returns: +, neutral, -
        'emission_stds': [np.array([0.01, 0.015, 0.02])],    # Low, med, high vol
    })


@pytest.fixture
def model_output_bearish():
    """Model output clearly indicating bearish regime."""
    return pd.DataFrame({
        'state': [2],  # REQUIRED: Current state index
        'confidence': [0.7],  # REQUIRED: Max probability
        'state_probabilities': [np.array([0.1, 0.2, 0.7])],
        'emission_means': [np.array([0.0005, 0.0, -0.002])],  # Positive, neutral, very negative
        'emission_stds': [np.array([0.01, 0.015, 0.025])],
    })


@pytest.fixture
def model_output_crisis():
    """Model output indicating crisis regime (high volatility)."""
    return pd.DataFrame({
        'state': [1],  # REQUIRED: Current state index
        'confidence': [0.7],  # REQUIRED: Max probability
        'state_probabilities': [np.array([0.2, 0.7, 0.1])],
        'emission_means': [np.array([0.001, -0.003, 0.0])],
        'emission_stds': [np.array([0.015, 0.05, 0.01])],  # State 1 has very high vol
    })


# ============================================================================
# Regime Label Assignment Tests
# ============================================================================


@pytest.mark.unit
def test_bullish_regime_identification(basic_config, model_output_bullish):
    """Test that positive returns + low vol => bullish regime."""
    interpreter = FinancialInterpreter(basic_config)

    result = interpreter.update(model_output_bullish)

    # Should identify bullish regime
    regime_type = result.iloc[0]['regime_type'].lower()
    assert 'bullish' in regime_type or 'bull' in regime_type or 'uptrend' in regime_type, \
        f"Should identify bullish regime, got: {regime_type}"


@pytest.mark.unit
def test_bearish_regime_identification(basic_config, model_output_bearish):
    """Test that negative returns => bearish regime."""
    interpreter = FinancialInterpreter(basic_config)

    result = interpreter.update(model_output_bearish)

    regime_type = result.iloc[0]['regime_type'].lower()
    assert 'bearish' in regime_type or 'bear' in regime_type or 'downtrend' in regime_type, \
        f"Should identify bearish regime, got: {regime_type}"


@pytest.mark.unit
def test_crisis_regime_identification(basic_config, model_output_crisis):
    """Test that high volatility => crisis or volatile regime."""
    interpreter = FinancialInterpreter(basic_config)

    result = interpreter.update(model_output_crisis)

    regime_type = result.iloc[0]['regime_type'].lower()
    # Should identify high volatility regime
    assert 'crisis' in regime_type or 'volatile' in regime_type or 'high_vol' in regime_type, \
        f"Should identify volatile/crisis regime, got: {regime_type}"


@pytest.mark.unit
def test_regime_confidence_scoring(basic_config, model_output_bullish):
    """Test that regime confidence is calculated correctly.

    High probability in single state => high confidence
    Spread across states => low confidence
    """
    interpreter = FinancialInterpreter(basic_config)

    result = interpreter.update(model_output_bullish)

    # State probability is 0.8 => high confidence
    confidence = result.iloc[0].get('regime_confidence', 0.0)

    assert confidence > 0.6, f"Should have high confidence with p=0.8, got {confidence}"
    assert confidence <= 1.0, "Confidence should not exceed 1.0"


@pytest.mark.unit
def test_regime_transition_detection():
    """Test detection of regime transitions."""
    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)

    # Create time series with regime change
    states = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # Transition at t=5
    confidences = [0.9] * 5 + [0.9] * 5
    model_output = pd.DataFrame({
        'state': states,  # REQUIRED: Current state index
        'confidence': confidences,  # REQUIRED: Max probability
        'state_probabilities': [np.array([0.9, 0.1])] * 5 + [np.array([0.1, 0.9])] * 5,
        'emission_means': [np.array([0.001, -0.001])] * 10,
        'emission_stds': [np.array([0.01, 0.02])] * 10,
    })

    result = interpreter.update(model_output)

    # Check for transition flag or different regimes
    regimes = result['regime_type'].values
    assert not all(r == regimes[0] for r in regimes), \
        "Should detect regime transition"


@pytest.mark.unit
def test_regime_stability_duration():
    """Test calculation of regime stability/duration."""
    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)

    # All in same regime
    model_output = pd.DataFrame({
        'state': [0] * 20,  # REQUIRED: All in state 0
        'confidence': [0.95] * 20,  # REQUIRED: High confidence
        'state_probabilities': [np.array([0.95, 0.05])] * 20,
        'emission_means': [np.array([0.001, -0.001])] * 20,
        'emission_stds': [np.array([0.01, 0.02])] * 20,
    })

    result = interpreter.update(model_output)

    # Should indicate stable regime
    regimes = result['regime_type'].values
    assert all(r == regimes[0] for r in regimes), \
        "All time steps should have same regime"


# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.unit
def test_low_confidence_regime_detection():
    """Test that low confidence regimes are detected with appropriate metrics."""
    config = InterpreterConfiguration(
        n_states=2
    )
    interpreter = FinancialInterpreter(config)

    # Low confidence scenario
    model_output = pd.DataFrame({
        'state': [0],  # REQUIRED: Current state index
        'confidence': [0.55],  # REQUIRED: Low confidence
        'state_probabilities': [np.array([0.55, 0.45])],  # Low confidence
        'emission_means': [np.array([0.001, -0.001])],
        'emission_stds': [np.array([0.01, 0.02])],
    })

    result = interpreter.update(model_output)

    # Should still produce valid output
    assert 'regime_type' in result.columns

    # Confidence should reflect the uncertainty
    if 'regime_confidence' in result.columns or 'regime_strength' in result.columns:
        confidence = result.iloc[0].get('regime_confidence', result.iloc[0].get('regime_strength', 0.0))
        assert confidence < 0.7, "Low state probability should yield low regime confidence"


@pytest.mark.unit
def test_force_regime_labels_override():
    """Test that manual regime labels can override automatic classification."""
    config = InterpreterConfiguration(
        n_states=2,
        force_regime_labels=["custom_bull", "custom_bear"],
        acknowledge_override=True  # REQUIRED when using force_regime_labels
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0],  # REQUIRED: Current state index
        'confidence': [0.8],  # REQUIRED: Max probability
        'state_probabilities': [np.array([0.8, 0.2])],
        'emission_means': [np.array([0.001, -0.001])],
        'emission_stds': [np.array([0.01, 0.02])],
    })

    result = interpreter.update(model_output)

    regime_type = result.iloc[0]['regime_type']

    # Should use forced label
    assert regime_type == "custom_bull", \
        f"Should use forced label 'custom_bull', got {regime_type}"


@pytest.mark.unit
def test_interpretation_method_threshold():
    """Test threshold-based interpretation method."""
    config = InterpreterConfiguration(
        n_states=3,
        interpretation_method="threshold"
    )
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0],  # REQUIRED: Current state index
        'confidence': [0.7],  # REQUIRED: Max probability
        'state_probabilities': [np.array([0.7, 0.2, 0.1])],
        'emission_means': [np.array([0.002, 0.0, -0.002])],
        'emission_stds': [np.array([0.01, 0.015, 0.02])],
    })

    result = interpreter.update(model_output)

    # Should produce valid regime type
    regime_type = result.iloc[0]['regime_type']
    assert isinstance(regime_type, str), "Regime type should be string"
    assert len(regime_type) > 0, "Regime type should not be empty"


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.unit
def test_mixed_regime_handling():
    """Test handling of ambiguous/mixed regimes.

    When state probabilities are similar, should handle gracefully.
    """
    config = InterpreterConfiguration(n_states=3)
    interpreter = FinancialInterpreter(config)

    # Very uncertain state distribution
    model_output = pd.DataFrame({
        'state': [0],  # REQUIRED: Current state index
        'confidence': [0.35],  # REQUIRED: Low confidence (max prob)
        'state_probabilities': [np.array([0.35, 0.33, 0.32])],  # Very mixed
        'emission_means': [np.array([0.001, 0.0, -0.001])],
        'emission_stds': [np.array([0.015, 0.015, 0.015])],
    })

    result = interpreter.update(model_output)

    # Should still produce valid output
    assert 'regime_type' in result.columns

    # Confidence should be low (check both possible column names)
    if 'regime_confidence' in result.columns:
        confidence = result.iloc[0]['regime_confidence']
        assert confidence < 0.5, f"Mixed regime should have low confidence, got {confidence}"
    elif 'regime_strength' in result.columns:
        strength = result.iloc[0]['regime_strength']
        assert strength < 0.5, f"Mixed regime should have low strength, got {strength}"


@pytest.mark.unit
def test_missing_emission_params_handling():
    """Test graceful handling when emission params are missing."""
    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)

    # Model output without emission parameters
    model_output = pd.DataFrame({
        'state': [0],  # REQUIRED: Current state index
        'confidence': [0.8],  # REQUIRED: Max probability
        'state_probabilities': [np.array([0.8, 0.2])],
        # Missing emission_means and emission_stds
    })

    result = interpreter.update(model_output)

    # Should use default labels
    assert 'regime_type' in result.columns
    regime_type = result.iloc[0]['regime_type']
    assert regime_type is not None and len(regime_type) > 0


@pytest.mark.unit
def test_two_state_minimum():
    """Test that interpreter requires minimum 2 states for meaningful interpretation."""
    config = InterpreterConfiguration(n_states=2)  # Minimum 2 states for regime switching
    interpreter = FinancialInterpreter(config)

    model_output = pd.DataFrame({
        'state': [0],  # REQUIRED: Current state index
        'confidence': [1.0],  # REQUIRED: Perfect confidence
        'state_probabilities': [np.array([1.0, 0.0])],
        'emission_means': [np.array([0.0005, -0.001])],
        'emission_stds': [np.array([0.015, 0.02])],
    })

    result = interpreter.update(model_output)

    # Should handle gracefully
    assert 'regime_type' in result.columns
    confidence = result.iloc[0].get('regime_confidence', result.iloc[0].get('regime_strength', 0.0))
    assert confidence == 1.0 or confidence > 0.9, \
        "Perfect state probability should yield high confidence"


@pytest.mark.unit
def test_extreme_volatility_classification():
    """Test classification of extreme volatility scenarios."""
    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)

    # Extreme volatility (market crash scenario)
    model_output = pd.DataFrame({
        'state': [0],  # REQUIRED: Current state index
        'confidence': [0.9],  # REQUIRED: High confidence
        'state_probabilities': [np.array([0.9, 0.1])],
        'emission_means': [np.array([-0.005, 0.001])],  # -5% daily return!
        'emission_stds': [np.array([0.08, 0.01])],      # 8% daily vol!
    })

    result = interpreter.update(model_output)

    regime_type = result.iloc[0]['regime_type'].lower()

    # Should recognize extreme condition
    assert 'crisis' in regime_type or 'bear' in regime_type or 'crash' in regime_type or 'volatile' in regime_type, \
        f"Should recognize crisis/crash/volatile regime, got: {regime_type}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
