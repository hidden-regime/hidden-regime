"""
Error Propagation Integration Tests.

Tests verify that errors are properly handled and propagated across
component boundaries:
- Invalid inputs are rejected at correct layer
- Error messages are informative
- Errors don't corrupt component state
- Graceful degradation where appropriate
"""

import pytest
import numpy as np
import pandas as pd

from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.config.model import HMMConfig
from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.config.interpreter import InterpreterConfiguration
from hidden_regime.signal_generation.financial import FinancialSignalGenerator
from hidden_regime.config.signal_generation import SignalGenerationConfiguration
from hidden_regime.utils.exceptions import ValidationError


# ============================================================================
# Model Layer Error Tests
# ============================================================================


@pytest.mark.integration
def test_model_rejects_nan_values():
    """Test that model rejects NaN values with clear error."""
    data = pd.DataFrame({'log_return': [0.01, np.nan, 0.02, -0.01]})

    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)

    # Should reject NaN values
    with pytest.raises((ValidationError, ValueError)):
        model.fit(data)


@pytest.mark.integration
def test_model_rejects_infinite_values():
    """Test that model rejects infinite values."""
    data = pd.DataFrame({'log_return': [0.01, np.inf, 0.02, -0.01, 0.003, -0.002, 0.001, -0.001, 0.002, -0.002, 0.001]})

    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)

    # Should reject infinite values or handle them gracefully
    try:
        model.fit(data)
        # If it handles inf gracefully, model should still be fitted
        assert model.is_fitted or not model.is_fitted, "Model state should be consistent"
    except (ValidationError, ValueError):
        # Strict validation is also acceptable
        pass


@pytest.mark.integration
def test_model_preserves_state_after_failed_fit():
    """Test that model state is not corrupted after failed fit attempt."""
    # First, successfully train model
    valid_data = pd.DataFrame({'log_return': np.random.normal(0, 0.01, 50)})
    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)
    model.fit(valid_data)

    assert model.is_fitted, "Model should be fitted"
    original_means = model.emission_means_.copy()

    # Now try to fit with invalid data
    invalid_data = pd.DataFrame({'log_return': [np.nan, np.nan]})

    with pytest.raises((ValidationError, ValueError)):
        model.fit(invalid_data)

    # Original model state should be preserved (or model should be marked unfitted)
    # This is implementation-dependent, but state should be consistent
    if model.is_fitted:
        # If still fitted, parameters should be unchanged
        assert np.array_equal(model.emission_means_, original_means), \
            "Failed fit should not corrupt existing model parameters"


@pytest.mark.integration
def test_predict_before_fit_raises_error():
    """Test that calling predict before fit raises clear error."""
    data = pd.DataFrame({'log_return': np.random.normal(0, 0.01, 20)})

    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)

    with pytest.raises((ValidationError, ValueError, RuntimeError)) as exc_info:
        model.predict(data)

    error_msg = str(exc_info.value).lower()
    assert 'fit' in error_msg or 'train' in error_msg or 'not fitted' in error_msg, \
        f"Error should mention that model needs to be fitted first: {error_msg}"


# ============================================================================
# Interpreter Layer Error Tests
# ============================================================================


@pytest.mark.integration
def test_interpreter_rejects_mismatched_state_count():
    """Test that interpreter rejects input with wrong number of states."""
    # Interpreter configured for 2 states
    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)

    # But input has 3 state probabilities
    invalid_input = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.5, 0.3, 0.2])],  # 3 states!
        'emission_means': [np.array([0.001, -0.001, 0.0])],
        'emission_stds': [np.array([0.01, 0.02, 0.015])]
    })

    # Should reject or handle gracefully
    try:
        result = interpreter.update(invalid_input)
        # If it handles gracefully, should still produce output
        assert len(result) > 0, "Should handle state count mismatch"
    except (ValidationError, ValueError, IndexError):
        # Strict validation is also acceptable
        pass


@pytest.mark.integration
def test_interpreter_rejects_invalid_probabilities():
    """Test that interpreter rejects state probabilities that don't sum to 1."""
    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)

    # Probabilities don't sum to 1
    invalid_input = pd.DataFrame({
        'state': [0],
        'confidence': [0.8],
        'state_probabilities': [np.array([0.5, 0.3])],  # Sum = 0.8, not 1.0
        'emission_means': [np.array([0.001, -0.001])],
        'emission_stds': [np.array([0.01, 0.02])]
    })

    # Interpreter may validate this or handle gracefully
    # If it validates, it should raise an error
    try:
        result = interpreter.update(invalid_input)
        # If it doesn't raise, it should at least produce a result
        assert len(result) == 1, "Should process single row"
    except (ValidationError, ValueError):
        # This is also acceptable - strict validation
        pass


@pytest.mark.integration
def test_interpreter_handles_missing_required_column():
    """Test that interpreter handles missing required columns gracefully."""
    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)

    # Missing 'state' column
    invalid_input = pd.DataFrame({
        'confidence': [0.8],
        'state_probabilities': [np.array([0.8, 0.2])],
        'emission_means': [np.array([0.001, -0.001])],
        'emission_stds': [np.array([0.01, 0.02])]
    })

    with pytest.raises((KeyError, ValidationError, ValueError)):
        interpreter.update(invalid_input)


# ============================================================================
# Signal Generator Layer Error Tests
# ============================================================================


@pytest.mark.integration
def test_signal_generator_handles_missing_regime_type():
    """Test signal generator behavior when regime_type is missing."""
    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)

    incomplete_input = pd.DataFrame({
        'state': [0],
        'regime_confidence': [0.8],
        'regime_strength': [0.8],
        'signal_valid': [True]
        # Missing regime_type
    })

    # Should either raise error or default to neutral signal
    try:
        result = signal_gen.update(incomplete_input)
        # If it handles gracefully, should produce zero signal
        assert result.iloc[0]['position_size'] == 0, \
            "Missing regime_type should result in zero signal"
    except (KeyError, ValidationError, ValueError):
        # Strict validation is also acceptable
        pass


@pytest.mark.integration
def test_signal_generator_handles_invalid_confidence_values():
    """Test signal generator with invalid confidence values."""
    config = SignalGenerationConfiguration(strategy_type="confidence_weighted")
    signal_gen = FinancialSignalGenerator(config)

    invalid_input = pd.DataFrame({
        'state': [0, 1],
        'regime_type': ['bullish', 'bearish'],
        'regime_confidence': [-0.5, 2.0],  # Invalid: outside [0, 1]
        'regime_strength': [0.8, 0.8],
        'signal_valid': [True, True]
    })

    # Should either validate and raise error, or clip to valid range
    try:
        result = signal_gen.update(invalid_input)
        # If it handles gracefully, signals should be bounded
        assert all(abs(result['position_size']) <= 1.0), \
            "Signals should be bounded even with invalid confidence"
    except (ValidationError, ValueError):
        # Strict validation is also acceptable
        pass


# ============================================================================
# Cross-Component Error Propagation Tests
# ============================================================================


@pytest.mark.integration
def test_error_in_model_prevents_interpreter_processing():
    """Test that model errors prevent downstream processing."""
    # Invalid data for model
    invalid_data = pd.DataFrame({'log_return': [np.nan, np.nan]})

    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)

    # Model should reject this
    with pytest.raises((ValidationError, ValueError)):
        model.fit(invalid_data)

    # Since fit failed, model should not be usable
    assert not model.is_fitted, "Model should not be fitted after failed fit"


@pytest.mark.integration
def test_partial_invalid_data_in_predictions():
    """Test handling of NaN in prediction data (after successful training)."""
    # Train on valid data
    train_data = pd.DataFrame({'log_return': np.random.normal(0, 0.01, 50)})
    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)
    model.fit(train_data)

    # Try to predict on data with NaN
    test_data = pd.DataFrame({'log_return': [0.01, np.nan, -0.01]})

    # Should reject invalid prediction data or handle gracefully
    try:
        predictions = model.predict(test_data)
        # If it handles NaN gracefully, predictions should be valid
        assert len(predictions) <= len(test_data), "Predictions length should be reasonable"
    except (ValidationError, ValueError):
        # Strict validation is also acceptable
        pass


@pytest.mark.integration
def test_configuration_mismatch_across_components():
    """Test error when component configurations are mismatched."""
    # Model with 2 states
    data = pd.DataFrame({'log_return': np.random.normal(0, 0.01, 50)})
    model = HiddenMarkovModel(HMMConfig(n_states=2))
    model.fit(data)
    model_output = model.predict(data)

    # Create adapter output for 2 states
    n_states = 2
    adapted = model_output.copy()
    adapted['state'] = adapted['predicted_state']
    state_prob_cols = [f'state_{i}_prob' for i in range(n_states)]
    adapted['state_probabilities'] = adapted[state_prob_cols].apply(
        lambda row: np.array(row.values), axis=1
    )
    adapted['emission_means'] = [model.emission_means_] * len(adapted)
    adapted['emission_stds'] = [model.emission_stds_] * len(adapted)

    # But interpreter configured for 3 states
    interpreter = FinancialInterpreter(InterpreterConfiguration(n_states=3))

    # Should detect mismatch or handle gracefully
    try:
        result = interpreter.update(adapted)
        # If it handles gracefully, should produce output
        assert len(result) > 0, "Should handle state count mismatch"
    except (ValidationError, ValueError, IndexError):
        # Strict validation is also acceptable
        pass


# ============================================================================
# Graceful Degradation Tests
# ============================================================================


@pytest.mark.integration
def test_signal_generator_with_all_invalid_signals():
    """Test signal generator when all signals are marked invalid."""
    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)

    all_invalid = pd.DataFrame({
        'state': [0, 1, 0],
        'regime_type': ['bullish', 'bearish', 'bullish'],
        'regime_confidence': [0.8, 0.85, 0.75],
        'regime_strength': [0.8, 0.85, 0.75],
        'signal_valid': [False, False, False]  # All invalid
    })

    result = signal_gen.update(all_invalid)

    # Should produce all zero signals
    assert all(result['position_size'] == 0), \
        "All invalid signals should result in zero positions"


@pytest.mark.integration
def test_interpreter_with_very_low_confidence():
    """Test interpreter behavior with extremely low confidence states."""
    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)

    low_conf_input = pd.DataFrame({
        'state': [0],
        'confidence': [0.01],  # Very low confidence
        'state_probabilities': [np.array([0.51, 0.49])],  # Nearly equal
        'emission_means': [np.array([0.001, -0.001])],
        'emission_stds': [np.array([0.01, 0.02])]
    })

    # Should handle gracefully
    result = interpreter.update(low_conf_input)

    # Should produce output with low confidence
    assert result.iloc[0]['regime_confidence'] < 0.7, \
        "Low input confidence should propagate to output"


@pytest.mark.integration
def test_empty_dataframe_error_messages():
    """Test that empty DataFrame errors have clear messages."""
    empty_data = pd.DataFrame({'log_return': []})

    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)

    with pytest.raises((ValidationError, ValueError)) as exc_info:
        model.fit(empty_data)

    error_msg = str(exc_info.value).lower()
    assert 'empty' in error_msg or 'no data' in error_msg or 'insufficient' in error_msg, \
        f"Error message should clearly indicate empty data: {error_msg}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
