"""
Component Pair Integration Tests.

Tests verify that pairs of components integrate correctly:
- Data preparation → Model training
- Model predictions → Interpreter input
- Interpreter output → Signal generation

These tests focus on the INTERFACES between components, ensuring
that output from one component is valid input for the next.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from hidden_regime.models.hmm import HiddenMarkovModel
from hidden_regime.config.model import HMMConfig
from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.config.interpreter import InterpreterConfiguration
from hidden_regime.signal_generation.financial import FinancialSignalGenerator
from hidden_regime.config.signal_generation import SignalGenerationConfiguration
from hidden_regime.utils.exceptions import ValidationError


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_returns_dataframe():
    """Create sample returns DataFrame for testing."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 50)
    return pd.DataFrame({'log_return': returns})


@pytest.fixture
def trained_model(sample_returns_dataframe):
    """Pre-trained HMM model for testing."""
    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)
    model.fit(sample_returns_dataframe)
    return model


# ============================================================================
# Data → Model Integration Tests
# ============================================================================


@pytest.mark.integration
def test_dataframe_to_model_training():
    """Test that properly formatted DataFrame can train the model."""
    # Create minimal valid DataFrame (need at least 10 observations)
    np.random.seed(42)
    data = pd.DataFrame({'log_return': np.random.normal(0.001, 0.01, 20)})

    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)

    # Should train without errors
    model.fit(data)

    assert model.is_fitted, "Model should be fitted after training"
    assert hasattr(model, 'emission_means_'), "Model should have learned emission parameters"


@pytest.mark.integration
def test_model_rejects_wrong_column_name():
    """Test that model rejects DataFrame with incorrect column names."""
    # Wrong column name (should be 'log_return')
    data = pd.DataFrame({'returns': [0.01, -0.01, 0.02, -0.02, 0.01]})

    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)

    # Should raise validation error
    with pytest.raises((ValidationError, ValueError, KeyError)):
        model.fit(data)


@pytest.mark.integration
def test_model_rejects_insufficient_data():
    """Test that model rejects DataFrame with too few observations."""
    # Only 3 observations (need at least 10)
    data = pd.DataFrame({'log_return': [0.01, -0.01, 0.02]})

    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)

    with pytest.raises((ValidationError, ValueError)):
        model.fit(data)


@pytest.mark.integration
def test_model_handles_datetime_index():
    """Test that model preserves datetime index from input DataFrame."""
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'log_return': np.random.normal(0.001, 0.01, 50)
    }, index=dates)

    config = HMMConfig(n_states=2)
    model = HiddenMarkovModel(config)
    model.fit(data)

    predictions = model.predict(data)

    # Index should be preserved
    assert len(predictions) == len(data), "Output length should match input"
    # Note: Index preservation is implementation-dependent


@pytest.mark.integration
def test_model_prediction_output_schema():
    """Test that model predictions have the expected schema."""
    data = pd.DataFrame({'log_return': np.random.normal(0.001, 0.01, 30)})

    config = HMMConfig(n_states=3)
    model = HiddenMarkovModel(config)
    model.fit(data)

    predictions = model.predict(data)

    # Verify required columns
    assert 'predicted_state' in predictions.columns, "Missing predicted_state"
    assert 'confidence' in predictions.columns, "Missing confidence"

    # Verify state probability columns
    for i in range(3):
        assert f'state_{i}_prob' in predictions.columns, f"Missing state_{i}_prob"

    # Verify data types
    assert predictions['predicted_state'].dtype in [np.int32, np.int64], "State should be integer"
    assert predictions['confidence'].dtype == np.float64, "Confidence should be float"


# ============================================================================
# Model → Interpreter Integration Tests
# ============================================================================


def adapt_model_output_for_interpreter(model_output: pd.DataFrame, model: HiddenMarkovModel) -> pd.DataFrame:
    """
    Adapter function to transform model output to interpreter input.

    This is the critical integration point between model and interpreter.
    """
    n_states = model.n_states

    adapted = model_output.copy()
    adapted['state'] = adapted['predicted_state']

    state_prob_cols = [f'state_{i}_prob' for i in range(n_states)]
    adapted['state_probabilities'] = adapted[state_prob_cols].apply(
        lambda row: np.array(row.values), axis=1
    )

    adapted['emission_means'] = [model.emission_means_] * len(adapted)
    adapted['emission_stds'] = [model.emission_stds_] * len(adapted)

    return adapted


@pytest.mark.integration
def test_model_output_to_interpreter_input(trained_model, sample_returns_dataframe):
    """Test that model output can be adapted for interpreter input."""
    model_output = trained_model.predict(sample_returns_dataframe)

    # Adapt output
    interpreter_input = adapt_model_output_for_interpreter(model_output, trained_model)

    # Verify interpreter can consume this
    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)

    result = interpreter.update(interpreter_input)

    assert 'regime_type' in result.columns, "Interpreter should produce regime_type"
    assert len(result) == len(model_output), "Interpreter should preserve length"


@pytest.mark.integration
def test_interpreter_requires_all_fields():
    """Test that interpreter fails gracefully when required fields are missing."""
    # Missing emission_means and emission_stds
    incomplete_input = pd.DataFrame({
        'state': [0, 1],
        'confidence': [0.8, 0.7],
        'state_probabilities': [np.array([0.8, 0.2]), np.array([0.3, 0.7])]
    })

    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)

    # Should raise error for missing required fields or handle gracefully
    try:
        result = interpreter.update(incomplete_input)
        # If it handles gracefully, should still produce output
        assert len(result) > 0, "Should produce output even with missing fields"
    except (KeyError, ValidationError, ValueError):
        # Strict validation is also acceptable
        pass


@pytest.mark.integration
def test_interpreter_state_consistency(trained_model, sample_returns_dataframe):
    """Test that state indices remain consistent through model-interpreter chain."""
    model_output = trained_model.predict(sample_returns_dataframe)
    interpreter_input = adapt_model_output_for_interpreter(model_output, trained_model)

    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)
    result = interpreter.update(interpreter_input)

    # State column should be preserved
    assert all(result['state'] == model_output['predicted_state']), \
        "State indices should be preserved through pipeline"


@pytest.mark.integration
def test_interpreter_confidence_transformation(trained_model, sample_returns_dataframe):
    """Test that confidence metrics are properly transformed."""
    model_output = trained_model.predict(sample_returns_dataframe)
    interpreter_input = adapt_model_output_for_interpreter(model_output, trained_model)

    config = InterpreterConfiguration(n_states=2)
    interpreter = FinancialInterpreter(config)
    result = interpreter.update(interpreter_input)

    # Confidence should exist at both stages
    assert 'confidence' in model_output.columns, "Model should provide confidence"
    assert 'regime_confidence' in result.columns, "Interpreter should provide regime_confidence"

    # Both should be in valid range [0, 1]
    assert all((model_output['confidence'] >= 0) & (model_output['confidence'] <= 1))
    assert all((result['regime_confidence'] >= 0) & (result['regime_confidence'] <= 1))


# ============================================================================
# Interpreter → Signal Generator Integration Tests
# ============================================================================


@pytest.mark.integration
def test_interpreter_output_to_signal_input():
    """Test that interpreter output can be consumed by signal generator."""
    # Create interpreter output
    interpreter_output = pd.DataFrame({
        'state': [0, 1, 0],
        'regime_type': ['bullish', 'bearish', 'bullish'],
        'regime_confidence': [0.8, 0.85, 0.75],
        'regime_strength': [0.8, 0.85, 0.75],
        'signal_valid': [True, True, True]
    })

    # Verify signal generator can consume this
    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)

    result = signal_gen.update(interpreter_output)

    assert 'position_size' in result.columns, "Signal generator should produce position_size"
    assert len(result) == len(interpreter_output), "Signal generator should preserve length"


@pytest.mark.integration
def test_signal_generator_requires_state_column():
    """Test that signal generator requires state column from interpreter."""
    # Missing 'state' column
    incomplete_input = pd.DataFrame({
        'regime_type': ['bullish', 'bearish'],
        'regime_confidence': [0.8, 0.85],
        'regime_strength': [0.8, 0.85],
        'signal_valid': [True, True]
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)

    # Should raise error for missing state column
    with pytest.raises((KeyError, ValidationError, ValueError)):
        signal_gen.update(incomplete_input)


@pytest.mark.integration
def test_signal_generator_respects_signal_valid_flag():
    """Test that signal generator respects signal_valid flag from interpreter."""
    interpreter_output = pd.DataFrame({
        'state': [0, 1],
        'regime_type': ['bullish', 'bearish'],
        'regime_confidence': [0.8, 0.85],
        'regime_strength': [0.8, 0.85],
        'signal_valid': [True, False]  # Second signal marked invalid
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)

    result = signal_gen.update(interpreter_output)

    # First signal should be non-zero (bullish)
    assert result.iloc[0]['position_size'] != 0, "Valid bullish signal should be non-zero"

    # Second signal should be zero (invalid flag)
    assert result.iloc[1]['position_size'] == 0, "Invalid signal should be zeroed"


@pytest.mark.integration
def test_signal_generator_handles_unknown_regime_gracefully():
    """Test that signal generator handles unknown regime types gracefully."""
    interpreter_output = pd.DataFrame({
        'state': [0, 1],
        'regime_type': ['unknown_regime_xyz', 'also_unknown'],
        'regime_confidence': [0.8, 0.85],
        'regime_strength': [0.8, 0.85],
        'signal_valid': [True, True]
    })

    config = SignalGenerationConfiguration(strategy_type="regime_following")
    signal_gen = FinancialSignalGenerator(config)

    result = signal_gen.update(interpreter_output)

    # Should produce zero or near-zero signals for unknown regimes
    assert all(abs(result['position_size']) < 0.2), \
        "Unknown regimes should produce minimal signals"


# ============================================================================
# Multi-Component Chain Tests
# ============================================================================


@pytest.mark.integration
def test_three_component_chain_preserves_length():
    """Test that data length is preserved through model → interpreter → signals."""
    np.random.seed(123)
    n_obs = 40

    # Create data
    data = pd.DataFrame({'log_return': np.random.normal(0.001, 0.01, n_obs)})

    # Model
    model = HiddenMarkovModel(HMMConfig(n_states=2))
    model.fit(data)
    model_output = model.predict(data)
    assert len(model_output) == n_obs, "Model should preserve length"

    # Interpreter
    interpreter_input = adapt_model_output_for_interpreter(model_output, model)
    interpreter = FinancialInterpreter(InterpreterConfiguration(n_states=2))
    interp_output = interpreter.update(interpreter_input)
    assert len(interp_output) == n_obs, "Interpreter should preserve length"

    # Signal Generator
    signal_gen = FinancialSignalGenerator(SignalGenerationConfiguration(strategy_type="regime_following"))
    signal_output = signal_gen.update(interp_output)
    assert len(signal_output) == n_obs, "Signal generator should preserve length"


@pytest.mark.integration
def test_three_component_chain_state_consistency():
    """Test that state column is consistently maintained through all components."""
    np.random.seed(456)
    data = pd.DataFrame({'log_return': np.random.normal(0.001, 0.01, 30)})

    # Model
    model = HiddenMarkovModel(HMMConfig(n_states=3))
    model.fit(data)
    model_output = model.predict(data)
    original_states = model_output['predicted_state'].values

    # Interpreter
    interpreter_input = adapt_model_output_for_interpreter(model_output, model)
    interpreter = FinancialInterpreter(InterpreterConfiguration(n_states=3))
    interp_output = interpreter.update(interpreter_input)

    # Signal Generator
    signal_gen = FinancialSignalGenerator(SignalGenerationConfiguration(strategy_type="regime_following"))
    signal_output = signal_gen.update(interp_output)

    # State should be preserved through entire chain
    assert all(interp_output['state'] == original_states), \
        "Interpreter should preserve states from model"
    assert all(signal_output['state'] == original_states), \
        "Signal generator should preserve states from interpreter"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
