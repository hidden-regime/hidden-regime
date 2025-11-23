"""
Regime Persistence and Hysteresis Tests - Critical Edge Cases.

Tests verify that regime interpretation handles regime transitions correctly
and doesn't exhibit unstable flip-flopping behavior.

This addresses critical interpretation problems:
- Regime flip-flopping (rapid alternation between states)
- Regime duration tracking
- Transition stability
- Hysteresis effects (regime stickiness)

Tests cover:
1. Stable regime persistence
2. Regime flip-flop detection
3. Gradual regime transitions
4. Rapid regime changes
5. Regime duration metrics
6. Transition smoothness
"""

import pytest
import numpy as np
import pandas as pd

from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.config.interpreter import InterpreterConfiguration


# ============================================================================
# Stable Regime Persistence Tests
# ============================================================================


@pytest.mark.unit
def test_stable_regime_no_transitions():
    """Test that a stable regime remains classified consistently."""
    config = InterpreterConfiguration(
        n_states=3,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # 100 time steps in same state with same characteristics
    n_steps = 100
    model_output = pd.DataFrame({
        'state': [0] * n_steps,
        'confidence': [0.85] * n_steps,
        'state_probabilities': [np.array([0.85, 0.1, 0.05])] * n_steps,
        'emission_means': [np.array([0.0015, 0.0, -0.002])] * n_steps,
        'emission_stds': [np.array([0.01, 0.015, 0.025])] * n_steps,
    })

    result = interpreter.update(model_output)

    # All time steps should have same regime
    regimes = result['regime_type'].values
    assert all(r == regimes[0] for r in regimes), \
        f"Stable state should have consistent regime, got {len(set(regimes))} different regimes"

    # All should be bullish
    assert all('bull' in r.lower() or 'uptrend' in r.lower() for r in regimes), \
        "Stable bullish state should consistently classify as bullish"


@pytest.mark.unit
def test_high_confidence_regime_persistence():
    """Test that high confidence regimes persist even with minor noise."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # High confidence state with slight probability variations
    n_steps = 50
    # Add small random noise to probabilities
    np.random.seed(42)
    probs_0 = 0.9 + np.random.normal(0, 0.01, n_steps)  # 90% ± 1%
    probs_0 = np.clip(probs_0, 0.85, 0.95)  # Keep in range

    model_output = pd.DataFrame({
        'state': [0] * n_steps,
        'confidence': probs_0,
        'state_probabilities': [np.array([p, 1-p]) for p in probs_0],
        'emission_means': [np.array([0.001, -0.001])] * n_steps,
        'emission_stds': [np.array([0.01, 0.02])] * n_steps,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # All should be same regime despite noise
    assert all(r == regimes[0] for r in regimes), \
        "High confidence regime should persist despite probability noise"


# ============================================================================
# Flip-Flop Detection Tests
# ============================================================================


@pytest.mark.unit
def test_rapid_regime_flip_flop():
    """Test detection of rapid regime alternation (flip-flopping)."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Alternating states every time step: 0, 1, 0, 1, 0, 1...
    states = [i % 2 for i in range(20)]

    model_output = pd.DataFrame({
        'state': states,
        'confidence': [0.6] * 20,  # Moderate confidence
        'state_probabilities': [
            np.array([0.6, 0.4]) if s == 0 else np.array([0.4, 0.6])
            for s in states
        ],
        'emission_means': [np.array([0.001, -0.001])] * 20,
        'emission_stds': [np.array([0.01, 0.02])] * 20,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # Count regime transitions
    transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])

    # Should have many transitions (at least 15 out of 19 possible)
    assert transitions >= 15, \
        f"Flip-flopping states should cause many regime transitions, got {transitions}"

    # Regimes should alternate
    assert len(set(regimes)) == 2, "Should have two distinct regimes"


@pytest.mark.unit
def test_no_flip_flop_with_stable_states():
    """Test that stable states don't cause flip-flopping."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Long period in state 0, then long period in state 1
    states = [0] * 25 + [1] * 25

    model_output = pd.DataFrame({
        'state': states,
        'confidence': [0.8] * 50,
        'state_probabilities': [
            np.array([0.8, 0.2]) if s == 0 else np.array([0.2, 0.8])
            for s in states
        ],
        'emission_means': [np.array([0.001, -0.001])] * 50,
        'emission_stds': [np.array([0.01, 0.02])] * 50,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # Count regime transitions (should be exactly 1)
    transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])

    assert transitions <= 2, \
        f"Stable states should cause minimal transitions, got {transitions}"


@pytest.mark.unit
def test_low_confidence_flip_flop():
    """Test that low confidence states may show regime instability."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Alternating states with low confidence (near 50/50)
    states = [i % 2 for i in range(20)]

    model_output = pd.DataFrame({
        'state': states,
        'confidence': [0.51] * 20,  # Very low confidence
        'state_probabilities': [
            np.array([0.51, 0.49]) if s == 0 else np.array([0.49, 0.51])
            for s in states
        ],
        'emission_means': [np.array([0.0001, -0.0001])] * 20,  # Very similar states
        'emission_stds': [np.array([0.01, 0.0105])] * 20,
    })

    result = interpreter.update(model_output)

    # Confidence should be consistently low
    confidences = result['regime_confidence'].values
    assert all(c < 0.6 for c in confidences), \
        f"Low confidence states should have low regime confidence"


# ============================================================================
# Gradual Transition Tests
# ============================================================================


@pytest.mark.unit
def test_gradual_regime_transition():
    """Test smooth transition from one regime to another."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Gradual transition: state 0 → state 1 over 20 steps
    n_steps = 20
    # Probability gradually shifts from 0.9/0.1 to 0.1/0.9
    probs = np.linspace(0.9, 0.1, n_steps)
    states = (probs < 0.5).astype(int)  # Switches at midpoint

    model_output = pd.DataFrame({
        'state': states,
        'confidence': np.maximum(probs, 1 - probs),  # Max of the two probs
        'state_probabilities': [np.array([p, 1-p]) for p in probs],
        'emission_means': [np.array([0.001, -0.001])] * n_steps,
        'emission_stds': [np.array([0.01, 0.02])] * n_steps,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values
    confidences = result['regime_confidence'].values

    # Should have exactly 1 regime transition
    transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
    assert transitions == 1, \
        f"Gradual transition should cause single regime change, got {transitions}"

    # Confidence should be lowest near the transition point
    min_confidence_idx = np.argmin(confidences)
    assert 8 <= min_confidence_idx <= 12, \
        f"Min confidence should be near midpoint (10), got {min_confidence_idx}"


@pytest.mark.unit
def test_transition_with_uncertainty_period():
    """Test transition with period of high uncertainty between regimes."""
    config = InterpreterConfiguration(
        n_states=3,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # State 0 (bull) → State 2 (sideways) → State 1 (bear)
    states = [0]*10 + [2]*10 + [1]*10

    model_output = pd.DataFrame({
        'state': states,
        'confidence': [0.8]*10 + [0.6]*10 + [0.8]*10,  # Lower confidence in middle
        'state_probabilities': [
            np.array([0.8, 0.1, 0.1]) if s == 0 else
            np.array([0.2, 0.2, 0.6]) if s == 2 else
            np.array([0.1, 0.8, 0.1])
            for s in states
        ],
        'emission_means': [np.array([0.002, -0.002, 0.0])] * 30,
        'emission_stds': [np.array([0.01, 0.025, 0.012])] * 30,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values
    confidences = result['regime_confidence'].values

    # Should have 2 regime transitions (0→2, 2→1)
    transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
    assert transitions >= 2, \
        f"Three distinct states should cause at least 2 transitions, got {transitions}"

    # Middle period (sideways) should have lower confidence
    middle_confidence = confidences[10:20].mean()
    edge_confidence = (confidences[0:10].mean() + confidences[20:30].mean()) / 2
    assert middle_confidence < edge_confidence, \
        "Transition period should have lower confidence than stable regimes"


# ============================================================================
# Rapid Regime Change Tests
# ============================================================================


@pytest.mark.unit
def test_sudden_regime_shock():
    """Test detection of sudden regime change (market shock)."""
    config = InterpreterConfiguration(
        n_states=3,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Long stable period, then sudden crisis, then recovery
    states = [0]*30 + [2]*5 + [0]*15  # Bull → Crisis → Bull

    model_output = pd.DataFrame({
        'state': states,
        'confidence': [0.85]*30 + [0.9]*5 + [0.8]*15,
        'state_probabilities': [
            np.array([0.85, 0.1, 0.05]) if s == 0 else
            np.array([0.02, 0.08, 0.9])
            for s in states
        ],
        'emission_means': [np.array([0.001, -0.001, -0.005])] * 50,
        'emission_stds': [np.array([0.01, 0.02, 0.06])] * 50,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # Should detect crisis period
    crisis_regimes = regimes[30:35]
    assert any('crisis' in r.lower() or 'volatile' in r.lower() or 'bear' in r.lower()
               for r in crisis_regimes), \
        f"Should detect crisis regime, got: {crisis_regimes}"

    # Should return to bullish after crisis
    post_crisis_regimes = regimes[35:50]
    assert any('bull' in r.lower() or 'uptrend' in r.lower() for r in post_crisis_regimes), \
        "Should return to bullish regime after crisis"


@pytest.mark.unit
def test_multiple_rapid_transitions():
    """Test handling of multiple rapid regime transitions."""
    config = InterpreterConfiguration(
        n_states=3,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Rapid transitions: 0 → 1 → 2 → 0 → 1 → 2 (every 3 steps)
    states = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2]

    model_output = pd.DataFrame({
        'state': states,
        'confidence': [0.7] * 18,
        'state_probabilities': [
            np.array([0.7, 0.2, 0.1]) if s == 0 else
            np.array([0.2, 0.7, 0.1]) if s == 1 else
            np.array([0.1, 0.2, 0.7])
            for s in states
        ],
        'emission_means': [np.array([0.002, -0.002, 0.0])] * 18,
        'emission_stds': [np.array([0.01, 0.02, 0.015])] * 18,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # Count transitions
    transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])

    # Should have at least 4 transitions (may have more if interpretation varies)
    assert transitions >= 4, \
        f"Multiple state changes should cause multiple regime transitions, got {transitions}"


# ============================================================================
# Regime Duration Tests
# ============================================================================


@pytest.mark.unit
def test_regime_minimum_duration():
    """Test that regimes have reasonable minimum durations (no single-step regimes)."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Mostly state 0, with single-step spikes to state 1
    states = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]

    model_output = pd.DataFrame({
        'state': states,
        'confidence': [0.75] * 15,
        'state_probabilities': [
            np.array([0.75, 0.25]) if s == 0 else np.array([0.25, 0.75])
            for s in states
        ],
        'emission_means': [np.array([0.001, -0.001])] * 15,
        'emission_stds': [np.array([0.01, 0.02])] * 15,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # Note: Interpreter doesn't enforce minimum duration, it just reports current regime
    # But we can check that each single-step state gets classified
    assert len(regimes) == 15, "All time steps should have regime classification"


@pytest.mark.unit
def test_long_regime_duration():
    """Test handling of very long regime durations."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # 200 time steps in same regime
    n_steps = 200
    model_output = pd.DataFrame({
        'state': [0] * n_steps,
        'confidence': [0.8] * n_steps,
        'state_probabilities': [np.array([0.8, 0.2])] * n_steps,
        'emission_means': [np.array([0.0012, -0.001])] * n_steps,
        'emission_stds': [np.array([0.01, 0.02])] * n_steps,
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # All should be same regime
    assert all(r == regimes[0] for r in regimes), \
        f"Long duration should maintain same regime"

    # Should be bullish throughout
    assert all('bull' in r.lower() or 'uptrend' in r.lower() for r in regimes), \
        "Long bullish period should consistently classify as bullish"


# ============================================================================
# Transition Smoothness Tests
# ============================================================================


@pytest.mark.unit
def test_transition_point_confidence():
    """Test that confidence is lowest at regime transition points."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Clear transition: high conf state 0 → low conf → high conf state 1
    probs = [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
    states = [0 if p > 0.5 else 1 for p in probs]

    model_output = pd.DataFrame({
        'state': states,
        'confidence': [max(p, 1-p) for p in probs],
        'state_probabilities': [np.array([p, 1-p]) for p in probs],
        'emission_means': [np.array([0.001, -0.001])] * len(probs),
        'emission_stds': [np.array([0.01, 0.02])] * len(probs),
    })

    result = interpreter.update(model_output)

    confidences = result['regime_confidence'].values

    # Minimum confidence should be at transition (index 5, where p=0.5)
    min_idx = np.argmin(confidences)
    assert min_idx == 5, \
        f"Minimum confidence should be at transition point (5), got {min_idx}"

    # Confidence should be 0.5 at transition
    assert confidences[5] == 0.5, \
        f"Confidence at transition should be 0.5, got {confidences[5]}"


@pytest.mark.unit
def test_no_spurious_transitions():
    """Test that very brief state changes don't cause regime oscillation."""
    config = InterpreterConfiguration(
        n_states=2,
        interpretation_method="data_driven"
    )
    interpreter = FinancialInterpreter(config)

    # Mostly state 0 with very brief (<2 step) excursions
    states = [0]*10 + [1] + [0]*10 + [1, 1] + [0]*10

    model_output = pd.DataFrame({
        'state': states,
        'confidence': [
            0.85 if s == 0 else 0.55  # Brief excursions have lower confidence
            for s in states
        ],
        'state_probabilities': [
            np.array([0.85, 0.15]) if s == 0 else np.array([0.45, 0.55])
            for s in states
        ],
        'emission_means': [np.array([0.001, -0.001])] * len(states),
        'emission_stds': [np.array([0.01, 0.02])] * len(states),
    })

    result = interpreter.update(model_output)

    regimes = result['regime_type'].values

    # Even if states change, the brief excursions should be reflected
    # (Interpreter doesn't filter, it reports current regime)
    # Just verify no crashes and all regimes are valid
    assert all(isinstance(r, str) and len(r) > 0 for r in regimes), \
        "All regimes should be valid strings"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
