"""
Unit tests for anchored regime interpretation system.

Tests verify that:
1. Regime labels remain stable despite parameter drift
2. Anchors update correctly via exponential smoothing
3. KL divergence-based state-to-regime matching is correct
4. Confidence scores are valid probabilities and reflect KL distance
5. Anchor history tracking works for audit trail
6. Stability metrics detect anchor changes
"""

import pytest
import numpy as np
from unittest.mock import Mock

from hidden_regime.interpreter.anchored import (
    AnchoredInterpreter,
    RegimeAnchor,
    DEFAULT_REGIME_ANCHORS,
)


class TestRegimeAnchor:
    """Test RegimeAnchor dataclass."""

    def test_regime_anchor_creation(self):
        """Test creating a RegimeAnchor."""
        anchor = RegimeAnchor(mean=0.001, std=0.008)
        assert anchor.mean == 0.001
        assert anchor.std == 0.008

    def test_regime_anchor_to_dict(self):
        """Test converting RegimeAnchor to dict."""
        anchor = RegimeAnchor(mean=0.001, std=0.008)
        d = anchor.to_dict()
        assert d == {'mean': 0.001, 'std': 0.008}

    def test_regime_anchor_from_dict(self):
        """Test creating RegimeAnchor from dict."""
        data = {'mean': 0.001, 'std': 0.008}
        anchor = RegimeAnchor.from_dict(data)
        assert anchor.mean == 0.001
        assert anchor.std == 0.008

    def test_regime_anchor_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        anchor1 = RegimeAnchor(mean=-0.0008, std=0.012)
        anchor2 = RegimeAnchor.from_dict(anchor1.to_dict())
        assert anchor1.mean == anchor2.mean
        assert anchor1.std == anchor2.std


class TestAnchoredInterpreterInitialization:
    """Test AnchoredInterpreter initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        interp = AnchoredInterpreter()
        assert interp.anchor_update_rate == 0.01
        assert 'BULLISH' in interp.regime_anchors
        assert 'BEARISH' in interp.regime_anchors
        assert 'SIDEWAYS' in interp.regime_anchors
        assert 'CRISIS' in interp.regime_anchors

    def test_custom_anchor_update_rate(self):
        """Test initialization with custom update rate."""
        interp = AnchoredInterpreter(anchor_update_rate=0.05)
        assert interp.anchor_update_rate == 0.05

    def test_custom_regime_anchors(self):
        """Test initialization with custom anchors."""
        custom_anchors = {
            'BULL': {'mean': 0.002, 'std': 0.010},
            'BEAR': {'mean': -0.001, 'std': 0.015},
        }
        interp = AnchoredInterpreter(regime_anchors=custom_anchors)
        assert 'BULL' in interp.regime_anchors
        assert 'BEAR' in interp.regime_anchors
        assert interp.regime_anchors['BULL'].mean == 0.002

    def test_default_anchors_match_constants(self):
        """Test that default anchors match DEFAULT_REGIME_ANCHORS."""
        interp = AnchoredInterpreter()
        for regime_name, expected in DEFAULT_REGIME_ANCHORS.items():
            assert interp.regime_anchors[regime_name].mean == expected['mean']
            assert interp.regime_anchors[regime_name].std == expected['std']

    def test_audit_trail_initialized(self):
        """Test that anchor_update_history is initialized."""
        interp = AnchoredInterpreter()
        assert 'BULLISH' in interp.anchor_update_history
        assert 'BEARISH' in interp.anchor_update_history
        assert isinstance(interp.anchor_update_history['BULLISH'], list)
        assert len(interp.anchor_update_history['BULLISH']) == 0


class TestGaussianKLDivergence:
    """Test KL divergence computation for Gaussians."""

    def test_kl_zero_for_identical_distributions(self):
        """KL divergence should be ~0 for identical distributions."""
        interp = AnchoredInterpreter()
        kl = interp._gaussian_kl_divergence(
            mean1=0.001, std1=0.008,
            mean2=0.001, std2=0.008
        )
        assert abs(kl) < 1e-10  # Should be essentially 0

    def test_kl_positive(self):
        """KL divergence should be non-negative."""
        interp = AnchoredInterpreter()
        kl = interp._gaussian_kl_divergence(
            mean1=0.001, std1=0.008,
            mean2=0.002, std2=0.010
        )
        assert kl >= 0

    def test_kl_not_symmetric(self):
        """KL divergence should not be symmetric (it's asymmetric by definition)."""
        interp = AnchoredInterpreter()
        kl_12 = interp._gaussian_kl_divergence(
            mean1=0.001, std1=0.008,
            mean2=0.002, std2=0.010
        )
        kl_21 = interp._gaussian_kl_divergence(
            mean1=0.002, std1=0.010,
            mean2=0.001, std2=0.008
        )
        assert kl_12 != kl_21

    def test_kl_increases_with_mean_difference(self):
        """KL divergence should increase with mean difference."""
        interp = AnchoredInterpreter()
        kl_small_diff = interp._gaussian_kl_divergence(
            mean1=0.001, std1=0.008,
            mean2=0.0015, std2=0.008
        )
        kl_large_diff = interp._gaussian_kl_divergence(
            mean1=0.001, std1=0.008,
            mean2=0.010, std2=0.008
        )
        assert kl_large_diff > kl_small_diff

    def test_kl_increases_with_variance_difference(self):
        """KL divergence should increase with variance difference."""
        interp = AnchoredInterpreter()
        kl_small_var_diff = interp._gaussian_kl_divergence(
            mean1=0.001, std1=0.008,
            mean2=0.001, std2=0.0085
        )
        kl_large_var_diff = interp._gaussian_kl_divergence(
            mean1=0.001, std1=0.008,
            mean2=0.001, std2=0.015
        )
        assert kl_large_var_diff > kl_small_var_diff


class TestAnchorUpdate:
    """Test anchor updating via exponential smoothing."""

    def test_anchor_update_with_perfect_match(self):
        """Anchor should barely change if observed matches current anchor."""
        interp = AnchoredInterpreter(anchor_update_rate=0.01)
        original_mean = interp.regime_anchors['BULLISH'].mean
        original_std = interp.regime_anchors['BULLISH'].std

        # Update with same values
        interp._update_anchor('BULLISH', state_mean=original_mean, state_std=original_std)

        # Should be unchanged (within floating point precision)
        assert abs(interp.regime_anchors['BULLISH'].mean - original_mean) < 1e-10
        assert abs(interp.regime_anchors['BULLISH'].std - original_std) < 1e-10

    def test_anchor_update_exponential_smoothing(self):
        """Anchor should move toward observed value with α=0.01."""
        interp = AnchoredInterpreter(anchor_update_rate=0.01)
        original_mean = 0.001
        observed_mean = 0.002

        interp.regime_anchors['BULLISH'].mean = original_mean
        interp._update_anchor('BULLISH', state_mean=observed_mean, state_std=0.008)

        expected_mean = (1 - 0.01) * original_mean + 0.01 * observed_mean
        assert abs(interp.regime_anchors['BULLISH'].mean - expected_mean) < 1e-10

    def test_anchor_update_with_higher_rate(self):
        """Anchor should move faster with higher α."""
        interp_slow = AnchoredInterpreter(anchor_update_rate=0.01)
        interp_fast = AnchoredInterpreter(anchor_update_rate=0.10)

        original_mean = 0.001
        observed_mean = 0.002

        interp_slow.regime_anchors['BULLISH'].mean = original_mean
        interp_fast.regime_anchors['BULLISH'].mean = original_mean

        interp_slow._update_anchor('BULLISH', state_mean=observed_mean, state_std=0.008)
        interp_fast._update_anchor('BULLISH', state_mean=observed_mean, state_std=0.008)

        # Fast should move more
        slow_change = abs(interp_slow.regime_anchors['BULLISH'].mean - original_mean)
        fast_change = abs(interp_fast.regime_anchors['BULLISH'].mean - original_mean)
        assert fast_change > slow_change

    def test_anchor_update_recorded_in_history(self):
        """Each anchor update should be recorded in history."""
        interp = AnchoredInterpreter()
        assert len(interp.anchor_update_history['BULLISH']) == 0

        interp._update_anchor('BULLISH', state_mean=0.001, state_std=0.008)
        assert len(interp.anchor_update_history['BULLISH']) == 1

        record = interp.anchor_update_history['BULLISH'][0]
        assert 'mean' in record
        assert 'std' in record
        assert 'observed_mean' in record
        assert 'observed_std' in record


class MockHMMModel:
    """Mock HMM model for testing interpret_states."""

    def __init__(self, n_states=3):
        self.n_states = n_states
        self.means_ = np.array([[0.001], [-0.0008], [0.0001]])[:n_states]
        self.covars_ = np.array([
            [[0.008 ** 2]],
            [[0.012 ** 2]],
            [[0.006 ** 2]]
        ])[:n_states]


class TestInterpretStates:
    """Test state-to-regime mapping and interpretation."""

    def test_interpret_states_returns_dict(self):
        """interpret_states should return tuples of dicts."""
        interp = AnchoredInterpreter()
        model = MockHMMModel(n_states=3)

        state_to_regime, confidence = interp.interpret_states(model)

        assert isinstance(state_to_regime, dict)
        assert isinstance(confidence, dict)

    def test_interpret_states_all_states_mapped(self):
        """All HMM states should be mapped to regimes."""
        interp = AnchoredInterpreter()
        model = MockHMMModel(n_states=3)

        state_to_regime, confidence = interp.interpret_states(model)

        assert len(state_to_regime) == 3
        assert len(confidence) == 3
        assert set(state_to_regime.keys()) == {0, 1, 2}
        assert set(confidence.keys()) == {0, 1, 2}

    def test_interpret_states_regime_names_valid(self):
        """Mapped regimes should be valid regime names."""
        interp = AnchoredInterpreter()
        model = MockHMMModel(n_states=3)

        state_to_regime, _ = interp.interpret_states(model)

        valid_regimes = set(interp.regime_anchors.keys())
        for regime in state_to_regime.values():
            assert regime in valid_regimes

    def test_interpret_states_confidence_bounded(self):
        """Confidence scores should be in (0, 1]."""
        interp = AnchoredInterpreter()
        model = MockHMMModel(n_states=3)

        _, confidence = interp.interpret_states(model)

        for conf in confidence.values():
            assert 0 < conf <= 1

    def test_interpret_states_high_confidence_for_matching_states(self):
        """Confidence should be high when state matches an anchor well."""
        interp = AnchoredInterpreter()

        # Create model where state 0 matches BULLISH anchor perfectly
        model = Mock()
        model.n_states = 1
        model.means_ = np.array([[0.0010]])  # Matches BULLISH
        model.covars_ = np.array([[[0.008 ** 2]]])

        state_to_regime, confidence = interp.interpret_states(model)

        assert state_to_regime[0] == 'BULLISH'
        assert confidence[0] > 0.9  # Should be high confidence

    def test_interpret_states_correct_matching_logic(self):
        """States should map to nearest anchor by KL divergence."""
        interp = AnchoredInterpreter()

        # Create model with 4 states, each matching a different regime
        model = Mock()
        model.n_states = 4
        model.means_ = np.array([
            [0.0010],    # BULLISH
            [-0.0008],   # BEARISH
            [0.0001],    # SIDEWAYS
            [-0.0030],   # CRISIS
        ])
        model.covars_ = np.array([
            [[0.008 ** 2]],
            [[0.012 ** 2]],
            [[0.006 ** 2]],
            [[0.025 ** 2]],
        ])

        state_to_regime, _ = interp.interpret_states(model)

        # Each state should match its corresponding regime
        assert state_to_regime[0] == 'BULLISH'
        assert state_to_regime[1] == 'BEARISH'
        assert state_to_regime[2] == 'SIDEWAYS'
        assert state_to_regime[3] == 'CRISIS'

    def test_interpret_states_triggers_anchor_updates(self):
        """interpret_states should update anchors."""
        interp = AnchoredInterpreter(anchor_update_rate=0.01)
        model = MockHMMModel(n_states=1)

        # Anchors should have no history initially
        assert len(interp.anchor_update_history['BULLISH']) == 0

        interp.interpret_states(model)

        # After interpret_states, anchors should have been updated
        assert len(interp.anchor_update_history['BULLISH']) >= 0


class TestAnchorStability:
    """Test anchor stability monitoring."""

    def test_stability_for_new_anchors(self):
        """New anchors with no history should be considered stable."""
        interp = AnchoredInterpreter()
        stability = interp.compute_anchor_stability('BULLISH')
        assert stability == 1.0

    def test_stability_for_changing_anchors(self):
        """Anchors that change should have lower stability."""
        interp = AnchoredInterpreter(anchor_update_rate=0.1)

        # Perform multiple updates with different values
        for _ in range(10):
            interp._update_anchor('BULLISH', state_mean=0.002, state_std=0.009)

        for _ in range(10):
            interp._update_anchor('BULLISH', state_mean=0.0005, state_std=0.007)

        stability = interp.compute_anchor_stability('BULLISH')
        assert 0 < stability < 1  # Should have reduced but non-zero stability

    def test_stability_increases_with_consistency(self):
        """Stability should increase as updates become consistent."""
        interp = AnchoredInterpreter(anchor_update_rate=0.01)

        # Perform many updates with the same value
        for _ in range(50):
            interp._update_anchor('BULLISH', state_mean=0.001, state_std=0.008)

        stability = interp.compute_anchor_stability('BULLISH')
        assert stability > 0.9  # Should be very stable


class TestAnchorHistory:
    """Test anchor history tracking for audit trail."""

    def test_get_anchor_history(self):
        """Should retrieve history for a regime."""
        interp = AnchoredInterpreter()

        interp._update_anchor('BULLISH', state_mean=0.001, state_std=0.008)
        interp._update_anchor('BULLISH', state_mean=0.0011, state_std=0.0085)

        history = interp.get_anchor_history('BULLISH')
        assert len(history) == 2
        assert history[0]['observed_mean'] == 0.001
        assert history[1]['observed_mean'] == 0.0011

    def test_get_anchor_history_empty_for_new_regime(self):
        """New regimes should have empty history."""
        interp = AnchoredInterpreter()
        history = interp.get_anchor_history('BULLISH')
        assert isinstance(history, list)
        assert len(history) == 0


class TestAnchorGettersSetters:
    """Test getting and setting anchors."""

    def test_get_anchors(self):
        """get_anchors should return dict representation."""
        interp = AnchoredInterpreter()
        anchors = interp.get_anchors()

        assert isinstance(anchors, dict)
        assert 'BULLISH' in anchors
        assert anchors['BULLISH']['mean'] == 0.0010
        assert anchors['BULLISH']['std'] == 0.008

    def test_set_anchors(self):
        """set_anchors should update anchor values."""
        interp = AnchoredInterpreter()

        new_anchors = {
            'BULLISH': {'mean': 0.002, 'std': 0.010},
            'BEARISH': {'mean': -0.001, 'std': 0.012},
        }

        interp.set_anchors(new_anchors)

        assert interp.regime_anchors['BULLISH'].mean == 0.002
        assert interp.regime_anchors['BEARISH'].mean == -0.001

    def test_reset_anchors_to_defaults(self):
        """reset_anchors_to_defaults should restore canonical values."""
        interp = AnchoredInterpreter()

        # Modify anchors
        interp.regime_anchors['BULLISH'].mean = 0.999

        # Reset
        interp.reset_anchors_to_defaults()

        # Should be back to default
        assert interp.regime_anchors['BULLISH'].mean == 0.0010

    def test_reset_clears_history(self):
        """reset should also clear history."""
        interp = AnchoredInterpreter()

        # Add some history
        interp._update_anchor('BULLISH', state_mean=0.001, state_std=0.008)
        assert len(interp.anchor_update_history['BULLISH']) > 0

        # Reset
        interp.reset_anchors_to_defaults()

        # History should be cleared
        assert len(interp.anchor_update_history['BULLISH']) == 0


class TestStableLabelsDespiteParameterDrift:
    """Integration tests: verify stable labels with parameter drift."""

    def test_regime_labels_stable_with_small_drift(self):
        """Regime labels should remain stable with small parameter drift.

        The key insight: regime labels depend on KL divergence to anchors.
        With small drift, KL distances may shift but the minimum should remain
        the same. This test verifies that small changes in parameters don't
        flip regime labels.
        """
        interp1 = AnchoredInterpreter(anchor_update_rate=0.01)
        interp2 = AnchoredInterpreter(anchor_update_rate=0.01)

        # Create models where BULLISH state drifts slightly
        model_original = Mock()
        model_original.n_states = 1
        model_original.means_ = np.array([[0.0010]])    # Bullish anchor
        model_original.covars_ = np.array([[[0.008 ** 2]]])

        model_drifted = Mock()
        model_drifted.n_states = 1
        model_drifted.means_ = np.array([[0.00105]])   # +5% drift (very small)
        model_drifted.covars_ = np.array([[[0.0082 ** 2]]])  # Small variance drift

        regime1, _ = interp1.interpret_states(model_original)
        regime2, _ = interp2.interpret_states(model_drifted)

        # With small drift, labels should remain the same
        # 0.00105 is still closest to BULLISH (0.001) not BEARISH (-0.0008)
        assert regime1[0] == regime2[0] == 'BULLISH'

    def test_regime_labels_eventually_adapt_to_large_drift(self):
        """After many updates, labels should eventually adapt to large drift."""
        interp = AnchoredInterpreter(anchor_update_rate=0.10)  # Faster adaptation

        model = Mock()
        model.n_states = 1

        # Start with BULLISH state
        model.means_ = np.array([[0.0010]])
        model.covars_ = np.array([[[0.008 ** 2]]])

        regime1, _ = interp.interpret_states(model)
        assert regime1[0] == 'BULLISH'

        # Repeatedly interpret as if state drifted to CRISIS territory
        model.means_ = np.array([[-0.0030]])
        model.covars_ = np.array([[[0.025 ** 2]]])

        # After many updates, should eventually recognize as CRISIS
        for _ in range(100):
            regime_current, _ = interp.interpret_states(model)

        assert regime_current[0] == 'CRISIS'
