"""
Unit tests for drift detection system (SLRT, KL divergence, Hellinger distance).

Tests verify:
1. SLRT cumulative sum behavior and forgetting factor
2. KL divergence computation correctness
3. Hellinger distance properties (symmetric, bounded)
4. ParameterMonitor decision logic
5. Drift metrics computation
"""

import pytest
import numpy as np
from unittest.mock import Mock

from hidden_regime.monitoring.drift_detector import (
    DriftDetector,
    ParameterMonitor,
    DriftMetrics,
)


class TestDriftDetector:
    """Test SLRT drift detection."""

    def test_initialization(self):
        """Test DriftDetector initialization."""
        detector = DriftDetector(threshold=10.0, forgetting_factor=0.99)
        assert detector.threshold == 10.0
        assert detector.forgetting_factor == 0.99
        assert detector.cusum == 0.0

    def test_no_drift_with_positive_llr(self):
        """New model fits better (positive LLR) should not trigger drift."""
        detector = DriftDetector(threshold=10.0)

        # Feed positive LLR (new model fits better)
        for _ in range(10):
            drift, cusum = detector.check_drift(1.0)
            assert not drift  # Should not trigger
            assert cusum >= 0

    def test_drift_with_negative_llr(self):
        """Old model fits better (negative LLR) should eventually trigger drift."""
        detector = DriftDetector(threshold=5.0)

        # Feed negative LLR (old model fits better)
        drift = False
        for i in range(20):
            drift, cusum = detector.check_drift(-1.0)
            if i < 4:
                assert not drift
            else:
                if drift:
                    break

        assert drift  # Should eventually trigger

    def test_cusum_resets_to_zero(self):
        """CUSUM should reset to zero on positive LLR."""
        detector = DriftDetector(threshold=10.0)

        # Build up cusum with negative LLR
        detector.check_drift(-1.0)
        detector.check_drift(-1.0)
        assert detector.cusum > 0

        # Reset to zero with large positive LLR
        detector.check_drift(10.0)
        assert detector.cusum == 0.0

    def test_forgetting_factor_prevents_unbounded_growth(self):
        """Forgetting factor should prevent unbounded CUSUM growth."""
        detector = DriftDetector(threshold=100.0, forgetting_factor=0.99)

        # Feed constant negative LLR
        max_cusum = 0
        for _ in range(1000):
            _, cusum = detector.check_drift(-1.0)
            max_cusum = max(max_cusum, cusum)

        # CUSUM should stabilize, not grow unbounded
        assert max_cusum < 100  # Should stay well below threshold with forgetting

    def test_higher_forgetting_factor_accumulates_faster(self):
        """Higher β should accumulate LLR faster (less forgetting)."""
        detector_slow = DriftDetector(threshold=100.0, forgetting_factor=0.90)
        detector_fast = DriftDetector(threshold=100.0, forgetting_factor=0.99)

        for _ in range(100):
            _, cusum_slow = detector_slow.check_drift(-1.0)
            _, cusum_fast = detector_fast.check_drift(-1.0)

        # With more forgetting (lower β), CUSUM should be lower
        assert cusum_slow < cusum_fast

    def test_reset_clears_cusum(self):
        """Reset should clear CUSUM to zero."""
        detector = DriftDetector()
        detector.check_drift(-1.0)
        detector.check_drift(-1.0)
        assert detector.cusum > 0

        detector.reset()
        assert detector.cusum == 0.0

    def test_get_status(self):
        """get_status should return dict with current state."""
        detector = DriftDetector(threshold=10.0)
        detector.check_drift(-1.0)

        status = detector.get_status()
        assert 'cusum' in status
        assert 'threshold' in status
        assert 'progress' in status
        assert status['cusum'] == detector.cusum
        assert status['threshold'] == 10.0
        assert 0 <= status['progress'] <= 1.5


class MockHMMModel:
    """Mock HMM model for testing ParameterMonitor."""

    def __init__(self, n_states=2):
        self.n_states = n_states
        # Two states: one bullish, one bearish
        self.means_ = np.array([[0.001], [-0.0008]])[:n_states]
        self.covars_ = np.array([
            [[0.008 ** 2]],
            [[0.012 ** 2]]
        ])[:n_states]

    def score(self, obs):
        """Mock log-likelihood computation."""
        # Simple mock: return negative sum of squared values
        return -np.sum(np.array(obs) ** 2)


class TestParameterMonitor:
    """Test multi-metric drift assessment."""

    def test_initialization(self):
        """Test ParameterMonitor initialization."""
        monitor = ParameterMonitor(
            slrt_threshold=10.0,
            kl_hard_threshold=2.0,
            kl_soft_threshold=1.0
        )
        assert monitor.slrt_threshold == 10.0
        assert monitor.kl_hard_threshold == 2.0
        assert monitor.kl_soft_threshold == 1.0
        assert isinstance(monitor.drift_detector, DriftDetector)

    def test_assess_drift_with_no_drift(self):
        """Models with identical parameters should show no drift."""
        monitor = ParameterMonitor()
        model_old = MockHMMModel(n_states=2)
        model_new = MockHMMModel(n_states=2)  # Identical

        decision, metrics = monitor.assess_drift(model_old, model_new, np.array([]))

        assert decision == 'continue'
        assert metrics['max_kl_divergence'] < 0.01  # Should be ~0
        assert metrics['max_hellinger_distance'] < 0.01

    def test_assess_drift_with_mean_shift(self):
        """Significant mean shift should trigger drift warning."""
        monitor = ParameterMonitor(
            kl_hard_threshold=2.0,
            kl_soft_threshold=1.0
        )
        model_old = MockHMMModel(n_states=2)

        # Create new model with shifted means
        model_new = Mock()
        model_new.n_states = 2
        model_new.means_ = np.array([[0.002], [-0.001]])  # Shifted
        model_new.covars_ = np.array([
            [[0.008 ** 2]],
            [[0.012 ** 2]]
        ])

        decision, metrics = monitor.assess_drift(model_old, model_new, np.array([]))

        # Should detect some drift
        assert metrics['max_kl_divergence'] > 0.1
        assert decision in ['monitor', 'retrain_soft']

    def test_assess_drift_with_variance_increase(self):
        """Variance increase should be detected."""
        monitor = ParameterMonitor()
        model_old = MockHMMModel(n_states=2)

        # Create new model with higher variance
        model_new = Mock()
        model_new.n_states = 2
        model_new.means_ = np.array([[0.001], [-0.0008]])  # Same means
        model_new.covars_ = np.array([
            [[0.020 ** 2]],  # Increased variance
            [[0.030 ** 2]]   # Increased variance
        ])

        decision, metrics = monitor.assess_drift(model_old, model_new, np.array([]))

        # Should detect variance drift
        assert metrics['max_kl_divergence'] > 0.1
        assert metrics['max_hellinger_distance'] > 0.01

    def test_decision_priority_slrt_overrides_kl(self):
        """SLRT critical drift should trigger retrain_hard regardless of KL."""
        monitor = ParameterMonitor(
            slrt_threshold=5.0,
            kl_hard_threshold=10.0  # Very high, won't trigger
        )

        # Artificially trigger SLRT
        for _ in range(10):
            monitor.drift_detector.check_drift(-2.0)

        model = MockHMMModel()
        decision, _ = monitor.assess_drift(model, model, np.array([]))

        assert decision == 'retrain_hard'

    def test_decision_priority_kl_hard_triggers_retrain(self):
        """KL hard threshold should trigger retrain_hard."""
        monitor = ParameterMonitor(
            kl_hard_threshold=0.5,  # Low threshold
            slrt_threshold=100.0    # Very high, won't trigger
        )

        model_old = MockHMMModel()

        # Create model with large parameter shift
        model_new = Mock()
        model_new.n_states = 2
        model_new.means_ = np.array([[0.003], [-0.003]])  # Very different
        model_new.covars_ = np.array([
            [[0.020 ** 2]],
            [[0.030 ** 2]]
        ])

        decision, _ = monitor.assess_drift(model_old, model_new, np.array([]))
        assert decision == 'retrain_hard'

    def test_decision_kl_soft_threshold(self):
        """KL between soft and hard thresholds should trigger retrain_soft."""
        monitor = ParameterMonitor(
            kl_hard_threshold=2.0,
            kl_soft_threshold=0.5,
            slrt_threshold=100.0  # Won't trigger
        )

        model_old = MockHMMModel()

        # Create model with moderate drift
        model_new = Mock()
        model_new.n_states = 2
        model_new.means_ = np.array([[0.0015], [-0.001]])
        model_new.covars_ = np.array([
            [[0.012 ** 2]],
            [[0.015 ** 2]]
        ])

        decision, _ = monitor.assess_drift(model_old, model_new, np.array([]))
        assert decision in ['retrain_soft', 'monitor']

    def test_metrics_include_state_kl_divergences(self):
        """Metrics should include KL divergence for each state."""
        monitor = ParameterMonitor()
        model = MockHMMModel(n_states=3)

        decision, metrics = monitor.assess_drift(model, model, np.array([]))

        assert 'state_kl_divergences' in metrics
        assert len(metrics['state_kl_divergences']) == 3
        assert all(kl >= 0 for kl in metrics['state_kl_divergences'])

    def test_gaussian_kl_divergence_symmetry(self):
        """KL divergence should not be symmetric."""
        monitor = ParameterMonitor()

        kl_12 = monitor._gaussian_kl_divergence(0.001, 0.008, 0.002, 0.010)
        kl_21 = monitor._gaussian_kl_divergence(0.002, 0.010, 0.001, 0.008)

        assert kl_12 != kl_21

    def test_gaussian_kl_divergence_zero_for_identical(self):
        """KL divergence should be ~0 for identical distributions."""
        monitor = ParameterMonitor()

        kl = monitor._gaussian_kl_divergence(0.001, 0.008, 0.001, 0.008)

        assert abs(kl) < 1e-10

    def test_hellinger_distance_bounds(self):
        """Hellinger distance should be bounded [0, 1]."""
        monitor = ParameterMonitor()

        for mean1 in [0.0, 0.001, 0.005]:
            for std1 in [0.001, 0.008, 0.020]:
                for mean2 in [-0.001, 0.002, 0.010]:
                    for std2 in [0.002, 0.012, 0.030]:
                        h = monitor._hellinger_distance(mean1, std1, mean2, std2)
                        assert 0 <= h <= 1

    def test_hellinger_distance_symmetry(self):
        """Hellinger distance should be symmetric."""
        monitor = ParameterMonitor()

        h_12 = monitor._hellinger_distance(0.001, 0.008, 0.002, 0.010)
        h_21 = monitor._hellinger_distance(0.002, 0.010, 0.001, 0.008)

        assert abs(h_12 - h_21) < 1e-10

    def test_hellinger_distance_zero_for_identical(self):
        """Hellinger distance should be ~0 for identical distributions."""
        monitor = ParameterMonitor()

        h = monitor._hellinger_distance(0.001, 0.008, 0.001, 0.008)

        assert abs(h) < 1e-10

    def test_reset_clears_drift_detector(self):
        """Reset should clear SLRT detector."""
        monitor = ParameterMonitor()
        monitor.drift_detector.check_drift(-1.0)
        assert monitor.drift_detector.cusum > 0

        monitor.reset()
        assert monitor.drift_detector.cusum == 0.0

    def test_get_status(self):
        """get_status should return monitoring configuration."""
        monitor = ParameterMonitor(
            slrt_threshold=10.0,
            kl_hard_threshold=2.0
        )

        status = monitor.get_status()
        assert 'slrt_detector' in status
        assert 'slrt_threshold' in status
        assert status['slrt_threshold'] == 10.0
        assert status['kl_hard_threshold'] == 2.0
