"""
Unit tests for retraining policy orchestration.

Tests verify:
1. Schedule adherence (correct day thresholds)
2. Drift trigger precedence (overrides schedule)
3. Min/max constraints enforced
4. Decision priority ordering
5. Error tracking for anomaly detection
"""

import pytest

from hidden_regime.monitoring.retraining_policy import (
    UpdateSchedule,
    RetrainingPolicy,
    create_moderate_policy,
    create_conservative_policy,
    create_aggressive_policy,
)


class TestUpdateSchedule:
    """Test UpdateSchedule configuration."""

    def test_default_schedule(self):
        """Test default schedule parameters."""
        schedule = UpdateSchedule()
        assert schedule.emission_frequency_days == 5
        assert schedule.transition_frequency_days == 21
        assert schedule.full_retrain_frequency_days == 63
        assert schedule.max_days_without_retrain == 90
        assert schedule.min_days_between_retrains == 14

    def test_custom_schedule(self):
        """Test custom schedule creation."""
        schedule = UpdateSchedule(
            emission_frequency_days=3,
            transition_frequency_days=10,
            full_retrain_frequency_days=30
        )
        assert schedule.emission_frequency_days == 3
        assert schedule.transition_frequency_days == 10


class TestRetrainingPolicy:
    """Test retraining policy decision logic."""

    def test_initialization(self):
        """Test policy initialization."""
        policy = RetrainingPolicy()
        assert policy.days_since_emission_update == 0
        assert policy.days_since_transition_update == 0
        assert policy.days_since_full_retrain == 0
        assert len(policy.recent_errors) == 0

    def test_no_update_initially(self):
        """No updates should be triggered initially."""
        policy = RetrainingPolicy()
        update_type, reason = policy.should_update('continue')
        assert update_type == 'none'

    def test_emission_update_scheduled(self):
        """Emission update should trigger after configured days."""
        schedule = UpdateSchedule(emission_frequency_days=5)
        policy = RetrainingPolicy(schedule)

        # Trigger 5 days of updates
        for i in range(5):
            update_type, _ = policy.should_update('continue')
            if i < 4:
                assert update_type == 'none'

        # On day 5 (index 4), emission update should trigger
        assert update_type == 'emission_only'

    def test_transition_update_scheduled(self):
        """Transition update should trigger after configured days."""
        schedule = UpdateSchedule(
            emission_frequency_days=100,  # Won't trigger
            transition_frequency_days=21
        )
        policy = RetrainingPolicy(schedule)

        # Trigger 21 days
        for i in range(21):
            update_type, _ = policy.should_update('continue')

        assert update_type == 'transition_only'

    def test_full_retrain_scheduled(self):
        """Full retrain should trigger after configured days."""
        schedule = UpdateSchedule(
            emission_frequency_days=100,
            transition_frequency_days=100,
            full_retrain_frequency_days=63
        )
        policy = RetrainingPolicy(schedule)

        # Trigger 63 days
        for i in range(63):
            update_type, _ = policy.should_update('continue')

        assert update_type == 'full_retrain'

    def test_max_days_constraint(self):
        """Hard constraint: max_days should trigger full retrain."""
        schedule = UpdateSchedule(
            emission_frequency_days=100,
            transition_frequency_days=100,
            full_retrain_frequency_days=200,  # Won't trigger normally
            max_days_without_retrain=90
        )
        policy = RetrainingPolicy(schedule)

        # After 90 days, should force full retrain
        for i in range(90):
            update_type, _ = policy.should_update('continue')

        assert update_type == 'full_retrain'

    def test_min_days_constraint(self):
        """Min constraint: should prevent too-frequent updates."""
        schedule = UpdateSchedule(
            emission_frequency_days=1,
            min_days_between_retrains=5
        )
        policy = RetrainingPolicy(schedule)

        # Trigger emission update (day 1)
        policy.should_update('continue')
        policy.record_emission_update()

        # Next few days should be blocked
        for _ in range(4):
            update_type, reason = policy.should_update('continue')
            assert update_type == 'none'
            assert 'min_days_not_met' in reason

        # After min_days, should allow next update
        for _ in range(5):
            update_type, _ = policy.should_update('continue')

        assert update_type != 'none'

    def test_drift_hard_triggers_full_retrain(self):
        """Critical drift should immediately trigger full retrain."""
        policy = RetrainingPolicy()

        # Even on day 0, hard drift should trigger retrain
        update_type, reason = policy.should_update('retrain_hard')
        assert update_type == 'full_retrain'
        assert 'critical_drift' in reason

    def test_drift_soft_with_time_triggers_retrain(self):
        """Soft drift + half retrain cycle should trigger retrain."""
        schedule = UpdateSchedule(full_retrain_frequency_days=60)
        policy = RetrainingPolicy(schedule)

        # Advance to halfway point
        for _ in range(30):
            policy.should_update('continue')

        # Soft drift at halfway should trigger retrain
        update_type, reason = policy.should_update('retrain_soft')
        assert update_type == 'full_retrain'
        assert 'soft_drift' in reason

    def test_record_emission_update(self):
        """Recording emission update should reset counters."""
        policy = RetrainingPolicy()
        policy.days_since_emission_update = 5
        policy.days_since_full_retrain = 10

        policy.record_emission_update()

        assert policy.days_since_emission_update == 0
        assert policy.days_since_full_retrain == 10  # Not reset

    def test_record_transition_update(self):
        """Recording transition update should reset counters."""
        policy = RetrainingPolicy()
        policy.days_since_transition_update = 21
        policy.days_since_full_retrain = 30

        policy.record_transition_update()

        assert policy.days_since_transition_update == 0
        assert policy.days_since_full_retrain == 30  # Not reset

    def test_record_full_retrain(self):
        """Recording full retrain should reset all counters."""
        policy = RetrainingPolicy()
        policy.days_since_emission_update = 5
        policy.days_since_transition_update = 21
        policy.days_since_full_retrain = 63

        policy.record_full_retrain()

        assert policy.days_since_emission_update == 0
        assert policy.days_since_transition_update == 0
        assert policy.days_since_full_retrain == 0

    def test_add_and_get_error_threshold(self):
        """Error tracking should compute threshold."""
        policy = RetrainingPolicy()

        # Add errors
        for i in range(20):
            policy.add_error(float(i))

        threshold = policy.get_error_threshold(quantile=0.95)
        assert threshold > 0
        assert threshold < float('inf')

    def test_insufficient_error_history(self):
        """Should return inf threshold with insufficient history."""
        policy = RetrainingPolicy()

        policy.add_error(1.0)
        policy.add_error(2.0)

        threshold = policy.get_error_threshold()
        assert threshold == float('inf')

    def test_get_status(self):
        """Status should reflect current state."""
        policy = RetrainingPolicy()
        policy.days_since_emission_update = 3
        policy.days_since_full_retrain = 10

        status = policy.get_status()
        assert status['days_since_emission_update'] == 3
        assert status['days_since_full_retrain'] == 10
        assert 'emission_threshold' in status
        assert 'max_days_without_retrain' in status

    def test_get_update_history(self):
        """Should track decision history."""
        policy = RetrainingPolicy()

        for _ in range(10):
            policy.should_update('continue')

        history = policy.get_update_history(last_n=5)
        assert len(history) <= 5
        assert all('update_type' in record for record in history)
        assert all('reason' in record for record in history)


class TestPolicyFactories:
    """Test factory functions for preconfigured policies."""

    def test_moderate_policy(self):
        """Moderate policy should have balanced settings."""
        policy = create_moderate_policy()
        assert policy.schedule.full_retrain_frequency_days == 63
        assert policy.schedule.max_days_without_retrain == 90
        assert policy.schedule.min_days_between_retrains == 14

    def test_conservative_policy(self):
        """Conservative policy should have longer cycles."""
        policy = create_conservative_policy()
        assert policy.schedule.full_retrain_frequency_days == 180
        assert policy.schedule.max_days_without_retrain == 180
        assert policy.schedule.min_days_between_retrains == 30

    def test_aggressive_policy(self):
        """Aggressive policy should have shorter cycles."""
        policy = create_aggressive_policy()
        assert policy.schedule.full_retrain_frequency_days == 30
        assert policy.schedule.max_days_without_retrain == 30
        assert policy.schedule.min_days_between_retrains == 7

    def test_policies_have_different_schedules(self):
        """Different policies should have different schedules."""
        moderate = create_moderate_policy()
        conservative = create_conservative_policy()
        aggressive = create_aggressive_policy()

        # Full retrain frequencies should differ
        assert moderate.schedule.full_retrain_frequency_days != conservative.schedule.full_retrain_frequency_days
        assert conservative.schedule.full_retrain_frequency_days != aggressive.schedule.full_retrain_frequency_days
        assert aggressive.schedule.full_retrain_frequency_days != moderate.schedule.full_retrain_frequency_days


class TestDecisionPriority:
    """Test decision priority ordering."""

    def test_max_days_has_highest_priority(self):
        """Max_days constraint should override drift."""
        schedule = UpdateSchedule(
            max_days_without_retrain=90,
            min_days_between_retrains=0
        )
        policy = RetrainingPolicy(schedule)

        # Advance past max_days
        for _ in range(90):
            policy.should_update('continue')

        # Should trigger full retrain even without drift
        update_type, reason = policy.should_update('continue')
        assert update_type == 'full_retrain'
        assert 'max_days' in reason

    def test_min_days_blocks_before_checking_drift(self):
        """Min_days constraint should block before drift check."""
        schedule = UpdateSchedule(
            emission_frequency_days=1,
            min_days_between_retrains=5
        )
        policy = RetrainingPolicy(schedule)

        # Trigger emission update
        policy.should_update('continue')
        policy.record_emission_update()

        # Try critical drift on day 1
        update_type, reason = policy.should_update('retrain_hard')
        assert update_type == 'none'
        assert 'min_days' in reason

    def test_hard_drift_overrides_schedule(self):
        """Hard drift should override schedule."""
        schedule = UpdateSchedule(
            full_retrain_frequency_days=100
        )
        policy = RetrainingPolicy(schedule)

        # On day 1, no scheduled update
        update_type, _ = policy.should_update('continue')
        assert update_type == 'none'

        # But hard drift should trigger immediately (after min_days)
        policy.days_since_last_update = 100  # Bypass min_days
        update_type, _ = policy.should_update('retrain_hard')
        assert update_type == 'full_retrain'
