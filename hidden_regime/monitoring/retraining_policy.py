"""
Retraining policy engine that orchestrates when and how often to update HMM.

Implements hierarchical update strategy:
- Weekly (day 5): Emission-only updates (~1% cost)
- Monthly (day 21): Transition-only updates (~5% cost)
- Quarterly (day 63): Full Baum-Welch retrains (~100% cost)

Plus drift-triggered retrains when market structure fundamentally changes.

Decision logic prioritizes:
1. Hard constraints (max_days exceeded)
2. Drift triggers (SLRT, KL divergence)
3. Time-based schedule
4. Constraint enforcement (min_days, max_days)
"""

from dataclasses import dataclass
from typing import Dict, Literal, Tuple
from collections import deque


@dataclass
class UpdateSchedule:
    """
    Update schedule configuration.

    Attributes:
        emission_frequency_days: Days between emission-only updates
        transition_frequency_days: Days between transition-only updates
        full_retrain_frequency_days: Days between full retrains
        max_days_without_retrain: Hard ceiling for full retrain
        min_days_between_retrains: Minimum days between any updates
    """
    emission_frequency_days: int = 5
    transition_frequency_days: int = 21
    full_retrain_frequency_days: int = 63
    max_days_without_retrain: int = 90
    min_days_between_retrains: int = 14


class RetrainingPolicy:
    """
    Orchestrates HMM update schedule with drift-triggered overrides.

    Maintains state about when each update type was last performed, evaluates
    drift metrics, and decides what action to take. Prevents thrashing through
    min_days constraint.

    Attributes:
        schedule: UpdateSchedule configuration
        days_since_emission_update: Days since last emission-only update
        days_since_transition_update: Days since last transition-only update
        days_since_full_retrain: Days since last full retrain
        recent_errors: Deque of recent prediction errors for anomaly detection
    """

    def __init__(self, schedule: UpdateSchedule = None):
        """
        Initialize retraining policy.

        Args:
            schedule: UpdateSchedule configuration
                If None, uses default Moderate schedule
        """
        self.schedule = schedule or UpdateSchedule()

        # Tracking variables
        self.days_since_emission_update = 0
        self.days_since_transition_update = 0
        self.days_since_full_retrain = 0
        self.days_since_last_update = 0

        # Error history for prediction quality monitoring
        self.recent_errors = deque(maxlen=20)

        # Audit trail
        self.update_history = []

    def should_update(
        self,
        drift_decision: str,
        drift_metrics: Dict = None
    ) -> Tuple[str, str]:
        """
        Determine what type of update to perform.

        Decision logic (priority order):
        1. Hard constraint: max_days exceeded → full retrain
        2. Minimum constraint: min_days not met → no update
        3. Critical drift (hard triggers) → full retrain immediately
        4. Soft drift + time → full retrain
        5. Scheduled updates → emission/transition/full retrain

        Args:
            drift_decision: One of 'retrain_hard', 'retrain_soft', 'monitor', 'continue'
            drift_metrics: Dict with drift metric values (optional)

        Returns:
            Tuple of:
            - update_type: One of 'emission_only', 'transition_only', 'full_retrain', 'none'
            - reason: String explaining the decision

        Example:
            >>> policy = RetrainingPolicy()
            >>> decision, reason = policy.should_update('continue')
            >>> print(f"Update: {decision} because {reason}")
        """
        # Increment counters
        self._tick_all_counters()

        # Hard constraint: max_days exceeded
        if self.days_since_full_retrain >= self.schedule.max_days_without_retrain:
            self._record_decision('full_retrain', 'max_days_exceeded')
            return 'full_retrain', 'max_days_exceeded'

        # Minimum constraint: too soon to update
        if self.days_since_last_update < self.schedule.min_days_between_retrains:
            self._record_decision('none', f'min_days_not_met ({self.days_since_last_update}d < {self.schedule.min_days_between_retrains}d)')
            return 'none', f'min_days_not_met ({self.days_since_last_update}d < {self.schedule.min_days_between_retrains}d)'

        # Critical drift detected
        if drift_decision == 'retrain_hard':
            self._record_decision('full_retrain', 'critical_drift_detected')
            return 'full_retrain', 'critical_drift_detected'

        # Soft drift + time passed
        if (drift_decision == 'retrain_soft' and
            self.days_since_full_retrain >= self.schedule.full_retrain_frequency_days / 2):
            self._record_decision('full_retrain', 'soft_drift_with_time')
            return 'full_retrain', 'soft_drift_with_time'

        # Scheduled updates (time-based)
        if self.days_since_full_retrain >= self.schedule.full_retrain_frequency_days:
            self._record_decision('full_retrain', 'scheduled_quarterly')
            return 'full_retrain', 'scheduled_quarterly'

        if self.days_since_transition_update >= self.schedule.transition_frequency_days:
            self._record_decision('transition_only', 'scheduled_monthly')
            return 'transition_only', 'scheduled_monthly'

        if self.days_since_emission_update >= self.schedule.emission_frequency_days:
            self._record_decision('emission_only', 'scheduled_weekly')
            return 'emission_only', 'scheduled_weekly'

        # No update needed
        self._record_decision('none', drift_decision)
        return 'none', drift_decision

    def record_emission_update(self) -> None:
        """Record that an emission-only update was performed."""
        self.days_since_emission_update = 0
        self.days_since_last_update = 0

    def record_transition_update(self) -> None:
        """Record that a transition-only update was performed."""
        self.days_since_transition_update = 0
        self.days_since_last_update = 0

    def record_full_retrain(self) -> None:
        """Record that a full retrain was performed."""
        self.days_since_emission_update = 0
        self.days_since_transition_update = 0
        self.days_since_full_retrain = 0
        self.days_since_last_update = 0

    def add_error(self, error: float) -> None:
        """
        Record a prediction error for anomaly detection.

        Args:
            error: Prediction error (e.g., negative log-likelihood)
        """
        self.recent_errors.append(error)

    def get_error_threshold(self, quantile: float = 0.95) -> float:
        """
        Get error threshold for anomaly detection.

        Args:
            quantile: Quantile to use for threshold (default 0.95)

        Returns:
            Error threshold (returns inf if insufficient history)
        """
        if len(self.recent_errors) < 10:
            return float('inf')

        import numpy as np
        return float(np.quantile(list(self.recent_errors), quantile))

    def _tick_all_counters(self) -> None:
        """Increment all day counters."""
        self.days_since_emission_update += 1
        self.days_since_transition_update += 1
        self.days_since_full_retrain += 1
        self.days_since_last_update += 1

    def _record_decision(self, update_type: str, reason: str) -> None:
        """Record decision in audit trail."""
        self.update_history.append({
            'update_type': update_type,
            'reason': reason,
            'days_since_emission': self.days_since_emission_update,
            'days_since_transition': self.days_since_transition_update,
            'days_since_full': self.days_since_full_retrain,
        })

    def get_status(self) -> Dict:
        """
        Get current policy status.

        Returns:
            Dict with current state of all counters and thresholds
        """
        return {
            'days_since_emission_update': self.days_since_emission_update,
            'days_since_transition_update': self.days_since_transition_update,
            'days_since_full_retrain': self.days_since_full_retrain,
            'days_since_last_update': self.days_since_last_update,
            'emission_threshold': self.schedule.emission_frequency_days,
            'transition_threshold': self.schedule.transition_frequency_days,
            'full_retrain_threshold': self.schedule.full_retrain_frequency_days,
            'max_days_without_retrain': self.schedule.max_days_without_retrain,
            'min_days_between_updates': self.schedule.min_days_between_retrains,
            'recent_errors_count': len(self.recent_errors),
        }

    def get_update_history(self, last_n: int = 10) -> list:
        """
        Get recent update decisions.

        Args:
            last_n: Number of recent decisions to return

        Returns:
            List of recent decision records
        """
        return list(self.update_history[-last_n:])

    def print_status(self) -> None:
        """Print current policy status in human-readable format."""
        status = self.get_status()
        print("Retraining Policy Status")
        print("=" * 60)
        print(f"Days since emission update:    {status['days_since_emission_update']:3d} / {status['emission_threshold']:3d}")
        print(f"Days since transition update:  {status['days_since_transition_update']:3d} / {status['transition_threshold']:3d}")
        print(f"Days since full retrain:       {status['days_since_full_retrain']:3d} / {status['full_retrain_threshold']:3d}")
        print(f"Days since last any update:    {status['days_since_last_update']:3d}")
        print(f"Max days without retrain:      {status['max_days_without_retrain']:3d}")
        print(f"Min days between updates:      {status['min_days_between_updates']:3d}")
        print(f"Recent errors recorded:        {status['recent_errors_count']:3d}")
        print("=" * 60)


# Factory functions for creating preconfigured policies

def create_moderate_policy() -> RetrainingPolicy:
    """
    Create Moderate policy (RECOMMENDED DEFAULT).

    90-day full retrain cycle with moderate drift thresholds.
    Good balance between adaptation and stability.

    Returns:
        RetrainingPolicy configured for moderate use
    """
    schedule = UpdateSchedule(
        emission_frequency_days=5,
        transition_frequency_days=21,
        full_retrain_frequency_days=63,
        max_days_without_retrain=90,
        min_days_between_retrains=14,
    )
    return RetrainingPolicy(schedule)


def create_conservative_policy() -> RetrainingPolicy:
    """
    Create Conservative policy.

    For regulatory/compliance: 180-day full retrain cycle, higher thresholds.
    Minimizes disruption but slower adaptation.

    Returns:
        RetrainingPolicy configured for conservative use
    """
    schedule = UpdateSchedule(
        emission_frequency_days=10,
        transition_frequency_days=42,
        full_retrain_frequency_days=180,
        max_days_without_retrain=180,
        min_days_between_retrains=30,
    )
    return RetrainingPolicy(schedule)


def create_aggressive_policy() -> RetrainingPolicy:
    """
    Create Aggressive policy.

    For high-frequency trading: 30-day full retrain cycle.
    Rapid updates but may cause label changes.

    Returns:
        RetrainingPolicy configured for aggressive use
    """
    schedule = UpdateSchedule(
        emission_frequency_days=2,
        transition_frequency_days=7,
        full_retrain_frequency_days=30,
        max_days_without_retrain=30,
        min_days_between_retrains=7,
    )
    return RetrainingPolicy(schedule)
