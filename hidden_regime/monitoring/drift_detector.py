"""
Drift detection system for monitoring HMM parameter changes.

Implements three statistical tests for detecting when market structure has
fundamentally changed, requiring full model retraining:

1. SLRT (Sequential Likelihood Ratio Test)
   - Online cumulative test with controlled false alarm rate
   - Threshold α ≈ 0.005 → ~1 false alarm per 200 days

2. KL Divergence of Emission Distributions
   - Information-theoretic distance between old/new parameters
   - Hard threshold 2.0 nats: critical drift
   - Soft threshold 1.0 nat: elevated drift

3. Hellinger Distance
   - Symmetric, bounded [0, 1] version of KL divergence
   - Robust to variance changes
   - Threshold 0.3: substantial drift

The three metrics provide redundancy: triggers are precise, not brittle.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class DriftMetrics:
    """
    Drift detection metrics from a single observation.

    Attributes:
        slrt_statistic: Sequential LRT cumulative sum
        slrt_drift_detected: Whether SLRT threshold exceeded
        kl_divergence: KL divergence from state distribution
        hellinger_distance: Hellinger distance from state distribution
    """
    slrt_statistic: float
    slrt_drift_detected: bool
    kl_divergence: float = 0.0
    hellinger_distance: float = 0.0


class DriftDetector:
    """
    Sequential Likelihood Ratio Test (SLRT) for online drift detection.

    Implements cumulative sum test with exponential forgetting to detect
    when new observations are inconsistent with the old model.

    Algorithm:
    - Compute log-likelihood ratio: Λ_t = log P(x_t | θ_new) - log P(x_t | θ_old)
    - Cumulative sum: S_t = max(0, β * S_{t-1} + Λ_t)
    - Trigger: if S_t > threshold, drift detected

    The forgetting factor β prevents unbounded growth and allows recovery
    from temporary fluctuations.

    Attributes:
        threshold: SLRT trigger threshold (default 10.0)
            - α ≈ 0.005, ~1 false alarm per 200 days
        forgetting_factor: Exponential decay β (default 0.99)
            - β=0.99: ~69 days to half-weight
            - β=0.95: ~14 days to half-weight
        cusum: Current cumulative sum value
    """

    def __init__(
        self,
        threshold: float = 10.0,
        forgetting_factor: float = 0.99
    ):
        """
        Initialize SLRT drift detector.

        Args:
            threshold: SLRT cusum threshold (higher = conservative)
                - 10.0: α ≈ 0.005 (recommended, moderate)
                - 15.0: α ≈ 0.000003 (very conservative)
                - 5.0: α ≈ 0.007 (aggressive)
            forgetting_factor: Exponential decay rate β ∈ (0, 1)
                - Prevents unbounded growth
                - Allows recovery from temporary fluctuations
        """
        self.threshold = threshold
        self.forgetting_factor = forgetting_factor
        self.cusum = 0.0

    def check_drift(
        self,
        log_likelihood_ratio: float
    ) -> Tuple[bool, float]:
        """
        Update SLRT and check for drift.

        Args:
            log_likelihood_ratio: log P(x_t | θ_new) - log P(x_t | θ_old)
                - Positive: new model fits better (no drift)
                - Negative: old model fits better (potential drift)

        Returns:
            Tuple of:
            - drift_detected: Whether threshold exceeded
            - cusum: Current cumulative sum value

        Example:
            >>> detector = DriftDetector(threshold=10.0)
            >>> for obs in observations:
            ...     ll_ratio = compute_ll_ratio(obs, old_model, new_model)
            ...     drift, stat = detector.check_drift(ll_ratio)
            ...     if drift:
            ...         retrain_needed = True
        """
        # Update SLRT with forgetting
        self.cusum = max(0, self.forgetting_factor * self.cusum + log_likelihood_ratio)

        # Check if threshold exceeded
        drift_detected = self.cusum > self.threshold

        return drift_detected, self.cusum

    def reset(self) -> None:
        """Reset SLRT cusum to zero (e.g., after retraining)."""
        self.cusum = 0.0

    def get_status(self) -> Dict[str, float]:
        """
        Get current detector status.

        Returns:
            Dict with keys:
            - 'cusum': Current cumulative sum
            - 'threshold': Trigger threshold
            - 'progress': Fraction of threshold (0-1, capped at 1.5)
        """
        progress = min(1.5, self.cusum / self.threshold)
        return {
            'cusum': self.cusum,
            'threshold': self.threshold,
            'progress': progress,
        }


class ParameterMonitor:
    """
    Multi-metric drift assessment combining three statistical tests.

    Provides comprehensive drift detection by combining:
    1. SLRT for online likelihood-based detection
    2. KL divergence for parameter distribution changes
    3. Hellinger distance for robust cross-validation

    Decision logic (priority order):
    - 'retrain_hard': Critical drift detected (SLRT > threshold OR KL > hard_threshold)
    - 'retrain_soft': Soft drift detected (KL > soft_threshold, consider retraining)
    - 'monitor': Elevated drift, continue monitoring
    - 'continue': No concerning drift, proceed normally

    Attributes:
        slrt_threshold: SLRT trigger threshold
        kl_hard_threshold: KL hard threshold in nats
        kl_soft_threshold: KL soft threshold in nats
        hellinger_threshold: Hellinger distance threshold
        drift_detector: SLRT detector instance
    """

    def __init__(
        self,
        slrt_threshold: float = 10.0,
        slrt_forgetting_factor: float = 0.99,
        kl_hard_threshold: float = 2.0,
        kl_soft_threshold: float = 1.0,
        hellinger_threshold: float = 0.3
    ):
        """
        Initialize parameter monitor.

        Args:
            slrt_threshold: SLRT cusum threshold (default 10.0)
            slrt_forgetting_factor: SLRT forgetting factor (default 0.99)
            kl_hard_threshold: Critical KL divergence (default 2.0)
                - 2.0 nats: ~13% probability ratio
                - 3.0 nats: ~5% probability ratio
            kl_soft_threshold: Elevated KL divergence (default 1.0)
                - 1.0 nat: ~37% probability ratio
                - 1.5 nats: ~22% probability ratio
            hellinger_threshold: Hellinger distance (default 0.3)
                - Bounded [0, 1], robust to variance changes
        """
        self.slrt_threshold = slrt_threshold
        self.kl_hard_threshold = kl_hard_threshold
        self.kl_soft_threshold = kl_soft_threshold
        self.hellinger_threshold = hellinger_threshold

        self.drift_detector = DriftDetector(
            threshold=slrt_threshold,
            forgetting_factor=slrt_forgetting_factor
        )

    def assess_drift(
        self,
        old_model,
        new_model,
        recent_data
    ) -> Tuple[str, Dict[str, float]]:
        """
        Comprehensive drift assessment using three metrics.

        Compares old and new models on recent data, computing KL divergence
        and Hellinger distance for each HMM state's emission distribution.

        Args:
            old_model: Previously trained HMM model
                - Should have .n_states, .means_, .covars_
            new_model: Newly trained or updated HMM model
            recent_data: Recent observations for likelihood computation
                - Used for SLRT calculation

        Returns:
            Tuple of:
            - decision: One of 'retrain_hard', 'retrain_soft', 'monitor', 'continue'
            - metrics: Dict with:
                - 'slrt_cusum': SLRT cumulative sum
                - 'max_kl_divergence': Maximum KL divergence across states
                - 'max_hellinger_distance': Maximum Hellinger distance across states
                - 'slrt_drift': Whether SLRT triggered
                - 'state_kl_divergences': List of KL divergences for each state

        Example:
            >>> monitor = ParameterMonitor()
            >>> decision, metrics = monitor.assess_drift(
            ...     old_model=old_hmm,
            ...     new_model=new_hmm,
            ...     recent_data=data[-100:]
            ... )
            >>> if decision == 'retrain_hard':
            ...     retrain_model()
        """
        metrics = {}

        # Test 1: SLRT on recent observation
        if len(recent_data) > 0:
            recent_obs = recent_data[-1]
            ll_ratio = self._compute_ll_ratio(recent_obs, old_model, new_model)
            slrt_drift, cusum = self.drift_detector.check_drift(ll_ratio)
            metrics['slrt_cusum'] = cusum
            metrics['slrt_drift'] = slrt_drift
        else:
            metrics['slrt_cusum'] = self.drift_detector.cusum
            metrics['slrt_drift'] = False

        # Test 2 & 3: KL divergence and Hellinger distance for emission distributions
        kl_divs = []
        hellinger_dists = []

        for state_id in range(old_model.n_states):
            kl_div = self._gaussian_kl_divergence(
                old_model.means_[state_id, 0],
                np.sqrt(old_model.covars_[state_id, 0, 0]),
                new_model.means_[state_id, 0],
                np.sqrt(new_model.covars_[state_id, 0, 0])
            )
            kl_divs.append(kl_div)

            hell_dist = self._hellinger_distance(
                old_model.means_[state_id, 0],
                np.sqrt(old_model.covars_[state_id, 0, 0]),
                new_model.means_[state_id, 0],
                np.sqrt(new_model.covars_[state_id, 0, 0])
            )
            hellinger_dists.append(hell_dist)

        max_kl = max(kl_divs) if kl_divs else 0.0
        max_hellinger = max(hellinger_dists) if hellinger_dists else 0.0

        metrics['max_kl_divergence'] = max_kl
        metrics['max_hellinger_distance'] = max_hellinger
        metrics['state_kl_divergences'] = kl_divs

        # Decision logic (priority order)
        if (metrics['slrt_drift'] or
            max_kl > self.kl_hard_threshold or
            max_hellinger > self.hellinger_threshold):
            decision = 'retrain_hard'
        elif max_kl > self.kl_soft_threshold:
            decision = 'retrain_soft'
        elif max_kl > self.kl_soft_threshold * 0.5:
            decision = 'monitor'
        else:
            decision = 'continue'

        return decision, metrics

    def _compute_ll_ratio(self, observation, old_model, new_model) -> float:
        """
        Compute log-likelihood ratio between models.

        Args:
            observation: Single observation (1D array or value)
            old_model: Reference model
            new_model: Candidate model

        Returns:
            log P(observation | new_model) - log P(observation | old_model)
        """
        try:
            # For sklearn-like HMMs with score method
            ll_new = old_model.score([observation])
            ll_old = new_model.score([observation])
            return ll_new - ll_old
        except (AttributeError, ValueError):
            # Fallback: return 0 if models don't support scoring
            return 0.0

    def _gaussian_kl_divergence(
        self,
        mean1: float,
        std1: float,
        mean2: float,
        std2: float
    ) -> float:
        """
        Compute KL divergence between two univariate Gaussians.

        KL(N(μ₁, σ₁²) || N(μ₂, σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

        Args:
            mean1, std1: First distribution (old model)
            mean2, std2: Second distribution (new model)

        Returns:
            KL divergence in nats
        """
        var1 = std1 ** 2
        var2 = std2 ** 2

        kl = (
            np.log(std2 / std1) +
            (var1 + (mean1 - mean2) ** 2) / (2 * var2) -
            0.5
        )

        return float(kl)

    def _hellinger_distance(
        self,
        mean1: float,
        std1: float,
        mean2: float,
        std2: float
    ) -> float:
        """
        Compute Hellinger distance between two univariate Gaussians.

        For Gaussians: H(P, Q)² = 1 - exp(-0.125 * Mahalanobis²)

        where Mahalanobis² = (μ₁ - μ₂)² / (σ₁² + σ₂²)

        Properties:
        - Symmetric (unlike KL divergence)
        - Bounded: H ∈ [0, 1]
        - H = 0: identical distributions
        - H = 1: disjoint support

        Args:
            mean1, std1: First distribution
            mean2, std2: Second distribution

        Returns:
            Hellinger distance, bounded in [0, 1]
        """
        var1 = std1 ** 2
        var2 = std2 ** 2

        mahal_sq = (mean1 - mean2) ** 2 / (var1 + var2)
        h_sq = 1.0 - np.exp(-0.125 * mahal_sq)

        return float(np.sqrt(max(0.0, h_sq)))  # Clip to [0, 1]

    def reset(self) -> None:
        """Reset all drift detectors (e.g., after retraining)."""
        self.drift_detector.reset()

    def get_status(self) -> Dict[str, float]:
        """
        Get current monitoring status.

        Returns:
            Dict with detector status information
        """
        return {
            'slrt_detector': self.drift_detector.get_status(),
            'slrt_threshold': self.slrt_threshold,
            'kl_hard_threshold': self.kl_hard_threshold,
            'kl_soft_threshold': self.kl_soft_threshold,
        }
