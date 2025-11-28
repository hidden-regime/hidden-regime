"""
Anchored regime interpretation for stable regime labels despite model parameter drift.

This module provides the AnchoredInterpreter class, which maintains stable regime
labels (BULLISH, BEARISH, etc.) even as HMM parameters evolve through online
updates. It accomplishes this by anchoring regime definitions to slowly-evolving
template distributions, and assigning HMM states to regimes based on minimum
KL divergence to these anchors.

Statistical Justification:
- Separates HMM inference uncertainty (fast, parameter-dependent)
  from regime interpretation stability (slow, anchor-dependent)
- Anchors update via exponential smoothing (α=0.01 → 69 days to half-adapt)
- KL divergence-based matching is information-theoretically principled
- Maintains high mutual information I(regime_label; future_returns)

Key Insight:
Without anchoring, online parameter updates cause regime labels to flip
unexpectedly, destroying user confidence. With anchoring, users see stable
labels while the model adapts underneath.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np
from scipy.special import kl_div


@dataclass
class RegimeAnchor:
    """
    Statistical template for a regime type.

    Attributes:
        mean: Mean daily return (e.g., 0.001 for +0.1% daily)
        std: Daily volatility (standard deviation of returns)
    """
    mean: float
    std: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {'mean': self.mean, 'std': self.std}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'RegimeAnchor':
        """Create from dictionary representation."""
        return cls(mean=data['mean'], std=data['std'])


class AnchoredInterpreter:
    """
    Interprets HMM states as financial regimes using slowly-adapting anchors.

    Maintains stable regime labels by:
    1. Defining regime anchors (templates: mean/std distribution for each regime)
    2. Mapping HMM states to regimes by minimum KL divergence to anchors
    3. Slowly updating anchors (exponential smoothing, α=0.01)

    Result: Regime labels remain stable despite parameter drift, while anchors
    eventually adapt to true market structure changes.

    Attributes:
        anchor_update_rate: Exponential smoothing coefficient (default 0.01)
            - α=0.01 → 69 days to half-adapt
            - α=0.05 → 14 days to half-adapt (more responsive)
            - α=0.001 → 690 days to half-adapt (very stable)
        regime_anchors: Dict[str, RegimeAnchor] mapping regime names to templates
    """

    def __init__(
        self,
        anchor_update_rate: float = 0.01,
        regime_anchors: Dict[str, Dict[str, float]] = None
    ):
        """
        Initialize anchored interpreter.

        Args:
            anchor_update_rate: Exponential smoothing coefficient α
                - Controls how fast anchors adapt to observed parameters
                - Default 0.01 means 69-day half-life (slow, stable)
            regime_anchors: Dict mapping regime names to {'mean': float, 'std': float}
                - If None, uses default anchors (BULLISH, BEARISH, SIDEWAYS, CRISIS)
                - Values are daily returns (e.g., 0.001 = +0.1% daily)
        """
        self.anchor_update_rate = anchor_update_rate

        # Initialize regime anchors (default values)
        if regime_anchors is None:
            self.regime_anchors = {
                'BULLISH': RegimeAnchor(mean=0.0010, std=0.008),
                'BEARISH': RegimeAnchor(mean=-0.0008, std=0.012),
                'SIDEWAYS': RegimeAnchor(mean=0.0001, std=0.006),
                'CRISIS': RegimeAnchor(mean=-0.0030, std=0.025),
            }
        else:
            self.regime_anchors = {
                name: RegimeAnchor.from_dict(params)
                for name, params in regime_anchors.items()
            }

        # Audit trail for anchor updates
        self.anchor_update_history: Dict[str, list] = {
            regime: [] for regime in self.regime_anchors.keys()
        }

    def interpret_states(
        self,
        hmm_model
    ) -> Tuple[Dict[int, str], Dict[int, float]]:
        """
        Map HMM states to regime labels based on KL divergence to anchors.

        Algorithm:
        1. For each HMM state, extract emission parameters (mean, std)
        2. Compute KL divergence from state distribution to each regime anchor
        3. Assign state to regime with minimum KL divergence
        4. Confidence = 1 / (1 + KL_distance) [bounded in (0, 1)]
        5. Update anchors slowly via exponential smoothing

        Args:
            hmm_model: Fitted HiddenMarkovModel instance
                - Should have .n_states, .means_, .covars_

        Returns:
            Tuple of:
            - state_to_regime: Dict[state_id] → regime_name (e.g., 'BULLISH')
            - interpretation_confidence: Dict[state_id] → confidence_score (0-1)

        Examples:
            >>> state_to_regime, confidence = interpreter.interpret_states(model)
            >>> regime = state_to_regime[current_state]  # e.g., 'BULLISH'
            >>> conf = confidence[current_state]          # e.g., 0.92
        """
        state_to_regime = {}
        interpretation_confidence = {}

        for state_id in range(hmm_model.n_states):
            # Extract state's emission parameters
            state_mean = hmm_model.means_[state_id, 0]
            state_std = np.sqrt(hmm_model.covars_[state_id, 0, 0])

            # Compute KL divergence to each regime anchor
            kl_scores = {}
            for regime_name, anchor in self.regime_anchors.items():
                kl_dist = self._gaussian_kl_divergence(
                    mean1=state_mean,
                    std1=state_std,
                    mean2=anchor.mean,
                    std2=anchor.std
                )
                kl_scores[regime_name] = kl_dist

            # Assign to regime with minimum KL divergence
            best_regime = min(kl_scores, key=kl_scores.get)
            best_kl = kl_scores[best_regime]

            state_to_regime[state_id] = best_regime

            # Confidence: inverse of KL distance
            # Higher KL → lower confidence. Bounded in (0, 1) via logistic
            interpretation_confidence[state_id] = 1.0 / (1.0 + best_kl)

            # Update anchor slowly toward observed state
            self._update_anchor(
                best_regime,
                state_mean=state_mean,
                state_std=state_std
            )

        return state_to_regime, interpretation_confidence

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

        This measures how much information is lost when using the second
        distribution (μ₂, σ₂) to approximate the first (μ₁, σ₁).

        Args:
            mean1, std1: First distribution (state parameters)
            mean2, std2: Second distribution (anchor parameters)

        Returns:
            KL divergence in nats (base-e logarithms)
            - KL = 0: distributions identical
            - KL = 1: ~37% probability ratio
            - KL = 2: ~13% probability ratio
        """
        var1 = std1 ** 2
        var2 = std2 ** 2

        kl = (
            np.log(std2 / std1) +
            (var1 + (mean1 - mean2) ** 2) / (2 * var2) -
            0.5
        )

        return float(kl)

    def _update_anchor(
        self,
        regime_name: str,
        state_mean: float,
        state_std: float
    ) -> None:
        """
        Update regime anchor using exponential smoothing.

        Formula:
            anchor_new = (1 - α) * anchor_old + α * state_observed

        where α = anchor_update_rate (default 0.01)

        Interpretation:
        - Small α (0.01): slow adaptation, stable labels (~69 day half-life)
        - Large α (0.1): fast adaptation, responsive to changes (~7 day half-life)

        Args:
            regime_name: Regime to update (e.g., 'BULLISH')
            state_mean: Observed mean from HMM state
            state_std: Observed std from HMM state
        """
        anchor = self.regime_anchors[regime_name]

        # Exponential smoothing update
        anchor.mean = (
            (1 - self.anchor_update_rate) * anchor.mean +
            self.anchor_update_rate * state_mean
        )
        anchor.std = (
            (1 - self.anchor_update_rate) * anchor.std +
            self.anchor_update_rate * state_std
        )

        # Log update for audit trail
        self.anchor_update_history[regime_name].append({
            'mean': anchor.mean,
            'std': anchor.std,
            'observed_mean': state_mean,
            'observed_std': state_std,
        })

    def get_anchors(self) -> Dict[str, Dict[str, float]]:
        """
        Get current regime anchors as dictionary.

        Returns:
            Dict mapping regime names to {'mean': float, 'std': float}

        Example:
            >>> anchors = interpreter.get_anchors()
            >>> anchors['BULLISH']  # {'mean': 0.00102, 'std': 0.0081}
        """
        return {
            name: anchor.to_dict()
            for name, anchor in self.regime_anchors.items()
        }

    def set_anchors(self, regime_anchors: Dict[str, Dict[str, float]]) -> None:
        """
        Set regime anchors from dictionary (e.g., from saved configuration).

        Args:
            regime_anchors: Dict mapping regime names to {'mean': float, 'std': float}

        Example:
            >>> interpreter.set_anchors({
            ...     'BULLISH': {'mean': 0.001, 'std': 0.008},
            ...     'BEARISH': {'mean': -0.0008, 'std': 0.012},
            ... })
        """
        self.regime_anchors = {
            name: RegimeAnchor.from_dict(params)
            for name, params in regime_anchors.items()
        }

    def reset_anchors_to_defaults(self) -> None:
        """Reset anchors to canonical default values."""
        self.regime_anchors = {
            'BULLISH': RegimeAnchor(mean=0.0010, std=0.008),
            'BEARISH': RegimeAnchor(mean=-0.0008, std=0.012),
            'SIDEWAYS': RegimeAnchor(mean=0.0001, std=0.006),
            'CRISIS': RegimeAnchor(mean=-0.0030, std=0.025),
        }
        self.anchor_update_history = {
            regime: [] for regime in self.regime_anchors.keys()
        }

    def get_anchor_history(self, regime_name: str) -> list:
        """
        Get audit trail of anchor updates for a specific regime.

        Useful for monitoring how anchors evolve over time and detecting
        when market structure changes.

        Args:
            regime_name: Regime to inspect (e.g., 'BULLISH')

        Returns:
            List of dicts with keys: 'mean', 'std', 'observed_mean', 'observed_std'

        Example:
            >>> history = interpreter.get_anchor_history('BULLISH')
            >>> history[-1]  # Most recent update
            # {'mean': 0.00102, 'std': 0.0081, 'observed_mean': 0.0011, ...}
        """
        return self.anchor_update_history.get(regime_name, [])

    def compute_anchor_stability(self, regime_name: str) -> float:
        """
        Compute stability of regime anchor (coefficient of variation).

        Higher stability = less change over time = more consistent regime definition.

        Args:
            regime_name: Regime to assess

        Returns:
            Stability score (0-1): 1.0 = perfectly stable, 0 = highly volatile
        """
        history = self.anchor_update_history.get(regime_name, [])

        if len(history) < 2:
            return 1.0  # Newly initialized anchors are "stable"

        means = [h['mean'] for h in history]
        mean_of_means = np.mean(means)
        std_of_means = np.std(means)

        if mean_of_means == 0:
            return 1.0  # Avoid division by zero

        # Coefficient of variation (lower = more stable)
        cv = std_of_means / abs(mean_of_means)

        # Convert to stability score (1.0 = stable, 0 = volatile)
        stability = 1.0 / (1.0 + cv)

        return float(stability)

    def print_anchor_summary(self) -> None:
        """Print summary of current anchors and their stability."""
        print("Regime Anchor Summary")
        print("=" * 60)
        print(f"Anchor update rate (α): {self.anchor_update_rate}")
        print()

        for regime_name in sorted(self.regime_anchors.keys()):
            anchor = self.regime_anchors[regime_name]
            stability = self.compute_anchor_stability(regime_name)
            history_len = len(self.anchor_update_history[regime_name])

            print(f"{regime_name:10} → Mean: {anchor.mean:+.5f}, "
                  f"Std: {anchor.std:.5f} | "
                  f"Stability: {stability:.2%}, "
                  f"Updates: {history_len}")

        print("=" * 60)


# Default regime anchors (moderate market conditions)
DEFAULT_REGIME_ANCHORS = {
    'BULLISH': {'mean': 0.0010, 'std': 0.008},      # +0.1% daily, 0.8% vol
    'BEARISH': {'mean': -0.0008, 'std': 0.012},     # -0.08% daily, 1.2% vol
    'SIDEWAYS': {'mean': 0.0001, 'std': 0.006},     # +0.01% daily, 0.6% vol
    'CRISIS': {'mean': -0.0030, 'std': 0.025},      # -0.3% daily, 2.5% vol
}
