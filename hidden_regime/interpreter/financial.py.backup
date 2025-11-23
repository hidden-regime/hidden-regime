"""Financial Interpreter implementation.

Implements data-driven regime interpretation using actual training data characteristics.
Maps HMM states to semantic financial regime labels based on emission parameters.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from hidden_regime.config.interpreter import InterpreterConfiguration
from hidden_regime.interpreter.base import BaseInterpreter


class FinancialInterpreter(BaseInterpreter):
    """Financial regime interpreter using data-driven labeling.

    Assigns regime labels based on actual HMM emission parameters:
    - Mean return (daily %) → determines bull/bear tendency
    - Volatility (daily std) → determines crisis vs normal
    - Combination → specific regime label

    This replaces hard-coded financial domain assumptions with
    data-driven characteristics derived from actual market behavior.
    """

    def __init__(self, config: InterpreterConfiguration):
        """Initialize financial interpreter.

        Args:
            config: InterpreterConfiguration object
        """
        super().__init__(config)
        self._emission_means: Optional[np.ndarray] = None
        self._emission_stds: Optional[np.ndarray] = None

    def _assign_regime_labels(
        self, model_output: pd.DataFrame
    ) -> Dict[int, str]:
        """Assign regime labels using data-driven characteristics.

        Args:
            model_output: Model output including emission parameters

        Returns:
            Dictionary mapping state index to regime label
        """
        # Check for manual override
        if self.config.force_regime_labels is not None:
            labels = {}
            for state_idx, label in enumerate(self.config.force_regime_labels):
                labels[state_idx] = label
            return labels

        # Extract emission parameters from model output
        if "emission_means" not in model_output.columns:
            return self._default_labels(self.config.n_states)

        # Use data-driven labeling based on emission parameters
        if self.config.interpretation_method == "data_driven":
            return self._data_driven_labels(model_output)
        elif self.config.interpretation_method == "threshold":
            return self._threshold_based_labels(model_output)
        else:
            return self._default_labels(self.config.n_states)

    def _data_driven_labels(self, model_output: pd.DataFrame) -> Dict[int, str]:
        """Assign labels based on actual regime characteristics.

        Analyzes the emission parameters to determine what each regime
        actually represents in the data.

        Args:
            model_output: Model output with emission parameters

        Returns:
            State-to-label mapping
        """
        n_states = self.config.n_states

        # Get emission parameters (assume they're the same for all rows,
        # so we just look at the first row)
        first_row = model_output.iloc[0]
        emission_means = first_row.get("emission_means")
        emission_stds = first_row.get("emission_stds")

        if emission_means is None or emission_stds is None:
            return self._default_labels(n_states)

        # Create regime profiles based on emission characteristics
        profiles = {}
        for state_idx in range(n_states):
            mean_return = emission_means[state_idx] * 252  # Annualize daily return
            volatility = emission_stds[state_idx] * np.sqrt(252)  # Annualize daily vol

            profiles[state_idx] = {
                "mean_return": mean_return,
                "volatility": volatility,
                "label": self._classify_regime(mean_return, volatility, n_states),
            }

        self._regime_profiles = profiles

        # Create label mapping
        labels = {state_idx: profiles[state_idx]["label"] for state_idx in range(n_states)}
        return labels

    def _threshold_based_labels(self, model_output: pd.DataFrame) -> Dict[int, str]:
        """Assign labels using fixed return thresholds.

        Args:
            model_output: Model output with emission parameters

        Returns:
            State-to-label mapping
        """
        first_row = model_output.iloc[0]
        emission_means = first_row.get("emission_means")

        if emission_means is None:
            return self._default_labels(self.config.n_states)

        labels = {}
        n_states = self.config.n_states

        # Threshold-based classification
        bear_threshold = -0.005  # -0.5% daily
        bull_threshold = 0.01    # +1.0% daily

        for state_idx in range(n_states):
            mean_return = emission_means[state_idx]

            if mean_return < bear_threshold:
                label = "Bear"
            elif mean_return > bull_threshold:
                label = "Bull"
            else:
                label = "Sideways"

            labels[state_idx] = label

        return labels

    def _classify_regime(
        self, mean_return: float, volatility: float, n_states: int
    ) -> str:
        """Classify regime based on return and volatility characteristics.

        Uses a decision tree approach to assign meaningful regime labels.

        Args:
            mean_return: Annualized daily return (%)
            volatility: Annualized volatility (%)
            n_states: Total number of states (affects label set)

        Returns:
            Regime label
        """
        # Crisis regime: high volatility regardless of return
        if volatility > 0.40:  # >40% annual volatility
            if mean_return < -0.10:
                return "Crisis Bear"
            else:
                return "Crisis"

        # Strong directional regimes
        if mean_return > 0.20:  # >20% annual return
            if volatility > 0.20:
                return "Euphoric Bull"
            else:
                return "Bull"
        elif mean_return < -0.10:  # <-10% annual return
            if volatility > 0.20:
                return "Crash Bear"
            else:
                return "Bear"

        # Moderate return regimes
        if mean_return > 0.05:  # 5-20% annual return
            return "Uptrend"
        elif mean_return < -0.05:  # -5% to -10% annual return
            return "Downtrend"

        # Low volatility, near-zero return
        if volatility < 0.10:
            return "Flat"
        else:
            return "Sideways"

    def _default_labels(self, n_states: int) -> Dict[int, str]:
        """Get default labels for states when data-driven labeling not possible.

        Args:
            n_states: Number of states

        Returns:
            Default state-to-label mapping
        """
        if n_states == 2:
            return {0: "Bear", 1: "Bull"}
        elif n_states == 3:
            return {0: "Bear", 1: "Sideways", 2: "Bull"}
        elif n_states == 4:
            return {0: "Crisis", 1: "Bear", 2: "Sideways", 3: "Bull"}
        elif n_states == 5:
            return {0: "Crisis", 1: "Bear", 2: "Sideways", 3: "Bull", 4: "Euphoric"}
        else:
            # For larger n_states, create numbered labels
            return {state_idx: f"Regime_{state_idx}" for state_idx in range(n_states)}

    def add_multitimeframe_columns(
        self, output_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add multi-timeframe regime columns to interpreter output.

        For use with MultiTimeframeRegime model. Expects columns:
        - daily_predicted_state, weekly_predicted_state, monthly_predicted_state
        - alignment_score, alignment_label

        Adds columns:
        - daily_regime_label, weekly_regime_label, monthly_regime_label
        - timeframe_alignment, alignment_description

        Args:
            output_df: Interpreter output DataFrame with multi-timeframe predictions

        Returns:
            DataFrame with added multi-timeframe regime columns
        """
        df = output_df.copy()

        # Get regime labels mapping
        state_labels = self.regime_labels

        # Add regime labels for each timeframe
        if "daily_predicted_state" in df.columns:
            df["daily_regime_label"] = df["daily_predicted_state"].map(state_labels)

        if "weekly_predicted_state" in df.columns:
            df["weekly_regime_label"] = df["weekly_predicted_state"].map(state_labels)

        if "monthly_predicted_state" in df.columns:
            df["monthly_regime_label"] = df["monthly_predicted_state"].map(state_labels)

        # Add alignment description
        if "alignment_score" in df.columns:
            df["timeframe_alignment"] = df["alignment_score"]
            df["alignment_description"] = df["alignment_label"].apply(
                self._describe_alignment
            )

        return df

    def _describe_alignment(self, alignment_label: str) -> str:
        """Create description of timeframe alignment.

        Args:
            alignment_label: "Perfect", "Partial", or "Misaligned"

        Returns:
            Human-readable description
        """
        descriptions = {
            "Perfect": "All timeframes agree - high conviction signal",
            "Partial": "Two timeframes agree - moderate conviction",
            "Misaligned": "Timeframes disagree - low conviction, skip trade",
        }
        return descriptions.get(alignment_label, "Unknown alignment")
