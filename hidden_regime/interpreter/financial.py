"""Financial Interpreter implementation.

Unified interpreter combining data-driven regime interpretation with comprehensive
performance analysis. Implements InterpreterComponent interface.

This replaces the dual FinancialAnalysis/FinancialInterpreter architecture with
a single, clean implementation.
"""

import warnings
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hidden_regime.config.interpreter import InterpreterConfiguration
from hidden_regime.interpreter.anchored import AnchoredInterpreter
from hidden_regime.interpreter.base import BaseInterpreter
from hidden_regime.utils.exceptions import ValidationError


class FinancialInterpreter(BaseInterpreter):
    """Financial regime interpreter with comprehensive performance analysis.

    Combines:
    - Data-driven regime labeling (from emission parameters)
    - Comprehensive performance analysis
    - Regime characterization and statistics
    - Visualization capabilities

    NO signal generation - that's handled by SignalGenerator component.
    """

    def __init__(self, config: InterpreterConfiguration):
        """Initialize financial interpreter.

        Args:
            config: InterpreterConfiguration object
        """
        super().__init__(config)

        # Performance analysis tracking
        self._last_model_output = None
        self._last_interpretation = None
        self._last_raw_data = None

        # Regime profiles cache
        self._regime_profiles: Dict[int, Dict] = {}
        self._emission_means: Optional[np.ndarray] = None
        self._emission_stds: Optional[np.ndarray] = None

        # Multivariate support
        self._emission_covs: Optional[np.ndarray] = None
        self._is_multivariate: bool = False
        self._feature_names: List[str] = []

        # Anchored interpretation (for stable regime labels)
        self._anchored_interpreter: Optional[AnchoredInterpreter] = None
        if config.use_anchored_interpretation:
            self._anchored_interpreter = AnchoredInterpreter(
                anchor_update_rate=config.anchor_update_rate
            )

        # Try to import performance analyzer
        try:
            from hidden_regime.analysis.performance import RegimePerformanceAnalyzer
            self.performance_analyzer = RegimePerformanceAnalyzer()
        except ImportError:
            self.performance_analyzer = None
            warnings.warn(
                "RegimePerformanceAnalyzer not available - "
                "performance metrics will be limited"
            )

    def update(
        self,
        model_output: pd.DataFrame,
        raw_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Interpret model output and add regime information.

        Args:
            model_output: Raw model predictions with columns:
                - timestamp: datetime
                - state or predicted_state: int (0, 1, 2, ...)
                - confidence: float (max probability)
                - [optional] emission_means: array
                - [optional] emission_stds: array
            raw_data: Optional raw OHLCV data for performance calculations

        Returns:
            DataFrame with added regime interpretation columns
        """
        if model_output.empty:
            raise ValidationError("Model output cannot be empty")

        # Normalize column names (support both 'state' and 'predicted_state')
        output = model_output.copy()
        if 'predicted_state' in output.columns and 'state' not in output.columns:
            output['state'] = output['predicted_state']
        elif 'state' in output.columns and 'predicted_state' not in output.columns:
            output['predicted_state'] = output['state']

        # Validate required columns
        required_cols = ["state", "confidence"]
        missing_cols = [col for col in required_cols if col not in output.columns]
        if missing_cols:
            raise ValidationError(
                f"Required columns missing from model output: {missing_cols}"
            )

        # Store references for performance analysis
        self._last_model_output = output.copy()
        self._last_raw_data = raw_data.copy() if raw_data is not None else None

        # Use BaseInterpreter.update() to add basic regime interpretation
        # This adds: regime_label, regime_color, regime_type, regime_strength
        interpreted = super().update(output)

        # Add regime statistics
        interpreted = self._add_regime_statistics(interpreted)

        # Add duration analysis
        interpreted = self._add_duration_analysis(interpreted)

        # Add regime stability metrics
        interpreted = self._add_regime_stability_metrics(interpreted)

        # Add return analysis if raw data available
        if raw_data is not None and 'log_return' in raw_data.columns:
            interpreted = self._add_return_analysis(interpreted)

        # Add volatility analysis if raw data available
        if raw_data is not None and 'log_return' in raw_data.columns:
            interpreted = self._add_volatility_analysis(interpreted)

        # Add multivariate analysis if multivariate model
        interpreted = self._add_multivariate_analysis(interpreted)

        # Store for API methods
        self._last_interpretation = interpreted.copy()

        return interpreted

    def _assign_regime_labels(
        self, model_output: pd.DataFrame
    ) -> Dict[int, str]:
        """Assign regime labels using data-driven characteristics.

        Implements BaseInterpreter abstract method.
        Supports both univariate and multivariate model outputs.
        Optionally uses anchored interpretation for stable, slowly-adapting labels.

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

        # Detect multivariate mode from model output
        self._is_multivariate = "emission_covs" in model_output.columns
        if self._is_multivariate and "feature_names" in model_output.columns:
            # Extract feature names for multivariate interpretation
            first_row = model_output.iloc[0]
            self._feature_names = first_row.get("feature_names", [])

        # Extract emission parameters from model output
        if "emission_means" not in model_output.columns:
            return self._default_labels(self.config.n_states)

        # Get base data-driven labels
        if self.config.interpretation_method == "data_driven":
            base_labels = self._data_driven_labels(model_output)
        elif self.config.interpretation_method == "threshold":
            base_labels = self._threshold_based_labels(model_output)
        else:
            base_labels = self._default_labels(self.config.n_states)

        # Apply anchored interpretation if enabled
        if self._anchored_interpreter is not None:
            first_row = model_output.iloc[0]
            emission_means = first_row.get("emission_means")
            emission_stds = first_row.get("emission_stds")

            if emission_means is not None and emission_stds is not None:
                # Use anchored interpreter to map states to regimes
                anchored_labels = self._apply_anchored_interpretation(
                    emission_means=np.array(emission_means),
                    emission_stds=np.array(emission_stds),
                    base_labels=base_labels
                )
                return anchored_labels

        return base_labels

    def _apply_anchored_interpretation(
        self,
        emission_means: np.ndarray,
        emission_stds: np.ndarray,
        base_labels: Dict[int, str]
    ) -> Dict[int, str]:
        """Apply anchored interpretation to stabilize regime labels.

        Uses AnchoredInterpreter to map HMM states to regime labels based on
        KL divergence to slowly-adapting anchors. This provides stable labels
        even as model parameters drift through online updates.

        Args:
            emission_means: Array of emission means for each state
            emission_stds: Array of emission standard deviations for each state
            base_labels: Data-driven regime labels from _data_driven_labels

        Returns:
            Anchored regime labels Dict[state_id -> regime_label]
        """
        anchored_labels = {}

        for state_id in range(len(emission_means)):
            state_mean = float(emission_means[state_id])
            state_std = float(emission_stds[state_id])

            # Compute KL divergence to each regime anchor
            kl_scores = {}
            for regime_name, anchor in self._anchored_interpreter.regime_anchors.items():
                kl_dist = self._anchored_interpreter._gaussian_kl_divergence(
                    mean1=state_mean,
                    std1=state_std,
                    mean2=anchor.mean,
                    std2=anchor.std
                )
                kl_scores[regime_name] = kl_dist

            # Assign to regime with minimum KL divergence
            best_regime = min(kl_scores, key=kl_scores.get)

            # Update anchor slowly toward observed state
            self._anchored_interpreter._update_anchor(
                best_regime,
                state_mean=state_mean,
                state_std=state_std
            )

            anchored_labels[state_id] = best_regime

        return anchored_labels

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

        # Get emission parameters (assume they're the same for all rows)
        first_row = model_output.iloc[0]
        emission_means = first_row.get("emission_means")
        emission_stds = first_row.get("emission_stds")

        if emission_means is None or emission_stds is None:
            return self._default_labels(n_states)

        # Store for later use
        self._emission_means = np.array(emission_means)
        self._emission_stds = np.array(emission_stds)

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

        # Store for later use
        self._emission_means = np.array(emission_means)

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

    def _add_regime_statistics(self, interpretation: pd.DataFrame) -> pd.DataFrame:
        """Add basic regime statistics.

        Args:
            interpretation: Interpreted regime data

        Returns:
            DataFrame with regime statistics added
        """
        # Calculate days in current regime
        interpretation["days_in_regime"] = self._calculate_days_in_regime(
            interpretation["state"]
        )

        # Add expected regime characteristics from emission parameters
        if self._regime_profiles:
            for state_idx, profile in self._regime_profiles.items():
                state_mask = interpretation["state"] == state_idx
                interpretation.loc[state_mask, "expected_return"] = profile["mean_return"]
                interpretation.loc[state_mask, "expected_volatility"] = profile["volatility"]

        # Calculate regime performance statistics using actual data
        if self._last_raw_data is not None and "log_return" in self._last_raw_data.columns:
            raw_data = self._last_raw_data

            for state_idx in range(self.config.n_states):
                state_mask = interpretation["state"] == state_idx
                if state_mask.sum() > 0:
                    # Get returns for this regime
                    regime_returns = raw_data.loc[state_mask, "log_return"]

                    # Win rate: percentage of positive return days
                    win_rate = (regime_returns > 0).sum() / len(regime_returns) if len(regime_returns) > 0 else 0.0
                    interpretation.loc[state_mask, "win_rate"] = win_rate

                    # Max drawdown: maximum peak-to-trough decline
                    if len(regime_returns) > 1:
                        cumulative = (1 + regime_returns).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative - running_max) / running_max
                        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
                    else:
                        max_drawdown = 0.0
                    interpretation.loc[state_mask, "max_drawdown"] = max_drawdown

                    # Regime strength: coefficient of variation (inverse)
                    # Higher strength = more consistent returns
                    if regime_returns.std() > 0 and len(regime_returns) > 1:
                        regime_strength = min(1.0, abs(regime_returns.mean()) / regime_returns.std())
                    else:
                        regime_strength = 0.0
                    interpretation.loc[state_mask, "regime_strength"] = regime_strength

        return interpretation

    def _calculate_days_in_regime(self, states: pd.Series) -> pd.Series:
        """Calculate number of consecutive days in current regime.

        Args:
            states: Series of state indices

        Returns:
            Series with days in regime count
        """
        days_in_regime = []
        current_count = 1

        for i in range(len(states)):
            if i == 0:
                days_in_regime.append(current_count)
            elif states.iloc[i] == states.iloc[i - 1]:
                current_count += 1
                days_in_regime.append(current_count)
            else:
                current_count = 1
                days_in_regime.append(current_count)

        return pd.Series(days_in_regime, index=states.index)

    def _add_duration_analysis(self, interpretation: pd.DataFrame) -> pd.DataFrame:
        """Add regime duration analysis.

        Args:
            interpretation: Interpreted regime data

        Returns:
            DataFrame with duration analysis added
        """
        # Calculate regime transitions
        transitions = (
            interpretation["state"] != interpretation["state"].shift(1)
        ).cumsum()
        interpretation["regime_episode"] = transitions

        # Calculate expected remaining duration if we have profiles
        if self._regime_profiles and "days_in_regime" in interpretation.columns:
            expected_durations = {}
            for state_idx in range(self.config.n_states):
                # Use a default expected duration of 10 days if not in profiles
                # (More sophisticated duration estimation could be added later)
                expected_durations[state_idx] = 10.0

            interpretation["expected_duration"] = interpretation["state"].map(
                expected_durations
            )

            interpretation["expected_remaining_duration"] = interpretation.apply(
                lambda row: max(1, row["expected_duration"] - row["days_in_regime"]),
                axis=1
            )

        return interpretation

    def _add_regime_stability_metrics(self, interpretation: pd.DataFrame) -> pd.DataFrame:
        """Add regime stability and transition metrics.

        Computes diagnostic metrics to assess market coherence and inform position sizing:
        - stability_score: Overall market coherence (0-1, >0.8=stable)
        - transition_rate_recent: % of days with regime changes (20-day window)
        - regime_transitions_all: Binary indicator of regime change
        - expected_regime_persistence: Expected days for current regime type

        Args:
            interpretation: Interpreted regime data

        Returns:
            DataFrame with stability metrics added
        """
        if "days_in_regime" not in interpretation.columns:
            return interpretation

        # Mark regime transitions
        interpretation["regime_transitions_all"] = (
            interpretation["state"] != interpretation["state"].shift(1)
        ).astype(int)

        # Calculate recent transition rate (20-day rolling window)
        window = min(20, len(interpretation))
        if window > 1:
            rolling_transitions = (
                interpretation["regime_transitions_all"]
                .rolling(window=window, min_periods=1)
                .sum()
            )
            interpretation["transition_rate_recent"] = (
                rolling_transitions / window
            )
        else:
            interpretation["transition_rate_recent"] = 0.0

        # Calculate stability score (inverse of transition frequency)
        # Formula: 1.0 / (1.0 + transitions_in_window)
        # High score (>0.8): Stable regime, good for trend-following
        # Medium score (0.5-0.8): Moderate conditions
        # Low score (<0.3): Regime collapse, go defensive
        transitions_window = interpretation["regime_transitions_all"].rolling(
            window=window, min_periods=1
        ).sum()
        interpretation["stability_score"] = 1.0 / (1.0 + transitions_window)

        # Calculate expected regime persistence based on historical data
        # Uses the current regime's observed persistence
        expected_durations = {}
        for state_idx in range(self.config.n_states):
            state_episodes = interpretation[interpretation["state"] == state_idx][
                "days_in_regime"
            ]
            if len(state_episodes) > 0:
                # Use median persistence for this regime type
                expected_durations[state_idx] = float(state_episodes.median())
            else:
                expected_durations[state_idx] = 10.0

        interpretation["expected_regime_persistence"] = interpretation["state"].map(
            expected_durations
        )

        # Calculate regime fit quality using KL divergence to anchor
        # (if anchored interpretation is enabled)
        if self._anchored_interpreter is not None and self._emission_means is not None:
            fit_qualities = []
            for i, row in interpretation.iterrows():
                state_idx = int(row["state"])
                if state_idx < len(self._emission_means):
                    state_mean = float(self._emission_means[state_idx])
                    state_std = float(self._emission_stds[state_idx]) if self._emission_stds is not None else 0.01

                    # Get KL distance to the assigned regime anchor
                    regime_label = row.get("regime_label", "Unknown")
                    anchor = self._anchored_interpreter.regime_anchors.get(regime_label)

                    if anchor is not None:
                        kl_dist = self._anchored_interpreter._gaussian_kl_divergence(
                            mean1=state_mean,
                            std1=state_std,
                            mean2=anchor.mean,
                            std2=anchor.std
                        )
                        # Confidence is inverse of KL distance
                        regime_fit_quality = 1.0 / (1.0 + kl_dist)
                    else:
                        regime_fit_quality = 0.5
                else:
                    regime_fit_quality = 0.5

                fit_qualities.append(regime_fit_quality)

            interpretation["regime_fit_quality"] = fit_qualities

        # Add interpretation for stability score
        def stability_interpretation(score):
            if score > 0.8:
                return "Stable"
            elif score > 0.5:
                return "Moderate"
            elif score > 0.3:
                return "Low"
            else:
                return "Collapse"

        interpretation["stability_interpretation"] = interpretation[
            "stability_score"
        ].apply(stability_interpretation)

        return interpretation

    def _add_return_analysis(self, interpretation: pd.DataFrame) -> pd.DataFrame:
        """Add return-based analysis using actual price data.

        Args:
            interpretation: Interpreted regime data

        Returns:
            DataFrame with return analysis added
        """
        if self._last_raw_data is None or "log_return" not in self._last_raw_data.columns:
            return interpretation

        raw_data = self._last_raw_data

        # Default return window
        return_window = 20

        if len(interpretation) >= return_window:
            # Calculate rolling cumulative returns
            rolling_log_returns = (
                raw_data["log_return"]
                .rolling(window=return_window)
                .sum()
            )
            interpretation["rolling_return"] = (
                np.exp(rolling_log_returns) - 1
            )  # Convert to percentage

            # Calculate regime-specific expected returns
            if self._state_labels:
                expected_returns = {}
                for state_idx in range(self.config.n_states):
                    state_mask = interpretation["state"] == state_idx
                    if state_mask.any():
                        expected_returns[state_idx] = interpretation[
                            state_mask
                        ]["rolling_return"].median()

                # Map expected returns to interpretation
                interpretation["expected_rolling_return"] = interpretation[
                    "state"
                ].map(expected_returns)

                # Calculate return vs expected
                interpretation["return_vs_expected"] = (
                    interpretation["rolling_return"] - interpretation["expected_rolling_return"]
                )
        else:
            interpretation["rolling_return"] = 0.0
            interpretation["return_vs_expected"] = 0.0

        return interpretation

    def _add_volatility_analysis(self, interpretation: pd.DataFrame) -> pd.DataFrame:
        """Add volatility analysis using actual price data.

        Args:
            interpretation: Interpreted regime data

        Returns:
            DataFrame with volatility analysis added
        """
        if self._last_raw_data is None or "log_return" not in self._last_raw_data.columns:
            return interpretation

        raw_data = self._last_raw_data

        # Default volatility window
        volatility_window = 20

        if len(interpretation) >= volatility_window:
            # Calculate rolling volatility
            rolling_vol = (
                raw_data["log_return"]
                .rolling(window=volatility_window)
                .std()
            )
            interpretation["rolling_volatility"] = rolling_vol * np.sqrt(
                252
            )  # Annualized volatility

            # Calculate regime-specific expected volatility
            if self._state_labels:
                expected_volatilities = {}
                for state_idx in range(self.config.n_states):
                    state_mask = interpretation["state"] == state_idx
                    if state_mask.any():
                        # Get log returns for this regime's periods
                        regime_dates = interpretation[state_mask].index
                        regime_returns = raw_data.loc[regime_dates, "log_return"]

                        # Calculate regime-specific volatility
                        regime_vol = regime_returns.std() * np.sqrt(
                            252
                        )  # Annualized
                        expected_volatilities[state_idx] = regime_vol

                # Map expected volatilities to interpretation
                interpretation["expected_volatility"] = interpretation["state"].map(
                    expected_volatilities
                )

                # Calculate volatility vs expected
                interpretation["volatility_vs_expected"] = (
                    interpretation["rolling_volatility"] - interpretation["expected_volatility"]
                )

                # Add volatility regime classification
                interpretation["volatility_regime"] = interpretation.apply(
                    lambda row: self._classify_volatility_regime(
                        row["rolling_volatility"],
                        row.get("expected_volatility", 0.15),
                    ),
                    axis=1,
                )
        else:
            interpretation["rolling_volatility"] = 0.0
            interpretation["volatility_vs_expected"] = 0.0
            interpretation["volatility_regime"] = "Unknown"

        return interpretation

    def _classify_volatility_regime(
        self, current_vol: float, expected_vol: float
    ) -> str:
        """Classify current volatility relative to expected.

        Args:
            current_vol: Current volatility level
            expected_vol: Expected volatility for regime

        Returns:
            Volatility regime classification
        """
        if current_vol > expected_vol * 1.5:
            return "High Volatility"
        elif current_vol > expected_vol * 1.2:
            return "Elevated Volatility"
        elif current_vol < expected_vol * 0.8:
            return "Low Volatility"
        else:
            return "Normal Volatility"

    def _add_multivariate_analysis(self, interpretation: pd.DataFrame) -> pd.DataFrame:
        """Add multivariate regime characteristics.

        Computes eigenvalue analysis and covariance-based regime metrics for
        multivariate HMM outputs. Provides financial interpretation of covariance structure.

        Args:
            interpretation: Interpreted regime data

        Returns:
            DataFrame with multivariate analysis columns added
        """
        if not self._is_multivariate or self._last_model_output is None:
            return interpretation

        # Extract covariance matrices from model output
        if "emission_covs" not in self._last_model_output.columns:
            return interpretation

        try:
            first_row = self._last_model_output.iloc[0]
            emission_covs = first_row.get("emission_covs")

            if emission_covs is None:
                return interpretation

            self._emission_covs = np.array(emission_covs)

            # Compute multivariate characteristics for each state
            multivariate_metrics = {}
            for state_idx in range(self.config.n_states):
                cov_matrix = self._emission_covs[state_idx]

                # Ensure matrix is 2D
                if cov_matrix.ndim < 2:
                    continue

                try:
                    # Eigenvalue decomposition
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                    # Sort by magnitude (descending)
                    idx = np.argsort(eigenvalues)[::-1]
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]

                    # Compute metrics
                    eigenvalue_ratio = (
                        eigenvalues[0] / eigenvalues[-1]
                        if len(eigenvalues) > 1 and eigenvalues[-1] > 1e-10
                        else 0.0
                    )

                    # PCA explained variance in first component
                    pca_explained = (
                        eigenvalues[0] / np.sum(eigenvalues)
                        if np.sum(eigenvalues) > 1e-10
                        else 0.0
                    )

                    # Correlation matrix from covariance
                    stds = np.sqrt(np.diag(cov_matrix))
                    if np.all(stds > 1e-10):
                        correlation_matrix = cov_matrix / np.outer(stds, stds)
                        avg_correlation = np.abs(
                            correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
                        ).mean()
                    else:
                        avg_correlation = 0.0

                    # Condition number
                    condition_number = np.linalg.cond(cov_matrix)

                    # Financial interpretation of eigenvalue ratio
                    variance_concentration = self._interpret_eigenvalue_ratio(eigenvalue_ratio)

                    # Interpret feature correlations
                    correlation_regime = self._interpret_correlation(avg_correlation)

                    multivariate_metrics[state_idx] = {
                        "eigenvalue_ratio": eigenvalue_ratio,
                        "pca_explained_variance": pca_explained,
                        "avg_feature_correlation": avg_correlation,
                        "condition_number": condition_number,
                        "covariance_trace": np.trace(cov_matrix),
                        "variance_concentration": variance_concentration,
                        "correlation_regime": correlation_regime,
                    }

                except (np.linalg.LinAlgError, ValueError):
                    # If decomposition fails, use defaults
                    multivariate_metrics[state_idx] = {
                        "eigenvalue_ratio": 0.0,
                        "pca_explained_variance": 0.0,
                        "avg_feature_correlation": 0.0,
                        "condition_number": 0.0,
                        "covariance_trace": np.trace(cov_matrix),
                        "variance_concentration": "Unknown",
                        "correlation_regime": "Unknown",
                    }

            # Add metrics to interpretation
            if multivariate_metrics:
                for metric_name in [
                    "eigenvalue_ratio",
                    "pca_explained_variance",
                    "avg_feature_correlation",
                    "condition_number",
                    "covariance_trace",
                    "variance_concentration",
                    "correlation_regime",
                ]:
                    interpretation[f"multivariate_{metric_name}"] = interpretation[
                        "state"
                    ].map(lambda s: multivariate_metrics.get(s, {}).get(metric_name, 0.0))

        except Exception as e:
            warnings.warn(
                f"Error computing multivariate analysis: {e}. "
                "Skipping multivariate metrics."
            )

        return interpretation

    def _interpret_eigenvalue_ratio(self, ratio: float) -> str:
        """Interpret eigenvalue ratio in financial terms.

        Higher ratio means one variance direction dominates (concentrated risk).
        Lower ratio means risk spread across multiple dimensions (diversified).

        Args:
            ratio: Ratio of largest to smallest eigenvalue

        Returns:
            Financial interpretation
        """
        if ratio < 1.5:
            return "Isotropic"  # Balanced risk across dimensions
        elif ratio < 3.0:
            return "Moderate Concentration"  # One direction somewhat dominant
        elif ratio < 10.0:
            return "High Concentration"  # One direction strongly dominant
        else:
            return "Extreme Concentration"  # Risk entirely in one direction

    def _interpret_correlation(self, avg_correlation: float) -> str:
        """Interpret average feature correlation in financial terms.

        Args:
            avg_correlation: Average absolute correlation between features

        Returns:
            Financial interpretation
        """
        if avg_correlation < 0.2:
            return "Uncorrelated"  # Features move independently
        elif avg_correlation < 0.5:
            return "Low Correlation"  # Weak relationship
        elif avg_correlation < 0.7:
            return "Moderate Correlation"  # Clear relationship
        else:
            return "High Correlation"  # Features move together

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
        state_labels = self._state_labels

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

    # API Methods for external access

    def get_current_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime state.

        Returns:
            Dictionary with current regime information
        """
        if self._last_interpretation is None or len(self._last_interpretation) == 0:
            return {"status": "No interpretation available"}

        current = self._last_interpretation.iloc[-1]

        summary = {
            "current_state": int(current["state"]),
            "regime_label": current.get("regime_label", f"State_{current['state']}"),
            "regime_type": current.get("regime_type", "Unknown"),
            "confidence": float(current["confidence"]),
            "days_in_regime": int(current.get("days_in_regime", 0)),
        }

        # Add expected characteristics if available
        for key in ["expected_return", "expected_volatility", "expected_duration"]:
            if key in current:
                summary[key] = float(current[key])

        return summary

    def get_comprehensive_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance analysis using RegimePerformanceAnalyzer.

        Returns:
            Dictionary with detailed performance metrics and analysis
        """
        if self._last_interpretation is None:
            return {"status": "No interpretation available"}

        if not self.performance_analyzer:
            return {"error": "RegimePerformanceAnalyzer not available"}

        try:
            # Get comprehensive performance analysis
            performance_metrics = self.performance_analyzer.analyze_regime_performance(
                analysis_results=self._last_interpretation,
                raw_data=self._last_raw_data
            )

            return performance_metrics

        except Exception as e:
            return {"error": f"Performance analysis failed: {str(e)}"}

    def get_regime_transition_matrix(self) -> Optional[Dict[str, Any]]:
        """Get regime transition matrix and statistics.

        Returns:
            Dictionary with transition analysis or None if unavailable
        """
        if self._last_interpretation is None:
            return None

        try:
            performance_metrics = self.get_comprehensive_performance_metrics()
            return performance_metrics.get("transition_analysis", {})
        except Exception:
            return None

    def get_regime_duration_statistics(self) -> Optional[Dict[str, Any]]:
        """Get regime duration statistics.

        Returns:
            Dictionary with duration statistics or None if unavailable
        """
        if self._last_interpretation is None:
            return None

        try:
            performance_metrics = self.get_comprehensive_performance_metrics()
            return performance_metrics.get("duration_analysis", {})
        except Exception:
            return None

    def get_regime_profiles(self) -> Optional[Dict[int, Any]]:
        """Get regime profiles for visualization.

        Returns:
            Dictionary mapping state IDs to regime profiles or None
        """
        return self._regime_profiles if self._regime_profiles else None

    @staticmethod
    def assess_data_for_regime_detection(
        raw_data: pd.DataFrame, observed_signal: str = "log_return"
    ) -> Dict[str, Any]:
        """Assess raw data to determine optimal regime detection approach.

        Args:
            raw_data: Raw financial data with price and return information
            observed_signal: Column name for returns (default: 'log_return')

        Returns:
            Dictionary with data assessment and recommendations
        """
        if observed_signal not in raw_data.columns:
            raise ValidationError(
                f"Observed signal '{observed_signal}' not found in data columns: "
                f"{list(raw_data.columns)}"
            )

        returns = raw_data[observed_signal].dropna()
        if len(returns) < 10:
            raise ValidationError(
                f"Insufficient data: only {len(returns)} valid returns available. "
                f"Need at least 10 observations."
            )

        # Convert to percentage space for analysis
        returns_pct = np.exp(returns) - 1

        # Basic statistics
        mean_return = returns_pct.mean()
        std_return = returns_pct.std()
        min_return = returns_pct.min()
        max_return = returns_pct.max()

        # Trend analysis
        positive_returns = returns_pct > 0.001  # > 0.1% daily
        negative_returns = returns_pct < -0.001  # < -0.1% daily

        pct_positive = positive_returns.mean()
        pct_negative = negative_returns.mean()

        # Regime switching indicators
        return_spread = max_return - min_return

        assessment = {
            "data_summary": {
                "n_observations": len(returns),
                "mean_daily_return": f"{mean_return:.3%}",
                "daily_volatility": f"{std_return:.3%}",
                "return_range": f"{min_return:.3%} to {max_return:.3%}",
                "return_spread": f"{return_spread:.3%}",
            },
            "return_distribution": {
                "pct_positive_days": f"{pct_positive:.1%}",
                "pct_negative_days": f"{pct_negative:.1%}",
            },
            "recommendations": {
                "suitable_for_regime_detection": return_spread > 0.02,
                "recommended_n_states": 3 if return_spread > 0.03 else 2,
            },
        }

        return assessment

    # Visualization methods (simplified - no signal plotting)

    def plot(self, ax=None, **kwargs) -> plt.Figure:
        """Generate visualization for interpretation results.

        Args:
            ax: Optional matplotlib axes for pipeline integration
            **kwargs: Additional plotting arguments

        Returns:
            matplotlib Figure with regime visualizations
        """
        if self._last_interpretation is None:
            if ax is not None:
                ax.text(
                    0.5, 0.5, "No interpretation results yet",
                    ha="center", va="center", fontsize=14,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                return ax.figure
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(
                    0.5, 0.5, "No interpretation results yet",
                    ha="center", va="center", fontsize=14,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                return fig

        # Compact plot for pipeline integration
        if ax is not None:
            return self._plot_compact(ax, **kwargs)

        # Full standalone plot
        return self._plot_full(**kwargs)

    def _plot_compact(self, ax, **kwargs):
        """Create compact plot for pipeline integration.

        Args:
            ax: Matplotlib axes
            **kwargs: Additional arguments

        Returns:
            Figure object
        """
        interpretation = self._last_interpretation

        # Plot regime sequence as colored bars
        for i, (idx, row) in enumerate(interpretation.iterrows()):
            regime_color = row.get("regime_color", "#808080")
            alpha = row["confidence"] * 0.7 + 0.3  # Scale alpha by confidence
            ax.bar(i, 1, color=regime_color, alpha=alpha, width=1, edgecolor="none")

        ax.set_title("Interpreter - Regime Sequence")
        ax.set_ylabel("Regime")
        ax.set_ylim(0, 1)

        # Add legend
        unique_regimes = interpretation["regime_label"].unique()
        unique_colors = interpretation.groupby("regime_label")["regime_color"].first()
        legend_elements = [
            plt.Rectangle(
                (0, 0), 1, 1,
                facecolor=unique_colors[regime],
                alpha=0.7,
                label=regime,
            )
            for regime in unique_regimes
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        return ax.figure

    def _plot_full(self, **kwargs):
        """Create full standalone plot with subplots.

        Args:
            **kwargs: Additional arguments

        Returns:
            Figure object
        """
        interpretation = self._last_interpretation

        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Plot 1: Regime sequence with confidence
        ax1 = axes[0]
        for i, (idx, row) in enumerate(interpretation.iterrows()):
            regime_color = row.get("regime_color", "#808080")
            alpha = row["confidence"] * 0.7 + 0.3
            ax1.bar(i, 1, color=regime_color, alpha=alpha, width=1, edgecolor="none")

        ax1.set_title("Regime Sequence (colored by type, opacity by confidence)")
        ax1.set_ylabel("Regime")
        ax1.set_ylim(0, 1)

        # Add legend
        unique_regimes = interpretation["regime_label"].unique()
        unique_colors = interpretation.groupby("regime_label")["regime_color"].first()
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=unique_colors[regime], alpha=0.7, label=regime)
            for regime in unique_regimes
        ]
        ax1.legend(handles=legend_elements, loc="upper right")

        # Plot 2: Confidence and days in regime
        ax2 = axes[1]
        ax2_twin = ax2.twinx()

        x_vals = range(len(interpretation))
        ax2.plot(
            x_vals, interpretation["confidence"], "b-", linewidth=2, label="Confidence"
        )

        if "days_in_regime" in interpretation.columns:
            ax2_twin.plot(
                x_vals,
                interpretation["days_in_regime"],
                "r--",
                linewidth=2,
                label="Days in Regime",
            )
            ax2_twin.set_ylabel("Days in Regime", color="red")
            ax2_twin.tick_params(axis="y", labelcolor="red")

        ax2.set_title("Confidence and Regime Duration")
        ax2.set_ylabel("Confidence", color="blue")
        ax2.set_xlabel("Time")
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = (
            ax2_twin.get_legend_handles_labels()
            if "days_in_regime" in interpretation.columns
            else ([], [])
        )
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        return fig

    def get_anchored_interpretation_status(self) -> Dict[str, Any]:
        """Get status of anchored interpretation system.

        Returns:
            Dictionary with anchored interpretation status and anchor parameters
        """
        if self._anchored_interpreter is None:
            return {"enabled": False, "status": "Anchored interpretation disabled"}

        status = {
            "enabled": True,
            "anchor_update_rate": self._anchored_interpreter.anchor_update_rate,
            "regime_anchors": {}
        }

        # Add regime anchor information
        for regime_name, anchor in self._anchored_interpreter.regime_anchors.items():
            status["regime_anchors"][regime_name] = {
                "mean": float(anchor.mean),
                "std": float(anchor.std)
            }

        # Add anchor update history summary
        status["anchor_update_history_counts"] = {
            regime_name: len(history)
            for regime_name, history in self._anchored_interpreter.anchor_update_history.items()
        }

        return status

    def reset_anchored_interpretation(self) -> None:
        """Reset anchored interpretation system to initial state.

        This clears all anchor update history and resets to default anchors.
        """
        if self._anchored_interpreter is not None:
            # Clear the update history
            for regime_name in self._anchored_interpreter.anchor_update_history:
                self._anchored_interpreter.anchor_update_history[regime_name].clear()

            # Reset anchors to default values
            from hidden_regime.interpreter.anchored import RegimeAnchor
            self._anchored_interpreter.regime_anchors = {
                'BULLISH': RegimeAnchor(mean=0.0010, std=0.008),
                'BEARISH': RegimeAnchor(mean=-0.0008, std=0.012),
                'SIDEWAYS': RegimeAnchor(mean=0.0001, std=0.006),
                'CRISIS': RegimeAnchor(mean=-0.0030, std=0.025),
            }
