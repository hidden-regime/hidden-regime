"""
Debug data accumulation and export for QuantConnect regime switching algorithms.

This module provides DebugDataAccumulator, which captures internal algorithm state
at every timestep and exports detailed CSVs for analysis and debugging.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class DebugDataAccumulator:
    """
    Accumulates debug data throughout a backtest and exports to CSV files.

    Captures:
    - Raw OHLCV data
    - Computed observations (log returns, features)
    - HMM model predictions and parameters
    - Interpreter regime labels and metrics
    - Trading signals

    Data is stored in-memory (lists of dicts) during backtest for performance,
    then exported to disk at end or on exception.

    Attributes:
        ticker: Asset ticker symbol
        n_states: Number of HMM states
        timesteps: List of per-bar data dicts
        state_probabilities: List of per-state probability dicts
        training_history: List of model training event dicts
        hmm_params: Dict of HMM parameters (updated on retrain)
    """

    def __init__(self, ticker: str, n_states: int) -> None:
        """
        Initialize debug data accumulator.

        Args:
            ticker: Asset ticker symbol (e.g., "SPY")
            n_states: Number of HMM states in model
        """
        self.ticker = ticker
        self.n_states = n_states

        # In-memory buffers (lists of dicts for speed)
        self.timesteps: List[Dict[str, Any]] = []
        self.state_probabilities: List[Dict[str, Any]] = []
        self.training_history: List[Dict[str, Any]] = []
        self.regime_changes: List[Dict[str, Any]] = []
        self.signals: List[Dict[str, Any]] = []

        # Static HMM info (updated on each retrain)
        self.hmm_params: Dict[str, Any] = {}

    def add_timestep(
        self,
        timestamp: pd.Timestamp,
        bar_data: Dict[str, float],
        observation_data: Dict[str, Any],
        model_output: Dict[str, Any],
        interpreter_output: Dict[str, Any],
        signal_data: Dict[str, Any],
    ) -> None:
        """
        Record data for one bar/timestep.

        Called at every bar with full pipeline state from data → model → interpreter.

        Args:
            timestamp: Bar close timestamp
            bar_data: Dict with keys: Open, High, Low, Close, Volume
            observation_data: Dict with computed features (e.g., log_return)
            model_output: Dict with HMM predictions (state, confidence, probs)
            interpreter_output: Dict with regime labels and metrics
            signal_data: Dict with trading signal info (direction, allocation, etc.)
        """
        # Detect regime change
        regime_changed = False
        if len(self.timesteps) > 0:
            prev_regime = self.timesteps[-1].get("regime_label")
            curr_regime = interpreter_output.get("regime_label")
            regime_changed = prev_regime != curr_regime

        # Build timestep dict
        timestep_dict = {
            "timestamp": timestamp,
            "regime_changed": regime_changed,
        }

        # Add bar data
        timestep_dict.update(bar_data)

        # Add observation data
        timestep_dict.update(observation_data)

        # Add model output (states, confidence, parameters)
        timestep_dict.update(model_output)

        # Add interpreter output (regime labels, metrics)
        timestep_dict.update(interpreter_output)

        # Add signal data
        timestep_dict.update(signal_data)

        self.timesteps.append(timestep_dict)

        # Record regime change for filtering later
        if regime_changed:
            self.regime_changes.append(timestep_dict.copy())

    def add_state_probabilities(
        self, timestamp: pd.Timestamp, state_probs: Dict[int, float]
    ) -> None:
        """
        Record per-state probabilities for this bar.

        Args:
            timestamp: Bar timestamp
            state_probs: Dict mapping state index to probability (0.0-1.0)
        """
        prob_row = {"timestamp": timestamp}
        for state_idx in range(self.n_states):
            prob_row[f"state_{state_idx}_prob"] = state_probs.get(state_idx, 0.0)
        self.state_probabilities.append(prob_row)

    def record_hmm_params(self, model: Any) -> None:
        """
        Capture HMM parameters from a fitted model.

        Called after model training or retraining.

        Args:
            model: HiddenMarkovModel instance with fitted parameters
        """
        self.hmm_params = {
            "n_states": self.n_states,
            "emission_means": model.emission_means_.tolist()
            if hasattr(model, "emission_means_")
            else [],
            "emission_stds": model.emission_stds_.tolist()
            if hasattr(model, "emission_stds_")
            else [],
            "transition_matrix": model.transition_matrix_.tolist()
            if hasattr(model, "transition_matrix_")
            else [],
            "initial_probs": model.initial_probs_.tolist()
            if hasattr(model, "initial_probs_")
            else [],
        }

        # Compute derived metrics
        if self.hmm_params.get("transition_matrix"):
            trans_matrix = np.array(self.hmm_params["transition_matrix"])
            diag = np.diag(trans_matrix)
            self.hmm_params["state_persistence"] = diag.tolist()
            # Expected duration = 1 / (1 - persistence)
            self.hmm_params["expected_durations"] = (
                1.0 / (1.0 - diag + 1e-10)
            ).tolist()

    def add_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Record model training or retraining event.

        Args:
            metrics: Dict with keys: timestamp, iteration, log_likelihood,
                    converged, training_time, etc.
        """
        self.training_history.append(metrics)

    def add_signal(self, signal_data: Dict[str, Any]) -> None:
        """
        Record a trading signal.

        Args:
            signal_data: Dict with trading signal info
        """
        self.signals.append(signal_data)

    def _flatten_hmm_params(self) -> pd.DataFrame:
        """
        Convert HMM parameter matrices to flattened CSV format.

        Returns:
            DataFrame with columns: [parameter, state, from_state, to_state, value]
        """
        rows = []

        if not self.hmm_params:
            return pd.DataFrame()

        n_states = self.hmm_params.get("n_states", self.n_states)

        # Emission means
        emission_means = self.hmm_params.get("emission_means", [])
        for state_idx in range(len(emission_means)):
            rows.append(
                {
                    "parameter": "emission_mean",
                    "state": state_idx,
                    "from_state": None,
                    "to_state": None,
                    "value": float(emission_means[state_idx]),
                }
            )

        # Emission stds
        emission_stds = self.hmm_params.get("emission_stds", [])
        for state_idx in range(len(emission_stds)):
            rows.append(
                {
                    "parameter": "emission_std",
                    "state": state_idx,
                    "from_state": None,
                    "to_state": None,
                    "value": float(emission_stds[state_idx]),
                }
            )

        # Transition matrix (as from_state → to_state probabilities)
        transition_matrix = self.hmm_params.get("transition_matrix", [])
        if transition_matrix:
            trans_array = np.array(transition_matrix)
            for from_state in range(trans_array.shape[0]):
                for to_state in range(trans_array.shape[1]):
                    rows.append(
                        {
                            "parameter": "transition_prob",
                            "state": None,
                            "from_state": from_state,
                            "to_state": to_state,
                            "value": float(trans_array[from_state, to_state]),
                        }
                    )

        # State persistence (diagonal of transition matrix)
        state_persistence = self.hmm_params.get("state_persistence", [])
        for state_idx in range(len(state_persistence)):
            rows.append(
                {
                    "parameter": "state_persistence",
                    "state": state_idx,
                    "from_state": None,
                    "to_state": None,
                    "value": float(state_persistence[state_idx]),
                }
            )

        # Expected duration
        expected_durations = self.hmm_params.get("expected_durations", [])
        for state_idx in range(len(expected_durations)):
            rows.append(
                {
                    "parameter": "expected_duration_days",
                    "state": state_idx,
                    "from_state": None,
                    "to_state": None,
                    "value": float(expected_durations[state_idx]),
                }
            )

        return pd.DataFrame(rows)

    def export_to_csv(self, output_dir: str) -> Dict[str, Path]:
        """
        Export accumulated data to CSV files.

        Creates multiple CSV files:
        - timesteps.csv: One row per bar with all data
        - state_probabilities.csv: State probabilities per bar
        - hmm_params.csv: HMM parameters in long format
        - training_history.csv: Model training events
        - regime_changes.csv: Subset of timesteps where regime changed
        - signals.csv: Trading signals

        Args:
            output_dir: Directory to write CSV files

        Returns:
            Dict mapping filename to Path of written CSV
        """
        os.makedirs(output_dir, exist_ok=True)

        csv_files = {}

        # 1. debug_timesteps.csv
        if self.timesteps:
            df_timesteps = pd.DataFrame(self.timesteps)
            # Ensure timestamp is first column
            cols = ["timestamp"] + [c for c in df_timesteps.columns if c != "timestamp"]
            df_timesteps = df_timesteps[cols]

            timesteps_path = os.path.join(output_dir, "timesteps.csv")
            df_timesteps.to_csv(timesteps_path, index=False)
            csv_files["timesteps.csv"] = Path(timesteps_path)

        # 2. debug_state_probabilities.csv
        if self.state_probabilities:
            df_probs = pd.DataFrame(self.state_probabilities)
            probs_path = os.path.join(output_dir, "state_probabilities.csv")
            df_probs.to_csv(probs_path, index=False)
            csv_files["state_probabilities.csv"] = Path(probs_path)

        # 3. debug_hmm_params.csv
        df_params = self._flatten_hmm_params()
        if not df_params.empty:
            params_path = os.path.join(output_dir, "hmm_params.csv")
            df_params.to_csv(params_path, index=False)
            csv_files["hmm_params.csv"] = Path(params_path)

        # 4. debug_training_history.csv
        if self.training_history:
            df_training = pd.DataFrame(self.training_history)
            training_path = os.path.join(output_dir, "training_history.csv")
            df_training.to_csv(training_path, index=False)
            csv_files["training_history.csv"] = Path(training_path)

        # 5. debug_regime_changes.csv
        if self.regime_changes:
            df_regime_changes = pd.DataFrame(self.regime_changes)
            regime_changes_path = os.path.join(output_dir, "regime_changes.csv")
            df_regime_changes.to_csv(regime_changes_path, index=False)
            csv_files["regime_changes.csv"] = Path(regime_changes_path)

        # 6. debug_signals.csv
        if self.signals:
            df_signals = pd.DataFrame(self.signals)
            signals_path = os.path.join(output_dir, "signals.csv")
            df_signals.to_csv(signals_path, index=False)
            csv_files["signals.csv"] = Path(signals_path)

        return csv_files

    def summary(self) -> str:
        """
        Get summary statistics about accumulated data.

        Returns:
            String summary for logging
        """
        lines = [
            f"Debug data for {self.ticker}:",
            f"  Timesteps: {len(self.timesteps)}",
            f"  State probability records: {len(self.state_probabilities)}",
            f"  Training events: {len(self.training_history)}",
            f"  Regime changes: {len(self.regime_changes)}",
            f"  Signals: {len(self.signals)}",
            f"  HMM params recorded: {len(self.hmm_params) > 0}",
        ]
        return "\n".join(lines)
