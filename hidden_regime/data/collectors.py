"""
Model data collection infrastructure for comprehensive pipeline state capture.

Provides ModelDataCollector for capturing detailed model states, parameters,
transitions, and signal generation rationale at each timestep to enable
thorough analysis and educational explanations.
"""

import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
import pandas as pd


@dataclass
class HMMStateSnapshot:
    """Snapshot of HMM model state at a specific timestep."""
    timestamp: str
    data_end_date: str
    transition_matrix: List[List[float]]
    emission_means: List[float]
    emission_stds: List[float]
    initial_probs: List[float]
    regime_probabilities: List[List[float]]  # Per-observation probabilities
    most_likely_states: List[int]  # Viterbi path
    current_regime_probs: List[float]  # Latest observation probabilities
    log_likelihood: float
    training_iterations: int
    converged: bool
    training_time: float
    n_observations: int
    parameter_changes: Dict[str, float]  # Delta from previous snapshot


@dataclass
class TechnicalIndicatorSnapshot:
    """Snapshot of technical indicators and signals at a specific timestep."""
    timestamp: str
    data_end_date: str
    indicators: Dict[str, float]  # Raw indicator values
    signals: Dict[str, str]  # Signal directions (buy/sell/neutral)
    signal_rationale: Dict[str, str]  # Human-readable explanations
    signal_confidence: Dict[str, float]  # Signal strength/conviction
    threshold_crossings: List[Dict[str, Any]]  # Recent threshold events


@dataclass
class RegimeAnalysisSnapshot:
    """Snapshot of regime analysis and interpretation."""
    timestamp: str
    data_end_date: str
    current_regime: str
    regime_confidence: float
    days_in_regime: int
    expected_regime_duration: float
    regime_characteristics: Dict[str, Any]
    transition_probabilities: Dict[str, float]


@dataclass
class TimestepSnapshot:
    """Complete snapshot of all model data at a specific timestep."""
    timestamp: str
    data_end_date: str
    hmm_state: Optional[HMMStateSnapshot]
    technical_indicators: Optional[TechnicalIndicatorSnapshot]
    regime_analysis: Optional[RegimeAnalysisSnapshot]
    execution_time: float
    data_quality_metrics: Dict[str, Any]


class ModelDataCollector:
    """
    Centralized data collection from all pipeline components.

    Captures detailed model states, parameters, and signals at each timestep
    for analysis, visualization, and educational explanation generation.
    """

    def __init__(self,
                 max_history: int = 1000,
                 collection_level: str = "detailed",
                 auto_export: bool = False,
                 export_path: Optional[str] = None):
        """
        Initialize model data collector.

        Args:
            max_history: Maximum number of timesteps to keep in memory
            collection_level: Level of data collection (basic/detailed/full)
            auto_export: Whether to automatically export data periodically
            export_path: Path for automatic exports
        """
        self.max_history = max_history
        self.collection_level = collection_level
        self.auto_export = auto_export
        self.export_path = export_path

        # Use deque for memory-efficient rolling buffer
        self.timestep_data: deque = deque(maxlen=max_history)
        self.collection_stats = {
            "total_timesteps": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "last_collection": None
        }

        # Previous state for computing deltas
        self._previous_hmm_state: Optional[HMMStateSnapshot] = None

    def collect_timestep_data(self,
                             pipeline,
                             data_end_date: str,
                             execution_start_time: float) -> TimestepSnapshot:
        """
        Collect comprehensive data for a single timestep.

        Args:
            pipeline: Pipeline object with all components
            data_end_date: End date of data for this timestep
            execution_start_time: When timestep execution started

        Returns:
            Complete timestep snapshot
        """
        collection_start = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Collect HMM state data
            hmm_snapshot = self._collect_hmm_state(pipeline, timestamp, data_end_date)

            # Collect technical indicator data
            indicator_snapshot = self._collect_indicator_state(pipeline, timestamp, data_end_date)

            # Collect regime analysis data
            regime_snapshot = self._collect_regime_analysis(pipeline, timestamp, data_end_date)

            # Calculate execution time
            execution_time = time.time() - execution_start_time

            # Collect data quality metrics
            quality_metrics = self._collect_data_quality_metrics(pipeline)

            # Create complete snapshot
            snapshot = TimestepSnapshot(
                timestamp=timestamp,
                data_end_date=data_end_date,
                hmm_state=hmm_snapshot,
                technical_indicators=indicator_snapshot,
                regime_analysis=regime_snapshot,
                execution_time=execution_time,
                data_quality_metrics=quality_metrics
            )

            # Add to history
            self.timestep_data.append(snapshot)

            # Update statistics
            self.collection_stats["total_timesteps"] += 1
            self.collection_stats["successful_collections"] += 1
            self.collection_stats["last_collection"] = timestamp

            # Auto-export if enabled
            if self.auto_export and len(self.timestep_data) % 50 == 0:
                self._auto_export()

            return snapshot

        except Exception as e:
            self.collection_stats["failed_collections"] += 1
            # Return minimal snapshot on error
            return TimestepSnapshot(
                timestamp=timestamp,
                data_end_date=data_end_date,
                hmm_state=None,
                technical_indicators=None,
                regime_analysis=None,
                execution_time=time.time() - execution_start_time,
                data_quality_metrics={"collection_error": str(e)}
            )

    def _collect_hmm_state(self, pipeline, timestamp: str, data_end_date: str) -> Optional[HMMStateSnapshot]:
        """Collect HMM model state and parameters."""
        try:
            if not hasattr(pipeline, 'model') or not hasattr(pipeline.model, 'is_fitted'):
                return None

            model = pipeline.model
            if not model.is_fitted:
                return None

            # Get current data
            current_data = pipeline.data.get_all_data()
            if len(current_data) == 0:
                return None

            # Calculate parameter changes from previous state
            parameter_changes = {}
            if self._previous_hmm_state is not None:
                prev_trans = np.array(self._previous_hmm_state.transition_matrix)
                current_trans = model.transition_matrix_
                parameter_changes["transition_matrix_l2_delta"] = float(np.linalg.norm(current_trans - prev_trans))

                prev_means = np.array(self._previous_hmm_state.emission_means)
                current_means = model.emission_means_
                parameter_changes["emission_means_l2_delta"] = float(np.linalg.norm(current_means - prev_means))

                prev_stds = np.array(self._previous_hmm_state.emission_stds)
                current_stds = model.emission_stds_
                parameter_changes["emission_stds_l2_delta"] = float(np.linalg.norm(current_stds - prev_stds))

            # Get regime probabilities and states
            regime_probs_df = model.predict_proba(current_data)
            regime_probs = regime_probs_df.values if isinstance(regime_probs_df, pd.DataFrame) else regime_probs_df

            most_likely_states = model.predict(current_data)
            current_regime_probs = regime_probs[-1].tolist() if len(regime_probs) > 0 else []

            # Get training history
            training_history = getattr(model, 'training_history_', {})

            snapshot = HMMStateSnapshot(
                timestamp=timestamp,
                data_end_date=data_end_date,
                transition_matrix=model.transition_matrix_.tolist(),
                emission_means=model.emission_means_.tolist(),
                emission_stds=model.emission_stds_.tolist(),
                initial_probs=model.initial_probs_.tolist(),
                regime_probabilities=regime_probs.tolist(),
                most_likely_states=most_likely_states.tolist(),
                current_regime_probs=current_regime_probs,
                log_likelihood=float(model.score(current_data)),
                training_iterations=training_history.get('iterations', 0),
                converged=training_history.get('converged', False),
                training_time=training_history.get('training_time', 0.0),
                n_observations=len(current_data),
                parameter_changes=parameter_changes
            )

            # Store for next comparison
            self._previous_hmm_state = snapshot

            return snapshot

        except Exception as e:
            # Return None on any error
            return None

    def _collect_indicator_state(self, pipeline, timestamp: str, data_end_date: str) -> Optional[TechnicalIndicatorSnapshot]:
        """Collect technical indicator values and signals."""
        try:
            # Check if analysis component has indicator data
            if not hasattr(pipeline, 'analysis'):
                return None

            analysis = pipeline.analysis

            # Try to get indicator data from analysis component
            # This is a placeholder - actual implementation depends on analysis structure
            indicators = {}
            signals = {}
            signal_rationale = {}
            signal_confidence = {}
            threshold_crossings = []

            # TODO: Extract actual indicator data from analysis component
            # This would need to be customized based on how indicators are stored

            return TechnicalIndicatorSnapshot(
                timestamp=timestamp,
                data_end_date=data_end_date,
                indicators=indicators,
                signals=signals,
                signal_rationale=signal_rationale,
                signal_confidence=signal_confidence,
                threshold_crossings=threshold_crossings
            )

        except Exception as e:
            return None

    def _collect_regime_analysis(self, pipeline, timestamp: str, data_end_date: str) -> Optional[RegimeAnalysisSnapshot]:
        """Collect regime analysis and interpretation."""
        try:
            if not hasattr(pipeline, 'analysis') or not hasattr(pipeline, 'model'):
                return None

            model = pipeline.model
            if not model.is_fitted:
                return None

            current_data = pipeline.data.get_all_data()
            if len(current_data) == 0:
                return None

            # Get current regime information
            regime_probs_df = model.predict_proba(current_data)
            regime_probs = regime_probs_df.values if isinstance(regime_probs_df, pd.DataFrame) else regime_probs_df
            current_regime_probs = regime_probs[-1] if len(regime_probs) > 0 else np.array([])

            # Map to regime names (basic implementation)
            regime_names = ["Bear", "Sideways", "Bull"]
            current_regime_idx = np.argmax(current_regime_probs) if len(current_regime_probs) > 0 else 0
            current_regime = regime_names[current_regime_idx] if current_regime_idx < len(regime_names) else f"State_{current_regime_idx}"
            regime_confidence = float(current_regime_probs[current_regime_idx]) if len(current_regime_probs) > 0 else 0.0

            # Calculate regime persistence (simplified)
            most_likely_states = model.predict(current_data)
            days_in_regime = 1
            if len(most_likely_states) > 1:
                # Count consecutive days in current regime
                for i in range(len(most_likely_states) - 1, 0, -1):
                    if most_likely_states[i] == most_likely_states[-1]:
                        days_in_regime += 1
                    else:
                        break

            # Estimate expected duration from transition matrix
            if hasattr(model, 'transition_matrix_'):
                diag_prob = model.transition_matrix_[current_regime_idx, current_regime_idx]
                expected_duration = 1.0 / (1.0 - diag_prob) if diag_prob < 1.0 else float('inf')
            else:
                expected_duration = 10.0  # Default

            return RegimeAnalysisSnapshot(
                timestamp=timestamp,
                data_end_date=data_end_date,
                current_regime=current_regime,
                regime_confidence=regime_confidence,
                days_in_regime=days_in_regime,
                expected_regime_duration=expected_duration,
                regime_characteristics={
                    "mean_return": float(model.emission_means_[current_regime_idx]) if hasattr(model, 'emission_means_') else 0.0,
                    "volatility": float(model.emission_stds_[current_regime_idx]) if hasattr(model, 'emission_stds_') else 0.0
                },
                transition_probabilities={
                    regime_names[i]: float(model.transition_matrix_[current_regime_idx, i])
                    for i in range(min(len(regime_names), model.transition_matrix_.shape[1]))
                } if hasattr(model, 'transition_matrix_') else {}
            )

        except Exception as e:
            return None

    def _collect_data_quality_metrics(self, pipeline) -> Dict[str, Any]:
        """Collect data quality and pipeline health metrics."""
        try:
            metrics = {}

            # Data component metrics
            if hasattr(pipeline, 'data'):
                data = pipeline.data.get_all_data()
                metrics["data_size"] = len(data)
                metrics["data_completeness"] = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
                metrics["data_start"] = str(data.index[0]) if len(data) > 0 else None
                metrics["data_end"] = str(data.index[-1]) if len(data) > 0 else None

            # Model component metrics
            if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'is_fitted'):
                metrics["model_fitted"] = pipeline.model.is_fitted
                if pipeline.model.is_fitted:
                    metrics["model_complexity"] = getattr(pipeline.model, 'n_states', 0)

            # Pipeline component availability
            metrics["components_available"] = {
                "data": hasattr(pipeline, 'data'),
                "observations": hasattr(pipeline, 'observations'),
                "model": hasattr(pipeline, 'model'),
                "analysis": hasattr(pipeline, 'analysis'),
                "report": hasattr(pipeline, 'report')
            }

            return metrics

        except Exception as e:
            return {"collection_error": str(e)}

    def _auto_export(self):
        """Automatically export data if configured."""
        if self.export_path:
            try:
                self.export_to_json(self.export_path)
            except Exception:
                pass  # Silent failure for auto-export

    def get_latest_snapshot(self) -> Optional[TimestepSnapshot]:
        """Get the most recent timestep snapshot."""
        return self.timestep_data[-1] if self.timestep_data else None

    def get_history(self, n_timesteps: Optional[int] = None) -> List[TimestepSnapshot]:
        """Get historical timestep data."""
        if n_timesteps is None:
            return list(self.timestep_data)
        else:
            return list(self.timestep_data)[-n_timesteps:]

    def get_hmm_parameter_evolution(self) -> Dict[str, List[Any]]:
        """Get evolution of HMM parameters over time."""
        evolution = {
            "timestamps": [],
            "transition_matrices": [],
            "emission_means": [],
            "emission_stds": [],
            "log_likelihoods": [],
            "parameter_deltas": []
        }

        for snapshot in self.timestep_data:
            if snapshot.hmm_state:
                evolution["timestamps"].append(snapshot.timestamp)
                evolution["transition_matrices"].append(snapshot.hmm_state.transition_matrix)
                evolution["emission_means"].append(snapshot.hmm_state.emission_means)
                evolution["emission_stds"].append(snapshot.hmm_state.emission_stds)
                evolution["log_likelihoods"].append(snapshot.hmm_state.log_likelihood)
                evolution["parameter_deltas"].append(snapshot.hmm_state.parameter_changes)

        return evolution

    def export_to_json(self, filepath: str):
        """Export collected data to JSON file."""
        # Convert dataclasses to dictionaries for JSON serialization
        export_data = {
            "collection_stats": self.collection_stats,
            "timesteps": [asdict(snapshot) for snapshot in self.timestep_data]
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

    def export_to_parquet(self, filepath: str):
        """Export collected data to Parquet file for efficient analysis."""
        # Flatten data structure for tabular format
        flattened_data = []

        for snapshot in self.timestep_data:
            row = {
                "timestamp": snapshot.timestamp,
                "data_end_date": snapshot.data_end_date,
                "execution_time": snapshot.execution_time
            }

            # Add HMM state data
            if snapshot.hmm_state:
                row.update({
                    "hmm_log_likelihood": snapshot.hmm_state.log_likelihood,
                    "hmm_training_iterations": snapshot.hmm_state.training_iterations,
                    "hmm_converged": snapshot.hmm_state.converged,
                    "hmm_n_observations": snapshot.hmm_state.n_observations,
                    "hmm_current_regime_0": snapshot.hmm_state.current_regime_probs[0] if snapshot.hmm_state.current_regime_probs else None,
                    "hmm_current_regime_1": snapshot.hmm_state.current_regime_probs[1] if len(snapshot.hmm_state.current_regime_probs) > 1 else None,
                    "hmm_current_regime_2": snapshot.hmm_state.current_regime_probs[2] if len(snapshot.hmm_state.current_regime_probs) > 2 else None
                })

            # Add regime analysis data
            if snapshot.regime_analysis:
                row.update({
                    "regime_current": snapshot.regime_analysis.current_regime,
                    "regime_confidence": snapshot.regime_analysis.regime_confidence,
                    "regime_days": snapshot.regime_analysis.days_in_regime,
                    "regime_expected_duration": snapshot.regime_analysis.expected_regime_duration
                })

            flattened_data.append(row)

        df = pd.DataFrame(flattened_data)
        df.to_parquet(filepath)

    def clear_history(self):
        """Clear all collected data."""
        self.timestep_data.clear()
        self._previous_hmm_state = None
        self.collection_stats = {
            "total_timesteps": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "last_collection": None
        }

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about collected data."""
        if not self.timestep_data:
            return {"message": "No data collected yet"}

        successful_hmm = sum(1 for s in self.timestep_data if s.hmm_state is not None)
        successful_indicators = sum(1 for s in self.timestep_data if s.technical_indicators is not None)
        successful_regime = sum(1 for s in self.timestep_data if s.regime_analysis is not None)

        execution_times = [s.execution_time for s in self.timestep_data]

        return {
            "total_timesteps": len(self.timestep_data),
            "successful_hmm_collection": successful_hmm,
            "successful_indicator_collection": successful_indicators,
            "successful_regime_collection": successful_regime,
            "collection_success_rate": successful_hmm / len(self.timestep_data),
            "avg_execution_time": np.mean(execution_times),
            "min_execution_time": np.min(execution_times),
            "max_execution_time": np.max(execution_times),
            "date_range": {
                "start": self.timestep_data[0].data_end_date,
                "end": self.timestep_data[-1].data_end_date
            }
        }