"""
Structured Data Export for Hidden Regime Analysis.

Provides comprehensive export capabilities for model data, analysis results,
and temporal snapshots in multiple formats including JSON, Parquet, and HDF5.
"""

import gzip
import json
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    warnings.warn("PyArrow not available - Parquet export disabled")

try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    warnings.warn("h5py not available - HDF5 export disabled")

from ..analysis.regime_evolution import RegimeEvolutionAnalyzer
from ..analysis.signal_attribution import SignalAttributionAnalyzer
from ..utils.exceptions import AnalysisError
from .collectors import HMMStateSnapshot, ModelDataCollector, TimestepSnapshot


class StructuredDataExporter:
    """
    Comprehensive data exporter for hidden regime analysis results.

    Supports multiple export formats and provides data structure optimization
    for different use cases including analysis reproduction, model persistence,
    and data sharing.
    """

    def __init__(self, base_output_dir: str = "exports", compress: bool = True):
        """
        Initialize data exporter.

        Args:
            base_output_dir: Base directory for exports
            compress: Whether to compress output files
        """
        self.base_output_dir = Path(base_output_dir)
        self.compress = compress
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Track exported files
        self.export_log: List[Dict[str, Any]] = []

    def export_temporal_snapshots(
        self,
        temporal_snapshots: List[Dict[str, Any]],
        format: str = "json",
        filename: Optional[str] = None,
    ) -> str:
        """
        Export temporal snapshots to specified format.

        Args:
            temporal_snapshots: List of timestep snapshots
            format: Export format ('json', 'parquet', 'hdf5', 'pickle')
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if not temporal_snapshots:
            raise AnalysisError("No temporal snapshots to export")

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"temporal_snapshots_{timestamp}"

        # Convert snapshots to exportable format
        exportable_data = self._prepare_temporal_data_for_export(temporal_snapshots)

        # Export based on format
        if format.lower() == "json":
            return self._export_json(exportable_data, filename, "temporal_snapshots")
        elif format.lower() == "parquet":
            return self._export_parquet(exportable_data, filename, "temporal_snapshots")
        elif format.lower() == "hdf5":
            return self._export_hdf5(exportable_data, filename, "temporal_snapshots")
        elif format.lower() == "pickle":
            return self._export_pickle(
                temporal_snapshots, filename, "temporal_snapshots"
            )  # Use original data
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def export_regime_evolution_results(
        self,
        evolution_results: Dict[str, Any],
        format: str = "json",
        filename: Optional[str] = None,
    ) -> str:
        """
        Export regime evolution analysis results.

        Args:
            evolution_results: Results from RegimeEvolutionAnalyzer
            format: Export format
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"regime_evolution_{timestamp}"

        # Prepare data for export
        exportable_data = self._prepare_evolution_data_for_export(evolution_results)

        # Export based on format
        if format.lower() == "json":
            return self._export_json(exportable_data, filename, "regime_evolution")
        elif format.lower() == "parquet":
            return self._export_parquet(exportable_data, filename, "regime_evolution")
        elif format.lower() == "hdf5":
            return self._export_hdf5(exportable_data, filename, "regime_evolution")
        elif format.lower() == "pickle":
            return self._export_pickle(evolution_results, filename, "regime_evolution")
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def export_signal_attribution_results(
        self,
        attribution_results: Dict[str, Any],
        format: str = "json",
        filename: Optional[str] = None,
    ) -> str:
        """
        Export signal attribution analysis results.

        Args:
            attribution_results: Results from SignalAttributionAnalyzer
            format: Export format
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"signal_attribution_{timestamp}"

        # Prepare data for export
        exportable_data = self._prepare_attribution_data_for_export(attribution_results)

        # Export based on format
        if format.lower() == "json":
            return self._export_json(exportable_data, filename, "signal_attribution")
        elif format.lower() == "parquet":
            return self._export_parquet(exportable_data, filename, "signal_attribution")
        elif format.lower() == "hdf5":
            return self._export_hdf5(exportable_data, filename, "signal_attribution")
        elif format.lower() == "pickle":
            return self._export_pickle(
                attribution_results, filename, "signal_attribution"
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def export_comprehensive_analysis(
        self,
        temporal_snapshots: List[Dict[str, Any]],
        evolution_results: Optional[Dict[str, Any]] = None,
        attribution_results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "json",
        filename: Optional[str] = None,
    ) -> str:
        """
        Export comprehensive analysis package with all results.

        Args:
            temporal_snapshots: List of timestep snapshots
            evolution_results: Optional regime evolution results
            attribution_results: Optional signal attribution results
            metadata: Optional additional metadata
            format: Export format
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_analysis_{timestamp}"

        # Build comprehensive package
        comprehensive_data = {
            "export_metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "format": format,
                "data_includes": [],
                "total_timesteps": len(temporal_snapshots),
                "exporter_version": "1.0.0",
            },
            "temporal_snapshots": None,
            "regime_evolution": None,
            "signal_attribution": None,
            "additional_metadata": metadata or {},
        }

        # Add temporal snapshots
        if temporal_snapshots:
            comprehensive_data["temporal_snapshots"] = (
                self._prepare_temporal_data_for_export(temporal_snapshots)
            )
            comprehensive_data["export_metadata"]["data_includes"].append(
                "temporal_snapshots"
            )

        # Add evolution results
        if evolution_results:
            comprehensive_data["regime_evolution"] = (
                self._prepare_evolution_data_for_export(evolution_results)
            )
            comprehensive_data["export_metadata"]["data_includes"].append(
                "regime_evolution"
            )

        # Add attribution results
        if attribution_results:
            comprehensive_data["signal_attribution"] = (
                self._prepare_attribution_data_for_export(attribution_results)
            )
            comprehensive_data["export_metadata"]["data_includes"].append(
                "signal_attribution"
            )

        # Export based on format
        if format.lower() == "json":
            return self._export_json(
                comprehensive_data, filename, "comprehensive_analysis"
            )
        elif format.lower() == "parquet":
            return self._export_parquet(
                comprehensive_data, filename, "comprehensive_analysis"
            )
        elif format.lower() == "hdf5":
            return self._export_hdf5(
                comprehensive_data, filename, "comprehensive_analysis"
            )
        elif format.lower() == "pickle":
            # For pickle, include original objects
            original_data = {
                "temporal_snapshots": temporal_snapshots,
                "evolution_results": evolution_results,
                "attribution_results": attribution_results,
                "metadata": metadata,
                "export_metadata": comprehensive_data["export_metadata"],
            }
            return self._export_pickle(
                original_data, filename, "comprehensive_analysis"
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def export_data_for_reproduction(
        self,
        temporal_snapshots: List[Dict[str, Any]],
        model_configs: Dict[str, Any],
        analysis_configs: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> str:
        """
        Export data package optimized for analysis reproduction.

        Args:
            temporal_snapshots: List of timestep snapshots
            model_configs: Model configuration parameters
            analysis_configs: Analysis configuration parameters
            filename: Optional custom filename

        Returns:
            Path to exported reproduction package
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reproduction_package_{timestamp}"

        # Build reproduction package
        reproduction_data = {
            "reproduction_metadata": {
                "created_timestamp": datetime.now().isoformat(),
                "package_version": "1.0.0",
                "purpose": "analysis_reproduction",
                "data_quality_checks": self._perform_data_quality_checks(
                    temporal_snapshots
                ),
            },
            "model_configurations": model_configs,
            "analysis_configurations": analysis_configs,
            "temporal_data": self._prepare_temporal_data_for_export(temporal_snapshots),
            "reproduction_instructions": {
                "steps": [
                    "1. Load temporal snapshots and configurations",
                    "2. Initialize models with provided configurations",
                    "3. Run analysis pipeline with temporal data",
                    "4. Compare results with provided baseline metrics",
                ],
                "required_packages": [
                    "hidden-regime",
                    "numpy",
                    "pandas",
                    "matplotlib",
                    "scikit-learn",
                ],
            },
        }

        # Always export as pickle for reproduction (preserves exact object state)
        return self._export_pickle(reproduction_data, filename, "reproduction_package")

    def _prepare_temporal_data_for_export(
        self, temporal_snapshots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare temporal snapshots for export by converting to serializable format."""
        exportable_snapshots = []

        for snapshot_data in temporal_snapshots:
            snapshot = snapshot_data.get("snapshot")
            exportable_snapshot = {
                "timestamp": snapshot_data.get("timestamp"),
                "temporal_context": snapshot_data.get("temporal_context", {}),
                "hmm_state": None,
                "technical_indicators": None,
                "regime_analysis": None,
            }

            # Convert HMM state
            if snapshot and hasattr(snapshot, "hmm_state") and snapshot.hmm_state:
                hmm_state = snapshot.hmm_state
                exportable_snapshot["hmm_state"] = {
                    "timestamp": hmm_state.timestamp,
                    "data_end_date": hmm_state.data_end_date,
                    "transition_matrix": hmm_state.transition_matrix,
                    "emission_means": hmm_state.emission_means,
                    "emission_stds": hmm_state.emission_stds,
                    "regime_probabilities": hmm_state.regime_probabilities,
                    "most_likely_states": hmm_state.most_likely_states,
                    "model_performance": hmm_state.model_performance,
                    "training_info": hmm_state.training_info,
                    "convergence_metrics": hmm_state.convergence_metrics,
                }

            # Convert technical indicators
            if (
                snapshot
                and hasattr(snapshot, "technical_indicators")
                and snapshot.technical_indicators
            ):
                exportable_snapshot["technical_indicators"] = (
                    snapshot.technical_indicators
                )

            # Convert regime analysis
            if (
                snapshot
                and hasattr(snapshot, "regime_analysis")
                and snapshot.regime_analysis
            ):
                regime_analysis = snapshot.regime_analysis
                exportable_snapshot["regime_analysis"] = {
                    "timestamp": regime_analysis.timestamp,
                    "current_regime": regime_analysis.current_regime,
                    "regime_confidence": regime_analysis.regime_confidence,
                    "regime_duration": regime_analysis.regime_duration,
                    "regime_characteristics": regime_analysis.regime_characteristics,
                    "transition_probabilities": regime_analysis.transition_probabilities,
                    "risk_metrics": regime_analysis.risk_metrics,
                }

            exportable_snapshots.append(exportable_snapshot)

        return {
            "snapshots": exportable_snapshots,
            "total_count": len(exportable_snapshots),
            "date_range": {
                "start": (
                    exportable_snapshots[0]["timestamp"]
                    if exportable_snapshots
                    else None
                ),
                "end": (
                    exportable_snapshots[-1]["timestamp"]
                    if exportable_snapshots
                    else None
                ),
            },
        }

    def _prepare_evolution_data_for_export(
        self, evolution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare regime evolution results for export."""
        return {
            "analysis_metadata": {
                "analysis_timestamp": evolution_results.get("analysis_timestamp"),
                "total_timesteps": evolution_results.get("total_timesteps", 0),
            },
            "parameter_evolution": evolution_results.get("parameter_evolution", {}),
            "regime_transitions": evolution_results.get("regime_transitions", []),
            "stability_metrics": evolution_results.get("stability_metrics", {}),
            "regime_characteristics": evolution_results.get(
                "regime_characteristics", {}
            ),
            "drift_analysis": evolution_results.get("drift_analysis", {}),
            "summary_insights": evolution_results.get("summary_insights", {}),
        }

    def _prepare_attribution_data_for_export(
        self, attribution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare signal attribution results for export."""
        return {
            "analysis_metadata": {
                "analysis_timestamp": attribution_results.get("analysis_timestamp"),
                "total_timesteps": attribution_results.get("total_timesteps", 0),
            },
            "signal_performances": attribution_results.get("signal_performances", {}),
            "signal_interactions": attribution_results.get("signal_interactions", {}),
            "time_based_attribution": attribution_results.get(
                "time_based_attribution", {}
            ),
            "portfolio_attribution": attribution_results.get(
                "portfolio_attribution", {}
            ),
            "attribution_insights": attribution_results.get("attribution_insights", {}),
            "quality_metrics": attribution_results.get("quality_metrics", {}),
        }

    def _export_json(self, data: Dict[str, Any], filename: str, data_type: str) -> str:
        """Export data to JSON format."""
        output_path = self.base_output_dir / f"{filename}.json"

        # Handle numpy arrays and other non-serializable objects
        json_data = self._convert_to_json_serializable(data)

        if self.compress:
            output_path = self.base_output_dir / f"{filename}.json.gz"
            with gzip.open(output_path, "wt", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, separators=(",", ": "))
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, separators=(",", ": "))

        self._log_export(str(output_path), "json", data_type, len(str(json_data)))
        return str(output_path)

    def _export_parquet(
        self, data: Dict[str, Any], filename: str, data_type: str
    ) -> str:
        """Export data to Parquet format."""
        if not PARQUET_AVAILABLE:
            raise AnalysisError("Parquet export requires pyarrow package")

        output_path = self.base_output_dir / f"{filename}.parquet"

        # Convert data to DataFrame format for Parquet
        parquet_data = self._convert_to_parquet_format(data)

        # Write to Parquet
        table = pa.Table.from_pandas(parquet_data)
        pq.write_table(
            table, output_path, compression="snappy" if self.compress else None
        )

        self._log_export(
            str(output_path), "parquet", data_type, output_path.stat().st_size
        )
        return str(output_path)

    def _export_hdf5(self, data: Dict[str, Any], filename: str, data_type: str) -> str:
        """Export data to HDF5 format."""
        if not HDF5_AVAILABLE:
            raise AnalysisError("HDF5 export requires h5py package")

        output_path = self.base_output_dir / f"{filename}.h5"

        with h5py.File(output_path, "w") as f:
            self._write_to_hdf5_group(f, data, "")

        self._log_export(
            str(output_path), "hdf5", data_type, output_path.stat().st_size
        )
        return str(output_path)

    def _export_pickle(self, data: Any, filename: str, data_type: str) -> str:
        """Export data to pickle format."""
        output_path = self.base_output_dir / f"{filename}.pkl"

        if self.compress:
            output_path = self.base_output_dir / f"{filename}.pkl.gz"
            with gzip.open(output_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(output_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._log_export(
            str(output_path), "pickle", data_type, output_path.stat().st_size
        )
        return str(output_path)

    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {
                key: self._convert_to_json_serializable(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        else:
            return obj

    def _convert_to_parquet_format(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert nested data to flattened DataFrame format for Parquet."""
        flattened_data = {}
        self._flatten_dict(data, "", flattened_data)

        # Convert to DataFrame
        max_length = max(
            len(v) if isinstance(v, list) else 1 for v in flattened_data.values()
        )

        df_data = {}
        for key, value in flattened_data.items():
            if isinstance(value, list):
                # Pad list to max length
                padded_list = value + [None] * (max_length - len(value))
                df_data[key] = padded_list
            else:
                # Repeat scalar value
                df_data[key] = [value] * max_length

        return pd.DataFrame(df_data)

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str, result: Dict[str, Any]
    ) -> None:
        """Flatten nested dictionary for DataFrame conversion."""
        for key, value in d.items():
            new_key = f"{parent_key}_{key}" if parent_key else key

            if isinstance(value, dict):
                self._flatten_dict(value, new_key, result)
            elif (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], dict)
            ):
                # Handle list of dictionaries
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._flatten_dict(item, f"{new_key}_{i}", result)
                    else:
                        result[f"{new_key}_{i}"] = item
            else:
                result[new_key] = value

    def _write_to_hdf5_group(
        self, group: h5py.Group, data: Dict[str, Any], prefix: str
    ) -> None:
        """Recursively write data to HDF5 group."""
        for key, value in data.items():
            full_key = f"{prefix}/{key}" if prefix else key

            if isinstance(value, dict):
                subgroup = group.create_group(full_key)
                self._write_to_hdf5_group(subgroup, value, "")
            elif isinstance(value, list):
                if len(value) > 0:
                    # Try to convert to numpy array
                    try:
                        array_data = np.array(value)
                        group.create_dataset(full_key, data=array_data)
                    except (ValueError, TypeError):
                        # Store as string if conversion fails
                        group.create_dataset(full_key, data=str(value))
                else:
                    group.create_dataset(full_key, data=[])
            elif isinstance(value, (np.ndarray, int, float, str, bool)):
                group.create_dataset(full_key, data=value)
            elif value is None:
                group.create_dataset(full_key, data="null")
            else:
                # Convert to string for unsupported types
                group.create_dataset(full_key, data=str(value))

    def _perform_data_quality_checks(
        self, temporal_snapshots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform data quality checks for reproduction package."""
        checks = {
            "total_snapshots": len(temporal_snapshots),
            "snapshots_with_hmm_data": 0,
            "snapshots_with_signal_data": 0,
            "temporal_consistency": True,
            "data_completeness_score": 0.0,
        }

        if not temporal_snapshots:
            return checks

        # Check data availability
        for snapshot_data in temporal_snapshots:
            snapshot = snapshot_data.get("snapshot")
            if snapshot:
                if hasattr(snapshot, "hmm_state") and snapshot.hmm_state:
                    checks["snapshots_with_hmm_data"] += 1
                if (
                    hasattr(snapshot, "technical_indicators")
                    and snapshot.technical_indicators
                ):
                    checks["snapshots_with_signal_data"] += 1

        # Calculate completeness score
        hmm_completeness = checks["snapshots_with_hmm_data"] / len(temporal_snapshots)
        signal_completeness = checks["snapshots_with_signal_data"] / len(
            temporal_snapshots
        )
        checks["data_completeness_score"] = (hmm_completeness + signal_completeness) / 2

        # Check temporal consistency
        timestamps = [pd.to_datetime(s["timestamp"]) for s in temporal_snapshots]
        timestamps.sort()
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i - 1]:
                checks["temporal_consistency"] = False
                break

        return checks

    def _log_export(
        self, filepath: str, format: str, data_type: str, file_size: int
    ) -> None:
        """Log export operation."""
        self.export_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "filepath": filepath,
                "format": format,
                "data_type": data_type,
                "file_size_bytes": file_size,
                "compressed": self.compress,
            }
        )

    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of all export operations."""
        if not self.export_log:
            return {"no_exports": True}

        total_files = len(self.export_log)
        total_size = sum(log["file_size_bytes"] for log in self.export_log)
        formats_used = list(set(log["format"] for log in self.export_log))
        data_types_exported = list(set(log["data_type"] for log in self.export_log))

        return {
            "total_exports": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "formats_used": formats_used,
            "data_types_exported": data_types_exported,
            "compression_enabled": self.compress,
            "export_directory": str(self.base_output_dir),
            "latest_export": (
                self.export_log[-1]["timestamp"] if self.export_log else None
            ),
        }

    def create_export_manifest(self) -> str:
        """Create manifest file listing all exports."""
        manifest_path = self.base_output_dir / "export_manifest.json"

        manifest_data = {
            "manifest_created": datetime.now().isoformat(),
            "export_summary": self.get_export_summary(),
            "exported_files": self.export_log,
            "supported_formats": {
                "json": {
                    "available": True,
                    "description": "JSON format with optional compression",
                },
                "parquet": {
                    "available": PARQUET_AVAILABLE,
                    "description": "Apache Parquet columnar format",
                },
                "hdf5": {
                    "available": HDF5_AVAILABLE,
                    "description": "HDF5 hierarchical data format",
                },
                "pickle": {
                    "available": True,
                    "description": "Python pickle format (preserves exact objects)",
                },
            },
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return str(manifest_path)


class DataImporter:
    """
    Complementary data importer for loading exported analysis results.

    Provides functionality to load and reconstruct analysis data from
    various export formats for further analysis or reproduction.
    """

    def __init__(self):
        """Initialize data importer."""
        pass

    def load_temporal_snapshots(self, filepath: str) -> List[Dict[str, Any]]:
        """Load temporal snapshots from exported file."""
        format = self._detect_format(filepath)

        if format == "json":
            return self._load_json(filepath)["snapshots"]
        elif format == "pickle":
            return self._load_pickle(filepath)
        elif format == "parquet":
            df = self._load_parquet(filepath)
            return self._reconstruct_temporal_data_from_df(df)
        elif format == "hdf5":
            return self._load_hdf5(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_analysis_results(self, filepath: str) -> Dict[str, Any]:
        """Load analysis results from exported file."""
        format = self._detect_format(filepath)

        if format == "json":
            return self._load_json(filepath)
        elif format == "pickle":
            return self._load_pickle(filepath)
        elif format == "parquet":
            df = self._load_parquet(filepath)
            return self._reconstruct_analysis_data_from_df(df)
        elif format == "hdf5":
            return self._load_hdf5(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _detect_format(self, filepath: str) -> str:
        """Detect file format from extension."""
        path = Path(filepath)
        if path.suffix == ".gz":
            # Handle compressed files
            if path.stem.endswith(".json"):
                return "json"
            elif path.stem.endswith(".pkl"):
                return "pickle"
        elif path.suffix == ".json":
            return "json"
        elif path.suffix == ".parquet":
            return "parquet"
        elif path.suffix in [".h5", ".hdf5"]:
            return "hdf5"
        elif path.suffix == ".pkl":
            return "pickle"
        else:
            raise ValueError(f"Cannot detect format for file: {filepath}")

    def _load_json(self, filepath: str) -> Dict[str, Any]:
        """Load JSON file (with optional compression)."""
        path = Path(filepath)
        if path.suffix == ".gz":
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)

    def _load_pickle(self, filepath: str) -> Any:
        """Load pickle file (with optional compression)."""
        path = Path(filepath)
        if path.suffix == ".gz":
            with gzip.open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            with open(filepath, "rb") as f:
                return pickle.load(f)

    def _load_parquet(self, filepath: str) -> pd.DataFrame:
        """Load Parquet file."""
        if not PARQUET_AVAILABLE:
            raise AnalysisError("Parquet loading requires pyarrow package")
        return pd.read_parquet(filepath)

    def _load_hdf5(self, filepath: str) -> Dict[str, Any]:
        """Load HDF5 file."""
        if not HDF5_AVAILABLE:
            raise AnalysisError("HDF5 loading requires h5py package")

        data = {}
        with h5py.File(filepath, "r") as f:
            self._read_hdf5_group(f, data)
        return data

    def _read_hdf5_group(self, group: h5py.Group, result: Dict[str, Any]) -> None:
        """Recursively read HDF5 group."""
        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                result[key] = {}
                self._read_hdf5_group(group[key], result[key])
            else:
                # Dataset
                data = group[key][()]
                if isinstance(data, bytes):
                    result[key] = data.decode("utf-8")
                else:
                    result[key] = data

    def _reconstruct_temporal_data_from_df(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Reconstruct temporal snapshots from DataFrame."""
        # This is a simplified reconstruction - full implementation would be more complex
        # depending on how the data was flattened
        return [{"reconstructed_from_parquet": True, "data": df.to_dict("records")}]

    def _reconstruct_analysis_data_from_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Reconstruct analysis results from DataFrame."""
        # This is a simplified reconstruction
        return {"reconstructed_from_parquet": True, "data": df.to_dict("records")}
