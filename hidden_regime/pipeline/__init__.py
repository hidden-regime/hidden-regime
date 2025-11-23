"""
Pipeline infrastructure for hidden-regime package.

Provides the core Pipeline class and supporting components for building
Data → Observations → Model → Interpreter → Signal Generator → Report workflows
with temporal data isolation for rigorous backtesting verification and validation.

Key Components:
- Pipeline: Main orchestrator class
- TemporalController: Prevents temporal data leakage during backtesting
- Component interfaces: Standardized interfaces for all pipeline components

Architecture:
    Data → Observations → Model → Interpreter → Signal Generator → Report

    - Model: Pure math (HMM algorithms)
    - Interpreter: Domain knowledge (regime labels, characteristics)
    - Signal Generator: Trading logic (position sizing, entry/exit)
    - Report: Metrics, visualizations, summaries
"""

from .core import Pipeline
from .interfaces import (
    DataComponent,
    InterpreterComponent,
    ModelComponent,
    ObservationComponent,
    PipelineComponent,
    ReportComponent,
    SignalGeneratorComponent,
)
from .schemas import (
    INTERPRETER_OUTPUT_SCHEMA,
    MODEL_OUTPUT_SCHEMA,
    SIGNAL_OUTPUT_SCHEMA,
    OutputSchema,
    assert_valid_output,
    validate_component_output,
)
from .temporal import TemporalController, TemporalDataStub

__all__ = [
    # Core pipeline
    "Pipeline",
    # Temporal V&V components
    "TemporalController",
    "TemporalDataStub",
    # Component interfaces
    "PipelineComponent",
    "DataComponent",
    "ObservationComponent",
    "ModelComponent",
    "InterpreterComponent",  # NEW
    # AnalysisComponent removed in 2.0.0 - use InterpreterComponent
    "SignalGeneratorComponent",  # NEW
    "ReportComponent",
    # Output schemas and validation
    "OutputSchema",
    "MODEL_OUTPUT_SCHEMA",
    "INTERPRETER_OUTPUT_SCHEMA",
    "SIGNAL_OUTPUT_SCHEMA",
    "validate_component_output",
    "assert_valid_output",
]
