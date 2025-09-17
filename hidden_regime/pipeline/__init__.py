"""
Pipeline infrastructure for hidden-regime package.

Provides the core Pipeline class and supporting components for building
Data → Observation → Model → Analysis → Report workflows with temporal
data isolation for rigorous backtesting verification and validation.

Key Components:
- Pipeline: Main orchestrator class
- TemporalController: Prevents temporal data leakage during backtesting
- Component interfaces: Standardized interfaces for all pipeline components
"""

from .core import Pipeline
from .temporal import TemporalController, TemporalDataStub
from .interfaces import (
    PipelineComponent,
    DataComponent,
    ObservationComponent, 
    ModelComponent,
    AnalysisComponent,
    ReportComponent,
)

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
    "AnalysisComponent",
    "ReportComponent",
]