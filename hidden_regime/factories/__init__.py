"""
Factory patterns for hidden-regime pipeline components.

Provides factory functions and classes for creating pipeline components
from configuration objects, enabling easy extensibility and consistent
component instantiation.
"""

from .pipeline import PipelineFactory, pipeline_factory
from .components import ComponentFactory, component_factory

__all__ = [
    "PipelineFactory",
    "ComponentFactory",
    "pipeline_factory",
    "component_factory",
]