"""
Factory patterns for hidden-regime pipeline components.

Provides factory functions and classes for creating pipeline components
from configuration objects, enabling easy extensibility and consistent
component instantiation.
"""

from .pipeline import PipelineFactory
from .components import ComponentFactory

__all__ = [
    "PipelineFactory",
    "ComponentFactory",
]