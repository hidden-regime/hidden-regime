"""
Visualization utilities for hidden-regime package.

Provides consistent plotting functionality and styling across all components
including data loaders, HMM models, and state standardizers.
"""

from .plotting import (
    create_regime_legend,
    format_financial_axis,
    get_regime_colors,
    plot_regime_heatmap,
    plot_regime_statistics,
    plot_returns_with_regimes,
    setup_financial_plot_style,
)

__all__ = [
    "setup_financial_plot_style",
    "plot_returns_with_regimes",
    "plot_regime_heatmap",
    "plot_regime_statistics",
    "get_regime_colors",
    "format_financial_axis",
    "create_regime_legend",
]
