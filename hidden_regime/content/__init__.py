"""
Blog Content Generation Module

Automated tools for creating compelling blog content, market analysis reports,
and comparative studies. Streamlines the process of turning HMM regime analysis
into publication-ready articles and visualizations.
"""

from .generators import (
    generate_comparative_report,
    generate_historical_analysis,
    generate_market_report,
    generate_regime_update,
)
from .templates import (
    ANALYSIS_TEMPLATES,
    format_analysis_for_blog,
    get_template,
)

__all__ = [
    # Content generators
    "generate_market_report",
    "generate_historical_analysis", 
    "generate_comparative_report",
    "generate_regime_update",
    # Template utilities
    "get_template",
    "format_analysis_for_blog",
    "ANALYSIS_TEMPLATES",
]