"""
Report generation components for hidden-regime pipeline.

Provides report components that implement the ReportComponent interface
for generating comprehensive reports from pipeline analysis results.
"""

from .markdown import MarkdownReportGenerator

__all__ = [
    "MarkdownReportGenerator",
]
