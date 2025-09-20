"""
Report configuration classes for pipeline report components.

Provides configuration for generating reports from analysis results including
output formatting, visualization options, and report content specifications.
"""

from dataclasses import dataclass
from typing import Optional, List, Literal, Any
import os

from .base import BaseConfig
from ..utils.exceptions import ConfigurationError


@dataclass(frozen=True)
class ReportConfig(BaseConfig):
    """
    Configuration for report generation components.
    """
    
    # Output configuration
    output_dir: Optional[str] = None
    output_format: Literal["markdown", "html", "pdf", "json"] = "markdown"
    
    # Visualization options
    show_plots: bool = False
    save_plots: bool = True
    plot_format: Literal["png", "pdf", "svg"] = "png"
    plot_dpi: int = 300
    
    # Report content
    include_summary: bool = True
    include_regime_analysis: bool = True
    include_performance_metrics: bool = True
    include_risk_analysis: bool = True
    include_trading_signals: bool = False
    include_data_quality: bool = True
    
    # Advanced features (for future implementation)
    include_llm_analysis: bool = False
    llm_provider: Optional[str] = None
    
    # Report styling
    title: Optional[str] = None
    template_style: Literal["professional", "academic", "minimal"] = "professional"
    
    def validate(self) -> None:
        """Validate report configuration."""
        super().validate()
        
        # Validate output directory
        if self.output_dir is not None:
            # Check if directory exists or can be created
            if not os.path.exists(self.output_dir):
                try:
                    os.makedirs(self.output_dir, exist_ok=True)
                except OSError:
                    raise ConfigurationError(f"Cannot create output directory: {self.output_dir}")
        
        # Validate plot parameters
        if self.plot_dpi < 72:
            raise ConfigurationError(f"plot_dpi must be at least 72, got {self.plot_dpi}")
        
        if self.plot_dpi > 600:
            raise ConfigurationError(f"plot_dpi should not exceed 600 for practical reasons, got {self.plot_dpi}")
        
        # Validate LLM configuration
        if self.include_llm_analysis and not self.llm_provider:
            raise ConfigurationError("llm_provider must be specified when include_llm_analysis=True")
        
        if self.llm_provider is not None:
            valid_providers = ["openai", "anthropic", "local", "custom"]
            if self.llm_provider not in valid_providers:
                raise ConfigurationError(f"llm_provider must be one of {valid_providers}")
    
    def create_component(self) -> Any:
        """Create report component."""
        from ..reports.markdown import MarkdownReportGenerator
        return MarkdownReportGenerator(self)
    
    def get_output_directory(self) -> str:
        """Get output directory, using temp directory if none specified."""
        if self.output_dir is None:
            import tempfile
            return tempfile.gettempdir()
        return self.output_dir
    
    def get_plot_filename(self, plot_name: str) -> str:
        """Generate filename for plot based on configuration."""
        output_dir = self.get_output_directory()
        filename = f"{plot_name}.{self.plot_format}"
        return os.path.join(output_dir, filename)
    
    def get_report_filename(self, base_name: str = "regime_analysis") -> str:
        """Generate filename for report based on configuration."""
        output_dir = self.get_output_directory()
        
        if self.output_format == "markdown":
            ext = "md"
        elif self.output_format == "html":
            ext = "html"
        elif self.output_format == "pdf":
            ext = "pdf"
        elif self.output_format == "json":
            ext = "json"
        else:
            ext = "txt"
        
        filename = f"{base_name}.{ext}"
        return os.path.join(output_dir, filename)
    
    @classmethod
    def create_minimal(cls) -> 'ReportConfig':
        """Create minimal report configuration."""
        return cls(
            output_format="markdown",
            show_plots=False,
            save_plots=False,
            include_summary=True,
            include_regime_analysis=True,
            include_performance_metrics=False,
            include_risk_analysis=False,
            include_trading_signals=False,
            template_style="minimal"
        )
    
    @classmethod
    def create_comprehensive(cls) -> 'ReportConfig':
        """Create comprehensive report configuration."""
        return cls(
            output_format="markdown",
            show_plots=False,
            save_plots=True,
            plot_format="png",
            plot_dpi=300,
            include_summary=True,
            include_regime_analysis=True,
            include_performance_metrics=True,
            include_risk_analysis=True,
            include_trading_signals=True,
            include_data_quality=True,
            template_style="professional"
        )
    
    @classmethod
    def create_presentation(cls) -> 'ReportConfig':
        """Create configuration for presentation-ready reports."""
        return cls(
            output_format="html",
            show_plots=True,
            save_plots=True,
            plot_format="svg",
            plot_dpi=300,
            include_summary=True,
            include_regime_analysis=True,
            include_performance_metrics=True,
            include_risk_analysis=True,
            include_trading_signals=False,
            template_style="professional"
        )