"""
Working tests for ReportConfig component.

Tests that work with the current implementation, focusing on coverage
and validation of report configuration functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import patch

from hidden_regime.config.report import ReportConfig
from hidden_regime.reports.markdown import MarkdownReportGenerator
from hidden_regime.utils.exceptions import ConfigurationError


class TestReportConfigWorking:
    """Working tests for ReportConfig that focus on coverage."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        config = ReportConfig()
        
        # Test default values
        assert config.output_dir is None
        assert config.output_format == "markdown"
        assert config.show_plots == False
        assert config.save_plots == True
        assert config.plot_format == "png"
        assert config.plot_dpi == 300
        assert config.include_summary == True
        assert config.include_regime_analysis == True
        assert config.include_performance_metrics == True
        assert config.include_risk_analysis == True
        assert config.include_trading_signals == False
        assert config.include_data_quality == True
        assert config.include_llm_analysis == False
        assert config.llm_provider is None
        assert config.title is None
        assert config.template_style == "professional"
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        config = ReportConfig(
            output_dir="/tmp/test_reports",
            output_format="html",
            show_plots=True,
            save_plots=False,
            plot_format="svg",
            plot_dpi=150,
            include_summary=False,
            include_regime_analysis=True,
            include_performance_metrics=False,
            include_risk_analysis=False,
            include_trading_signals=True,
            include_data_quality=False,
            title="Custom Analysis Report",
            template_style="minimal"
        )
        
        assert config.output_dir == "/tmp/test_reports"
        assert config.output_format == "html"
        assert config.show_plots == True
        assert config.save_plots == False
        assert config.plot_format == "svg"
        assert config.plot_dpi == 150
        assert config.include_summary == False
        assert config.include_regime_analysis == True
        assert config.include_performance_metrics == False
        assert config.include_risk_analysis == False
        assert config.include_trading_signals == True
        assert config.include_data_quality == False
        assert config.title == "Custom Analysis Report"
        assert config.template_style == "minimal"
    
    def test_validation_success(self):
        """Test successful validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ReportConfig(
                output_dir=temp_dir,
                plot_dpi=200,
                include_llm_analysis=False
            )
            
            # Should not raise any exceptions
            config.validate()
    
    def test_validation_output_directory_creation(self):
        """Test output directory creation during validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new_report_dir")
            
            # Create config without triggering validation yet
            config = ReportConfig()
            config.output_dir = new_dir  # Set directory without validation
            
            # Directory should not exist yet
            assert not os.path.exists(new_dir)
            
            # Validation should create the directory
            config.validate()
            assert os.path.exists(new_dir)
    
    def test_validation_invalid_output_directory(self):
        """Test validation with invalid output directory."""
        # Create config first, then set invalid directory to bypass constructor validation
        config = ReportConfig()
        config.output_dir = "/root/definitely/does/not/exist/and/cannot/create"
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        
        assert "Cannot create output directory" in str(exc_info.value)
    
    def test_validation_plot_dpi_limits(self):
        """Test validation of plot DPI limits."""
        # Test DPI too low - create config then modify to bypass constructor
        config_low = ReportConfig()
        config_low.plot_dpi = 50
        with pytest.raises(ConfigurationError) as exc_info:
            config_low.validate()
        assert "plot_dpi must be at least 72" in str(exc_info.value)
        
        # Test DPI too high - create config then modify to bypass constructor  
        config_high = ReportConfig()
        config_high.plot_dpi = 800
        with pytest.raises(ConfigurationError) as exc_info:
            config_high.validate()
        assert "plot_dpi should not exceed 600" in str(exc_info.value)
        
        # Test valid DPI values
        valid_dpis = [72, 150, 300, 600]
        for dpi in valid_dpis:
            config_valid = ReportConfig(plot_dpi=dpi)
            config_valid.validate()  # Should not raise
    
    def test_validation_llm_configuration(self):
        """Test validation of LLM configuration."""
        # Test LLM analysis enabled without provider - modify after creation
        config_no_provider = ReportConfig()
        config_no_provider.include_llm_analysis = True
        config_no_provider.llm_provider = None
        with pytest.raises(ConfigurationError) as exc_info:
            config_no_provider.validate()
        assert "llm_provider must be specified when include_llm_analysis=True" in str(exc_info.value)
        
        # Test invalid LLM provider - modify after creation
        config_invalid_provider = ReportConfig()
        config_invalid_provider.include_llm_analysis = True
        config_invalid_provider.llm_provider = "invalid_provider"
        with pytest.raises(ConfigurationError) as exc_info:
            config_invalid_provider.validate()
        assert "llm_provider must be one of" in str(exc_info.value)
        
        # Test valid LLM providers
        valid_providers = ["openai", "anthropic", "local", "custom"]
        for provider in valid_providers:
            config_valid = ReportConfig(
                include_llm_analysis=True,
                llm_provider=provider
            )
            config_valid.validate()  # Should not raise
    
    def test_create_component(self):
        """Test component creation."""
        config = ReportConfig()
        component = config.create_component()
        
        assert isinstance(component, MarkdownReportGenerator)
        assert component.config is config
    
    def test_get_output_directory(self):
        """Test output directory retrieval."""
        # Test with specified output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_with_dir = ReportConfig(output_dir=temp_dir)
            assert config_with_dir.get_output_directory() == temp_dir
        
        # Test with no output directory (should use temp directory)
        config_no_dir = ReportConfig(output_dir=None)
        output_dir = config_no_dir.get_output_directory()
        
        assert output_dir == tempfile.gettempdir()
    
    def test_get_plot_filename(self):
        """Test plot filename generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ReportConfig(
                output_dir=temp_dir,
                plot_format="svg"
            )
            
            filename = config.get_plot_filename("test_plot")
            expected_filename = os.path.join(temp_dir, "test_plot.svg")
            
            assert filename == expected_filename
    
    def test_get_report_filename(self):
        """Test report filename generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test different output formats
            format_extensions = {
                "markdown": "md",
                "html": "html", 
                "pdf": "pdf",
                "json": "json",
                "unknown": "txt"  # Should default to txt
            }
            
            for format_type, expected_ext in format_extensions.items():
                config = ReportConfig(
                    output_dir=temp_dir,
                    output_format=format_type
                )
                
                filename = config.get_report_filename("test_report")
                expected_filename = os.path.join(temp_dir, f"test_report.{expected_ext}")
                
                assert filename == expected_filename
    
    def test_create_minimal_config(self):
        """Test creation of minimal configuration."""
        config = ReportConfig.create_minimal()
        
        assert config.output_format == "markdown"
        assert config.show_plots == False
        assert config.save_plots == False
        assert config.include_summary == True
        assert config.include_regime_analysis == True
        assert config.include_performance_metrics == False
        assert config.include_risk_analysis == False
        assert config.include_trading_signals == False
        assert config.template_style == "minimal"
        
        # Should validate successfully
        config.validate()
    
    def test_create_comprehensive_config(self):
        """Test creation of comprehensive configuration."""
        config = ReportConfig.create_comprehensive()
        
        assert config.output_format == "markdown"
        assert config.show_plots == False
        assert config.save_plots == True
        assert config.plot_format == "png"
        assert config.plot_dpi == 300
        assert config.include_summary == True
        assert config.include_regime_analysis == True
        assert config.include_performance_metrics == True
        assert config.include_risk_analysis == True
        assert config.include_trading_signals == True
        assert config.include_data_quality == True
        assert config.template_style == "professional"
        
        # Should validate successfully
        config.validate()
    
    def test_create_presentation_config(self):
        """Test creation of presentation configuration."""
        config = ReportConfig.create_presentation()
        
        assert config.output_format == "html"
        assert config.show_plots == True
        assert config.save_plots == True
        assert config.plot_format == "svg"
        assert config.plot_dpi == 300
        assert config.include_summary == True
        assert config.include_regime_analysis == True
        assert config.include_performance_metrics == True
        assert config.include_risk_analysis == True
        assert config.include_trading_signals == False
        assert config.template_style == "professional"
        
        # Should validate successfully
        config.validate()
    
    def test_all_output_formats(self):
        """Test all supported output formats."""
        formats = ["markdown", "html", "pdf", "json"]
        
        for fmt in formats:
            config = ReportConfig(output_format=fmt)
            config.validate()  # Should not raise
            
            # Test filename generation
            with tempfile.TemporaryDirectory() as temp_dir:
                config.output_dir = temp_dir
                filename = config.get_report_filename()
                assert filename.endswith(f".{fmt}" if fmt != "markdown" else ".md")
    
    def test_all_plot_formats(self):
        """Test all supported plot formats."""
        formats = ["png", "pdf", "svg"]
        
        for fmt in formats:
            config = ReportConfig(plot_format=fmt)
            config.validate()  # Should not raise
            
            # Test filename generation
            with tempfile.TemporaryDirectory() as temp_dir:
                config.output_dir = temp_dir
                filename = config.get_plot_filename("test")
                assert filename.endswith(f".{fmt}")
    
    def test_all_template_styles(self):
        """Test all supported template styles."""
        styles = ["professional", "academic", "minimal"]
        
        for style in styles:
            config = ReportConfig(template_style=style)
            config.validate()  # Should not raise
            
            assert config.template_style == style


class TestReportConfigCoverage:
    """Additional tests to improve code coverage."""
    
    def test_edge_case_configurations(self):
        """Test edge case configurations."""
        # Test minimum valid DPI
        config_min_dpi = ReportConfig(plot_dpi=72)
        config_min_dpi.validate()
        
        # Test maximum valid DPI
        config_max_dpi = ReportConfig(plot_dpi=600)
        config_max_dpi.validate()
        
        # Test all boolean combinations for report sections
        boolean_params = [
            'include_summary',
            'include_regime_analysis', 
            'include_performance_metrics',
            'include_risk_analysis',
            'include_trading_signals',
            'include_data_quality',
            'show_plots',
            'save_plots',
            'include_llm_analysis'
        ]
        
        # Test with all features disabled
        disabled_config = ReportConfig(**{param: False for param in boolean_params})
        disabled_config.validate()  # Should still be valid
        
        # Test with all features enabled (except LLM without provider)
        enabled_params = {param: True for param in boolean_params if param != 'include_llm_analysis'}
        enabled_config = ReportConfig(**enabled_params)
        enabled_config.validate()  # Should be valid
    
    def test_filename_generation_edge_cases(self):
        """Test filename generation with edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ReportConfig(output_dir=temp_dir)
            
            # Test with special characters in base name
            special_names = ["report_with_underscores", "report-with-dashes", "report123"]
            
            for name in special_names:
                filename = config.get_report_filename(name)
                assert name in filename
                assert filename.endswith(".md")
            
            # Test plot filename with special characters
            plot_filename = config.get_plot_filename("plot_with_underscores")
            assert "plot_with_underscores" in plot_filename
            assert plot_filename.endswith(".png")
    
    def test_config_inheritance(self):
        """Test that ReportConfig properly inherits from BaseConfig."""
        config = ReportConfig()
        
        # Should have BaseConfig functionality
        assert hasattr(config, 'validate')
        assert hasattr(config, 'create_component')
        
        # Should be able to call parent validation
        config.validate()
    
    def test_configuration_combinations(self):
        """Test various configuration combinations."""
        # Test academic style with comprehensive features
        academic_config = ReportConfig(
            template_style="academic",
            include_summary=True,
            include_regime_analysis=True,
            include_performance_metrics=True,
            include_risk_analysis=True,
            include_trading_signals=True,
            include_data_quality=True
        )
        academic_config.validate()
        assert academic_config.template_style == "academic"
        
        # Test minimal style with limited features
        minimal_config = ReportConfig(
            template_style="minimal",
            include_summary=True,
            include_regime_analysis=False,
            include_performance_metrics=False,
            include_risk_analysis=False,
            include_trading_signals=False,
            include_data_quality=False
        )
        minimal_config.validate()
        assert minimal_config.template_style == "minimal"
        
        # Test presentation style with visual features
        presentation_config = ReportConfig(
            template_style="professional",
            output_format="html",
            show_plots=True,
            save_plots=True,
            plot_format="svg",
            include_summary=True,
            include_regime_analysis=True,
            include_performance_metrics=True
        )
        presentation_config.validate()
        assert presentation_config.output_format == "html"
        assert presentation_config.plot_format == "svg"
    
    def test_directory_creation_permissions(self):
        """Test directory creation under different scenarios."""
        # Test creating directory in temp location (should work)
        import tempfile
        temp_base = tempfile.gettempdir()
        test_dir = os.path.join(temp_base, "test_hidden_regime_reports")
        
        # Clean up if exists from previous test
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
        
        config = ReportConfig(output_dir=test_dir)
        config.validate()  # Should create directory
        
        assert os.path.exists(test_dir)
        
        # Clean up
        os.rmdir(test_dir)
    
    def test_component_creation_integration(self):
        """Test integration between config and component creation."""
        # Test that created component receives the configuration
        config = ReportConfig(
            title="Integration Test Report",
            template_style="academic",
            include_trading_signals=True
        )
        
        component = config.create_component()
        
        assert isinstance(component, MarkdownReportGenerator)
        assert component.config is config
        assert component.config.title == "Integration Test Report"
        assert component.config.template_style == "academic"
        assert component.config.include_trading_signals == True
    
    def test_comprehensive_workflow(self):
        """Test comprehensive configuration workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a comprehensive configuration
            config = ReportConfig(
                output_dir=temp_dir,
                output_format="markdown",
                title="Comprehensive Workflow Test",
                template_style="professional",
                plot_format="png",
                plot_dpi=300,
                include_summary=True,
                include_regime_analysis=True,
                include_performance_metrics=True,
                include_risk_analysis=True,
                include_trading_signals=True,
                include_data_quality=True,
                save_plots=True,
                show_plots=False
            )
            
            # Validate configuration
            config.validate()
            
            # Test filename generation
            report_filename = config.get_report_filename("comprehensive_test")
            plot_filename = config.get_plot_filename("test_plot")
            
            # Verify paths are correctly formed
            assert report_filename.startswith(temp_dir)
            assert report_filename.endswith(".md")
            assert plot_filename.startswith(temp_dir)
            assert plot_filename.endswith(".png")
            
            # Test component creation
            component = config.create_component()
            assert isinstance(component, MarkdownReportGenerator)
            
            # Verify component has correct configuration
            assert component.config.title == "Comprehensive Workflow Test"
            assert component.config.include_trading_signals == True
            assert component.config.template_style == "professional"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])