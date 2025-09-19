"""
Working tests for MarkdownReportGenerator component.

Tests that work with the current implementation, focusing on coverage
and validation of markdown report generation functionality.
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import Mock, patch
from datetime import datetime
import tempfile
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from hidden_regime.reports.markdown import MarkdownReportGenerator
from hidden_regime.config.report import ReportConfig
from hidden_regime.utils.exceptions import ValidationError


class TestMarkdownReportGeneratorWorking:
    """Working tests for MarkdownReportGenerator that focus on coverage."""
    
    def test_initialization(self):
        """Test basic initialization."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        assert generator.config is config
        assert generator._last_report is None
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = ReportConfig(
            title="Custom Analysis Report",
            template_style="academic",
            include_trading_signals=True
        )
        generator = MarkdownReportGenerator(config)
        
        assert generator.config.title == "Custom Analysis Report"
        assert generator.config.template_style == "academic"
        assert generator.config.include_trading_signals == True
    
    def test_basic_report_generation(self):
        """Test basic report generation functionality."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        
        # Simple analysis data
        analysis = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 30),
            'confidence': np.random.uniform(0.6, 0.9, 30),
            'regime_name': np.random.choice(['Bear', 'Sideways', 'Bull'], 30)
        }, index=dates)
        
        # Generate report
        report = generator.update(analysis=analysis)
        
        # Basic validation
        assert isinstance(report, str)
        assert len(report) > 0
        assert "# Hidden Regime Analysis Report" in report
        assert "## Executive Summary" in report
        assert generator._last_report == report
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation with all components."""
        config = ReportConfig(
            title="Comprehensive Test Report",
            include_summary=True,
            include_regime_analysis=True,
            include_performance_metrics=True,
            include_risk_analysis=True,
            include_trading_signals=True,
            include_data_quality=True
        )
        generator = MarkdownReportGenerator(config)
        
        # Create comprehensive test data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # Raw data
        prices = 100 + np.cumsum(np.random.normal(0.1, 1.0, 50))
        data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
        
        # Observations
        observations = pd.DataFrame({
            'log_return': np.random.normal(0.001, 0.02, 50),
            'volatility': np.random.uniform(0.01, 0.04, 50)
        }, index=dates)
        
        # Model output
        model_output = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 50),
            'confidence': np.random.uniform(0.5, 0.95, 50),
            'state_0_prob': np.random.uniform(0, 1, 50),
            'state_1_prob': np.random.uniform(0, 1, 50),
            'state_2_prob': np.random.uniform(0, 1, 50)
        }, index=dates)
        
        # Analysis results
        analysis = pd.DataFrame({
            'predicted_state': model_output['predicted_state'],
            'confidence': model_output['confidence'],
            'regime_name': np.random.choice(['Bear', 'Sideways', 'Bull'], 50),
            'regime_type': np.random.choice(['Bear', 'Sideways', 'Bull'], 50),
            'expected_return': np.random.normal(0.0005, 0.002, 50),
            'expected_volatility': np.random.uniform(0.01, 0.03, 50),
            'days_in_regime': np.random.randint(1, 15, 50),
            'expected_duration': np.random.uniform(5, 20, 50),
            'position_signal': np.random.uniform(-1, 1, 50)
        }, index=dates)
        
        # Generate comprehensive report
        report = generator.update(
            data=data,
            observations=observations,
            model_output=model_output,
            analysis=analysis
        )
        
        # Validate comprehensive sections
        assert isinstance(report, str)
        assert "# Comprehensive Test Report" in report
        assert "## Executive Summary" in report
        assert "## Regime Analysis" in report
        assert "## Performance Metrics" in report
        assert "## Risk Analysis" in report
        assert "## Trading Signals" in report
        assert "## Data Quality Assessment" in report
        
        # Validate specific content
        assert "**Current Regime**:" in report
        assert "**Confidence**:" in report
        assert "**Days in Current Regime**:" in report
    
    def test_title_section_generation(self):
        """Test title section generation."""
        config = ReportConfig(title="Custom Title", template_style="academic")
        generator = MarkdownReportGenerator(config)
        
        title_section = generator._generate_title_section()
        
        assert "# Custom Title" in title_section
        assert "**Template Style**: academic" in title_section
        assert "**Generated**:" in title_section
        assert "---" in title_section
    
    def test_summary_section_generation(self):
        """Test summary section generation."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.normal(0, 1, 20))
        }, index=dates)
        
        analysis = pd.DataFrame({
            'predicted_state': [1, 1, 2, 2, 0] * 4,
            'confidence': np.random.uniform(0.7, 0.9, 20),
            'regime_name': ['Sideways', 'Sideways', 'Bull', 'Bull', 'Bear'] * 4,
            'days_in_regime': np.random.randint(1, 10, 20),
            'expected_return': np.random.normal(0.001, 0.002, 20),
            'expected_volatility': np.random.uniform(0.015, 0.025, 20)
        }, index=dates)
        
        summary = generator._generate_summary_section(data, analysis)
        
        assert "## Executive Summary" in summary
        assert "**Data Period**:" in summary
        assert "**Total Observations**: 20" in summary
        assert "**Current Regime**:" in summary
        assert "**Confidence**:" in summary
        assert "**Days in Current Regime**:" in summary
        assert "**Expected Daily Return**:" in summary
        assert "**Expected Volatility**:" in summary
    
    def test_regime_analysis_section(self):
        """Test regime analysis section generation."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        analysis = pd.DataFrame({
            'predicted_state': [0] * 10 + [1] * 15 + [2] * 5,
            'confidence': np.random.uniform(0.6, 0.9, 30),
            'regime_name': ['Bear'] * 10 + ['Sideways'] * 15 + ['Bull'] * 5,
            'days_in_regime': np.random.randint(1, 8, 30),
            'expected_duration': np.random.uniform(5, 15, 30)
        }, index=dates)
        
        regime_section = generator._generate_regime_analysis_section(analysis)
        
        assert "## Regime Analysis" in regime_section
        assert "### Regime Distribution" in regime_section
        assert "| Regime | Occurrences | Percentage |" in regime_section
        assert "Sideways" in regime_section  # Most frequent regime
        assert "### Regime Stability" in regime_section
        assert "### Current Regime Details" in regime_section
    
    def test_performance_section_generation(self):
        """Test performance section generation."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        dates = pd.date_range('2023-01-01', periods=25, freq='D')
        model_output = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 25),
            'confidence': np.random.uniform(0.5, 0.95, 25)
        }, index=dates)
        
        performance_section = generator._generate_performance_section(model_output, None)
        
        assert "## Performance Metrics" in performance_section
        assert "### Model Confidence" in performance_section
        assert "| Metric | Value |" in performance_section
        assert "| Mean |" in performance_section
        assert "| Std Dev |" in performance_section
        assert "### State Predictions" in performance_section
        assert "| State | Count | Percentage |" in performance_section
    
    def test_risk_analysis_section(self):
        """Test risk analysis section generation."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        analysis = pd.DataFrame({
            'regime_type': ['Bear', 'Bear', 'Sideways', 'Sideways', 'Bull'] * 4,
            'expected_volatility': np.random.uniform(0.01, 0.05, 20)
        }, index=dates)
        
        risk_section = generator._generate_risk_analysis_section(analysis)
        
        assert "## Risk Analysis" in risk_section
        assert "### Risk by Regime Type" in risk_section
        assert "| Regime | Avg Volatility | Std Dev | Observations |" in risk_section
        assert "### Current Risk Assessment" in risk_section
        assert "**Risk Level**:" in risk_section
    
    def test_trading_signals_section(self):
        """Test trading signals section generation."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        dates = pd.date_range('2023-01-01', periods=15, freq='D')
        analysis = pd.DataFrame({
            'position_signal': np.random.uniform(-1, 1, 15)
        }, index=dates)
        
        trading_section = generator._generate_trading_signals_section(analysis)
        
        assert "## Trading Signals" in trading_section
        assert "### Current Position Recommendation" in trading_section
        assert "**Position Signal**:" in trading_section
        assert "**Signal Strength**:" in trading_section
        assert "**Recommendation**:" in trading_section
        assert "### Signal Statistics" in trading_section
    
    def test_data_quality_section(self):
        """Test data quality section generation."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        
        # Create data with some missing values
        data = pd.DataFrame({
            'close': [100 + i for i in range(20)],
            'volume': [1000000] * 15 + [np.nan] * 5  # Some missing volume data
        }, index=dates)
        
        observations = pd.DataFrame({
            'log_return': np.random.normal(0, 0.02, 20),
            'volatility': np.random.uniform(0.01, 0.03, 20)
        }, index=dates)
        
        quality_section = generator._generate_data_quality_section(data, observations)
        
        assert "## Data Quality Assessment" in quality_section
        assert "### Data Overview" in quality_section
        assert "**Total Observations**: 20" in quality_section
        assert "### Missing Data" in quality_section
        assert "volume" in quality_section  # Missing volume should be reported
        assert "### Observation Quality" in quality_section
    
    def test_footer_section_generation(self):
        """Test footer section generation."""
        config = ReportConfig(template_style="minimal")
        generator = MarkdownReportGenerator(config)
        
        footer = generator._generate_footer_section()
        
        assert "---" in footer
        assert "*Report generated by Hidden Regime Analysis Pipeline*" in footer
        assert "*Template: minimal*" in footer
        assert "*Timestamp:" in footer
    
    def test_empty_data_handling(self):
        """Test handling of empty data inputs."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        # Test with completely empty inputs
        report = generator.update()
        assert isinstance(report, str)
        assert "# Hidden Regime Analysis Report" in report
        
        # Test with empty DataFrames - create with proper datetime index to avoid date() errors
        dates = pd.date_range('2023-01-01', periods=0, freq='D')  # Empty range
        empty_df = pd.DataFrame(index=dates)
        report_empty = generator.update(
            data=empty_df,
            analysis=empty_df
        )
        assert isinstance(report_empty, str)
        assert "# Hidden Regime Analysis Report" in report_empty
    
    def test_minimal_config_report(self):
        """Test report generation with minimal configuration."""
        config = ReportConfig.create_minimal()
        generator = MarkdownReportGenerator(config)
        
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        analysis = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 10),
            'confidence': np.random.uniform(0.7, 0.9, 10),
            'regime_name': np.random.choice(['Bear', 'Sideways', 'Bull'], 10)
        }, index=dates)
        
        report = generator.update(analysis=analysis)
        
        assert isinstance(report, str)
        assert "## Executive Summary" in report
        assert "## Regime Analysis" in report
        # Should not include performance metrics, risk analysis, trading signals
        assert "## Performance Metrics" not in report
        assert "## Risk Analysis" not in report
        assert "## Trading Signals" not in report
    
    def test_comprehensive_config_report(self):
        """Test report generation with comprehensive configuration."""
        config = ReportConfig.create_comprehensive()
        generator = MarkdownReportGenerator(config)
        
        dates = pd.date_range('2023-01-01', periods=15, freq='D')
        
        # Create comprehensive test data
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.normal(0, 1, 15))
        }, index=dates)
        
        model_output = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 15),
            'confidence': np.random.uniform(0.6, 0.9, 15)
        }, index=dates)
        
        analysis = pd.DataFrame({
            'predicted_state': model_output['predicted_state'],
            'confidence': model_output['confidence'],
            'regime_name': np.random.choice(['Bear', 'Sideways', 'Bull'], 15),
            'regime_type': np.random.choice(['Bear', 'Sideways', 'Bull'], 15),
            'expected_volatility': np.random.uniform(0.01, 0.03, 15),
            'position_signal': np.random.uniform(-0.5, 0.5, 15)
        }, index=dates)
        
        report = generator.update(
            data=data,
            model_output=model_output,
            analysis=analysis
        )
        
        # Should include all sections for comprehensive report
        assert "## Executive Summary" in report
        assert "## Regime Analysis" in report
        assert "## Performance Metrics" in report
        assert "## Risk Analysis" in report
        assert "## Trading Signals" in report
        assert "## Data Quality Assessment" in report
    
    def test_plot_functionality(self):
        """Test plot generation functionality."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        # Test with no report generated
        fig1 = generator.plot()
        assert fig1 is not None
        
        import matplotlib.pyplot as plt
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Test with report generated
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        analysis = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 10),
            'confidence': np.random.uniform(0.7, 0.9, 10)
        }, index=dates)
        
        generator.update(analysis=analysis)
        
        fig2 = generator.plot()
        assert fig2 is not None
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
    
    def test_file_saving_functionality(self):
        """Test report file saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ReportConfig(output_dir=temp_dir)
            generator = MarkdownReportGenerator(config)
            
            dates = pd.date_range('2023-01-01', periods=5, freq='D')
            analysis = pd.DataFrame({
                'predicted_state': [0, 1, 2, 1, 0],
                'confidence': [0.8, 0.9, 0.7, 0.85, 0.75]
            }, index=dates)
            
            # Generate report
            report = generator.update(analysis=analysis)
            
            # Check if file was created
            expected_file = config.get_report_filename()
            assert os.path.exists(expected_file)
            
            # Verify file content
            with open(expected_file, 'r', encoding='utf-8') as f:
                saved_content = f.read()
            
            assert saved_content == report
            assert len(saved_content) > 0


class TestMarkdownReportGeneratorCoverage:
    """Additional tests to improve code coverage."""
    
    def test_error_handling_in_file_operations(self):
        """Test error handling in file operations."""
        # Test with invalid output directory - use a config that bypasses validation
        config = ReportConfig()
        config.output_dir = "/invalid/path/that/does/not/exist"  # Set after creation to bypass validation
        generator = MarkdownReportGenerator(config)
        
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        analysis = pd.DataFrame({
            'predicted_state': [1, 2, 0, 1, 2],
            'confidence': [0.8, 0.7, 0.9, 0.6, 0.8]
        }, index=dates)
        
        # Should not fail even with invalid output directory - report generation should continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress potential warnings
            report = generator.update(analysis=analysis)
            assert isinstance(report, str)
            assert len(report) > 0
    
    def test_various_template_styles(self):
        """Test report generation with different template styles."""
        styles = ["professional", "academic", "minimal"]
        
        for style in styles:
            config = ReportConfig(template_style=style, title=f"Test {style.title()} Report")
            generator = MarkdownReportGenerator(config)
            
            dates = pd.date_range('2023-01-01', periods=8, freq='D')
            analysis = pd.DataFrame({
                'predicted_state': np.random.randint(0, 3, 8),
                'confidence': np.random.uniform(0.6, 0.9, 8),
                'regime_name': np.random.choice(['Bear', 'Sideways', 'Bull'], 8)
            }, index=dates)
            
            report = generator.update(analysis=analysis)
            
            assert isinstance(report, str)
            assert f"# Test {style.title()} Report" in report
            assert f"*Template: {style}*" in report
    
    def test_edge_case_data_patterns(self):
        """Test with edge case data patterns."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        # Test with single regime state
        analysis_single_regime = pd.DataFrame({
            'predicted_state': [1] * 10,  # All same state
            'confidence': np.random.uniform(0.8, 0.9, 10),
            'regime_name': ['Sideways'] * 10
        }, index=dates)
        
        report_single = generator.update(analysis=analysis_single_regime)
        assert isinstance(report_single, str)
        assert "Sideways" in report_single
        assert "100.0%" in report_single  # Should show 100% for single regime
        
        # Test with extreme confidence values - include model_output for performance metrics
        model_output_extreme = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 10),
            'confidence': [0.0, 0.1, 0.5, 0.9, 1.0] * 2  # Include extreme values
        }, index=dates)
        
        analysis_extreme = pd.DataFrame({
            'predicted_state': model_output_extreme['predicted_state'],
            'confidence': model_output_extreme['confidence'],
            'regime_name': np.random.choice(['Bear', 'Sideways', 'Bull'], 10)
        }, index=dates)
        
        report_extreme = generator.update(
            model_output=model_output_extreme, 
            analysis=analysis_extreme
        )
        assert isinstance(report_extreme, str)
        assert "## Performance Metrics" in report_extreme
    
    def test_missing_optional_columns(self):
        """Test handling of missing optional columns in analysis data."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        dates = pd.date_range('2023-01-01', periods=12, freq='D')
        
        # Minimal analysis data (missing optional columns)
        minimal_analysis = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 12),
            'confidence': np.random.uniform(0.6, 0.9, 12)
            # Missing: regime_name, expected_return, expected_volatility, etc.
        }, index=dates)
        
        report = generator.update(analysis=minimal_analysis)
        
        assert isinstance(report, str)
        assert "## Executive Summary" in report
        assert "## Regime Analysis" in report
        # Should handle missing columns gracefully
    
    def test_advanced_performance_section_mock(self):
        """Test advanced performance section with mocked analysis object."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        # Create mock analysis object with performance metrics method
        mock_analysis = Mock()
        mock_analysis.get_comprehensive_performance_metrics.return_value = {
            'summary': {
                'overall_quality': 'Good',
                'quality_score': 75,
                'average_confidence': 0.82,
                'balance_score': 0.65,
                'stability_rating': 'Stable'
            },
            'regime_distribution': {
                'regime_name_percentages': {'Bear': 30.0, 'Sideways': 45.0, 'Bull': 25.0},
                'regime_name_counts': {'Bear': 15, 'Sideways': 22, 'Bull': 13},
                'regime_dominance': {
                    'most_frequent_state': 'Sideways',
                    'dominance_percentage': 45.0,
                    'is_dominated': False,
                    'balance_score': 0.65
                }
            },
            'transition_analysis': {
                'total_transitions': 8,
                'transition_rate': 0.16,
                'average_persistence': 0.84,
                'stability_score': 0.78,
                'persistence_by_state': {'Bear': 0.80, 'Sideways': 0.85, 'Bull': 0.82}
            }
        }
        
        advanced_section = generator._generate_advanced_performance_section(mock_analysis)
        
        assert "## Advanced Performance Analysis" in advanced_section
        assert "### Performance Summary" in advanced_section
        assert "Overall Quality | Good" in advanced_section
        assert "### Detailed Regime Distribution" in advanced_section
        assert "### Regime Transition Analysis" in advanced_section
    
    def test_advanced_performance_section_error_handling(self):
        """Test error handling in advanced performance section."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        # Mock analysis that raises exception
        mock_analysis = Mock()
        mock_analysis.get_comprehensive_performance_metrics.side_effect = Exception("Test error")
        
        advanced_section = generator._generate_advanced_performance_section(mock_analysis)
        
        assert "## Advanced Performance Analysis" in advanced_section
        assert "Advanced performance analysis failed: Test error" in advanced_section
    
    def test_no_data_scenarios(self):
        """Test various no-data scenarios."""
        config = ReportConfig()
        generator = MarkdownReportGenerator(config)
        
        # Test regime analysis with empty data
        empty_analysis = pd.DataFrame()
        regime_section = generator._generate_regime_analysis_section(empty_analysis)
        assert "No regime analysis data available." in regime_section
        
        # Test performance section with empty data
        empty_model_output = pd.DataFrame()
        perf_section = generator._generate_performance_section(empty_model_output, None)
        assert "No performance data available." in perf_section
        
        # Test risk analysis with empty data
        risk_section = generator._generate_risk_analysis_section(empty_analysis)
        assert "No risk analysis data available." in risk_section
        
        # Test trading signals with no position signal
        no_signals_analysis = pd.DataFrame({'other_col': [1, 2, 3]})
        trading_section = generator._generate_trading_signals_section(no_signals_analysis)
        assert "No trading signals generated." in trading_section
        
        # Test data quality with None data
        quality_section = generator._generate_data_quality_section(None, None)
        assert "No data available for quality assessment." in quality_section
    
    def test_comprehensive_workflow(self):
        """Test comprehensive workflow with realistic data patterns."""
        config = ReportConfig.create_comprehensive()
        config.title = "Comprehensive Workflow Test"
        generator = MarkdownReportGenerator(config)
        
        # Create realistic time series data
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        
        # Simulate regime-based price movements
        price_changes = np.concatenate([
            np.random.normal(-0.02, 0.03, 20),  # Bear regime
            np.random.normal(0.001, 0.015, 25), # Sideways regime
            np.random.normal(0.015, 0.025, 15)  # Bull regime
        ])
        prices = 100 * np.exp(np.cumsum(price_changes))
        
        data = pd.DataFrame({
            'open': prices * 0.998,
            'high': prices * 1.025,
            'low': prices * 0.975,
            'close': prices,
            'volume': np.random.randint(500000, 5000000, 60)
        }, index=dates)
        
        observations = pd.DataFrame({
            'log_return': price_changes,
            'volatility': np.abs(price_changes) + np.random.uniform(0.005, 0.015, 60)
        }, index=dates)
        
        model_output = pd.DataFrame({
            'predicted_state': np.concatenate([[0]*20, [1]*25, [2]*15]),
            'confidence': np.random.uniform(0.65, 0.92, 60)
        }, index=dates)
        
        analysis = pd.DataFrame({
            'predicted_state': model_output['predicted_state'],
            'confidence': model_output['confidence'],
            'regime_name': np.concatenate([['Bear']*20, ['Sideways']*25, ['Bull']*15]),
            'regime_type': np.concatenate([['Bear']*20, ['Sideways']*25, ['Bull']*15]),
            'expected_return': np.concatenate([[-0.02]*20, [0.001]*25, [0.015]*15]),
            'expected_volatility': np.concatenate([[0.03]*20, [0.015]*25, [0.025]*15]),
            'days_in_regime': np.concatenate([list(range(1, 21)), list(range(1, 26)), list(range(1, 16))]),
            'expected_duration': np.concatenate([[20]*20, [25]*25, [15]*15]),
            'position_signal': np.concatenate([[-0.8]*20, [0.1]*25, [0.7]*15])
        }, index=dates)
        
        # Generate comprehensive report
        report = generator.update(
            data=data,
            observations=observations,
            model_output=model_output,
            analysis=analysis
        )
        
        # Validate comprehensive report content
        assert isinstance(report, str)
        assert len(report) > 2000  # Should be substantial report
        assert "# Comprehensive Workflow Test" in report
        
        # Validate all major sections
        sections = ["Executive Summary", "Regime Analysis", "Performance Metrics", 
                   "Risk Analysis", "Trading Signals", "Data Quality Assessment"]
        for section in sections:
            assert f"## {section}" in report
        
        # Validate specific regime content
        assert "Bear" in report
        assert "Sideways" in report
        assert "Bull" in report
        
        # Validate performance metrics
        assert "Model Confidence" in report
        assert "State Predictions" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])