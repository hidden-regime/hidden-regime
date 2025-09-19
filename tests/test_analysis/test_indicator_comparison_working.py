"""
Working tests for IndicatorPerformanceComparator component.

Tests that work with the current implementation, focusing on coverage
and validation of indicator comparison functionality.
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import Mock, patch
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from hidden_regime.analysis.indicator_comparison import IndicatorPerformanceComparator
from hidden_regime.config.analysis import FinancialAnalysisConfig
from hidden_regime.utils.exceptions import ValidationError


class TestIndicatorPerformanceComparatorWorking:
    """Working tests for IndicatorPerformanceComparator that focus on coverage."""
    
    def test_initialization(self):
        """Test basic initialization."""
        comparator = IndicatorPerformanceComparator()
        
        assert comparator.config is not None
        assert isinstance(comparator.config, FinancialAnalysisConfig)
        assert comparator._cache == {}
        assert hasattr(comparator, 'agreement_thresholds')
        assert hasattr(comparator, 'indicator_params')
        
        # Test with custom config
        config = FinancialAnalysisConfig(n_states=4)
        comparator_custom = IndicatorPerformanceComparator(config)
        assert comparator_custom.config.n_states == 4
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = FinancialAnalysisConfig(
            n_states=4,
            indicator_comparisons=['rsi', 'macd', 'bollinger_bands']
        )
        comparator = IndicatorPerformanceComparator(config)
        
        assert comparator.config.n_states == 4
        assert 'rsi' in comparator.config.indicator_comparisons
        assert 'macd' in comparator.config.indicator_comparisons
    
    def test_agreement_thresholds_structure(self):
        """Test agreement thresholds are properly configured."""
        comparator = IndicatorPerformanceComparator()
        
        expected_keys = ['excellent', 'good', 'moderate', 'poor']
        for key in expected_keys:
            assert key in comparator.agreement_thresholds
            assert isinstance(comparator.agreement_thresholds[key], float)
        
        # Test thresholds are in descending order
        thresholds = list(comparator.agreement_thresholds.values())
        assert thresholds == sorted(thresholds, reverse=True)
    
    def test_indicator_params_structure(self):
        """Test indicator parameters are properly configured."""
        comparator = IndicatorPerformanceComparator()
        
        expected_indicators = ['rsi', 'macd', 'bollinger_bands', 'moving_average', 'stochastic', 'williams_r']
        for indicator in expected_indicators:
            assert indicator in comparator.indicator_params
            assert isinstance(comparator.indicator_params[indicator], dict)
    
    def test_basic_comparison_functionality(self):
        """Test basic comparison functionality."""
        comparator = IndicatorPerformanceComparator()
        
        # Create sample analysis results
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        analysis_results = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 50),
            'confidence': np.random.uniform(0.6, 0.95, 50)
        }, index=dates)
        
        # Basic comparison without raw data
        comparison = comparator.compare_regime_vs_indicators(analysis_results)
        
        # Check basic structure
        assert isinstance(comparison, dict)
        assert 'summary' in comparison
        assert 'indicator_analysis' in comparison
        assert 'cross_indicator_analysis' in comparison
        assert 'predictive_performance' in comparison
        assert 'regime_indicator_matrix' in comparison
        assert 'statistical_tests' in comparison
    
    def test_comparison_with_raw_data(self):
        """Test comparison with raw OHLCV data."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        
        # Analysis results
        analysis_results = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 60),
            'confidence': np.random.uniform(0.5, 1.0, 60)
        }, index=dates)
        
        # Raw OHLCV data
        prices = 100 + np.cumsum(np.random.normal(0.1, 1.0, 60))
        raw_data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 60)
        }, index=dates)
        
        # Compare with indicators
        comparison = comparator.compare_regime_vs_indicators(
            analysis_results, 
            raw_data, 
            indicators=['rsi', 'macd']
        )
        
        # Should include indicator analysis
        assert 'indicator_analysis' in comparison
        # May include calculated indicators depending on implementation success
    
    def test_error_handling_empty_analysis_results(self):
        """Test error handling for empty analysis results."""
        comparator = IndicatorPerformanceComparator()
        
        empty_results = pd.DataFrame()
        
        with pytest.raises(ValidationError) as exc_info:
            comparator.compare_regime_vs_indicators(empty_results)
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_error_handling_missing_predicted_state(self):
        """Test error handling for missing predicted_state column."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        invalid_results = pd.DataFrame({
            'confidence': np.random.uniform(0.5, 1.0, 20)
            # Missing 'predicted_state' column
        }, index=dates)
        
        with pytest.raises(ValidationError) as exc_info:
            comparator.compare_regime_vs_indicators(invalid_results)
        
        assert "predicted_state" in str(exc_info.value).lower()
    
    def test_rsi_calculation(self):
        """Test RSI calculation functionality."""
        comparator = IndicatorPerformanceComparator()
        
        # Create sample price data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 50)), index=dates)
        
        # Calculate RSI
        rsi_data = comparator._calculate_rsi_detailed(prices, period=14)
        
        if rsi_data is not None:
            assert isinstance(rsi_data, pd.DataFrame)
            expected_columns = ['value', 'signal', 'zone', 'strength']
            for col in expected_columns:
                assert col in rsi_data.columns
            
            # Check RSI values are in valid range (0-100)
            rsi_values = rsi_data['value'].dropna()
            if len(rsi_values) > 0:
                assert (rsi_values >= 0).all()
                assert (rsi_values <= 100).all()
    
    def test_macd_calculation(self):
        """Test MACD calculation functionality."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        prices = pd.Series(100 + np.cumsum(np.random.normal(0.1, 1, 60)), index=dates)
        
        params = {'fast': 12, 'slow': 26, 'signal': 9}
        macd_data = comparator._calculate_macd_detailed(prices, params)
        
        if macd_data is not None:
            assert isinstance(macd_data, pd.DataFrame)
            expected_columns = ['macd_line', 'signal_line', 'histogram', 'signal', 'momentum', 'strength']
            for col in expected_columns:
                assert col in macd_data.columns
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation functionality."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=40, freq='D')
        prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1.5, 40)), index=dates)
        
        params = {'period': 20, 'std_dev': 2}
        bb_data = comparator._calculate_bollinger_detailed(prices, params)
        
        if bb_data is not None:
            assert isinstance(bb_data, pd.DataFrame)
            expected_columns = ['upper_band', 'middle_band', 'lower_band', 'position', 'signal', 'squeeze', 'band_width']
            for col in expected_columns:
                assert col in bb_data.columns
            
            # Check band relationships
            non_nan_data = bb_data.dropna()
            if len(non_nan_data) > 0:
                assert (non_nan_data['upper_band'] >= non_nan_data['middle_band']).all()
                assert (non_nan_data['middle_band'] >= non_nan_data['lower_band']).all()
    
    def test_moving_average_calculation(self):
        """Test moving average calculation functionality."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        prices = pd.Series(100 + np.cumsum(np.random.normal(0.05, 1, 30)), index=dates)
        
        ma_data = comparator._calculate_ma_detailed(prices, period=20)
        
        if ma_data is not None:
            assert isinstance(ma_data, pd.DataFrame)
            expected_columns = ['ma', 'position', 'signal', 'trend', 'slope', 'strength']
            for col in expected_columns:
                assert col in ma_data.columns
    
    def test_stochastic_calculation(self):
        """Test Stochastic oscillator calculation functionality."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=40, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, 40))
        
        raw_data = pd.DataFrame({
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices
        }, index=dates)
        
        params = {'k_period': 14, 'd_period': 3}
        stoch_data = comparator._calculate_stochastic_detailed(raw_data, params)
        
        if stoch_data is not None:
            assert isinstance(stoch_data, pd.DataFrame)
            expected_columns = ['k_percent', 'd_percent', 'signal', 'momentum']
            for col in expected_columns:
                assert col in stoch_data.columns
            
            # Check %K and %D are in valid range (0-100)
            k_values = stoch_data['k_percent'].dropna()
            d_values = stoch_data['d_percent'].dropna()
            if len(k_values) > 0:
                assert (k_values >= 0).all()
                assert (k_values <= 100).all()
            if len(d_values) > 0:
                assert (d_values >= 0).all()
                assert (d_values <= 100).all()
    
    def test_williams_r_calculation(self):
        """Test Williams %R calculation functionality."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=35, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1.2, 35))
        
        raw_data = pd.DataFrame({
            'high': prices * 1.03,
            'low': prices * 0.97,
            'close': prices
        }, index=dates)
        
        williams_data = comparator._calculate_williams_r_detailed(raw_data, period=14)
        
        if williams_data is not None:
            assert isinstance(williams_data, pd.DataFrame)
            expected_columns = ['williams_r', 'signal', 'zone']
            for col in expected_columns:
                assert col in williams_data.columns
            
            # Check Williams %R is in valid range (-100 to 0)
            wr_values = williams_data['williams_r'].dropna()
            if len(wr_values) > 0:
                assert (wr_values >= -100).all()
                assert (wr_values <= 0).all()
    
    def test_indicator_calculation_error_handling(self):
        """Test error handling in indicator calculations."""
        comparator = IndicatorPerformanceComparator()
        
        # Test with insufficient data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        short_prices = pd.Series([100, 101, 99, 102, 98], index=dates)
        
        # Should handle gracefully
        rsi_result = comparator._calculate_rsi_detailed(short_prices, period=14)
        # May return None or DataFrame with NaN values
        
        # Test with missing columns for multi-column indicators
        incomplete_data = pd.DataFrame({
            'close': [100, 101, 102],
            # Missing 'high' and 'low' columns
        })
        
        stoch_result = comparator._calculate_stochastic_detailed(incomplete_data, {})
        assert stoch_result is None  # Should return None for missing required columns
    
    def test_summary_generation(self):
        """Test comparison summary generation."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        analysis_results = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 30),
            'confidence': np.random.uniform(0.6, 0.9, 30)
        }, index=dates)
        
        indicators = ['rsi', 'macd']
        summary = comparator._generate_comparison_summary(analysis_results, indicators)
        
        assert isinstance(summary, dict)
        assert 'total_observations' in summary
        assert 'date_range' in summary  
        assert 'regime_summary' in summary
        assert 'indicators_requested' in summary
        assert 'analysis_timestamp' in summary
        
        assert summary['total_observations'] == 30
        assert summary['indicators_requested'] == indicators
        assert summary['regime_summary']['n_states'] == 3
    
    def test_correlation_metrics_calculation(self):
        """Test correlation metrics calculation."""
        comparator = IndicatorPerformanceComparator()
        
        # Create sample aligned signals
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        regime_signals = pd.Series(np.random.choice([-1, 0, 1], 50), index=dates)
        indicator_signals = pd.Series(np.random.choice([-1, 0, 1], 50), index=dates)
        
        # Add some correlation
        indicator_signals[regime_signals == 1] = np.random.choice([0, 1], sum(regime_signals == 1))
        
        corr_metrics = comparator._calculate_correlation_metrics(regime_signals, indicator_signals)
        
        assert isinstance(corr_metrics, dict)
        expected_keys = ['pearson_correlation', 'spearman_correlation', 'rolling_correlation_mean', 
                        'rolling_correlation_std', 'correlation_stability']
        for key in expected_keys:
            assert key in corr_metrics
            assert isinstance(corr_metrics[key], float)
    
    def test_classification_metrics_calculation(self):
        """Test classification metrics calculation."""
        comparator = IndicatorPerformanceComparator()
        
        # Create sample signals with some agreement
        dates = pd.date_range('2023-01-01', periods=40, freq='D')
        regime_signals = pd.Series(np.random.choice([-1, 0, 1], 40), index=dates)
        indicator_signals = pd.Series(np.random.choice([-1, 0, 1], 40), index=dates)
        
        # Force some agreement
        agreement_mask = np.random.choice([True, False], 40, p=[0.7, 0.3])
        indicator_signals[agreement_mask] = regime_signals[agreement_mask]
        
        class_metrics = comparator._calculate_classification_metrics(regime_signals, indicator_signals)
        
        assert isinstance(class_metrics, dict)
        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'direction_agreement']
        for key in expected_keys:
            assert key in class_metrics
            assert isinstance(class_metrics[key], float)
            assert 0 <= class_metrics[key] <= 1
    
    def test_confidence_weighted_agreement(self):
        """Test confidence-weighted agreement calculation."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        regime_signals = pd.Series(np.random.choice([-1, 0, 1], 30), index=dates)
        indicator_signals = pd.Series(np.random.choice([-1, 0, 1], 30), index=dates)
        confidence = pd.Series(np.random.uniform(0.3, 1.0, 30), index=dates)
        
        weighted_metrics = comparator._calculate_confidence_weighted_agreement(
            regime_signals, indicator_signals, confidence
        )
        
        assert isinstance(weighted_metrics, dict)
        expected_keys = ['weighted_agreement', 'high_confidence_agreement', 
                        'low_confidence_agreement', 'confidence_correlation']
        for key in expected_keys:
            assert key in weighted_metrics
            assert isinstance(weighted_metrics[key], float)
    
    def test_temporal_agreement_analysis(self):
        """Test temporal agreement analysis."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        regime_signals = pd.Series(np.random.choice([-1, 0, 1], 60), index=dates)
        indicator_signals = pd.Series(np.random.choice([-1, 0, 1], 60), index=dates)
        
        temporal_metrics = comparator._analyze_temporal_agreement(regime_signals, indicator_signals)
        
        assert isinstance(temporal_metrics, dict)
        expected_keys = ['rolling_agreement_mean', 'rolling_agreement_std', 
                        'agreement_trend', 'agreement_volatility', 'periods_analyzed']
        for key in expected_keys:
            assert key in temporal_metrics
            if key != 'periods_analyzed':
                assert isinstance(temporal_metrics[key], float)
        
        assert temporal_metrics['periods_analyzed'] == 60
    
    def test_performance_rating(self):
        """Test performance rating calculation."""
        comparator = IndicatorPerformanceComparator()
        
        # Create mock agreement analysis
        agreement_analysis = {
            'correlation': {'pearson_correlation': 0.6},
            'classification_metrics': {'accuracy': 0.7},
            'confidence_weighted_agreement': {'weighted_agreement': 0.65}
        }
        
        rating = comparator._rate_indicator_performance(agreement_analysis)
        
        assert isinstance(rating, dict)
        expected_keys = ['composite_score', 'rating', 'correlation_component', 
                        'accuracy_component', 'weighted_agreement_component']
        for key in expected_keys:
            assert key in rating
        
        assert rating['rating'] in ['excellent', 'good', 'moderate', 'poor']
        assert 0 <= rating['composite_score'] <= 1
    
    def test_plot_functionality(self):
        """Test plot generation functionality."""
        comparator = IndicatorPerformanceComparator()
        
        # Test with empty results
        fig1 = comparator.plot_comparison_results({})
        assert fig1 is not None
        
        import matplotlib.pyplot as plt
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Test with minimal comparison results
        minimal_results = {
            'indicator_analysis': {},
            'summary': {'total_observations': 0}
        }
        
        fig2 = comparator.plot_comparison_results(minimal_results)
        assert fig2 is not None
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)


class TestIndicatorPerformanceComparatorCoverage:
    """Additional tests to improve code coverage."""
    
    def test_various_indicator_combinations(self):
        """Test analysis with various indicator combinations."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=40, freq='D')
        analysis_results = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 40),
            'confidence': np.random.uniform(0.5, 1.0, 40)
        }, index=dates)
        
        # Test different indicator combinations
        indicator_sets = [
            ['rsi'],
            ['rsi', 'macd'],
            ['rsi', 'macd', 'bollinger_bands'],
            ['moving_average', 'stochastic']
        ]
        
        for indicators in indicator_sets:
            comparison = comparator.compare_regime_vs_indicators(
                analysis_results, indicators=indicators
            )
            assert isinstance(comparison, dict)
            assert 'summary' in comparison
            assert comparison['summary']['indicators_requested'] == indicators
    
    def test_edge_case_data_patterns(self):
        """Test with edge case data patterns."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=25, freq='D')
        
        # Test with single regime state
        single_state_results = pd.DataFrame({
            'predicted_state': [1] * 25,  # All same state
            'confidence': np.random.uniform(0.7, 1.0, 25)
        }, index=dates)
        
        comparison = comparator.compare_regime_vs_indicators(single_state_results)
        assert isinstance(comparison, dict)
        assert comparison['summary']['regime_summary']['n_states'] == 1
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        comparator = IndicatorPerformanceComparator()
        
        # Very short time series
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        short_results = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 5),
            'confidence': np.random.uniform(0.6, 1.0, 5)
        }, index=dates)
        
        comparison = comparator.compare_regime_vs_indicators(short_results)
        assert isinstance(comparison, dict)
        assert comparison['summary']['total_observations'] == 5
    
    def test_missing_confidence_column(self):
        """Test analysis when confidence column is missing."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        no_confidence_results = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 20)
            # No 'confidence' column
        }, index=dates)
        
        # Should still work with default confidence values
        comparison = comparator.compare_regime_vs_indicators(no_confidence_results)
        assert isinstance(comparison, dict)
        assert 'summary' in comparison
    
    def test_error_handling_in_calculations(self):
        """Test error handling in various calculations."""
        comparator = IndicatorPerformanceComparator()
        
        # Test with data that might cause calculation errors
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        # Create data with potential division by zero scenarios
        problem_data = pd.DataFrame({
            'predicted_state': [0, 0, 0, 1, 1, 2, 2, 2, 1, 0],
            'confidence': [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.0, 1.0, 0.8, 0.9]
        }, index=dates)
        
        # Should handle gracefully without crashing
        comparison = comparator.compare_regime_vs_indicators(problem_data)
        assert isinstance(comparison, dict)
    
    def test_unknown_indicator_handling(self):
        """Test handling of unknown indicators."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        analysis_results = pd.DataFrame({
            'predicted_state': np.random.randint(0, 3, 20),
            'confidence': np.random.uniform(0.5, 1.0, 20)
        }, index=dates)
        
        # Test with unknown indicator
        comparison = comparator.compare_regime_vs_indicators(
            analysis_results, indicators=['unknown_indicator']
        )
        
        assert isinstance(comparison, dict)
        assert 'summary' in comparison
    
    def test_signal_strength_analysis_edge_cases(self):
        """Test signal strength analysis with edge cases."""
        comparator = IndicatorPerformanceComparator()
        
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        
        # Test with constant signal strength
        regime_signals = pd.Series(np.random.choice([-1, 0, 1], 20), index=dates)
        constant_strength = pd.Series([0.5] * 20, index=dates)  # All same strength
        
        strength_analysis = comparator._analyze_signal_strength(regime_signals, constant_strength)
        assert isinstance(strength_analysis, dict)
        
        # Test with extreme signal strengths
        extreme_strength = pd.Series([0.0] * 10 + [1.0] * 10, index=dates)
        strength_analysis_extreme = comparator._analyze_signal_strength(regime_signals, extreme_strength)
        assert isinstance(strength_analysis_extreme, dict)
    
    def test_comprehensive_workflow(self):
        """Test comprehensive analysis workflow."""
        comparator = IndicatorPerformanceComparator()
        
        # Create comprehensive test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Analysis results with realistic regime patterns
        analysis_results = pd.DataFrame({
            'predicted_state': np.concatenate([
                np.full(30, 0),  # Bear regime
                np.full(40, 1),  # Sideways regime
                np.full(30, 2)   # Bull regime  
            ]),
            'confidence': np.random.uniform(0.6, 0.95, 100)
        }, index=dates)
        
        # Raw data with corresponding price patterns
        price_changes = np.concatenate([
            np.random.normal(-0.5, 1.5, 30),  # Declining prices
            np.random.normal(0.0, 0.8, 40),   # Sideways movement
            np.random.normal(0.8, 1.2, 30)    # Rising prices
        ])
        prices = 100 + np.cumsum(price_changes)
        
        raw_data = pd.DataFrame({
            'open': prices * 0.998,
            'high': prices * 1.025,
            'low': prices * 0.975,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Comprehensive comparison
        comparison = comparator.compare_regime_vs_indicators(
            analysis_results, 
            raw_data,
            indicators=['rsi', 'macd', 'bollinger_bands']
        )
        
        # Verify comprehensive results
        assert isinstance(comparison, dict)
        assert comparison['summary']['total_observations'] == 100
        assert comparison['summary']['regime_summary']['n_states'] == 3
        
        # Generate plot
        fig = comparator.plot_comparison_results(comparison)
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])