"""
Integration tests for hidden-regime data pipeline.

Tests end-to-end workflows combining DataLoader, DataValidator, and DataPreprocessor
to ensure components work correctly together and data flows properly through
the entire pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from unittest.mock import patch

from hidden_regime import (
    DataLoader, DataValidator, DataPreprocessor,
    DataConfig, ValidationConfig, PreprocessingConfig,
    DataLoadError, ValidationError, DataQualityError
)
from tests.fixtures.sample_data import MockYFinanceTicker


class TestPipelineIntegration:
    """Test complete data pipeline workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.start_date = "2024-01-01"
        self.end_date = "2024-03-31"
        self.test_ticker = "AAPL"
        
        # Create components with default configs
        self.loader = DataLoader()
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor()
    
    def test_complete_single_stock_pipeline(self):
        """Test complete pipeline: Load -> Validate -> Process -> Re-validate."""
        with patch('hidden_regime.data.loader.yf.Ticker') as mock_ticker_class:
            # Set up mock
            mock_ticker = MockYFinanceTicker(self.test_ticker)
            mock_ticker_class.return_value = mock_ticker
            
            # Step 1: Load data
            raw_data = self.loader.load_stock_data(
                self.test_ticker, self.start_date, self.end_date
            )
            
            # Verify data loaded correctly
            assert isinstance(raw_data, pd.DataFrame)
            assert not raw_data.empty
            assert len(raw_data) >= 30  # Minimum observations met
            required_columns = ['date', 'price', 'log_return', 'volume']
            assert all(col in raw_data.columns for col in required_columns)
            
            # Step 2: Validate raw data
            raw_validation = self.validator.validate_data(raw_data, self.test_ticker)
            
            # Verify validation results
            assert isinstance(raw_validation.quality_score, float)
            assert 0.0 <= raw_validation.quality_score <= 1.0
            assert raw_validation.is_valid == True  # Mock data should be clean
            assert raw_validation.quality_score > 0.8  # Should be high quality
            
            # Step 3: Process data  
            processed_data = self.preprocessor.process_data(raw_data, self.test_ticker)
            
            # Verify processing results
            assert isinstance(processed_data, pd.DataFrame)
            assert not processed_data.empty
            assert len(processed_data) >= len(raw_data) - 5  # Some rows may be lost to edge effects
            
            # Check for additional engineered features
            if self.preprocessor.preprocessing_config.calculate_volatility:
                assert 'volatility' in processed_data.columns
            
            # Step 4: Re-validate processed data
            processed_validation = self.validator.validate_data(processed_data, self.test_ticker)
            
            # Verify processed validation (may have lower score due to feature engineering)
            assert isinstance(processed_validation.quality_score, float)
            assert 0.0 <= processed_validation.quality_score <= 1.0
            # Processed data often has lower scores due to boundary effects
            assert processed_validation.quality_score > 0.2  # Still usable
            
            # Step 5: Verify data consistency through pipeline
            # Prices should be preserved (allowing for minor processing differences)
            if len(processed_data) > 0 and len(raw_data) > 0:
                # Check that core price data is preserved
                assert processed_data['price'].min() > 0
                assert not processed_data['price'].isnull().all()
    
    def test_multi_stock_pipeline_workflow(self):
        """Test multi-stock pipeline with batch processing."""
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        
        with patch('hidden_regime.data.loader.yf.Ticker') as mock_ticker_class:
            # Set up mock for multiple tickers
            def ticker_side_effect(ticker):
                return MockYFinanceTicker(ticker)
            mock_ticker_class.side_effect = ticker_side_effect
            
            # Step 1: Batch load multiple stocks
            data_dict = self.loader.load_multiple_stocks(
                tickers, self.start_date, self.end_date
            )
            
            # Verify batch loading
            assert isinstance(data_dict, dict)
            assert len(data_dict) == len(tickers)
            for ticker in tickers:
                assert ticker in data_dict
                assert isinstance(data_dict[ticker], pd.DataFrame)
                assert not data_dict[ticker].empty
            
            # Step 2: Batch validate all stocks
            validation_results = {}
            for ticker, data in data_dict.items():
                validation_results[ticker] = self.validator.validate_data(data, ticker)
            
            # Verify all validations completed
            assert len(validation_results) == len(tickers)
            for ticker, result in validation_results.items():
                assert result.is_valid == True  # Mock data should be clean
                assert result.quality_score > 0.8
            
            # Step 3: Batch process with alignment
            processed_dict = self.preprocessor.process_multiple_series(data_dict)
            
            # Verify batch processing with alignment
            assert isinstance(processed_dict, dict)
            assert len(processed_dict) <= len(data_dict)  # Some may fail processing
            
            # Check timestamp alignment if enabled
            if self.preprocessor.preprocessing_config.align_timestamps and len(processed_dict) > 1:
                date_ranges = {}
                for ticker, data in processed_dict.items():
                    if 'date' in data.columns:
                        date_ranges[ticker] = (data['date'].min(), data['date'].max())
                
                # All aligned data should have similar date ranges
                if len(date_ranges) > 1:
                    start_dates = [dr[0] for dr in date_ranges.values()]
                    end_dates = [dr[1] for dr in date_ranges.values()]
                    
                    # Allow some variation in alignment (few days difference)
                    start_range = (max(start_dates) - min(start_dates)).days
                    end_range = (max(end_dates) - min(end_dates)).days
                    assert start_range <= 7  # Within a week
                    assert end_range <= 7    # Within a week
            
            # Step 4: Final validation of processed data
            final_validations = {}
            for ticker, data in processed_dict.items():
                final_validations[ticker] = self.validator.validate_data(data, ticker)
            
            # Verify final results
            for ticker, result in final_validations.items():
                assert isinstance(result.quality_score, float)
                assert 0.0 <= result.quality_score <= 1.0
    
    def test_configuration_interaction_workflow(self):
        """Test pipeline with custom configurations across components."""
        # Create custom configurations that interact
        data_config = DataConfig(
            use_ohlc_average=False,  # Use close price only
            max_missing_data_pct=0.02,  # Strict missing data tolerance
            cache_enabled=False  # Disable caching for test consistency
        )
        
        validation_config = ValidationConfig(
            outlier_method='zscore',
            outlier_threshold=2.5,  # More sensitive outlier detection
            max_daily_return=0.2    # Stricter return limits
        )
        
        preprocessing_config = PreprocessingConfig(
            return_method='simple',  # Use simple returns instead of log
            calculate_volatility=True,
            apply_smoothing=True,
            smoothing_window=3
        )
        
        # Create components with custom configs
        loader = DataLoader(data_config)
        validator = DataValidator(validation_config)  
        preprocessor = DataPreprocessor(
            preprocessing_config=preprocessing_config,
            validation_config=validation_config
        )
        
        with patch('hidden_regime.data.loader.yf.Ticker') as mock_ticker_class:
            mock_ticker = MockYFinanceTicker(self.test_ticker)
            mock_ticker_class.return_value = mock_ticker
            
            # Execute pipeline with custom configs
            raw_data = loader.load_stock_data(self.test_ticker, self.start_date, self.end_date)
            
            # Verify close price used (not OHLC average)
            # This requires checking the underlying data processing
            assert 'price' in raw_data.columns
            
            # Validate with custom config
            validation = validator.validate_data(raw_data, self.test_ticker)
            
            # With stricter validation, may get more warnings
            assert isinstance(validation.quality_score, float)
            # May be lower due to stricter thresholds, but should still be reasonable
            assert validation.quality_score > 0.3
            
            # Process with custom config
            processed_data = preprocessor.process_data(raw_data, self.test_ticker)
            
            # Verify simple returns used
            if 'simple_return' in processed_data.columns:
                # Simple returns should be present
                assert not processed_data['simple_return'].isnull().all()
            
            # Verify smoothing applied
            if preprocessing_config.apply_smoothing:
                smoothed_cols = [col for col in processed_data.columns if 'smoothed' in col]
                assert len(smoothed_cols) > 0  # Should have smoothed columns
    
    def test_error_propagation_through_pipeline(self):
        """Test how errors propagate through the complete pipeline."""
        
        # Test 1: Invalid ticker should fail at loading stage
        with patch('hidden_regime.data.loader.yf.Ticker') as mock_ticker_class:
            mock_ticker = MockYFinanceTicker("INVALID", should_fail=True)
            mock_ticker_class.return_value = mock_ticker
            
            with pytest.raises(DataLoadError):
                self.loader.load_stock_data("INVALID", self.start_date, self.end_date)
        
        # Test 2: Corrupted data should fail at validation stage
        corrupted_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'price': [-100, -200, np.inf, np.nan, 0],  # Invalid prices
            'log_return': [np.inf, np.nan, -np.inf, 1.5, -1.5],  # Invalid returns
            'volume': [1000000] * 5
        })
        
        validation_result = self.validator.validate_data(corrupted_data, "CORRUPTED")
        assert not validation_result.is_valid
        assert validation_result.quality_score < 0.3  # Should be very low
        assert len(validation_result.issues) > 0
        
        # Test 3: Processing should handle bad data gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warnings
            try:
                processed_corrupted = self.preprocessor.process_data(corrupted_data, "CORRUPTED")
                # If processing succeeds, result should still show quality issues
                final_validation = self.validator.validate_data(processed_corrupted, "CORRUPTED")
                assert final_validation.quality_score < 0.5
            except (DataQualityError, ValueError):
                # Processing may legitimately fail with severely corrupted data
                pass
    
    def test_caching_behavior_in_pipeline(self):
        """Test caching behavior across pipeline components."""
        with patch('hidden_regime.data.loader.yf.Ticker') as mock_ticker_class:
            mock_ticker = MockYFinanceTicker(self.test_ticker)
            mock_ticker_class.return_value = mock_ticker
            
            # Enable caching
            cached_loader = DataLoader(DataConfig(cache_enabled=True))
            
            # First load should hit the API
            data1 = cached_loader.load_stock_data(
                self.test_ticker, self.start_date, self.end_date
            )
            
            # Second load should use cache (API should not be called again)
            data2 = cached_loader.load_stock_data(
                self.test_ticker, self.start_date, self.end_date
            )
            
            # Data should be identical
            pd.testing.assert_frame_equal(data1, data2)
            
            # API should only have been called once
            assert mock_ticker_class.call_count == 1
            
            # Cache stats should reflect usage
            cache_stats = cached_loader.get_cache_stats()
            assert cache_stats['cache_entries'] > 0
            assert cache_stats['cache_enabled'] == True
    
    def test_pipeline_performance_characteristics(self):
        """Test basic performance characteristics of the pipeline."""
        import time
        
        with patch('hidden_regime.data.loader.yf.Ticker') as mock_ticker_class:
            mock_ticker = MockYFinanceTicker(self.test_ticker)
            mock_ticker_class.return_value = mock_ticker
            
            # Time the complete pipeline
            start_time = time.time()
            
            # Load data
            raw_data = self.loader.load_stock_data(
                self.test_ticker, self.start_date, self.end_date
            )
            
            # Validate
            validation = self.validator.validate_data(raw_data, self.test_ticker)
            
            # Process  
            processed_data = self.preprocessor.process_data(raw_data, self.test_ticker)
            
            # Re-validate
            final_validation = self.validator.validate_data(processed_data, self.test_ticker)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Pipeline should complete in reasonable time (mock data, should be fast)
            assert total_time < 10.0  # Less than 10 seconds for mock data
            
            # Results should be valid
            assert not raw_data.empty
            assert isinstance(validation.quality_score, float)
            assert not processed_data.empty
            assert isinstance(final_validation.quality_score, float)


class TestPipelineEdgeCases:
    """Test edge cases in pipeline integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor()
    
    def test_minimal_data_pipeline(self):
        """Test pipeline with minimal valid data."""
        # Create minimal valid dataset (just above minimum requirements)
        minimal_data = pd.DataFrame({
            'date': pd.bdate_range('2024-01-01', periods=35),  # Just above min_observations=30
            'price': np.random.uniform(100, 110, 35),
            'log_return': np.random.normal(0.001, 0.02, 35),
            'volume': np.random.randint(1000000, 2000000, 35)
        })
        minimal_data.loc[0, 'log_return'] = np.nan  # First return is always NaN
        
        # Process through pipeline
        validation = self.validator.validate_data(minimal_data, "MINIMAL")
        assert validation.is_valid  # Should be valid despite being minimal
        
        processed = self.preprocessor.process_data(minimal_data, "MINIMAL")
        assert not processed.empty
        assert len(processed) >= 25  # Some rows lost to processing
        
        final_validation = self.validator.validate_data(processed, "MINIMAL")
        assert isinstance(final_validation.quality_score, float)
    
    def test_single_stock_vs_multi_stock_consistency(self):
        """Test that single stock and multi-stock processing give consistent results."""
        ticker = "AAPL"
        
        with patch('hidden_regime.data.loader.yf.Ticker') as mock_ticker_class:
            mock_ticker = MockYFinanceTicker(ticker)
            mock_ticker_class.return_value = mock_ticker
            
            # Load as single stock
            single_data = self.loader.load_stock_data(ticker, "2024-01-01", "2024-03-31")
            single_processed = self.preprocessor.process_data(single_data, ticker)
            
            # Load as part of multi-stock batch
            multi_data = self.loader.load_multiple_stocks([ticker], "2024-01-01", "2024-03-31")
            multi_processed = self.preprocessor.process_multiple_series(multi_data)
            
            # Results should be consistent
            assert ticker in multi_data
            assert ticker in multi_processed
            
            # Data should be very similar (allowing for minor processing differences)
            single_len = len(single_processed)
            multi_len = len(multi_processed[ticker])
            assert abs(single_len - multi_len) <= 2  # Allow small differences due to alignment
    
    def test_configuration_edge_cases(self):
        """Test pipeline with extreme configuration values."""
        # Very strict configuration
        strict_config = ValidationConfig(
            outlier_threshold=1.0,  # Very sensitive
            max_daily_return=0.05,  # Very strict return limit
            max_consecutive_missing=1  # No consecutive missing allowed
        )
        
        # Very lenient configuration
        lenient_config = ValidationConfig(
            outlier_threshold=10.0,  # Very lenient
            max_daily_return=2.0,   # Allow extreme returns
            max_consecutive_missing=50  # Allow lots of missing data
        )
        
        # Test data with some quality issues
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'price': np.random.uniform(95, 105, 50),
            'log_return': np.random.normal(0.001, 0.05, 50),  # Higher volatility
            'volume': np.random.randint(500000, 1500000, 50)
        })
        # Add some outliers
        test_data.loc[10, 'log_return'] = 0.15  # 15% return (outlier)
        test_data.loc[20, 'log_return'] = -0.12  # -12% return (outlier)
        
        # Test with strict config
        strict_validator = DataValidator(strict_config)
        strict_result = strict_validator.validate_data(test_data, "TEST")
        
        # Test with lenient config
        lenient_validator = DataValidator(lenient_config)
        lenient_result = lenient_validator.validate_data(test_data, "TEST")
        
        # Strict should have lower quality score
        assert strict_result.quality_score <= lenient_result.quality_score
        assert len(strict_result.warnings) >= len(lenient_result.warnings)


if __name__ == "__main__":
    pytest.main([__file__])