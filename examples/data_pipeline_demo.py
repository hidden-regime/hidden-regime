"""
Data Pipeline Demo for hidden-regime package.

Demonstrates how to use the data loading, preprocessing, and validation
functionality with real market data.
"""

import sys
import warnings
from datetime import datetime, timedelta

# Import the hidden-regime package
import hidden_regime as hr


def main():
    """Demonstrate the data pipeline functionality."""
    print("Hidden Regime Data Pipeline Demo")
    print("=" * 40)
    
    # Example 1: Basic data loading
    print("\n1. Basic Data Loading")
    print("-" * 20)
    
    try:
        # Load Apple stock data for a reliable historical period
        end_date = "2024-06-30"
        start_date = "2024-01-01"
        
        print(f"Loading AAPL data from {start_date} to {end_date}...")
        
        # Use the convenience function
        data = hr.load_stock_data('AAPL', start_date, end_date)
        
        print(f"✓ Successfully loaded {len(data)} days of data")
        print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"  Price range: ${data['price'].min():.2f} to ${data['price'].max():.2f}")
        print(f"  Columns: {list(data.columns)}")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Note: This demo requires yfinance. Install with: pip install yfinance")
        return
    
    # Example 2: Data validation
    print("\n2. Data Quality Validation")
    print("-" * 25)
    
    validation_result = hr.validate_data(data, 'AAPL')
    
    print(f"Data is valid: {validation_result.is_valid}")
    print(f"Quality score: {validation_result.quality_score:.2f}")
    
    if validation_result.issues:
        print("Issues found:")
        for issue in validation_result.issues[:3]:  # Show first 3 issues
            print(f"  - {issue}")
    
    if validation_result.warnings:
        print("Warnings:")
        for warning in validation_result.warnings[:3]:  # Show first 3 warnings
            print(f"  - {warning}")
    
    if validation_result.recommendations:
        print("Recommendations:")
        for rec in validation_result.recommendations[:2]:  # Show first 2 recommendations
            print(f"  - {rec}")
    
    # Example 3: Using individual classes with custom configuration
    print("\n3. Custom Configuration Example")
    print("-" * 30)
    
    # Create custom configurations
    data_config = hr.DataConfig(
        use_ohlc_average=False,  # Use close price instead
        cache_enabled=True,
        max_missing_data_pct=0.02  # Stricter missing data tolerance
    )
    
    validation_config = hr.ValidationConfig(
        outlier_method='zscore',
        outlier_threshold=2.5,
        max_daily_return=0.2  # 20% max daily return
    )
    
    preprocessing_config = hr.PreprocessingConfig(
        return_method='simple',  # Use simple returns instead of log
        calculate_volatility=True,
        volatility_window=30,
        apply_smoothing=True,
        smoothing_window=5
    )
    
    # Use individual classes with custom config
    loader = hr.DataLoader(config=data_config)
    validator = hr.DataValidator(config=validation_config)
    preprocessor = hr.DataPreprocessor(
        preprocessing_config=preprocessing_config,
        validation_config=validation_config
    )
    
    print("Loading data with custom configuration...")
    
    # Load a different stock
    try:
        custom_data = loader.load_stock_data('MSFT', start_date, end_date)
        print(f"✓ Loaded {len(custom_data)} days of MSFT data")
        
        # Preprocess the data
        processed_data = preprocessor.process_data(custom_data, ticker='MSFT')
        print(f"✓ Processed data - added {len(processed_data.columns) - len(custom_data.columns)} new columns")
        
        # Get data summary
        summary = preprocessor.get_data_summary(processed_data)
        print(f"✓ Data summary: {summary['n_observations']} observations over {summary['date_range']['days']} days")
        
        # Validate processed data
        custom_validation = validator.validate_data(processed_data, 'MSFT')
        print(f"✓ Custom validation - Quality score: {custom_validation.quality_score:.2f}")
        
    except Exception as e:
        print(f"✗ Error with custom configuration: {e}")
    
    # Example 4: Multi-stock analysis
    print("\n4. Multi-Stock Analysis")
    print("-" * 22)
    
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    try:
        multi_data = loader.load_multiple_stocks(tickers, start_date, end_date)
        
        print(f"✓ Successfully loaded data for {len(multi_data)} stocks:")
        for ticker, stock_data in multi_data.items():
            print(f"  - {ticker}: {len(stock_data)} days")
        
        # Process multiple series
        processed_multi = preprocessor.process_multiple_series(multi_data)
        
        print("✓ Processed all stocks with alignment and feature engineering")
        
        # Validate each stock
        print("\nValidation results:")
        for ticker, stock_data in processed_multi.items():
            validation = validator.validate_data(stock_data, ticker)
            print(f"  - {ticker}: Quality {validation.quality_score:.2f}, Valid: {validation.is_valid}")
    
    except Exception as e:
        print(f"✗ Error with multi-stock analysis: {e}")
    
    # Example 5: Error handling demonstration
    print("\n5. Error Handling Demo")
    print("-" * 21)
    
    try:
        # Try to load invalid ticker
        print("Attempting to load invalid ticker...")
        bad_data = hr.load_stock_data('INVALID_TICKER_XYZ', start_date, end_date)
    except hr.DataLoadError as e:
        print(f"✓ Correctly caught DataLoadError: {e}")
    except Exception as e:
        print(f"✓ Caught other error: {e}")
    
    try:
        # Try invalid date range
        print("Attempting invalid date range...")
        hr.load_stock_data('AAPL', end_date, start_date)  # Reversed dates
    except hr.ValidationError as e:
        print(f"✓ Correctly caught ValidationError: {e}")
    except Exception as e:
        print(f"✓ Caught other error: {e}")
    
    print("\n" + "=" * 40)
    print("Demo completed successfully!")
    print("The hidden-regime data pipeline is ready for use.")
    print("\nNext steps:")
    print("- Install with: pip install hidden-regime")
    print("- Import with: import hidden_regime as hr")
    print("- Load data with: hr.load_stock_data('AAPL', '2023-01-01', '2023-12-31')")
    print("- Validate with: hr.validate_data(data, 'AAPL')")


if __name__ == "__main__":
    # Suppress warnings for cleaner demo output
    warnings.filterwarnings('ignore')
    main()