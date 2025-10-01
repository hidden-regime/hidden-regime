# Hidden Regime Data Pipeline Documentation

The data pipeline is the foundation of the Hidden Regime package, providing robust data loading, preprocessing, and validation capabilities for financial market data analysis.

## Table of Contents

1. [Overview](#overview)
2. [Data Pipeline Architecture](#data-pipeline-architecture)
3. [Data Loading System](#data-loading-system)
4. [Data Validation Framework](#data-validation-framework)
5. [Quality Score System](#quality-score-system)
6. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
7. [Configuration System](#configuration-system)
8. [Best Practices](#best-practices)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [API Reference](#api-reference)

## Overview

The Hidden Regime data pipeline is designed to handle the complexities of financial market data with enterprise-grade reliability and flexibility. It addresses common challenges in financial data:

- **Data Quality Issues**: Missing values, outliers, price errors, stock splits
- **Multiple Data Sources**: Extensible architecture supporting various data providers
- **Real-time Processing**: Streaming data capabilities with caching and rate limiting
- **Validation Complexity**: Comprehensive quality assessment for trading applications
- **Preprocessing Needs**: Feature engineering, outlier handling, and data transformation

### Design Philosophy

1. **Robustness First**: Graceful error handling and comprehensive validation
2. **Configurability**: Extensive configuration options for different use cases
3. **Performance**: Efficient caching, vectorized operations, and memory management
4. **Extensibility**: Plugin architecture for new data sources and validation rules
5. **Transparency**: Detailed logging, metrics, and quality scoring

## Data Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│   Data Loader    │───▶│  Raw Data Store │
│  (yfinance,     │    │  - Caching       │    │  (DataFrame)    │
│   Alpha Vantage,│    │  - Rate Limiting │    │                 │
│   Custom APIs)  │    │  - Retry Logic   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │                        │
                                 ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Validation      │◀───│  Data Validator  │◀───│ Quality Control │
│ Results         │    │  - Multi-layer   │    │ - Threshold     │
│ - Issues        │    │  - Configurable  │    │   Checking      │
│ - Warnings      │    │  - Scoring       │    │ - Anomaly       │
│ - Quality Score │    │  - Metrics       │    │   Detection     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Processed     │◀───│ Data Preprocessor│◀───│  Feature        │
│   Data Store    │    │ - Outlier Handle │    │  Engineering    │
│  (Clean Data)   │    │ - Missing Data   │    │ - Volatility    │
│                 │    │ - Feature Eng.   │    │ - Technical     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

- **DataLoader**: Multi-source data acquisition with robust error handling
- **DataValidator**: Comprehensive data quality assessment and scoring
- **DataPreprocessor**: Advanced data cleaning and feature engineering
- **Configuration System**: Flexible, dataclass-based configuration management

## Data Loading System

### DataLoader Class

The `DataLoader` is the entry point for all data acquisition operations. It provides a unified interface across multiple data sources while handling common challenges like rate limiting, caching, and error recovery.

#### Core Features

1. **Multi-Source Support**
   ```python
   # Currently supported
   loader = DataLoader()
   data = loader.load_stock_data('AAPL', '2024-01-01', '2024-12-31', source='yfinance')
   
   # Extensible for future sources
   # data = loader.load_stock_data('AAPL', '2024-01-01', '2024-12-31', source='alpha_vantage')
   ```

2. **Intelligent Caching**
   - In-memory caching with configurable expiry
   - Automatic cache key generation based on parameters
   - Cache hit/miss statistics for monitoring
   
3. **Rate Limiting & Retry Logic**
   - Configurable requests per minute limits
   - Exponential backoff retry strategy
   - Graceful degradation under API constraints

4. **Data Processing Pipeline**
   ```
   Raw API Data → Column Standardization → OHLC Processing → Return Calculation → Quality Validation
   ```

#### Data Schema

All loaded data follows a consistent schema:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Trading date (timezone-aware) |
| `price` | float | Price (OHLC average or Close based on config) |
| `log_return` | float | Natural log return (price_t / price_t-1) |
| `volume` | int | Trading volume (optional, based on config) |

#### Configuration Options

```python
from hidden_regime import DataConfig

config = DataConfig(
    default_source='yfinance',           # Data source preference
    use_ohlc_average=True,               # True: (O+H+L+C)/4, False: Close only
    include_volume=True,                 # Include volume data
    max_missing_data_pct=0.05,          # 5% maximum missing data tolerance
    min_observations=30,                 # Minimum data points required
    cache_enabled=True,                  # Enable in-memory caching
    cache_expiry_hours=24,               # Cache validity period
    requests_per_minute=60,              # API rate limiting
    retry_attempts=3,                    # Number of retry attempts
    retry_delay_seconds=1.0              # Base retry delay
)

loader = DataLoader(config=config)
```

### Multi-Stock Loading

The system provides optimized batch loading for multiple securities:

```python
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
data_dict = loader.load_multiple_stocks(tickers, '2024-01-01', '2024-12-31')

# Returns: {'AAPL': DataFrame, 'GOOGL': DataFrame, ...}
# Automatic error handling - failed tickers excluded with warnings
```

## Data Validation Framework

### Validation Philosophy

The validation system implements a **multi-layered, penalty-based approach** that evaluates financial data quality across six critical dimensions. Rather than simple pass/fail validation, it provides nuanced quality assessment suitable for quantitative finance applications.

### Validation Layers

#### 1. Structure Validation
**Purpose**: Ensure basic DataFrame integrity and required schema compliance

**Checks**:
- Non-empty DataFrame
- Required columns present (`price` is mandatory)
- Reasonable column count (<20 to detect data corruption)
- Proper data types

**Failure Impact**: Critical - marks data as invalid

#### 2. Date/Temporal Validation  
**Purpose**: Assess time series consistency and trading calendar alignment

**Checks**:
- Valid datetime format
- Chronological ordering
- No duplicate dates
- Business day coverage analysis
- Gap detection (>14 day gaps flagged)

**Key Metrics**:
```python
{
    'date_range_days': int,              # Total calendar days covered
    'n_dates': int,                      # Number of observations
    'duplicate_dates': int,              # Duplicate date count
    'date_coverage_ratio': float         # Actual/expected business days
}
```

**Quality Impact**: 
- Issues: Critical (e.g., duplicates, invalid formats)
- Warnings: Moderate (e.g., gaps, low coverage)

#### 3. Price Validation
**Purpose**: Identify price data anomalies and potential data errors

**Checks**:
- Non-positive prices (critical error)
- Extremely low prices (<$0.01 by default)
- Large price changes (>50% daily - potential splits/errors)
- Price distribution analysis

**Algorithm for Split Detection**:
```python
price_changes = prices.pct_change().abs()
large_changes = (price_changes > 0.5).sum()
# Flags potential stock splits or data errors
```

**Quality Impact**: Logarithmic penalty based on anomaly count

#### 4. Return Validation
**Purpose**: Statistical validation of return series properties

**Checks**:
- Infinite returns (critical - indicates division by zero)
- Extreme returns (>50% daily by default)
- Volatility assessment (very high >10% or very low <0.5%)
- Distribution properties (skewness, kurtosis)

**Statistical Metrics**:
```python
{
    'return_mean': float,                # Average daily return
    'return_std': float,                 # Daily volatility
    'return_skewness': float,            # Distribution asymmetry
    'return_kurtosis': float,            # Tail heaviness
    'extreme_returns': int,              # Count of extreme observations
    'infinite_returns': int              # Count of infinite values
}
```

#### 5. Missing Data Assessment
**Purpose**: Evaluate data completeness and identify problematic patterns

**Missing Data Hierarchy**:
1. **Overall Missing**: Total percentage across all cells
2. **Column-wise Missing**: Per-column completeness assessment  
3. **Consecutive Missing**: Temporal gaps in critical columns
4. **Pattern Analysis**: Systematic vs. random missing data

**Thresholds**:
- Overall >10% missing: Critical issue
- Column >20% missing: Critical issue  
- Overall >5% missing: Warning
- Column >10% missing: Warning
- Consecutive missing >5: Critical issue

**Quality Impact**: Heavy penalty (missing_percentage × 2.0)

#### 6. Outlier Detection
**Purpose**: Identify anomalous observations that may impact model performance

**Detection Methods** (configurable):

1. **IQR Method** (default):
   ```python
   Q1 = returns.quantile(0.25)
   Q3 = returns.quantile(0.75)
   IQR = Q3 - Q1
   outliers = (returns < Q1 - 1.5*IQR) | (returns > Q3 + 1.5*IQR)
   ```

2. **Z-Score Method**:
   ```python
   z_scores = np.abs(stats.zscore(returns))
   outliers = z_scores > 3.0  # configurable threshold
   ```

3. **Isolation Forest** (requires scikit-learn):
   ```python
   from sklearn.ensemble import IsolationForest
   iso_forest = IsolationForest(contamination=0.1)
   outliers = iso_forest.fit_predict(returns) == -1
   ```

**Quality Impact**: Proportional penalty (outlier_percentage × 1.0)

### Volume Validation (Optional)

When volume data is available, additional checks include:
- Negative volume detection
- Zero-volume days analysis
- Extreme volume spikes (>100x median)
- Volume-return correlation assessment

## Quality Score System

### Scoring Formula

The quality score uses a **penalty-based calculation** starting from perfection (1.0):

```python
def calculate_quality_score(issues, warnings, metrics):
    score = 1.0
    
    # Major penalties (critical issues)
    score -= len(issues) * 0.2              # -20% per critical issue
    
    # Minor penalties (warnings)  
    score -= len(warnings) * 0.05           # -5% per warning
    
    # Data-specific penalties
    score -= metrics.get('missing_percentage', 0) * 2.0      # -200% of missing data %
    score -= metrics.get('outlier_percentage', 0) * 1.0      # -100% of outlier %
    score -= metrics.get('extreme_returns', 0) * 0.1         # -10% per extreme return
    
    return max(0.0, min(1.0, score))  # Clamp to [0,1]
```

### Score Interpretation Guide

| Score Range | Quality Level | Description | Recommended Action |
|-------------|---------------|-------------|--------------------|
| **0.90-1.00** | **Excellent** | High-quality data suitable for all applications | Proceed with confidence |
| **0.70-0.89** | **Good** | Minor issues that don't affect most analyses | Use with standard preprocessing |
| **0.50-0.69** | **Moderate** | Noticeable quality issues requiring attention | Apply data cleaning, monitor results |
| **0.30-0.49** | **Poor** | Significant quality problems | Extensive preprocessing required |
| **0.00-0.29** | **Very Poor** | Likely unusable for quantitative analysis | Consider alternative data sources |

### Real-World Score Examples

Based on empirical testing with market data:

```python
# Typical scores for different scenarios:

# Clean daily stock data (6 months, major stock)
# Score: 0.85-0.95
# - Few missing days (holidays/weekends expected)
# - Normal volatility patterns
# - Clean price action

# Processed data with engineered features  
# Score: 0.40-0.60 (appears lower due to boundary effects)
# - Volatility calculations create NaN at edges
# - Multiple columns increase warning potential
# - Still high-quality, just more complex

# Cryptocurrency data (24/7 trading)
# Score: 0.70-0.90  
# - Higher volatility triggers warnings
# - More outliers due to market nature
# - Complete temporal coverage

# Low-volume/penny stocks
# Score: 0.30-0.70
# - Many zero-volume days
# - Price manipulation artifacts
# - Higher missing data rates
```

## Data Preprocessing Pipeline

### DataPreprocessor Class

The preprocessor applies sophisticated data cleaning and feature engineering techniques optimized for financial time series.

### Processing Pipeline

```python
def process_data(data, ticker=None):
    """
    1. Handle Missing Values    → Interpolation/forward-fill strategies
    2. Outlier Detection       → IQR/Z-score/Isolation Forest methods  
    3. Outlier Treatment       → Winsorization (capping at percentiles)
    4. Return Calculation      → Log or simple returns
    5. Feature Engineering     → Volatility, technical indicators
    6. Data Smoothing          → Rolling averages (optional)
    7. Final Validation        → Quality checks on processed data
    """
```

### Missing Value Handling

**Strategy Selection** (configurable):

1. **Linear Interpolation** (default for prices):
   ```python
   data['price'] = data['price'].interpolate(method='linear')
   ```

2. **Forward Fill** (carry last observation forward):
   ```python
   data['price'] = data['price'].ffill()
   ```

3. **Backward Fill** (use next observation):
   ```python
   data['price'] = data['price'].bfill()
   ```

**Volume Handling**: Missing volume filled with median volume (less critical than prices)

### Outlier Treatment

**Detection-Treatment Pipeline**:
1. **Detect outliers** using configured method (IQR/Z-score/Isolation Forest)
2. **Assess impact** - warn if >5% outliers detected
3. **Apply winsorization** - cap outliers at 1st/99th percentiles
4. **Preserve data integrity** - log all modifications

**Winsorization Example**:
```python
# Instead of removing outliers, cap them at reasonable bounds
returns = data['log_return']
lower_bound = returns.quantile(0.01)  # 1st percentile
upper_bound = returns.quantile(0.99)  # 99th percentile
data['log_return'] = returns.clip(lower_bound, upper_bound)
```

### Feature Engineering

**Standard Features** (always calculated):
- `log_return`: Natural log returns for statistical modeling
- `date`: Properly formatted datetime index

**Optional Features** (configurable):
- `volatility`: Rolling standard deviation of returns
- `abs_return`: Absolute value of returns (alternative volatility measure)
- `avg_abs_return`: Rolling average of absolute returns
- `price_smoothed`: Smoothed price series (moving average)
- `log_return_smoothed`: Smoothed return series

**Volatility Calculation**:
```python
# Rolling volatility (20-day window default)
data['volatility'] = data['log_return'].rolling(window=20).std()

# Absolute return measures
data['abs_return'] = data['log_return'].abs()
data['avg_abs_return'] = data['abs_return'].rolling(window=20).mean()
```

### Multi-Asset Processing

For portfolio analysis, the preprocessor provides:

1. **Timestamp Alignment**: Synchronize multiple time series to common business day calendar
2. **Missing Data Coordination**: Handle gaps consistently across assets
3. **Feature Standardization**: Apply same transformations to all assets
4. **Cross-Asset Validation**: Ensure data quality consistency

```python
# Process multiple assets with alignment
processed_data = preprocessor.process_multiple_series({
    'AAPL': apple_data,
    'GOOGL': google_data,
    'MSFT': microsoft_data
})
# Returns aligned DataFrames with consistent timestamps
```

## Configuration System

### Design Philosophy

The configuration system uses **dataclasses** for type safety and clear documentation, with **sensible defaults** for immediate usability and **extensive customization** for advanced use cases.

### DataConfig

```python
@dataclass
class DataConfig:
    """Configuration for data loading operations."""
    
    # Data source settings
    default_source: str = 'yfinance'         # Primary data provider
    use_ohlc_average: bool = True            # Price calculation method
    include_volume: bool = True              # Volume data inclusion
    
    # Data quality settings  
    max_missing_data_pct: float = 0.05       # 5% maximum missing data
    min_observations: int = 30               # Minimum data points required
    
    # Caching settings
    cache_enabled: bool = True               # Enable in-memory caching
    cache_expiry_hours: int = 24            # Cache validity period
    
    # Rate limiting
    requests_per_minute: int = 60            # API rate limiting
    retry_attempts: int = 3                  # Number of retry attempts
    retry_delay_seconds: float = 1.0         # Base retry delay
```

### ValidationConfig

```python
@dataclass
class ValidationConfig:
    """Configuration for data validation operations."""
    
    # Outlier detection
    outlier_method: str = 'iqr'              # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 3.0           # Standard deviations for zscore
    iqr_multiplier: float = 1.5              # IQR multiplier for outlier detection
    
    # Price validation
    min_price: float = 0.01                  # Minimum valid price
    max_daily_return: float = 0.5            # 50% max daily return
    
    # Date validation
    min_trading_days_per_month: int = 15     # Minimum trading days expected
    
    # Missing data handling
    max_consecutive_missing: int = 5         # Max consecutive missing values
    interpolation_method: str = 'linear'     # 'linear', 'forward', 'backward'
```

### PreprocessingConfig

```python
@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing operations."""
    
    # Return calculation
    return_method: str = 'log'               # 'log', 'simple'
    
    # Smoothing/filtering
    apply_smoothing: bool = False            # Apply data smoothing
    smoothing_window: int = 5                # Smoothing window size
    
    # Feature engineering
    calculate_volatility: bool = True        # Calculate volatility features
    volatility_window: int = 20              # Volatility calculation window
    
    # Data alignment
    align_timestamps: bool = True            # Align multi-asset timestamps
    fill_method: str = 'forward'             # 'forward', 'backward', 'interpolate'
```

### Configuration Examples

**Conservative Trading Setup**:
```python
# For low-risk trading strategies
conservative_config = ValidationConfig(
    outlier_method='zscore',
    outlier_threshold=2.0,           # More sensitive outlier detection
    max_daily_return=0.1,            # 10% max daily return
    max_consecutive_missing=2        # Stricter missing data tolerance
)
```

**Research/Backtesting Setup**:
```python
# For historical analysis and research
research_config = ValidationConfig(
    outlier_method='iqr',
    outlier_threshold=5.0,           # More lenient outlier detection
    max_daily_return=1.0,            # Allow extreme historical events
    max_consecutive_missing=10       # More flexible missing data
)
```

**High-Frequency Setup**:
```python
# For intraday/high-frequency applications
hf_config = DataConfig(
    use_ohlc_average=False,          # Use close prices only
    min_observations=100,            # Require more data points
    max_missing_data_pct=0.01       # Very strict missing data tolerance
)
```

## Best Practices

### Data Loading Best Practices

1. **Always Validate After Loading**:
   ```python
   data = hr.load_stock_data('AAPL', '2024-01-01', '2024-12-31')
   validation_result = hr.validate_data(data, 'AAPL')
   
   if not validation_result.is_valid:
       print(f"Data quality issues: {validation_result.issues}")
   ```

2. **Use Appropriate Date Ranges**:
   ```python
   # Good: Reasonable historical period
   data = loader.load_stock_data('AAPL', '2023-01-01', '2024-01-01')
   
   # Avoid: Very short periods (may not meet minimum observations)
   # data = loader.load_stock_data('AAPL', '2024-01-01', '2024-01-07')
   ```

3. **Handle Failed Loads Gracefully**:
   ```python
   try:
       data = hr.load_stock_data(ticker, start_date, end_date)
   except hr.DataLoadError as e:
       logger.warning(f"Failed to load {ticker}: {e}")
       # Implement fallback strategy
   ```

### Validation Best Practices

1. **Interpret Scores in Context**:
   ```python
   result = hr.validate_data(data, ticker)
   
   if result.quality_score > 0.8:
       # High quality - proceed
       pass
   elif result.quality_score > 0.5:
       # Moderate quality - check warnings
       for warning in result.warnings:
           print(f"Warning: {warning}")
   else:
       # Low quality - investigate issues
       for issue in result.issues:
           print(f"Issue: {issue}")
   ```

2. **Use Recommendations**:
   ```python
   result = hr.validate_data(processed_data, ticker)
   
   for recommendation in result.recommendations:
       print(f"Recommendation: {recommendation}")
   # Example output:
   # "High kurtosis detected - consider fat-tailed distributions for modeling"
   # "Outliers detected - consider winsorization or robust modeling"
   ```

3. **Monitor Quality Over Time**:
   ```python
   # Track quality scores for ongoing monitoring
   quality_history = []
   for month_data in monthly_batches:
       result = validator.validate_data(month_data)
       quality_history.append({
           'date': month_data['date'].max(),
           'quality_score': result.quality_score,
           'issues_count': len(result.issues)
       })
   ```

### Preprocessing Best Practices

1. **Choose Processing Based on Application**:
   ```python
   # For volatility modeling
   vol_config = PreprocessingConfig(
       calculate_volatility=True,
       volatility_window=20,
       apply_smoothing=False      # Keep raw volatility
   )
   
   # For trend following
   trend_config = PreprocessingConfig(
       apply_smoothing=True,
       smoothing_window=10,
       calculate_volatility=False
   )
   ```

2. **Validate After Processing**:
   ```python
   # Processing can introduce edge effects
   processed_data = preprocessor.process_data(raw_data)
   
   # Use lenient validation for processed data
   lenient_config = ValidationConfig(max_consecutive_missing=10)
   result = DataValidator(lenient_config).validate_data(processed_data)
   ```

3. **Handle Multi-Asset Alignment**:
   ```python
   # For portfolio analysis, ensure proper alignment
   processor = DataPreprocessor(
       preprocessing_config=PreprocessingConfig(align_timestamps=True)
   )
   
   aligned_data = processor.process_multiple_series(multi_stock_data)
   # All assets now have consistent timestamps
   ```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Low Quality Scores on Processed Data

**Symptom**: Quality scores drop significantly after preprocessing (e.g., from 0.9 to 0.4)

**Cause**: Feature engineering creates boundary effects and NaN values

**Solution**:
```python
# Use lenient validation config for processed data
lenient_config = ValidationConfig(
    max_consecutive_missing=10,        # Allow more missing at boundaries
    outlier_threshold=3.0              # Less sensitive outlier detection
)

validator = DataValidator(lenient_config)
result = validator.validate_data(processed_data)
```

#### Issue: "Insufficient data" Errors

**Symptom**: `DataLoadError: Insufficient data for TICKER: X < 30`

**Cause**: Not enough business days in the requested period

**Solutions**:
```python
# Option 1: Extend the date range
data = loader.load_stock_data('AAPL', '2024-01-01', '2024-03-31')  # 3 months

# Option 2: Lower minimum observations requirement
config = DataConfig(min_observations=10)
loader = DataLoader(config)
data = loader.load_stock_data('AAPL', '2024-01-01', '2024-01-31')
```

#### Issue: Cache-Related Problems

**Symptom**: Stale data or unexpected cache behavior

**Solutions**:
```python
# Option 1: Clear cache manually
loader.clear_cache()

# Option 2: Disable caching temporarily
config = DataConfig(cache_enabled=False)
loader = DataLoader(config)

# Option 3: Reduce cache expiry
config = DataConfig(cache_expiry_hours=1)  # 1 hour instead of 24
```

#### Issue: Rate Limiting Errors

**Symptom**: API errors or timeouts during bulk loading

**Solutions**:
```python
# Reduce request rate
config = DataConfig(
    requests_per_minute=30,        # Half the default rate
    retry_attempts=5,              # More retry attempts
    retry_delay_seconds=2.0        # Longer delays
)

# Add delays between bulk operations
import time
for ticker in tickers:
    data = loader.load_stock_data(ticker, start_date, end_date)
    time.sleep(1)  # Additional delay
```

#### Issue: Validation Failures on Good Data

**Symptom**: Data appears clean but fails validation

**Investigation Steps**:
```python
# Get detailed metrics
result = validator.validate_data(data, ticker)
print("Issues:", result.issues)
print("Warnings:", result.warnings)  
print("Metrics:", result.metrics)

# Check specific problems
if 'price_min' in result.metrics:
    print(f"Price range: ${result.metrics['price_min']:.2f} - ${result.metrics['price_max']:.2f}")

if 'missing_percentage' in result.metrics:
    print(f"Missing data: {result.metrics['missing_percentage']:.2%}")
```

#### Issue: Memory Issues with Large Datasets

**Symptom**: Out of memory errors during processing

**Solutions**:
```python
# Process in chunks
def process_large_dataset(data, chunk_size=1000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        processed_chunk = preprocessor.process_data(chunk)
        results.append(processed_chunk)
    return pd.concat(results, ignore_index=True)

# Disable caching for large operations
config = DataConfig(cache_enabled=False)
```

### Edge Cases

#### Single Row Datasets
```python
# System handles gracefully but with warnings
single_row = data.iloc[:1]
result = validator.validate_data(single_row)
# Expect warnings about insufficient data for statistical analysis
```

#### Weekend/Holiday Data
```python
# Cryptocurrency or forex data with weekend trading
# May trigger date coverage warnings - this is expected
result = validator.validate_data(crypto_data)
# Check warnings for "unusual trading pattern" messages
```

#### Stock Splits and Dividends
```python
# Large price changes may trigger warnings
# Review warnings for "large price changes" messages
# These may be legitimate corporate actions, not data errors
```

## API Reference

### Quick Start Functions

```python
import hidden_regime as hr

# Load data
data = hr.load_stock_data('AAPL', '2024-01-01', '2024-12-31')

# Validate data  
result = hr.validate_data(data, 'AAPL')
print(f"Quality Score: {result.quality_score:.2f}")
```

### DataLoader API

```python
from hidden_regime import DataLoader, DataConfig

# Initialize with custom config
config = DataConfig(use_ohlc_average=False, cache_enabled=True)
loader = DataLoader(config)

# Load single stock
data = loader.load_stock_data(
    ticker='AAPL',
    start_date='2024-01-01',  # String or datetime
    end_date='2024-12-31',
    use_ohlc_avg=None,        # Override config setting
    source='yfinance'         # Override default source
)

# Load multiple stocks
data_dict = loader.load_multiple_stocks(
    tickers=['AAPL', 'GOOGL', 'MSFT'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Cache management
stats = loader.get_cache_stats()  # Returns cache statistics
loader.clear_cache()              # Clear all cached data
```

### DataValidator API

```python
from hidden_regime import DataValidator, ValidationConfig

# Initialize with custom config
config = ValidationConfig(outlier_method='zscore', outlier_threshold=2.5)
validator = DataValidator(config)

# Validate data
result = validator.validate_data(data, ticker='AAPL')

# Access results
print(f"Valid: {result.is_valid}")
print(f"Score: {result.quality_score:.2f}")
print(f"Issues: {len(result.issues)}")
print(f"Warnings: {len(result.warnings)}")

# Detailed metrics
for key, value in result.metrics.items():
    print(f"{key}: {value}")

# Utility functions
is_valid_ticker = validator.validate_ticker_format('AAPL')
is_valid_range, issues = validator.validate_date_range('2024-01-01', '2024-12-31')
```

### DataPreprocessor API

```python
from hidden_regime import DataPreprocessor, PreprocessingConfig, ValidationConfig

# Initialize with config
preprocessing_config = PreprocessingConfig(
    calculate_volatility=True,
    apply_smoothing=True,
    return_method='log'
)
validation_config = ValidationConfig(outlier_method='iqr')

preprocessor = DataPreprocessor(
    preprocessing_config=preprocessing_config,
    validation_config=validation_config
)

# Process single dataset
processed_data = preprocessor.process_data(data, ticker='AAPL')

# Process multiple datasets
processed_dict = preprocessor.process_multiple_series({
    'AAPL': apple_data,
    'GOOGL': google_data
})

# Get data summary
summary = preprocessor.get_data_summary(processed_data)
print(f"Observations: {summary['n_observations']}")
print(f"Date range: {summary['date_range']['days']} days")
```

### ValidationResult Structure

```python
@dataclass
class ValidationResult:
    is_valid: bool                    # True if no critical issues
    issues: List[str]                 # Critical problems requiring attention
    warnings: List[str]               # Minor issues affecting quality score
    recommendations: List[str]        # Suggested actions for improvement
    quality_score: float              # Overall quality score (0.0-1.0)
    metrics: Dict[str, Any]           # Detailed quantitative metrics
```

### Exception Handling

```python
from hidden_regime import DataLoadError, ValidationError, DataQualityError

try:
    data = hr.load_stock_data('INVALID_TICKER', '2024-01-01', '2024-12-31')
except DataLoadError as e:
    print(f"Data loading failed: {e}")

try:
    result = hr.validate_data(empty_dataframe)
except ValidationError as e:
    print(f"Validation error: {e}")

try:
    processed = preprocessor.process_data(corrupted_data)
except DataQualityError as e:
    print(f"Data quality issue: {e}")
```

---

## Summary

The Hidden Regime data pipeline provides enterprise-grade data management for quantitative finance applications. Its comprehensive validation system, flexible configuration, and robust error handling make it suitable for both research and production trading systems.

Key strengths:
- **Reliability**: Extensive error handling and validation
- **Flexibility**: Configurable for diverse use cases  
- **Transparency**: Detailed quality scoring and metrics
- **Performance**: Efficient caching and processing
- **Extensibility**: Plugin architecture for future enhancements

For additional support or advanced configuration needs, refer to the test suite in `/tests/` for comprehensive usage examples.