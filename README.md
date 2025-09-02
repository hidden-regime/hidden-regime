# Hidden Regime

**Market regime detection using Hidden Markov Models with Bayesian uncertainty quantification.**

🚀 **Production-ready data pipeline with comprehensive validation and preprocessing** 🚀

## Features

### ✅ Data Pipeline (Complete)
- **Multi-source data loading** with yfinance integration and extensible architecture
- **Comprehensive data validation** with 6-layer quality assessment system
- **Advanced preprocessing** with outlier detection, missing value handling, and feature engineering
- **Intelligent caching** with rate limiting and automatic retry logic
- **Quality scoring system** providing quantitative data quality assessment (0.0-1.0 scale)
- **Multi-asset support** with timestamp alignment and batch processing

### 🔄 Regime Detection (Coming Soon)
- **Real-time regime detection** using Online Hidden Markov Models
- **Bayesian uncertainty quantification** for parameter estimates  
- **Multi-asset regime analysis** and correlation detection
- **Model Context Protocol integration** for AI systems

### 📋 Planned Advanced Features
- Online HMM learning with streaming market data
- Regime-based trading signal generation
- Fat-tailed emission models for crisis detection
- Duration modeling for regime persistence analysis

## Quick Start

### Installation

```bash
pip install hidden-regime
```

### Basic Usage

```python
import hidden_regime as hr

# Load and validate stock data
data = hr.load_stock_data('AAPL', '2024-01-01', '2024-12-31')
validation_result = hr.validate_data(data, 'AAPL')

print(f"Data quality score: {validation_result.quality_score:.2f}")
print(f"Data is valid: {validation_result.is_valid}")

# Basic info
print(f"Loaded {len(data)} days of data")
print(f"Columns: {list(data.columns)}")
```

### Advanced Usage

```python
from hidden_regime import DataLoader, DataValidator, DataPreprocessor
from hidden_regime import DataConfig, ValidationConfig, PreprocessingConfig

# Custom configuration for strict validation
config = ValidationConfig(
    outlier_method='zscore',
    outlier_threshold=2.0,
    max_daily_return=0.1
)

# Load and process multiple stocks
loader = DataLoader()
validator = DataValidator(config)
preprocessor = DataPreprocessor()

# Multi-stock analysis
tickers = ['AAPL', 'GOOGL', 'MSFT']
data_dict = loader.load_multiple_stocks(tickers, '2024-01-01', '2024-12-31')

# Process and validate each stock
results = {}
for ticker, data in data_dict.items():
    processed_data = preprocessor.process_data(data, ticker)
    validation = validator.validate_data(processed_data, ticker)
    results[ticker] = {
        'data': processed_data,
        'quality_score': validation.quality_score,
        'is_valid': validation.is_valid
    }

for ticker, result in results.items():
    print(f"{ticker}: Quality {result['quality_score']:.2f}, Valid: {result['is_valid']}")
```

## Data Pipeline Features

### Data Loading
- **Multi-source support**: Currently yfinance, extensible for Alpha Vantage, custom APIs
- **Intelligent caching**: In-memory caching with configurable expiry (default 24h)  
- **Rate limiting**: Configurable requests per minute with exponential backoff
- **Error handling**: Comprehensive retry logic with graceful degradation

### Data Validation
- **6-layer validation system**: Structure, Date, Price, Return, Missing Data, Outlier Detection
- **Quality scoring**: Penalty-based scoring system (0.0-1.0 scale)
- **Configurable thresholds**: Customizable validation rules for different use cases
- **Detailed reporting**: Issues, warnings, recommendations, and quantitative metrics

### Data Preprocessing  
- **Missing value handling**: Linear interpolation, forward/backward fill strategies
- **Outlier treatment**: Winsorization using IQR, Z-score, or Isolation Forest methods
- **Feature engineering**: Volatility calculation, technical indicators, smoothing
- **Multi-asset alignment**: Timestamp synchronization for portfolio analysis

### Quality Score System

**Scoring Formula:**
```
score = 1.0 - (issues × 0.2) - (warnings × 0.05) - (missing_data_pct × 2.0) - (outlier_pct × 1.0)
```

The quality score (0.0-1.0) is calculated using a **penalty-based system** across **six validation layers**:

#### Validation Layers
1. **Structure Validation**: Required columns, data types, basic format checks
2. **Date Validation**: Date formats, chronological order, coverage gaps
3. **Price Validation**: Non-positive prices, extreme price changes, data types
4. **Return Validation**: Infinite returns, extreme daily returns, volatility checks
5. **Missing Data**: Overall missing percentage, column-specific gaps, consecutive missing values
6. **Outlier Detection**: Statistical outliers using IQR, Z-score, or Isolation Forest methods

#### Critical Issues (−0.2 points each)
- Missing required columns (`date`, `price`)
- Empty DataFrame or no valid data
- Invalid date formats or non-positive prices
- Excessive missing data (>10% of total data)
- Too many consecutive missing values (>5 by default)
- More than 20% missing data in any single column
- Date range too short (<7 days) or unrealistic

#### Warning Conditions (−0.05 points each)
- Dates not in chronological order
- Large gaps in time series (>14 days)
- Very high volatility (>10% daily) or very low (<0.5% daily)
- Moderate outliers detected (5-10% of data)
- High percentage of zero-volume trading days (>10%)
- Low date coverage (<80% of business days)
- Very low prices (below configured minimum)

#### Metric-Based Penalties
- **Missing Data**: Heavy penalty of −2.0 × percentage (e.g., 5% missing = −0.1 points)
- **Outliers**: Moderate penalty of −1.0 × percentage (e.g., 3% outliers = −0.03 points)
- **Extreme Returns**: Minor penalty of −0.1 per occurrence of returns exceeding threshold

#### Quality Score Interpretation
| Score | Quality | Description | Typical Characteristics |
|-------|---------|-------------|------------------------|
| 0.90-1.00 | **Excellent** | Ready for all applications | No issues, minimal warnings, <2% missing data |
| 0.70-0.89 | **Good** | Suitable for most analysis | 1-2 minor warnings, <5% missing data, good coverage |
| 0.50-0.69 | **Moderate** | Usable with preprocessing | Some issues or warnings, 5-10% missing data |
| 0.30-0.49 | **Poor** | Requires extensive cleaning | Multiple issues, >10% missing, significant outliers |
| 0.00-0.29 | **Very Poor** | Likely unusable | Major structural problems, consider alternative sources |

#### Practical Examples

**Perfect Quality (Score: 1.00)**
```
✓ Complete OHLC data with volume
✓ No missing values, proper chronological order
✓ Reasonable price movements and volatility
✓ No outliers or extreme returns detected
```

**Good Quality (Score: 0.85)**
```
✓ Complete required data
⚠ 1-2 days with large price gaps (holidays)
⚠ Moderate volatility during earnings periods
✓ <3% missing volume data (filled with median)
```

**Moderate Quality (Score: 0.60)**
```
✓ Core price data complete
⚠ Several outlier returns during volatile periods
⚠ 8% missing volume data
⚠ 3-day gap in data (market closure)
✗ 1 extreme return event exceeding 50% threshold
```

**Poor Quality (Score: 0.25)**
```
✗ 15% missing price data
✗ Multiple extreme returns and outliers
✗ Chronological order issues
✗ Very high volatility (15% daily average)
⚠ Low date coverage (60% of business days)
```

## Configuration

### Data Loading Configuration
```python
from hidden_regime import DataConfig

config = DataConfig(
    use_ohlc_average=True,           # Use OHLC average vs close price
    include_volume=True,             # Include volume data
    max_missing_data_pct=0.05,      # 5% max missing data tolerance
    min_observations=30,             # Minimum required data points
    cache_enabled=True,              # Enable caching
    requests_per_minute=60,          # API rate limiting
    retry_attempts=3                 # Retry failed requests
)
```

### Validation Configuration  
```python
from hidden_regime import ValidationConfig

config = ValidationConfig(
    outlier_method='iqr',            # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold=3.0,           # Z-score threshold  
    max_daily_return=0.5,            # 50% max daily return
    max_consecutive_missing=5,       # Max consecutive missing values
    interpolation_method='linear'    # Missing value interpolation
)
```

#### Configuration Impact on Quality Scoring

**Conservative Settings (Higher Quality Standards):**
```python
config = ValidationConfig(
    outlier_threshold=2.0,           # Stricter outlier detection
    max_daily_return=0.1,            # 10% max daily return (vs 50% default)
    max_consecutive_missing=3        # Fewer missing values allowed
)
# Result: Lower scores for same data, higher quality bar
```

**Lenient Settings (Lower Quality Standards):**
```python  
config = ValidationConfig(
    outlier_threshold=4.0,           # More tolerant of outliers
    max_daily_return=1.0,            # 100% max daily return allowed
    max_consecutive_missing=10       # More missing data tolerance
)
# Result: Higher scores for same data, lower quality bar
```

**Real-World Examples:**

| Setting | Conservative | Default | Lenient | Impact on Score |
|---------|-------------|---------|---------|-----------------|
| **Outlier Threshold** | 2.0 | 3.0 | 4.0 | ±0.03 per 3% outliers |
| **Max Daily Return** | 10% | 50% | 100% | ±0.05 per extreme return |
| **Max Missing** | 3 days | 5 days | 10 days | −0.2 if exceeded |
| **Outlier Method** | zscore | iqr | isolation_forest | Method-dependent detection |

**Recommended Settings by Use Case:**
- **HMM Training**: Conservative settings for clean training data
- **Exploratory Analysis**: Default settings for balanced validation
- **Crisis Period Data**: Lenient settings to handle extreme market events
- **High-Frequency Data**: Stricter missing data limits, moderate outlier tolerance

### Preprocessing Configuration
```python
from hidden_regime import PreprocessingConfig

config = PreprocessingConfig(
    return_method='log',             # 'log' or 'simple' returns
    calculate_volatility=True,       # Add volatility features
    volatility_window=20,            # Volatility calculation window
    apply_smoothing=False,           # Apply data smoothing
    align_timestamps=True            # Multi-asset timestamp alignment
)
```

## Examples

Run the comprehensive demo:
```bash
python examples/data_pipeline_demo.py
```

## Documentation

- **[Data Pipeline Guide](hidden_regime/data/README.md)**: Comprehensive documentation of the data pipeline
- **API Reference**: Complete API documentation with examples
- **Configuration Guide**: Detailed configuration options and use cases  
- **Troubleshooting**: Common issues and solutions

## Testing

The package includes comprehensive tests with 87 test cases covering:
- Data loading from multiple sources
- Validation across all quality dimensions  
- Preprocessing with various configurations
- Edge cases and error conditions

Run tests:
```bash
pytest tests/ -v
```

## Dependencies

- **pandas** >= 1.3.0: Data manipulation and analysis
- **numpy** >= 1.20.0: Numerical computing
- **scipy** >= 1.7.0: Scientific computing and statistics
- **yfinance** >= 0.2.0: Financial data provider
- **matplotlib** >= 3.4.0: Plotting and visualization

## Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

### Phase 1: Data Pipeline ✅ (Complete)
- Multi-source data loading
- Comprehensive validation framework  
- Advanced preprocessing capabilities
- Quality scoring and reporting

### Phase 2: Basic Regime Detection 🔄 (In Progress)
- Simple threshold-based regime classification
- 3-state Hidden Markov Model implementation
- Baum-Welch parameter estimation
- Viterbi algorithm for state inference

### Phase 3: Advanced Regime Detection 📋 (Planned)
- Online learning with streaming data updates
- Bayesian uncertainty quantification
- Model selection and complexity optimization
- Multi-asset regime correlation analysis

### Phase 4: AI Integration 🎯 (Future)
- Model Context Protocol (MCP) server implementation
- Integration with LLM-based financial analysis
- Real-time trading signal generation
- Risk management and portfolio optimization

## Support

- **Documentation**: Comprehensive guides and API reference
- **Examples**: Working demonstrations and tutorials
- **Issues**: Report bugs and request features on GitHub
- **Community**: Join discussions and share experiences

---

**Hidden Regime** - Transforming market regime detection through advanced statistical modeling and AI integration.
