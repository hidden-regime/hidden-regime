"""
Portfolio Analysis Example for hidden-regime package.

Demonstrates advanced multi-stock analysis including:
- Loading and validating multiple assets
- Cross-asset quality comparison
- Data preprocessing with alignment
- Portfolio-level metrics and insights
"""

import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the hidden-regime package
import hidden_regime as hr
from hidden_regime import DataConfig, ValidationConfig, PreprocessingConfig


def load_portfolio_data(tickers, start_date, end_date):
    """Load data for a portfolio of stocks."""
    print(f"Loading data for {len(tickers)} stocks from {start_date} to {end_date}...")

    # Configure for reliable loading
    data_config = DataConfig(
        use_ohlc_average=True,  # Use OHLC average for stability
        include_volume=True,  # Include volume for analysis
        max_missing_data_pct=0.1,  # Allow up to 10% missing data
        min_observations=50,  # Require at least 50 data points
        cache_enabled=True,  # Enable caching for efficiency
    )

    loader = hr.DataLoader(data_config)

    try:
        data_dict = loader.load_multiple_stocks(tickers, start_date, end_date)
        print(f"✓ Successfully loaded {len(data_dict)} out of {len(tickers)} stocks")

        # Show loading results
        for ticker in tickers:
            if ticker in data_dict:
                days = len(data_dict[ticker])
                print(f"  - {ticker}: {days} days of data")
            else:
                print(f"  - {ticker}: Failed to load")

        return data_dict

    except Exception as e:
        print(f"✗ Error loading portfolio data: {e}")
        return {}


def validate_portfolio_data(data_dict):
    """Validate data quality across the portfolio."""
    print(f"\nValidating data quality for {len(data_dict)} stocks...")

    # Use moderate validation settings suitable for portfolio analysis
    validation_config = ValidationConfig(
        outlier_method="iqr",
        outlier_threshold=2.5,  # Slightly more sensitive than default
        max_daily_return=0.3,  # Allow 30% daily returns
        max_consecutive_missing=3,  # Allow some missing data gaps
    )

    validator = hr.DataValidator(validation_config)
    validation_results = {}

    for ticker, data in data_dict.items():
        validation_results[ticker] = validator.validate_data(data, ticker)

    # Analyze validation results
    quality_scores = {
        ticker: result.quality_score for ticker, result in validation_results.items()
    }

    print(f"\nData Quality Summary:")
    print(f"  Average quality score: {np.mean(list(quality_scores.values())):.3f}")
    print(
        f"  Best quality: {max(quality_scores, key=quality_scores.get)} ({max(quality_scores.values()):.3f})"
    )
    print(
        f"  Worst quality: {min(quality_scores, key=quality_scores.get)} ({min(quality_scores.values()):.3f})"
    )

    # Show detailed results for each stock
    print(f"\nIndividual Stock Quality:")
    for ticker in sorted(quality_scores.keys()):
        result = validation_results[ticker]
        quality = result.quality_score
        valid = result.is_valid
        issues = len(result.issues)
        warnings_count = len(result.warnings)

        status = "✓" if valid else "⚠"
        print(
            f"  {status} {ticker:6s}: Quality {quality:.3f} | Issues: {issues} | Warnings: {warnings_count}"
        )

        # Show specific issues for low-quality stocks
        if quality < 0.7 or not valid:
            for issue in result.issues[:2]:  # Show first 2 issues
                print(f"    Issue: {issue}")
            for warning in result.warnings[:2]:  # Show first 2 warnings
                print(f"    Warning: {warning}")

    return validation_results


def process_portfolio_data(data_dict):
    """Process portfolio data with alignment and feature engineering."""
    print(f"\nProcessing portfolio data with feature engineering...")

    # Configure preprocessing for portfolio analysis
    preprocessing_config = PreprocessingConfig(
        return_method="log",  # Use log returns for analysis
        calculate_volatility=True,  # Add volatility features
        volatility_window=20,  # 20-day volatility window
        apply_smoothing=False,  # Keep raw data for regime detection
        align_timestamps=True,  # Critical for portfolio analysis
        fill_method="forward",  # Forward-fill missing data
    )

    # Use lenient validation for processed data
    validation_config = ValidationConfig(
        max_consecutive_missing=10,  # More lenient for processed data
        outlier_threshold=3.0,  # Standard outlier detection
    )

    preprocessor = hr.DataPreprocessor(
        preprocessing_config=preprocessing_config, validation_config=validation_config
    )

    try:
        processed_dict = preprocessor.process_multiple_series(data_dict)

        print(f"✓ Successfully processed {len(processed_dict)} stocks")

        # Analyze processing results
        for ticker, processed_data in processed_dict.items():
            original_data = data_dict[ticker]

            print(
                f"  - {ticker}: {len(original_data)} → {len(processed_data)} rows, "
                f"{len(processed_data.columns)} features"
            )

        return processed_dict

    except Exception as e:
        print(f"✗ Error processing portfolio data: {e}")
        return {}


def analyze_portfolio_correlations(processed_dict):
    """Analyze return correlations across the portfolio."""
    print(f"\nAnalyzing portfolio correlations...")

    if len(processed_dict) < 2:
        print("Need at least 2 stocks for correlation analysis")
        return

    # Extract returns for correlation analysis
    returns_df = pd.DataFrame()

    for ticker, data in processed_dict.items():
        if "log_return" in data.columns:
            returns_df[ticker] = data["log_return"]

    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()

    print(f"\nReturn Correlations:")
    print(correlation_matrix.round(3))

    # Find highest and lowest correlations
    correlations = []
    tickers = list(correlation_matrix.columns)

    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            if i < j:  # Avoid duplicates and self-correlation
                corr = correlation_matrix.loc[ticker1, ticker2]
                correlations.append((ticker1, ticker2, corr))

    # Sort by correlation strength
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\nStrongest correlations:")
    for ticker1, ticker2, corr in correlations[:3]:
        print(f"  {ticker1} - {ticker2}: {corr:.3f}")

    print(f"\nWeakest correlations:")
    for ticker1, ticker2, corr in correlations[-3:]:
        print(f"  {ticker1} - {ticker2}: {corr:.3f}")

    return correlation_matrix


def analyze_portfolio_volatility(processed_dict):
    """Analyze volatility patterns across the portfolio."""
    print(f"\nAnalyzing portfolio volatility patterns...")

    volatility_stats = {}

    for ticker, data in processed_dict.items():
        if "log_return" in data.columns:
            returns = data["log_return"].dropna()

            if len(returns) > 0:
                volatility_stats[ticker] = {
                    "daily_vol": returns.std(),
                    "annualized_vol": returns.std() * np.sqrt(252),
                    "avg_abs_return": returns.abs().mean(),
                    "max_daily_return": returns.max(),
                    "min_daily_return": returns.min(),
                    "return_skewness": returns.skew() if len(returns) > 2 else np.nan,
                    "return_kurtosis": (
                        returns.kurtosis() if len(returns) > 2 else np.nan
                    ),
                }

    # Display volatility analysis
    print(f"\nVolatility Analysis:")
    print(
        f"{'Ticker':<8} {'Daily Vol':<10} {'Annual Vol':<11} {'Max Return':<11} {'Min Return':<11} {'Skew':<7} {'Kurt':<7}"
    )
    print("-" * 70)

    for ticker in sorted(volatility_stats.keys()):
        stats = volatility_stats[ticker]
        print(
            f"{ticker:<8} {stats['daily_vol']:<10.4f} {stats['annualized_vol']:<11.1%} "
            f"{stats['max_daily_return']:<11.3f} {stats['min_daily_return']:<11.3f} "
            f"{stats['return_skewness']:<7.2f} {stats['return_kurtosis']:<7.2f}"
        )

    # Portfolio-level analysis
    all_returns = pd.concat(
        [data["log_return"].dropna() for data in processed_dict.values()],
        ignore_index=True,
    )

    if len(all_returns) > 0:
        portfolio_vol = all_returns.std()
        portfolio_annual_vol = portfolio_vol * np.sqrt(252)

        print(f"\nPortfolio-Level Statistics:")
        print(
            f"  Average daily volatility: {portfolio_vol:.4f} ({portfolio_annual_vol:.1%} annualized)"
        )
        print(f"  Return skewness: {all_returns.skew():.3f}")
        print(f"  Return kurtosis: {all_returns.kurtosis():.3f}")

        # Identify high/low volatility stocks
        daily_vols = {
            ticker: stats["daily_vol"] for ticker, stats in volatility_stats.items()
        }
        highest_vol = max(daily_vols, key=daily_vols.get)
        lowest_vol = min(daily_vols, key=daily_vols.get)

        print(f"  Highest volatility: {highest_vol} ({daily_vols[highest_vol]:.4f})")
        print(f"  Lowest volatility: {lowest_vol} ({daily_vols[lowest_vol]:.4f})")

    return volatility_stats


def generate_portfolio_insights(data_dict, processed_dict, validation_results):
    """Generate high-level portfolio insights and recommendations."""
    print(f"\nGenerating Portfolio Insights...")

    insights = []
    recommendations = []

    # Data quality insights
    quality_scores = [result.quality_score for result in validation_results.values()]
    avg_quality = np.mean(quality_scores)

    if avg_quality > 0.8:
        insights.append(
            f"✓ Excellent data quality across portfolio (avg: {avg_quality:.3f})"
        )
    elif avg_quality > 0.6:
        insights.append(
            f"⚠ Moderate data quality across portfolio (avg: {avg_quality:.3f})"
        )
        recommendations.append(
            "Consider data cleaning or filtering for low-quality stocks"
        )
    else:
        insights.append(
            f"✗ Poor data quality across portfolio (avg: {avg_quality:.3f})"
        )
        recommendations.append(
            "Significant data quality issues detected - review data sources"
        )

    # Portfolio size insights
    n_stocks = len(processed_dict)
    if n_stocks >= 10:
        insights.append(f"✓ Well-diversified portfolio with {n_stocks} stocks")
    elif n_stocks >= 5:
        insights.append(f"⚠ Moderately diversified portfolio with {n_stocks} stocks")
        recommendations.append("Consider adding more stocks for better diversification")
    else:
        insights.append(f"✗ Limited diversification with only {n_stocks} stocks")
        recommendations.append("Add more stocks to improve diversification")

    # Data coverage insights
    data_lengths = [len(data) for data in processed_dict.values()]
    if data_lengths:
        min_length = min(data_lengths)
        max_length = max(data_lengths)

        if min_length < 50:
            insights.append(
                f"⚠ Some stocks have limited data (minimum: {min_length} days)"
            )
            recommendations.append("Extend date range for more robust analysis")
        elif min_length > 200:
            insights.append(
                f"✓ Comprehensive data coverage (minimum: {min_length} days)"
            )

    # Volatility insights
    if len(processed_dict) > 1:
        returns_data = []
        for data in processed_dict.values():
            if "log_return" in data.columns:
                returns_data.extend(data["log_return"].dropna().tolist())

        if returns_data:
            portfolio_vol = np.std(returns_data) * np.sqrt(252)

            if portfolio_vol > 0.4:  # >40% annualized volatility
                insights.append(
                    f"⚠ High portfolio volatility ({portfolio_vol:.1%} annualized)"
                )
                recommendations.append(
                    "Consider risk management strategies for high-volatility portfolio"
                )
            elif portfolio_vol < 0.15:  # <15% annualized volatility
                insights.append(
                    f"✓ Low portfolio volatility ({portfolio_vol:.1%} annualized)"
                )
            else:
                insights.append(
                    f"✓ Moderate portfolio volatility ({portfolio_vol:.1%} annualized)"
                )

    # Display insights and recommendations
    print(f"\nKey Insights:")
    for insight in insights:
        print(f"  {insight}")

    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")

    print(f"\nNext Steps:")
    print(f"  • Use this processed data for regime detection analysis")
    print(f"  • Monitor data quality over time for any degradation")
    print(f"  • Consider expanding analysis with additional technical indicators")
    print(f"  • Implement portfolio optimization based on correlation patterns")

    return insights, recommendations


def main():
    """Run comprehensive portfolio analysis example."""
    print("Hidden Regime Portfolio Analysis Example")
    print("=" * 50)

    # Define portfolio - mix of different sectors and market caps
    portfolio_tickers = [
        "AAPL",  # Technology - Large Cap
        "GOOGL",  # Technology - Large Cap
        "MSFT",  # Technology - Large Cap
        "JPM",  # Financial - Large Cap
        "JNJ",  # Healthcare - Large Cap
        "XOM",  # Energy - Large Cap
        "TSLA",  # Consumer Discretionary - Large Cap
        "WMT",  # Consumer Staples - Large Cap
    ]

    # Use historical period with good data availability
    end_date = "2024-06-30"
    start_date = "2024-01-01"

    print(f"Analyzing portfolio of {len(portfolio_tickers)} stocks:")
    print(f"  Tickers: {', '.join(portfolio_tickers)}")
    print(f"  Period: {start_date} to {end_date}")

    try:
        # Step 1: Load portfolio data
        data_dict = load_portfolio_data(portfolio_tickers, start_date, end_date)

        if not data_dict:
            print("No data loaded - exiting analysis")
            return

        # Step 2: Validate data quality
        validation_results = validate_portfolio_data(data_dict)

        # Step 3: Process data with feature engineering
        processed_dict = process_portfolio_data(data_dict)

        if not processed_dict:
            print("No processed data available - exiting analysis")
            return

        # Step 4: Analyze correlations
        correlation_matrix = analyze_portfolio_correlations(processed_dict)

        # Step 5: Analyze volatility patterns
        volatility_stats = analyze_portfolio_volatility(processed_dict)

        # Step 6: Generate insights and recommendations
        insights, recommendations = generate_portfolio_insights(
            data_dict, processed_dict, validation_results
        )

        print(f"\n" + "=" * 50)
        print("Portfolio Analysis Complete!")
        print(
            "Use the processed data for regime detection and trading strategy development."
        )

    except Exception as e:
        print(f"✗ Error during portfolio analysis: {e}")
        print("This example requires yfinance and network connectivity.")
        print("Install with: pip install yfinance")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
