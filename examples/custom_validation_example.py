"""
Custom Validation Example for hidden-regime package.

Demonstrates advanced validation configuration scenarios including:
- Custom validation rules for different asset types
- Configuration profiles for different trading strategies
- Edge case handling and validation customization
- Performance optimization for large datasets
"""

import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the hidden-regime package
import hidden_regime as hr
from hidden_regime import DataConfig, ValidationConfig, PreprocessingConfig


def create_crypto_validation_config():
    """Create validation configuration optimized for cryptocurrency data."""
    return ValidationConfig(
        # Crypto markets trade 24/7, so different patterns expected
        outlier_method="iqr",  # IQR works well for crypto volatility
        outlier_threshold=2.0,  # More sensitive due to higher volatility
        iqr_multiplier=2.0,  # Wider bounds for crypto outliers
        min_price=0.0001,  # Allow very low prices (altcoins)
        max_daily_return=2.0,  # Allow extreme daily returns (200%)
        min_trading_days_per_month=28,  # Nearly every day (24/7 trading)
        max_consecutive_missing=1,  # Very strict - continuous trading expected
        interpolation_method="linear",  # Linear interpolation for missing data
    )


def create_penny_stock_validation_config():
    """Create validation configuration for penny stocks and low-price securities."""
    return ValidationConfig(
        outlier_method="zscore",  # Z-score better for penny stocks
        outlier_threshold=3.5,  # More lenient outlier detection
        iqr_multiplier=2.5,  # Wider bounds for price manipulation
        min_price=0.001,  # Very low minimum prices allowed
        max_daily_return=1.0,  # Allow 100% daily returns
        min_trading_days_per_month=10,  # Lower liquidity expected
        max_consecutive_missing=5,  # More missing data tolerated
        interpolation_method="forward",  # Forward fill for sparse data
    )


def create_blue_chip_validation_config():
    """Create validation configuration for large-cap, stable stocks."""
    return ValidationConfig(
        outlier_method="zscore",  # Z-score good for stable stocks
        outlier_threshold=2.5,  # Moderate sensitivity
        iqr_multiplier=1.5,  # Standard IQR bounds
        min_price=1.0,  # Higher minimum price expectation
        max_daily_return=0.15,  # Stricter return limits (15%)
        min_trading_days_per_month=20,  # High liquidity expected
        max_consecutive_missing=2,  # Very strict missing data tolerance
        interpolation_method="linear",  # Linear interpolation preferred
    )


def create_etf_validation_config():
    """Create validation configuration for ETFs and index funds."""
    return ValidationConfig(
        outlier_method="iqr",  # IQR works well for diversified assets
        outlier_threshold=3.0,  # Standard sensitivity
        iqr_multiplier=1.5,  # Standard IQR bounds
        min_price=1.0,  # ETFs typically trade above $1
        max_daily_return=0.3,  # Allow moderate returns (30%)
        min_trading_days_per_month=20,  # High liquidity expected
        max_consecutive_missing=3,  # Moderate missing data tolerance
        interpolation_method="linear",  # Linear interpolation
    )


def create_conservative_trading_config():
    """Create validation configuration for conservative trading strategies."""
    return ValidationConfig(
        outlier_method="zscore",  # Precise outlier detection
        outlier_threshold=2.0,  # Very sensitive to outliers
        iqr_multiplier=1.0,  # Tight bounds
        min_price=0.01,  # Standard minimum price
        max_daily_return=0.1,  # Very strict return limits (10%)
        min_trading_days_per_month=18,  # High data quality expected
        max_consecutive_missing=1,  # Minimal missing data tolerance
        interpolation_method="linear",  # Precise interpolation
    )


def create_aggressive_trading_config():
    """Create validation configuration for aggressive trading strategies."""
    return ValidationConfig(
        outlier_method="iqr",  # Handle high volatility
        outlier_threshold=4.0,  # Less sensitive to outliers
        iqr_multiplier=3.0,  # Wide outlier bounds
        min_price=0.001,  # Allow low-priced stocks
        max_daily_return=5.0,  # Allow extreme returns
        min_trading_days_per_month=15,  # More flexible data requirements
        max_consecutive_missing=7,  # Tolerate more missing data
        interpolation_method="forward",  # Simple forward fill
    )


def demonstrate_asset_type_validation():
    """Demonstrate validation with different asset type configurations."""
    print("🏦 Asset Type Validation Demonstration")
    print("=" * 45)

    # Create different validation configs
    configs = {
        "Cryptocurrency": create_crypto_validation_config(),
        "Penny Stock": create_penny_stock_validation_config(),
        "Blue Chip": create_blue_chip_validation_config(),
        "ETF": create_etf_validation_config(),
    }

    # Create sample datasets mimicking different asset types
    datasets = {}

    # Crypto-like data (high volatility, 24/7 trading)
    crypto_dates = pd.date_range(
        "2024-01-01", "2024-01-31", freq="D"
    )  # Daily including weekends
    crypto_returns = np.random.normal(0.002, 0.08, len(crypto_dates))  # High volatility
    crypto_prices = 1000 * np.exp(np.cumsum(crypto_returns))

    datasets["Cryptocurrency"] = pd.DataFrame(
        {
            "date": crypto_dates,
            "price": crypto_prices,
            "log_return": crypto_returns,
            "volume": np.random.lognormal(15, 1, len(crypto_dates)),
        }
    )

    # Penny stock data (low prices, high volatility, sparse trading)
    penny_dates = pd.bdate_range("2024-01-01", "2024-01-31")
    # Add missing trading days
    missing_days = np.random.choice(
        len(penny_dates), size=len(penny_dates) // 4, replace=False
    )
    penny_trading_days = [
        date for i, date in enumerate(penny_dates) if i not in missing_days
    ]

    penny_returns = np.random.normal(
        0.001, 0.12, len(penny_trading_days)
    )  # Very high volatility
    penny_prices = 0.50 * np.exp(np.cumsum(penny_returns))  # Low starting price

    datasets["Penny Stock"] = pd.DataFrame(
        {
            "date": penny_trading_days,
            "price": penny_prices,
            "log_return": penny_returns,
            "volume": np.random.lognormal(
                10, 1.5, len(penny_trading_days)
            ),  # Lower volume
        }
    )

    # Blue chip data (stable, high liquidity)
    bluechip_dates = pd.bdate_range("2024-01-01", "2024-01-31")
    bluechip_returns = np.random.normal(
        0.0008, 0.015, len(bluechip_dates)
    )  # Low volatility
    bluechip_prices = 150 * np.exp(np.cumsum(bluechip_returns))  # High starting price

    datasets["Blue Chip"] = pd.DataFrame(
        {
            "date": bluechip_dates,
            "price": bluechip_prices,
            "log_return": bluechip_returns,
            "volume": np.random.lognormal(
                16, 0.3, len(bluechip_dates)
            ),  # High, stable volume
        }
    )

    # ETF data (moderate volatility, very stable)
    etf_dates = pd.bdate_range("2024-01-01", "2024-01-31")
    etf_returns = np.random.normal(0.0005, 0.018, len(etf_dates))  # Moderate volatility
    etf_prices = 100 * np.exp(np.cumsum(etf_returns))  # Standard starting price

    datasets["ETF"] = pd.DataFrame(
        {
            "date": etf_dates,
            "price": etf_prices,
            "log_return": etf_returns,
            "volume": np.random.lognormal(15.5, 0.4, len(etf_dates)),  # Moderate volume
        }
    )

    # Validate each dataset with its appropriate configuration
    results = {}

    for asset_type in configs.keys():
        config = configs[asset_type]
        dataset = datasets[asset_type]

        validator = hr.DataValidator(config)
        validation_result = validator.validate_data(dataset, asset_type.upper())

        results[asset_type] = validation_result

        print(f"\n💰 {asset_type} Validation Results:")
        print(f"  Quality Score: {validation_result.quality_score:.3f}")
        print(f"  Valid: {'✅' if validation_result.is_valid else '❌'}")
        print(f"  Issues: {len(validation_result.issues)}")
        print(f"  Warnings: {len(validation_result.warnings)}")
        print(f"  Data Points: {len(dataset)}")

        # Show configuration highlights
        print(f"  Config Highlights:")
        print(f"    Max Daily Return: {config.max_daily_return:.0%}")
        print(f"    Outlier Method: {config.outlier_method}")
        print(f"    Min Price: ${config.min_price:.4f}")

        # Show specific issues/warnings for this asset type
        if validation_result.issues:
            print(f"  Issues:")
            for issue in validation_result.issues[:2]:
                print(f"    • {issue}")

        if validation_result.warnings:
            print(f"  Warnings:")
            for warning in validation_result.warnings[:2]:
                print(f"    • {warning}")

    return results


def demonstrate_trading_strategy_validation():
    """Demonstrate validation configurations for different trading strategies."""
    print("\n📊 Trading Strategy Validation Demonstration")
    print("=" * 50)

    # Create trading strategy configs
    strategy_configs = {
        "Conservative": create_conservative_trading_config(),
        "Aggressive": create_aggressive_trading_config(),
    }

    # Create a test dataset with some quality issues
    dates = pd.bdate_range("2024-01-01", "2024-02-29")
    base_returns = np.random.normal(0.001, 0.025, len(dates))

    # Add some outliers and quality issues
    outlier_indices = np.random.choice(len(dates), size=5, replace=False)
    base_returns[outlier_indices] = np.random.choice(
        [-0.15, 0.18], size=5
    )  # ±15-18% returns

    prices = 100 * np.exp(np.cumsum(base_returns))

    # Add some missing data
    missing_indices = np.random.choice(len(dates), size=3, replace=False)
    prices[missing_indices] = np.nan

    test_dataset = pd.DataFrame(
        {
            "date": dates,
            "price": prices,
            "log_return": base_returns,
            "volume": np.random.lognormal(15, 0.5, len(dates)),
        }
    )

    print(f"Test dataset: {len(test_dataset)} days with intentional quality issues")
    print(f"  Added {len(outlier_indices)} outlier returns")
    print(f"  Added {len(missing_indices)} missing price values")

    # Validate with both strategy configurations
    strategy_results = {}

    for strategy_name, config in strategy_configs.items():
        validator = hr.DataValidator(config)
        validation_result = validator.validate_data(
            test_dataset, f"STRATEGY_{strategy_name.upper()}"
        )

        strategy_results[strategy_name] = validation_result

        print(f"\n🎯 {strategy_name} Strategy Validation:")
        print(f"  Quality Score: {validation_result.quality_score:.3f}")
        print(f"  Valid: {'✅' if validation_result.is_valid else '❌'}")
        print(f"  Issues: {len(validation_result.issues)}")
        print(f"  Warnings: {len(validation_result.warnings)}")

        # Strategy-specific insights
        print(f"  Strategy Insights:")
        if strategy_name == "Conservative":
            if validation_result.quality_score < 0.8:
                print(f"    ⚠️ Quality too low for conservative strategy")
                print(f"    💡 Consider more stringent data filtering")
            else:
                print(f"    ✅ Suitable for conservative trading approach")
        else:  # Aggressive
            if validation_result.quality_score < 0.4:
                print(f"    ⚠️ Even aggressive strategy needs minimum quality")
                print(f"    💡 Consider data source improvement")
            else:
                print(f"    ✅ Acceptable for aggressive trading approach")

        # Show differences in validation approach
        print(f"  Config Differences:")
        print(f"    Outlier Threshold: {config.outlier_threshold:.1f}")
        print(f"    Max Daily Return: {config.max_daily_return:.0%}")
        print(f"    Missing Data Tolerance: {config.max_consecutive_missing}")

    # Compare results
    print(f"\n🔄 Strategy Comparison:")
    conservative_score = strategy_results["Conservative"].quality_score
    aggressive_score = strategy_results["Aggressive"].quality_score

    score_difference = aggressive_score - conservative_score
    print(f"  Conservative Quality: {conservative_score:.3f}")
    print(f"  Aggressive Quality: {aggressive_score:.3f}")
    print(f"  Score Difference: {score_difference:+.3f}")

    if score_difference > 0.1:
        print(
            f"  💡 Insight: Conservative strategy rejects this data, aggressive accepts"
        )
    elif abs(score_difference) < 0.05:
        print(f"  💡 Insight: Both strategies agree on data quality assessment")

    return strategy_results


def demonstrate_custom_validation_rules():
    """Demonstrate creating completely custom validation rules."""
    print("\n⚙️ Custom Validation Rules Demonstration")
    print("=" * 45)

    # Create extreme custom configurations for specific use cases

    # Ultra-strict configuration for algorithmic trading
    algo_trading_config = ValidationConfig(
        outlier_method="zscore",
        outlier_threshold=1.5,  # Extremely sensitive
        iqr_multiplier=0.75,  # Very tight bounds
        min_price=5.0,  # High minimum price
        max_daily_return=0.05,  # Only 5% daily returns allowed
        min_trading_days_per_month=22,  # Nearly perfect trading days
        max_consecutive_missing=0,  # No missing data allowed
        interpolation_method="linear",
    )

    # Ultra-lenient configuration for research/backtesting
    research_config = ValidationConfig(
        outlier_method="iqr",
        outlier_threshold=10.0,  # Very insensitive
        iqr_multiplier=5.0,  # Extremely wide bounds
        min_price=0.0001,  # Allow any positive price
        max_daily_return=10.0,  # Allow 1000% returns
        min_trading_days_per_month=5,  # Very sparse data OK
        max_consecutive_missing=50,  # Allow lots of missing data
        interpolation_method="forward",
    )

    # Create problematic test data
    problematic_data = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", "2024-01-31"),
            "price": [
                100,
                102,
                np.nan,
                np.nan,
                98,
                200,
                95,
                97,
                np.nan,
                300,
                50,
                105,
                108,
                np.nan,
                400,
                90,
                85,
                np.nan,
                np.nan,
                110,
                115,
                120,
            ],  # Mix of missing values and extreme changes
            "log_return": [
                0.02,
                np.nan,
                np.nan,
                -0.02,
                0.5,
                -0.6,
                0.02,
                np.nan,
                2.0,
                -0.8,
                0.8,
                0.03,
                np.nan,
                3.0,
                -0.9,
                -0.05,
                np.nan,
                np.nan,
                0.27,
                0.04,
                0.041,
            ],  # Extreme returns and missing values
            "volume": np.random.lognormal(15, 0.5, 22),
        }
    )

    configs = {
        "Algorithmic Trading": algo_trading_config,
        "Research/Backtesting": research_config,
    }

    print(f"Problematic test dataset: {len(problematic_data)} days")
    print(f"  Missing prices: {problematic_data['price'].isna().sum()}")
    print(f"  Missing returns: {problematic_data['log_return'].isna().sum()}")
    print(f"  Max return: {problematic_data['log_return'].max():.1%}")
    print(f"  Min return: {problematic_data['log_return'].min():.1%}")

    for config_name, config in configs.items():
        validator = hr.DataValidator(config)
        validation_result = validator.validate_data(
            problematic_data, f"CUSTOM_{config_name.replace(' ', '_').upper()}"
        )

        print(f"\n🔧 {config_name} Configuration:")
        print(f"  Quality Score: {validation_result.quality_score:.3f}")
        print(f"  Valid: {'✅' if validation_result.is_valid else '❌'}")
        print(f"  Issues: {len(validation_result.issues)}")
        print(f"  Warnings: {len(validation_result.warnings)}")

        print(f"  Configuration Details:")
        print(f"    Outlier Threshold: {config.outlier_threshold:.1f}")
        print(f"    Max Return: {config.max_daily_return:.0%}")
        print(f"    Max Missing: {config.max_consecutive_missing}")
        print(f"    Min Price: ${config.min_price:.2f}")

        # Show key issues identified
        if validation_result.issues:
            print(f"  Key Issues Identified:")
            for issue in validation_result.issues[:3]:
                print(f"    ❌ {issue}")

        if validation_result.warnings:
            print(f"  Warnings:")
            for warning in validation_result.warnings[:3]:
                print(f"    ⚠️ {warning}")

        # Custom recommendations based on configuration
        if config_name == "Algorithmic Trading":
            if not validation_result.is_valid:
                print(
                    f"  🤖 Algo Trading Recommendation: REJECT - Data unsuitable for algorithmic trading"
                )
            else:
                print(
                    f"  🤖 Algo Trading Recommendation: ACCEPT - Data meets strict algorithmic requirements"
                )
        else:  # Research
            if validation_result.quality_score < 0.2:
                print(
                    f"  🔬 Research Recommendation: Data has significant issues but may be usable for research"
                )
            else:
                print(
                    f"  🔬 Research Recommendation: Data acceptable for backtesting and research"
                )


def main():
    """Run custom validation configuration examples."""
    print("Hidden Regime Custom Validation Configuration Examples")
    print("=" * 65)

    print("This example demonstrates how to customize validation for different:")
    print("• Asset types (crypto, penny stocks, blue chips, ETFs)")
    print("• Trading strategies (conservative vs aggressive)")
    print("• Use cases (algorithmic trading vs research)")
    print()

    try:
        # Demonstrate asset type-specific validation
        asset_results = demonstrate_asset_type_validation()

        # Demonstrate trading strategy-specific validation
        strategy_results = demonstrate_trading_strategy_validation()

        # Demonstrate completely custom validation rules
        demonstrate_custom_validation_rules()

        print(f"\n" + "=" * 65)
        print("Custom Validation Examples Complete! 🎉")

        print(f"\n💡 Key Takeaways:")
        print("• Different asset types need different validation approaches")
        print("• Trading strategy determines appropriate quality thresholds")
        print("• Custom configurations enable specialized use cases")
        print("• Quality scores should be interpreted in context")

        print(f"\n🔧 Implementation Tips:")
        print("• Start with default configs and adjust based on results")
        print("• Monitor quality over time to refine thresholds")
        print("• Document custom configurations for reproducibility")
        print("• Test configurations with known good/bad data samples")

    except Exception as e:
        print(f"✗ Error during custom validation demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
