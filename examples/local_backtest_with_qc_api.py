#!/usr/bin/env python
"""
Local regime-based backtest using QuantConnect API.

This example shows how to:
1. Pull historical data from QuantConnect using your PRO API key
2. Run regime detection locally (no Docker needed)
3. Simulate portfolio rebalancing based on detected regimes
4. Export results for analysis

Prerequisites:
- QuantConnect PRO account (you get this when you sign up)
- API key and secret from https://www.quantconnect.com/account
- Set environment variables or create ~/.qc-credentials.json:
  {
    "api_key": "your-key",
    "api_secret": "your-secret"
  }

Usage:
    source /home/aoaustin/hidden-regime-pyenv/bin/activate
    python examples/local_backtest_with_qc_api.py
"""

import logging
import sys
from pathlib import Path

from hidden_regime.local.backtester import LocalBacktester
from hidden_regime.quantconnect.credentials import QCCredentials

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run example local backtest."""
    logger.info("Starting local regime-based backtest example...")

    # Step 1: Verify credentials are available
    logger.info("Checking QuantConnect credentials...")
    try:
        creds = QCCredentials()
        logger.info("QuantConnect credentials loaded successfully")
    except ValueError as e:
        logger.error(f"Credentials error: {e}")
        logger.error("\nTo fix this, do one of the following:")
        logger.error("1. Set environment variables:")
        logger.error("   export QC_API_KEY='your-key'")
        logger.error("   export QC_API_SECRET='your-secret'")
        logger.error("\n2. Or create ~/.qc-credentials.json with your API credentials")
        logger.error("   (Get credentials from https://www.quantconnect.com/account)")
        return 1

    # Step 2: Initialize backtester
    logger.info("Initializing backtester...")
    backtester = LocalBacktester(
        ticker="SPY",
        initial_cash=100000,
        cache_enabled=True,  # Cache data locally to avoid repeated API calls
    )

    # Step 3: Run backtest
    logger.info("Running backtest...")
    results = backtester.run(
        start_date="2020-01-01",
        end_date="2023-12-31",
        n_states=3,  # 3-state HMM: Bull, Sideways, Bear
        lookback_days=252,  # 1 year of training data
        allocations={
            "Bull": 1.0,  # 100% long in bull regime
            "Bear": 0.0,  # Cash in bear regime
            "Sideways": 0.5,  # 50% long in sideways regime
        },
    )

    # Step 4: Export results
    logger.info("Exporting results...")
    output_dir = backtester.export_results()
    logger.info(f"Results exported to: {output_dir}")

    # Step 5: Print summary
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Ticker: SPY")
    print(f"Period: 2020-01-01 to 2023-12-31")
    print(f"Initial Capital: ${100000:,.2f}")
    print(f"Final Value: ${results['metrics']['final_value']:,.2f}")
    print(f"Total Return: {results['metrics']['total_return']:.2%}")
    print(f"Annualized Return: {results['metrics']['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"Win Rate: {results['metrics']['win_rate']:.1%}")
    print(f"Total Trades: {results['metrics']['total_trades']}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)

    # Step 6: Show how to analyze results
    print("\nNext steps:")
    print("1. View results: head -20", output_dir / "results.csv")
    print("2. View trades: cat", output_dir / "trades.csv")
    print("3. View metrics: cat", output_dir / "metrics.json")
    print("\nTip: Results are cached locally, so re-running this script will use cached data.")
    print("     Use backtester.api_client.clear_cache() to refresh data from QC API.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception(f"Backtest failed with error: {e}")
        sys.exit(1)
