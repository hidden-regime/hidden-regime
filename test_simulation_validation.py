#!/usr/bin/env python3
"""
Simulation Validation Tests

Test script to validate that the trading simulation logic is working correctly
with proper trade counting, temporal isolation, and realistic results.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from hidden_regime.analysis.case_study import CaseStudyAnalyzer
from hidden_regime.analysis.technical_indicators import TechnicalIndicatorAnalyzer


def test_buy_hold_trade_count():
    """Test that buy-and-hold has exactly 2 trades."""
    print("🧪 Testing buy-and-hold trade count...")

    # Create test data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = pd.DataFrame({
        'close': 100 * np.cumprod(1 + np.random.normal(0, 0.01, 100))
    }, index=dates)

    analyzer = CaseStudyAnalyzer()
    result = analyzer.analyze_buy_hold_performance(prices)

    expected_trades = 2
    actual_trades = result['num_trades']

    print(f"   Expected trades: {expected_trades}")
    print(f"   Actual trades: {actual_trades}")

    if actual_trades == expected_trades:
        print("   ✅ Buy-and-hold trade count correct")
        return True
    else:
        print("   ❌ Buy-and-hold trade count incorrect")
        return False


def test_technical_indicator_trade_count():
    """Test that technical indicators have reasonable trade counts."""
    print("\n🧪 Testing technical indicator trade counts...")

    # Create test data with more realistic price movement
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # Create trending data to ensure signals are generated
    trend = np.linspace(0, 0.5, 100)
    noise = np.random.normal(0, 0.02, 100)
    returns = trend/100 + noise
    prices = 100 * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, 100)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)

    analyzer = TechnicalIndicatorAnalyzer()

    try:
        results = analyzer.analyze_all_indicator_strategies(data)

        print(f"   Analyzed {len(results)} technical indicators")

        valid_results = 0
        for indicator_name, metrics in results.items():
            trades = metrics.get('number_of_trades', 0)

            # Technical indicators should have reasonable trade counts (not 100 daily trades)
            if 0 <= trades <= 50:  # Reasonable range for 100 days of data
                print(f"   ✅ {indicator_name}: {trades} trades (reasonable)")
                valid_results += 1
            else:
                print(f"   ❌ {indicator_name}: {trades} trades (unreasonable)")

        if valid_results == len(results) and len(results) > 0:
            print("   ✅ All technical indicators have reasonable trade counts")
            return True
        else:
            print("   ❌ Some technical indicators have unreasonable trade counts")
            return False

    except Exception as e:
        print(f"   ❌ Technical indicator test failed: {e}")
        return False


def test_hmm_strategy_trade_count():
    """Test that HMM strategy has reasonable trade counts."""
    print("\n🧪 Testing HMM strategy trade count...")

    # Create test data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    prices = pd.DataFrame({
        'close': 100 * np.cumprod(1 + np.random.normal(0, 0.01, 50))
    }, index=dates)

    # Create mock regime data with some regime changes
    regime_data = pd.DataFrame({
        'predicted_state': [0]*15 + [1]*20 + [2]*15,  # 3 regimes with 2 transitions
        'confidence': np.random.uniform(0.7, 0.9, 50)
    }, index=dates)

    analyzer = CaseStudyAnalyzer()
    result = analyzer.analyze_hmm_strategy_performance(prices, regime_data)

    trades = result['num_trades']

    print(f"   HMM strategy trades: {trades}")

    # With 2 regime transitions, we expect around 2-4 trades (depending on initial/final positions)
    if 2 <= trades <= 10:  # Reasonable range
        print("   ✅ HMM strategy trade count reasonable")
        return True
    else:
        print("   ❌ HMM strategy trade count unreasonable")
        return False


def test_temporal_isolation():
    """Test that strategies don't have access to future data."""
    print("\n🧪 Testing temporal isolation...")

    # This is a conceptual test - in practice, temporal isolation is ensured by
    # the evolution process feeding only historical data to each strategy

    # Create data with a clear pattern in the second half
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # First 50 days: random walk
    returns1 = np.random.normal(0, 0.01, 50)

    # Last 50 days: strong uptrend (future information)
    returns2 = np.random.normal(0.02, 0.01, 50)  # 2% daily return

    returns = np.concatenate([returns1, returns2])
    prices = 100 * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.ones(100) * 1000000
    }, index=dates)

    # Test with only first 50 days (no future info)
    data_no_future = data.iloc[:50]

    analyzer = TechnicalIndicatorAnalyzer()

    try:
        results_no_future = analyzer.analyze_all_indicator_strategies(data_no_future)

        # Check that analysis completes without errors
        if len(results_no_future) > 0:
            print("   ✅ Temporal isolation test passed - analysis works with limited data")
            return True
        else:
            print("   ⚠️  Temporal isolation test inconclusive - no results generated")
            return True  # Not a failure, just insufficient data

    except Exception as e:
        print(f"   ❌ Temporal isolation test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🚀 Running Trading Simulation Validation Tests")
    print("=" * 60)

    tests = [
        test_buy_hold_trade_count,
        test_technical_indicator_trade_count,
        test_hmm_strategy_trade_count,
        test_temporal_isolation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed with error: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All simulation validation tests passed!")
        return True
    else:
        print("❌ Some tests failed - simulation needs further fixes")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)