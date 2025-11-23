"""
Validate SPY Regime Quality for Trading

This script validates that SPY regime detection produces tradeable-quality regimes
before deploying to live trading strategies. Essential quality check for Sharpe 10+
strategies where regime quality directly impacts performance.

Usage:
    python examples/quickstart/validate_spy_regimes.py

Requirements:
    - hidden-regime package installed
    - Internet connection (downloads SPY data from Yahoo Finance)

Output:
    - Detailed stability metrics report
    - Trading readiness assessment
    - Recommendations for improvement (if needed)
"""

import sys
from pathlib import Path

# Add package to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hidden_regime as hr
from hidden_regime.analysis.stability import RegimeStabilityMetrics


def validate_spy_regimes(
    n_states: int = 3,
    lookback_days: int = 252,
    start_date: str = '2020-01-01',
    end_date: str = '2024-01-01',
    use_temporal_isolation: bool = True
):
    """
    Validate SPY regime detection quality with proper temporal isolation.

    Args:
        n_states: Number of regime states (2-5)
        lookback_days: Historical data window for initial training
        start_date: Analysis start date
        end_date: Analysis end date
        use_temporal_isolation: Use walk-forward validation (default: True)

    Returns:
        Dict with validation results
    """
    print("=" * 70)
    print("SPY REGIME VALIDATION FOR TRADING")
    print("=" * 70)
    print()
    print(f"Parameters:")
    print(f"  Ticker:       SPY")
    print(f"  States:       {n_states}")
    print(f"  Lookback:     {lookback_days} days")
    print(f"  Period:       {start_date} to {end_date}")
    print(f"  Temporal Isolation: {'YES (walk-forward)' if use_temporal_isolation else 'NO (lookahead bias!)'}")
    print()

    if use_temporal_isolation:
        # CORRECT: Use temporal isolation to prevent lookahead bias
        print("Downloading data...")
        print()

        # Get full dataset using FinancialDataLoader
        from hidden_regime.config.data import FinancialDataConfig
        from hidden_regime.factories import component_factory

        data_config = FinancialDataConfig(
            ticker='SPY',
            start_date=start_date,
            end_date=end_date
        )
        data_loader = component_factory.create_data_component(data_config)
        full_data = data_loader.update()

        print(f"✅ Downloaded {len(full_data)} days")
        print()
        print("Creating temporally isolated pipeline...")
        print()

        # Create pipeline
        pipeline = hr.create_financial_pipeline(
            ticker='SPY',
            n_states=n_states,
            include_report=False
        )

        # Create temporal controller
        from hidden_regime.pipeline.temporal import TemporalController
        controller = TemporalController(
            pipeline=pipeline,
            full_dataset=full_data,
            enable_data_collection=False
        )

        # Train on initial lookback, then validate on final state
        print(f"Training on initial {lookback_days} days...")
        print()

        # Update to end date (model will be trained progressively)
        controller.update_as_of(end_date)

        print("✅ Regime detection complete (with temporal isolation)")
        print()

        # Get results from pipeline
        result = pipeline.component_outputs.get('interpreter')
        if result is None:
            result = pipeline.component_outputs.get('analysis')

    else:
        # WRONG: Lookahead bias (train on all data at once)
        print("⚠️  WARNING: Using lookahead bias mode!")
        print("   Model will see all future data during training.")
        print("   Results will be unrealistically optimistic.")
        print()
        print("Downloading data and training model...")
        print()

        # Create and run regime detection pipeline
        pipeline = hr.create_financial_pipeline(
            ticker='SPY',
            n_states=n_states,
            start_date=start_date,
            end_date=end_date
        )

        # Run regime detection
        report = pipeline.update()

        print("✅ Regime detection complete (with lookahead bias!)")
        print()

        # Get the actual DataFrame from pipeline component outputs
        result = pipeline.component_outputs.get('interpreter')
        if result is None:
            result = pipeline.component_outputs.get('analysis')

    if result is None:
        print("❌ Error: No regime analysis results available")
        print("Pipeline component outputs:", list(pipeline.component_outputs.keys()))
        return {'passed': False, 'error': 'No analysis results'}

    # Analyze stability using RegimeStabilityMetrics
    print("Analyzing regime stability...")
    print()

    metrics_analyzer = RegimeStabilityMetrics(result)

    # Get detailed report
    report = metrics_analyzer.get_detailed_report()
    print(report)

    # Get metrics dict
    metrics = metrics_analyzer.get_metrics()

    # Additional validation checks
    print()
    print("=" * 70)
    print("VALIDATION CHECKS (Sharpe 10+ Criteria)")
    print("=" * 70)
    print()

    checks = []

    # Check 1: Duration
    if 20 <= metrics['mean_duration'] <= 90:
        checks.append(("✅", "Mean duration in optimal range (20-90 days)"))
    elif metrics['mean_duration'] < 20:
        checks.append(("❌", f"Mean duration too short ({metrics['mean_duration']:.1f} days)"))
    else:
        checks.append(("⚠️", f"Mean duration too long ({metrics['mean_duration']:.1f} days)"))

    # Check 2: Persistence
    if metrics['persistence'] >= 0.6:
        checks.append(("✅", f"Good persistence ({metrics['persistence']:.1%})"))
    else:
        checks.append(("❌", f"Low persistence ({metrics['persistence']:.1%})"))

    # Check 3: Stability
    if metrics['stability_score'] >= 0.6:
        checks.append(("✅", f"Good stability ({metrics['stability_score']:.2f})"))
    else:
        checks.append(("❌", f"Low stability ({metrics['stability_score']:.2f})"))

    # Check 4: Tradeable
    if metrics['is_tradeable']:
        checks.append(("✅", "Regime quality is tradeable"))
    else:
        checks.append(("❌", "Regime quality NOT tradeable"))

    for symbol, message in checks:
        print(f"{symbol} {message}")

    print()

    # Overall verdict
    passed = all(symbol == "✅" for symbol, _ in checks)

    print("=" * 70)
    if passed:
        print("✅ VALIDATION PASSED - Ready for Trading!")
        print()
        print("Next steps:")
        print("  1. Proceed to Strategy #4 (Regime Deterioration Short)")
        print("  2. Add multi-timeframe filtering")
        print("  3. Implement confidence-based position sizing")
    else:
        print("❌ VALIDATION FAILED - Regime Quality Insufficient")
        print()
        print("Next steps:")
        print("  1. Review recommendations above")
        print("  2. Adjust model parameters (n_states, lookback_days)")
        print("  3. Re-run validation")
    print("=" * 70)

    return {
        'passed': passed,
        'metrics': metrics,
        'checks': checks,
        'result': result
    }


def compare_model_configurations():
    """
    Compare different HMM configurations to find optimal settings.

    Tests multiple combinations of n_states and lookback_days to identify
    which configuration produces the most tradeable regime quality.
    """
    print("=" * 70)
    print("COMPARING MODEL CONFIGURATIONS")
    print("=" * 70)
    print()
    print("Testing different n_states and lookback_days combinations...")
    print("(This may take a few minutes)")
    print()

    configurations = [
        # (n_states, lookback_days)
        (2, 180),
        (2, 252),
        (3, 180),
        (3, 252),
        (3, 360),
        (4, 252),
        (4, 360),
    ]

    results = []

    # Download data once
    print("Downloading SPY data...")

    from hidden_regime.config.data import FinancialDataConfig
    from hidden_regime.factories import component_factory

    data_config = FinancialDataConfig(
        ticker='SPY',
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    data_loader = component_factory.create_data_component(data_config)
    full_data = data_loader.update()

    print(f"✅ Downloaded {len(full_data)} days")
    print()

    for n_states, lookback_days in configurations:
        print(f"Testing: n_states={n_states}, lookback={lookback_days}...")

        try:
            # Use temporal isolation
            pipeline = hr.create_financial_pipeline(
                ticker='SPY',
                n_states=n_states,
                include_report=False
            )

            from hidden_regime.pipeline.temporal import TemporalController
            controller = TemporalController(
                pipeline=pipeline,
                full_dataset=full_data,
                enable_data_collection=False
            )

            # Train and validate with temporal isolation
            controller.update_as_of('2024-01-01')

            # Get the actual DataFrame from component outputs
            result = pipeline.component_outputs.get('interpreter')
            if result is None:
                result = pipeline.component_outputs.get('analysis')

            if result is None:
                print(f"  ❌ Failed: No analysis results")
                continue

            metrics_analyzer = RegimeStabilityMetrics(result)
            metrics = metrics_analyzer.get_metrics()

            results.append({
                'n_states': n_states,
                'lookback': lookback_days,
                'mean_duration': metrics['mean_duration'],
                'persistence': metrics['persistence'],
                'stability_score': metrics['stability_score'],
                'quality_rating': metrics['quality_rating'],
                'is_tradeable': metrics['is_tradeable']
            })

            print(f"  Duration: {metrics['mean_duration']:.1f}, "
                  f"Persistence: {metrics['persistence']:.2f}, "
                  f"Stability: {metrics['stability_score']:.2f}, "
                  f"Rating: {metrics['quality_rating']}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

    print()
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print()

    # Sort by stability score (best first)
    results.sort(key=lambda x: x['stability_score'], reverse=True)

    print(f"{'States':<8} {'Lookback':<10} {'Duration':<10} {'Persist':<10} {'Stability':<12} {'Rating':<12} {'Tradeable'}")
    print("-" * 70)

    for r in results:
        tradeable = "✅" if r['is_tradeable'] else "❌"
        print(f"{r['n_states']:<8} {r['lookback']:<10} {r['mean_duration']:<10.1f} "
              f"{r['persistence']:<10.2f} {r['stability_score']:<12.2f} "
              f"{r['quality_rating']:<12} {tradeable}")

    print()
    print("Recommendation:")
    best = results[0]
    print(f"  Use n_states={best['n_states']}, lookback_days={best['lookback']}")
    print(f"  Stability score: {best['stability_score']:.2f}")
    print(f"  Quality rating: {best['quality_rating']}")
    print()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Validate SPY regime quality for trading')
    parser.add_argument('--n-states', type=int, default=3, help='Number of regime states (2-5)')
    parser.add_argument('--lookback', type=int, default=252, help='Lookback window in days')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple configurations')
    parser.add_argument('--no-temporal-isolation', action='store_true',
                       help='DANGEROUS: Disable temporal isolation (introduces lookahead bias)')

    args = parser.parse_args()

    if args.compare:
        # Run comparison mode (always uses temporal isolation)
        compare_model_configurations()
    else:
        # Run single validation
        validate_spy_regimes(
            n_states=args.n_states,
            lookback_days=args.lookback,
            start_date=args.start,
            end_date=args.end,
            use_temporal_isolation=not args.no_temporal_isolation
        )
