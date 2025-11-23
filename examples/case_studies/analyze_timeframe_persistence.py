"""
Analyze persistence and duration metrics across different timeframes.

This script downloads crypto data at multiple timeframes (daily, hourly, 15-min, 1-min)
and fits HMMs to derive empirical thresholds for quality assessment.

Goal: Replace arbitrary thresholds (0.6, 2.0 days) with data-driven values that
account for different asset frequencies and volatility patterns.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hidden_regime.factories import component_factory
from hidden_regime.config.model import HMMConfig
from hidden_regime.data.loader import prepare_features


def download_crypto_data(symbol: str, interval: str, period: str = None, days_back: int = None):
    """
    Download crypto data at specified interval.

    Args:
        symbol: Crypto symbol (e.g., 'BTC-USD')
        interval: Data interval ('1d', '1h', '15m', '1m')
        period: Period string for yfinance ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        days_back: Alternative to period - number of days to go back
    """
    if days_back is not None:
        end = datetime.now()
        start = end - timedelta(days=days_back)
        data = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    else:
        data = yf.download(symbol, period=period, interval=interval, progress=False)

    return data


def analyze_timeframe(symbol: str, interval: str, period: str = None, days_back: int = None, n_states: int = 3):
    """
    Analyze persistence and duration for a single timeframe.

    Returns:
        Dict with persistence, duration, and other metrics
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {symbol} at {interval} interval")
    print(f"{'='*80}")

    # Download data
    try:
        if days_back is not None:
            raw_data = download_crypto_data(symbol, interval, days_back=days_back)
            print(f"Downloaded {len(raw_data)} observations ({days_back} days)")
        else:
            raw_data = download_crypto_data(symbol, interval, period=period)
            print(f"Downloaded {len(raw_data)} observations (period: {period})")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    if len(raw_data) < 100:
        print(f"Insufficient data: {len(raw_data)} observations")
        return None

    # Prepare features
    try:
        data = prepare_features(raw_data, price_method='close')
    except Exception as e:
        print(f"Error preparing features: {e}")
        return None

    # Create and fit HMM
    config = HMMConfig(
        n_states=n_states,
        observed_signal='log_return',
        max_iterations=100,
        tolerance=1e-6,
        initialization_method='kmeans'
    )

    # Use factory to create model component (recommended)
    model = component_factory.create_model_component(config)

    try:
        model.fit(data)
        print(f" Model fitted successfully")
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None

    # Get quality metrics
    try:
        metrics = model.get_quality_metrics(data)
    except Exception as e:
        print(f"Error getting metrics: {e}")
        return None

    # Extract key statistics
    avg_persistence = metrics['regime_persistence']['average']
    avg_duration = metrics['regime_durations']['average']
    ll_per_obs = metrics['log_likelihood']['per_observation']

    # Get per-state statistics
    persistence_values = list(metrics['regime_persistence']['by_regime'].values())
    duration_values = list(metrics['regime_durations']['by_regime'].values())

    # Compute frequency of observations per unit time
    if interval == '1d':
        obs_per_day = 1
    elif interval == '1h':
        obs_per_day = 24
    elif interval == '15m':
        obs_per_day = 96  # 24 * 4
    elif interval == '1m':
        obs_per_day = 1440  # 24 * 60
    else:
        obs_per_day = 1  # fallback

    # Convert duration from observation units to days
    avg_duration_days = avg_duration / obs_per_day
    duration_values_days = [d / obs_per_day for d in duration_values]

    results = {
        'symbol': symbol,
        'interval': interval,
        'n_observations': len(data),
        'obs_per_day': obs_per_day,

        # Core metrics
        'avg_persistence': avg_persistence,
        'avg_duration_obs': avg_duration,
        'avg_duration_days': avg_duration_days,
        'log_likelihood_per_obs': ll_per_obs,

        # Distributions
        'persistence_values': persistence_values,
        'persistence_min': min(persistence_values),
        'persistence_max': max(persistence_values),
        'persistence_std': np.std(persistence_values),

        'duration_values_obs': duration_values,
        'duration_values_days': duration_values_days,
        'duration_min_days': min(duration_values_days),
        'duration_max_days': max(duration_values_days),
        'duration_std_days': np.std(duration_values_days),

        # Quality assessment (using current thresholds)
        'quality_assessment': metrics['overall_assessment'],
        'quality_issues': metrics['quality_issues']
    }

    # Print summary
    print(f"\nResults:")
    print(f"  Average Persistence: {avg_persistence:.3f} ({avg_persistence:.1%})")
    print(f"  Average Duration: {avg_duration:.1f} obs = {avg_duration_days:.2f} days")
    print(f"  Log-Likelihood/obs: {ll_per_obs:.4f}")
    print(f"  Persistence range: [{min(persistence_values):.3f}, {max(persistence_values):.3f}]")
    print(f"  Duration range: [{min(duration_values_days):.2f}, {max(duration_values_days):.2f}] days")
    print(f"\n  Current Assessment: {metrics['overall_assessment']}")
    if metrics['quality_issues']:
        print(f"  Issues: {', '.join(metrics['quality_issues'])}")

    return results


def main():
    """Run analysis across multiple cryptocurrencies and timeframes."""

    print("EMPIRICAL ANALYSIS OF PERSISTENCE AND DURATION ACROSS TIMEFRAMES")
    print("="*80)
    print("\nObjective: Derive data-driven thresholds for quality assessment")
    print("that account for different observation frequencies.\n")

    # Define analysis matrix
    # Format: (symbol, interval, period/days_back)
    analyses = [
        # Daily data - longer history
        ('BTC-USD', '1d', {'period': '2y'}),
        ('ETH-USD', '1d', {'period': '2y'}),

        # Hourly data - ~90 days
        ('BTC-USD', '1h', {'days_back': 90}),
        ('ETH-USD', '1h', {'days_back': 90}),

        # 15-minute data - ~30 days (limited by yfinance)
        ('BTC-USD', '15m', {'days_back': 30}),
        ('ETH-USD', '15m', {'days_back': 30}),

        # 1-minute data - ~7 days (limited by yfinance)
        ('BTC-USD', '1m', {'days_back': 7}),
        ('ETH-USD', '1m', {'days_back': 7}),
    ]

    # Run analyses
    all_results = []

    for symbol, interval, kwargs in analyses:
        result = analyze_timeframe(symbol, interval, **kwargs)
        if result is not None:
            all_results.append(result)

    # Aggregate results by timeframe
    print("\n\n")
    print("="*80)
    print("AGGREGATE STATISTICS BY TIMEFRAME")
    print("="*80)

    df = pd.DataFrame(all_results)

    if len(df) == 0:
        print("No successful analyses")
        return

    # Group by interval
    for interval in df['interval'].unique():
        interval_data = df[df['interval'] == interval]

        print(f"\n{interval.upper()} DATA ({len(interval_data)} symbols)")
        print("-" * 80)

        # Aggregate statistics
        avg_pers = interval_data['avg_persistence'].mean()
        std_pers = interval_data['avg_persistence'].std()
        min_pers = interval_data['avg_persistence'].min()
        max_pers = interval_data['avg_persistence'].max()

        avg_dur_days = interval_data['avg_duration_days'].mean()
        std_dur_days = interval_data['avg_duration_days'].std()
        min_dur_days = interval_data['avg_duration_days'].min()
        max_dur_days = interval_data['avg_duration_days'].max()

        avg_ll = interval_data['log_likelihood_per_obs'].mean()

        print(f"  Persistence:")
        print(f"    Mean: {avg_pers:.3f} ± {std_pers:.3f}")
        print(f"    Range: [{min_pers:.3f}, {max_pers:.3f}]")

        print(f"  Duration (days):")
        print(f"    Mean: {avg_dur_days:.2f} ± {std_dur_days:.2f}")
        print(f"    Range: [{min_dur_days:.2f}, {max_dur_days:.2f}]")

        print(f"  Log-Likelihood/obs:")
        print(f"    Mean: {avg_ll:.4f}")

        # Compute percentiles for threshold suggestions
        all_pers = interval_data['avg_persistence'].values
        all_dur = interval_data['avg_duration_days'].values

        pers_p25 = np.percentile(all_pers, 25)
        pers_p50 = np.percentile(all_pers, 50)
        pers_p75 = np.percentile(all_pers, 75)

        dur_p25 = np.percentile(all_dur, 25)
        dur_p50 = np.percentile(all_dur, 50)
        dur_p75 = np.percentile(all_dur, 75)

        print(f"\n  Suggested Thresholds (based on percentiles):")
        print(f"    Persistence - Poor: < {pers_p25:.3f} | Acceptable: {pers_p25:.3f}-{pers_p75:.3f} | Good: > {pers_p75:.3f}")
        print(f"    Duration    - Poor: < {dur_p25:.2f} days | Acceptable: {dur_p25:.2f}-{dur_p75:.2f} days | Good: > {dur_p75:.2f} days")

    # Save results
    output_file = Path(__file__).parent.parent / 'output' / 'timeframe_persistence_analysis.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n\n Results saved to: {output_file}")

    # Summary recommendations
    print("\n\n")
    print("="*80)
    print("RECOMMENDATIONS FOR ADAPTIVE THRESHOLDS")
    print("="*80)
    print("""
Based on empirical analysis, we should implement adaptive thresholds that:

1. **Detect Timeframe Automatically**
   - Infer from data frequency (observations per day)
   - 1d: ~1 obs/day
   - 1h: ~24 obs/day
   - 15m: ~96 obs/day
   - 1m: ~1440 obs/day

2. **Use Timeframe-Specific Thresholds**
   - Replace fixed 0.6/0.8 persistence cutoffs with data-driven values
   - Account for natural increase in persistence at higher frequencies
   - Use duration in calendar days (not raw observations)

3. **Weighted Averaging**
   - Weight by regime frequency (not simple mean)
   - Rare states shouldn't distort overall assessment

4. **Asset-Type Awareness** (future enhancement)
   - Crypto typically more volatile than equities
   - Indices more stable than individual stocks
    """)


if __name__ == '__main__':
    main()
