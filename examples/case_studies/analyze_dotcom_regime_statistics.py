"""
Analyze Dotcom 2000 Regime Statistics

Extracts regime duration statistics from case study outputs
and validates blog post claims against real data.

Usage:
    source $HOME/hidden-regime-pyenv/bin/activate
    python examples/analyze_dotcom_regime_statistics.py
"""

import pandas as pd
import numpy as np


def analyze_dotcom_regimes(csv_path='/mnt/c/Workspace/hidden-regime/examples/output/dotcom_study/regime_history.csv'):
    """Extract and analyze regime statistics from dotcom case study."""

    # Load data
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    tickers = ['QQQ', 'MSFT', 'INTC', 'CSCO', 'AMZN', 'SPY']
    regime_names_map = {0: 'Bearish', 1: 'Sideways', 2: 'Bullish'}

    print("=" * 80)
    print("DOTCOM 2000 REGIME STATISTICS ANALYSIS")
    print("=" * 80)
    print(f"\nData Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total Days: {len(df)}\n")

    # Key dates analysis
    key_dates = ['2000-03-10', '2000-10-09', '2002-10-09']
    date_labels = ['NASDAQ Peak', 'Market Trough', 'Bear Bottom']

    print("=" * 80)
    print("KEY DATE REGIME SNAPSHOTS")
    print("=" * 80)

    for date_str, label in zip(key_dates, date_labels):
        date_val = pd.to_datetime(date_str)
        row = df[df['date'] == date_val]
        if len(row) > 0:
            print(f"\n{date_str} - {label}:")
            for ticker in tickers:
                regime_num = row[f'{ticker}_regime'].values[0]
                regime_name = row[f'{ticker}_regime_name'].values[0]
                confidence = row[f'{ticker}_confidence'].values[0]
                price = row[f'{ticker}_price'].values[0]
                print(f"  {ticker:5} | Regime: {regime_name:8} ({regime_num}) | Conf: {confidence:.3f} | Price: ${price:8.2f}")

    # Regime duration statistics
    print("\n" + "=" * 80)
    print("REGIME DURATION AND DISTRIBUTION STATISTICS")
    print("=" * 80)

    for ticker in tickers:
        print(f"\n{ticker}:")
        print("-" * 60)

        regimes = df[f'{ticker}_regime'].values
        transitions = np.where(np.diff(regimes) != 0)[0] + 1
        transition_indices = [0] + list(transitions) + [len(regimes)]

        regime_stats = {0: {'durations': [], 'days': 0},
                       1: {'durations': [], 'days': 0},
                       2: {'durations': [], 'days': 0}}

        for i in range(len(transition_indices) - 1):
            start_idx = transition_indices[i]
            end_idx = transition_indices[i + 1]
            regime = regimes[start_idx]
            duration = end_idx - start_idx
            regime_stats[regime]['durations'].append(duration)
            regime_stats[regime]['days'] += duration

        # Print statistics
        for regime_num in [0, 1, 2]:
            stats = regime_stats[regime_num]
            regime_name = regime_names_map[regime_num]
            total_days = len(regimes)
            pct = (stats['days'] / total_days) * 100

            if stats['durations']:
                avg_duration = np.mean(stats['durations'])
                max_duration = np.max(stats['durations'])
                min_duration = np.min(stats['durations'])
                count = len(stats['durations'])
                print(f"  {regime_name:8}: {stats['days']:3} days ({pct:5.1f}%) | "
                      f"Avg: {avg_duration:6.1f} | Max: {max_duration:4} | Count: {count}")

    # Summary table for blog post
    print("\n" + "=" * 80)
    print("SUMMARY FOR BLOG POST")
    print("=" * 80)
    print("\nRegime Duration Comparison Table (Markdown format):\n")

    summary_data = {}
    for ticker in tickers:
        regimes = df[f'{ticker}_regime'].values
        transitions = np.where(np.diff(regimes) != 0)[0] + 1
        transition_indices = [0] + list(transitions) + [len(regimes)]

        regime_stats = {0: {'durations': []}, 1: {'durations': []}, 2: {'durations': []}}
        regime_days = {0: 0, 1: 0, 2: 0}

        for i in range(len(transition_indices) - 1):
            start_idx = transition_indices[i]
            end_idx = transition_indices[i + 1]
            regime = regimes[start_idx]
            duration = end_idx - start_idx
            regime_stats[regime]['durations'].append(duration)
            regime_days[regime] += duration

        summary_data[ticker] = {
            'bull_avg': int(np.mean(regime_stats[2]['durations'])) if regime_stats[2]['durations'] else 0,
            'bear_avg': int(np.mean(regime_stats[0]['durations'])) if regime_stats[0]['durations'] else 0,
            'crisis_avg': int(np.mean(regime_stats[1]['durations'])) if regime_stats[1]['durations'] else 0,
            'bear_pct': (regime_days[0] / len(regimes)) * 100 if regime_days[0] > 0 else 0,
        }

    print("| Stock | Bull Avg | Bear Avg | Crisis Avg | Bear %  |")
    print("|-------|----------|----------|-----------|---------|")
    for ticker in tickers:
        data = summary_data[ticker]
        print(f"| {ticker:5} | {data['bull_avg']:8} days | {data['bear_avg']:8} days | {data['crisis_avg']:10} days | {data['bear_pct']:5.1f}% |")


if __name__ == '__main__':
    analyze_dotcom_regimes()
