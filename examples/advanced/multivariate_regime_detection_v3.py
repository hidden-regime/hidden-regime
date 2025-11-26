"""
Multivariate Regime Detection Example - Production Version

Demonstrates multivariate HMM regime detection using the proper pipeline architecture.
This is the RECOMMENDED way to use multivariate HMM in production.

Key Features:
1. Uses pipeline factory for proper component composition
2. Returns + realized volatility (best practice feature combination)
3. Automatic feature standardization handled by the pipeline
4. Interpreter adds financial domain knowledge to raw model states

This example compares:
1. Univariate pipeline (returns only)
2. Multivariate pipeline (returns + realized volatility)

Key Insight: Volatility clusters with regime structure. High-vol periods indicate bear/crisis
regimes, low-vol periods indicate bull regimes. This correlation is regime-informative.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hidden_regime as hr


def main():
    """Main execution."""
    print("=" * 80)
    print("MULTIVARIATE REGIME DETECTION - PRODUCTION VERSION")
    print("=" * 80)

    # Configuration
    ticker = 'SPY'
    n_states = 3
    start_date = '2020-01-01'
    end_date = '2024-01-01'

    print(f"\nConfiguration:")
    print(f"  Ticker: {ticker}")
    print(f"  States: {n_states}")
    print(f"  Date range: {start_date} to {end_date}")

    # 1. Create univariate pipeline (returns only)
    print("\n" + "=" * 80)
    print("CREATING UNIVARIATE PIPELINE (Returns Only)")
    print("=" * 80)

    univariate_pipeline = hr.create_financial_pipeline(
        ticker=ticker,
        n_states=n_states,
        start_date=start_date,
        end_date=end_date,
        include_report=False,
        observation_config_overrides={
            'generators': ['log_return'],
            'normalize_features': False
        }
    )

    print("Running univariate pipeline...")
    univariate_result = univariate_pipeline.update()
    print("✓ Univariate pipeline completed")

    # Get model metrics
    uni_model = univariate_pipeline.model
    print(f"\nUnivariate Model Metrics:")
    print(f"  Converged: {uni_model.training_history_['converged']}")
    print(f"  Iterations: {uni_model.training_history_['iterations']}")
    print(f"  Final log-likelihood: {uni_model.training_history_['log_likelihoods'][-1]:.2f}")

    # Get predictions
    uni_predictions = univariate_pipeline.component_outputs['model']
    uni_transitions = np.sum(np.diff(uni_predictions['predicted_state']) != 0)
    print(f"  Regime transitions: {uni_transitions}")

    # 2. Create multivariate pipeline (returns + realized volatility)
    print("\n" + "=" * 80)
    print("CREATING MULTIVARIATE PIPELINE (Returns + Realized Volatility)")
    print("=" * 80)

    multivariate_pipeline = hr.create_multivariate_pipeline(
        ticker=ticker,
        n_states=n_states,
        features=['log_return', 'realized_vol'],
        start_date=start_date,
        end_date=end_date
    )

    print("Running multivariate pipeline...")
    multivariate_result = multivariate_pipeline.update()
    print("✓ Multivariate pipeline completed")

    # Get model metrics
    multi_model = multivariate_pipeline.model
    print(f"\nMultivariate Model Metrics:")
    print(f"  Converged: {multi_model.training_history_['converged']}")
    print(f"  Iterations: {multi_model.training_history_['iterations']}")
    print(f"  Final log-likelihood: {multi_model.training_history_['log_likelihoods'][-1]:.2f}")

    # Get predictions
    multi_predictions = multivariate_pipeline.component_outputs['model']
    multi_transitions = np.sum(np.diff(multi_predictions['predicted_state']) != 0)
    print(f"  Regime transitions: {multi_transitions}")

    # 3. Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    # Prediction agreement
    agreement = np.mean(
        uni_predictions['predicted_state'].values == multi_predictions['predicted_state'].values
    )
    print(f"\nPrediction agreement: {agreement:.1%}")

    # Confidence comparison
    uni_confidence = uni_predictions['confidence'].mean()
    multi_confidence = multi_predictions['confidence'].mean()
    print(f"Average confidence (univariate): {uni_confidence:.1%}")
    print(f"Average confidence (multivariate): {multi_confidence:.1%}")

    # Transition comparison
    print(f"\nRegime transitions:")
    print(f"  Univariate: {uni_transitions}")
    print(f"  Multivariate: {multi_transitions}")
    print(f"  Reduction: {(1 - multi_transitions/uni_transitions)*100:.1f}%")

    # Transaction cost analysis (assuming 0.5% per trade)
    cost_per_trade = 0.005
    trading_days = len(uni_predictions)
    annual_days = 252

    uni_annual_cost = (uni_transitions / trading_days) * annual_days * cost_per_trade * 100
    multi_annual_cost = (multi_transitions / trading_days) * annual_days * cost_per_trade * 100

    print(f"\nTransaction Cost Analysis (0.5% per regime change):")
    print(f"  Univariate annual cost: {uni_annual_cost:.2f}%")
    print(f"  Multivariate annual cost: {multi_annual_cost:.2f}%")
    print(f"  Annual savings: {uni_annual_cost - multi_annual_cost:.2f}%")

    # 4. Visualize comparison
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    # Get data for plotting
    data = univariate_pipeline.component_outputs['data']

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Price and regimes (univariate)
    ax1 = axes[0]
    ax1.plot(data.index, data['close'], color='black', alpha=0.3, linewidth=0.5)

    for state in range(n_states):
        mask = uni_predictions['predicted_state'] == state
        ax1.scatter(
            data.index[mask], data['close'][mask],
            c=f'C{state}', alpha=0.5, s=10, label=f'State {state}'
        )

    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{ticker} - Univariate HMM (Returns Only)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Price and regimes (multivariate)
    ax2 = axes[1]
    ax2.plot(data.index, data['close'], color='black', alpha=0.3, linewidth=0.5)

    for state in range(n_states):
        mask = multi_predictions['predicted_state'] == state
        ax2.scatter(
            data.index[mask], data['close'][mask],
            c=f'C{state}', alpha=0.5, s=10, label=f'State {state}'
        )

    ax2.set_ylabel('Price ($)')
    ax2.set_title(f'{ticker} - Multivariate HMM (Returns + Realized Vol)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Confidence comparison
    ax3 = axes[2]
    ax3.plot(uni_predictions.index, uni_predictions['confidence'], label='Univariate', alpha=0.7)
    ax3.plot(multi_predictions.index, multi_predictions['confidence'], label='Multivariate', alpha=0.7)
    ax3.set_ylabel('Confidence')
    ax3.set_title('Prediction Confidence Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Regime transition indicator
    ax4 = axes[3]
    uni_changes = np.diff(uni_predictions['predicted_state'], prepend=uni_predictions['predicted_state'].iloc[0]) != 0
    multi_changes = np.diff(multi_predictions['predicted_state'], prepend=multi_predictions['predicted_state'].iloc[0]) != 0

    ax4.scatter(uni_predictions.index[uni_changes], np.ones(np.sum(uni_changes)) * 1,
               marker='|', s=100, c='blue', alpha=0.5, label=f'Univariate ({uni_transitions} transitions)')
    ax4.scatter(multi_predictions.index[multi_changes], np.ones(np.sum(multi_changes)) * 0,
               marker='|', s=100, c='red', alpha=0.5, label=f'Multivariate ({multi_transitions} transitions)')

    ax4.set_ylabel('Model')
    ax4.set_xlabel('Date')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Multivariate', 'Univariate'])
    ax4.set_title('Regime Transitions (vertical lines = regime change)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('multivariate_regime_comparison_v3.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved plot to multivariate_regime_comparison_v3.png")

    # 5. Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. Multivariate HMMs (returns + realized_vol) provide more stable regime detection
2. Realized volatility is regime-informative: high vol → bear/crisis, low vol → bull
3. Feature standardization prevents numerical instability (handled automatically)
4. Fewer regime transitions = lower transaction costs in trading strategies
5. Pipeline architecture ensures proper component separation and V&V compliance

Production Recommendation:
- Use hr.create_multivariate_pipeline() for proper architecture alignment
- Default features ['log_return', 'realized_vol'] are best practice
- Automatic feature standardization prevents scale mismatch issues
- Interpreter adds financial domain knowledge (regime labels, characteristics)
    """)

    plt.show()


if __name__ == '__main__':
    main()
