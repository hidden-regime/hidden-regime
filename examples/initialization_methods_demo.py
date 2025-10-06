"""
Demonstration of HMM Initialization Methods
============================================

Shows the three initialization approaches for Hidden Markov Models:
1. KMeans: Data-driven clustering (DEFAULT, recommended for most cases)
2. Random: Data-driven quantile-based initialization
3. Custom: User-specified parameters (for domain knowledge or transfer learning)

Also demonstrates:
- Transfer learning: Using parameters from one asset to initialize another
- Convenience factory: from_regime_specs() for ergonomic custom initialization
- Initialization diagnostics: Understanding what each method does

Requirements:
    pip install hidden-regime yfinance pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from hidden_regime.config.model import HMMConfig
from hidden_regime.models.hmm import HiddenMarkovModel


def download_data(ticker: str, days: int = 504) -> pd.DataFrame:
    """Download historical data and compute log returns."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Download with auto_adjust to avoid MultiIndex columns
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    # Compute log returns - squeeze to ensure 1D
    close_prices = data['Close'].squeeze()
    log_returns = np.log(close_prices / close_prices.shift(1))

    # Create clean DataFrame with proper column name
    result = pd.DataFrame({'log_return': log_returns}, index=data.index)
    return result.dropna()


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")


def print_diagnostics(diagnostics: dict, method_name: str):
    """Print initialization diagnostics in readable format."""
    print(f"\n{method_name} Initialization Diagnostics:")
    print(f"  Method: {diagnostics['method']}")
    print(f"  Number of States: {diagnostics['n_states']}")

    # Only custom initialization has regime_characteristics
    if 'regime_characteristics' in diagnostics:
        print("\n  Regime Characteristics:")
        for i, regime in enumerate(diagnostics['regime_characteristics']):
            print(f"\n    State {i}:")
            print(f"      Mean Return: {regime['mean_return_pct']:.2f}% daily")
            print(f"      Volatility: {regime['volatility_pct']:.2f}% daily")
            print(f"      Persistence: {regime['persistence']:.1%}")
            print(f"      Expected Duration: {regime['expected_duration_days']:.1f} days")

    # KMeans and Random methods have different diagnostic fields
    if 'kmeans_inertia' in diagnostics:
        print("\n  KMeans Clustering Quality:")
        print(f"    Silhouette Score: {diagnostics.get('silhouette_score', 'N/A'):.3f}")
        print(f"    Cluster Sizes: {diagnostics.get('cluster_sizes', {})}")

    if diagnostics.get('initialization_warnings'):
        print(f"\n  Warnings: {len(diagnostics['initialization_warnings'])} warning(s)")
        for warning in diagnostics['initialization_warnings']:
            print(f"    • {warning}")


# =============================================================================
# Example 1: KMeans Initialization (DEFAULT - Recommended)
# =============================================================================
print_section("Example 1: KMeans Initialization (Data-Driven Clustering)")

print("KMeans initialization is the DEFAULT and RECOMMENDED approach.")
print("It automatically clusters historical returns to find optimal regime centers.\n")

# Download SPY data
spy_data = download_data('SPY', days=504)  # ~2 years

# Create model with KMeans initialization (default)
kmeans_config = HMMConfig(
    n_states=3,
    initialization_method='kmeans',  # This is the default
    max_iterations=50,
    random_seed=42,
)

print(f"Training on {len(spy_data)} days of SPY data...")
hmm_kmeans = HiddenMarkovModel(kmeans_config)
hmm_kmeans.fit(spy_data)

# Show diagnostics
diagnostics_kmeans = hmm_kmeans.get_initialization_diagnostics()
print_diagnostics(diagnostics_kmeans, "KMeans")

print("\nWhen to use KMeans:")
print("   Default choice for most applications")
print("   No prior knowledge about regime characteristics")
print("   Want data to automatically determine regime structure")
print("   Training on sufficient historical data (>200 observations)")


# =============================================================================
# Example 2: Random Initialization (Fallback)
# =============================================================================
print_section("Example 2: Random Initialization (Quantile-Based)")

print("Random initialization uses quantiles of the data to set initial parameters.")
print("This is the fallback method when KMeans isn't available.\n")

# Create model with random initialization
random_config = HMMConfig(
    n_states=3,
    initialization_method='random',
    max_iterations=50,
    random_seed=42,
)

print(f"Training on {len(spy_data)} days of SPY data...")
hmm_random = HiddenMarkovModel(random_config)
hmm_random.fit(spy_data)

# Show diagnostics
diagnostics_random = hmm_random.get_initialization_diagnostics()
print_diagnostics(diagnostics_random, "Random")

print("\nWhen to use Random:")
print("   Fallback when KMeans dependencies not available")
print("   Testing and debugging (simpler, more predictable)")
print("   Very small datasets where KMeans might struggle")
print("   Quick prototyping without sklearn dependency")


# =============================================================================
# Example 3: Custom Initialization (Domain Knowledge)
# =============================================================================
print_section("Example 3: Custom Initialization (Expert-Specified Parameters)")

print("Custom initialization allows you to specify exact starting parameters.")
print("Useful for incorporating domain knowledge or transfer learning.\n")

# Define regime specifications based on financial expertise
print("Defining 3 regimes based on typical market behavior:")
print("  Bear Market:   -1.5% daily return, 2.5% volatility")
print("  Sideways:       0.0% daily return, 1.5% volatility")
print("  Bull Market:   +1.2% daily return, 2.0% volatility\n")

custom_config = HMMConfig(
    n_states=3,
    initialization_method='custom',
    # Specify mean returns (in log space)
    custom_emission_means=[
        -0.015,  # Bear: -1.5% daily
        0.0,     # Sideways: neutral
        0.012,   # Bull: +1.2% daily
    ],
    # Specify volatilities
    custom_emission_stds=[
        0.025,   # Bear: high volatility (2.5%)
        0.015,   # Sideways: low volatility (1.5%)
        0.020,   # Bull: moderate volatility (2.0%)
    ],
    # Optional: Specify transition matrix (if not provided, defaults to 80% persistence)
    custom_transition_matrix=[
        [0.85, 0.10, 0.05],  # Bear → Bear (85%), Sideways (10%), Bull (5%)
        [0.15, 0.70, 0.15],  # Sideways → more transitions
        [0.05, 0.10, 0.85],  # Bull → Bull (85%), Sideways (10%), Bear (5%)
    ],
    # Optional: Specify initial probabilities (uniform if not provided)
    custom_initial_probs=[0.3, 0.4, 0.3],  # Start assuming sideways most likely
    max_iterations=50,
    random_seed=42,
)

print(f"Training on {len(spy_data)} days of SPY data with custom initialization...")
hmm_custom = HiddenMarkovModel(custom_config)
hmm_custom.fit(spy_data)

# Show diagnostics
diagnostics_custom = hmm_custom.get_initialization_diagnostics()
print_diagnostics(diagnostics_custom, "Custom")

print("\nIMPORTANT: Custom parameters are STARTING values, not final values!")
print("Baum-Welch training will update them based on actual data.")
print("\nFinal trained parameters:")
for i in range(3):
    print(f"  State {i}: μ={hmm_custom.emission_means_[i]:.4f}, σ={hmm_custom.emission_stds_[i]:.4f}")

print("\nWhen to use Custom:")
print("   Transfer learning (use trained params from similar asset)")
print("   Incorporating expert domain knowledge")
print("   Research reproducibility (deterministic starting point)")
print("   Testing specific regime hypotheses")
print("   Warm-starting optimization from known good parameters")


# =============================================================================
# Example 4: Convenience Factory Method (from_regime_specs)
# =============================================================================
print_section("Example 4: Convenience Factory - from_regime_specs()")

print("The from_regime_specs() factory provides an ergonomic API for custom initialization.")
print("Just specify mean and std for each regime - all else is optional.\n")

# Create config using factory method
factory_config = HMMConfig.from_regime_specs(
    regime_specs=[
        {'mean': -0.020, 'std': 0.030},  # Crisis/Bear
        {'mean': 0.000, 'std': 0.015},   # Sideways
        {'mean': 0.015, 'std': 0.022},   # Bull
    ],
    # Can still override other config parameters
    max_iterations=50,
    random_seed=42,
)

print("Config created with from_regime_specs():")
print(f"  n_states: {factory_config.n_states}")
print(f"  initialization_method: {factory_config.initialization_method}")
print(f"  custom_emission_means: {factory_config.custom_emission_means}")
print(f"  custom_emission_stds: {factory_config.custom_emission_stds}")

hmm_factory = HiddenMarkovModel(factory_config)
hmm_factory.fit(spy_data)

diagnostics_factory = hmm_factory.get_initialization_diagnostics()
print_diagnostics(diagnostics_factory, "Factory")


# =============================================================================
# Example 5: Transfer Learning (SPY → AAPL)
# =============================================================================
print_section("Example 5: Transfer Learning (Using SPY parameters for AAPL)")

print("Transfer learning: Use trained parameters from SPY to initialize AAPL model.")
print("This can speed up convergence when assets have similar regime structure.\n")

# Get trained parameters from SPY model (using KMeans)
spy_means = hmm_kmeans.emission_means_
spy_stds = hmm_kmeans.emission_stds_
spy_transitions = hmm_kmeans.transition_matrix_

print("Trained SPY parameters (from KMeans initialization):")
for i in range(3):
    print(f"  State {i}: μ={spy_means[i]:.4f}, σ={spy_stds[i]:.4f}")

# Download AAPL data
aapl_data = download_data('AAPL', days=504)

# Create AAPL model using SPY parameters
transfer_config = HMMConfig(
    n_states=3,
    initialization_method='custom',
    custom_emission_means=spy_means.tolist(),
    custom_emission_stds=spy_stds.tolist(),
    custom_transition_matrix=spy_transitions.tolist(),
    max_iterations=50,
    random_seed=42,
)

print(f"\nTraining AAPL model ({len(aapl_data)} days) with SPY-derived parameters...")
hmm_transfer = HiddenMarkovModel(transfer_config)
hmm_transfer.fit(aapl_data)

print("\nFinal AAPL parameters (after Baum-Welch from SPY initialization):")
for i in range(3):
    print(f"  State {i}: μ={hmm_transfer.emission_means_[i]:.4f}, σ={hmm_transfer.emission_stds_[i]:.4f}")

print("\nTransfer learning benefits:")
print("   Faster convergence (good starting point)")
print("   More stable training (initialized in reasonable region)")
print("   Consistent regime interpretation across assets")
print("   Useful when limited data available for target asset")


# =============================================================================
# Example 6: Comparing Initialization Methods
# =============================================================================
print_section("Example 6: Comparing All Methods on Same Data")

print("Training 3 models on SPY with different initialization methods...\n")

methods = {
    'KMeans': hmm_kmeans,
    'Random': hmm_random,
    'Custom': hmm_custom,
}

print(f"{'Method':<15} {'State 0 μ':<12} {'State 1 μ':<12} {'State 2 μ':<12} {'Log-Likelihood':<15}")
print("-" * 80)

for name, model in methods.items():
    ll = model.training_history_.get('final_log_likelihood', 'N/A')
    ll_str = f"{ll:>14.2f}" if isinstance(ll, (int, float)) else f"{ll:>14}"
    print(f"{name:<15} {model.emission_means_[0]:>11.4f} {model.emission_means_[1]:>11.4f} "
          f"{model.emission_means_[2]:>11.4f} {ll_str}")

print("\nObservations:")
print("  • All methods converge to similar final parameters (Baum-Welch finds the same optima)")
print("  • KMeans often converges faster (better starting point)")
print("  • Custom initialization allows controlling regime interpretation")
print("  • Log-likelihood values are comparable across methods")


# =============================================================================
# Summary
# =============================================================================
print_section("Summary: Choosing an Initialization Method")

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ DECISION TREE: Which Initialization Method?                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Do you have specific domain knowledge about regime characteristics?       │
│    YES → Use Custom Initialization                                         │
│          • HMMConfig.from_regime_specs() for convenience                   │
│          • Specify means, stds, and optionally transitions                 │
│          • Good for transfer learning from other assets                    │
│                                                                             │
│    NO  → Do you have sklearn installed and >200 observations?              │
│            YES → Use KMeans (DEFAULT)                                       │
│                  • Best automatic initialization                           │
│                  • Data-driven regime discovery                            │
│                  • Fastest convergence in most cases                       │
│                                                                             │
│            NO  → Use Random                                                 │
│                  • Quantile-based initialization                           │
│                  • Works with minimal dependencies                         │
│                  • Good for small datasets                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

KEY CONCEPTS:

1. Initialization sets STARTING parameters, not final parameters
   - Baum-Welch algorithm updates parameters during fit()
   - Different initializations can converge to same final parameters
   - Good initialization → faster convergence, more stable training

2. Custom initialization with safety rails
   - Validation checks for financially realistic parameters
   - Warnings for extreme values (but doesn't block expert overrides)
   - Automatic normalization of probability distributions
   - Rich diagnostics for transparency

3. Transfer learning workflow
   - Train model on asset with lots of data (e.g., SPY)
   - Extract trained parameters
   - Use as custom initialization for related assets (e.g., AAPL)
   - Faster convergence and more consistent regime interpretation

4. All methods produce initialization diagnostics
   - Access via hmm.get_initialization_diagnostics()
   - Shows regime characteristics, persistence, expected durations
   - Useful for understanding and validating model behavior
""")

print("\n" + "="*80)
print("Demo complete! Try different initialization methods on your own data.")
print("="*80 + "\n")
