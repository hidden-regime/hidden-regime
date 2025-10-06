"""Test script for KMeans initialization diagnostics."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

from hidden_regime.config.model import HMMConfig
from hidden_regime.models.hmm import HiddenMarkovModel

np.random.seed(42)

print("=" * 80)
print("TEST 1: Financial-like data (should work well)")
print("=" * 80)

# Generate financial-like Gaussian mixture
NUM_SAMPLES = 100
observations_financial = []
for i in range(NUM_SAMPLES):
    if i % 30 < 15:
        obs = np.random.normal(-0.2, 0.05)  # Bear regime
    else:
        obs = np.random.normal(0.2, 0.05)   # Bull regime
    observations_financial.append(obs)

df_financial = pd.DataFrame({'returns': observations_financial})

# Train HMM
hmm_financial = HiddenMarkovModel(HMMConfig(n_states=2, observed_signal='returns'))
hmm_financial.fit(df_financial)

# Print diagnostics
hmm_financial.print_initialization_report()

print("\n" + "=" * 80)
print("TEST 2: Non-financial data (should show constraint distortion)")
print("=" * 80)

# Generate non-financial Gaussian mixture (original HMM101 scenario)
observations_non_financial = []
for i in range(NUM_SAMPLES):
    if i % 30 < 15:
        obs = np.random.normal(-1.0, 0.5)  # State 0
    else:
        obs = np.random.normal(1.0, 0.5)   # State 1
    observations_non_financial.append(obs)

df_non_financial = pd.DataFrame({'returns': observations_non_financial})

# Train HMM
hmm_non_financial = HiddenMarkovModel(HMMConfig(n_states=2, observed_signal='returns'))
hmm_non_financial.fit(df_non_financial)

# Print diagnostics
hmm_non_financial.print_initialization_report()

print("\n" + "=" * 80)
print("TEST 3: Random initialization (no KMeans)")
print("=" * 80)

# Test random initialization
hmm_random = HiddenMarkovModel(
    HMMConfig(n_states=2, observed_signal='returns', initialization_method='random')
)
hmm_random.fit(df_financial)
hmm_random.print_initialization_report()

print("\nAll tests complete!")
