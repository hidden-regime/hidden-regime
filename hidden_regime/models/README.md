# Model Components - Pure Mathematical Layer

This directory contains the **pure mathematical model layer** of Hidden Regime. All code in this directory follows the **Model Purity Principle**: models contain only mathematical logic with zero domain-specific knowledge.

## Architectural Principle: Domain-Agnostic Models

**CRITICAL**: Models must be completely domain-agnostic. They should work for:
- Financial time series (the primary use case)
- Weather pattern detection
- Speech recognition states
- Manufacturing process monitoring
- Any time series with hidden states

All domain knowledge (financial, meteorological, etc.) belongs in the **Interpreter** layer, not here.

## Model Purity Rules

### ✅ ALLOWED in models/
- Mathematical algorithms (Baum-Welch, Viterbi, Forward-Backward)
- Statistical computations (Gaussian emissions, likelihoods, transitions)
- Numerical stability techniques (log-space, regularization)
- Generic state terminology ("State 0", "State 1", "high-return state", "low-return state")
- Data preprocessing (outlier filtering, normalization)
- Clustering algorithms (K-means, GMM, quantile-based)
- Parameter initialization and validation

### ❌ FORBIDDEN in models/
- Financial terminology ("bull", "bear", "crisis", "sideways")
- Trading concepts ("buy", "sell", "long", "short", "signal")
- Economic indicators ("Sharpe ratio", "alpha", "beta", "drawdown")
- Market-specific logic ("earnings", "splits", "dividends")
- Portfolio concepts ("position", "allocation", "portfolio")
- Risk metrics ("VaR", "CVaR", "max drawdown")

## File Structure

```
models/
├── README.md                   # This file - architectural documentation
├── __init__.py                 # Public API exports
├── hmm.py                      # Hidden Markov Model implementation
├── multitimeframe.py           # Multi-timeframe HMM wrapper
├── utils.py                    # Mathematical utilities and initialization
└── algorithms/                 # Core HMM algorithms (future)
```

## Core Classes

### `HiddenMarkovModel` (hmm.py)
Pure mathematical HMM implementation:
- **Inputs**: Numeric observations (typically log returns, but domain-agnostic)
- **Outputs**: State sequences (0, 1, 2, ...) and emission parameters (means, stds)
- **No knowledge of**: What the states represent, financial concepts, trading logic

**Example Output:**
```python
{
    'predicted_state': [0, 1, 2, 1, 0, ...],  # Just state IDs
    'emission_means': [-0.015, 0.001, 0.012],  # Mathematical means
    'emission_stds': [0.025, 0.015, 0.020],    # Mathematical standard deviations
    'transition_matrix': [[0.9, 0.08, 0.02], ...]  # Transition probabilities
}
```

### `MultiTimeframeRegime` (multitimeframe.py)
Multi-horizon state detection:
- Trains independent HMMs on daily/weekly/monthly data
- Computes alignment scores (0-1) between timeframes
- **No knowledge of**: Why alignment matters, trading implications

## Model-Interpreter Separation

The architectural separation is crystal clear:

```
┌─────────────────────────────────────────┐
│         MODEL COMPONENT                 │
│  (models/hmm.py)                        │
│                                         │
│  Input:  [0.01, -0.02, 0.005, ...]     │
│  Output: states=[0, 1, 2, ...]         │
│          means=[-0.015, 0.001, 0.012]  │
│          stds=[0.025, 0.015, 0.020]    │
│                                         │
│  NO KNOWLEDGE OF FINANCE                │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│      INTERPRETER COMPONENT              │
│  (interpreter/financial.py)             │
│                                         │
│  Input:  Model output (states + params)│
│  Output: regime_labels=['Bear', ...]   │
│          win_rates=[0.35, 0.55, 0.72]  │
│          sharpe_ratios=[-0.5, 0.1, 1.2]│
│                                         │
│  ALL FINANCIAL KNOWLEDGE HERE           │
└─────────────────────────────────────────┘
```

### Why This Separation Matters

1. **Reusability**: The same HMM can detect weather patterns, speech states, or machine states
2. **Testability**: Mathematical correctness can be verified independently of domain interpretation
3. **Maintainability**: All financial logic is concentrated in one layer
4. **Flexibility**: Change financial interpretation without touching model code

## Example: Correct vs Incorrect Design

### ❌ INCORRECT (Violates Model Purity)
```python
# BAD - Financial knowledge in Model
class HiddenMarkovModel:
    def predict(self, returns):
        states = self._viterbi(returns)

        # ❌ Financial interpretation in model
        if self.emission_means_[states[-1]] > 0.001:
            return "BULLISH"
        elif self.emission_means_[states[-1]] < -0.001:
            return "BEARISH"
        else:
            return "SIDEWAYS"
```

### ✅ CORRECT (Maintains Model Purity)
```python
# GOOD - Model stays pure
class HiddenMarkovModel:
    def predict(self, returns):
        states = self._viterbi(returns)
        return {
            'predicted_state': states,
            'state_probabilities': self._state_probs,
            'emission_means': self.emission_means_,
            'emission_stds': self.emission_stds_
        }

# Financial interpretation happens in Interpreter
class FinancialInterpreter:
    def interpret(self, model_output):
        state = model_output['predicted_state'][-1]
        mean = model_output['emission_means'][state]

        # ✅ Financial logic only in interpreter
        if mean > 0.001:
            return RegimeType.BULLISH
        elif mean < -0.001:
            return RegimeType.BEARISH
        else:
            return RegimeType.SIDEWAYS
```

## Initialization Methods

All initialization methods in `utils.py` are domain-agnostic:

- **`initialize_parameters_quantile`**: Splits data into equal-sized percentile bins
- **`initialize_parameters_kmeans`**: K-means clustering on returns + volatility features
- **`initialize_parameters_gmm`**: Gaussian Mixture Model clustering
- **`initialize_parameters_random`**: Random parameter initialization

While these use constraints appropriate for typical daily return ranges (-8% to +5%), the logic is mathematical, not financial.

## Validation and Quality Metrics

### Mathematical Validation (models/utils.py)
```python
# ✅ Domain-agnostic quality checks
validate_state_quality(emission_params)  # Checks state separation (Cohen's d)
analyze_state_transitions(states, transition_matrix)  # Persistence analysis
calculate_state_statistics(states, returns)  # Duration, frequency statistics
```

### Financial Validation (interpreter/)
```python
# ✅ Financial metrics computed in interpreter
interpreter.calculate_sharpe_ratio(state_returns)
interpreter.calculate_max_drawdown(state_returns)
interpreter.calculate_win_rate(state_returns)
```

## Testing

Architectural compliance tests ensure model purity:

```bash
# Run architectural tests
pytest tests/architecture/test_model_purity.py -v

# Tests enforce:
# 1. No financial terminology in models/ code
# 2. HMM API outputs only mathematical values
# 3. Config parameters are domain-agnostic
```

## Contributing

When adding new model components:

1. **Ask**: "Could this code work for non-financial time series?"
2. **If no**: Move the logic to `interpreter/` instead
3. **If yes**: Ensure terminology is mathematical, not domain-specific
4. **Always**: Run architectural tests before committing

### Code Review Checklist
- [ ] No financial terms in variable names
- [ ] No financial terms in function names
- [ ] No financial logic in model code
- [ ] Outputs are mathematical (states, parameters, probabilities)
- [ ] All tests in `tests/architecture/` pass

## References

- **Pipeline Architecture**: See `ARCHITECTURE.md` for the full pipeline design
- **Interpreter Component**: See `interpreter/README.md` for financial interpretation
- **Factory Pattern**: See `factories/README.md` for component creation

---

**Remember**: Models are mathematics. Interpreters are domain knowledge. Never mix the two.

*Last Updated: 2025-11-21*
