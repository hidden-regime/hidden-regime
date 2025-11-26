# Multivariate HMM Examples - Learning Progression

## Overview

The multivariate examples are organized as a learning progression from "what not to do" to "production-ready". Each example builds on lessons learned from previous ones.

---

## Example Progression

### 1. `multivariate_regime_detection.py` - ⚠️ Educational: What NOT to Do

**Purpose:** Demonstrate why feature choice matters and how poor choices fail catastrophically

**Key Points:**
- ❌ Uses `volume_change` as second feature
- ❌ Volume has 405x larger variance than returns (numerical instability)
- ❌ Results in 636 regime transitions (catastrophic overtrading)
- ❌ Transaction costs: 2.38% annually (destroys alpha)
- ❌ Kept for educational purposes ONLY

**What You'll Learn:**
- Feature scaling matters enormously
- Not all features are good for multivariate models
- Volume is regime-ambiguous
- How to recognize poor model behavior (too many transitions)

**When to Use This Example:**
- Understanding what to AVOID
- Educational discussions about feature selection
- Debugging problematic implementations

**Code Highlights:**
```python
# ❌ This is wrong - volume has much larger variance
config = HMMConfig(
    n_states=2,
    observed_signals=['log_return', 'volume_change'],  # BAD!
    max_iterations=50
)
```

**Key Takeaway:**
_"Don't use raw volume or features with extreme scale differences. Use realized_vol or other
regime-informative features instead."_

---

### 2. `multivariate_regime_detection_v2.py` - ✓ Improved: Best Practices

**Purpose:** Show the correct feature choice and best practices with direct model instantiation

**Key Points:**
- ✓ Uses `realized_vol` instead of `volume_change`
- ✓ Realized volatility is regime-informative
- ✓ Automatic feature standardization prevents scale mismatch
- ✓ Shows proper model convergence with fewer transitions
- ✓ Demonstrates how to interpret multivariate outputs

**What You'll Learn:**
- Realized volatility is a superior feature choice
- How to interpret multivariate model outputs
- Covariance matrices reveal regime structure
- Feature standardization handling

**When to Use This Example:**
- Understanding correct multivariate implementation
- Learning which features work well together
- Seeing direct HiddenMarkovModel usage
- Model interpretation practices

**Code Highlights:**
```python
# ✓ This is correct - realized_vol is regime-informative
config = HMMConfig(
    n_states=2,
    observation_mode=ObservationMode.MULTIVARIATE,
    observed_signals=['log_return', 'realized_vol'],
    max_iterations=100
)

# Access covariance matrices
model.fit(data)
covs = model.emission_covs_  # (n_states, 2, 2) shape
```

**Key Takeaway:**
_"Use regime-informative features like realized_vol. The covariance structure itself reveals
regime characteristics."_

---

### 3. `multivariate_regime_detection_v3.py` - 🚀 Production Ready: Pipeline Architecture

**Purpose:** Recommended approach using the full pipeline for production use

**Key Points:**
- ✓ Uses pipeline factory (`create_multivariate_pipeline`)
- ✓ Automatic feature generation (doesn't require pre-computing realized_vol)
- ✓ Proper component composition (Data → Observation → Model → Interpreter)
- ✓ Financial domain interpretation automatically added
- ✓ Regime profiles with multivariate characteristics
- ✓ Ready for production deployment

**What You'll Learn:**
- Using the recommended `create_multivariate_pipeline` factory
- Full pipeline architecture in action
- How interpretation enriches raw model output
- Best practices for production use

**When to Use This Example:**
- Building production systems
- Understanding the complete pipeline flow
- Feature generation and interpretation
- Starting new multivariate projects

**Code Highlights:**
```python
# ✓ This is the recommended approach
import hidden_regime as hr

pipeline = hr.create_multivariate_pipeline(
    'SPY',
    features=['log_return', 'realized_vol'],
    n_states=2
)

result = pipeline.update()
print(result[['regime_label', 'confidence', 'multivariate_eigenvalue_ratio']])
```

**Key Takeaway:**
_"Use the pipeline factory for production. It handles everything: feature generation,
standardization, model training, and interpretation."_

---

## Jupyter Notebooks

### `multivariate_hmm_deep_dive.ipynb`

**Purpose:** Interactive exploration of multivariate HMM mechanics and theory

**Contains:**
- Mathematical foundations of multivariate Gaussians
- Covariance matrix properties
- Eigenvalue decomposition intuition
- Step-by-step HMM algorithm walkthrough
- Interactive visualizations

**When to Use:**
- Understanding the mathematics
- Teaching or learning multivariate concepts
- Visualizing covariance structures
- Algorithm exploration

**Estimated Time:** 30-45 minutes

---

### `multivariate_hmm_validation.ipynb`

**Purpose:** Validation testing and performance metrics

**Contains:**
- Model convergence analysis
- Parameter stability checks
- Prediction accuracy metrics
- Covariance matrix properties verification
- Numerical stability tests

**When to Use:**
- Validating your own multivariate models
- Understanding convergence behavior
- Testing implementation correctness
- Performance benchmarking

**Estimated Time:** 20-30 minutes

---

## Which Example Should I Use?

### For Learning (Complete Beginner)

1. Start with `README_MULTIVARIATE_EXAMPLES.md` (this file)
2. Read through `multivariate_regime_detection.py` to understand what NOT to do
3. Study `multivariate_regime_detection_v2.py` to see correct implementation
4. Finally, use `multivariate_regime_detection_v3.py` as your template

**Total Time:** 1-2 hours

### For Understanding the Pipeline

1. Read the main `CLAUDE.md` in project root
2. Examine `multivariate_regime_detection_v3.py` (pipeline factory)
3. Review the pipeline architecture in `docs/guides/MULTIVARIATE_HMM_GUIDE.md`

**Total Time:** 30-45 minutes

### For Building a Production System

1. Start with `multivariate_regime_detection_v3.py` as your template
2. Reference `docs/reference/MULTIVARIATE_CONFIG_REFERENCE.md` for configuration
3. Read `docs/guides/MULTIVARIATE_HMM_GUIDE.md` for best practices
4. Use notebooks for validation of your models

**Total Time:** 2-4 hours (setup and testing)

### For Deep Understanding (Math/Theory)

1. Read `multivariate_hmm_deep_dive.ipynb` interactively
2. Study both v2 and v3 examples in parallel
3. Review `ARCHITECTURE.md` for system design
4. Experiment with your own feature combinations

**Total Time:** 3-5 hours

---

## Common Patterns Across Examples

### Pattern 1: Feature Selection

**v1 (Wrong):**
```python
observed_signals=['log_return', 'volume_change']  # ❌ 405x scale difference
```

**v2 (Correct):**
```python
observed_signals=['log_return', 'realized_vol']  # ✓ Regime-informative
```

**v3 (Recommended):**
```python
features=['log_return', 'realized_vol']  # ✓ Auto-generated by pipeline
```

**Lesson:** Feature choice is critical. Use regime-informative features with reasonable scale.

---

### Pattern 2: Configuration

**v1 (Low-Level):**
```python
config = HMMConfig(
    n_states=2,
    observed_signals=['log_return', 'volume_change']
    # No observation_mode specified (defaults to UNIVARIATE)
)
```

**v2 (Explicit):**
```python
from hidden_regime import ObservationMode

config = HMMConfig(
    n_states=2,
    observation_mode=ObservationMode.MULTIVARIATE,
    observed_signals=['log_return', 'realized_vol']
)
```

**v3 (Factory):**
```python
pipeline = hr.create_multivariate_pipeline(
    'SPY',
    features=['log_return', 'realized_vol'],
    n_states=2
    # All configuration handled automatically
)
```

**Lesson:** Use the factory for production. Explicit config for experimentation.

---

### Pattern 3: Model Usage

**v1 & v2 (Direct):**
```python
model = HiddenMarkovModel(config)
model.fit(data)
predictions = model.predict(data)
# Get: predicted_state, confidence, emission_means, emission_covs
```

**v3 (Pipeline):**
```python
pipeline = hr.create_multivariate_pipeline(...)
result = pipeline.update()
# Get: regime_label, confidence, multivariate_eigenvalue_ratio,
#      multivariate_correlation_regime, ... and everything else
```

**Lesson:** Pipeline adds financial interpretation. Use for production.

---

## Quick Reference

| Aspect | v1 (Educational) | v2 (Improved) | v3 (Production) |
|--------|------------------|---------------|-----------------|
| Feature Choice | ❌ volume_change | ✓ realized_vol | ✓ realized_vol |
| Feature Scaling | ❌ Manual (fails) | ✓ Auto (StandardScaler) | ✓ Auto (Pipeline) |
| Configuration | Minimal | Explicit | Factory |
| Model Usage | Direct HMM | Direct HMM | Pipeline |
| Interpretation | Raw output | Minimal | Full (eigenvalue analysis) |
| Production Ready | ❌ No | ⚠️ Partial | ✓ Yes |
| Code Complexity | Simple | Medium | Low (factory handles it) |
| Learning Value | High (what NOT to do) | High | High (best practices) |

---

## Debugging Guide

### "Model not converging" Error

See `multivariate_regime_detection_v1.py` - demonstrates how poor feature choice causes this.

**Solution:** Use better features like v2/v3 examples.

---

### "Singular covariance matrix"

Check your features:
- Are they highly correlated (>0.95)? Choose one.
- Do they have extreme scale differences? Pipeline should handle, but verify.
- Do you have enough data? Need 200+ obs for 2 features.

---

### "Too many regime transitions"

This is the key problem shown in v1.

**Indicators:**
- >5% of days are regime changes
- Transaction costs exceed alpha
- Model doesn't capture meaningful regimes

**Solution:** Use better features (v2/v3) and increase `max_iterations`.

---

## Key Takeaways

1. **Feature selection is everything** - v1 shows what happens with poor features
2. **Realized volatility works** - v2 demonstrates the correct choice
3. **Use the pipeline** - v3 shows the production-ready approach
4. **Scale matters** - Pipeline handles it, but understand why
5. **Interpretation is crucial** - Eigenvalues and correlations reveal regime structure

---

## Next Steps

1. **Start with v3** if you want production code
2. **Study v1 → v2 → v3** if you want to understand progression
3. **Use notebooks** for deep mathematical understanding
4. **Reference docs** for configuration and best practices

For more information:
- Production Guide: `docs/guides/MULTIVARIATE_HMM_GUIDE.md`
- Configuration Reference: `docs/reference/MULTIVARIATE_CONFIG_REFERENCE.md`
- Main Documentation: `README.md`
