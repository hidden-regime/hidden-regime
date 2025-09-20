# Aspirational Tests

This directory contains tests for features that are planned but not yet implemented.

## Files:
- `test_hmm_aspirational.py` - Tests for future HMM model enhancements

## Missing HMM Methods (Future Implementation):

### Core ML API Methods:
- `predict_proba()` - Probability predictions for regime states
- `score()` - Model likelihood scoring
- `decode_states()` - Viterbi decoding for most likely state sequence

### Model Persistence:
- `save_model()` / `load_model()` - Model serialization

### Advanced Features:
- `partial_fit()` - Online/incremental learning
- `cross_validate()` - Cross-validation utilities
- `aic()` / `bic()` - Model selection criteria
- `get_regime_analysis()` - Detailed regime analysis
- `get_performance_monitoring()` - Performance tracking metrics

## Implementation Priority:
1. **High**: `predict_proba`, `score`, `decode_states` (core ML API)
2. **Medium**: `save_model`, `load_model` (persistence)
3. **Low**: Advanced features (can be external utilities)

## Current Status:
- Working implementation available in `test_hmm_working.py`
- Aspirational tests moved here to avoid blocking current development
- Implement features based on user needs and roadmap priorities