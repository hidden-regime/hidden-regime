# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-09-30

### Added
- **Complete Documentation Overhaul**
  - Rewrote root README.md focused on v1.0.0 capabilities (removed all aspirational content)
  - Created 10 comprehensive module READMEs with accurate API documentation
  - Completely rewrote examples/README.md documenting all 12 working examples
  - Added installation instructions, quick start guides, and learning paths

- **Production-Ready Examples** (12 total)
  - Basic examples: 00_basic_regime_detection, 01_real_market_analysis, 02_regime_comparison_analysis
  - Intermediate examples: 03_trading_strategy_demo, 04_multi_stock_comparative_study
  - Advanced examples: 05_advanced_analysis_showcase, improved_features
  - Case study examples: case_study, case_study_basic, case_study_comprehensive, case_study_multi_asset, financial_case_study
  - All examples tested and verified working with documented runtimes (15s-180s)

- **Proper Dependency Management**
  - Moved scikit-learn to main dependencies (used for KMeans initialization and metrics)
  - Created visualization optional dependency group (seaborn, plotly, Pillow)
  - Clear separation between core, visualization, dev, and docs dependencies

- **Test Infrastructure**
  - Organized test suite with proper markers (slow, network, performance)
  - 177 passing fast tests (100% pass rate)
  - Comprehensive coverage across all modules

### Changed
- **Data Validation**
  - Relaxed NaN validation threshold from 0% to 10% for practical use with pct_change() calculations
  - More realistic handling of missing data in time series

- **Package Metadata**
  - Updated classifier to "Development Status :: 5 - Production/Stable"
  - Enhanced optional dependencies with visualization group

### Fixed
- **Test Suite Fixes** (5 total)
  - 4 model tests: Removed references to deleted online learning attributes (_current_state_probs, _last_observation, _sufficient_stats)
  - 1 analysis test: Corrected attribute name from indicator_calculator to indicator_analyzer

- **Example Fixes**
  - Fixed 5 examples requiring model_component parameter for data-driven regime interpretation
  - Fixed 1 example with incorrect API usage (old DataLoader pattern)
  - All 12 examples now execute successfully

### Documentation
- **Module Documentation**: Complete README for each of 10 modules (config, data, observations, models, analysis, reporting, pipeline, utils, visualization, financial)
- **API Documentation**: Full docstring coverage with examples
- **Examples Guide**: Comprehensive examples/README.md with:
  - Runtime estimates for each example
  - Complexity levels (Beginner/Intermediate/Advanced)
  - Learning paths for different user types
  - Command reference for running tests
- **Configuration Guide**: Documentation of all config classes and presets

### Technical Details
- **Architecture**: Pipeline-based (Data → Observation → Model → Analysis → Report)
- **HMM Implementation**: Gaussian emissions with Baum-Welch training
- **Financial Regime Types**: Bear, Bull, Sideways, Crisis (data-driven classification)
- **Trading Simulation**: Regime-based strategies with realistic backtesting
- **Temporal Control**: Proper data isolation for V&V backtesting

## [Unreleased]

### Added
- Enhanced PyPI packaging with pyproject.toml support
- Comprehensive CI/CD pipeline with GitHub Actions
- Performance testing and benchmarking infrastructure
- Security scanning with Bandit
- Code quality checks with Black, isort, flake8

### Changed
- Improved package metadata with project URLs and enhanced classifiers
- Centralized version management in `_version.py`
- Enhanced MANIFEST.in for better file inclusion control

### Fixed
- Unit test compatibility issues across Python versions
- Log-likelihood sign assertions in test suite
- DataFrame assignment warnings in preprocessing
- Performance test rate limiting issues

## [0.1.0] - 2025-01-XX

### Added
- **Complete Data Pipeline**
  - Multi-source data loading with yfinance integration
  - Comprehensive 6-layer data validation system
  - Advanced preprocessing with outlier detection and missing value handling
  - Intelligent caching with rate limiting and automatic retry logic
  - Quality scoring system (0.0-1.0 scale) for quantitative data assessment
  - Multi-asset support with timestamp alignment and batch processing

- **Hidden Markov Models (HMM)**
  - 3-state regime detection optimized for Bear, Sideways, Bull markets
  - Baum-Welch EM training algorithm with numerical stability enhancements
  - Viterbi algorithm for most likely regime sequence inference
  - Forward-Backward algorithm for state probability computation
  - Real-time regime tracking with online state probability updates
  - Model persistence with save/load functionality (JSON/Pickle formats)

- **Regime Analysis & Interpretation**
  - Comprehensive regime analysis with duration statistics
  - Automatic regime interpretation (Bull/Bear/Sideways with volatility levels)
  - Regime transition analysis and timing predictions
  - Performance statistics by regime type

- **Configuration System**
  - Flexible configuration classes for data loading, validation, and preprocessing
  - Factory methods for common use cases (market data, high-frequency data)
  - Validation of configuration parameters against data characteristics

- **Error Handling & Exceptions**
  - Comprehensive exception hierarchy for different error types
  - Graceful error handling with informative error messages
  - Data quality warnings and validation feedback

- **Testing & Quality Assurance**
  - Comprehensive unit test suite (80%+ coverage)
  - Integration tests for end-to-end workflows
  - Performance tests and benchmarking
  - Mock data generation for consistent testing
  - Type hints and static analysis support

- **Examples & Documentation**
  - Complete API documentation in models/README.md
  - Practical examples for common use cases
  - Data pipeline demonstrations
  - HMM training and analysis examples
  - Portfolio analysis integration examples

### Technical Details
- **Dependencies**: numpy, pandas, scipy, matplotlib, yfinance
- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Operating Systems**: Linux, macOS, Windows
- **Architecture**: Modular design with clear separation of concerns
- **Performance**: Optimized algorithms with numerical stability enhancements

### Package Structure
```
hidden_regime/
├── config/          # Configuration management
├── data/            # Data loading, validation, preprocessing
├── models/          # HMM implementation and algorithms
├── utils/           # Shared utilities and exceptions
examples/            # Usage examples and demonstrations
tests/               # Comprehensive test suite
```

### Known Limitations
- Currently supports Gaussian emission models only
- Online learning capabilities are basic (planned for future releases)
- Limited to 3-state models by default (configurable up to 6 states)
- Fat-tailed emission models not yet implemented
- Duration modeling for regime persistence is planned for v0.2.0

## [0.0.1] - 2025-01-XX

### Added
- Initial project structure
- Basic package configuration
- Placeholder PyPI package (published as placeholder)

---

## Release Notes

### Version 0.1.0 - "Foundation Release"

This is the first production-ready release of Hidden Regime, providing a complete toolkit for market regime detection using Hidden Markov Models. The package includes:

** Production-Ready Components:**
- Robust data pipeline with comprehensive validation
- Numerically stable HMM implementation
- Real-time regime detection capabilities
- Extensive testing and documentation

** Proven Performance:**
- Successfully tested on major market indices (SPY, QQQ, etc.)
- Handles various market conditions and time periods
- Reliable regime detection with quantified uncertainty

**🔧 Developer-Friendly:**
- Clean, well-documented API
- Comprehensive examples and tutorials
- Flexible configuration system
- Extensive error handling

**🎯 Target Users:**
- Quantitative analysts and researchers
- Algorithmic trading developers
- Financial data scientists
- Academic researchers in finance

### Upgrade Path

This is the first stable release. Future versions will maintain backward compatibility for the public API while adding new features and improvements.

### Support

- **Documentation**: Available in the `models/README.md` and inline docstrings
- **Examples**: See the `examples/` directory for practical usage patterns
- **Issues**: Report bugs and feature requests on GitHub
- **Community**: Join discussions in GitHub Discussions

### Acknowledgments

Special thanks to the open-source community and the contributors who helped make this release possible.