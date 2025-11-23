# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2025-11-23

### BREAKING CHANGES

This release completes the architecture migration to a unified Interpreter + Signal Generation pattern. **All backward compatibility code has been removed.**

See [Migration Guide](docs/reference/MIGRATION_2.0.md) for detailed upgrade instructions.

### Removed

- **Legacy Components (66.6KB of code)**
  - `FinancialAnalysis` component - replaced by `FinancialInterpreter`
  - `AnalysisConfig` and `FinancialAnalysisConfig` - replaced by `InterpreterConfiguration`
  - `analysis` parameter in `Pipeline.__init__()`
  - `analysis_config` parameter in `PipelineFactory.create_pipeline()`
  - `analysis_output` property (use `interpreter_output` instead)
  - `_format_analysis_output()` method (use `_format_interpreter_output()` instead)

### Changed

- **Architecture**: Unified Interpreter + Signal Generation pattern
  - **Pipeline flow**: Data → Observation → Model → Interpreter → SignalGenerator → Report
  - **FinancialInterpreter** (895 lines): Unified regime interpretation with performance metrics
  - **FinancialSignalGenerator**: Dedicated trading signal generation component
  - **Clean separation**: Interpreter adds domain knowledge, SignalGenerator creates trading signals

- **Factory Methods**: Simplified signatures
  - `create_pipeline()` now requires `interpreter_config` (no longer accepts `analysis_config`)
  - `Pipeline()` now requires `interpreter` parameter (no longer accepts `analysis`)
  - `create_financial_pipeline()` uses new architecture by default

- **Column Names**: Standardized output naming
  - `regime_name` → `regime_label`
  - `regime_return` → `expected_return`
  - `regime_volatility` → `expected_volatility`
  - Note: MCP tools still support both naming conventions

- **Configuration System**
  - Removed: `AnalysisConfig`, `FinancialAnalysisConfig`
  - Added: `InterpreterConfiguration`, `SignalGenerationConfiguration`
  - Updated package exports in `hidden_regime.config`

### Added

- **Migration Guide**: Comprehensive guide for upgrading from v1.x to v2.0
  - Located at `docs/reference/MIGRATION_2.0.md`
  - Step-by-step migration checklist
  - Before/after code examples
  - Common pitfalls and solutions

### Fixed

- **Performance Metrics**: Data-driven calculations for win_rate, max_drawdown, regime_strength
- **Pipeline Output**: Proper formatting for interpreter output columns
- **MCP Tools**: Dynamic column name handling (supports both old and new conventions)
- **Import Paths**: Cleaned up all imports after removing legacy code

### Technical Details

- **Files Deleted**: 2 (financial.py, analysis.py)
- **Files Modified**: 7 (factories, pipeline core, config exports, imports)
- **Code Reduction**: 66.6KB removed
- **Architecture**: Single responsibility principle maintained throughout

- **Major Documentation Reorganization** - Consolidated 35+ root-level docs into organized structure
  - **New Structure**: 4 root docs (README, ARCHITECTURE, CONTRIBUTING, CHANGELOG) + organized `docs/` subdirectories
  - **docs/guides/**: User guides (quickstart, configuration, troubleshooting, OS-specific setup)
  - **docs/advanced/**: Advanced topics (features, trading, case studies, optimization, community examples)
  - **docs/reference/**: Technical reference (mathematical foundations, deployment, error handling)
  - **docs/integrations/**: Integration guides (MCP server in `docs/integrations/mcp/`)
  - **docs/experimental/**: Experimental features (QuantConnect integration, alpha strategies)
  - **CONTRIBUTING.md**: New comprehensive contributing guide consolidating 8+ testing docs
  - **Removed**: 15+ session artifacts and redundant documentation files
  - **Updated**: README.md with complete documentation index and navigation
  - **Result**: Reduced from 35 root .md files to 4 canonical docs + organized subdirectories

### Added
- **MCP Logging Interface** - Structured logging for transparency (4th of 5 MCP interfaces)
  - Logs cache operations (hit/miss/set) for all tools
  - Logs data loading with observation counts and date ranges
  - Logs model training start and completion
  - Logs operation completion with caching status
  - Uses Python's standard logging module (captured by FastMCP)
  - Added comprehensive test suite with 20+ test cases
  - Documentation in README_MCP.md with example log output

- **MCP Prompts Interface** - Complete implementation of MCP prompts (3rd of 5 MCP interfaces)
  - 7 curated prompt templates for expert-level regime analysis workflows
  - `regime_quick_check` - Fast market health check for day-to-day decisions
  - `regime_deep_dive` - Comprehensive multi-tool analysis with historical context
  - `regime_strategy_advisor` - Trading strategy recommendations based on regime + risk tolerance
  - `regime_multi_asset_comparison` - Cross-asset regime comparison and diversification analysis
  - `regime_risk_assessment` - Quantitative risk scoring (0-100) with mitigation strategies
  - `regime_historical_analogs` - Find similar past regime patterns and predict outcomes
  - `regime_portfolio_review` - Portfolio-wide regime analysis with rebalancing recommendations
  - New `hidden_regime_mcp/prompts.py` module with 7 async prompt functions
  - Comprehensive test suite: `tests/test_mcp/test_prompts.py` with 50+ test cases
  - Extended README_MCP.md with detailed prompts documentation (350+ lines)

### Documentation
- **MCP Prompts Technical Specification** - 40-page implementation guide (`MCP_PROMPTS_TECHNICAL_SPEC.md`)
  - Complete MCP Prompts interface overview and design rationale
  - Detailed specification for all 7 prompts with workflows and output formats
  - Implementation plan, testing strategy, and success metrics
  - Roadmap for future enhancements (Phase 2 prompts, prompt composition, dynamic prompts)
- **README_MCP.md Updates** - Comprehensive prompts section with:
  - Why use prompts vs tools comparison
  - Detailed documentation for each prompt with arguments, workflows, and sample outputs
  - "Prompts vs Tools: When to Use Each" decision guide
  - Usage examples showing real-world scenarios

### Technical Details
- Prompts registered in `hidden_regime_mcp/server.py` using FastMCP decorator pattern
- All prompts are async functions returning structured instruction text
- Argument validation with sensible defaults (e.g., risk_tolerance="medium", lookback_period="2y")
- Prompts reference actual MCP tools (detect_regime, get_regime_statistics, get_transition_probabilities)
- Modular architecture: prompts are instructional layer above operational tools

## [1.1.0] - 2025-11-04

### Added
- **Model Context Protocol (MCP) Server**
  - FastMCP-based server for AI assistant integration (Claude Desktop, others)
  - 3 core tools: `detect_regime`, `get_regime_statistics`, `get_transition_probabilities`
  - 2 resource endpoints: `regime://{ticker}/current`, `regime://{ticker}/transitions`
  - Rich temporal context: regime duration, expected transitions, confidence scores
  - Comprehensive 805-line documentation with examples and troubleshooting guide

- **Case Study Scripts** (3 historical market events)
  - `examples/case_study_dotcom_2000.py` - Dot-com bubble analysis (1999-2003)
  - `examples/case_study_2008_financial_crisis.py` - 2008 crisis analysis (2005-2009)
  - `examples/case_study_covid_2020.py` - COVID-19 rapid regime shift (2019-2021)
  - All scripts produce detailed regime analysis, regime transitions, and performance metrics

- **Enhanced Observation Features**
  - `momentum_strength`: Bull/bear momentum detection (23.8% improvement in bull market detection)
  - `trend_persistence`: Sideways regime identification
  - `volatility_context`: Crisis period detection
  - `directional_consistency`: Return sign patterns for regime interpretation

- **Test Infrastructure Improvements**
  - 36 test files with comprehensive coverage
  - Test markers: unit, integration, e2e, network, slow, performance
  - Enhanced error handling and validation testing
  - MCP tool testing (tools, resources, error cases)

### Changed
- Enhanced MCP server with caching support and error recovery
- Improved regime interpretation with financial context features
- More robust yfinance integration with retry logic

### Fixed
- Test compatibility issues with enhanced observation features
- MCP resource response formatting and error messages
- Case study script data alignment and validation

### Documentation
- Added comprehensive README_MCP.md (22K) with quick start, tool docs, scenarios, troubleshooting
- Updated examples/README.md with case study documentation
- Added example Claude conversations showing MCP usage patterns

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