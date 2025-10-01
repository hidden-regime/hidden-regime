# Reports Module

The reports module provides comprehensive report generation from pipeline analysis results, supporting multiple output formats and customizable content sections.

## Overview

The reports layer transforms pipeline outputs into formatted documents with visualizations, statistics, and actionable insights:

```
Pipeline Results → Report Generation → Formatted Output
        ↓                  ↓                    ↓
  Analysis Data      Section Building      Markdown/HTML
  Performance        Visualization         PDF/JSON
  Risk Metrics       Formatting            Files
```

## Core Components

### MarkdownReportGenerator

Main report component implementing ReportComponent interface.

```python
from hidden_regime.reports import MarkdownReportGenerator
from hidden_regime.config import ReportConfig

# Configure report
config = ReportConfig(
    output_dir='./reports',
    template_style='professional',
    include_summary=True,
    include_regime_analysis=True,
    include_performance_metrics=True,
    save_plots=True
)

# Create generator
generator = MarkdownReportGenerator(config)

# Generate report from pipeline results
report = generator.update(
    data=price_data,
    observations=observations,
    model_output=hmm_predictions,
    analysis=analysis_results
)

print(report)  # Markdown-formatted report
```

**Key Features:**
- Pipeline component integration
- Configurable sections
- Automatic file saving
- Multiple template styles
- Data quality assessment
- Trading signal reporting

### CaseStudyReportGenerator

Specialized generator for comprehensive case study analysis.

```python
from hidden_regime.reports.case_study import CaseStudyReportGenerator
from hidden_regime.config import CaseStudyConfig

# Configure case study
config = CaseStudyConfig(
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    n_states=3,
    n_training=252,
    create_animations=True
)

# Create generator
generator = CaseStudyReportGenerator(config)

# Generate comprehensive report
report = generator.generate_case_study_report(
    evolution_results=evolution_data,
    performance_evolution=performance_data,
    final_comparison=comparison_results,
    execution_time=elapsed_seconds
)
```

**Includes:**
- Evolution analysis (regime progression over time)
- Performance metrics (returns, risk, Sharpe ratio)
- Strategy comparison (HMM vs. indicators vs. buy-hold)
- Methodology section
- Technical appendix
- Configuration details

## Report Configuration

### ReportConfig

```python
from hidden_regime.config import ReportConfig

config = ReportConfig(
    # Output settings
    output_dir='./reports',
    output_format='markdown',  # markdown, html, pdf, json

    # Visualization
    show_plots=False,
    save_plots=True,
    plot_format='png',  # png, pdf, svg
    plot_dpi=300,

    # Content sections
    include_summary=True,
    include_regime_analysis=True,
    include_performance_metrics=True,
    include_risk_analysis=True,
    include_trading_signals=False,
    include_data_quality=True,

    # Styling
    title='Market Regime Analysis Report',
    template_style='professional'  # professional, academic, minimal
)
```

### Factory Methods

```python
# Minimal report (summary and regime analysis only)
config = ReportConfig.create_minimal()

# Comprehensive report (all sections)
config = ReportConfig.create_comprehensive()

# Presentation-ready (HTML with embedded visuals)
config = ReportConfig.create_presentation()
```

## Report Sections

### Executive Summary

High-level overview of analysis results.

```markdown
## Executive Summary

**Data Period**: 2023-01-01 to 2024-01-01
**Total Observations**: 252
**Current Regime**: Bull Market
**Confidence**: 87.3%
**Days in Current Regime**: 12
**Expected Daily Return**: 0.12%
**Expected Volatility**: 1.8%
```

**Includes:**
- Data period and observation count
- Current regime identification
- Confidence levels
- Expected characteristics

### Regime Analysis

Detailed regime statistics and distribution.

```markdown
## Regime Analysis

### Regime Distribution
| Regime | Occurrences | Percentage |
|--------|-------------|------------|
| Bear   | 45          | 17.9%      |
| Sideways | 102       | 40.5%      |
| Bull   | 105         | 41.7%      |

### Current Regime Details
- **Regime**: Bull Market
- **Confidence**: 87.3%
- **Duration**: 12 periods
- **Expected Total Duration**: 18.5 periods
```

**Includes:**
- Regime distribution statistics
- Regime stability metrics
- Current regime details
- Transition analysis

### Performance Metrics

Model confidence and prediction statistics.

```markdown
## Performance Metrics

### Model Confidence
| Metric | Value |
|--------|-------|
| Mean   | 0.782 |
| Std Dev | 0.124 |
| Min    | 0.412 |
| Max    | 0.973 |

### State Predictions
| State | Count | Percentage |
|-------|-------|------------|
| 0     | 45    | 17.9%      |
| 1     | 102   | 40.5%      |
| 2     | 105   | 41.7%      |
```

**Includes:**
- Model confidence statistics
- State prediction distribution
- Advanced performance analysis (if available)

### Advanced Performance Analysis

Comprehensive metrics from FinancialAnalysis component.

```markdown
## Advanced Performance Analysis

### Performance Summary
| Metric | Value |
|--------|-------|
| Overall Quality | High |
| Quality Score | 85/100 |
| Average Confidence | 78.2% |
| Balance Score | 0.897 |
| Stability Rating | Stable |

### Regime Persistence
| State | Persistence |
|-------|-------------|
| Bear  | 92.3%       |
| Sideways | 95.8%    |
| Bull  | 88.7%       |
```

**Includes:**
- Regime dominance analysis
- Transition frequency
- Duration statistics
- Confidence quality
- Return performance by regime

### Risk Analysis

Risk assessment by regime type.

```markdown
## Risk Analysis

### Risk by Regime Type
| Regime | Avg Volatility | Std Dev | Observations |
|--------|---------------|---------|--------------|
| Bear   | 2.45%         | 0.38%   | 45           |
| Sideways | 1.52%       | 0.22%   | 102          |
| Bull   | 1.87%         | 0.31%   | 105          |

### Current Risk Assessment
- **Current Regime Type**: Bull Market
- **Expected Volatility**: 1.87%
- **Risk Level**: Moderate
```

**Includes:**
- Volatility by regime
- Current risk assessment
- Risk classification (Low, Moderate, High, Very High)

### Trading Signals

Position recommendations based on regime.

```markdown
## Trading Signals

### Current Position Recommendation
- **Position Signal**: 0.850
- **Signal Strength**: 0.850
- **Recommendation**: Strong Long

### Signal Statistics
| Metric | Value |
|--------|-------|
| Mean   | 0.234 |
| Std Dev | 0.512 |
| Min    | -0.650 |
| Max    | 0.920 |
```

**Includes:**
- Current position signal
- Signal interpretation
- Historical signal statistics

### Data Quality Assessment

Data quality and completeness analysis.

```markdown
## Data Quality Assessment

### Data Overview
- **Total Observations**: 252
- **Date Range**: 2023-01-01 to 2024-01-01
- **Data Columns**: open, high, low, close, volume

### Missing Data
✅ No missing data detected.

### Observation Quality
✅ All observations computed successfully.
```

**Includes:**
- Basic data statistics
- Missing data analysis
- Observation quality check

## Pipeline Integration

### Automatic Report Generation

```python
import hidden_regime as hr

# Create pipeline with reporting
pipeline = hr.create_financial_pipeline(
    'AAPL',
    n_states=3,
    generate_reports=True,
    report_output_dir='./reports'
)

# Update generates report automatically
result = pipeline.update()

# Report saved to: ./reports/regime_analysis.md
```

### Custom Report Component

```python
from hidden_regime.pipeline import Pipeline
from hidden_regime.reports import MarkdownReportGenerator
from hidden_regime.config import ReportConfig

# Configure custom report
report_config = ReportConfig(
    output_dir='./custom_reports',
    template_style='academic',
    include_trading_signals=True
)

# Create pipeline with custom report component
pipeline = Pipeline(
    data=data_component,
    observation=observation_component,
    model=model_component,
    analysis=analysis_component,
    report=MarkdownReportGenerator(report_config)
)

# Run pipeline
result = pipeline.update()
```

## Usage Examples

### Example 1: Basic Report

```python
from hidden_regime.reports import MarkdownReportGenerator
from hidden_regime.config import ReportConfig

# Minimal configuration
config = ReportConfig.create_minimal()

# Create generator
generator = MarkdownReportGenerator(config)

# Generate report
report = generator.update(
    data=price_data,
    model_output=hmm_output
)

print(report)
```

### Example 2: Comprehensive Report

```python
from hidden_regime.config import ReportConfig

# Comprehensive configuration
config = ReportConfig.create_comprehensive()
config = config._replace(
    output_dir='./output/reports',
    title='AAPL Market Regime Analysis',
    plot_dpi=600
)

generator = MarkdownReportGenerator(config)

# Generate with all pipeline outputs
report = generator.update(
    data=price_data,
    observations=observations,
    model_output=model_predictions,
    analysis=analysis_results
)

# Report automatically saved to ./output/reports/regime_analysis.md
```

### Example 3: Case Study Report

```python
from hidden_regime.reports.case_study import CaseStudyReportGenerator
from hidden_regime.config import CaseStudyConfig

# Configure case study
config = CaseStudyConfig(
    ticker='SPY',
    start_date='2020-01-01',
    end_date='2023-12-31',
    n_states=3,
    n_training=252,
    include_technical_indicators=True,
    indicators_to_compare=['RSI', 'MACD'],
    create_animations=True,
    output_directory='./case_studies/spy'
)

# Generate case study
generator = CaseStudyReportGenerator(config)

# Run case study analysis (generates data)
# Then create report
report = generator.generate_case_study_report(
    evolution_results=evolution_data,
    performance_evolution=perf_data,
    final_comparison=comparison_data,
    execution_time=120.5
)

# Comprehensive report saved to ./case_studies/spy/reports/
```

### Example 4: Custom Template Style

```python
config = ReportConfig(
    template_style='academic',  # More formal style
    include_regime_analysis=True,
    include_performance_metrics=True,
    include_risk_analysis=True,
    title='Academic Market Regime Study',
    save_plots=True,
    plot_format='pdf'  # Publication-quality
)

generator = MarkdownReportGenerator(config)
report = generator.update(**pipeline_outputs)
```

## Best Practices

### 1. Choose Appropriate Template Style

```python
# For professional trading reports
config = ReportConfig(template_style='professional')

# For academic research
config = ReportConfig(template_style='academic')

# For quick analysis
config = ReportConfig(template_style='minimal')
```

### 2. Enable Relevant Sections

```python
# Trading focus
config = ReportConfig(
    include_regime_analysis=True,
    include_trading_signals=True,
    include_risk_analysis=True,
    include_performance_metrics=False  # Skip if not needed
)

# Research focus
config = ReportConfig(
    include_performance_metrics=True,
    include_data_quality=True,
    include_trading_signals=False
)
```

### 3. Organize Output Files

```python
config = ReportConfig(
    output_dir='./output/reports/2024',
    title=f'AAPL Regime Analysis {datetime.now().strftime("%Y-%m-%d")}',
    save_plots=True
)

# Creates organized directory structure:
# ./output/reports/2024/
#   ├── regime_analysis.md
#   └── plots/
```

### 4. Validate Before Generating

```python
# Check configuration is valid
try:
    config.validate()
except ConfigurationError as e:
    print(f"Configuration error: {e}")

# Now safe to use
generator = MarkdownReportGenerator(config)
```

## Output Formats

### Markdown (Default)

```python
config = ReportConfig(output_format='markdown')
# Generates .md files
```

**Use Cases:**
- Easy to read in plain text
- Version control friendly
- Convert to other formats
- Documentation

### HTML

```python
config = ReportConfig(output_format='html')
# Generates .html files
```

**Use Cases:**
- Web viewing
- Embedded visualizations
- Interactive elements
- Presentations

### PDF

```python
config = ReportConfig(output_format='pdf')
# Generates .pdf files (requires additional dependencies)
```

**Use Cases:**
- Professional reports
- Printable documents
- Fixed formatting
- Distribution

### JSON

```python
config = ReportConfig(output_format='json')
# Generates .json files
```

**Use Cases:**
- Programmatic access
- API responses
- Data integration
- Further processing

## Module Structure

```
reports/
├── __init__.py            # Public API
├── markdown.py            # MarkdownReportGenerator
└── case_study.py          # CaseStudyReportGenerator
```

## Related Modules

- **[pipeline](../pipeline/README.md)**: Pipeline integration (report as final component)
- **[analysis](../analysis/README.md)**: Analysis results (input to reports)
- **[visualization](../visualization/README.md)**: Plot generation
- **[config](../config/README.md)**: ReportConfig and CaseStudyConfig

## Key Concepts

### Pipeline Integration

Reports implement the ReportComponent interface:

```python
class ReportComponent(PipelineComponent):
    def update(self, **kwargs) -> str:
        """Generate report from pipeline results."""
        pass

    def plot(self, **kwargs) -> plt.Figure:
        """Visualize report generation status."""
        pass
```

This allows seamless integration as the final pipeline stage.

### Configurable Content

Reports are highly configurable:

```python
# Choose exactly what you need
config = ReportConfig(
    include_summary=True,           # Executive summary
    include_regime_analysis=True,   # Regime stats
    include_performance_metrics=True, # Model performance
    include_risk_analysis=False,    # Skip if not needed
    include_trading_signals=False,  # Skip if not trading
    include_data_quality=True       # Data validation
)
```

### Template Styles

Three built-in styles:

| Style | Description | Use Case |
|-------|-------------|----------|
| **professional** | Clean, business-oriented | Trading reports, presentations |
| **academic** | Formal, detailed | Research papers, studies |
| **minimal** | Concise, essential only | Quick analysis, summaries |

### Case Study Reports

Specialized for comprehensive analysis:

1. **Evolution Analysis**: Regime progression over time
2. **Performance Tracking**: Strategy returns and metrics
3. **Comparative Analysis**: HMM vs. indicators vs. benchmarks
4. **Methodology**: Detailed analysis approach
5. **Technical Appendix**: Configuration and reproducibility

---

For complete examples using reports, see `examples/case_study_comprehensive.py` and `examples/financial_case_study.py` in the project root.