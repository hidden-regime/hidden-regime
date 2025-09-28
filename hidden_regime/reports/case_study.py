"""
Case study report generation module.

Provides specialized report generation for comprehensive case studies
including markdown and HTML output with embedded visualizations.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import base64
import io
import matplotlib.pyplot as plt

from .markdown import MarkdownReportGenerator
from ..config.case_study import CaseStudyConfig
from ..utils.exceptions import ReportGenerationError


class CaseStudyReportGenerator(MarkdownReportGenerator):
    """
    Specialized report generator for case study analysis.

    Extends the standard markdown report generator with case study specific
    sections, visualizations, and comparative analysis summaries.
    """

    def __init__(self, config: CaseStudyConfig):
        """
        Initialize case study report generator.

        Args:
            config: Case study configuration
        """
        # Initialize parent with case study specific report config
        from ..config.report import ReportConfig

        report_config = ReportConfig(
            template_style='academic',
            include_performance_metrics=True,
            include_risk_analysis=True,
            save_plots=True,
            output_directory=config.output_directory
        )

        super().__init__(report_config)
        self.case_study_config = config

    def generate_case_study_report(
        self,
        evolution_results: List[Dict[str, Any]],
        performance_evolution: List[Dict[str, Any]],
        final_comparison: Dict[str, Any],
        execution_time: float
    ) -> str:
        """
        Generate comprehensive case study report.

        Args:
            evolution_results: List of evolution analysis results
            performance_evolution: List of performance metrics over time
            final_comparison: Final strategy comparison results
            execution_time: Total execution time in seconds

        Returns:
            Markdown formatted report string
        """
        try:
            # Build report sections
            report_sections = []

            # Header and executive summary
            report_sections.append(self._generate_header())
            report_sections.append(self._generate_executive_summary(
                evolution_results, performance_evolution, final_comparison, execution_time
            ))

            # Configuration details
            report_sections.append(self._generate_configuration_section())

            # Evolution analysis results
            if evolution_results:
                report_sections.append(self._generate_evolution_section(evolution_results))

            # Performance analysis
            if performance_evolution:
                report_sections.append(self._generate_performance_section(performance_evolution))

            # Comparative analysis
            if final_comparison:
                report_sections.append(self._generate_comparison_section(final_comparison))

            # Visualizations and outputs
            report_sections.append(self._generate_outputs_section())

            # Methodology and appendices
            report_sections.append(self._generate_methodology_section())
            report_sections.append(self._generate_technical_appendix())

            # Combine all sections
            full_report = "\n\n".join(report_sections)

            return full_report

        except Exception as e:
            raise ReportGenerationError(f"Failed to generate case study report: {str(e)}")

    def _generate_header(self) -> str:
        """Generate report header with metadata."""
        return f"""# Case Study Report: {self.case_study_config.ticker}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Type**: Comprehensive Market Regime Case Study
**Ticker Symbol**: {self.case_study_config.ticker}
**Analysis Period**: {self.case_study_config.start_date} to {self.case_study_config.end_date}
**Model Configuration**: {self.case_study_config.n_states}-State Hidden Markov Model
**Training Period**: {self.case_study_config.n_training} days

---"""

    def _generate_executive_summary(
        self,
        evolution_results: List[Dict[str, Any]],
        performance_evolution: List[Dict[str, Any]],
        final_comparison: Dict[str, Any],
        execution_time: float
    ) -> str:
        """Generate executive summary section."""
        summary_lines = [
            "## Executive Summary",
            "",
            f"This case study analyzes market regime behavior for **{self.case_study_config.ticker}** using a systematic 4-phase approach:",
            "",
            "1. **Training Phase**: HMM model trained on historical data with temporal isolation",
            f"2. **Evolution Phase**: {len(evolution_results)} evaluation periods analyzed",
            "3. **Visualization Phase**: Static plots and animated regime evolution created",
            "4. **Comparison Phase**: Performance compared against baselines",
            ""
        ]

        # Key findings
        if evolution_results and len(evolution_results) > 0:
            final_result = evolution_results[-1]
            if 'regime_data' in final_result and len(final_result['regime_data']) > 0:
                regime_data = final_result['regime_data']
                current_regime = regime_data['predicted_state'].iloc[-1]
                current_confidence = regime_data.get('confidence', pd.Series([0])).iloc[-1]

                regime_names = ['Bear', 'Sideways', 'Bull', 'Strong Bull'][:self.case_study_config.n_states]
                current_regime_name = regime_names[int(current_regime)] if current_regime < len(regime_names) else f"Regime {current_regime}"

                summary_lines.extend([
                    "### Key Findings",
                    "",
                    f"- **Current Market Regime**: {current_regime_name} (Confidence: {current_confidence:.1%})",
                ])

        # Performance summary
        if final_comparison and 'comparison_summary' in final_comparison:
            summary = final_comparison['comparison_summary']
            if 'strategy_ranking' in summary and len(summary['strategy_ranking']) > 0:
                best_strategy = summary['strategy_ranking'][0]
                summary_lines.extend([
                    f"- **Best Performing Strategy**: {best_strategy[0]} (Sharpe Ratio: {best_strategy[1]:.3f})",
                ])

        # Analysis statistics
        summary_lines.extend([
            f"- **Analysis Execution Time**: {execution_time:.1f} seconds",
            f"- **Data Quality**: {len(evolution_results)} successful evaluation periods",
        ])

        if self.case_study_config.include_technical_indicators:
            summary_lines.append(f"- **Technical Indicators Analyzed**: {len(self.case_study_config.indicators_to_compare)}")

        if self.case_study_config.create_animations:
            summary_lines.append("- **Visualizations**: Static plots and animated GIFs generated")

        return "\n".join(summary_lines)

    def _generate_configuration_section(self) -> str:
        """Generate detailed configuration section."""
        config_info = self.case_study_config.get_summary_info()

        config_lines = [
            "## Configuration Details",
            "",
            "### Analysis Parameters",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| **Ticker Symbol** | {config_info['ticker']} |",
            f"| **Analysis Start** | {config_info['analysis_period']['start']} |",
            f"| **Analysis End** | {config_info['analysis_period']['end']} |",
            f"| **Total Periods** | {config_info['analysis_period']['total_periods']} |",
            f"| **Training Days** | {config_info['training_period']['n_training_days']} |",
            f"| **Data Frequency** | {config_info['configuration']['frequency']} |",
            f"| **Model States** | {config_info['configuration']['n_states']} |",
            "",
            "### Feature Configuration",
            "",
        ]

        # Technical indicators
        if config_info['configuration']['indicators']:
            indicators_str = ", ".join(config_info['configuration']['indicators'])
            config_lines.append(f"- **Technical Indicators**: {indicators_str}")
        else:
            config_lines.append("- **Technical Indicators**: None")

        # Visualization settings
        config_lines.extend([
            f"- **Animations Created**: {config_info['configuration']['include_animations']}",
            f"- **Buy-and-Hold Comparison**: {config_info['configuration']['include_buy_hold']}",
            "",
            "### Output Configuration",
            "",
            f"- **Output Directory**: `{config_info['output_directory']}`",
        ])

        if self.case_study_config.create_animations:
            config_lines.extend([
                f"- **Animation FPS**: {self.case_study_config.animation_fps}",
                f"- **Save Individual Frames**: {self.case_study_config.save_individual_frames}",
            ])

        return "\n".join(config_lines)

    def _generate_evolution_section(self, evolution_results: List[Dict[str, Any]]) -> str:
        """Generate evolution analysis section."""
        evolution_lines = [
            "## Evolution Analysis Results",
            "",
            f"The regime detection model was stepped through {len(evolution_results)} evaluation periods,",
            "maintaining strict temporal isolation to prevent data leakage.",
            ""
        ]

        # Regime distribution analysis
        if evolution_results:
            regime_counts = {}
            total_periods = 0

            for result in evolution_results:
                if 'regime_data' in result and len(result['regime_data']) > 0:
                    regime_data = result['regime_data']
                    current_regime = regime_data['predicted_state'].iloc[-1]
                    regime_counts[current_regime] = regime_counts.get(current_regime, 0) + 1
                    total_periods += 1

            if total_periods > 0:
                evolution_lines.extend([
                    "### Regime Distribution Over Analysis Period",
                    "",
                    "| Regime | Periods | Percentage |",
                    "|--------|---------|------------|",
                ])

                regime_names = ['Bear', 'Sideways', 'Bull', 'Strong Bull'][:self.case_study_config.n_states]
                for regime in range(self.case_study_config.n_states):
                    count = regime_counts.get(regime, 0)
                    percentage = (count / total_periods) * 100 if total_periods > 0 else 0
                    regime_name = regime_names[regime] if regime < len(regime_names) else f"Regime {regime}"
                    evolution_lines.append(f"| {regime_name} | {count} | {percentage:.1f}% |")

                evolution_lines.append("")

        # Temporal progression
        if len(evolution_results) >= 5:
            evolution_lines.extend([
                "### Temporal Progression Analysis",
                "",
                "Key periods during the analysis:",
                "",
            ])

            # Sample key periods
            key_indices = [0, len(evolution_results)//4, len(evolution_results)//2,
                          3*len(evolution_results)//4, -1]

            for i, idx in enumerate(key_indices):
                if idx < len(evolution_results):
                    result = evolution_results[idx]
                    period_name = ["Start", "25%", "50%", "75%", "End"][i]

                    if 'regime_data' in result and len(result['regime_data']) > 0:
                        regime_data = result['regime_data']
                        current_regime = regime_data['predicted_state'].iloc[-1]
                        current_confidence = regime_data.get('confidence', pd.Series([0])).iloc[-1]
                        regime_name = regime_names[int(current_regime)] if current_regime < len(regime_names) else f"Regime {current_regime}"

                        evolution_lines.append(
                            f"- **{period_name} ({result['date']})**: {regime_name} "
                            f"(Confidence: {current_confidence:.1%})"
                        )

        return "\n".join(evolution_lines)

    def _generate_performance_section(self, performance_evolution: List[Dict[str, Any]]) -> str:
        """Generate performance analysis section."""
        if not performance_evolution:
            return "## Performance Analysis\n\nNo performance data available for analysis."

        perf_lines = [
            "## Performance Analysis",
            "",
            f"Performance metrics were calculated across {len(performance_evolution)} periods,",
            "showing how the HMM-based trading strategy evolved over time.",
            ""
        ]

        # Final performance metrics
        if performance_evolution:
            final_perf = performance_evolution[-1]

            perf_lines.extend([
                "### Final Strategy Performance",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| **Total Return** | {final_perf.get('total_return', 0):.2%} |",
                f"| **Annualized Return** | {final_perf.get('annual_return', 0):.2%} |",
                f"| **Annualized Volatility** | {final_perf.get('annual_volatility', 0):.2%} |",
                f"| **Sharpe Ratio** | {final_perf.get('sharpe_ratio', 0):.3f} |",
                f"| **Sortino Ratio** | {final_perf.get('sortino_ratio', 0):.3f} |",
                f"| **Maximum Drawdown** | {final_perf.get('max_drawdown', 0):.2%} |",
                f"| **Win Rate** | {final_perf.get('win_rate', 0):.2%} |",
                "",
                "### Risk Metrics",
                "",
                f"- **Value at Risk (95%)**: {final_perf.get('var_95', 0):.2%}",
                f"- **Value at Risk (99%)**: {final_perf.get('var_99', 0):.2%}",
                f"- **Skewness**: {final_perf.get('skewness', 0):.3f}",
                f"- **Kurtosis**: {final_perf.get('kurtosis', 0):.3f}",
                ""
            ])

        # Performance evolution summary
        if len(performance_evolution) > 1:
            returns = [p.get('total_return', 0) for p in performance_evolution]
            sharpe_ratios = [p.get('sharpe_ratio', 0) for p in performance_evolution]

            perf_lines.extend([
                "### Performance Evolution",
                "",
                f"- **Return Range**: {min(returns):.2%} to {max(returns):.2%}",
                f"- **Sharpe Ratio Range**: {min(sharpe_ratios):.3f} to {max(sharpe_ratios):.3f}",
                f"- **Performance Stability**: {'High' if max(returns) - min(returns) < 0.1 else 'Moderate' if max(returns) - min(returns) < 0.3 else 'Low'}",
                ""
            ])

        return "\n".join(perf_lines)

    def _generate_comparison_section(self, final_comparison: Dict[str, Any]) -> str:
        """Generate strategy comparison section."""
        if not final_comparison or 'individual_results' not in final_comparison:
            return "## Strategy Comparison\n\nNo comparison data available."

        comp_lines = [
            "## Strategy Comparison Analysis",
            "",
            "The HMM regime detection strategy was compared against baseline strategies",
            "to evaluate relative performance and risk characteristics.",
            ""
        ]

        # Strategy rankings
        if 'comparison_summary' in final_comparison:
            summary = final_comparison['comparison_summary']

            if 'strategy_ranking' in summary:
                comp_lines.extend([
                    "### Strategy Rankings (by Sharpe Ratio)",
                    "",
                    "| Rank | Strategy | Sharpe Ratio |",
                    "|------|----------|--------------|",
                ])

                for i, (strategy, sharpe) in enumerate(summary['strategy_ranking']):
                    comp_lines.append(f"| {i+1} | {strategy.replace('_', ' ').title()} | {sharpe:.3f} |")

                comp_lines.append("")

        # Detailed comparison table
        results = final_comparison['individual_results']
        if results:
            comp_lines.extend([
                "### Detailed Performance Comparison",
                "",
                "| Strategy | Total Return | Annual Return | Sharpe | Max DD | Win Rate |",
                "|----------|--------------|---------------|--------|--------|----------|",
            ])

            for strategy_name, metrics in results.items():
                comp_lines.append(
                    f"| {strategy_name.replace('_', ' ').title()} | "
                    f"{metrics.get('total_return', 0):.2%} | "
                    f"{metrics.get('annual_return', 0):.2%} | "
                    f"{metrics.get('sharpe_ratio', 0):.3f} | "
                    f"{metrics.get('max_drawdown', 0):.2%} | "
                    f"{metrics.get('win_rate', 0):.2%} |"
                )

            comp_lines.append("")

        # Key insights
        if 'comparison_summary' in final_comparison:
            summary = final_comparison['comparison_summary']
            comp_lines.extend([
                "### Key Insights",
                ""
            ])

            # Best performers by metric
            if 'best_total_return' in summary:
                best_return = summary['best_total_return']
                comp_lines.append(f"- **Best Total Return**: {best_return['strategy']} ({best_return['value']:.2%})")

            if 'best_sharpe_ratio' in summary:
                best_sharpe = summary['best_sharpe_ratio']
                comp_lines.append(f"- **Best Sharpe Ratio**: {best_sharpe['strategy']} ({best_sharpe['value']:.3f})")

            if 'best_max_drawdown' in summary:
                best_dd = summary['best_max_drawdown']
                comp_lines.append(f"- **Lowest Drawdown**: {best_dd['strategy']} ({best_dd['value']:.2%})")

            # Strategy type analysis
            hmm_strategies = summary.get('hmm_strategies', [])
            if hmm_strategies:
                comp_lines.append(f"- **HMM Strategies Tested**: {len(hmm_strategies)}")

            indicator_strategies = summary.get('indicator_strategies', [])
            if indicator_strategies:
                comp_lines.append(f"- **Technical Indicator Strategies**: {len(indicator_strategies)}")

        return "\n".join(comp_lines)

    def _generate_outputs_section(self) -> str:
        """Generate outputs and files section."""
        output_lines = [
            "## Generated Outputs",
            "",
            "This case study generated the following outputs for further analysis:",
            "",
            "### Visualizations",
            "",
            f"📊 **Static Analysis**: `plots/{self.case_study_config.ticker}_complete_analysis.png`",
        ]

        if self.case_study_config.create_animations:
            output_lines.extend([
                f"🎬 **Regime Evolution**: `animations/{self.case_study_config.ticker}_regime_evolution.gif`",
                f"📈 **Performance Evolution**: `animations/{self.case_study_config.ticker}_performance_evolution.gif`",
            ])

        output_lines.extend([
            "",
            "### Data Files",
            "",
            "📄 **Evolution Results**: `data/evolution_results.json`",
            "📊 **Performance Metrics**: `data/performance_evolution.json`",
            "🔍 **Comparison Analysis**: `data/final_comparison.json`",
            "",
            "### Reports",
            "",
            "📝 **This Report**: `reports/" + f"{self.case_study_config.ticker}_case_study_report.md`",
        ])

        if self.case_study_config.save_individual_frames:
            output_lines.append("🖼️  **Individual Frames**: `frames/frame_*.png`")

        return "\n".join(output_lines)

    def _generate_methodology_section(self) -> str:
        """Generate methodology section."""
        return """## Methodology

This case study follows a systematic 4-phase approach designed to ensure temporal integrity and comprehensive analysis:

### Phase 1: Configuration and Setup
- Load complete dataset spanning training and evaluation periods
- Configure pipeline components for case study requirements
- Initialize temporal controller for proper data isolation

### Phase 2: Model Training
- Train HMM model exclusively on pre-analysis training data
- Establish baseline regime detection capabilities
- Validate model convergence and stability

### Phase 3: Evolution Analysis
- Step through evaluation period day-by-day
- Maintain strict temporal boundaries (no future data leakage)
- Update model incrementally with each new observation
- Record regime predictions and confidence levels

### Phase 4: Comparative Analysis
- Generate technical indicator baselines
- Calculate buy-and-hold benchmark performance
- Compare all strategies using consistent metrics
- Analyze risk-adjusted returns and drawdown characteristics

### Key Principles

1. **Temporal Integrity**: No future data used in any predictions
2. **Reproducibility**: Fixed random seeds and deterministic processing
3. **Comprehensive Evaluation**: Multiple baseline comparisons
4. **Statistical Rigor**: Proper risk adjustment and uncertainty quantification

### Performance Metrics

- **Return Metrics**: Total return, annualized return, excess return
- **Risk Metrics**: Volatility, maximum drawdown, Value at Risk
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Distributional**: Skewness, kurtosis, win rate

All metrics calculated using standard financial industry practices."""

    def _generate_technical_appendix(self) -> str:
        """Generate technical appendix."""
        model_config = self.case_study_config.get_model_config_with_overrides()
        analysis_config = self.case_study_config.get_analysis_config_with_overrides()

        return f"""## Technical Appendix

### Model Configuration

```json
{self._format_config_for_display(model_config)}
```

### Analysis Configuration

```json
{self._format_config_for_display(analysis_config)}
```

### Dependencies

This analysis was performed using the `hidden-regime` package with the following key components:

- **Hidden Markov Model**: Baum-Welch algorithm with {model_config.get('n_states', 3)} states
- **Temporal Controller**: Ensures no data leakage during backtesting
- **Performance Analyzer**: Comprehensive risk and return metrics
- **Visualization Engine**: Static plots and animated GIFs

### Data Sources

- **Price Data**: Yahoo Finance via yfinance package
- **Technical Indicators**: Custom implementations in hidden-regime
- **Frequency**: {self.case_study_config.frequency}

### Reproducibility

To reproduce this analysis:

1. Install hidden-regime package
2. Use the configuration parameters shown above
3. Run with the same random seed ({model_config.get('random_seed', 42)})
4. Ensure data availability for the specified time period

---

*Report generated by hidden-regime case study system*
*For more information: https://github.com/hidden-regime*"""

    def _format_config_for_display(self, config_dict: Dict[str, Any]) -> str:
        """Format configuration dictionary for display."""
        import json
        try:
            return json.dumps(config_dict, indent=2, default=str)
        except Exception:
            return str(config_dict)