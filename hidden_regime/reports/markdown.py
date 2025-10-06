"""
Markdown report generation component for pipeline architecture.

Provides MarkdownReportGenerator that implements ReportComponent interface for
generating comprehensive markdown reports from pipeline analysis results.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..config.report import ReportConfig
from ..pipeline.interfaces import ReportComponent


class MarkdownReportGenerator(ReportComponent):
    """
    Markdown report generator component for pipeline architecture.

    Implements ReportComponent interface to generate comprehensive markdown reports
    from pipeline data, observations, model output, and analysis results.
    """

    def __init__(self, config: ReportConfig):
        """
        Initialize markdown report generator with configuration.

        Args:
            config: ReportConfig with report generation parameters
        """
        self.config = config
        self._last_report = None

    def update(self, **kwargs) -> str:
        """
        Generate report from pipeline results.

        Args:
            **kwargs: Pipeline component outputs (data, observations, model_output, analysis)

        Returns:
            Generated markdown report string
        """
        # Extract component outputs
        data = kwargs.get("data")
        observations = kwargs.get("observations")
        model_output = kwargs.get("model_output")
        analysis = kwargs.get("analysis")

        # Generate report sections
        report_sections = []

        # Title and metadata
        report_sections.append(self._generate_title_section())

        # Summary section
        if self.config.include_summary:
            report_sections.append(self._generate_summary_section(data, analysis))

        # Regime analysis section
        if self.config.include_regime_analysis and analysis is not None:
            report_sections.append(self._generate_regime_analysis_section(analysis))

        # Performance metrics section
        if self.config.include_performance_metrics and model_output is not None:
            report_sections.append(
                self._generate_performance_section(model_output, analysis)
            )

        # Advanced performance analysis (if analysis component supports it)
        if (
            self.config.include_performance_metrics
            and analysis is not None
            and hasattr(analysis, "get_comprehensive_performance_metrics")
        ):
            report_sections.append(
                self._generate_advanced_performance_section(analysis)
            )

        # Risk analysis section
        if self.config.include_risk_analysis and analysis is not None:
            report_sections.append(self._generate_risk_analysis_section(analysis))

        # Trading signals section
        if self.config.include_trading_signals and analysis is not None:
            report_sections.append(self._generate_trading_signals_section(analysis))

        # Data quality section
        if self.config.include_data_quality and data is not None:
            report_sections.append(
                self._generate_data_quality_section(data, observations)
            )

        # Footer
        report_sections.append(self._generate_footer_section())

        # Combine all sections
        full_report = "\n\n".join(report_sections)

        # Save report if output directory specified
        if self.config.output_dir is not None:
            self._save_report(full_report)

        # Store for reference
        self._last_report = full_report

        return full_report

    def _generate_title_section(self) -> str:
        """Generate report title and metadata."""
        title = self.config.title or "Hidden Regime Analysis Report"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""# {title}

**Generated**: {timestamp}  
**Template Style**: {self.config.template_style}

---"""

    def _generate_summary_section(
        self, data: Optional[pd.DataFrame], analysis: Optional[pd.DataFrame]
    ) -> str:
        """Generate executive summary section."""
        summary = ["## Executive Summary"]

        if data is not None:
            summary.append(
                f"**Data Period**: {data.index.min().date()} to {data.index.max().date()}"
            )
            summary.append(f"**Total Observations**: {len(data)}")

        if analysis is not None and len(analysis) > 0:
            current = analysis.iloc[-1]

            current_regime = current.get(
                "regime_name", f"State {current.get('predicted_state', 'Unknown')}"
            )
            confidence = current.get("confidence", 0) * 100
            days_in_regime = current.get("days_in_regime", 0)

            summary.extend(
                [
                    f"**Current Regime**: {current_regime}",
                    f"**Confidence**: {confidence:.1f}%",
                    f"**Days in Current Regime**: {days_in_regime}",
                ]
            )

            # Add expected characteristics if available
            if "expected_return" in current:
                expected_return = current["expected_return"] * 100
                summary.append(f"**Expected Daily Return**: {expected_return:.3f}%")

            if "expected_volatility" in current:
                expected_vol = current["expected_volatility"] * 100
                summary.append(f"**Expected Volatility**: {expected_vol:.3f}%")

        return "\n".join(summary)

    def _generate_regime_analysis_section(self, analysis: pd.DataFrame) -> str:
        """Generate regime analysis section."""
        section = ["## Regime Analysis"]

        if len(analysis) == 0:
            section.append("No regime analysis data available.")
            return "\n".join(section)

        # Regime distribution
        if "regime_name" in analysis.columns:
            regime_counts = analysis["regime_name"].value_counts()
            section.append("### Regime Distribution")
            section.append("| Regime | Occurrences | Percentage |")
            section.append("|--------|-------------|------------|")

            for regime, count in regime_counts.items():
                percentage = (count / len(analysis)) * 100
                section.append(f"| {regime} | {count} | {percentage:.1f}% |")

        # Recent regime transitions
        if "predicted_state" in analysis.columns and len(analysis) > 1:
            transitions = analysis["predicted_state"].diff().dropna()
            regime_changes = (transitions != 0).sum()

            section.append("### Regime Stability")
            section.append(f"- **Total Regime Changes**: {regime_changes}")
            section.append(
                f"- **Average Regime Duration**: {len(analysis) / max(regime_changes, 1):.1f} periods"
            )

        # Current regime details
        if len(analysis) > 0:
            current = analysis.iloc[-1]
            section.append("### Current Regime Details")

            current_regime = current.get("regime_name", "Unknown")
            confidence = current.get("confidence", 0) * 100

            section.extend(
                [
                    f"- **Regime**: {current_regime}",
                    f"- **Confidence**: {confidence:.1f}%",
                ]
            )

            if "days_in_regime" in current:
                section.append(f"- **Duration**: {current['days_in_regime']} periods")

            if "expected_duration" in current:
                section.append(
                    f"- **Expected Total Duration**: {current['expected_duration']:.1f} periods"
                )

        return "\n".join(section)

    def _generate_performance_section(
        self, model_output: pd.DataFrame, analysis: Optional[pd.DataFrame]
    ) -> str:
        """Generate performance metrics section."""
        section = ["## Performance Metrics"]

        if len(model_output) == 0:
            section.append("No performance data available.")
            return "\n".join(section)

        # Model confidence statistics
        if "confidence" in model_output.columns:
            confidence_stats = model_output["confidence"].describe()

            section.append("### Model Confidence")
            section.append("| Metric | Value |")
            section.append("|--------|-------|")
            section.append(f"| Mean | {confidence_stats['mean']:.3f} |")
            section.append(f"| Std Dev | {confidence_stats['std']:.3f} |")
            section.append(f"| Min | {confidence_stats['min']:.3f} |")
            section.append(f"| Max | {confidence_stats['max']:.3f} |")

        # State prediction distribution
        if "predicted_state" in model_output.columns:
            state_counts = model_output["predicted_state"].value_counts().sort_index()

            section.append("### State Predictions")
            section.append("| State | Count | Percentage |")
            section.append("|-------|-------|------------|")

            for state, count in state_counts.items():
                percentage = (count / len(model_output)) * 100
                section.append(f"| {state} | {count} | {percentage:.1f}% |")

        return "\n".join(section)

    def _generate_advanced_performance_section(self, analysis) -> str:
        """Generate advanced performance analysis section using comprehensive metrics."""
        section = ["## Advanced Performance Analysis"]

        try:
            # Get comprehensive performance metrics
            if hasattr(analysis, "get_comprehensive_performance_metrics"):
                performance_metrics = analysis.get_comprehensive_performance_metrics()

                if "error" in performance_metrics:
                    section.append(
                        f"Performance analysis unavailable: {performance_metrics['error']}"
                    )
                    return "\n".join(section)

                # Summary section
                if "summary" in performance_metrics:
                    summary = performance_metrics["summary"]
                    section.append("### Performance Summary")
                    section.append("| Metric | Value |")
                    section.append("|--------|-------|")
                    section.append(
                        f"| Overall Quality | {summary.get('overall_quality', 'Unknown')} |"
                    )
                    section.append(
                        f"| Quality Score | {summary.get('quality_score', 0)}/100 |"
                    )
                    section.append(
                        f"| Average Confidence | {summary.get('average_confidence', 0):.1%} |"
                    )
                    section.append(
                        f"| Balance Score | {summary.get('balance_score', 0):.3f} |"
                    )
                    section.append(
                        f"| Stability Rating | {summary.get('stability_rating', 'Unknown')} |"
                    )

                # Regime Distribution
                if "regime_distribution" in performance_metrics:
                    dist = performance_metrics["regime_distribution"]
                    section.append("### Detailed Regime Distribution")

                    if "regime_name_percentages" in dist:
                        section.append("| Regime | Percentage | Periods |")
                        section.append("|--------|------------|---------|")
                        for regime, pct in dist["regime_name_percentages"].items():
                            count = dist.get("regime_name_counts", {}).get(regime, 0)
                            section.append(f"| {regime} | {pct:.1f}% | {count} |")

                    if "regime_dominance" in dist:
                        dominance = dist["regime_dominance"]
                        section.append("#### Regime Balance Analysis")
                        section.append(
                            f"- **Most Frequent**: {dominance.get('most_frequent_state', 'Unknown')} ({dominance.get('dominance_percentage', 0):.1f}%)"
                        )
                        section.append(
                            f"- **Market Balance**: {'Dominated' if dominance.get('is_dominated', False) else 'Balanced'}"
                        )
                        section.append(
                            f"- **Balance Score**: {dominance.get('balance_score', 0):.3f} (higher = more balanced)"
                        )

                # Transition Analysis
                if "transition_analysis" in performance_metrics:
                    trans = performance_metrics["transition_analysis"]
                    section.append("### Regime Transition Analysis")
                    section.append(
                        f"- **Total Transitions**: {trans.get('total_transitions', 0)}"
                    )
                    section.append(
                        f"- **Transition Rate**: {trans.get('transition_rate', 0):.1%} per period"
                    )
                    section.append(
                        f"- **Average Persistence**: {trans.get('average_persistence', 0):.1%}"
                    )
                    section.append(
                        f"- **Stability Score**: {trans.get('stability_score', 0):.3f}"
                    )

                    if "persistence_by_state" in trans:
                        section.append("#### Regime Persistence")
                        section.append("| State | Persistence |")
                        section.append("|-------|-------------|")
                        for state, persistence in trans["persistence_by_state"].items():
                            section.append(f"| {state} | {persistence:.1%} |")

                # Duration Analysis
                if "duration_analysis" in performance_metrics:
                    duration = performance_metrics["duration_analysis"]
                    section.append("### Regime Duration Analysis")

                    if "duration_stats_by_state" in duration:
                        section.append(
                            "| State | Avg Duration | Episodes | Time Share |"
                        )
                        section.append(
                            "|-------|-------------|----------|------------|"
                        )
                        for state, stats in duration["duration_stats_by_state"].items():
                            section.append(
                                f"| {state} | {stats['mean_duration']:.1f} | {stats['count']} | {stats['percentage_of_time']:.1f}% |"
                            )

                    overall = duration.get("overall_duration_stats", {})
                    section.append("#### Overall Duration Statistics")
                    section.append(
                        f"- **Mean Episode Duration**: {overall.get('mean', 0):.1f} periods"
                    )
                    section.append(
                        f"- **Median Episode Duration**: {overall.get('median', 0):.1f} periods"
                    )
                    section.append(
                        f"- **Longest Episode**: {overall.get('max', 0)} periods"
                    )

                # Confidence Analysis
                if "confidence_analysis" in performance_metrics:
                    conf = performance_metrics["confidence_analysis"]
                    section.append("### Model Confidence Analysis")

                    if "confidence_quality" in conf:
                        quality = conf["confidence_quality"]
                        section.append("| Confidence Level | Percentage |")
                        section.append("|------------------|------------|")
                        section.append(
                            f"| High (>70%) | {quality.get('high_confidence_pct', 0):.1f}% |"
                        )
                        section.append(
                            f"| Medium (50-70%) | {quality.get('medium_confidence_pct', 0):.1f}% |"
                        )
                        section.append(
                            f"| Low (<50%) | {quality.get('low_confidence_pct', 0):.1f}% |"
                        )
                        section.append(
                            f"- **Confidence Stability**: {quality.get('confidence_stability', 0):.3f}"
                        )

                # Return Performance (if available)
                if "regime_performance" in performance_metrics:
                    perf = performance_metrics["regime_performance"]
                    if "regime_performance" in perf:
                        section.append("### Return Performance by Regime")
                        section.append(
                            "| Regime | Avg Return | Volatility | Sharpe | Win Rate |"
                        )
                        section.append(
                            "|--------|------------|------------|--------|----------|"
                        )
                        for state, stats in perf["regime_performance"].items():
                            section.append(
                                f"| {state} | {stats['mean_return']:.2%} | {stats['std_return']:.2%} | {stats['sharpe_ratio']:.2f} | {stats['positive_days_pct']:.1f}% |"
                            )

                        if "performance_summary" in perf:
                            summary = perf["performance_summary"]
                            section.append("#### Performance Highlights")
                            section.append(
                                f"- **Best Performer**: {summary.get('best_performing_regime', 'Unknown')}"
                            )
                            section.append(
                                f"- **Most Volatile**: {summary.get('most_volatile_regime', 'Unknown')}"
                            )
                            section.append(
                                f"- **Least Volatile**: {summary.get('least_volatile_regime', 'Unknown')}"
                            )

        except Exception as e:
            section.append(f"Advanced performance analysis failed: {str(e)}")

        return "\n".join(section)

    def _generate_risk_analysis_section(self, analysis: pd.DataFrame) -> str:
        """Generate risk analysis section."""
        section = ["## Risk Analysis"]

        if len(analysis) == 0:
            section.append("No risk analysis data available.")
            return "\n".join(section)

        # Risk by regime type
        if (
            "regime_type" in analysis.columns
            and "expected_volatility" in analysis.columns
        ):
            risk_by_regime = analysis.groupby("regime_type")["expected_volatility"].agg(
                ["mean", "std", "count"]
            )

            section.append("### Risk by Regime Type")
            section.append("| Regime | Avg Volatility | Std Dev | Observations |")
            section.append("|--------|---------------|---------|--------------|")

            for regime, stats in risk_by_regime.iterrows():
                avg_vol = stats["mean"] * 100
                std_vol = stats["std"] * 100 if not pd.isna(stats["std"]) else 0
                count = int(stats["count"])
                section.append(
                    f"| {regime} | {avg_vol:.2f}% | {std_vol:.2f}% | {count} |"
                )

        # Current risk assessment
        if len(analysis) > 0:
            current = analysis.iloc[-1]

            section.append("### Current Risk Assessment")

            regime_type = current.get("regime_type", "Unknown")
            expected_vol = current.get("expected_volatility", 0) * 100

            section.extend(
                [
                    f"- **Current Regime Type**: {regime_type}",
                    f"- **Expected Volatility**: {expected_vol:.2f}%",
                ]
            )

            # Risk classification
            if expected_vol < 1.0:
                risk_level = "Low"
            elif expected_vol < 2.5:
                risk_level = "Moderate"
            elif expected_vol < 4.0:
                risk_level = "High"
            else:
                risk_level = "Very High"

            section.append(f"- **Risk Level**: {risk_level}")

        return "\n".join(section)

    def _generate_trading_signals_section(self, analysis: pd.DataFrame) -> str:
        """Generate trading signals section."""
        section = ["## Trading Signals"]

        if "position_signal" not in analysis.columns:
            section.append("No trading signals generated.")
            return "\n".join(section)

        current_signal = (
            analysis.iloc[-1]["position_signal"] if len(analysis) > 0 else 0
        )
        signal_strength = abs(current_signal)

        section.append("### Current Position Recommendation")

        if current_signal > 0.5:
            recommendation = "Strong Long"
        elif current_signal > 0.1:
            recommendation = "Moderate Long"
        elif current_signal > -0.1:
            recommendation = "Neutral/Cash"
        elif current_signal > -0.5:
            recommendation = "Moderate Short"
        else:
            recommendation = "Strong Short"

        section.extend(
            [
                f"- **Position Signal**: {current_signal:.3f}",
                f"- **Signal Strength**: {signal_strength:.3f}",
                f"- **Recommendation**: {recommendation}",
            ]
        )

        # Signal statistics
        signal_stats = analysis["position_signal"].describe()

        section.append("### Signal Statistics")
        section.append("| Metric | Value |")
        section.append("|--------|-------|")
        section.append(f"| Mean | {signal_stats['mean']:.3f} |")
        section.append(f"| Std Dev | {signal_stats['std']:.3f} |")
        section.append(f"| Min | {signal_stats['min']:.3f} |")
        section.append(f"| Max | {signal_stats['max']:.3f} |")

        return "\n".join(section)

    def _generate_data_quality_section(
        self, data: pd.DataFrame, observations: Optional[pd.DataFrame]
    ) -> str:
        """Generate data quality section."""
        section = ["## Data Quality Assessment"]

        if data is None or len(data) == 0:
            section.append("No data available for quality assessment.")
            return "\n".join(section)

        # Basic data statistics
        section.append("### Data Overview")
        section.append(f"- **Total Observations**: {len(data)}")
        section.append(
            f"- **Date Range**: {data.index.min().date()} to {data.index.max().date()}"
        )
        section.append(f"- **Data Columns**: {', '.join(data.columns)}")

        # Missing data analysis
        missing_data = data.isnull().sum()
        if missing_data.any():
            section.append("### Missing Data")
            section.append("| Column | Missing Count | Percentage |")
            section.append("|--------|---------------|------------|")

            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    percentage = (missing_count / len(data)) * 100
                    section.append(f"| {col} | {missing_count} | {percentage:.1f}% |")
        else:
            section.append("### Missing Data")
            section.append(" No missing data detected.")

        # Observation quality (if available)
        if observations is not None and len(observations) > 0:
            section.append("### Observation Quality")
            obs_missing = observations.isnull().sum()

            if obs_missing.any():
                section.append("| Observation | Missing Count |")
                section.append("|-------------|---------------|")

                for col, missing_count in obs_missing.items():
                    if missing_count > 0:
                        section.append(f"| {col} | {missing_count} |")
            else:
                section.append(" All observations computed successfully.")

        return "\n".join(section)

    def _generate_footer_section(self) -> str:
        """Generate report footer."""
        return f"""---

*Report generated by Hidden Regime Analysis Pipeline*  
*Template: {self.config.template_style}*  
*Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*"""

    def _save_report(self, report_content: str) -> None:
        """Save report to configured output directory."""
        try:
            output_dir = self.config.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)

            filename = self.config.get_report_filename()

            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)

        except Exception as e:
            # Log error but don't fail report generation
            print(f"Warning: Failed to save report to file: {e}")

    def plot(self, **kwargs) -> plt.Figure:
        """Generate visualization for report component."""
        fig, ax = plt.subplots(figsize=(8, 6))

        if self._last_report is not None:
            # Show report statistics
            lines = self._last_report.split("\n")
            sections = [line for line in lines if line.startswith("##")]

            ax.text(
                0.5,
                0.7,
                f"Report Generated Successfully",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
            )
            ax.text(
                0.5,
                0.5,
                f"Sections: {len(sections)}",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.text(
                0.5,
                0.3,
                f"Total Lines: {len(lines)}",
                ha="center",
                va="center",
                fontsize=12,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No report generated yet",
                ha="center",
                va="center",
                fontsize=14,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Report Generation Status")

        return fig
