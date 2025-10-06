#!/usr/bin/env python3
"""
Case Study Orchestrator Script

Implements the comprehensive case study system as specified in case_study.md.
Performs the complete 4-phase workflow:
1. Training: Train HMM model on n_training days before start_date
2. Evolution: Step through evaluation period with daily updates
3. Visualization: Generate evolving charts and animations
4. Analysis: Compare against buy-and-hold and technical indicators

This script serves as the main interface for running case studies with proper
temporal isolation and comprehensive performance analysis.
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hidden_regime.analysis.case_study import CaseStudyAnalyzer

# Import hidden-regime components
from hidden_regime.config.case_study import CaseStudyConfig
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.config.simulation import SimulationConfig
from hidden_regime.data.financial import FinancialDataLoader
from hidden_regime.factories.pipeline import pipeline_factory
from hidden_regime.pipeline.temporal import TemporalController
from hidden_regime.simulation.simulation_orchestrator import SimulationOrchestrator
from hidden_regime.visualization.animations import (
    RegimeAnimator,
    create_regime_comparison_gif,
)
from hidden_regime.visualization.plotting import create_multi_panel_regime_plot


class CaseStudyOrchestrator:
    """
    Main orchestrator for comprehensive case study analysis.

    Manages the complete workflow from configuration to final analysis,
    ensuring temporal integrity and generating comprehensive outputs.
    """

    def __init__(self, config: CaseStudyConfig):
        """
        Initialize case study orchestrator.

        Args:
            config: Case study configuration
        """
        self.config = config
        self.case_study_analyzer = CaseStudyAnalyzer()
        self.regime_animator = RegimeAnimator(
            color_scheme=config.color_scheme, style=config.plot_style
        )

        # Results storage
        self.full_dataset = None
        self.pipeline = None
        self.temporal_controller = None
        self.evolution_results = []
        self.performance_evolution = []
        self.final_comparison = None
        self.simulation_results = None

        # Create simulation configuration if simulation is enabled
        self.simulation_config = None
        if config.enable_simulation:
            self.simulation_config = self._create_simulation_config()

    def run_complete_case_study(self) -> Dict[str, Any]:
        """
        Execute the complete case study workflow.

        Returns:
            Dictionary with complete case study results
        """
        print(f" Starting Case Study: {self.config.ticker}")
        print("=" * 60)

        start_time = datetime.now()

        try:
            # Phase 1: Configuration and Data Loading
            print("\n📋 Phase 1: Configuration and Data Setup")
            self._setup_data_and_pipeline()

            # Phase 2: Training Period
            print("\n🎓 Phase 2: Model Training")
            self._train_initial_model()

            # Phase 3: Evolution Analysis
            print("\n Phase 3: Evolution Analysis")
            self._run_evolution_analysis()

            # Phase 4: Visualization Generation
            print("\n🎨 Phase 4: Visualization Generation")
            self._generate_visualizations()

            # Phase 5: Trading Simulation (if enabled)
            if self.config.enable_simulation:
                print("\n💰 Phase 5: Trading Simulation")
                self._run_trading_simulation()

            # Phase 6: Comparative Analysis
            print("\n Phase 6: Comparative Analysis")
            self._run_comparative_analysis()

            # Phase 7: Report Generation
            print("\n📝 Phase 7: Report Generation")
            final_report = self._generate_final_report()

            total_time = (datetime.now() - start_time).total_seconds()
            print(f"\n Case Study Complete! ({total_time:.1f} seconds)")

            return {
                "config": self.config,
                "evolution_results": self.evolution_results,
                "performance_evolution": self.performance_evolution,
                "final_comparison": self.final_comparison,
                "simulation_results": self.simulation_results,
                "final_report": final_report,
                "execution_time": total_time,
            }

        except Exception as e:
            print(f"\n Case Study Failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _setup_data_and_pipeline(self):
        """Phase 1: Setup data loading and pipeline configuration."""
        print("   Loading market data...")

        # Create output directory structure
        output_structure = self.config.create_output_structure()
        print(f"   Created output structure at: {self.config.output_directory}")

        # Load complete dataset (training + evaluation period)
        training_start, training_end = self.config.get_training_date_range()

        # Map case study frequency to data config frequency
        frequency_mapping = {
            "business_days": "days",
            "daily": "days",
            "hourly": "hours",
        }

        data_config = FinancialDataConfig(
            ticker=self.config.ticker,
            start_date=training_start,
            end_date=self.config.end_date,
            frequency=frequency_mapping.get(self.config.frequency, "days"),
        )

        data_loader = FinancialDataLoader(data_config)
        self.full_dataset = data_loader.update()

        print(
            f"   Loaded {len(self.full_dataset)} days of data ({training_start} to {self.config.end_date})"
        )

        # Assess data for regime detection feasibility
        self._assess_data_for_regime_detection()

        # Create pipeline for case study
        model_overrides = self.config.get_model_config_with_overrides()
        # Remove n_states from overrides to avoid conflict
        model_overrides.pop("n_states", None)

        # Prepare analysis config overrides for regime label handling
        analysis_config_overrides = {}
        if self.config.force_regime_labels is not None:
            analysis_config_overrides["force_regime_labels"] = (
                self.config.force_regime_labels
            )
            analysis_config_overrides["acknowledge_override"] = (
                self.config.acknowledge_override
            )

        self.pipeline = pipeline_factory.create_financial_pipeline(
            ticker=self.config.ticker,
            n_states=self.config.n_states,
            include_report=False,  # We'll generate our own reports
            model_config_overrides=model_overrides,
            analysis_config_overrides=analysis_config_overrides,
        )

        # Create temporal controller for proper isolation
        self.temporal_controller = TemporalController(self.pipeline, self.full_dataset)

        print("    Data and pipeline setup complete")

    def _assess_data_for_regime_detection(self):
        """Assess data quality and regime detection feasibility."""
        from hidden_regime.analysis.financial import FinancialAnalysis

        print("   Assessing data for regime detection feasibility...")

        try:
            # Perform data assessment
            assessment = FinancialAnalysis.assess_data_for_regime_detection(
                self.full_dataset, observed_signal="log_return"
            )

            # Store assessment for later reporting
            self.data_assessment = assessment

            # Check for critical warnings
            recommendations = assessment["recommendations"]

            if not recommendations["suitable_for_regime_detection"]:
                print(f"   [WARNING]  WARNING: Data may not be suitable for regime detection")
                for warning in recommendations["warnings"]:
                    print(f"       • {warning}")

                # Check if user wants to proceed
                if recommendations["recommended_n_states"] != self.config.n_states:
                    print(
                        f"   Note: RECOMMENDATION: Use {recommendations['recommended_n_states']} states instead of {self.config.n_states}"
                    )
                    print(
                        f"       Approach: {recommendations['regime_detection_approach']}"
                    )

            elif recommendations["warnings"]:
                print(f"   [WARNING]  Data assessment warnings:")
                for warning in recommendations["warnings"]:
                    print(f"       • {warning}")

            else:
                print(f"    Data assessment: Suitable for regime detection")
                print(
                    f"       Recommended states: {recommendations['recommended_n_states']}"
                )

                # Show positive indicators
                for rationale in recommendations["rationale"]:
                    print(f"       • {rationale}")

        except Exception as e:
            print(f"   [WARNING]  Data assessment failed: {e}")
            print(f"       Proceeding with standard configuration...")
            self.data_assessment = None

    def _train_initial_model(self):
        """Phase 2: Train the initial model on training data."""
        print("   Training initial HMM model...")

        training_start, training_end = self.config.get_training_date_range()

        # Train model using only training data
        try:
            training_result = self.temporal_controller.update_as_of(training_end)
            print(f"   Model trained on data through {training_end}")
            print(f"    Initial model training complete")
        except Exception as e:
            print(f"   [WARNING]  Training warning: {e}")
            # Continue with available data

    def _run_evolution_analysis(self):
        """Phase 3: Step through evaluation period with temporal isolation."""
        print("   Running temporal evolution analysis...")

        evaluation_dates = self.config.get_evaluation_dates()
        total_dates = len(evaluation_dates)

        print(f"   Analyzing {total_dates} evaluation dates...")

        # Step through each evaluation date
        for i, eval_date in enumerate(evaluation_dates):
            if i % 10 == 0 or i == total_dates - 1:
                print(f"   Progress: {i+1}/{total_dates} ({(i+1)/total_dates:.1%})")

            try:
                # Update model with data up to eval_date (maintaining temporal isolation)
                report = self.temporal_controller.update_as_of(eval_date)

                # Check if pipeline has valid outputs
                if (
                    hasattr(self.temporal_controller.pipeline, "data_output")
                    and hasattr(self.temporal_controller.pipeline, "analysis_output")
                    and hasattr(self.temporal_controller.pipeline, "model_output")
                ):

                    # Store results
                    pipeline_outputs = {
                        "date": eval_date,
                        "data": self.temporal_controller.pipeline.data_output.copy(),
                        "regime_data": self.temporal_controller.pipeline.analysis_output.copy(),
                        "model_output": self.temporal_controller.pipeline.model_output.copy(),
                    }

                    self.evolution_results.append(pipeline_outputs)

                    # Debug logging
                    print(
                        f"    Stored evolution result for {eval_date} (total: {len(self.evolution_results)})"
                    )

                    # Calculate evolving performance
                    if (
                        len(self.evolution_results) >= 30
                    ):  # Need minimum data for performance calc
                        current_price_data = pipeline_outputs["data"]
                        current_regime_data = pipeline_outputs["regime_data"]

                        try:
                            perf_metrics = self.case_study_analyzer.analyze_hmm_strategy_performance(
                                current_price_data, current_regime_data
                            )
                            perf_metrics["evaluation_date"] = eval_date
                            self.performance_evolution.append(perf_metrics)
                            print(f"    Performance calculated for {eval_date}")
                        except Exception as e:
                            print(
                                f"   [WARNING]  Performance calc warning for {eval_date}: {e}"
                            )
                else:
                    print(f"   [WARNING]  Pipeline outputs missing for {eval_date}")

            except Exception as e:
                print(f"   [WARNING]  Evolution warning for {eval_date}: {e}")
                import traceback

                print(f"   📝 Full error trace: {traceback.format_exc()}")
                continue

        print(
            f"    Evolution analysis complete ({len(self.evolution_results)} results)"
        )

        # Save evolution data to JSON
        self._save_evolution_data()

    def _save_evolution_data(self):
        """Save evolution results and performance data to JSON files."""
        import json
        from datetime import datetime

        def make_serializable(obj):
            """Convert pandas/numpy objects to serializable format."""
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict("records")
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, "isoformat"):  # datetime objects
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj

        try:
            # Prepare evolution results for serialization
            serializable_evolution = []
            for result in self.evolution_results:
                serializable_result = {}
                for key, value in result.items():
                    serializable_result[key] = make_serializable(value)
                serializable_evolution.append(serializable_result)

            # Save evolution results
            if serializable_evolution:
                evolution_path = os.path.join(
                    self.config.output_directory, "data", "evolution_results.json"
                )
                with open(evolution_path, "w") as f:
                    json.dump(serializable_evolution, f, indent=2, default=str)
                print(f"   💾 Evolution data saved: {evolution_path}")

            # Save performance evolution
            if self.performance_evolution:
                serializable_performance = []
                for perf in self.performance_evolution:
                    serializable_perf = {}
                    for key, value in perf.items():
                        serializable_perf[key] = make_serializable(value)
                    serializable_performance.append(serializable_perf)

                performance_path = os.path.join(
                    self.config.output_directory, "data", "performance_evolution.json"
                )
                with open(performance_path, "w") as f:
                    json.dump(serializable_performance, f, indent=2, default=str)
                print(f"   💾 Performance data saved: {performance_path}")

        except Exception as e:
            print(f"   [WARNING]  Data save warning: {e}")

    def _generate_visualizations(self):
        """Phase 4: Generate visualizations and animations."""
        if not self.evolution_results:
            print("   [WARNING]  No evolution results to visualize")
            return

        print("   Generating visualizations...")

        # Extract data for visualizations
        regime_data_sequence = [
            result["regime_data"] for result in self.evolution_results
        ]
        evaluation_dates = [result["date"] for result in self.evolution_results]

        # Get final data for static plots
        final_data = self.evolution_results[-1]["data"]
        final_regime_data = self.evolution_results[-1]["regime_data"]

        # Generate static comprehensive plot
        try:
            static_fig = create_multi_panel_regime_plot(
                final_data,
                final_regime_data,
                title=f"{self.config.ticker} - Complete Regime Analysis",
            )

            static_path = os.path.join(
                self.config.output_directory,
                "plots",
                f"{self.config.ticker}_complete_analysis.png",
            )
            static_fig.savefig(static_path, dpi=300, bbox_inches="tight")
            plt.close(static_fig)
            print(f"    Static analysis saved: {static_path}")

        except Exception as e:
            print(f"   [WARNING]  Static plot warning: {e}")

        # Generate animations if requested
        if self.config.create_animations and len(regime_data_sequence) > 3:
            try:
                print("   Creating regime evolution animation...")

                # Create regime evolution GIF
                regime_anim = self.regime_animator.create_evolving_regime_animation(
                    final_data,  # Use final_data instead of self.full_dataset
                    regime_data_sequence,
                    evaluation_dates,
                    title=f"{self.config.ticker} Regime Evolution",
                    save_path=os.path.join(
                        self.config.output_directory,
                        "animations",
                        f"{self.config.ticker}_regime_evolution.gif",
                    ),
                    fps=self.config.animation_fps,
                )

                print(f"   🎬 Regime evolution GIF created")

                # Create performance evolution animation if we have performance data
                if self.performance_evolution:
                    perf_anim = (
                        self.regime_animator.create_performance_evolution_animation(
                            self.performance_evolution,
                            [p["evaluation_date"] for p in self.performance_evolution],
                            title=f"{self.config.ticker} Performance Evolution",
                            save_path=os.path.join(
                                self.config.output_directory,
                                "animations",
                                f"{self.config.ticker}_performance_evolution.gif",
                            ),
                            fps=self.config.animation_fps,
                        )
                    )
                    print(f"    Performance evolution GIF created")

            except Exception as e:
                print(f"   [WARNING]  Animation warning: {e}")
                # Animation failure is not critical - continue with the rest of the analysis

        print("    Visualization generation complete")

    def _create_simulation_config(self) -> SimulationConfig:
        """Create simulation configuration from case study settings."""
        # Calculate initial capital if not specified
        initial_capital = self.config.simulation_initial_capital
        if initial_capital is None:
            # Use default calculation: 100 shares * estimated first price
            initial_capital = (
                100 * 100
            )  # Conservative estimate, will be adjusted in simulation

        return SimulationConfig(
            initial_capital=initial_capital,
            default_shares=100,
            transaction_cost=self.config.simulation_transaction_cost,
            enable_shorting=True,
            max_position_pct=self.config.simulation_max_position_pct,
            stop_loss_pct=self.config.simulation_stop_loss_pct,
            signal_generators=["buy_and_hold", "hmm_regime_following"],
            hmm_strategy_types=["regime_following"],
            technical_indicators=(
                ["rsi", "macd", "sma"]
                if self.config.simulation_include_technical_indicators
                else []
            ),
            save_trade_journal=True,
            save_daily_signals=True,
            save_portfolio_history=True,
        )

    def _run_trading_simulation(self):
        """Run capital-based trading simulation."""
        if not self.evolution_results or not self.simulation_config:
            print(
                "   [WARNING]  No evolution results or simulation config for trading simulation"
            )
            return

        print("   Running capital-based trading simulation...")

        try:
            # Get final price and regime data for simulation
            final_price_data = self.evolution_results[-1]["data"]
            final_regime_data = self.evolution_results[-1]["regime_data"]

            # Initialize simulation orchestrator
            simulation_orchestrator = SimulationOrchestrator(self.simulation_config)

            # Run simulation
            self.simulation_results = simulation_orchestrator.run_simulation(
                price_data=final_price_data,
                regime_data=final_regime_data,
                symbol=self.config.ticker,
            )

            # Print simulation summary
            if self.simulation_results.simulation_success:
                print(f"   💰 Simulation completed successfully!")
                print(
                    f"    Total Return: {self.simulation_results.total_return_pct:.2f}%"
                )
                print(f"    Sharpe Ratio: {self.simulation_results.sharpe_ratio:.3f}")
                print(f"   🏆 Best Strategy: {self.simulation_results.best_strategy}")
                print(f"   💼 Final Value: ${self.simulation_results.final_value:,.2f}")

                if self.simulation_results.benchmark_comparison:
                    benchmark = self.simulation_results.benchmark_comparison
                    print(
                        f"    vs {benchmark['benchmark_strategy']}: +{benchmark['excess_return']:.2f}% excess return"
                    )
            else:
                print(f"    Simulation failed")

        except Exception as e:
            print(f"   [WARNING]  Trading simulation warning: {e}")
            import traceback

            print(f"   📝 Simulation error trace: {traceback.format_exc()}")

    def _run_comparative_analysis(self):
        """Phase 5: Compare HMM strategy against baselines with proper temporal isolation."""
        if not self.evolution_results:
            print("   [WARNING]  No evolution results for comparison")
            return

        print("   Running comparative performance analysis...")

        # Use temporal evolution data for proper comparison
        # This ensures all strategies see the same data progression as the HMM model

        # Get the price data that matches the HMM analysis period
        final_price_data = self.evolution_results[-1]["data"]
        final_regime_data = self.evolution_results[-1]["regime_data"]

        # Remove the legacy technical indicator generation that causes look-ahead bias
        # The comprehensive TA analysis in compare_all_strategies handles temporal isolation properly
        indicators = {}

        # Run comprehensive comparison with temporal isolation
        try:
            self.final_comparison = self.case_study_analyzer.compare_all_strategies(
                final_price_data,
                final_regime_data,
                indicators,  # Empty - use comprehensive TA analysis instead
                hmm_strategy_types=["regime_following"],
                n_best_indicators=5,
            )

            print("    Comparative analysis complete")

            # Print summary
            if "comparison_summary" in self.final_comparison:
                summary = self.final_comparison["comparison_summary"]
                if "strategy_ranking" in summary:
                    print(
                        f"    Top strategy: {summary['strategy_ranking'][0][0]} "
                        + f"(Sharpe: {summary['strategy_ranking'][0][1]:.3f})"
                    )

        except Exception as e:
            print(f"   [WARNING]  Comparison analysis warning: {e}")
            self.final_comparison = {"error": str(e)}

    def _generate_final_report(self) -> str:
        """Phase 6: Generate comprehensive markdown report."""
        print("   Generating final case study report...")

        # Generate performance report
        report_content = []

        # Header
        report_content.extend(
            [
                f"# Case Study Report: {self.config.ticker}",
                "",
                f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Ticker**: {self.config.ticker}",
                f"**Analysis Period**: {self.config.start_date} to {self.config.end_date}",
                f"**Training Period**: {self.config.n_training} days",
                f"**Model States**: {self.config.n_states}",
                "",
            ]
        )

        # Configuration summary
        config_summary = self.config.get_summary_info()
        report_content.extend(
            [
                "## Configuration Summary",
                "",
                f"- **Analysis Period**: {config_summary['analysis_period']['total_periods']} periods",
                f"- **Training Days**: {config_summary['training_period']['n_training_days']}",
                f"- **Frequency**: {config_summary['configuration']['frequency']}",
                f"- **Technical Indicators**: {config_summary['configuration']['indicators']}",
                f"- **Animations Created**: {config_summary['configuration']['include_animations']}",
                f"- **Trading Simulation**: {'Enabled' if self.config.enable_simulation else 'Disabled'}",
                "",
            ]
        )

        # Evolution results summary
        if self.evolution_results:
            report_content.extend(
                [
                    "## Evolution Analysis Results",
                    "",
                    f"- **Total Evaluation Dates**: {len(self.evolution_results)}",
                    f"- **First Evaluation**: {self.evolution_results[0]['date']}",
                    f"- **Last Evaluation**: {self.evolution_results[-1]['date']}",
                    "",
                ]
            )

            # Final regime state
            if "regime_data" in self.evolution_results[-1]:
                final_regime_data = self.evolution_results[-1]["regime_data"]
                if len(final_regime_data) > 0:
                    current_regime = final_regime_data["predicted_state"].iloc[-1]
                    current_confidence = final_regime_data.get(
                        "confidence", pd.Series([0])
                    ).iloc[-1]

                    regime_names = ["Bear", "Sideways", "Bull", "Strong Bull"][
                        : self.config.n_states
                    ]
                    current_regime_name = (
                        regime_names[int(current_regime)]
                        if current_regime < len(regime_names)
                        else f"Regime {current_regime}"
                    )

                    report_content.extend(
                        [
                            "### Current Market State",
                            "",
                            f"- **Current Regime**: {current_regime_name}",
                            f"- **Confidence**: {current_confidence:.1%}",
                            "",
                        ]
                    )

        # Simulation results
        if self.simulation_results and self.simulation_results.simulation_success:
            report_content.extend(
                [
                    "## Trading Simulation Results",
                    "",
                    "### Capital-Based Performance",
                    "",
                    f"- **Initial Capital**: ${self.simulation_results.initial_capital:,.2f}",
                    f"- **Final Value**: ${self.simulation_results.final_value:,.2f}",
                    f"- **Total Return**: {self.simulation_results.total_return_pct:.2f}%",
                    f"- **Annualized Return**: {self.simulation_results.annualized_return:.2f}%",
                    f"- **Volatility**: {self.simulation_results.volatility:.2f}%",
                    f"- **Sharpe Ratio**: {self.simulation_results.sharpe_ratio:.3f}",
                    f"- **Sortino Ratio**: {self.simulation_results.sortino_ratio:.3f}",
                    f"- **Maximum Drawdown**: {self.simulation_results.max_drawdown_pct:.2f}%",
                    "",
                    "### Trading Activity",
                    "",
                    f"- **Total Trades**: {self.simulation_results.total_trades}",
                    f"- **Win Rate**: {self.simulation_results.win_rate:.1f}%",
                    f"- **Profit Factor**: {self.simulation_results.profit_factor:.2f}",
                    f"- **Average Trade P&L**: ${self.simulation_results.avg_trade_pnl:.2f}",
                    "",
                    "### Strategy Comparison",
                    "",
                    f"- **Best Strategy**: {self.simulation_results.best_strategy}",
                ]
            )

            if self.simulation_results.benchmark_comparison:
                benchmark = self.simulation_results.benchmark_comparison
                report_content.extend(
                    [
                        f"- **Benchmark**: {benchmark['benchmark_strategy']} ({benchmark['benchmark_return']:.2f}%)",
                        f"- **Excess Return**: +{benchmark['excess_return']:.2f}%",
                    ]
                )

            # Strategy breakdown
            if self.simulation_results.strategy_results:
                report_content.extend(["", "#### Individual Strategy Performance", ""])

                for (
                    strategy_name,
                    strategy_result,
                ) in self.simulation_results.strategy_results.items():
                    return_pct = strategy_result.get("total_return_pct", 0)
                    sharpe = strategy_result.get("sharpe_ratio", 0)
                    trades = strategy_result.get("total_trades", 0)
                    report_content.append(
                        f"- **{strategy_name}**: {return_pct:.2f}% return, {sharpe:.3f} Sharpe, {trades} trades"
                    )

            report_content.append("")

        # Performance comparison
        if self.final_comparison and "individual_results" in self.final_comparison:
            performance_report = self.case_study_analyzer.generate_performance_report(
                self.final_comparison,
                title="Strategy Performance Comparison (Percentage-Based)",
            )
            report_content.append(performance_report)

        # Files generated
        report_content.extend(
            [
                "## Generated Files",
                "",
                f"📁 **Output Directory**: `{self.config.output_directory}`",
                "",
                "### Visualizations",
                f"- Static analysis: `plots/{self.config.ticker}_complete_analysis.png`",
            ]
        )

        if self.config.create_animations:
            report_content.extend(
                [
                    f"- Regime evolution: `animations/{self.config.ticker}_regime_evolution.gif`",
                    f"- Performance evolution: `animations/{self.config.ticker}_performance_evolution.gif`",
                ]
            )

        report_content.extend(
            [
                "",
                "### Data",
                "- Evolution results: `data/evolution_results.json`",
                "- Performance metrics: `data/performance_evolution.json`",
                "",
            ]
        )

        # Methodology
        report_content.extend(
            [
                "## Methodology",
                "",
            ]
        )

        if self.config.enable_simulation:
            report_content.extend(
                [
                    "This case study follows a rigorous 5-phase approach:",
                    "",
                    "1. **Training**: HMM model trained on historical data with proper temporal isolation",
                    f"2. **Evolution**: Daily regime detection over {len(self.evolution_results)} evaluation periods",
                    "3. **Visualization**: Static plots and animated regime evolution",
                    "4. **Simulation**: Capital-based trading simulation with realistic execution",
                    "5. **Analysis**: Performance comparison using both percentage-based and capital-based metrics",
                ]
            )
        else:
            report_content.extend(
                [
                    "This case study follows a rigorous 4-phase approach:",
                    "",
                    "1. **Training**: HMM model trained on historical data with proper temporal isolation",
                    f"2. **Evolution**: Daily regime detection over {len(self.evolution_results)} evaluation periods",
                    "3. **Visualization**: Static plots and animated regime evolution",
                    "4. **Analysis**: Performance comparison against buy-and-hold and technical indicators",
                ]
            )

        report_content.extend(
            [
                "",
                "All analysis maintains strict temporal boundaries to prevent data leakage.",
            ]
        )

        if self.config.enable_simulation:
            report_content.extend(
                [
                    "",
                    "### Trading Simulation Details",
                    "",
                    "- Capital-based simulation with realistic buy/sell timing",
                    "- Risk management with stop-losses and position limits",
                    "- Multiple signal generators including HMM regime-following and buy-and-hold",
                    "- Comprehensive performance analytics including Sharpe ratio and drawdown analysis",
                ]
            )

        report_content.append("")

        # Save report
        report_text = "\n".join(report_content)
        report_path = os.path.join(
            self.config.output_directory,
            "reports",
            f"{self.config.ticker}_case_study_report.md",
        )

        with open(report_path, "w") as f:
            f.write(report_text)

        print(f"   📝 Final report saved: {report_path}")

        return report_text


def run_case_study_from_config(config: CaseStudyConfig) -> Dict[str, Any]:
    """
    Run complete case study from configuration.

    Args:
        config: Case study configuration

    Returns:
        Complete case study results
    """
    orchestrator = CaseStudyOrchestrator(config)
    return orchestrator.run_complete_case_study()


def main():
    """Main function for running case studies."""
    print(" Hidden Regime Case Study System")
    print("=" * 50)

    # Example configurations
    configs = [
        # Quick study example
        CaseStudyConfig.create_quick_study(ticker="NVDA", days_back=60, n_states=3),
        # Comprehensive study example (commented out for quick demo)
        # CaseStudyConfig.create_comprehensive_study(
        #     ticker="SPY",
        #     start_date="2024-01-01",
        #     end_date="2024-06-01",
        #     n_states=4
        # )
    ]

    results = {}

    for i, config in enumerate(configs):
        try:
            print(f"\n Running Case Study {i+1}: {config.ticker}")
            result = run_case_study_from_config(config)
            results[config.ticker] = result
            print(f" Case Study {i+1} Complete!")

        except Exception as e:
            print(f" Case Study {i+1} Failed: {e}")
            continue

    # Summary
    print(f"\n Case Study Summary:")
    print("=" * 30)

    for ticker, result in results.items():
        print(f"\n{ticker}:")
        if (
            "final_comparison" in result
            and "comparison_summary" in result["final_comparison"]
        ):
            summary = result["final_comparison"]["comparison_summary"]
            if "strategy_ranking" in summary:
                best_strategy = summary["strategy_ranking"][0]
                print(
                    f"  Best Strategy: {best_strategy[0]} (Sharpe: {best_strategy[1]:.3f})"
                )

        print(f"  Execution Time: {result.get('execution_time', 0):.1f} seconds")
        print(f"  Output Directory: {result['config'].output_directory}")

    print("\n🎉 All Case Studies Complete!")

    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "=" * 50)
        print(" Case Study System: SUCCESS")
        print("=" * 50)
    except Exception as e:
        print(f"\n Error running case studies: {e}")
        import traceback

        traceback.print_exc()
