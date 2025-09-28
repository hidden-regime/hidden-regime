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

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import hidden-regime components
from hidden_regime.config.case_study import CaseStudyConfig
from hidden_regime.factories.pipeline import pipeline_factory
from hidden_regime.pipeline.temporal import TemporalController
from hidden_regime.data.financial import FinancialDataLoader
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.analysis.case_study import CaseStudyAnalyzer
from hidden_regime.visualization.animations import RegimeAnimator, create_regime_comparison_gif
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
            color_scheme=config.color_scheme,
            style=config.plot_style
        )

        # Results storage
        self.full_dataset = None
        self.pipeline = None
        self.temporal_controller = None
        self.evolution_results = []
        self.performance_evolution = []
        self.final_comparison = None

    def run_complete_case_study(self) -> Dict[str, Any]:
        """
        Execute the complete case study workflow.

        Returns:
            Dictionary with complete case study results
        """
        print(f"🚀 Starting Case Study: {self.config.ticker}")
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
            print("\n📈 Phase 3: Evolution Analysis")
            self._run_evolution_analysis()

            # Phase 4: Visualization Generation
            print("\n🎨 Phase 4: Visualization Generation")
            self._generate_visualizations()

            # Phase 5: Comparative Analysis
            print("\n📊 Phase 5: Comparative Analysis")
            self._run_comparative_analysis()

            # Phase 6: Report Generation
            print("\n📝 Phase 6: Report Generation")
            final_report = self._generate_final_report()

            total_time = (datetime.now() - start_time).total_seconds()
            print(f"\n✅ Case Study Complete! ({total_time:.1f} seconds)")

            return {
                'config': self.config,
                'evolution_results': self.evolution_results,
                'performance_evolution': self.performance_evolution,
                'final_comparison': self.final_comparison,
                'final_report': final_report,
                'execution_time': total_time
            }

        except Exception as e:
            print(f"\n❌ Case Study Failed: {e}")
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
            "hourly": "hours"
        }

        data_config = FinancialDataConfig(
            ticker=self.config.ticker,
            start_date=training_start,
            end_date=self.config.end_date,
            frequency=frequency_mapping.get(self.config.frequency, "days")
        )

        data_loader = FinancialDataLoader(data_config)
        self.full_dataset = data_loader.update()

        print(f"   Loaded {len(self.full_dataset)} days of data ({training_start} to {self.config.end_date})")

        # Assess data for regime detection feasibility
        self._assess_data_for_regime_detection()

        # Create pipeline for case study
        model_overrides = self.config.get_model_config_with_overrides()
        # Remove n_states from overrides to avoid conflict
        model_overrides.pop('n_states', None)

        # Prepare analysis config overrides for regime label handling
        analysis_config_overrides = {}
        if self.config.force_regime_labels is not None:
            analysis_config_overrides['force_regime_labels'] = self.config.force_regime_labels
            analysis_config_overrides['acknowledge_override'] = self.config.acknowledge_override

        self.pipeline = pipeline_factory.create_financial_pipeline(
            ticker=self.config.ticker,
            n_states=self.config.n_states,
            include_report=False,  # We'll generate our own reports
            model_config_overrides=model_overrides,
            analysis_config_overrides=analysis_config_overrides
        )

        # Create temporal controller for proper isolation
        self.temporal_controller = TemporalController(self.pipeline, self.full_dataset)

        print("   ✅ Data and pipeline setup complete")

    def _assess_data_for_regime_detection(self):
        """Assess data quality and regime detection feasibility."""
        from hidden_regime.analysis.financial import FinancialAnalysis

        print("   Assessing data for regime detection feasibility...")

        try:
            # Perform data assessment
            assessment = FinancialAnalysis.assess_data_for_regime_detection(
                self.full_dataset, observed_signal='log_return'
            )

            # Store assessment for later reporting
            self.data_assessment = assessment

            # Check for critical warnings
            recommendations = assessment['recommendations']

            if not recommendations['suitable_for_regime_detection']:
                print(f"   ⚠️  WARNING: Data may not be suitable for regime detection")
                for warning in recommendations['warnings']:
                    print(f"       • {warning}")

                # Check if user wants to proceed
                if recommendations['recommended_n_states'] != self.config.n_states:
                    print(f"   💡 RECOMMENDATION: Use {recommendations['recommended_n_states']} states instead of {self.config.n_states}")
                    print(f"       Approach: {recommendations['regime_detection_approach']}")

            elif recommendations['warnings']:
                print(f"   ⚠️  Data assessment warnings:")
                for warning in recommendations['warnings']:
                    print(f"       • {warning}")

            else:
                print(f"   ✅ Data assessment: Suitable for regime detection")
                print(f"       Recommended states: {recommendations['recommended_n_states']}")

                # Show positive indicators
                for rationale in recommendations['rationale']:
                    print(f"       • {rationale}")

        except Exception as e:
            print(f"   ⚠️  Data assessment failed: {e}")
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
            print(f"   ✅ Initial model training complete")
        except Exception as e:
            print(f"   ⚠️  Training warning: {e}")
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
                if (hasattr(self.temporal_controller.pipeline, 'data_output') and
                    hasattr(self.temporal_controller.pipeline, 'analysis_output') and
                    hasattr(self.temporal_controller.pipeline, 'model_output')):

                    # Store results
                    pipeline_outputs = {
                        'date': eval_date,
                        'data': self.temporal_controller.pipeline.data_output.copy(),
                        'regime_data': self.temporal_controller.pipeline.analysis_output.copy(),
                        'model_output': self.temporal_controller.pipeline.model_output.copy()
                    }

                    self.evolution_results.append(pipeline_outputs)

                    # Debug logging
                    print(f"   📊 Stored evolution result for {eval_date} (total: {len(self.evolution_results)})")

                    # Calculate evolving performance
                    if len(self.evolution_results) >= 30:  # Need minimum data for performance calc
                        current_price_data = pipeline_outputs['data']
                        current_regime_data = pipeline_outputs['regime_data']

                        try:
                            perf_metrics = self.case_study_analyzer.analyze_hmm_strategy_performance(
                                current_price_data, current_regime_data
                            )
                            perf_metrics['evaluation_date'] = eval_date
                            self.performance_evolution.append(perf_metrics)
                            print(f"   📈 Performance calculated for {eval_date}")
                        except Exception as e:
                            print(f"   ⚠️  Performance calc warning for {eval_date}: {e}")
                else:
                    print(f"   ⚠️  Pipeline outputs missing for {eval_date}")

            except Exception as e:
                print(f"   ⚠️  Evolution warning for {eval_date}: {e}")
                import traceback
                print(f"   📝 Full error trace: {traceback.format_exc()}")
                continue

        print(f"   ✅ Evolution analysis complete ({len(self.evolution_results)} results)")

        # Save evolution data to JSON
        self._save_evolution_data()

    def _save_evolution_data(self):
        """Save evolution results and performance data to JSON files."""
        import json
        from datetime import datetime

        def make_serializable(obj):
            """Convert pandas/numpy objects to serializable format."""
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'isoformat'):  # datetime objects
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
                evolution_path = os.path.join(self.config.output_directory, 'data', 'evolution_results.json')
                with open(evolution_path, 'w') as f:
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

                performance_path = os.path.join(self.config.output_directory, 'data', 'performance_evolution.json')
                with open(performance_path, 'w') as f:
                    json.dump(serializable_performance, f, indent=2, default=str)
                print(f"   💾 Performance data saved: {performance_path}")

        except Exception as e:
            print(f"   ⚠️  Data save warning: {e}")

    def _generate_visualizations(self):
        """Phase 4: Generate visualizations and animations."""
        if not self.evolution_results:
            print("   ⚠️  No evolution results to visualize")
            return

        print("   Generating visualizations...")

        # Extract data for visualizations
        regime_data_sequence = [result['regime_data'] for result in self.evolution_results]
        evaluation_dates = [result['date'] for result in self.evolution_results]

        # Get final data for static plots
        final_data = self.evolution_results[-1]['data']
        final_regime_data = self.evolution_results[-1]['regime_data']

        # Generate static comprehensive plot
        try:
            static_fig = create_multi_panel_regime_plot(
                final_data, final_regime_data,
                title=f"{self.config.ticker} - Complete Regime Analysis"
            )

            static_path = os.path.join(self.config.output_directory, 'plots',
                                     f'{self.config.ticker}_complete_analysis.png')
            static_fig.savefig(static_path, dpi=300, bbox_inches='tight')
            plt.close(static_fig)
            print(f"   📊 Static analysis saved: {static_path}")

        except Exception as e:
            print(f"   ⚠️  Static plot warning: {e}")

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
                    save_path=os.path.join(self.config.output_directory, 'animations',
                                         f'{self.config.ticker}_regime_evolution.gif'),
                    fps=self.config.animation_fps
                )

                print(f"   🎬 Regime evolution GIF created")

                # Create performance evolution animation if we have performance data
                if self.performance_evolution:
                    perf_anim = self.regime_animator.create_performance_evolution_animation(
                        self.performance_evolution,
                        [p['evaluation_date'] for p in self.performance_evolution],
                        title=f"{self.config.ticker} Performance Evolution",
                        save_path=os.path.join(self.config.output_directory, 'animations',
                                             f'{self.config.ticker}_performance_evolution.gif'),
                        fps=self.config.animation_fps
                    )
                    print(f"   📈 Performance evolution GIF created")

            except Exception as e:
                print(f"   ⚠️  Animation warning: {e}")
                # Animation failure is not critical - continue with the rest of the analysis

        print("   ✅ Visualization generation complete")

    def _run_comparative_analysis(self):
        """Phase 5: Compare HMM strategy against baselines with proper temporal isolation."""
        if not self.evolution_results:
            print("   ⚠️  No evolution results for comparison")
            return

        print("   Running comparative performance analysis...")

        # Use temporal evolution data for proper comparison
        # This ensures all strategies see the same data progression as the HMM model

        # Get the price data that matches the HMM analysis period
        final_price_data = self.evolution_results[-1]['data']
        final_regime_data = self.evolution_results[-1]['regime_data']

        # Remove the legacy technical indicator generation that causes look-ahead bias
        # The comprehensive TA analysis in compare_all_strategies handles temporal isolation properly
        indicators = {}

        # Run comprehensive comparison with temporal isolation
        try:
            self.final_comparison = self.case_study_analyzer.compare_all_strategies(
                final_price_data,
                final_regime_data,
                indicators,  # Empty - use comprehensive TA analysis instead
                hmm_strategy_types=['regime_following'],
                n_best_indicators=5
            )

            print("   ✅ Comparative analysis complete")

            # Print summary
            if 'comparison_summary' in self.final_comparison:
                summary = self.final_comparison['comparison_summary']
                if 'strategy_ranking' in summary:
                    print(f"   📊 Top strategy: {summary['strategy_ranking'][0][0]} " +
                          f"(Sharpe: {summary['strategy_ranking'][0][1]:.3f})")

        except Exception as e:
            print(f"   ⚠️  Comparison analysis warning: {e}")
            self.final_comparison = {'error': str(e)}

    def _generate_final_report(self) -> str:
        """Phase 6: Generate comprehensive markdown report."""
        print("   Generating final case study report...")

        # Generate performance report
        report_content = []

        # Header
        report_content.extend([
            f"# Case Study Report: {self.config.ticker}",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Ticker**: {self.config.ticker}",
            f"**Analysis Period**: {self.config.start_date} to {self.config.end_date}",
            f"**Training Period**: {self.config.n_training} days",
            f"**Model States**: {self.config.n_states}",
            "",
        ])

        # Configuration summary
        config_summary = self.config.get_summary_info()
        report_content.extend([
            "## Configuration Summary",
            "",
            f"- **Analysis Period**: {config_summary['analysis_period']['total_periods']} periods",
            f"- **Training Days**: {config_summary['training_period']['n_training_days']}",
            f"- **Frequency**: {config_summary['configuration']['frequency']}",
            f"- **Technical Indicators**: {config_summary['configuration']['indicators']}",
            f"- **Animations Created**: {config_summary['configuration']['include_animations']}",
            ""
        ])

        # Evolution results summary
        if self.evolution_results:
            report_content.extend([
                "## Evolution Analysis Results",
                "",
                f"- **Total Evaluation Dates**: {len(self.evolution_results)}",
                f"- **First Evaluation**: {self.evolution_results[0]['date']}",
                f"- **Last Evaluation**: {self.evolution_results[-1]['date']}",
                ""
            ])

            # Final regime state
            if 'regime_data' in self.evolution_results[-1]:
                final_regime_data = self.evolution_results[-1]['regime_data']
                if len(final_regime_data) > 0:
                    current_regime = final_regime_data['predicted_state'].iloc[-1]
                    current_confidence = final_regime_data.get('confidence', pd.Series([0])).iloc[-1]

                    regime_names = ['Bear', 'Sideways', 'Bull', 'Strong Bull'][:self.config.n_states]
                    current_regime_name = regime_names[int(current_regime)] if current_regime < len(regime_names) else f"Regime {current_regime}"

                    report_content.extend([
                        "### Current Market State",
                        "",
                        f"- **Current Regime**: {current_regime_name}",
                        f"- **Confidence**: {current_confidence:.1%}",
                        ""
                    ])

        # Performance comparison
        if self.final_comparison and 'individual_results' in self.final_comparison:
            performance_report = self.case_study_analyzer.generate_performance_report(
                self.final_comparison,
                title="Strategy Performance Comparison"
            )
            report_content.append(performance_report)

        # Files generated
        report_content.extend([
            "## Generated Files",
            "",
            f"📁 **Output Directory**: `{self.config.output_directory}`",
            "",
            "### Visualizations",
            f"- Static analysis: `plots/{self.config.ticker}_complete_analysis.png`",
        ])

        if self.config.create_animations:
            report_content.extend([
                f"- Regime evolution: `animations/{self.config.ticker}_regime_evolution.gif`",
                f"- Performance evolution: `animations/{self.config.ticker}_performance_evolution.gif`",
            ])

        report_content.extend([
            "",
            "### Data",
            "- Evolution results: `data/evolution_results.json`",
            "- Performance metrics: `data/performance_evolution.json`",
            ""
        ])

        # Methodology
        report_content.extend([
            "## Methodology",
            "",
            "This case study follows a rigorous 4-phase approach:",
            "",
            "1. **Training**: HMM model trained on historical data with proper temporal isolation",
            f"2. **Evolution**: Daily regime detection over {len(self.evolution_results)} evaluation periods",
            "3. **Visualization**: Static plots and animated regime evolution",
            "4. **Analysis**: Performance comparison against buy-and-hold and technical indicators",
            "",
            "All analysis maintains strict temporal boundaries to prevent data leakage.",
            ""
        ])

        # Save report
        report_text = "\n".join(report_content)
        report_path = os.path.join(self.config.output_directory, 'reports',
                                 f'{self.config.ticker}_case_study_report.md')

        with open(report_path, 'w') as f:
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
    print("📊 Hidden Regime Case Study System")
    print("=" * 50)

    # Example configurations
    configs = [
        # Quick study example
        CaseStudyConfig.create_quick_study(
            ticker="NVDA",
            days_back=60,
            n_states=3
        ),

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
            print(f"\n🚀 Running Case Study {i+1}: {config.ticker}")
            result = run_case_study_from_config(config)
            results[config.ticker] = result
            print(f"✅ Case Study {i+1} Complete!")

        except Exception as e:
            print(f"❌ Case Study {i+1} Failed: {e}")
            continue

    # Summary
    print(f"\n📊 Case Study Summary:")
    print("=" * 30)

    for ticker, result in results.items():
        print(f"\n{ticker}:")
        if 'final_comparison' in result and 'comparison_summary' in result['final_comparison']:
            summary = result['final_comparison']['comparison_summary']
            if 'strategy_ranking' in summary:
                best_strategy = summary['strategy_ranking'][0]
                print(f"  Best Strategy: {best_strategy[0]} (Sharpe: {best_strategy[1]:.3f})")

        print(f"  Execution Time: {result.get('execution_time', 0):.1f} seconds")
        print(f"  Output Directory: {result['config'].output_directory}")

    print("\n🎉 All Case Studies Complete!")

    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "="*50)
        print("🚀 Case Study System: SUCCESS")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Error running case studies: {e}")
        import traceback
        traceback.print_exc()