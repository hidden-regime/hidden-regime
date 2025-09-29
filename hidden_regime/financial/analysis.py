"""
Unified financial regime analysis system.

Provides single entry point for complete financial market regime detection,
characterization, signal generation, and trading simulation workflow.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import time
import os
import warnings
warnings.filterwarnings('ignore')

from .config import FinancialRegimeConfig
from .regime_characterizer import FinancialRegimeCharacterizer, RegimeProfile
from .signal_generation import FinancialSignalGenerator, AdaptiveSignalGenerator

from ..data.financial import FinancialDataLoader
from ..config.data import FinancialDataConfig
from ..factories.pipeline import pipeline_factory
from ..pipeline.temporal import TemporalController
from ..simulation.simulation_orchestrator import SimulationOrchestrator
from ..config.simulation import SimulationConfig
from ..analysis.technical_indicators import TechnicalIndicatorAnalyzer
from ..visualization.plotting import create_multi_panel_regime_plot
from ..utils.formatting import format_strategy_name
from ..visualization.animations import RegimeAnimator
from ..utils.exceptions import AnalysisError


@dataclass
class FinancialAnalysisResult:
    """
    Complete results from financial regime analysis.

    Contains all outputs from regime detection through trading simulation
    with financial interpretation and performance metrics.
    """
    # Configuration and metadata
    config: FinancialRegimeConfig
    analysis_success: bool
    execution_time: float

    # Market data
    market_data: pd.DataFrame

    # Regime analysis
    regime_detection: pd.DataFrame
    regime_profiles: Dict[int, RegimeProfile]
    regime_summary: str

    # Signal generation
    trading_signals: pd.DataFrame
    signal_summary: Dict[str, Any]

    # Trading simulation (if enabled)
    simulation_results: Optional[Any] = None

    # Performance comparison
    strategy_comparison: Optional[Dict[str, Any]] = None

    # Technical analysis (if enabled)
    technical_analysis: Optional[Dict[str, Any]] = None

    # Visualizations
    static_plots: Optional[Dict[str, str]] = None
    animations: Optional[Dict[str, str]] = None

    # Comprehensive report
    report_content: Optional[str] = None

    @property
    def current_regime_info(self) -> Dict[str, Any]:
        """Get current regime information from analysis results."""
        if not self.analysis_success or self.regime_detection is None or not self.regime_profiles:
            return {"regime_type": "Unknown", "confidence": 0.0, "expected_return": 0.0, "volatility": 0.0}

        # Get most recent regime prediction
        latest_state = self.regime_detection['predicted_state'].iloc[-1]
        latest_confidence = self.regime_detection.get('confidence', pd.Series([0.5])).iloc[-1]

        if latest_state in self.regime_profiles:
            profile = self.regime_profiles[latest_state]
            return {
                "regime_type": profile.regime_type.value,
                "confidence": latest_confidence,
                "expected_return": profile.annualized_return,
                "volatility": profile.annualized_volatility
            }
        else:
            return {"regime_type": "Unknown", "confidence": 0.0, "expected_return": 0.0, "volatility": 0.0}

    def _get_current_regime_info(self) -> Dict[str, Any]:
        """Get current regime information based on most recent data."""
        if self.regime_detection is None or self.regime_profiles is None:
            return {"regime_type": "Unknown", "confidence": 0.0, "expected_return": 0.0, "volatility": 0.0}

        # Get most recent regime prediction
        latest_state = self.regime_detection['predicted_state'].iloc[-1]
        latest_confidence = self.regime_detection.get('confidence', pd.Series([0.5])).iloc[-1]

        # Get regime profile
        if latest_state in self.regime_profiles:
            profile = self.regime_profiles[latest_state]
            return {
                "regime_type": profile.regime_type.value,
                "confidence": latest_confidence,
                "expected_return": profile.annualized_return,
                "volatility": profile.annualized_volatility
            }
        else:
            return {"regime_type": "Unknown", "confidence": latest_confidence, "expected_return": 0.0, "volatility": 0.0}


class FinancialRegimeAnalysis:
    """
    Single entry point for complete financial regime analysis.

    Provides unified interface for market regime detection, characterization,
    signal generation, and trading simulation with financial domain focus.
    """

    def __init__(self, config: FinancialRegimeConfig):
        """
        Initialize financial regime analysis system.

        Args:
            config: Unified financial regime configuration
        """
        self.config = config
        self.config.validate()

        # Initialize components
        self.regime_characterizer = FinancialRegimeCharacterizer(
            min_regime_days=self.config.min_regime_days
        )

        # Create signal generators based on configured strategies
        self.signal_generators = self._create_signal_generators()

        # Technical analysis (if enabled)
        self.technical_analyzer = None
        if self.config.include_technical_indicators:
            self.technical_analyzer = TechnicalIndicatorAnalyzer()

        # Visualization components
        if self.config.generate_visualizations:
            self.regime_animator = RegimeAnimator(
                color_scheme=self.config.color_scheme,
                style=self.config.plot_style
            )

        # Results storage
        self.market_data = None
        self.regime_detection = None
        self.regime_profiles = None
        self.trading_signals = None

    def run_complete_analysis(self) -> FinancialAnalysisResult:
        """
        Execute complete financial regime analysis workflow.

        Returns:
            Comprehensive financial analysis results
        """
        print(f"🚀 Starting Financial Regime Analysis: {self.config.ticker}")
        print("=" * 60)

        start_time = time.time()

        try:
            # Phase 1: Market Data Loading
            print("\n📊 Phase 1: Market Data Loading")
            self._load_market_data()

            # Phase 2: Regime Detection
            print("\n🎯 Phase 2: Regime Detection")
            self._detect_regimes()

            # Phase 3: Regime Characterization
            print("\n🧠 Phase 3: Financial Regime Characterization")
            self._characterize_regimes()

            # Phase 4: Signal Generation
            print("\n📈 Phase 4: Trading Signal Generation")
            self._generate_trading_signals()

            # Phase 5: Technical Analysis (if enabled)
            technical_analysis = None
            if self.config.include_technical_indicators:
                print("\n📊 Phase 5: Technical Analysis")
                technical_analysis = self._run_technical_analysis()

            # Phase 6: Trading Simulation (if enabled)
            simulation_results = None
            if self.config.enable_simulation:
                print("\n💰 Phase 6: Trading Simulation")
                simulation_results = self._run_trading_simulation()

            # Phase 7: Performance Comparison
            print("\n📋 Phase 7: Strategy Performance Comparison")
            strategy_comparison = self._compare_strategies(simulation_results, technical_analysis)

            # Phase 8: Visualization Generation
            static_plots, animations = None, None
            if self.config.generate_visualizations:
                print("\n🎨 Phase 8: Visualization Generation")
                static_plots, animations = self._generate_visualizations()

            # Phase 9: Report Generation
            print("\n📝 Phase 9: Comprehensive Report Generation")
            report_content = self._generate_report(
                simulation_results, strategy_comparison, technical_analysis
            )

            execution_time = time.time() - start_time
            print(f"\n✅ Financial Regime Analysis Complete! ({execution_time:.2f} seconds)")

            return FinancialAnalysisResult(
                config=self.config,
                analysis_success=True,
                execution_time=execution_time,
                market_data=self.market_data,
                regime_detection=self.regime_detection,
                regime_profiles=self.regime_profiles,
                regime_summary=self.regime_characterizer.get_regime_summary(self.regime_profiles),
                trading_signals=self.trading_signals,
                signal_summary=self._get_signal_summary(),
                simulation_results=simulation_results,
                strategy_comparison=strategy_comparison,
                technical_analysis=technical_analysis,
                static_plots=static_plots,
                animations=animations,
                report_content=report_content
            )

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"\n❌ Financial Regime Analysis Failed: {e}")
            import traceback
            traceback.print_exc()

            return FinancialAnalysisResult(
                config=self.config,
                analysis_success=False,
                execution_time=execution_time,
                market_data=None,
                regime_detection=None,
                regime_profiles={},
                regime_summary=f"Analysis failed: {e}",
                trading_signals=None,
                signal_summary={}
            )

    def _load_market_data(self):
        """Load and prepare market data for analysis."""
        print("   Loading market data...")

        # Calculate extended date range for training
        training_start, training_end = self.config.get_training_date_range()

        # Create data configuration
        data_config = FinancialDataConfig(
            ticker=self.config.ticker,
            start_date=training_start,
            end_date=self.config.end_date,
            source=self.config.data_source
        )

        # Load data
        data_loader = FinancialDataLoader(data_config)
        self.market_data = data_loader.update()

        print(f"   Loaded {len(self.market_data)} days of data ({training_start} to {self.config.end_date})")
        print(f"   ✅ Market data loaded successfully")

    def _detect_regimes(self):
        """Detect market regimes using HMM."""
        print("   Training HMM model for regime detection...")

        # Create HMM pipeline
        self.pipeline = pipeline_factory.create_financial_pipeline(
            ticker=self.config.ticker,
            n_states=self.config.n_regimes,
            tolerance=self.config.hmm_tolerance,
            max_iterations=self.config.hmm_max_iterations,
            random_seed=self.config.hmm_random_seed,
            include_report=False
        )

        # Create temporal controller for proper training
        self.temporal_controller = TemporalController(self.pipeline, self.market_data)

        # Train model up to analysis start date
        training_start, training_end = self.config.get_training_date_range()
        self.temporal_controller.update_as_of(training_end)

        # Run regime detection on full analysis period
        self.temporal_controller.update_as_of(self.config.end_date)

        # Extract regime detection results
        if hasattr(self.temporal_controller.pipeline, 'analysis_output'):
            self.regime_detection = self.temporal_controller.pipeline.analysis_output.copy()
            print(f"   Detected regimes over {len(self.regime_detection)} days")
            print(f"   ✅ Regime detection completed")
        else:
            raise AnalysisError("Failed to generate regime detection results")

    def _characterize_regimes(self):
        """Characterize detected regimes using financial analysis."""
        print("   Analyzing financial characteristics of detected regimes...")

        # Get price data for the analysis period
        analysis_start_dt = pd.to_datetime(self.config.start_date)

        # Handle timezone-aware index
        if self.market_data.index.tz is not None:
            analysis_start_dt = analysis_start_dt.tz_localize(self.market_data.index.tz)

        analysis_data = self.market_data[self.market_data.index >= analysis_start_dt]

        # Characterize regimes
        self.regime_profiles = self.regime_characterizer.characterize_regimes(
            self.regime_detection, analysis_data
        )

        print(f"   Characterized {len(self.regime_profiles)} regime states")

        # Print regime summary
        for state_id, profile in self.regime_profiles.items():
            print(f"   State {state_id}: {profile.regime_type.value} "
                  f"(Return: {profile.annualized_return:.1%}, "
                  f"Vol: {profile.annualized_volatility:.1%})")

        print(f"   ✅ Regime characterization completed")

    def _generate_trading_signals(self):
        """Generate trading signals based on regime characteristics."""
        print("   Generating intelligent trading signals...")

        # Get analysis period data
        analysis_start_dt = pd.to_datetime(self.config.start_date)

        # Handle timezone-aware index
        if self.market_data.index.tz is not None:
            analysis_start_dt = analysis_start_dt.tz_localize(self.market_data.index.tz)

        analysis_data = self.market_data[self.market_data.index >= analysis_start_dt]
        analysis_regime_data = self.regime_detection[self.regime_detection.index >= analysis_start_dt]

        all_signals = {}

        # Generate signals from each configured strategy
        for strategy_name in self.config.signal_strategies:
            try:
                if strategy_name in self.signal_generators:
                    generator = self.signal_generators[strategy_name]

                    # Set regime profiles for financial generators
                    if hasattr(generator, 'regime_profiles'):
                        generator.regime_profiles = self.regime_profiles

                    signals = generator.generate_signals(analysis_data, analysis_regime_data)
                    all_signals[strategy_name] = signals

                    print(f"   Generated {strategy_name} signals")

            except Exception as e:
                print(f"   ⚠️ Warning: Failed to generate {strategy_name} signals: {e}")
                # Create neutral signals as fallback
                all_signals[strategy_name] = pd.Series(0.0, index=analysis_data.index)

        # Combine signals into DataFrame
        self.trading_signals = pd.DataFrame(all_signals)

        print(f"   Generated signals for {len(self.trading_signals.columns)} strategies")
        print(f"   ✅ Trading signal generation completed")

    def _run_technical_analysis(self) -> Dict[str, Any]:
        """Run technical analysis if enabled."""
        print("   Running technical analysis...")

        analysis_start_dt = pd.to_datetime(self.config.start_date)

        # Handle timezone-aware index
        if self.market_data.index.tz is not None:
            analysis_start_dt = analysis_start_dt.tz_localize(self.market_data.index.tz)

        analysis_data = self.market_data[self.market_data.index >= analysis_start_dt]

        try:
            technical_results = self.technical_analyzer.analyze_comprehensive_indicators(
                analysis_data, indicators=self.config.technical_indicators
            )

            print(f"   Analyzed {len(self.config.technical_indicators)} technical indicators")
            print(f"   ✅ Technical analysis completed")

            return technical_results

        except Exception as e:
            print(f"   ⚠️ Technical analysis warning: {e}")
            return {}

    def _run_trading_simulation(self) -> Optional[Any]:
        """Run trading simulation if enabled."""
        print("   Running capital-based trading simulation...")

        try:
            # Create simulation configuration
            # Include technical indicators in simulation if enabled
            technical_indicators = self.config.technical_indicators if self.config.include_technical_indicators else []

            sim_config = SimulationConfig(
                **self.config.get_simulation_config_dict(),
                signal_generators=list(self.trading_signals.columns),
                hmm_strategy_types=['regime_following'],  # Will be ignored by financial signals
                technical_indicators=technical_indicators  # Include technical indicators for comparison
            )

            # Create simulation orchestrator
            sim_orchestrator = SimulationOrchestrator(sim_config)

            # Get analysis period data
            analysis_start_dt = pd.to_datetime(self.config.start_date)

            # Handle timezone-aware index
            if self.market_data.index.tz is not None:
                analysis_start_dt = analysis_start_dt.tz_localize(self.market_data.index.tz)

            analysis_data = self.market_data[self.market_data.index >= analysis_start_dt]
            analysis_regime_data = self.regime_detection[self.regime_detection.index >= analysis_start_dt]

            # Run simulation with pre-generated signals
            simulation_result = sim_orchestrator.run_simulation(
                price_data=analysis_data,
                regime_data=analysis_regime_data,
                symbol=self.config.ticker
            )

            if simulation_result.simulation_success:
                print(f"   💰 Simulation successful: {simulation_result.total_return_pct:.2f}% return")
                print(f"   🏆 Best strategy: {simulation_result.best_strategy}")
                print(f"   📊 Sharpe ratio: {simulation_result.sharpe_ratio:.3f}")

                # Save trade journals if enabled
                if self.config.save_trade_journal:
                    self._save_trade_journals(simulation_result)

                print(f"   ✅ Trading simulation completed")
            else:
                print(f"   ❌ Simulation failed")

            return simulation_result

        except Exception as e:
            print(f"   ⚠️ Trading simulation warning: {e}")
            return None

    def _compare_strategies(self,
                          simulation_results: Optional[Any],
                          technical_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare strategy performance."""
        print("   Comparing strategy performance...")

        comparison = {
            'strategies_analyzed': list(self.trading_signals.columns),
            'regime_strategies': len([s for s in self.trading_signals.columns if 'financial' in s]),
            'traditional_strategies': len([s for s in self.trading_signals.columns if s == 'buy_and_hold'])
        }

        if simulation_results and simulation_results.simulation_success:
            comparison.update({
                'best_strategy': simulation_results.best_strategy,
                'best_return': simulation_results.total_return_pct,
                'total_trades': simulation_results.total_trades,
                'sharpe_ratio': simulation_results.sharpe_ratio
            })

            if simulation_results.benchmark_comparison:
                benchmark = simulation_results.benchmark_comparison
                comparison['excess_return'] = benchmark['excess_return']

        print(f"   Compared {len(self.trading_signals.columns)} strategies")
        print(f"   ✅ Strategy comparison completed")

        return comparison

    def _generate_visualizations(self) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
        """Generate static plots and animations."""
        print("   Generating visualizations...")

        static_plots = {}
        animations = {}

        # Create output directory structure
        output_structure = self.config.create_output_structure()

        try:
            # Static comprehensive plot
            analysis_start_dt = pd.to_datetime(self.config.start_date)

            # Handle timezone-aware index
            if self.market_data.index.tz is not None:
                analysis_start_dt = analysis_start_dt.tz_localize(self.market_data.index.tz)

            analysis_data = self.market_data[self.market_data.index >= analysis_start_dt]
            analysis_regime_data = self.regime_detection[self.regime_detection.index >= analysis_start_dt]

            static_fig = create_multi_panel_regime_plot(
                analysis_data,
                analysis_regime_data,
                title=f"{self.config.ticker} - Financial Regime Analysis",
                color_scheme=self.config.color_scheme,
                regime_profiles=self.regime_profiles
            )

            static_path = f"{output_structure['plots']}/{self.config.ticker}_regime_analysis.png"
            static_fig.savefig(static_path, dpi=300, bbox_inches='tight')
            plt.close(static_fig)
            static_plots['regime_analysis'] = static_path

            print(f"   📊 Static plots generated")

            # Animations (if enabled)
            if self.config.create_animations:
                try:
                    # Create regime evolution animation
                    animation_path = os.path.join(output_structure['animations'],
                                                 f"{self.config.ticker}_regime_evolution.gif")

                    # Create animation data sequence for regime evolution
                    # Generate time-evolving sequence showing how regime detection builds up
                    regime_data_sequence = []
                    evaluation_dates = []

                    # Create frames showing progressive regime detection
                    analysis_start = self.market_data.index[self.config.training_days] if len(self.market_data) > self.config.training_days else self.market_data.index[0]
                    total_analysis_days = len(self.market_data.index[self.market_data.index >= analysis_start])

                    # Create frames every 5-10 days for smoother animation
                    frame_interval = max(5, total_analysis_days // 20)  # Aim for ~20 frames

                    for i in range(frame_interval, total_analysis_days + 1, frame_interval):
                        # Get data up to this point
                        frame_end_date = self.market_data.index[self.market_data.index >= analysis_start][i-1]

                        # Create regime data for this frame (showing evolution up to this date)
                        frame_regime_data = self.regime_detection[self.regime_detection.index <= frame_end_date].copy()

                        if len(frame_regime_data) > 0:
                            regime_data_sequence.append(frame_regime_data)
                            evaluation_dates.append(frame_end_date.strftime('%Y-%m-%d'))

                    # Always include the final frame
                    if len(regime_data_sequence) == 0 or evaluation_dates[-1] != self.market_data.index[-1].strftime('%Y-%m-%d'):
                        regime_data_sequence.append(self.regime_detection)
                        evaluation_dates.append(self.market_data.index[-1].strftime('%Y-%m-%d'))

                    print(f"   🎬 Creating animation with {len(regime_data_sequence)} frames")

                    # Generate regime evolution animation
                    anim = self.regime_animator.create_evolving_regime_animation(
                        data=self.market_data,
                        regime_data_sequence=regime_data_sequence,
                        evaluation_dates=evaluation_dates,
                        title=f"{self.config.ticker} Regime Analysis",
                        save_path=animation_path,
                        fps=self.config.animation_fps,
                        regime_profiles=self.regime_profiles
                    )

                    if anim is not None:
                        animations['regime_evolution'] = animation_path
                        print(f"   🎬 Regime evolution animation saved to {animation_path}")
                    else:
                        print(f"   ⚠️ Animation generation failed")

                    # Save individual frames if requested
                    if self.config.save_individual_frames and output_structure.get('frames'):
                        from ..visualization.animations import save_individual_frames
                        frame_paths = save_individual_frames(
                            data=self.market_data,
                            regime_data_sequence=regime_data_sequence,
                            evaluation_dates=evaluation_dates,
                            output_directory=output_structure['frames'],
                            color_scheme=self.config.color_scheme
                        )
                        animations['individual_frames'] = frame_paths
                        print(f"   🖼️ {len(frame_paths)} individual frames saved")

                except Exception as anim_error:
                    print(f"   ⚠️ Animation generation failed: {anim_error}")

                print(f"   🎬 Animation processing completed")

            print(f"   ✅ Visualization generation completed")

            return static_plots, animations

        except Exception as e:
            print(f"   ⚠️ Visualization warning: {e}")
            return None, None


    def _generate_report(self,
                        simulation_results: Optional[Any],
                        strategy_comparison: Dict[str, Any],
                        technical_analysis: Optional[Dict[str, Any]]) -> str:
        """Generate comprehensive markdown report."""
        print("   Generating comprehensive financial report...")

        # Create output directory
        output_structure = self.config.create_output_structure()

        # Get current market context
        if self.regime_detection is not None and self.regime_profiles is not None:
            # Get most recent regime prediction
            latest_state = self.regime_detection['predicted_state'].iloc[-1]
            latest_confidence = self.regime_detection.get('confidence', pd.Series([0.5])).iloc[-1]

            if latest_state in self.regime_profiles:
                profile = self.regime_profiles[latest_state]
                current_regime = {
                    "regime_type": profile.regime_type.value,
                    "confidence": latest_confidence,
                    "expected_return": profile.annualized_return,
                    "volatility": profile.annualized_volatility
                }
            else:
                current_regime = {"regime_type": "Unknown", "confidence": 0.0, "expected_return": 0.0, "volatility": 0.0}
        else:
            current_regime = {"regime_type": "Unknown", "confidence": 0.0, "expected_return": 0.0, "volatility": 0.0}
        analysis_days = (pd.to_datetime(self.config.end_date) - pd.to_datetime(self.config.start_date)).days

        report_lines = [
            f"# Comprehensive Financial Regime Analysis: {self.config.ticker}",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Analysis Period**: {self.config.start_date} to {self.config.end_date} ({analysis_days} days)",
            f"**Training Period**: {self.config.training_days} days",
            f"**Initial Capital**: ${self.config.initial_capital:,.2f}",
            f"**Analysis Type**: {self.config.n_regimes}-state Hidden Markov Model with Financial Characterization",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"This report presents a comprehensive financial regime analysis of **{self.config.ticker}** using an advanced ",
            f"Hidden Markov Model (HMM) approach that intelligently characterizes market regimes based on actual ",
            f"financial metrics rather than naive state assumptions. The analysis combines regime detection, ",
            f"financial characterization, and systematic trading simulation to provide actionable market intelligence.",
            "",
            f"**Current Market Assessment**: {self.config.ticker} is currently in a **{current_regime['regime_type']}** regime ",
            f"with {current_regime['confidence']:.1%} confidence, expecting {current_regime['expected_return']:.1%} annual returns ",
            f"with {current_regime['volatility']:.1%} annual volatility.",
            "",
            "### Key Analytical Innovations",
            "",
            "- **Financial-First Architecture**: Regime states are characterized by actual returns, volatility, and persistence rather than arbitrary labels",
            "- **Intelligent Signal Generation**: Trading signals based on regime characteristics and confidence levels",
            "- **Single-Asset Optimization**: 100% capital allocation capability for dedicated asset analysis",
            "- **Zero Transaction Costs**: Retail-friendly defaults for commission-free trading environments",
            "- **Uncertainty Quantification**: Confidence intervals and regime transition probabilities",
            "",
            "### Key Findings",
            ""
        ]

        # Add comprehensive regime analysis
        if self.regime_profiles:
            report_lines.extend([
                "#### Detected Market Regimes",
                "",
                f"The {self.config.n_regimes}-state HMM identified the following distinct market regimes based on ",
                f"financial characteristics analysis:",
                ""
            ])

            # Regime performance summary table
            regime_summary = []
            for state_id, profile in self.regime_profiles.items():
                regime_summary.append({
                    'state': state_id,
                    'type': profile.regime_type.value.title(),
                    'return': profile.annualized_return,
                    'volatility': profile.annualized_volatility,
                    'win_rate': profile.win_rate,
                    'duration': profile.avg_duration,
                    'strength': profile.regime_strength,
                    'confidence': profile.confidence_score
                })

            # Create regime comparison table
            report_lines.extend([
                "| Regime | Type | Annual Return | Volatility | Win Rate | Avg Duration | Strength | Confidence |",
                "|--------|------|---------------|------------|----------|--------------|----------|------------|"
            ])

            for regime in sorted(regime_summary, key=lambda x: x['return'], reverse=True):
                report_lines.append(
                    f"| State {regime['state']} | {regime['type']} | {regime['return']:.1%} | "
                    f"{regime['volatility']:.1%} | {regime['win_rate']:.1%} | {regime['duration']:.1f} days | "
                    f"{regime['strength']:.2f} | {regime['confidence']:.2f} |"
                )

            report_lines.extend(["", "**Detailed Regime Characterization:**", ""])

            for state_id, profile in self.regime_profiles.items():
                regime_type = profile.regime_type.value.title()

                # Risk-adjusted metrics
                sharpe_equivalent = profile.annualized_return / max(profile.annualized_volatility, 0.01) if profile.annualized_volatility > 0 else 0

                report_lines.extend([
                    f"**State {state_id} - {regime_type} Regime:**",
                    "",
                    f"- **Performance**: {profile.annualized_return:.1%} annual return with {profile.annualized_volatility:.1%} volatility",
                    f"- **Risk Profile**: {sharpe_equivalent:.2f} risk-adjusted return ratio",
                    f"- **Consistency**: {profile.win_rate:.1%} win rate, {profile.regime_strength:.2f} regime strength",
                    f"- **Persistence**: {getattr(profile, 'avg_duration', profile.persistence_days):.1f} days average duration, {getattr(profile, 'max_drawdown', 0.0):.1%} max drawdown",
                    f"- **Trading Characteristics**: {profile.confidence_score:.2f} characterization confidence",
                    ""
                ])

                # Add regime-specific insights
                if profile.regime_type.value == 'bullish':
                    report_lines.extend([
                        f"  *Bullish periods show strong upward momentum. Recommended strategy: Long positions with trend following.*",
                        ""
                    ])
                elif profile.regime_type.value == 'bearish':
                    report_lines.extend([
                        f"  *Bearish periods show sustained decline. Recommended strategy: Defensive positioning or short exposure.*",
                        ""
                    ])
                elif profile.regime_type.value == 'sideways':
                    report_lines.extend([
                        f"  *Sideways periods show range-bound behavior. Recommended strategy: Mean reversion or reduced exposure.*",
                        ""
                    ])
                elif profile.regime_type.value == 'crisis':
                    report_lines.extend([
                        f"  *Crisis periods show extreme volatility. Recommended strategy: Cash position or defensive hedging.*",
                        ""
                    ])
                elif profile.regime_type.value == 'mixed':
                    report_lines.extend([
                        f"  *Mixed periods show uncertain directional bias. Recommended strategy: Reduced position sizing.*",
                        ""
                    ])

        # Add comprehensive simulation results
        if simulation_results and simulation_results.simulation_success:
            trading_days = (pd.to_datetime(self.config.end_date) - pd.to_datetime(self.config.start_date)).days
            annualized_factor = 252 / trading_days if trading_days > 0 else 1

            report_lines.extend([
                "## Comprehensive Trading Simulation Results",
                "",
                f"The simulation tested {len(self.config.signal_strategies)} distinct trading strategies plus ",
                f"{len(self.config.technical_indicators) if self.config.include_technical_indicators else 0} technical indicators ",
                f"over {trading_days} trading days with ${self.config.initial_capital:,.0f} initial capital.",
                "",
                "### Overall Performance Summary",
                "",
                f"- **🏆 Best Strategy**: {simulation_results.best_strategy.replace('_', ' ').title()}",
                f"- **💰 Total Return**: {simulation_results.total_return_pct:.2f}% (${simulation_results.initial_capital:,.0f} → ${getattr(simulation_results, 'final_value', 0):,.0f})",
                f"- **📈 Annualized Return**: {getattr(simulation_results, 'annualized_return', 0):.2f}%",
                f"- **⚡ Sharpe Ratio**: {simulation_results.sharpe_ratio:.3f}",
                f"- **🔻 Maximum Drawdown**: {simulation_results.max_drawdown_pct:.2f}%",
                f"- **🎯 Total Trades**: {simulation_results.total_trades}",
                f"- **🎲 Win Rate**: {getattr(simulation_results, 'win_rate', 0):.1f}%",
                ""
            ])

            # Strategy comparison table
            if hasattr(simulation_results, 'strategy_results') and simulation_results.strategy_results:
                report_lines.extend([
                    "### Strategy Performance Comparison",
                    "",
                    "| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Trades | Win Rate | Final Value |",
                    "|----------|--------------|--------------|--------------|--------|----------|-------------|"
                ])

                # Sort strategies by performance
                strategy_performance = []
                for strategy_name, metrics in simulation_results.strategy_results.items():
                    strategy_performance.append({
                        'name': format_strategy_name(strategy_name),
                        'return': metrics.get('total_return_pct', 0),
                        'sharpe': metrics.get('sharpe_ratio', 0),
                        'drawdown': metrics.get('max_drawdown_pct', 0),
                        'trades': metrics.get('total_trades', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'final_value': metrics.get('final_value', 0)
                    })

                strategy_performance.sort(key=lambda x: x['sharpe'], reverse=True)

                for strategy in strategy_performance:
                    report_lines.append(
                        f"| {strategy['name']} | {strategy['return']:.2f}% | {strategy['sharpe']:.3f} | "
                        f"{strategy['drawdown']:.2f}% | {strategy['trades']} | {strategy['win_rate']:.1f}% | "
                        f"${strategy['final_value']:,.0f} |"
                    )

                report_lines.extend([""])

            # Technical indicators analysis if included
            if self.config.include_technical_indicators and technical_analysis:
                report_lines.extend([
                    "### Technical Indicator Analysis",
                    "",
                    f"**Indicators Analyzed**: {', '.join(self.config.technical_indicators).upper()}",
                    "",
                    "Technical indicators provide additional market context and comparative signals:",
                    ""
                ])

                for indicator in self.config.technical_indicators:
                    indicator_name = indicator.upper()
                    report_lines.extend([
                        f"- **{indicator_name}**: Standard parameters, systematic buy/sell signals",
                        f"  - Used as benchmark for regime-based approach comparison",
                        f"  - Provides traditional technical analysis perspective",
                        ""
                    ])

            # Risk analysis
            report_lines.extend([
                "### Risk Analysis",
                "",
                f"**Risk-Adjusted Performance**: The Sharpe ratio of {simulation_results.sharpe_ratio:.3f} indicates ",
                f"{'strong' if simulation_results.sharpe_ratio > 1.5 else 'moderate' if simulation_results.sharpe_ratio > 1.0 else 'modest'} ",
                f"risk-adjusted returns relative to volatility.",
                "",
                f"**Drawdown Analysis**: Maximum drawdown of {simulation_results.max_drawdown_pct:.2f}% demonstrates ",
                f"{'excellent' if simulation_results.max_drawdown_pct < 5 else 'good' if simulation_results.max_drawdown_pct < 10 else 'moderate'} ",
                f"downside protection during adverse market conditions.",
                "",
                f"**Trade Efficiency**: {simulation_results.total_trades} total trades with {getattr(simulation_results, 'win_rate', 0):.1f}% win rate ",
                f"indicates {'high' if getattr(simulation_results, 'win_rate', 0) > 60 else 'moderate' if getattr(simulation_results, 'win_rate', 0) > 50 else 'selective'} ",
                f"trading frequency and execution success.",
                ""
            ])

        # Add comprehensive methodology section
        report_lines.extend([
            "## Advanced Methodology & Model Architecture",
            "",
            "### Financial-First Regime Detection Framework",
            "",
            "This analysis employs a sophisticated financial-first approach that revolutionizes traditional regime detection:",
            "",
            f"**1. Hidden Markov Model Foundation** ({self.config.n_regimes} States)",
            f"- **Training Period**: {self.config.training_days} days of historical data",
            f"- **Observation Model**: Gaussian emissions capturing daily log returns",
            f"- **State Transitions**: Markov property with estimated transition probabilities",
            f"- **Parameter Estimation**: Baum-Welch expectation-maximization algorithm",
            "",
            "**2. Financial Regime Characterization** (Key Innovation)",
            "",
            "Unlike traditional approaches that assign arbitrary labels (Bear/Bull), this system:",
            "- Analyzes actual financial metrics: returns, volatility, win rates, persistence",
            "- Classifies regimes based on empirical performance characteristics",
            "- Calculates regime strength and confidence scores",
            "- Determines optimal trading strategies per regime type",
            "",
            "**3. Intelligent Signal Generation**",
            "",
            f"Signals are generated using regime characteristics rather than naive state assumptions:",
            "- **Regime-Based Signals**: Position sizing based on regime type and confidence",
            "- **Adaptive Scaling**: Signal strength varies with regime uncertainty",
            "- **Duration Modeling**: Consider regime persistence for position timing",
            "- **Risk Management**: Integrate regime confidence into position sizing",
            "",
            "**4. Single-Asset Trading Simulation**",
            "",
            f"Optimized for dedicated capital allocation:",
            f"- **Initial Capital**: ${self.config.initial_capital:,.0f} dedicated to {self.config.ticker}",
            f"- **Position Sizing**: Up to {self.config.max_position_pct:.0%} allocation capability",
            f"- **Transaction Costs**: {self.config.transaction_cost_type.title()} (retail-friendly)",
            f"- **Risk Controls**: {self.config.stop_loss_pct:.1%} stop-loss protection",
            "",
            "**5. Performance Evaluation Framework**",
            "",
            "Comprehensive metrics beyond simple returns:",
            "- **Risk-Adjusted Returns**: Sharpe and Sortino ratios",
            "- **Drawdown Analysis**: Maximum and average drawdown periods",
            "- **Trade Analysis**: Win rates, holding periods, P&L distribution",
            "- **Regime Attribution**: Performance attribution by market regime",
            "",
            "### Model Validation & Robustness",
            "",
            f"**Data Quality**: {len(self.market_data)} days of market data validated",
            f"**Training Stability**: {self.config.hmm_max_iterations} max iterations with {self.config.hmm_tolerance} tolerance",
            f"**Out-of-Sample Testing**: Analysis period separate from training data",
            f"**Regime Stability**: Minimum {self.config.min_regime_days} days required for reliable characterization",
            "",
            "### Key Advantages Over Traditional Approaches",
            "",
            "1. **Financial Intelligence**: Regime labels based on actual market behavior",
            "2. **Uncertainty Quantification**: Confidence intervals on all regime predictions",
            "3. **Adaptive Position Sizing**: Signal strength varies with regime confidence",
            "4. **Zero-Cost Optimization**: Designed for commission-free trading environments",
            "5. **Single-Asset Focus**: Eliminates portfolio dilution for concentrated analysis",
            ""
        ])

        # Add current market assessment
        if current_regime['regime_type'] != 'Unknown':
            report_lines.extend([
                "## Current Market Assessment",
                "",
                f"**Current Regime**: {current_regime['regime_type'].title()}",
                f"**Confidence Level**: {current_regime['confidence']:.1%}",
                f"**Expected Annual Return**: {current_regime['expected_return']:.1%}",
                f"**Expected Annual Volatility**: {current_regime['volatility']:.1%}",
                "",
                "**Trading Implications**:",
                ""
            ])

            if current_regime['regime_type'] == 'bullish':
                report_lines.extend([
                    "- Consider increased long exposure",
                    "- Monitor for trend continuation signals",
                    "- Maintain stop-losses below recent support levels",
                ])
            elif current_regime['regime_type'] == 'bearish':
                report_lines.extend([
                    "- Consider defensive positioning",
                    "- Monitor for potential reversal signals",
                    "- Tighten stop-losses and risk management",
                ])
            elif current_regime['regime_type'] == 'sideways':
                report_lines.extend([
                    "- Consider range-trading strategies",
                    "- Monitor support and resistance levels",
                    "- Reduce position sizes due to limited directional bias",
                ])
            else:
                report_lines.extend([
                    "- Exercise caution with position sizing",
                    "- Monitor for regime transitions",
                    "- Consider diversified approaches",
                ])

            report_lines.extend([""])

        # Disclaimers and limitations
        report_lines.extend([
            "---",
            "",
            "## Important Disclaimers",
            "",
            "**Educational Purpose**: This analysis is for educational and research purposes only. Not financial advice.",
            "",
            "**Risk Warning**: Past performance does not guarantee future results. All trading involves risk of loss.",
            "",
            "**Model Limitations**: HMM models assume regime persistence and may not capture sudden structural changes.",
            "",
            "**Market Conditions**: Analysis based on historical data and may not reflect future market conditions.",
            "",
            f"**Generated by**: Hidden Regime Financial Analysis System v2.0 on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])

        # Save report
        report_content = "\n".join(report_lines)
        report_path = f"{output_structure['reports']}/{self.config.ticker}_financial_analysis_report.md"

        with open(report_path, 'w') as f:
            f.write(report_content)

        print(f"   📝 Report saved: {report_path}")
        print(f"   ✅ Report generation completed")

        return report_content

    def _create_signal_generators(self) -> Dict[str, Any]:
        """Create signal generators based on configured strategies."""
        generators = {}

        for strategy in self.config.signal_strategies:
            if strategy == 'hmm_regime_following' or strategy == 'financial_regime_following':
                generators[strategy] = FinancialSignalGenerator(
                    strategy_type='regime_following',
                    min_confidence=self.config.regime_confidence_threshold
                )
            elif strategy == 'hmm_regime_contrarian' or strategy == 'financial_regime_contrarian':
                generators[strategy] = FinancialSignalGenerator(
                    strategy_type='regime_contrarian',
                    min_confidence=self.config.regime_confidence_threshold
                )
            elif strategy == 'financial_confidence_weighted':
                generators[strategy] = FinancialSignalGenerator(
                    strategy_type='confidence_weighted',
                    min_confidence=self.config.regime_confidence_threshold
                )
            elif strategy == 'adaptive_financial':
                generators[strategy] = AdaptiveSignalGenerator(
                    min_confidence=self.config.regime_confidence_threshold
                )
            elif strategy == 'buy_and_hold':
                from ..simulation.signal_generators import BuyHoldSignalGenerator
                generators[strategy] = BuyHoldSignalGenerator()

        return generators

    def _get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of generated trading signals."""
        if self.trading_signals is None or self.trading_signals.empty:
            return {}

        summary = {}
        for strategy in self.trading_signals.columns:
            signals = self.trading_signals[strategy]
            summary[strategy] = {
                'total_signals': len(signals),
                'buy_signals': (signals > 0.1).sum(),
                'sell_signals': (signals < -0.1).sum(),
                'hold_signals': (abs(signals) <= 0.1).sum(),
                'avg_signal_strength': abs(signals).mean(),
                'max_long_signal': signals.max(),
                'max_short_signal': signals.min()
            }

        return summary

    def _save_trade_journals(self, simulation_result) -> None:
        """Save trade journals and related data to files."""
        try:
            # Create output directory structure
            output_structure = self.config.create_output_structure()
            journal_dir = output_structure['trade_journals']

            # Save strategy results with trade journals
            if hasattr(simulation_result, 'strategy_results') and simulation_result.strategy_results:
                for strategy_name, strategy_data in simulation_result.strategy_results.items():
                    # Save individual strategy trade journal
                    if 'trade_journal' in strategy_data:
                        trade_journal = strategy_data['trade_journal']
                        if hasattr(trade_journal, 'get_all_trades'):
                            trades = trade_journal.get_all_trades()
                            if trades:
                                # Convert trades to DataFrame for easy saving
                                trades_data = []
                                for trade in trades:
                                    trades_data.append({
                                        'strategy': strategy_name,
                                        'symbol': trade.symbol,
                                        'shares': trade.shares,
                                        'entry_price': trade.entry_price,
                                        'exit_price': trade.exit_price,
                                        'entry_date': trade.entry_date,
                                        'exit_date': trade.exit_date,
                                        'pnl': trade.pnl,
                                        'pnl_pct': trade.pnl_pct,
                                        'hold_days': trade.hold_days,
                                        'exit_reason': trade.exit_reason,
                                        'total_cost': trade.shares * trade.entry_price
                                    })

                                trades_df = pd.DataFrame(trades_data)

                                # Save as CSV
                                csv_path = os.path.join(journal_dir, f"{strategy_name}_trades.csv")
                                trades_df.to_csv(csv_path, index=False)

                                # Save as JSON for more detailed data
                                json_path = os.path.join(journal_dir, f"{strategy_name}_trades.json")
                                trades_df.to_json(json_path, orient='records', date_format='iso', indent=2)

                                print(f"   💾 {strategy_name}: {len(trades)} trades saved to {csv_path}")

            # Save best strategy summary
            if hasattr(simulation_result, 'best_strategy') and simulation_result.best_strategy:
                summary_data = {
                    'analysis_info': {
                        'ticker': self.config.ticker,
                        'analysis_period': f"{self.config.start_date} to {self.config.end_date}",
                        'generated_at': datetime.now().isoformat(),
                        'initial_capital': self.config.initial_capital,
                        'best_strategy': simulation_result.best_strategy
                    },
                    'performance_summary': {
                        'total_return_pct': simulation_result.total_return_pct,
                        'annualized_return': getattr(simulation_result, 'annualized_return', 0.0),
                        'sharpe_ratio': simulation_result.sharpe_ratio,
                        'sortino_ratio': getattr(simulation_result, 'sortino_ratio', 0.0),
                        'max_drawdown_pct': simulation_result.max_drawdown_pct,
                        'total_trades': simulation_result.total_trades,
                        'win_rate': getattr(simulation_result, 'win_rate', 0.0),
                        'final_value': getattr(simulation_result, 'final_value', 0.0)
                    }
                }

                summary_path = os.path.join(journal_dir, f"{self.config.ticker}_simulation_summary.json")
                import json
                with open(summary_path, 'w') as f:
                    json.dump(summary_data, f, indent=2)

                print(f"   📋 Simulation summary saved to {summary_path}")

            # Save portfolio history if available
            if hasattr(simulation_result, 'strategy_results'):
                for strategy_name, strategy_data in simulation_result.strategy_results.items():
                    if 'portfolio_history' in strategy_data:
                        portfolio_history = strategy_data['portfolio_history']
                        if portfolio_history:
                            portfolio_df = pd.DataFrame(portfolio_history)
                            portfolio_path = os.path.join(journal_dir, f"{strategy_name}_portfolio_history.csv")
                            portfolio_df.to_csv(portfolio_path, index=False)
                            print(f"   📈 {strategy_name}: Portfolio history saved to {portfolio_path}")

        except Exception as e:
            print(f"   ⚠️ Trade journal saving warning: {e}")
            import traceback
            traceback.print_exc()