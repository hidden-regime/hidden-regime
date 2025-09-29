"""
Unified configuration for financial regime analysis.

Provides single configuration entry point for complete financial market
regime detection, analysis, and trading simulation workflow.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timedelta
import pandas as pd

from ..config.base import BaseConfig
from ..utils.exceptions import ConfigurationError


@dataclass(frozen=True)
class FinancialRegimeConfig(BaseConfig):
    """
    Unified configuration for complete financial regime analysis.

    Combines regime detection, characterization, signal generation,
    and trading simulation into single financial-focused configuration.
    """

    # Market data configuration
    ticker: str = "NVDA"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-01"
    data_source: str = "yfinance"

    # Regime detection configuration
    n_regimes: int = 3
    training_days: int = 252  # One year of training data
    regime_detection_method: str = "hmm"  # Future: support other methods

    # HMM-specific parameters
    hmm_tolerance: float = 1e-6
    hmm_max_iterations: int = 1000
    hmm_random_seed: int = 42
    hmm_initialization: str = "kmeans"

    # Regime characterization
    min_regime_days: int = 5  # Minimum days for reliable characterization
    regime_confidence_threshold: float = 0.6  # Minimum confidence for full position

    # Trading signal generation
    signal_strategies: List[str] = field(default_factory=lambda: [
        'hmm_regime_following',
        'buy_and_hold',
        'adaptive_financial'
    ])

    # Single-asset trading simulation (optimized for dedicated capital)
    enable_simulation: bool = True
    initial_capital: float = 100000.0  # $100k default
    enable_shorting: bool = True
    max_position_pct: float = 1.0  # 100% allocation for single-asset
    stop_loss_pct: float = 0.05  # 5% stop-loss

    # Transaction costs (retail-friendly defaults)
    transaction_cost_type: Literal["free", "flat", "percentage"] = "free"
    transaction_cost_flat: float = 0.0  # Flat fee per trade
    transaction_cost_percentage: float = 0.0  # Percentage of trade value

    # Risk management (through regime confidence, not artificial position caps)
    regime_transition_delay: int = 1  # Days to wait before acting on regime change
    crisis_cash_allocation: float = 1.0  # Percentage to cash in crisis regimes
    mixed_regime_scaling: float = 0.5  # Position scaling for unclear regimes

    # Technical analysis integration
    include_technical_indicators: bool = True
    technical_indicators: List[str] = field(default_factory=lambda: [
        'rsi', 'macd', 'sma', 'bollinger_bands'
    ])

    # Analysis and comparison
    benchmark_comparison: bool = True
    benchmark_strategy: str = 'buy_and_hold'
    calculate_risk_metrics: bool = True
    include_drawdown_analysis: bool = True

    # Visualization configuration
    generate_visualizations: bool = True
    color_scheme: str = "colorblind_safe"
    plot_style: Literal["professional", "academic", "presentation"] = "professional"
    create_animations: bool = True
    animation_fps: int = 2
    save_individual_frames: bool = False

    # Output configuration
    output_directory: Optional[str] = None
    save_intermediate_results: bool = True
    generate_comprehensive_report: bool = True
    save_trade_journal: bool = True

    def __post_init__(self):
        """Post-initialization validation and default setting."""
        # Set default output directory if not provided
        if self.output_directory is None:
            object.__setattr__(self, 'output_directory', f"./output/financial_analysis/{self.ticker}_{self.start_date}_{self.end_date}")

    def validate(self) -> None:
        """Validate financial configuration parameters."""
        super().validate()

        # Validate ticker
        if not self.ticker or len(self.ticker.strip()) == 0:
            raise ConfigurationError("ticker cannot be empty")

        # Validate dates
        try:
            start_dt = pd.to_datetime(self.start_date)
            end_dt = pd.to_datetime(self.end_date)
        except ValueError as e:
            raise ConfigurationError(f"Invalid date format: {e}")

        if start_dt >= end_dt:
            raise ConfigurationError(f"start_date {self.start_date} must be before end_date {self.end_date}")

        # Validate training period
        if self.training_days <= 0:
            raise ConfigurationError(f"training_days must be positive, got {self.training_days}")

        if self.training_days < 30:
            raise ConfigurationError(f"training_days should be at least 30 days for meaningful analysis, got {self.training_days}")

        # Validate regimes
        if self.n_regimes < 2 or self.n_regimes > 10:
            raise ConfigurationError(f"n_regimes must be between 2 and 10, got {self.n_regimes}")

        # Validate capital allocation
        if self.initial_capital <= 0:
            raise ConfigurationError(f"initial_capital must be positive, got {self.initial_capital}")

        # Validate position sizing (should allow 100% for single-asset)
        if not 0 < self.max_position_pct <= 1.0:
            raise ConfigurationError(f"max_position_pct must be between 0 and 1, got {self.max_position_pct}")

        # Validate risk management
        if not 0 < self.stop_loss_pct <= 1.0:
            raise ConfigurationError(f"stop_loss_pct must be between 0 and 1, got {self.stop_loss_pct}")

        if not 0 <= self.regime_confidence_threshold <= 1.0:
            raise ConfigurationError(f"regime_confidence_threshold must be between 0 and 1, got {self.regime_confidence_threshold}")

        # Validate transaction costs
        if self.transaction_cost_flat < 0:
            raise ConfigurationError(f"transaction_cost_flat cannot be negative, got {self.transaction_cost_flat}")

        if not 0 <= self.transaction_cost_percentage <= 1.0:
            raise ConfigurationError(f"transaction_cost_percentage must be between 0 and 1, got {self.transaction_cost_percentage}")

        # Validate technical indicators
        valid_indicators = [
            'rsi', 'macd', 'sma', 'ema', 'bollinger_bands', 'stochastic',
            'williams_r', 'atr', 'adx', 'aroon', 'cci', 'roc', 'vwap'
        ]
        for indicator in self.technical_indicators:
            if indicator not in valid_indicators:
                raise ConfigurationError(f"Unknown technical indicator: {indicator}. Valid indicators: {valid_indicators}")

        # Validate signal strategies
        valid_strategies = [
            'hmm_regime_following', 'hmm_regime_contrarian', 'hmm_confidence_weighted',
            'financial_regime_following', 'financial_regime_contrarian', 'financial_confidence_weighted',
            'adaptive_financial', 'buy_and_hold'
        ]
        for strategy in self.signal_strategies:
            if strategy not in valid_strategies:
                raise ConfigurationError(f"Unknown signal strategy: {strategy}. Valid strategies: {valid_strategies}")

    def get_training_date_range(self) -> tuple[str, str]:
        """
        Calculate the training date range based on start_date and training_days.

        Returns:
            Tuple of (training_start_date, training_end_date) as strings
        """
        start_dt = pd.to_datetime(self.start_date)

        # Calculate training start with buffer for weekends
        training_start = start_dt - pd.Timedelta(days=int(self.training_days * 1.4))
        training_start = pd.bdate_range(end=start_dt, periods=self.training_days + 1)[0]

        # Training ends the day before evaluation starts
        training_end = start_dt - pd.Timedelta(days=1)

        return training_start.strftime('%Y-%m-%d'), training_end.strftime('%Y-%m-%d')

    def get_analysis_period_days(self) -> int:
        """Get total number of days in the analysis period."""
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        return (end_dt - start_dt).days

    def create_output_structure(self) -> Dict[str, str]:
        """
        Create output directory structure for financial analysis results.

        Returns:
            Dictionary mapping output types to directory paths
        """
        import os

        base_dir = self.output_directory
        structure = {
            'base': base_dir,
            'reports': os.path.join(base_dir, 'reports'),
            'plots': os.path.join(base_dir, 'plots'),
            'animations': os.path.join(base_dir, 'animations'),
            'data': os.path.join(base_dir, 'data'),
            'trade_journals': os.path.join(base_dir, 'trade_journals'),
            'frames': os.path.join(base_dir, 'frames') if self.save_individual_frames else None
        }

        # Create directories
        for dir_path in structure.values():
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

        return structure

    def get_simulation_config_dict(self) -> Dict[str, Any]:
        """Get simulation configuration as dictionary for SimulationConfig."""
        return {
            'initial_capital': self.initial_capital,
            'enable_shorting': self.enable_shorting,
            'max_position_pct': self.max_position_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'transaction_cost': self.transaction_cost_flat if self.transaction_cost_type == "flat" else 0.0,
            'save_trade_journal': self.save_trade_journal,
            'save_daily_signals': True,
            'save_portfolio_history': True
        }

    def get_summary_info(self) -> Dict[str, Any]:
        """
        Get summary information about the financial analysis configuration.

        Returns:
            Dictionary with configuration summary information
        """
        training_start, training_end = self.get_training_date_range()

        return {
            'asset': self.ticker,
            'analysis_period': {
                'start': self.start_date,
                'end': self.end_date,
                'total_days': self.get_analysis_period_days()
            },
            'training_period': {
                'start': training_start,
                'end': training_end,
                'training_days': self.training_days
            },
            'regime_detection': {
                'n_regimes': self.n_regimes,
                'method': self.regime_detection_method,
                'min_confidence': self.regime_confidence_threshold
            },
            'trading_simulation': {
                'enabled': self.enable_simulation,
                'initial_capital': self.initial_capital,
                'max_position': f"{self.max_position_pct:.0%}",
                'transaction_costs': self.transaction_cost_type,
                'shorting_enabled': self.enable_shorting
            },
            'signal_strategies': self.signal_strategies,
            'technical_indicators': self.technical_indicators,
            'output_directory': self.output_directory
        }

    @classmethod
    def create_quick_analysis(
        cls,
        ticker: str = "SPY",
        days_back: int = 90,
        n_regimes: int = 3,
        initial_capital: float = 50000.0
    ) -> 'FinancialRegimeConfig':
        """
        Create configuration for quick financial regime analysis.

        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to analyze
            n_regimes: Number of regime states
            initial_capital: Starting capital for simulation

        Returns:
            FinancialRegimeConfig for quick analysis
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        return cls(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            training_days=max(60, days_back // 2),
            n_regimes=n_regimes,
            initial_capital=initial_capital,
            create_animations=False,  # Quick analysis skips animations
            include_technical_indicators=False,
            signal_strategies=['hmm_regime_following', 'buy_and_hold']
        )

    @classmethod
    def create_comprehensive_analysis(
        cls,
        ticker: str,
        start_date: str,
        end_date: str,
        n_regimes: int = 4,
        initial_capital: float = 100000.0
    ) -> 'FinancialRegimeConfig':
        """
        Create configuration for comprehensive financial regime analysis.

        Args:
            ticker: Stock ticker symbol
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
            n_regimes: Number of regime states
            initial_capital: Starting capital for simulation

        Returns:
            FinancialRegimeConfig for comprehensive analysis
        """
        return cls(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            training_days=252,  # One year of training
            n_regimes=n_regimes,
            initial_capital=initial_capital,
            include_technical_indicators=True,
            create_animations=True,
            signal_strategies=[
                'hmm_regime_following',
                'adaptive_financial',
                'buy_and_hold'
            ],
            technical_indicators=['rsi', 'macd', 'sma', 'bollinger_bands'],
            calculate_risk_metrics=True,
            generate_comprehensive_report=True,
            save_intermediate_results=True
        )

    @classmethod
    def create_single_asset_study(
        cls,
        ticker: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        aggressive: bool = False
    ) -> 'FinancialRegimeConfig':
        """
        Create configuration optimized for single-asset regime analysis.

        Args:
            ticker: Stock ticker symbol
            start_date: Analysis start date
            end_date: Analysis end date
            initial_capital: Dedicated capital for this asset
            aggressive: Whether to use aggressive position sizing

        Returns:
            FinancialRegimeConfig optimized for single-asset analysis
        """
        return cls(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            max_position_pct=1.0,  # Full allocation for single asset
            enable_shorting=True,
            transaction_cost_type="free",  # Retail-friendly
            signal_strategies=[
                'hmm_regime_following',
                'adaptive_financial',
                'buy_and_hold'
            ] + (['hmm_regime_contrarian'] if aggressive else []),
            regime_confidence_threshold=0.5 if aggressive else 0.6,
            stop_loss_pct=0.08 if aggressive else 0.05,
            color_scheme="colorblind_safe"
        )

    def create_component(self) -> Any:
        """Create FinancialRegimeAnalysis component from this configuration."""
        from .analysis import FinancialRegimeAnalysis
        return FinancialRegimeAnalysis(self)