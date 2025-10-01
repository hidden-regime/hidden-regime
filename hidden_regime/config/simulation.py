"""
Configuration classes for trading simulation system.

Provides configuration for simulation parameters, risk management,
and signal generation as specified in simulation.md.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from ..config.base import BaseConfig
from ..utils.exceptions import ConfigurationError


@dataclass(frozen=True)
class SimulationConfig(BaseConfig):
    """
    Configuration for trading simulation engine.

    Defines parameters for capital allocation, risk management, signal generation,
    and performance analysis in the trading simulation system.
    """

    # Capital and position management
    initial_capital: Optional[float] = (
        None  # If None, calculated from default_shares * first_price
    )
    default_shares: int = 100  # Default number of shares when capital not specified
    enable_shorting: bool = True  # Whether to allow short selling
    transaction_cost: float = 0.0  # Cost per trade (flat fee)

    # Risk management (optimized for single-asset dedicated capital)
    max_position_pct: float = 1.0  # Maximum 100% allocation for single-asset analysis
    max_portfolio_risk: float = 0.02  # Maximum 2% portfolio risk per trade
    stop_loss_pct: float = 0.05  # Default 5% stop-loss
    max_total_exposure: float = (
        1.0  # Maximum 100% portfolio exposure (appropriate for single asset)
    )
    max_drawdown_pct: float = 0.2  # Maximum 20% drawdown before reducing size

    # Signal generation
    signal_generators: List[str] = field(
        default_factory=lambda: ["buy_and_hold", "hmm_regime_following"]
    )
    hmm_strategy_types: List[str] = field(
        default_factory=lambda: ["regime_following"]
    )  # regime_following, regime_contrarian, confidence_weighted
    technical_indicators: List[str] = field(
        default_factory=lambda: ["rsi", "macd", "sma"]
    )  # TA library indicators to include

    # Performance analysis
    risk_free_rate: float = 0.02  # Annual risk-free rate for Sharpe calculation
    benchmark_strategy: str = "buy_and_hold"  # Strategy to use as benchmark
    calculate_trade_metrics: bool = (
        True  # Whether to calculate detailed trade-level metrics
    )

    # Output configuration
    save_trade_journal: bool = True  # Save detailed trade journal
    save_daily_signals: bool = True  # Save daily signal DataFrame
    save_portfolio_history: bool = True  # Save daily portfolio values
    create_performance_charts: bool = True  # Generate performance visualization

    # Advanced simulation options
    enable_trailing_stops: bool = False  # Enable trailing stop-loss
    trailing_stop_pct: float = 0.05  # Trailing stop percentage
    enable_volatility_stops: bool = False  # Enable volatility-based stops
    volatility_multiplier: float = 2.0  # Volatility multiplier for dynamic stops

    def validate(self) -> None:
        """Validate simulation configuration parameters."""
        super().validate()

        # Validate capital parameters
        if self.initial_capital is not None and self.initial_capital <= 0:
            raise ConfigurationError(
                f"initial_capital must be positive, got {self.initial_capital}"
            )

        if self.default_shares <= 0:
            raise ConfigurationError(
                f"default_shares must be positive, got {self.default_shares}"
            )

        if self.transaction_cost < 0:
            raise ConfigurationError(
                f"transaction_cost cannot be negative, got {self.transaction_cost}"
            )

        # Validate risk management parameters
        if not 0 < self.max_position_pct <= 1:
            raise ConfigurationError(
                f"max_position_pct must be between 0 and 1, got {self.max_position_pct}"
            )

        if not 0 < self.max_portfolio_risk <= 1:
            raise ConfigurationError(
                f"max_portfolio_risk must be between 0 and 1, got {self.max_portfolio_risk}"
            )

        if not 0 < self.stop_loss_pct <= 1:
            raise ConfigurationError(
                f"stop_loss_pct must be between 0 and 1, got {self.stop_loss_pct}"
            )

        if not 0 < self.max_total_exposure <= 2:
            raise ConfigurationError(
                f"max_total_exposure must be between 0 and 2, got {self.max_total_exposure}"
            )

        if not 0 < self.max_drawdown_pct <= 1:
            raise ConfigurationError(
                f"max_drawdown_pct must be between 0 and 1, got {self.max_drawdown_pct}"
            )

        # Validate signal generators
        valid_base_generators = [
            "buy_and_hold",
            "hmm_regime_following",
            "hmm_regime_contrarian",
            "hmm_confidence_weighted",
            "adaptive_financial",
        ]
        for generator in self.signal_generators:
            if (
                not generator.startswith("ta_")
                and generator not in valid_base_generators
            ):
                raise ConfigurationError(f"Unknown signal generator: {generator}")

        # Validate HMM strategy types
        valid_hmm_strategies = [
            "regime_following",
            "regime_contrarian",
            "confidence_weighted",
        ]
        for strategy in self.hmm_strategy_types:
            if strategy not in valid_hmm_strategies:
                raise ConfigurationError(f"Unknown HMM strategy type: {strategy}")

        # Validate technical indicators
        valid_indicators = [
            "rsi",
            "macd",
            "sma",
            "ema",
            "bollinger_bands",
            "stochastic",
            "williams_r",
            "atr",
            "adx",
            "aroon",
            "cci",
            "mfi",
            "roc",
            "vwap",
            "keltner_channels",
        ]
        for indicator in self.technical_indicators:
            if indicator not in valid_indicators:
                raise ConfigurationError(f"Unknown technical indicator: {indicator}")

        # Validate performance parameters
        if self.risk_free_rate < 0 or self.risk_free_rate > 1:
            raise ConfigurationError(
                f"risk_free_rate must be between 0 and 1, got {self.risk_free_rate}"
            )

        # Validate trailing stop parameters
        if self.enable_trailing_stops:
            if not 0 < self.trailing_stop_pct <= 1:
                raise ConfigurationError(
                    f"trailing_stop_pct must be between 0 and 1, got {self.trailing_stop_pct}"
                )

        # Validate volatility stop parameters
        if self.enable_volatility_stops:
            if self.volatility_multiplier <= 0:
                raise ConfigurationError(
                    f"volatility_multiplier must be positive, got {self.volatility_multiplier}"
                )

    @classmethod
    def create_conservative(cls) -> "SimulationConfig":
        """Create conservative simulation configuration for single-asset analysis."""
        return cls(
            max_position_pct=0.5,  # 50% max position (conservative for single asset)
            max_portfolio_risk=0.01,  # 1% max risk
            stop_loss_pct=0.03,  # 3% stop-loss
            max_total_exposure=0.5,  # 50% max exposure
            signal_generators=["buy_and_hold", "hmm_regime_following"],
            hmm_strategy_types=["regime_following"],
            technical_indicators=["sma", "rsi"],
        )

    @classmethod
    def create_aggressive(cls) -> "SimulationConfig":
        """Create aggressive simulation configuration for single-asset analysis."""
        return cls(
            max_position_pct=1.0,  # 100% max position (full allocation for single asset)
            max_portfolio_risk=0.05,  # 5% max risk
            stop_loss_pct=0.08,  # 8% stop-loss
            max_total_exposure=1.5,  # 150% max exposure (leverage allowed)
            enable_shorting=True,
            signal_generators=[
                "buy_and_hold",
                "hmm_regime_following",
                "hmm_regime_contrarian",
            ],
            hmm_strategy_types=[
                "regime_following",
                "regime_contrarian",
                "confidence_weighted",
            ],
            technical_indicators=[
                "rsi",
                "macd",
                "sma",
                "ema",
                "bollinger_bands",
                "stochastic",
            ],
            enable_trailing_stops=True,
            enable_volatility_stops=True,
        )

    @classmethod
    def create_comprehensive(cls) -> "SimulationConfig":
        """Create comprehensive simulation configuration for single-asset thorough testing."""
        return cls(
            max_position_pct=1.0,  # 100% max position (full allocation for single asset)
            max_portfolio_risk=0.02,  # 2% max risk
            stop_loss_pct=0.05,  # 5% stop-loss
            max_total_exposure=1.0,  # 100% max exposure
            signal_generators=[
                "buy_and_hold",
                "hmm_regime_following",
                "hmm_regime_contrarian",
                "hmm_confidence_weighted",
            ],
            hmm_strategy_types=[
                "regime_following",
                "regime_contrarian",
                "confidence_weighted",
            ],
            technical_indicators=[
                "rsi",
                "macd",
                "sma",
                "ema",
                "bollinger_bands",
                "stochastic",
                "williams_r",
                "atr",
                "adx",
                "aroon",
                "cci",
                "roc",
                "vwap",
            ],
            calculate_trade_metrics=True,
            save_trade_journal=True,
            save_daily_signals=True,
            save_portfolio_history=True,
            create_performance_charts=True,
        )

    def get_risk_limits(self) -> Dict[str, float]:
        """Get risk limits as dictionary for RiskManager."""
        return {
            "max_position_pct": self.max_position_pct,
            "max_portfolio_risk": self.max_portfolio_risk,
            "stop_loss_pct": self.stop_loss_pct,
            "max_total_exposure": self.max_total_exposure,
            "max_drawdown_pct": self.max_drawdown_pct,
        }

    def get_signal_generator_configs(self) -> List[Dict[str, Any]]:
        """Get signal generator configurations."""
        configs = []

        # Add base signal generators
        for generator in self.signal_generators:
            if generator == "buy_and_hold":
                configs.append({"type": "buy_hold", "name": "buy_and_hold"})
            elif generator.startswith("hmm_"):
                strategy_type = generator.replace("hmm_", "")
                if strategy_type in self.hmm_strategy_types:
                    configs.append(
                        {
                            "type": "hmm",
                            "name": generator,
                            "strategy_type": strategy_type,
                        }
                    )

        # Add technical indicator generators
        for indicator in self.technical_indicators:
            configs.append(
                {
                    "type": "technical",
                    "name": f"ta_{indicator}",
                    "indicator_name": indicator,
                }
            )

        return configs

    def create_component(self) -> Any:
        """Create simulation orchestrator from this configuration."""
        from ..simulation.simulation_orchestrator import SimulationOrchestrator

        return SimulationOrchestrator(self)


@dataclass(frozen=True)
class SimulationOutputConfig(BaseConfig):
    """
    Configuration for simulation output and reporting.

    Defines what outputs to generate and where to save them.
    """

    # Output directory structure
    output_directory: str = "./output/simulations"
    create_subdirectories: bool = True  # Create dated subdirectories

    # File formats
    save_formats: List[Literal["csv", "json", "parquet"]] = field(
        default_factory=lambda: ["csv", "json"]
    )

    # Data to save
    save_trade_journal: bool = True
    save_daily_signals: bool = True
    save_portfolio_history: bool = True
    save_performance_metrics: bool = True
    save_risk_metrics: bool = True

    # Charts and visualizations
    create_performance_charts: bool = True
    create_drawdown_charts: bool = True
    create_trade_analysis_charts: bool = True
    create_signal_comparison_charts: bool = True

    # Report generation
    generate_summary_report: bool = True
    include_detailed_analysis: bool = True
    report_format: Literal["markdown", "html", "pdf"] = "markdown"

    def validate(self) -> None:
        """Validate output configuration."""
        super().validate()

        # Validate save formats
        valid_formats = ["csv", "json", "parquet"]
        for fmt in self.save_formats:
            if fmt not in valid_formats:
                raise ConfigurationError(
                    f"Invalid save format: {fmt}. Must be one of {valid_formats}"
                )

        # Validate report format
        valid_report_formats = ["markdown", "html", "pdf"]
        if self.report_format not in valid_report_formats:
            raise ConfigurationError(
                f"Invalid report format: {self.report_format}. Must be one of {valid_report_formats}"
            )

    def create_component(self) -> Any:
        """Create output configuration component."""
        # This config doesn't create a specific component, it's just configuration
        return self


@dataclass
class SimulationResult:
    """
    Container for simulation results.

    Holds all simulation outputs including performance metrics,
    trade journals, and analysis results.
    """

    # Basic simulation info
    simulation_success: bool
    symbol: str
    start_date: str
    end_date: str

    # Capital and returns
    initial_capital: float
    final_value: float
    total_return: float
    total_return_pct: float

    # Performance metrics
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float

    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float

    # Strategy comparison
    strategy_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    best_strategy: Optional[str] = None
    benchmark_comparison: Optional[Dict[str, float]] = None

    # Detailed data
    trade_journal: Optional[Any] = None
    portfolio_history: Optional[List[Dict]] = None
    daily_signals: Optional[Any] = None  # DataFrame

    # Metadata
    simulation_config: Optional[SimulationConfig] = None
    execution_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert simulation result to dictionary for serialization."""
        return {
            "simulation_success": self.simulation_success,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_pnl": self.avg_trade_pnl,
            "strategy_results": self.strategy_results,
            "best_strategy": self.best_strategy,
            "benchmark_comparison": self.benchmark_comparison,
            "execution_time": self.execution_time,
        }
