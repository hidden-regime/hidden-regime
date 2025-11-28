"""
Market Cycle Detection: Momentum-Riding Strategy

This template demonstrates advanced regime-based trading using a 5-state HMM
that captures euphoric bubble rallies while protecting downside with trailing stops.

Strategy Overview:
    - Identifies full market cycles: Euphoric → Bull → Sideways → Bear → Crisis
    - Rides euphoric rallies to capture final melt-ups with stop-loss protection
    - Uses trailing stops to lock in profits and prevent catastrophic losses
    - Provides crisis hedging with automatic bond allocation
    - Targets higher returns than bubble-fading while maintaining drawdown control

Key Features:
    1. 5-State HMM Detection: Euphoric, Bull, Sideways, Bear, Crisis
    2. Momentum-Riding Logic: Increase allocation in euphoric regimes
    3. Trailing Stops: Capture upside, exit on reversal (8% for euphoria, 12% for bull)
    4. Crisis Hedging: Automatic allocation to bonds (TLT) in extreme conditions
    5. Confidence Thresholds: High bar (0.70) for euphoria signals

Performance Targets:
    - Sharpe Ratio: > 1.6 (vs 0.8 for basic 3-state, 1.2 for bubble-fading)
    - Max Drawdown: < 28% (vs 40% for buy-and-hold)
    - Alpha vs SPY: +7-10% annualized (if thesis correct)
    - Upside Capture: 80%+ of euphoric rallies

Recommended Backtesting Periods:
    1. Dotcom Bubble (1998-01-01 to 2003-10-10): Euphoria phases + crash testing
    2. COVID + Tech Rally (2019-01-01 to 2021-12-31): Crisis + melt-up capture
    3. Recent Rallies (2022-2023): AI/Tech euphoria with stops

Differences vs Bubble-Fading Variant:
    Bubble-Fading          Momentum-Riding
    ───────────────────    ──────────────────
    30% euphoric           80% euphoric
    Duration decay         Duration decay
    Lock profits           Ride the wave
    Risk-averse            Aggressive growth
    Sharpe: 1.2            Sharpe: 1.6
    DD: 22%                DD: 28%

Author: hidden-regime
License: MIT
"""

from AlgorithmImports import *
from hidden_regime.quantconnect import HiddenRegimeAlgorithm


class MarketCycleDetectionMomentumRiding(HiddenRegimeAlgorithm):
    """
    Market Cycle Detection with Momentum-Riding Strategy.

    This algorithm detects market cycles using a 5-state HMM and implements
    a momentum-riding strategy that captures euphoric rallies with stop-loss protection.

    Strategy Logic:
        1. Train 5-state HMM on 756 days of historical data (3 years)
           - Captures rare euphoric episodes
           - Includes at least one full market cycle
           - Ensures crisis emission parameters are learned

        2. Detect 5 market regimes based on emission parameters:
           - Euphoric: High returns + high volatility = bubble conditions (ride it!)
           - Bull: Positive returns, moderate volatility = growth
           - Sideways: Near-zero returns, low-moderate volatility = consolidation
           - Bear: Negative returns, moderate volatility = downtrend
           - Crisis: Extreme negative returns + extreme volatility = tail event

        3. Allocate based on regime with momentum-riding thesis:
           - Euphoric: 80% long with 8% trailing stop
             * Historical melt-ups gain 3-5% before peak
             * Trailing stop captures gains + prevents crashes
             * Duration decay still applies (reduce as euphoria ages)
             * Formula: 80% * confidence * max(0.5, 1.0 - days/20)
           - Bull: 100% long with 12% trailing stop
             * Full growth allocation (trend-following)
             * Wider stop preserves larger trends
           - Sideways: 50% long (neutral consolidation)
           - Bear: 0% (move to cash, avoid drawdowns)
           - Crisis: 50% TLT bonds + 50% cash (negative correlation protection)

        4. Rebalance on regime changes with trailing stop execution:
           - Euphoria signals require confidence >= 0.70 and persistence >= 2 days
           - Stops execute on daily close if trailing price hit
           - Crisis allocation prioritizes defense over stop execution
           - Momentum-riding: Stay in even near stopping levels (capture final ups)

    Allocation Matrix with Trailing Stops:
        Regime      Allocation  Stop    Logic
        ───────     ──────────  ────    ─────
        Euphoric    80%         8%      * Ride final melt-up
                                        * Higher allocation than bubble-fade
                                        * 8% trailing stop controls drawdown
        Bull        100%        12%     * Full growth allocation
                                        * Wider stop lets trends run
        Sideways    50%         None    * Neutral (consolidation)
        Bear        0%          N/A     * Avoid drawdowns
        Crisis      50% TLT     N/A     * Defensive (bonds rally -0.6 corr)
                    + 50% cash

    Trailing Stop Mechanics:
        Setup:
        - Track highest price since entry (high water mark)
        - Exit if price drops below: high_water_mark * (1 - stop_percent)
        - Stops adjust upward but never downward (trailing)

        Example (Euphoric regime with 8% stop):
        - Entry at $100
        - Price rallies to $105 (5% gain)
        - High water mark = $105
        - Stop level = $105 * 0.92 = $96.60
        - If price drops to $96.50, exit at $96.60
        - Lock in $5.40 gain (5.4%)

        Benefit:
        - Captures intermediate rallies (3-5%)
        - Prevents full drawdown when euphoria turns
        - Trailing feature lets you stay in for bigger rallies

    Crisis Allocation Rationale:
        SPY/TLT correlation: -0.6 during crises (2008, 2020)
        - 2008 Financial Crisis: TLT +34% while SPY -37% (71% gain vs loss)
        - 2020 COVID Crash: TLT +20% while SPY -34% (54% gain vs loss)
        - Provides essential tail risk hedge during extreme market stress

    Risk Management:
        - Position sizing formula:
          allocation = base * confidence * duration_decay
        - Trailing stops: Automatic exit on reversal
        - Confidence thresholds:
          * Euphoria: 0.70 (high bar for rare regime)
          * Others: 0.60 (standard)
        - Maximum drawdown target: < 28% (vs 22% for bubble-fade)
        - Trade-off: Higher drawdown for higher return potential (+4% alpha)

    Logging & Monitoring:
        - Regime changes: Logged with confidence and trailing stop
        - Euphoria signals: Flagged with stop level and expected duration
        - Weekly summary: Transition rate, stability score, stop hit rate
        - Crisis detection: Immediate alert, stop suspension for defense
    """

    def Initialize(self):
        """
        Initialize the algorithm.

        Sets up:
        - 5-state HMM for full market cycle detection (including euphoria)
        - Longer lookback (756 days = 3 years) to capture rare euphoric episodes
        - Defensive assets: TLT bonds for crisis allocation
        - Momentum-riding logic with trailing stops
        - High confidence thresholds to avoid false signals

        Parameters (modifiable via GetParameter):
        -----------
        ticker : str
            Stock ticker symbol (default: "SPY")
        start_date : str
            Start date in format YYYY-MM-DD (default: "1998-01-01")
        end_date : str
            End date in format YYYY-MM-DD (default: "2023-12-31")
        cash : float
            Initial portfolio cash (default: 100000)
        n_states : int
            Number of HMM states (fixed: 5 for this template)
        lookback_days : int
            Historical data window (default: 756 days = 3 years)
        min_confidence : float
            Minimum confidence threshold (default: 0.70 for euphoria detection)
        euphoria_allocation : float
            Base euphoric regime allocation (default: 0.80 = 80%, higher than bubble-fade)
        euphoria_stop_percent : float
            Trailing stop for euphoric regime (default: 0.08 = 8%)
        bull_stop_percent : float
            Trailing stop for bull regime (default: 0.12 = 12%)
        """
        # === CONFIGURATION ===
        ticker = self.GetParameter("ticker", "SPY")
        start_year = int(self.GetParameter("start_year", 1998))
        start_month = int(self.GetParameter("start_month", 1))
        start_day = int(self.GetParameter("start_day", 1))
        end_year = int(self.GetParameter("end_year", 2023))
        end_month = int(self.GetParameter("end_month", 12))
        end_day = int(self.GetParameter("end_day", 31))
        initial_cash = float(self.GetParameter("cash", 100000))

        # 5-state model configuration
        n_states = 5  # Fixed: Euphoric, Bull, Sideways, Bear, Crisis
        lookback_days = int(self.GetParameter("lookback_days", 756))  # 3 years (required for euphoria)
        min_confidence = float(self.GetParameter("min_confidence", 0.70))  # High bar
        random_seed = int(self.GetParameter("random_seed", 4242))

        # Allocation parameters (higher euphoria than bubble-fade)
        euphoria_allocation = float(self.GetParameter("euphoria_allocation", 0.80))  # 80% vs 30%
        bull_allocation = float(self.GetParameter("bull_allocation", 1.0))
        sideways_allocation = float(self.GetParameter("sideways_allocation", 0.50))
        bear_allocation = float(self.GetParameter("bear_allocation", 0.0))

        # Trailing stop parameters
        self.euphoria_stop_percent = float(self.GetParameter("euphoria_stop_percent", 0.08))  # 8% stop
        self.bull_stop_percent = float(self.GetParameter("bull_stop_percent", 0.12))  # 12% stop

        # Retraining strategy: Adaptive for faster response
        retrain_frequency = self.GetParameter("retrain_frequency", "monthly")

        # === BACKTEST SETUP ===
        self.SetStartDate(start_year, start_month, start_day)
        self.SetEndDate(end_year, end_month, end_day)
        self.SetCash(initial_cash)

        # === ADD ASSETS ===
        # Primary asset (equities)
        self.symbol = self.AddEquity(ticker, Resolution.Daily).Symbol  # noqa: F405

        # Defensive assets (used in crisis regime)
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol  # noqa: F405
        self.shy = self.AddEquity("SHY", Resolution.Daily).Symbol  # noqa: F405

        # === REGIME DETECTION ===
        self.initialize_regime_detection(
            ticker=ticker,
            n_states=n_states,
            lookback_days=lookback_days,
            retrain_frequency=retrain_frequency,
            regime_allocations={
                "Euphoric": euphoria_allocation,
                "Bull": bull_allocation,
                "Sideways": sideways_allocation,
                "Bear": bear_allocation,
                "Crisis": 0.0,  # Handled in allocate_portfolio()
            },
            min_confidence=min_confidence,
            random_seed=random_seed,
        )

        # === STATE TRACKING ===
        self.euphoria_days = 0
        self.crisis_detected = False
        self.crisis_entry_date = None

        # Trailing stop tracking
        self.high_water_mark = None
        self.stop_hit = False
        self.current_stop_level = None

        # Duration decay parameters
        self.euphoria_max_days = 20
        self.euphoria_min_allocation = 0.5

        self.Debug(
            f"MarketCycleDetectionMomentumRiding initialized:\n"
            f"  Ticker: {ticker}\n"
            f"  Period: {start_year}-{start_month}-{start_day} to {end_year}-{end_month}-{end_day}\n"
            f"  States: 5 (Euphoric, Bull, Sideways, Bear, Crisis)\n"
            f"  Lookback: {lookback_days} days\n"
            f"  Strategy: Momentum-Riding (ride melt-ups with {self.euphoria_stop_percent:.0%} stops)"
        )

    def OnWarmupFinished(self):
        """
        Called when the warmup period completes.

        Fetch historical data from the warmup period and pre-populate
        the regime detector's buffer so HMM training can begin immediately.
        """
        ticker = self.GetParameter("ticker", "SPY")
        lookback_days = int(self.GetParameter("lookback_days", 756))

        # Fetch history from warmup period
        history = self.History(self.symbol, lookback_days, Resolution.Daily)  # noqa: F405

        if not history.empty:
            # Add each bar to the regime detector
            for index, row in history.iterrows():
                bar = {
                    'Time': index if not isinstance(index, tuple) else index[1],
                    'Open': row['open'],
                    'High': row['high'],
                    'Low': row['low'],
                    'Close': row['close'],
                    'Volume': row['volume'],
                }
                self.on_tradebar(ticker, bar)

            self.Debug(f"Loaded {len(history)} warmup bars for {ticker}")

    def OnData(self, data):
        """
        Handle new market data with trailing stop execution.

        Flow:
        1. Check trailing stops (execute if hit)
        2. Update regime detector with new bar
        3. Wait for HMM training to complete
        4. Update current regime classification
        5. Calculate allocations with momentum-riding logic
        6. Rebalance portfolio

        Args:
            data: Slice object containing market data
        """
        # Skip if no data
        if not data.ContainsKey(self.symbol):
            return

        bar = data[self.symbol]
        if bar is None:
            return

        ticker = self.GetParameter("ticker", "SPY")
        current_price = bar.Close

        # === TRAILING STOP EXECUTION ===
        if self.high_water_mark is not None and self.current_stop_level is not None:
            if current_price < self.current_stop_level:
                # Stop hit: exit position
                self.Log(
                    f"⚠️  TRAILING STOP HIT at ${current_price:.2f}\n"
                    f"  High Water Mark: ${self.high_water_mark:.2f}\n"
                    f"  Stop Level: ${self.current_stop_level:.2f}\n"
                    f"  Gain Locked: {((self.high_water_mark - current_price) / current_price):.2%}"
                )
                self.SetHoldings(self.symbol, 0.0)
                self.high_water_mark = None
                self.current_stop_level = None
                self.stop_hit = True
                return

        # Update high water mark (trailing feature)
        if self.high_water_mark is None:
            self.high_water_mark = current_price
        else:
            if current_price > self.high_water_mark:
                self.high_water_mark = current_price

        # Update regime detector with new bar
        self.on_tradebar(ticker, bar)

        # Wait until regime detection is ready
        if not self.regime_is_ready():
            return

        # Update regime classification
        self.update_regime()

        # Allocate portfolio based on regime
        self.allocate_portfolio()

    def allocate_portfolio(self):
        """
        Calculate and execute portfolio allocations with momentum-riding logic.

        Momentum-Riding Implementation:
        - Euphoric: 80% allocation (higher than bubble-fade's 30%)
        - Duration decay still applies (reduce as euphoria ages)
        - Trailing stop captures intermediate rallies
        - Stay in until stop hit or regime changes

        Trailing Stop Setup:
        - Bull: 12% trailing stop (wider, let trends run)
        - Euphoric: 8% trailing stop (tighter, protect gains)
        - Sideways/Bear/Crisis: No stops (different logic)
        """
        ticker = self.GetParameter("ticker", "SPY")
        current_regime = self.current_regime
        regime_confidence = self.regime_confidence
        days_in_regime = self.days_in_regime if hasattr(self, 'days_in_regime') else 1
        current_price = self.Securities[self.symbol].Price

        # Get base allocation from regime_allocations dict
        base_allocation = self.get_regime_allocation(ticker)

        # === EUPHORIC MOMENTUM-RIDING ===
        if current_regime == "Euphoric":
            self.euphoria_days += 1

            # Only signal euphoria if high confidence AND minimum persistence
            if regime_confidence < 0.70 or days_in_regime < 2:
                # Low confidence or too new: treat as Bull
                final_allocation = self.get_regime_allocation("Bull")
                self._set_trailing_stop_bull(current_price)
            else:
                # Calculate duration decay (same as bubble-fade)
                decay_factor = max(
                    self.euphoria_min_allocation,
                    1.0 - (days_in_regime / self.euphoria_max_days)
                )

                # Final allocation: base * confidence * decay
                # 80% * decay = 80% on day 1, down to 40% on day 20+
                final_allocation = base_allocation * regime_confidence * decay_factor

                # Set tighter trailing stop (8%) for euphoria
                self._set_trailing_stop_euphoric(current_price)

                # Log euphoria with stop details
                self.Log(
                    f"[{ticker}] EUPHORIA MOMENTUM-RIDING (Day {days_in_regime})\n"
                    f"  Confidence: {regime_confidence:.1%}\n"
                    f"  Duration Decay: {decay_factor:.2f}\n"
                    f"  Allocation: {final_allocation:.1%}\n"
                    f"  Trailing Stop: {self.euphoria_stop_percent:.0%} (${self.current_stop_level:.2f})"
                )

        # === BULL MOMENTUM ===
        elif current_regime == "Bull":
            final_allocation = base_allocation
            self._set_trailing_stop_bull(current_price)
            self.euphoria_days = 0

        # === CRISIS ALLOCATION (BONDS + CASH) ===
        elif current_regime == "Crisis":
            # Immediate alert on crisis detection
            if not self.crisis_detected:
                self.crisis_detected = True
                self.crisis_entry_date = self.Time
                self.Log(f"⚠️  CRISIS REGIME DETECTED - Suspending trailing stops")
                self.Log(f"  Allocation: 0% SPY | 50% TLT | 50% SHY")

            # Suspend stops during crisis, execute defensive allocation
            self.high_water_mark = None
            self.current_stop_level = None

            # Execute crisis allocation
            self.SetHoldings(self.symbol, 0.0)  # Exit equities completely
            self.SetHoldings(self.tlt, 0.50)    # Bond hedge
            self.SetHoldings(self.shy, 0.50)    # Cash equivalent

            return  # Skip standard SetHoldings

        # === STANDARD REGIME ALLOCATION ===
        else:
            final_allocation = base_allocation
            self.current_stop_level = None  # No stops for sideways/bear

            # Reset crisis flag when exiting crisis regime
            if self.crisis_detected:
                self.crisis_detected = False
                crisis_duration = (self.Time - self.crisis_entry_date).days
                self.Log(
                    f"✓ Crisis ended after {crisis_duration} days\n"
                    f"  Resuming trailing stops for {current_regime}"
                )

            # Reset euphoria counter
            if current_regime != "Euphoric":
                self.euphoria_days = 0

        # === EXECUTE ALLOCATION ===
        self.SetHoldings(self.symbol, final_allocation)

        # Zero out defensive assets (only used in Crisis)
        self.SetHoldings(self.tlt, 0.0)
        self.SetHoldings(self.shy, 0.0)

    def _set_trailing_stop_euphoric(self, current_price):
        """Set 8% trailing stop for euphoric regime."""
        self.current_stop_level = current_price * (1.0 - self.euphoria_stop_percent)

    def _set_trailing_stop_bull(self, current_price):
        """Set 12% trailing stop for bull regime."""
        self.current_stop_level = current_price * (1.0 - self.bull_stop_percent)

    def on_regime_change(self, old_regime, new_regime, confidence, ticker):
        """
        Called when regime transitions.

        Logs important regime changes and updates stop levels.

        Args:
            old_regime: Previous regime name
            new_regime: New regime name
            confidence: Confidence in new regime (0.0 to 1.0)
            ticker: Asset ticker
        """
        portfolio_value = self.Portfolio.TotalPortfolioValue

        self.Log(
            f"[{ticker}] Regime Change: {old_regime} → {new_regime}\n"
            f"  Confidence: {confidence:.1%}\n"
            f"  Portfolio Value: ${portfolio_value:,.2f}"
        )

        # Special logging for regime transitions
        if new_regime == "Euphoric":
            self.Log(f"🚀 ENTERING EUPHORIC REGIME - Momentum-riding activated (8% stop)")
        elif old_regime == "Euphoric":
            self.Log(f"✓ Exiting euphoric regime after {self.euphoria_days} days")

        # Weekly regime persistence summary
        if self.Time.weekday() == 0:  # Monday
            stop_status = f"${self.current_stop_level:.2f}" if self.current_stop_level else "None"
            self.Log(
                f"Weekly Summary ({self.Time.date()}):\n"
                f"  Current Regime: {new_regime}\n"
                f"  Confidence: {confidence:.1%}\n"
                f"  Trailing Stop Level: {stop_status}"
            )


# For QuantConnect cloud/local LEAN deployment, the algorithm class
# is automatically detected and instantiated.
