#!/usr/bin/env python3
"""
Hidden Regime Debug CSV Analysis Suite

Diagnoses why high win rates produce low Sharpe ratios by analyzing:
1. Regime-specific performance
2. Confidence calibration
3. Regime transition risk
4. Regime persistence effects
5. Drawdown analysis
6. Actionable recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional


class RegimeDebugAnalyzer:
    """
    Comprehensive diagnostics for HMM regime detection performance.
    Analyzes why 80% win rate might produce only 0.2-1.2 Sharpe ratio.
    """

    def __init__(self, csv_path: str, output_dir: str = "debug_analysis"):
        """
        Initialize analyzer with debug CSV export.

        Parameters
        ----------
        csv_path : str
            Path to timesteps.csv from backtest debug output
        output_dir : str
            Directory for output CSVs and PNG plots
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load data
        print(f"Loading {csv_path}...")
        self.df = pd.read_csv(csv_path)

        # Parse dates
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        elif self.df.index.name == "timestamp":
            self.df.index = pd.to_datetime(self.df.index)

        print(f"Loaded {len(self.df)} observations")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")

        # Validate required columns
        required = ["Close", "regime_label", "signal_confidence"]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Calculate returns if not present
        if "returns" not in self.df.columns:
            self.df["returns"] = self.df["Close"].pct_change()
            print("Calculated returns from Close prices")

    def run_full_analysis(self) -> None:
        """Execute complete diagnostic suite."""
        print("\n" + "="*80)
        print("HIDDEN REGIME DEBUG ANALYSIS SUITE")
        print("="*80)

        # 1. Overall summary
        print("\n### STEP 1: OVERALL PERFORMANCE ###")
        self.print_overall_summary()

        # 2. Regime analysis
        print("\n### STEP 2: REGIME-SEGMENTED ANALYSIS ###")
        regime_analysis = self.analyze_returns_by_regime()
        regime_analysis.to_csv(self.output_dir / "regime_analysis.csv")
        print(regime_analysis)

        # 3. Confidence calibration
        print("\n### STEP 3: CONFIDENCE CALIBRATION ###")
        conf_analysis = self.analyze_returns_by_confidence()
        conf_analysis.to_csv(self.output_dir / "confidence_analysis.csv")
        print(conf_analysis)

        # 4. Regime transitions
        print("\n### STEP 4: REGIME TRANSITION RISK ###")
        transition_analysis = self.analyze_regime_transition_risk()
        transition_analysis.to_csv(self.output_dir / "transition_analysis.csv")
        print(transition_analysis)

        # 5. Regime persistence
        print("\n### STEP 5: REGIME PERSISTENCE ANALYSIS ###")
        persistence_analysis = self.analyze_regime_persistence()
        persistence_analysis.to_csv(self.output_dir / "persistence_analysis.csv")
        print(persistence_analysis)

        # 6. Drawdown analysis
        print("\n### STEP 6: DRAWDOWN ANALYSIS ###")
        dd_periods, dd_summary = self.analyze_drawdowns_by_regime()
        dd_periods.to_csv(self.output_dir / "drawdown_periods.csv", index=False)
        dd_summary.to_csv(self.output_dir / "drawdown_summary.csv")

        # 7. Visualizations
        print("\n### STEP 7: GENERATING VISUALIZATIONS ###")
        self.plot_regime_cumulative_returns()
        self.plot_confidence_vs_returns()
        self.plot_rolling_sharpe_by_regime()

        # 8. Recommendations
        print("\n### STEP 8: GENERATING RECOMMENDATIONS ###")
        self.generate_recommendations(regime_analysis, conf_analysis, transition_analysis)

        print(f"\n✓ Analysis complete. Results saved to {self.output_dir}/")

    def print_overall_summary(self) -> None:
        """Print high-level performance metrics."""
        returns = self.df["returns"].dropna()

        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Win rate metrics
        win_rate = (returns > 0).sum() / len(returns)
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).any() else 0

        # Risk metrics
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else 0

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

        print(f"""
Overall Performance Summary
{'='*80}
Total Return:          {total_return:>10.2%}
Annualized Return:     {annualized_return:>10.2%}
Annualized Volatility: {annualized_vol:>10.2%}
Sharpe Ratio:          {sharpe_ratio:>10.2f}  {'[⚠️ LOW]' if sharpe_ratio < 1.0 else '[OK]' if sharpe_ratio < 2.0 else '[EXCELLENT]'}

Win Rate Metrics
{'='*80}
Win Rate:              {win_rate:>10.1%}  {'[EXCELLENT]' if win_rate > 0.75 else '[GOOD]' if win_rate > 0.60 else '[LOW]'}
Average Win:           {avg_win:>10.4f}
Average Loss:          {avg_loss:>10.4f}
Profit Factor:         {profit_factor:>10.2f}  {'[GOOD]' if profit_factor > 1.5 else '[LOW]'}
Expectancy:            {expectancy:>10.6f}  {'[POSITIVE]' if expectancy > 0 else '[NEGATIVE]'}

Risk Metrics
{'='*80}
Maximum Drawdown:      {max_drawdown:>10.2%}  {'[SEVERE]' if max_drawdown < -0.25 else '[HIGH]' if max_drawdown < -0.15 else '[ACCEPTABLE]'}
VaR (95%):             {var_95:>10.2%}
CVaR (95%):            {cvar_95:>10.2%}

DIAGNOSIS
{'='*80}""")

        if win_rate > 0.75 and sharpe_ratio < 1.2:
            print("🔴 HIGH WIN RATE, LOW SHARPE DETECTED")
            print("   Likely causes:")
            print("   1. Asymmetric payoff: Small wins, large losses")
            print("   2. Slow wins, fast losses (high return volatility)")
            print("   3. Regime-specific failures (rare but catastrophic)")
            print("   4. Poor position sizing (not scaling by volatility)")
            print("\n   → Review regime-segmented analysis below for specifics")
        elif win_rate > 0.75 and sharpe_ratio >= 2.0:
            print("✓✓ EXCELLENT: High win rate AND strong risk-adjusted returns")
        else:
            print(f"⚠️ Win rate: {win_rate:.1%}, Sharpe: {sharpe_ratio:.2f}")

    def analyze_returns_by_regime(self) -> pd.DataFrame:
        """Segment returns by regime and measure quality."""
        results = {}

        for regime in self.df["regime_label"].unique():
            regime_mask = self.df["regime_label"] == regime
            regime_returns = self.df.loc[regime_mask, "returns"].dropna()

            if len(regime_returns) == 0:
                continue

            # Basic stats
            count = len(regime_returns)
            mean_return = regime_returns.mean()
            median_return = regime_returns.median()
            std_return = regime_returns.std()

            # Sharpe components
            sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0

            # Sortino ratio (only downside volatility)
            down_returns = regime_returns[regime_returns < 0]
            down_vol = down_returns.std()
            sortino = (mean_return / down_vol * np.sqrt(252)) if down_vol > 0 else 0

            # Win rate metrics
            wins = (regime_returns > 0).sum()
            win_rate = wins / len(regime_returns)
            avg_win = regime_returns[regime_returns > 0].mean() if wins > 0 else 0
            avg_loss = regime_returns[regime_returns < 0].mean() if (regime_returns < 0).any() else 0

            # Profit factor
            if (regime_returns < 0).any():
                profit_factor = regime_returns[regime_returns > 0].sum() / abs(regime_returns[regime_returns < 0].sum())
            else:
                profit_factor = np.inf

            # Tail risk
            var_95 = regime_returns.quantile(0.05)
            cvar_95 = regime_returns[regime_returns <= var_95].mean() if (regime_returns <= var_95).any() else 0
            max_loss = regime_returns.min()
            max_win = regime_returns.max()

            # Expectancy
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

            results[regime] = {
                "count": count,
                "mean_return": mean_return,
                "median_return": median_return,
                "std_return": std_return,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "max_loss": max_loss,
                "max_win": max_win,
                "expectancy": expectancy,
            }

        return pd.DataFrame(results).T

    def analyze_returns_by_confidence(self) -> pd.DataFrame:
        """Check if signal confidence predicts return quality."""
        # Create confidence bins
        self.df["confidence_bin"] = pd.qcut(
            self.df["signal_confidence"],
            q=4,
            labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"],
            duplicates="drop"
        )

        results = {}
        for conf_level in sorted(self.df["confidence_bin"].unique()):
            conf_mask = self.df["confidence_bin"] == conf_level
            conf_returns = self.df.loc[conf_mask, "returns"].dropna()

            if len(conf_returns) == 0:
                continue

            count = len(conf_returns)
            mean_return = conf_returns.mean()
            std_return = conf_returns.std()
            sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            win_rate = (conf_returns > 0).sum() / len(conf_returns)
            max_dd = (conf_returns.cumsum() - conf_returns.cumsum().cummax()).min()

            results[str(conf_level)] = {
                "count": count,
                "mean_return": mean_return,
                "sharpe_ratio": sharpe,
                "win_rate": win_rate,
                "std_return": std_return,
                "max_drawdown": max_dd,
            }

        result_df = pd.DataFrame(results).T

        # Calculate correlation (ensure both arrays have same length)
        valid_mask = self.df["returns"].notna() & self.df["signal_confidence"].notna()
        if valid_mask.sum() > 1:
            corr, pval = spearmanr(
                self.df.loc[valid_mask, "signal_confidence"],
                self.df.loc[valid_mask, "returns"]
            )
            print(f"\nSpearman Correlation (Confidence vs Returns): {corr:.3f} (p={pval:.4f})")
            print(f"Interpretation: {'PREDICTIVE' if abs(corr) > 0.15 and pval < 0.05 else 'NOT PREDICTIVE'}")
        else:
            print("\nWarning: Not enough data for correlation analysis")

        return result_df

    def analyze_regime_transition_risk(self) -> pd.DataFrame:
        """Analyze performance around regime transitions."""
        # Detect regime changes
        self.df["regime_changed"] = (
            self.df["regime_label"] != self.df["regime_label"].shift(1)
        ).astype(int)

        # Create windows around transitions
        self.df["days_since_transition"] = self.df.groupby(
            (self.df["regime_changed"]).cumsum()
        ).cumcount()

        # Bin by days since transition
        self.df["transition_window"] = pd.cut(
            self.df["days_since_transition"],
            bins=[-1, 0, 2, 5, 10, np.inf],
            labels=["Day 0 (New)", "Days 1-2", "Days 3-5", "Days 6-10", "Days 10+"],
        )

        results = {}
        for window in sorted(self.df["transition_window"].unique()):
            if pd.isna(window):
                continue
            window_mask = self.df["transition_window"] == window
            window_returns = self.df.loc[window_mask, "returns"].dropna()

            if len(window_returns) == 0:
                continue

            count = len(window_returns)
            mean_return = window_returns.mean()
            sharpe = (
                (mean_return / window_returns.std() * np.sqrt(252))
                if window_returns.std() > 0
                else 0
            )
            win_rate = (window_returns > 0).sum() / len(window_returns)
            volatility = window_returns.std()

            results[str(window)] = {
                "count": count,
                "mean_return": mean_return,
                "sharpe_ratio": sharpe,
                "win_rate": win_rate,
                "volatility": volatility,
            }

        return pd.DataFrame(results).T

    def analyze_regime_persistence(self) -> pd.DataFrame:
        """Do stable regimes produce better returns than unstable ones?"""
        # Calculate regime run lengths
        self.df["regime_group"] = (
            (self.df["regime_label"] != self.df["regime_label"].shift(1)).cumsum()
        )
        self.df["regime_persistence"] = self.df.groupby("regime_group").cumcount() + 1

        # Bin by persistence
        self.df["persistence_bin"] = pd.cut(
            self.df["regime_persistence"],
            bins=[0, 1, 3, 7, 14, np.inf],
            labels=["Day 1 (New)", "Days 2-3", "Days 4-7", "Days 8-14", "Days 14+"],
        )

        results = {}
        for persistence in sorted(self.df["persistence_bin"].unique()):
            if pd.isna(persistence):
                continue
            pers_mask = self.df["persistence_bin"] == persistence
            pers_returns = self.df.loc[pers_mask, "returns"].dropna()

            if len(pers_returns) == 0:
                continue

            count = len(pers_returns)
            mean_return = pers_returns.mean()
            sharpe = (
                (mean_return / pers_returns.std() * np.sqrt(252))
                if pers_returns.std() > 0
                else 0
            )
            win_rate = (pers_returns > 0).sum() / len(pers_returns)
            volatility = pers_returns.std()

            results[str(persistence)] = {
                "count": count,
                "mean_return": mean_return,
                "sharpe_ratio": sharpe,
                "win_rate": win_rate,
                "volatility": volatility,
            }

        return pd.DataFrame(results).T

    def analyze_drawdowns_by_regime(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Identify worst drawdowns and which regimes they occur in."""
        # Calculate cumulative returns and drawdowns
        self.df["cumulative_return"] = (1 + self.df["returns"].fillna(0)).cumprod() - 1
        self.df["running_max"] = self.df["cumulative_return"].cummax()
        self.df["drawdown"] = self.df["cumulative_return"] - self.df["running_max"]

        # Identify drawdown periods
        self.df["in_drawdown"] = self.df["drawdown"] < 0
        self.df["drawdown_group"] = (
            (self.df["in_drawdown"] != self.df["in_drawdown"].shift(1)).cumsum()
        )

        # Analyze drawdown periods
        drawdown_data = []
        for dd_group, group_df in self.df[self.df["in_drawdown"]].groupby("drawdown_group"):
            if len(group_df) == 0:
                continue

            regime = group_df["regime_label"].mode()[0] if len(group_df["regime_label"].mode()) > 0 else "Unknown"

            drawdown_data.append({
                "max_drawdown": group_df["drawdown"].min(),
                "duration_days": len(group_df),
                "regime": regime,
                "start_date": group_df.index[0] if len(group_df) > 0 else None,
            })

        dd_periods = pd.DataFrame(drawdown_data).sort_values("max_drawdown")

        # Drawdown summary by regime
        regime_dd_summary = {}
        for regime in self.df["regime_label"].unique():
            regime_mask = self.df["regime_label"] == regime
            regime_dd = self.df.loc[regime_mask, "drawdown"]

            if len(regime_dd) == 0:
                continue

            regime_dd_summary[regime] = {
                "mean_drawdown": regime_dd[regime_dd < 0].mean() if (regime_dd < 0).any() else 0,
                "max_drawdown": regime_dd.min(),
                "num_drawdown_periods": (regime_dd < 0).sum(),
                "avg_recovery_days": (regime_dd < 0).sum() // max(1, ((regime_dd < 0).astype(int).diff() < 0).sum()),
            }

        dd_summary = pd.DataFrame(regime_dd_summary).T

        print("\nTop 10 Worst Drawdowns:")
        print(dd_periods.head(10).to_string())

        return dd_periods, dd_summary

    def plot_regime_cumulative_returns(self) -> None:
        """Plot cumulative returns by regime with drawdown visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        regimes = self.df["regime_label"].unique()[:4]

        for idx, regime in enumerate(regimes):
            ax = axes[idx // 2, idx % 2]
            regime_data = self.df[self.df["regime_label"] == regime].copy()

            if len(regime_data) == 0:
                ax.text(0.5, 0.5, f"No {regime} periods", ha="center", va="center")
                ax.set_title(f"{regime} Regime (N=0)")
                continue

            # Cumulative returns
            regime_data["cum_return"] = (1 + regime_data["returns"].fillna(0)).cumprod() - 1
            regime_data["running_max"] = regime_data["cum_return"].cummax()
            regime_data["drawdown"] = regime_data["cum_return"] - regime_data["running_max"]

            # Plot
            ax.plot(regime_data.index, regime_data["cum_return"], label="Cumulative Return", linewidth=2)
            ax.axhline(0, color="black", linestyle="--", alpha=0.3)

            # Drawdown fill
            ax.fill_between(regime_data.index, regime_data["drawdown"], 0, alpha=0.3, color="red", label="Drawdown")

            # Stats
            returns = regime_data["returns"].dropna()
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            max_dd = regime_data["drawdown"].min()
            win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

            ax.text(
                0.02,
                0.98,
                f"Sharpe: {sharpe:.2f}\nMax DD: {max_dd:.1%}\nWin Rate: {win_rate:.1%}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            ax.set_title(f"{regime} Regime (N={len(regime_data)})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative Return")
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "regime_cumulative_returns.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved: regime_cumulative_returns.png")

    def plot_confidence_vs_returns(self) -> None:
        """Plot confidence vs returns correlation."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Scatter
        ax1 = axes[0]
        colors = {
            "BULLISH": "green",
            "BEARISH": "red",
            "SIDEWAYS": "gray",
            "CRISIS": "black",
        }

        for regime in self.df["regime_label"].unique():
            regime_mask = self.df["regime_label"] == regime
            ax1.scatter(
                self.df.loc[regime_mask, "signal_confidence"],
                self.df.loc[regime_mask, "returns"],
                alpha=0.5,
                s=10,
                label=regime,
                color=colors.get(regime, "blue"),
            )

        # Calculate correlation with aligned data
        valid_mask = self.df["returns"].notna() & self.df["signal_confidence"].notna()
        if valid_mask.sum() > 1:
            corr, pval = spearmanr(
                self.df.loc[valid_mask, "signal_confidence"],
                self.df.loc[valid_mask, "returns"]
            )
        else:
            corr, pval = 0, 1

        ax1.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax1.set_xlabel("Signal Confidence", fontsize=12)
        ax1.set_ylabel("Return", fontsize=12)
        ax1.set_title(f"Confidence vs Returns (Correlation: {corr:.3f}, p={pval:.4f})", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: Boxplot by confidence quartile
        ax2 = axes[1]
        self.df["confidence_quartile"] = pd.qcut(
            self.df["signal_confidence"],
            q=4,
            labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"],
            duplicates="drop"
        )

        unique_quarters = sorted([q for q in self.df["confidence_quartile"].unique() if pd.notna(q)])
        bp_data = [self.df[self.df["confidence_quartile"] == q]["returns"].dropna().values for q in unique_quarters]
        if bp_data:
            ax2.boxplot(bp_data, labels=[str(q) for q in unique_quarters])
        ax2.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Confidence Quartile", fontsize=12)
        ax2.set_ylabel("Returns", fontsize=12)
        ax2.set_title("Return Distribution by Confidence Level", fontsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / "confidence_vs_returns.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved: confidence_vs_returns.png")

    def plot_rolling_sharpe_by_regime(self, window: int = 60) -> None:
        """Plot rolling Sharpe ratio by regime."""
        fig, ax = plt.subplots(figsize=(16, 8))

        # Calculate rolling Sharpe for each regime
        for regime in self.df["regime_label"].unique():
            regime_data = self.df[self.df["regime_label"] == regime].copy()

            if len(regime_data) < window:
                continue

            rolling_mean = regime_data["returns"].rolling(window=window).mean()
            rolling_std = regime_data["returns"].rolling(window=window).std()
            rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(252))

            ax.plot(regime_data.index, rolling_sharpe, label=f"{regime} ({window}-day)", linewidth=2)

        # Overall rolling Sharpe
        overall_mean = self.df["returns"].rolling(window=window).mean()
        overall_std = self.df["returns"].rolling(window=window).std()
        overall_sharpe = (overall_mean / overall_std * np.sqrt(252))

        ax.plot(
            self.df.index,
            overall_sharpe,
            label=f"Overall ({window}-day)",
            linewidth=3,
            color="black",
            linestyle="--",
            alpha=0.7,
        )

        # Reference lines
        ax.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax.axhline(1.0, color="green", linestyle="--", alpha=0.3, label="Sharpe=1.0 (Good)")
        ax.axhline(2.0, color="blue", linestyle="--", alpha=0.3, label="Sharpe=2.0 (Excellent)")

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(f"{window}-Day Rolling Sharpe Ratio", fontsize=12)
        ax.set_title("Rolling Sharpe Ratio by Regime Over Time", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "rolling_sharpe_by_regime.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved: rolling_sharpe_by_regime.png")

    def generate_recommendations(
        self,
        regime_df: pd.DataFrame,
        conf_df: pd.DataFrame,
        transition_df: pd.DataFrame,
    ) -> None:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Check for regime-specific failures
        if not regime_df.empty and "sharpe_ratio" in regime_df.columns:
            worst_regime = regime_df["sharpe_ratio"].idxmin()
            worst_sharpe = regime_df.loc[worst_regime, "sharpe_ratio"]
            worst_count = regime_df.loc[worst_regime, "count"] if "count" in regime_df.columns else 0

            if worst_sharpe < 0:
                recommendations.append(
                    f"🔴 CRITICAL: {worst_regime} regime has negative Sharpe ({worst_sharpe:.2f}) "
                    f"over {int(worst_count)} periods.\n"
                    f"   ACTION: Consider filtering out {worst_regime} signals entirely, or:\n"
                    f"   - Reduce position size by 50-75% in {worst_regime}\n"
                    f"   - Add regime stability filter (only trade after 5+ days in regime)"
                )

        # Check confidence calibration
        if not conf_df.empty and "sharpe_ratio" in conf_df.columns:
            conf_sharpe_range = conf_df["sharpe_ratio"].max() - conf_df["sharpe_ratio"].min()
            if conf_sharpe_range < 0.5:
                recommendations.append(
                    f"⚠️ Confidence scores are NOT predictive (Sharpe range: {conf_sharpe_range:.2f}).\n"
                    f"   ACTION: Do NOT use confidence for position sizing. Instead:\n"
                    f"   - Use fixed position sizing (ignore signal_confidence)\n"
                    f"   - Or use regime-based volatility targeting: position_size *= (target_vol / current_vol)\n"
                    f"   - Consider recalibrating confidence using forward-backward probabilities"
                )

        # Check transition risk
        if not transition_df.empty and "sharpe_ratio" in transition_df.columns:
            if "Day 0 (New)" in transition_df.index:
                transition_day0_sharpe = transition_df.loc["Day 0 (New)", "sharpe_ratio"]
                if transition_day0_sharpe < 0:
                    recommendations.append(
                        f"⚠️ Regime transitions have negative Sharpe ({transition_day0_sharpe:.2f}) on Day 0.\n"
                        f"   ACTION: Implement transition skip logic:\n"
                        f"   - Skip trading on Day 0 of regime change\n"
                        f"   - Wait for regime stability (2-3 days) before entering\n"
                        f"   - Use multi-timeframe alignment to filter false transitions"
                    )

        # Check profit factor
        if not regime_df.empty and "profit_factor" in regime_df.columns:
            overall_pf = regime_df["profit_factor"].mean()
            if overall_pf < 1.5 and not np.isinf(overall_pf):
                recommendations.append(
                    f"⚠️ Low profit factor ({overall_pf:.2f}) suggests poor win/loss asymmetry.\n"
                    f"   ACTION: Implement volatility-based position sizing:\n"
                    f"   - Calculate current volatility (20-day rolling std)\n"
                    f"   - Scale position_size *= (15% target_vol / current_volatility)\n"
                    f"   - This naturally reduces size during volatile periods (when losses are bigger)"
                )

        # Print and save recommendations
        print("\n" + "="*80)
        print("ACTIONABLE RECOMMENDATIONS")
        print("="*80)

        if not recommendations:
            print("\n✓ No critical issues identified. Consider these optimizations:")
            print("   - Implement volatility targeting for better risk-adjusted returns")
            print("   - Test regime-conditional position sizing")
            print("   - Verify confidence calibration with out-of-sample data")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec}")

        # Save recommendations
        with open(self.output_dir / "recommendations.txt", "w") as f:
            f.write("Hidden Regime Debug Analysis - Recommendations\n")
            f.write("="*80 + "\n\n")
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n\n")
            else:
                f.write("✓ No critical issues. Optimization recommendations:\n")
                f.write("   - Implement volatility targeting\n")
                f.write("   - Test regime-conditional position sizing\n")
                f.write("   - Verify confidence calibration\n")


def main():
    """Command-line entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_debug_csv.py <path/to/timesteps.csv> [output_dir]")
        print("\nExample:")
        print("  python analyze_debug_csv.py backtest_results/debug_SPY/timesteps.csv debug_analysis/")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "debug_analysis"

    try:
        analyzer = RegimeDebugAnalyzer(csv_path, output_dir)
        analyzer.run_full_analysis()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
