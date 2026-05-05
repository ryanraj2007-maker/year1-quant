# =============================================================================
# V2 Core — Visualisation
# =============================================================================
# Plotting functions for trade-level analysis. All plots use matplotlib and
# follow the same convention: create the figure but do NOT call plt.show().
# The caller (main script or test file) calls plt.show() once at the end so
# all figures open simultaneously.
#
# plot_all() is the standard entry point used by strategy runners.
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import os

from core.performance import equity_curve_from_r


def plot_equity_curve(trades: pd.DataFrame, starting_equity: float = 1.0) -> None:
    """
    Plot the cumulative R equity curve across all trades.

    The equity curve shows how the strategy grew (or shrank) trade by trade.
    A smooth upward slope = consistent edge. Flat/jagged = inconsistent results.
    The x-axis is trade number, not calendar time.
    """
    # Build the equity + drawdown DataFrame from R-multiples
    eq = equity_curve_from_r(trades, starting_equity=starting_equity)

    plt.figure(figsize=(10, 5))
    plt.plot(eq.index, eq["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity (R Units)")
    plt.grid(True)


def plot_drawdown_curve(trades: pd.DataFrame, starting_equity: float = 1.0) -> None:
    """
    Plot the drawdown at each trade — how far below the peak the strategy is.

    Drawdown = current equity - running maximum equity (always ≤ 0).
    The deeper the trough, the larger the drawdown.
    This is critical for prop firm analysis: a drawdown breach = instant fail.
    """
    eq = equity_curve_from_r(trades, starting_equity=starting_equity)

    plt.figure(figsize=(10, 5))
    plt.plot(eq.index, eq["drawdown"])
    plt.title("Drawdown Curve")
    plt.xlabel("Trade Number")
    plt.ylabel("Drawdown (R Units)")
    plt.grid(True)


def plot_r_distribution(trades: pd.DataFrame) -> None:
    """
    Histogram of R-multiples across all trades.

    Shows the shape of the trade outcome distribution:
    - A tight cluster around +2R / -1R = clean risk management
    - A fat left tail = occasional large losses (bad stops or gap risk)
    - Positive skew (right tail) = rare big winners
    """
    plt.figure(figsize=(10, 5))
    plt.hist(trades["r_multiple"], bins=20)
    plt.title("Distribution of Trade R-Multiples")
    plt.xlabel("R-Multiple")
    plt.ylabel("Frequency")
    plt.grid(True)


def plot_equity_curve_with_mc_bands(
    trades: pd.DataFrame,
    simulations: pd.DataFrame,
    starting_equity: float = 1.0,
    style="bands"
    ) -> None:
    """
    Overlay the real equity curve on top of Monte Carlo percentile bands.

    The shaded bands show the 5th–95th percentile range of simulated outcomes
    at each trade step. The actual backtest path is drawn on top in black.

    Seeing the real path near or outside the p5 band is a red flag —
    the backtest may be showing a lucky sequence, not the strategy's true edge.
    """
    eq = equity_curve_from_r(trades, starting_equity=starting_equity)

    # Percentile bands across all simulated paths at each trade step
    p5  = simulations.quantile(0.05, axis=0)
    p25 = simulations.quantile(0.25, axis=0)
    p75 = simulations.quantile(0.75, axis=0)
    p95 = simulations.quantile(0.95, axis=0)

    trade_steps = range(len(p5))

    plt.figure(figsize=(12, 6))
    if style == "bands":
        # Outer band: p5–p95 (light shading)
        plt.fill_between(trade_steps, p5, p95, alpha=0.15, color="steelblue", label="p5–p95 range")

        # Inner band: p25–p75 (darker shading)
        plt.fill_between(trade_steps, p25, p75, alpha=0.25, color="steelblue", label="p25–p75 range")
    elif style == "paths":
        for i in range(len(simulations)):
                plt.plot(simulations.iloc[i].values, alpha=0.05, color="steelblue", linewidth=0.5)  # very faint lines for all paths
        # Actual backtest equity curve
    plt.plot(eq.index, eq["equity"], color="black", linewidth=1.5, label="Actual backtest")

    plt.title("Equity Curve vs Monte Carlo Bands")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity (R Units)")
    plt.legend()
    plt.grid(True)


def plot_win_loss_streak(trades: pd.DataFrame) -> None:
    """
    Bar chart showing the running win/loss streak at each trade.

    Positive bars = consecutive wins at that point in the sequence.
    Negative bars = consecutive losses.

    Useful for spotting clustering — if losses are bunched together
    rather than randomly distributed, it may indicate regime dependency
    (the strategy breaks down in certain market conditions).
    """
    streaks = []
    current = 0

    for win in trades["win"]:
        if win == 1:
            # Extend win streak or start a new one from zero
            current = current + 1 if current > 0 else 1
        else:
            # Extend loss streak or start a new one from zero
            current = current - 1 if current < 0 else -1
        streaks.append(current)

    colours = ["green" if s > 0 else "red" for s in streaks]

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(streaks)), streaks, color=colours, width=0.8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Win / Loss Streak by Trade")
    plt.xlabel("Trade Number")
    plt.ylabel("Streak Length (+win / −loss)")
    plt.grid(True, axis="y")


def save_all_figures(output_dir: str = "results") -> None:
    """
    Save every currently open matplotlib figure to disk as a PNG.

    Figures are named figure_1.png, figure_2.png, etc. in output_dir.
    Creates the directory if it does not exist.

    Call this before plt.show() so figures are saved even if the window
    is closed without manually saving.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, fig in enumerate(map(plt.figure, plt.get_fignums()), start=1):
        filepath = os.path.join(output_dir, f"figure_{i}.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")


def plot_all(trades: pd.DataFrame, starting_equity: float = 1.0, simulations: pd.DataFrame = None, style: str = "bands") -> None:
    """
    Generate all standard plots in one call.

    Does not call plt.show() — the caller handles that so all figures
    open at the same time.
    """
    if simulations is not None:
        plot_equity_curve_with_mc_bands(trades, simulations, starting_equity=starting_equity, style=style)   
    plot_equity_curve(trades, starting_equity=starting_equity)
    plot_drawdown_curve(trades, starting_equity=starting_equity)
    plot_r_distribution(trades)
    plot_win_loss_streak(trades)
