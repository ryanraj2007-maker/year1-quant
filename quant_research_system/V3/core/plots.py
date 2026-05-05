# V2 Core — Visualisation
# All plot functions create figures but don't call plt.show().
# Call plt.show() once at the end of main.py so all charts open together.

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import os

from core.performance import equity_curve_from_r


def plot_equity_curve(trades: pd.DataFrame, starting_equity: float = 1.0) -> None:
    """Cumulative R equity curve across all trades."""
    eq = equity_curve_from_r(trades, starting_equity=starting_equity)
    plt.figure(figsize=(10, 5))
    plt.plot(eq.index, eq["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity (R Units)")
    plt.grid(True)


def plot_drawdown_curve(trades: pd.DataFrame, starting_equity: float = 1.0) -> None:
    """Drawdown at each trade — how far below peak equity the strategy is."""
    eq = equity_curve_from_r(trades, starting_equity=starting_equity)
    plt.figure(figsize=(10, 5))
    plt.plot(eq.index, eq["drawdown"])
    plt.title("Drawdown Curve")
    plt.xlabel("Trade Number")
    plt.ylabel("Drawdown (R Units)")
    plt.grid(True)


def plot_r_distribution(trades: pd.DataFrame) -> None:
    """Histogram of R-multiples across all trades."""
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
    style: str = "bands"
) -> None:
    """
    Overlay the actual equity curve on Monte Carlo simulation results.
    style="bands" draws shaded percentile bands.
    style="paths" draws individual simulation paths as faint lines.
    """
    eq = equity_curve_from_r(trades, starting_equity=starting_equity)

    p5  = simulations.quantile(0.05, axis=0)
    p25 = simulations.quantile(0.25, axis=0)
    p75 = simulations.quantile(0.75, axis=0)
    p95 = simulations.quantile(0.95, axis=0)

    trade_steps = range(len(p5))

    plt.figure(figsize=(12, 6))

    if style == "bands":
        plt.fill_between(trade_steps, p5, p95, alpha=0.15, color="steelblue", label="p5–p95")
        plt.fill_between(trade_steps, p25, p75, alpha=0.25, color="steelblue", label="p25–p75")
    elif style == "paths":
        for i in range(len(simulations)):
            plt.plot(simulations.iloc[i].values, alpha=0.05, color="steelblue", linewidth=0.5)

    plt.plot(eq.index, eq["equity"], color="black", linewidth=1.5, label="Actual backtest")
    plt.title("Equity Curve vs Monte Carlo")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity (R Units)")
    plt.legend()
    plt.grid(True)


def plot_win_loss_streak(trades: pd.DataFrame) -> None:
    """Bar chart of running win/loss streak at each trade."""
    streaks = []
    current = 0

    for win in trades["win"]:
        if win == 1:
            current = current + 1 if current > 0 else 1
        else:
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
    """Save all open figures to disk as PNGs. Call before plt.show()."""
    os.makedirs(output_dir, exist_ok=True)
    for i, fig in enumerate(map(plt.figure, plt.get_fignums()), start=1):
        filepath = os.path.join(output_dir, f"figure_{i}.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")


def plot_all(
    trades: pd.DataFrame,
    starting_equity: float = 1.0,
    simulations: pd.DataFrame = None,
    style: str = "bands"
) -> None:
    """Generate all standard plots. Caller handles plt.show()."""
    if simulations is not None:
        plot_equity_curve_with_mc_bands(trades, simulations, starting_equity=starting_equity, style=style)
    plot_equity_curve(trades, starting_equity=starting_equity)
    plot_drawdown_curve(trades, starting_equity=starting_equity)
    plot_r_distribution(trades)
    plot_win_loss_streak(trades)
