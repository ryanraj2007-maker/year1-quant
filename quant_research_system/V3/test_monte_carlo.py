# =============================================================================
# V2 — Monte Carlo Test Script
# =============================================================================
# Exercises the full V2 core pipeline with a small set of hand-crafted trades.
# Used to verify that trade_log, performance, plots, and monte_carlo all work
# together correctly before running a real backtest.
#
# Run from the V2/ directory:
#   python test_monte_carlo.py
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt

from core.trade_log import create_trade, trades_to_dataframe
from core.performance import summary_stats
from core.plots import plot_all
import core.monte_carlo as mc


# ── Mock trade log ────────────────────────────────────────────────────────────
# Four hand-crafted trades covering the main outcomes a strategy can produce:
#   Trade 1: Long winner  — exits above entry before target (+2R)
#   Trade 2: Short winner — exits below entry before target (+2R)
#   Trade 3: Long loser   — exits at stop (−1R)
#   Trade 4: Short loser  — exits beyond stop due to slippage (−1.5R)

trades_list = [
    # Long: entry 100, stop 99 → risk = 1pt. Exit at 102 → pnl = 2pts = +2R
    create_trade(
        entry_time=pd.Timestamp("2024-01-01 09:30"),
        exit_time=pd.Timestamp("2024-01-01 10:00"),
        direction="long",
        entry_price=100.0,
        exit_price=102.0,
        stop_price=99.0,
        target_price=104.0,
    ),
    # Short: entry 105, stop 106 → risk = 1pt. Exit at 103 → pnl = 2pts = +2R
    create_trade(
        entry_time=pd.Timestamp("2024-01-02 09:30"),
        exit_time=pd.Timestamp("2024-01-02 10:15"),
        direction="short",
        entry_price=105.0,
        exit_price=103.0,
        stop_price=106.0,
        target_price=101.0,
    ),
    # Long: entry 110, stop 109 → risk = 1pt. Exit at 109 (stopped out) = −1R
    create_trade(
        entry_time=pd.Timestamp("2024-01-03 09:30"),
        exit_time=pd.Timestamp("2024-01-03 11:00"),
        direction="long",
        entry_price=110.0,
        exit_price=109.0,
        stop_price=109.0,
        target_price=112.0,
    ),
    # Short: entry 120, stop 121 → risk = 1pt. Exit at 121.5 (stop with
    # slippage — filled 0.5pts beyond stop) → pnl = 120 - 121.5 = -1.5 = −1.5R
    create_trade(
        entry_time=pd.Timestamp("2024-01-04 09:30"),
        exit_time=pd.Timestamp("2024-01-04 10:30"),
        direction="short",
        entry_price=120.0,
        exit_price=121.5,
        stop_price=121.0,
        target_price=117.0,
    ),
]

# Convert list of Trade objects to a DataFrame for the analysis modules
trades = trades_to_dataframe(trades_list)

print("Trade Log:")
print(trades)
print()

# ── Performance summary ───────────────────────────────────────────────────────
stats = summary_stats(trades)
print("Performance Summary:")
print(stats)
print()

# ── Monte Carlo simulation ────────────────────────────────────────────────────
# With only 4 trades this is a toy example, but it tests the pipeline end-to-end.
# In real use, you'd have 100+ trades for meaningful simulation results.

simulations = mc.run_monte_carlo(
    trades=trades,
    n_simulations=200,
    n_trades=50,         # simulate longer sequences than the 4 real trades
    starting_equity=1.0,
)

print("Monte Carlo Simulations Shape:", simulations.shape)
print()

mc_stats = mc.final_equity_stats(simulations)
print("Monte Carlo Final Equity Stats:")
print(mc_stats)
print()

mc_summary = mc.monte_carlo_summary(
    trades=trades,
    n_simulations=500,
    n_trades=50,
    starting_equity=1.0,
)
print("Monte Carlo Summary:")
print(mc_summary)
print("Opening plots now... Close each plot window to continue.")
print()

# ── Plots ─────────────────────────────────────────────────────────────────────
# plot_all() creates equity curve, drawdown, and R-distribution figures.
# plot_monte_carlo_paths() creates the fan chart of simulated paths.
# plt.show() is called once here so all windows open simultaneously.

plot_all(trades)
mc.plot_monte_carlo_paths(simulations, n_paths_to_plot=100)

plt.show()
