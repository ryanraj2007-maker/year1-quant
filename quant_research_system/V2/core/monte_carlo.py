# V2 Core — Monte Carlo Simulation
# Bootstrapped resampling — randomly shuffles the trade sequence 1000 times
# to show the range of outcomes the strategy could produce, not just the one
# historical sequence that happened.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_equity_path(
    r_multiples: np.ndarray,
    n_trades: int,
    starting_equity: float = 1.0
) -> np.ndarray:
    """One simulated equity path — draws n_trades from the historical R set with replacement."""
    sampled_trades = np.random.choice(r_multiples, size=n_trades, replace=True)
    equity_path = starting_equity + np.cumsum(sampled_trades)
    return equity_path


def run_monte_carlo(
    trades: pd.DataFrame,
    n_simulations: int = 1000,
    n_trades: int | None = None,
    starting_equity: float = 1.0
) -> pd.DataFrame:
    """
    Run n_simulations equity paths.
    Returns a DataFrame with shape (n_simulations, n_trades) —
    each row is one complete path.
    """
    r_multiples = trades["r_multiple"].to_numpy()

    if n_trades is None:
        n_trades = len(r_multiples)

    simulation_results = []
    for _ in range(n_simulations):
        equity_path = simulate_equity_path(
            r_multiples=r_multiples,
            n_trades=n_trades,
            starting_equity=starting_equity
        )
        simulation_results.append(equity_path)

    return pd.DataFrame(simulation_results)


def final_equity_stats(simulations: pd.DataFrame) -> dict:
    """Distribution of final equity values across all simulated paths."""
    final_equities = simulations.iloc[:, -1]
    return {
        "mean_final_equity":   final_equities.mean(),
        "median_final_equity": final_equities.median(),
        "min_final_equity":    final_equities.min(),
        "max_final_equity":    final_equities.max(),
        "p5_final_equity":     final_equities.quantile(0.05),
        "p95_final_equity":    final_equities.quantile(0.95),
    }


def probability_of_ruin(
    simulations: pd.DataFrame,
    ruin_threshold: float = 0.0
) -> float:
    """
    Fraction of paths where equity ever hit or dropped below ruin_threshold.
    Default 0.0 = account went negative. For a 10% prop firm drawdown limit,
    pass 0.9 (starting from 1.0).
    """
    ever_ruined = (simulations <= ruin_threshold).any(axis=1)
    return ever_ruined.mean()


def drawdown_distribution(
    simulations: pd.DataFrame,
    starting_equity: float = 1.0
) -> dict:
    """Worst drawdown for each path, summarised as a distribution."""
    worst_drawdowns = []

    for i in range(len(simulations)):
        path = simulations.iloc[i].to_numpy()
        equity = np.concatenate([[starting_equity], path])
        rolling_max = np.maximum.accumulate(equity)
        pct_dd = (equity - rolling_max) / rolling_max * 100
        worst_drawdowns.append(pct_dd.min())

    worst_drawdowns = np.array(worst_drawdowns)

    return {
        "mean_worst_drawdown_pct":   worst_drawdowns.mean(),
        "median_worst_drawdown_pct": np.median(worst_drawdowns),
        "p5_worst_drawdown_pct":     np.percentile(worst_drawdowns, 5),
        "p95_worst_drawdown_pct":    np.percentile(worst_drawdowns, 95),
        "min_worst_drawdown_pct":    worst_drawdowns.min(),
    }


def plot_monte_carlo_paths(simulations: pd.DataFrame, n_paths_to_plot: int = 100) -> None:
    """Plot a sample of simulated paths."""
    plt.figure(figsize=(10, 6))
    n_paths = min(n_paths_to_plot, len(simulations))
    for i in range(n_paths):
        plt.plot(simulations.iloc[i], alpha=0.2)
    plt.title("Monte Carlo Simulated Equity Paths")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity (R Units)")
    plt.grid(True)


def monte_carlo_summary(
    trades: pd.DataFrame,
    n_simulations: int = 1000,
    n_trades: int | None = None,
    starting_equity: float = 1.0,
    ruin_threshold: float = 0.0,
    simulations: pd.DataFrame | None = None
) -> dict:
    """
    Full MC summary in one call. Pass in pre-computed simulations to avoid
    running them twice if you already called run_monte_carlo().
    """
    if simulations is None:
        simulations = run_monte_carlo(
            trades=trades,
            n_simulations=n_simulations,
            n_trades=n_trades,
            starting_equity=starting_equity
        )

    stats     = final_equity_stats(simulations)
    dd_dist   = drawdown_distribution(simulations, starting_equity=starting_equity)
    prob_ruin = probability_of_ruin(simulations, ruin_threshold=ruin_threshold)

    return {
        "n_simulations":       n_simulations,
        "n_trades":            n_trades if n_trades is not None else len(trades),
        "probability_of_ruin": prob_ruin,
        **stats,
        **dd_dist,
    }
