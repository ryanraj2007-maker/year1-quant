# =============================================================================
# V2 Core — Monte Carlo Simulation
# =============================================================================
# Uses bootstrapped resampling to simulate thousands of possible trade
# sequences and estimate the distribution of outcomes.
#
# Why Monte Carlo?
# A backtest gives you ONE sequence of trades — the exact order they happened
# historically. But in the future, the same trades could arrive in a different
# order. A long losing streak early can wipe you out even if the strategy is
# profitable overall. Monte Carlo shows you the RANGE of possible outcomes,
# not just the one that happened.
#
# Method: sample trades with replacement (bootstrapping).
# This preserves the real distribution of R-multiples while randomising order.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_equity_path(
    r_multiples: np.ndarray,
    n_trades: int,
    starting_equity: float = 1.0
) -> np.ndarray:
    """
    Simulate one possible equity path by resampling trades randomly.

    Draws n_trades from r_multiples with replacement — each draw is
    independent, so the same trade can appear multiple times (or not at all).
    The equity path starts at starting_equity and grows/falls additively.

    Returns an array of length n_trades representing equity after each trade.
    """
    # Sample n_trades R-multiples at random from the historical trade set
    sampled_trades = np.random.choice(r_multiples, size=n_trades, replace=True)

    # Accumulate them to build the equity path
    equity_path = starting_equity + np.cumsum(sampled_trades)

    return equity_path


def run_monte_carlo(
    trades: pd.DataFrame,
    n_simulations: int = 1000,
    n_trades: int | None = None,
    starting_equity: float = 1.0
) -> pd.DataFrame:
    """
    Run n_simulations equity paths and return them all as a DataFrame.

    Parameters
    ----------
    trades        : Trades DataFrame with an r_multiple column.
    n_simulations : How many simulated paths to generate. 1000 is fast;
                    10,000 gives a smoother distribution.
    n_trades      : How many trades to include in each simulated path.
                    Defaults to the number of actual trades in the backtest.
    starting_equity : Starting R equity level (default 1.0).

    Returns
    -------
    DataFrame where each row is one simulated path and each column is
    the equity after that many trades. Shape: (n_simulations, n_trades).
    """
    r_multiples = trades["r_multiple"].to_numpy()

    # Default: simulate the same number of trades as the actual backtest
    if n_trades is None:
        n_trades = len(r_multiples)

    simulation_results = []

    for sim_id in range(n_simulations):
        equity_path = simulate_equity_path(
            r_multiples=r_multiples,
            n_trades=n_trades,
            starting_equity=starting_equity
        )
        simulation_results.append(equity_path)

    return pd.DataFrame(simulation_results)


def final_equity_stats(simulations: pd.DataFrame) -> dict:
    """
    Summarise the distribution of final equity values across all simulations.

    The 5th percentile (p5) is particularly important — it represents the
    outcome in the worst 5% of simulated sequences. Use this to estimate
    worst-case scenarios for prop firm challenges.
    """
    # Last column = equity after all n_trades trades
    final_equities = simulations.iloc[:, -1]

    return {
        "mean_final_equity":   final_equities.mean(),
        "median_final_equity": final_equities.median(),
        "min_final_equity":    final_equities.min(),
        "max_final_equity":    final_equities.max(),
        "p5_final_equity":     final_equities.quantile(0.05),   # worst 5%
        "p95_final_equity":    final_equities.quantile(0.95),   # best 5%
    }


def probability_of_ruin(
    simulations: pd.DataFrame,
    ruin_threshold: float = 0.0
) -> float:
    """
    Fraction of simulated paths where equity ever fell to or below ruin_threshold.

    "Ruin" in a prop firm context is not necessarily zero — it could be the
    maximum drawdown limit (e.g. 10% drawdown = account blown). Pass the
    relevant threshold for your use case.

    Default ruin_threshold=0.0 means the account went negative (catastrophic ruin).
    For a prop firm with a 10% drawdown limit on a 1.0R starting equity, use 0.9.

    Returns a value between 0.0 and 1.0 (e.g. 0.12 = 12% of paths hit ruin).
    """
    # Check if any equity value in each path ever hit the threshold
    # simulations rows = paths, columns = trade steps
    ever_ruined = (simulations <= ruin_threshold).any(axis=1)
    return ever_ruined.mean()


def drawdown_distribution(
    simulations: pd.DataFrame,
    starting_equity: float = 1.0
) -> dict:
    """
    Compute the worst drawdown for each simulated path and summarise the distribution.

    For each path, drawdown is computed as the largest peak-to-trough decline
    expressed as a percentage of the peak equity at that point.

    This tells you not just what the worst case was in the backtest, but the
    full distribution of worst-case drawdowns across all simulated sequences.
    Key output for prop firm challenge modelling.
    """
    worst_drawdowns = []

    for i in range(len(simulations)):
        path = simulations.iloc[i].to_numpy()

        # Prepend starting equity so drawdown from the very first trade is captured
        equity = np.concatenate([[starting_equity], path])

        rolling_max = np.maximum.accumulate(equity)

        # Percentage drawdown at each step
        pct_dd = (equity - rolling_max) / rolling_max * 100

        worst_drawdowns.append(pct_dd.min())

    worst_drawdowns = np.array(worst_drawdowns)

    return {
        "mean_worst_drawdown_pct":   worst_drawdowns.mean(),
        "median_worst_drawdown_pct": np.median(worst_drawdowns),
        "p5_worst_drawdown_pct":     np.percentile(worst_drawdowns, 5),   # worst 5% of paths
        "p95_worst_drawdown_pct":    np.percentile(worst_drawdowns, 95),  # mildest 5% of paths
        "min_worst_drawdown_pct":    worst_drawdowns.min(),               # absolute worst path
    }


def plot_monte_carlo_paths(simulations: pd.DataFrame, n_paths_to_plot: int = 100) -> None:
    """
    Plot a sample of simulated equity paths on one chart.

    Plotting all n_simulations at once would create an unreadable mess;
    n_paths_to_plot caps the number drawn (default 100).
    Each path is semi-transparent (alpha=0.2) so overlapping paths are visible.
    """
    plt.figure(figsize=(10, 6))

    # Cap at the number of simulations available
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
    Run a full Monte Carlo simulation and return all summary statistics.

    Convenience wrapper that combines run_monte_carlo(), final_equity_stats(),
    probability_of_ruin(), and drawdown_distribution() into a single call.

    Parameters
    ----------
    ruin_threshold : Equity level considered "ruin". Default 0.0 (account goes
                     negative). For prop firm use, pass e.g. 0.9 to flag any
                     path that loses more than 10% from starting_equity=1.0.
    """

    if simulations is None:
        simulations = run_monte_carlo(
            trades=trades,
            n_simulations=n_simulations,
            n_trades=n_trades,
            starting_equity=starting_equity
        )
    stats        = final_equity_stats(simulations)
    dd_dist      = drawdown_distribution(simulations, starting_equity=starting_equity)
    prob_ruin    = probability_of_ruin(simulations, ruin_threshold=ruin_threshold)

    return {
        "n_simulations":   n_simulations,
        "n_trades":        n_trades if n_trades is not None else len(trades),
        "probability_of_ruin": prob_ruin,
        **stats,
        **dd_dist,
    }
