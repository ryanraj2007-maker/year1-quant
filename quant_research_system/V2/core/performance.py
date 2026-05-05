# V2 Core — Performance Analytics
# All metrics use R-multiples rather than raw P&L.
# This makes strategies comparable regardless of position size or account size.

import pandas as pd
import numpy as np


def total_r(trades: pd.DataFrame) -> float:
    """Total R earned across all trades."""
    return trades["r_multiple"].sum()


def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of trades that were profitable."""
    if len(trades) == 0:
        return np.nan
    return trades["win"].mean()


def average_r(trades: pd.DataFrame) -> float:
    """Average R per trade — same as expectancy."""
    if len(trades) == 0:
        return np.nan
    return trades["r_multiple"].mean()


def average_win_r(trades: pd.DataFrame) -> float:
    """Average R on winning trades only."""
    wins = trades[trades["win"] == 1]
    if len(wins) == 0:
        return np.nan
    return wins["r_multiple"].mean()


def average_loss_r(trades: pd.DataFrame) -> float:
    """Average R on losing trades only. Should be close to -1.0 for clean stop discipline."""
    losses = trades[trades["win"] == 0]
    if len(losses) == 0:
        return np.nan
    return losses["r_multiple"].mean()


def expectancy_r(trades: pd.DataFrame) -> float:
    """Expected R per trade — the strategy's mathematical edge."""
    if len(trades) == 0:
        return np.nan
    return trades["r_multiple"].mean()


def profit_factor(trades: pd.DataFrame) -> float:
    """Gross profit / gross loss. Above 1.5 is solid."""
    gross_profit = trades.loc[trades["r_multiple"] > 0, "r_multiple"].sum()
    gross_loss   = -trades.loc[trades["r_multiple"] < 0, "r_multiple"].sum()

    if gross_loss == 0:
        return np.nan

    return gross_profit / gross_loss


def equity_curve_from_r(trades: pd.DataFrame, starting_equity: float = 1.0) -> pd.DataFrame:
    """Build a cumulative R equity curve from the trade list."""
    df = trades.copy()
    df["equity"]      = starting_equity + df["r_multiple"].cumsum()
    df["rolling_max"] = df["equity"].cummax()
    df["drawdown"]    = df["equity"] - df["rolling_max"]
    return df


def max_drawdown(trades: pd.DataFrame, starting_equity: float = 1.0) -> float:
    """Worst peak-to-trough decline in R units."""
    if len(trades) == 0:
        return np.nan
    eq = equity_curve_from_r(trades, starting_equity=starting_equity)
    return eq["drawdown"].min()


def max_drawdown_pct(trades: pd.DataFrame, starting_equity: float = 1.0) -> float:
    """Worst drawdown as a percentage of peak equity."""
    if len(trades) == 0:
        return np.nan
    eq = equity_curve_from_r(trades, starting_equity=starting_equity)
    pct_dd = (eq["equity"] - eq["rolling_max"]) / eq["rolling_max"] * 100
    return pct_dd.min()


def sharpe_ratio(trades: pd.DataFrame, trades_per_year: int = 252) -> float:
    """Annualised Sharpe from trade-level R-multiples."""
    if len(trades) < 2:
        return np.nan
    r = trades["r_multiple"]
    std = r.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return (r.mean() / std) * np.sqrt(trades_per_year)


def longest_losing_streak(trades: pd.DataFrame) -> int:
    """Maximum consecutive losing trades."""
    if len(trades) == 0:
        return 0
    max_streak = 0
    current_streak = 0
    for win in trades["win"]:
        if win == 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


def recovery_factor(trades: pd.DataFrame, starting_equity: float = 1.0) -> float:
    """Total R / abs(max drawdown). How efficiently the strategy recovers from its worst hole."""
    if len(trades) == 0:
        return np.nan
    mdd = max_drawdown(trades, starting_equity=starting_equity)
    if mdd == 0:
        return np.nan
    return total_r(trades) / abs(mdd)


def summary_stats(trades: pd.DataFrame) -> dict:
    """Run all metrics and return as a dict."""
    return {
        "num_trades":            len(trades),
        "total_r":               total_r(trades),
        "win_rate":              win_rate(trades),
        "average_r":             average_r(trades),
        "average_win_r":         average_win_r(trades),
        "average_loss_r":        average_loss_r(trades),
        "expectancy_r":          expectancy_r(trades),
        "profit_factor":         profit_factor(trades),
        "max_drawdown_r":        max_drawdown(trades),
        "max_drawdown_pct":      max_drawdown_pct(trades),
        "sharpe_ratio":          sharpe_ratio(trades),
        "longest_losing_streak": longest_losing_streak(trades),
        "recovery_factor":       recovery_factor(trades),
    }
