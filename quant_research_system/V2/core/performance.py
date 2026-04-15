# =============================================================================
# V2 Core — Performance Analytics
# =============================================================================
# All performance metrics are computed at the trade level using R-multiples.
# This is standard practice in prop trading — you evaluate strategies by how
# many R they make per trade, not by raw dollar P&L (which depends on position
# size and account size).
#
# The summary_stats() function is the main entry point used by strategies and
# the dashboard. Individual functions can be called directly for custom reports.
# =============================================================================

import pandas as pd
import numpy as np


def total_r(trades: pd.DataFrame) -> float:
    """
    Sum of all R-multiples across every trade.

    +10R over 20 trades means the strategy made 10× its average risk.
    This is the raw "edge" of the strategy before position sizing.
    """
    return trades["r_multiple"].sum()


def win_rate(trades: pd.DataFrame) -> float:
    """
    Fraction of trades that were profitable.

    A high win rate alone doesn't mean a strategy is good — a 70% win rate
    with small winners and large losers can still be negative expectancy.
    Win rate must be read alongside average_win_r and average_loss_r.
    """
    if len(trades) == 0:
        return np.nan
    return trades["win"].mean()


def average_r(trades: pd.DataFrame) -> float:
    """
    Average R-multiple per trade (same as expectancy_r).

    This is the expected return per trade in units of risk.
    Positive average R = positive expectancy = strategy has an edge.
    """
    if len(trades) == 0:
        return np.nan
    return trades["r_multiple"].mean()


def average_win_r(trades: pd.DataFrame) -> float:
    """
    Average R-multiple of winning trades only.

    Tells you how big wins are relative to risk. Comparing this to
    average_loss_r shows the asymmetry of the strategy's outcomes.
    """
    wins = trades[trades["win"] == 1]
    if len(wins) == 0:
        return np.nan
    return wins["r_multiple"].mean()


def average_loss_r(trades: pd.DataFrame) -> float:
    """
    Average R-multiple of losing trades only.

    Should be close to -1.0 for a well-disciplined strategy (always taking
    stops). A value much worse than -1.0 suggests stop-outs with slippage
    or positions held past the stop.
    """
    losses = trades[trades["win"] == 0]
    if len(losses) == 0:
        return np.nan
    return losses["r_multiple"].mean()


def expectancy_r(trades: pd.DataFrame) -> float:
    """
    Expected R per trade: the mathematical edge of the strategy.

    Equivalent to average_r — kept as a separate function for clarity
    since "expectancy" is the standard term used in prop trading.

    Formula: (win_rate × avg_win) + ((1 - win_rate) × avg_loss)
    Same result as mean(r_multiples).
    """
    if len(trades) == 0:
        return np.nan
    return trades["r_multiple"].mean()


def profit_factor(trades: pd.DataFrame) -> float:
    """
    Ratio of total gross profit to total gross loss (both in R).

    profit_factor > 1.0 = strategy made more than it lost overall.
    profit_factor = 2.0 means for every $1 lost, $2 was made.
    A robust strategy typically has a profit factor above 1.5.
    """
    gross_profit = trades.loc[trades["r_multiple"] > 0, "r_multiple"].sum()
    gross_loss   = -trades.loc[trades["r_multiple"] < 0, "r_multiple"].sum()

    if gross_loss == 0:
        # No losing trades — technically infinite profit factor
        return np.nan

    return gross_profit / gross_loss


def equity_curve_from_r(trades: pd.DataFrame, starting_equity: float = 1.0) -> pd.DataFrame:
    """
    Build an equity curve by additively accumulating R-multiples.

    This models fixed-fraction sizing where each trade risks exactly 1R
    unit of capital. The curve starts at starting_equity and rises/falls
    by each trade's R-multiple.

    Also adds rolling_max and drawdown columns for drawdown analysis.
    """
    df = trades.copy()

    # Cumulative sum of R-multiples added to starting equity
    df["equity"]      = starting_equity + df["r_multiple"].cumsum()

    # Rolling maximum — the highest equity reached so far at each point
    df["rolling_max"] = df["equity"].cummax()

    # Drawdown = how far below the peak we currently are (in R units)
    # Negative values indicate underwater periods
    df["drawdown"]    = df["equity"] - df["rolling_max"]

    return df


def max_drawdown(trades: pd.DataFrame, starting_equity: float = 1.0) -> float:
    """
    Worst peak-to-trough decline in the equity curve (in R units).

    The most negative value of the drawdown column.
    e.g. -5.0 means at the worst point the strategy was 5R underwater
    from its previous high.
    """
    if len(trades) == 0:
        return np.nan

    eq = equity_curve_from_r(trades, starting_equity=starting_equity)
    return eq["drawdown"].min()


def max_drawdown_pct(trades: pd.DataFrame, starting_equity: float = 1.0) -> float:
    """
    Worst peak-to-trough decline expressed as a percentage of the peak equity.

    Converts the R-unit drawdown into a percentage so it can be compared
    across strategies with different risk-per-trade assumptions.

    e.g. if peak equity was 1.5R and trough was 1.0R → drawdown = -33.3%
    """
    if len(trades) == 0:
        return np.nan

    eq = equity_curve_from_r(trades, starting_equity=starting_equity)

    # Percentage drawdown = (equity - rolling_max) / rolling_max × 100
    pct_dd = (eq["equity"] - eq["rolling_max"]) / eq["rolling_max"] * 100
    return pct_dd.min()


def sharpe_ratio(trades: pd.DataFrame, trades_per_year: int = 252) -> float:
    """
    Annualised Sharpe ratio computed from trade-level R-multiples.

    Uses R-multiples as the return series (risk-free rate = 0, since R already
    represents excess return over the stop-loss floor). Scales by assuming
    trades_per_year trading opportunities per year.

    A Sharpe above 1.0 is decent; above 2.0 is strong for a discretionary
    strategy. Be sceptical of values above 3.0 without a large trade sample.
    """
    if len(trades) < 2:
        return np.nan

    r = trades["r_multiple"]
    std = r.std()

    if std == 0 or pd.isna(std):
        return np.nan

    # Mean R per trade / std R per trade × sqrt(annual trade frequency)
    return (r.mean() / std) * np.sqrt(trades_per_year)


def longest_losing_streak(trades: pd.DataFrame) -> int:
    """
    Maximum number of consecutive losing trades in the trade log.

    Important for psychology and drawdown analysis — a strategy might have
    positive expectancy but a 10-trade losing streak that breaks discipline.
    Prop firm challenges typically have daily loss limits that a streak can hit.
    """
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
    """
    Total R earned divided by the absolute value of the max drawdown (in R).

    Measures how efficiently the strategy recovers from its worst drawdown.
    recovery_factor = 3.0 means the strategy made 3× its worst drawdown.

    Higher is better. A value below 1.0 means the strategy never fully
    recovered from its worst loss period within the backtest window.
    """
    if len(trades) == 0:
        return np.nan

    mdd = max_drawdown(trades, starting_equity=starting_equity)

    # If there was no drawdown at all the recovery factor is undefined (infinite)
    if mdd == 0:
        return np.nan

    return total_r(trades) / abs(mdd)


def summary_stats(trades: pd.DataFrame) -> dict:
    """
    Compute all key performance metrics and return as a dictionary.

    This is the standard output used by strategy runners, the dashboard,
    and the prop firm simulator. All values are in R-multiple units unless
    noted (max_drawdown_pct is in %, sharpe_ratio is dimensionless).
    """
    return {
        "num_trades":          len(trades),
        "total_r":             total_r(trades),
        "win_rate":            win_rate(trades),
        "average_r":           average_r(trades),
        "average_win_r":       average_win_r(trades),
        "average_loss_r":      average_loss_r(trades),
        "expectancy_r":        expectancy_r(trades),
        "profit_factor":       profit_factor(trades),
        "max_drawdown_r":      max_drawdown(trades),
        "max_drawdown_pct":    max_drawdown_pct(trades),
        "sharpe_ratio":        sharpe_ratio(trades),
        "longest_losing_streak": longest_losing_streak(trades),
        "recovery_factor":     recovery_factor(trades),
    }
