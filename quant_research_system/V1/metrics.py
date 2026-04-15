# =============================================================================
# V1 — Performance Metrics
# =============================================================================
# Functions that compute standard performance statistics from an equity curve
# or a returns series. Each function is standalone so they can be called
# independently or combined in a report.
# =============================================================================

import pandas as pd
import numpy as np


def calculate_total_return(equity_curve: pd.Series) -> float:
    """
    Total return over the entire period.

    Assumes the equity curve starts at 1.0 (e.g. from cumprod of returns).
    Result: 0.50 means +50%, -0.20 means -20%.
    """
    return equity_curve.iloc[-1] - 1


def calculate_annualised_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate (CAGR) of the equity curve.

    Formula: ending_value ^ (periods_per_year / total_periods) - 1
    Works correctly only if the curve starts at 1.0.

    periods_per_year = 252 for daily data (trading days in a year).
    """
    total_periods = len(equity_curve)
    ending_value  = equity_curve.iloc[-1]

    if total_periods == 0 or ending_value <= 0:
        return np.nan

    return ending_value ** (periods_per_year / total_periods) - 1


def calculate_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualised Sharpe ratio: risk-adjusted return per unit of volatility.

    Formula: (mean_excess_return / std_excess_return) * sqrt(periods_per_year)

    A Sharpe above 1.0 is generally considered good for a systematic strategy.
    Above 2.0 is excellent. Below 0 means the strategy underperformed cash.

    risk_free_rate : Annual risk-free rate (e.g. 0.05 for 5%).
                     Converted to per-period internally.
    """
    daily_rf      = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    std_return     = excess_returns.std()

    if std_return == 0 or pd.isna(std_return):
        return np.nan

    return (excess_returns.mean() / std_return) * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown: the largest peak-to-trough decline in the equity curve.

    Computed as a fraction (e.g. -0.25 = a 25% drawdown).
    Rolling max tracks the highest point reached so far;
    drawdown is how far below that peak the curve currently sits.
    """
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve / rolling_max) - 1
    return drawdown.min()


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Fraction of active bars (non-zero returns) where the strategy made money.

    Bars with 0 return are excluded because they represent periods where the
    strategy was flat (no position) — they don't tell us about the strategy's
    ability to pick winners.
    """
    active_returns = returns[returns != 0]

    if len(active_returns) == 0:
        return np.nan

    return (active_returns > 0).mean()
