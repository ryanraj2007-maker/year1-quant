# V1 — Performance Metrics
# Standalone functions for measuring strategy performance from an equity curve or returns series.

import pandas as pd
import numpy as np


def calculate_total_return(equity_curve: pd.Series) -> float:
    """Total return over the period. Assumes curve starts at 1.0."""
    return equity_curve.iloc[-1] - 1


def calculate_annualised_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR of the equity curve. 252 periods/year for daily data."""
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
    """Annualised Sharpe ratio. risk_free_rate is annual (e.g. 0.05 = 5%)."""
    daily_rf       = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    std_return     = excess_returns.std()
    if std_return == 0 or pd.isna(std_return):
        return np.nan
    return (excess_returns.mean() / std_return) * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline as a fraction (e.g. -0.25 = 25% drawdown)."""
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve / rolling_max) - 1
    return drawdown.min()


def calculate_win_rate(returns: pd.Series) -> float:
    """Fraction of active (non-zero) bars where the strategy made money."""
    active_returns = returns[returns != 0]
    if len(active_returns) == 0:
        return np.nan
    return (active_returns > 0).mean()
