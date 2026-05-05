# V1 — Backtester
# Converts a signal column into a simulated equity curve.
# Vectorised — uses pandas operations rather than bar-by-bar loops.

import pandas as pd
import numpy as np


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "signal",
    return_col: str = "returns"
) -> pd.DataFrame:
    """
    Simulate a strategy's equity curve from a signal column.
    Adds position, strategy_return, strategy_equity, and buy_and_hold_equity columns.
    """
    df = df.copy()

    # Shift signal by 1 bar — signal fires at close of bar N, we trade at open of N+1
    df["position"] = df[signal_col].shift(1).fillna(0)

    # Strategy return = position * bar return (0 when flat, full return when long)
    df["strategy_return"]     = df["position"] * df[return_col]
    df["buy_and_hold_return"] = df[return_col]

    # Compound the returns to get equity curves starting at 1.0
    df["strategy_equity"]     = (1 + df["strategy_return"]).cumprod()
    df["buy_and_hold_equity"] = (1 + df["buy_and_hold_return"]).cumprod()

    return df
