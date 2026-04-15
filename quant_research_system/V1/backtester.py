# =============================================================================
# V1 — Backtester
# =============================================================================
# Converts a signal column into a simulated equity curve.
#
# This is a vectorised backtester — instead of looping through each bar,
# it uses pandas operations to compute all returns at once, which is much
# faster and easier to read.
# =============================================================================

import pandas as pd
import numpy as np


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "signal",
    return_col: str = "returns"
) -> pd.DataFrame:
    """
    Simulate a strategy's equity curve from a signal column.

    Parameters
    ----------
    df         : DataFrame with a signal column and a returns column.
    signal_col : Column with trade signals (0 = flat, 1 = long).
    return_col : Column with bar-by-bar percentage returns.

    Returns
    -------
    The same DataFrame with four new columns added:
      position          — the active position on each bar
      strategy_return   — the return earned by the strategy on each bar
      strategy_equity   — cumulative equity of the strategy (starts at 1.0)
      buy_and_hold_equity — cumulative equity of just holding (benchmark)
    """
    df = df.copy()

    # ── Position (signal lagged by 1 bar) ─────────────────────────────────
    # We shift the signal by one bar to avoid look-ahead bias.
    # The signal is generated at the CLOSE of bar N, so we can only act
    # at the OPEN of bar N+1. Shifting by 1 models this correctly.
    df["position"] = df[signal_col].shift(1)
    df["position"] = df["position"].fillna(0)  # first bar has no prior signal

    # ── Returns ───────────────────────────────────────────────────────────
    # Strategy return = position * bar return
    # If position = 1 (long), we capture the full bar return.
    # If position = 0 (flat), we earn 0 regardless of how the bar moved.
    df["strategy_return"]    = df["position"] * df[return_col]
    df["buy_and_hold_return"] = df[return_col]  # always long, no timing

    # ── Equity curves (compounded) ────────────────────────────────────────
    # cumprod of (1 + return) gives the compounded growth factor.
    # Starting value is 1.0, so 1.5 means +50% from the start.
    df["strategy_equity"]     = (1 + df["strategy_return"]).cumprod()
    df["buy_and_hold_equity"] = (1 + df["buy_and_hold_return"]).cumprod()

    return df
