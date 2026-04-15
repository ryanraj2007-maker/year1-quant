# =============================================================================
# V1 — Technical Indicators
# =============================================================================
# Functions that add derived columns (indicators) to a price DataFrame.
#
# Design pattern: every function accepts a DataFrame, makes a copy (so the
# original is never mutated), adds one column, and returns the new DataFrame.
# This lets you chain calls cleanly in main.py without side effects.
# =============================================================================

import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'returns' column: the percentage change in Close from bar to bar.

    pct_change() = (today - yesterday) / yesterday
    The first row will be NaN because there's no prior close to compare to.
    """
    df = df.copy()
    df["returns"] = df["Close"].pct_change()
    return df


def add_moving_average(df: pd.DataFrame, window: int, column_name: str) -> pd.DataFrame:
    """
    Add a simple moving average of Close over the last `window` bars.

    rolling(window).mean() produces NaN for the first (window - 1) rows
    because there aren't enough prior bars to fill the window yet.
    """
    df = df.copy()
    df[column_name] = df["Close"].rolling(window=window).mean()
    return df


def add_volatility(df: pd.DataFrame, window: int, column_name: str) -> pd.DataFrame:
    """
    Add a rolling standard deviation of returns over the last `window` bars.

    Standard deviation of returns is the most common measure of short-term
    price volatility — a higher value means more day-to-day price swing.
    Uses pandas default (ddof=1, unbiased / sample std dev).
    """
    df = df.copy()
    df[column_name] = df["returns"].rolling(window=window).std()
    return df


def add_momentum(df: pd.DataFrame, window: int, column_name: str) -> pd.DataFrame:
    """
    Add a rate-of-change momentum indicator over the last `window` bars.

    Formula: (Close_today / Close_N_bars_ago) - 1
    Positive value = price is higher than it was N bars ago (upward momentum).
    Negative value = price is lower than it was N bars ago (downward momentum).
    """
    df = df.copy()
    df[column_name] = df["Close"] / df["Close"].shift(window) - 1
    return df
