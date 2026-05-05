# V1 — Technical Indicators
# Each function takes a DataFrame, adds one indicator column, and returns it.
# Copying first means the original DataFrame is never modified.

import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add bar-to-bar percentage return on Close. First row is NaN."""
    df = df.copy()
    df["returns"] = df["Close"].pct_change()
    return df


def add_moving_average(df: pd.DataFrame, window: int, column_name: str) -> pd.DataFrame:
    """Add a simple moving average of Close over the last `window` bars."""
    df = df.copy()
    df[column_name] = df["Close"].rolling(window=window).mean()
    return df


def add_volatility(df: pd.DataFrame, window: int, column_name: str) -> pd.DataFrame:
    """Add rolling standard deviation of returns over the last `window` bars."""
    df = df.copy()
    df[column_name] = df["returns"].rolling(window=window).std()
    return df


def add_momentum(df: pd.DataFrame, window: int, column_name: str) -> pd.DataFrame:
    """Add rate-of-change momentum: (Close_today / Close_N_bars_ago) - 1."""
    df = df.copy()
    df[column_name] = df["Close"] / df["Close"].shift(window) - 1
    return df
