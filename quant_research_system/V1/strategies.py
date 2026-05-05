# V1 — Strategies
# Signal generation. Each function reads indicator columns and writes a signal column.
# 0 = flat, 1 = long.

import pandas as pd


def moving_average_crossover_strategy(
    df: pd.DataFrame,
    short_ma_col: str,
    long_ma_col: str,
    signal_col: str = "signal"
) -> pd.DataFrame:
    """
    Long when short MA is above long MA, flat otherwise.
    Trend-following — doesn't go short.
    """
    df = df.copy()
    df[signal_col] = 0
    df.loc[df[short_ma_col] > df[long_ma_col], signal_col] = 1
    return df
