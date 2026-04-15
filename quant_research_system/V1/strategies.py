# =============================================================================
# V1 — Strategies
# =============================================================================
# Signal generation logic. Each strategy function reads indicator columns from
# the DataFrame and writes a signal column (0 = flat, 1 = long).
#
# Keeping strategies in their own module means you can swap the strategy
# without touching the backtester or metrics.
# =============================================================================

import pandas as pd


def moving_average_crossover_strategy(
    df: pd.DataFrame,
    short_ma_col: str,
    long_ma_col: str,
    signal_col: str = "signal"
) -> pd.DataFrame:
    """
    Generate a long signal when the short MA is above the long MA.

    Logic:
      signal = 1  when short_ma > long_ma  (uptrend — be long)
      signal = 0  when short_ma <= long_ma (downtrend or flat — stay out)

    This is a trend-following strategy: it buys when price momentum is up
    and exits (goes flat) when momentum reverses. It does NOT go short.

    Parameters
    ----------
    df           : DataFrame with indicator columns already added.
    short_ma_col : Column name of the faster (shorter window) moving average.
    long_ma_col  : Column name of the slower (longer window) moving average.
    signal_col   : Name for the output signal column (default "signal").
    """
    df = df.copy()

    # Initialise all rows to 0 (no position)
    df[signal_col] = 0

    # Set signal to 1 wherever the short MA is above the long MA
    df.loc[df[short_ma_col] > df[long_ma_col], signal_col] = 1

    return df
