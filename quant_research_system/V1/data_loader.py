# =============================================================================
# V1 — Data Loader
# =============================================================================
# Responsible for fetching raw OHLCV price data from Yahoo Finance.
# All other V1 modules receive data from this one, so keeping it simple and
# consistent here makes the rest of the pipeline easier to reason about.
# =============================================================================

import yfinance as yf
import pandas as pd


def load_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily OHLCV data for a given ticker and date range.

    Parameters
    ----------
    ticker     : e.g. "SPY", "AAPL", "QQQ"
    start_date : "YYYY-MM-DD"
    end_date   : "YYYY-MM-DD"

    Returns
    -------
    DataFrame with columns [Open, High, Low, Close, Volume] and a DatetimeIndex.
    """
    df = yf.download(ticker, start=start_date, end=end_date)

    # yfinance returns a MultiIndex when given a single ticker — drop the
    # redundant ticker level so we get simple column names (Open, Close, etc.)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Strip any accidental whitespace from column names
    df.columns = [col.strip() for col in df.columns]

    return df
