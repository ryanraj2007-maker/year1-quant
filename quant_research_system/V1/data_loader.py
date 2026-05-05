# V1 — Data Loader
# Fetches OHLCV data from Yahoo Finance for a given ticker and date range.

import yfinance as yf
import pandas as pd


def load_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily OHLCV data for a ticker. Returns a DataFrame with a DatetimeIndex."""
    df = yf.download(ticker, start=start_date, end=end_date)

    # yfinance returns a MultiIndex for single tickers — drop the redundant level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.columns = [col.strip() for col in df.columns]
    return df
