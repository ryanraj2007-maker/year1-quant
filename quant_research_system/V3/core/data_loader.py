# V2 Core — Data Loader
# DataProvider ABC lets strategies swap data sources without changing any other code.
# YFinanceProvider = free equities data. DatabentoProvider = paid futures data.

import pandas as pd
from abc import ABC, abstractmethod


class DataProvider(ABC):
    @abstractmethod
    def fetch(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol over a date range."""
        pass


class YFinanceProvider(DataProvider):
    def fetch(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        """Download data from yFinance, chunked to respect API history limits."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("pip install yfinance to use YFinanceProvider")

        _chunk = {"1m": 7, "5m": 59, "15m": 59, "1h": 729}.get(interval, 59)
        _limit = {"1m": 30, "5m": 60, "15m": 60, "1h": 730}.get(interval, 60)

        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)

        earliest = pd.Timestamp.now().normalize() - pd.Timedelta(days=_limit - 1)
        if start_dt < earliest:
            print(f"[yfinance] clipping start date to {earliest.date()} due to API limitations"
                  f"({interval} limit = {_limit} days)")
            start_dt = earliest

        chunks, s = [], start_dt
        while s < end_dt:
            e = min(s + pd.Timedelta(days=_chunk), end_dt)
            chunk = yf.download(symbol, start=s, end=e, interval=interval,
                                auto_adjust=True, progress=False, multi_level_index=False)
            if not chunk.empty:
                chunks.append(chunk)
            s = e

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="first")]
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)
        return df


class DatabentoProvider(DataProvider):
    def __init__(self, api_key: str):
        """Store the Databento API key for use in fetch()."""
        self._api_key = api_key

    def fetch(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        """Fetch futures OHLCV data from Databento's historical API."""
        try:
            import databento as db
        except ImportError:
            raise ImportError("pip install databento to use DatabentoProvider")

        _schema_map = {
            "1m":  "ohlcv-1m",
            "5m":  "ohlcv-5m",
            "15m": "ohlcv-15m",
            "1h":  "ohlcv-1h",
            "1d":  "ohlcv-1d",
        }
        schema    = _schema_map.get(interval, "ohlcv-5m")
        db_symbol = f"{symbol}.c.0"  # continuous futures format

        client = db.Historical(self._api_key)
        data   = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[db_symbol],
            schema=schema,
            start=start,
            end=end,
            stype_in="continuous",
        )
        df = data.to_df()

        if df.empty:
            return pd.DataFrame()

        df.index   = pd.to_datetime(df.index, utc=True)
        df.index   = df.index.tz_convert("America/New_York").tz_localize(None)
        df.columns = [c.lower() for c in df.columns]

        # Databento prices are in nanodollars — convert if values look too large
        for col in ["open", "high", "low", "close"]:
            if col in df.columns and df[col].max() > 100_000:
                df[col] = df[col] / 1e9

        cols = ["open", "high", "low", "close", "volume"] if "volume" in df.columns else ["open", "high", "low", "close"]
        return df[cols]
