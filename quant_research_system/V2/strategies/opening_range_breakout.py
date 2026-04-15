"""
Opening Range Breakout (ORB) Strategy

Logic:
  1. Define the opening range using the first N minutes of each trading session.
  2. Long signal:  a candle closes ABOVE the opening range high  → enter long,
                   stop = opening range low, target = entry + RR × risk
  3. Short signal: a candle closes BELOW the opening range low   → enter short,
                   stop = opening range high, target = entry - RR × risk
  4. At most one trade per day (first breakout wins).
  5. Trade is exited when price hits target, stop, or the session ends.

Outputs a pd.DataFrame of trades compatible with the V2 core framework
(performance, monte_carlo, plots).
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Make core importable when running this file directly
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.trade_log import Trade, create_trade, trades_to_dataframe


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

# Yahoo Finance hard limits (calendar days of history available).
# These aren't arbitrary — they're documented API constraints; going beyond them
# silently returns empty data, so we clip the request date before calling the API.
_YF_LIMITS = {
    "1m":  30,    # last 30 days only
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,   # ~2 years
    "60m": 730,
    "1d":  99999,
}
# Max days per single yfinance request — yfinance rejects requests that span more
# than this many days for sub-daily intervals, so we break long ranges into chunks.
_YF_CHUNK = {
    "1m": 7, "2m": 59, "5m": 59, "15m": 59, "30m": 59,
    "1h": 729, "60m": 729, "1d": 3650,
}


def fetch_data_yfinance(
    ticker: str,
    start: str,
    end: str,
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Fetch intraday OHLC data via yfinance (free, no API key needed).

    Yahoo Finance history limits:
      1m / 5m / 15m / 30m  →  last 60 days only
      1h                   →  last ~2 years
      1d                   →  unlimited

    For 5-year backtests use fetch_data_alpaca() instead.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    limit_days = _YF_LIMITS.get(interval, 60)
    chunk_days = _YF_CHUNK.get(interval, 59)

    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)

    # Yahoo's history window is a rolling lookback from TODAY, not from the
    # data's earliest date — so we compute "earliest available" dynamically.
    earliest = pd.Timestamp.now().normalize() - pd.Timedelta(days=limit_days - 1)

    # Silently clipping (rather than raising) is friendlier: the user still gets
    # whatever data is available, and a clear message explains the truncation.
    if start_dt < earliest:
        print(f"  [yfinance] {interval} data only available from "
              f"{earliest.date()} — clipping start date.")
        start_dt = earliest

    # After clipping, if start is still >= end the entire range is unavailable.
    if start_dt >= end_dt:
        print(f"  [yfinance] Requested range is outside the available history "
              f"for {interval} data.")
        return pd.DataFrame()

    # Chunked download: yfinance enforces per-request day limits for sub-daily
    # intervals.  We tile the range into chunk_days-wide windows and stitch them.
    chunks = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), end_dt)
        df_chunk = yf.download(
            ticker,
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,       # adjust for splits/dividends — always on for backtesting
            progress=False,         # suppress tqdm bar in scripts/notebooks
            multi_level_index=False,  # flatten the (ticker, field) MultiIndex yfinance adds
        )
        if not df_chunk.empty:
            chunks.append(df_chunk)
        chunk_start = chunk_end

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks)
    # Chunk boundaries can overlap by one day — deduplicate keeping the first
    # occurrence so we don't double-count bars.
    df = df[~df.index.duplicated(keep="first")].sort_index()
    # Normalise column names to lowercase so the rest of the code works regardless
    # of which yfinance version returned "Close" vs "close".
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df


def fetch_data_alpaca(
    ticker: str,
    start: str,
    end: str,
    interval: str = "5Min",
    api_key: str = "",
    api_secret: str = "",
) -> pd.DataFrame:
    """
    Fetch intraday OHLC data via Alpaca Markets (free account, 5+ years history).

    Sign up free at https://alpaca.markets — get API keys from the dashboard.

    Parameters
    ----------
    ticker     : e.g. "SPY", "AAPL"
    start      : "YYYY-MM-DD"
    end        : "YYYY-MM-DD"
    interval   : Alpaca timeframe string — "1Min", "5Min", "15Min", "1Hour", "1Day"
    api_key    : Alpaca API key ID
    api_secret : Alpaca secret key

    Returns
    -------
    DataFrame with columns [open, high, low, close, volume] and tz-aware index.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        raise ImportError(
            "alpaca-py is required for 5-year backtests:\n"
            "  pip install alpaca-py\n"
            "Then sign up free at https://alpaca.markets for API keys."
        )

    # Map human-readable interval strings to Alpaca SDK TimeFrame objects.
    # Alpaca's SDK doesn't accept plain strings — it requires typed objects.
    _tf_map = {
        "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1,  TimeFrameUnit.Day),
    }
    if interval not in _tf_map:
        raise ValueError(f"interval must be one of {list(_tf_map)}")

    client = StockHistoricalDataClient(api_key, api_secret)
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=_tf_map[interval],
        # Alpaca requires tz-aware timestamps; Eastern is the exchange's native zone.
        start=pd.Timestamp(start, tz="America/New_York"),
        end=pd.Timestamp(end,   tz="America/New_York"),
        adjustment="all",   # apply splits + dividends — essential for multi-year backtests
    )
    bars = client.get_stock_bars(request).df

    if bars.empty:
        return pd.DataFrame()

    # alpaca-py returns a MultiIndex (symbol, timestamp) when multiple symbols
    # are requested — flatten to a single-level index for consistent downstream handling.
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level="symbol")

    bars.index.name = "Datetime"
    bars.columns = [c.lower() for c in bars.columns]
    return bars[["open", "high", "low", "close", "volume"]]


def fetch_data_tradingview_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV exported directly from TradingView.

    How to export from TradingView:
      1. Open any chart at the timeframe you want (e.g. 5m SPY).
      2. Click the camera/export icon at the top-right → "Export chart data".
      3. Save the .csv file anywhere on your machine.
      4. Pass the file path here.

    TradingView CSVs have columns: time, open, high, low, close, volume
    The 'time' column is a Unix timestamp (seconds) or a date string —
    both formats are handled automatically.

    Returns
    -------
    DataFrame with columns [open, high, low, close, volume] and a tz-aware
    DatetimeIndex in America/New_York (US Eastern).
    """
    df = pd.read_csv(filepath)
    # Strip whitespace from column names — TradingView CSV headers sometimes
    # have a trailing space, which would silently break .columns lookups below.
    df.columns = [c.lower().strip() for c in df.columns]

    if "time" not in df.columns:
        raise ValueError(
            "CSV must have a 'time' column. "
            "Make sure you exported from TradingView (Export chart data)."
        )

    # TradingView exports Unix timestamps (seconds) or ISO strings depending on
    # the chart version; handle both so the loader works regardless of TV version.
    if pd.api.types.is_numeric_dtype(df["time"]):
        # Unix epoch in seconds → UTC datetime
        df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    else:
        df.index = pd.to_datetime(df["time"], utc=True)

    # Convert UTC → Eastern so between_time("09:30") filters the correct session bars.
    df.index = df.index.tz_convert("America/New_York")
    df.index.name = "Datetime"
    df = df.drop(columns=["time"])

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Include volume if present but don't require it — ORB only needs OHLC.
    return df[["open", "high", "low", "close", "volume"]
              if "volume" in df.columns else ["open", "high", "low", "close"]]


def fetch_data_tvdatafeed(
    ticker: str,
    exchange: str,
    start: str,
    end: str,
    interval: str = "5m",
    username: str = "",
    password: str = "",
) -> pd.DataFrame:
    """
    Fetch data via tvdatafeed — an unofficial TradingView scraper.
    Gives several years of intraday data with no API key required.

    Install:  pip install tvdatafeed

    Parameters
    ----------
    ticker   : TradingView symbol, e.g. "SPY"
    exchange : TradingView exchange, e.g. "AMEX", "NASDAQ", "NYSE", "BINANCE"
    start    : "YYYY-MM-DD" — used to filter after fetching
    end      : "YYYY-MM-DD" — used to filter after fetching
    interval : "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1W"
    username : TradingView username (optional — anonymous login works for most data)
    password : TradingView password (optional)

    Notes
    -----
    - tvdatafeed fetches a fixed number of recent bars (~5000 max per call).
      For 5 years of 5m data (~98k bars) you will need multiple calls or use
      the CSV export method instead.
    - This library is unofficial and may break if TradingView changes its API.
    """
    try:
        from tvdatafeed import TvDatafeed, Interval
    except ImportError:
        raise ImportError(
            "tvdatafeed is required:\n"
            "  pip install tvdatafeed\n"
            "No API key needed — anonymous login works for most symbols."
        )

    # tvdatafeed uses an enum for intervals, not plain strings — this dict is the
    # translation layer so callers can use the same "5m" convention as yfinance.
    _interval_map = {
        "1m":  Interval.in_1_minute,
        "3m":  Interval.in_3_minute,
        "5m":  Interval.in_5_minute,
        "15m": Interval.in_15_minute,
        "30m": Interval.in_30_minute,
        "45m": Interval.in_45_minute,
        "1h":  Interval.in_1_hour,
        "2h":  Interval.in_2_hour,
        "3h":  Interval.in_3_hour,
        "4h":  Interval.in_4_hour,
        "1d":  Interval.in_daily,
        "1W":  Interval.in_weekly,
        "1M":  Interval.in_monthly,
    }
    if interval not in _interval_map:
        raise ValueError(f"interval must be one of {list(_interval_map)}")

    # Pass None (not an empty string) when credentials are omitted — the library
    # interprets empty string as a failed login attempt, causing an auth error.
    tv = TvDatafeed(username or None, password or None)

    # Estimate bars needed to cover the requested range.
    # tvdatafeed always returns the N most-recent bars, so we over-request slightly
    # (×1.1 buffer) and then trim to the exact date range after fetching.
    start_dt  = pd.Timestamp(start)
    end_dt    = pd.Timestamp(end)
    days      = (end_dt - start_dt).days
    mins_per_bar = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
                    "1h": 60, "4h": 240, "1d": 1440}.get(interval, 5)
    # US market ~390 min/day, 252 trading days/year; days/7*5 converts calendar
    # days → trading days; +500 guards against holidays and thin weeks.
    est_bars  = int(days / 7 * 5 * 390 / mins_per_bar * 1.1) + 500
    n_bars    = min(est_bars, 20000)   # tvdatafeed cap — requesting more is silently ignored

    df = tv.get_hist(
        symbol=ticker,
        exchange=exchange,
        interval=_interval_map[interval],
        n_bars=n_bars,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    # tvdatafeed may return tz-naive or tz-aware index depending on the version;
    # normalise to Eastern either way so session filtering works correctly.
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]
            if "volume" in df.columns else ["open", "high", "low", "close"]]

    # Trim to the exact requested window — tvdatafeed always returns recent bars
    # counting backward from today, so the raw result may include dates outside
    # the backtest window.
    start_ts = pd.Timestamp(start, tz="America/New_York")
    end_ts   = pd.Timestamp(end,   tz="America/New_York")
    df = df[(df.index >= start_ts) & (df.index < end_ts)]
    return df


def fetch_data(
    ticker: str,
    start: str = "",
    end: str = "",
    interval: str = "5m",
    source: str = "yfinance",
    # yfinance — no extra args needed
    # alpaca
    alpaca_key: str = "",
    alpaca_secret: str = "",
    # tradingview csv
    tv_csv_path: str = "",
    # tvdatafeed
    tv_exchange: str = "AMEX",
    tv_username: str = "",
    tv_password: str = "",
) -> pd.DataFrame:
    """
    Unified data fetcher.  Pick source based on how much history you need:

      "yfinance"   — free, no setup. Limit: ~60 days (5m), ~2 years (1h).
      "alpaca"     — free account, 5+ years minute data. Needs pip install alpaca-py.
      "tv_csv"     — export CSV from TradingView chart. Any history, any timeframe.
      "tvdatafeed" — unofficial TV scraper, no key needed. pip install tvdatafeed.
                     Max ~5000 bars per call; for 5yr/5m use tv_csv instead.
    """
    if source == "alpaca":
        # Alpaca uses capitalised interval names ("5Min") while the rest of the
        # codebase uses lowercase ("5m") — translate before passing through.
        _compat = {"1m": "1Min", "5m": "5Min", "15m": "15Min",
                   "30m": "30Min", "1h": "1Hour", "1d": "1Day"}
        return fetch_data_alpaca(ticker, start, end,
                                 _compat.get(interval, interval),
                                 alpaca_key, alpaca_secret)
    elif source == "tv_csv":
        if not tv_csv_path:
            raise ValueError("tv_csv_path must be set when source='tv_csv'")
        return fetch_data_tradingview_csv(tv_csv_path)
    elif source == "tvdatafeed":
        return fetch_data_tvdatafeed(ticker, tv_exchange, start, end, interval,
                                     tv_username, tv_password)
    else:
        # Default: yfinance — zero config, best for quick/recent backtests.
        return fetch_data_yfinance(ticker, start, end, interval)


# ---------------------------------------------------------------------------
# Core strategy
# ---------------------------------------------------------------------------

def run_orb(
    df: pd.DataFrame,
    or_minutes: int = 15,
    session_start: str = "09:30",
    session_end: str = "16:00",
    rr_ratio: float = 2.0,
    direction: str = "both",       # "long", "short", or "both"
    slippage_per_share: float = 0.02,  # $0.02 entry slippage (~1 tick on SPY 5m bars)
    commission_per_share: float = 0.0, # $0.00 round-trip (Alpaca stocks = free)
) -> pd.DataFrame:
    """
    Run the Opening Range Breakout strategy on intraday OHLC data.

    Parameters
    ----------
    df             : DataFrame with columns [open, high, low, close] and a
                     DatetimeIndex (tz-aware or tz-naive both work).
    or_minutes     : Length of the opening range window in minutes.
    session_start  : Start of the trading session (HH:MM, local exchange time).
    session_end    : End of the trading session (HH:MM, local exchange time).
    rr_ratio       : Reward-to-risk ratio for target calculation.
    direction      : Which breakout direction(s) to trade.
    slippage_per_share   : One-way entry slippage in dollars per share.
                           $0.02 ≈ 1 tick on SPY 5m bars (half the $0.01 bid-ask spread
                           + a little market impact). Applied to entry fill only.
    commission_per_share : Round-trip commission in dollars per share.
                           Alpaca stocks = $0.00 (truly free).

    Returns
    -------
    pd.DataFrame of trades compatible with the V2 core framework.
    """
    # Work on a copy so the caller's DataFrame is never mutated — important when
    # the same df is reused across parameter sweeps (e.g., OR length optimisation).
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    # Ensure index is in Eastern time, then strip tz for simple date arithmetic.
    # between_time() works correctly on tz-naive Eastern timestamps; tz-aware
    # would still work but adds unnecessary complexity in groupby + Timestamp math.
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    # If already tz-naive, assume it was pre-converted to Eastern (e.g. TV CSV)

    trades: list[Trade] = []

    # Process one calendar day at a time — ORB is inherently a daily reset strategy;
    # the opening range and trade limit reset every session.
    for date, day_df in df.groupby(df.index.date):
        # Restrict to regular trading hours; pre/post-market bars would distort
        # the opening range and produce unrealistic fills.
        day_df = day_df.between_time(session_start, session_end)
        if day_df.empty:
            continue

        session_open = pd.Timestamp(str(date) + " " + session_start)
        # or_end is the first timestamp AFTER the opening range period — we use
        # strict less-than when filtering, so this is the exclusive boundary.
        or_end       = session_open + pd.Timedelta(minutes=or_minutes)

        # Opening range candles (strictly within OR window)
        or_df = day_df[day_df.index < or_end]
        if len(or_df) < 1:
            # Safeguard: skip days where no bars fell inside the OR window
            # (e.g. a half-day that opened late, or a data gap at the open).
            continue

        # The opening range is simply the widest high/low over the OR period.
        # Using max(high) and min(low) — not just open/close — captures the true
        # price extremes, making the breakout levels more meaningful.
        or_high = or_df["high"].max()
        or_low  = or_df["low"].min()

        # A flat opening range (or_high == or_low) means no meaningful range exists
        # — this can happen on extremely thin trading days or data errors.
        # Entering a breakout trade with zero risk would cause a division-by-zero later.
        if or_high == or_low:
            continue  # Flat open – no range to trade

        # Candles available for entries (after OR closes)
        post_or = day_df[day_df.index >= or_end]
        if post_or.empty:
            continue

        # One trade per day enforced by this flag — first valid breakout wins.
        # This avoids "revenge trading" the second breakout on the same day,
        # which tends to have worse edge (market has already shown its hand).
        trade_taken = False

        for ts, row in post_or.iterrows():
            if trade_taken:
                break

            # ── Long breakout ──────────────────────────────────────────────
            # Signal: a candle CLOSES above the OR high, confirming bullish momentum.
            # We use close (not high) to avoid entering on intrabar wicks that fail
            # to sustain — a genuine breakout needs a full bar to commit above the level.
            if direction in ("long", "both") and row["close"] > or_high:
                # Realistic entry: we can only act AFTER the bar closes,
                # so we enter at the OPEN of the NEXT bar (+ slippage).
                # This avoids look-ahead bias — we cannot trade on the close of the
                # signal bar itself without knowing it's the close in real time.
                future_bars = post_or[post_or.index > ts]
                if future_bars.empty:
                    # Signal fired on the very last bar of the session — no room to enter.
                    continue
                next_bar   = future_bars.iloc[0]
                entry      = next_bar["open"] + slippage_per_share  # market order pays the ask
                entry_time = future_bars.index[0]

                # Stop at OR low: the opening range low is the most logical invalidation
                # point — if price falls back to the low of the range the bullish thesis
                # (breakout above OR high) is clearly wrong.
                stop   = or_low
                # Risk = distance from entry to stop, in price points.
                # This is the "1R" unit used to size the position and set the target.
                risk   = entry - stop
                if risk <= 0:
                    # Entry filled BELOW the stop after slippage — skip this trade
                    # to avoid a negative-risk anomaly that would corrupt R-multiples.
                    continue
                # Target: entry + RR × risk.  At RR=2, we make 2R for every 1R risked.
                # Using the same risk unit on both sides keeps the R-multiple maths clean.
                target = entry + rr_ratio * risk

                # Walk forward bar by bar to find the first stop/target hit.
                # We pass future_bars.iloc[1:] because iloc[0] is the entry bar itself —
                # we can't exit on the same bar we enter (fill is at the open).
                exit_price, exit_time = _simulate_exit(
                    future_bars.iloc[1:], "long", stop, target, entry,
                    slippage_per_share=slippage_per_share,
                )

                trade = create_trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    direction="long",
                    entry_price=entry,
                    exit_price=exit_price,
                    stop_price=stop,
                    target_price=target,
                )
                # Express commission as a fraction of 1R so it subtracts cleanly
                # from the R-multiple (e.g. $0.04 commission on a $0.40 stop = 0.1R).
                if risk > 0 and commission_per_share > 0:
                    trade = _deduct_commission(trade, commission_per_share / risk)
                trades.append(trade)
                trade_taken = True

            # ── Short breakout ─────────────────────────────────────────────
            # Mirror image of the long: close below OR low signals bearish momentum.
            elif direction in ("short", "both") and row["close"] < or_low:
                future_bars = post_or[post_or.index > ts]
                if future_bars.empty:
                    continue
                next_bar   = future_bars.iloc[0]
                # Short entry: we sell at the open of the next bar MINUS slippage
                # (we get a slightly worse fill than the posted bid — realistic for shorts).
                entry      = next_bar["open"] - slippage_per_share
                entry_time = future_bars.index[0]

                # Stop above OR high: if price rallies back to the top of the range,
                # the bearish breakdown thesis has failed.
                stop   = or_high
                # For a short: risk = stop - entry (how far price must move against us).
                risk   = stop - entry
                if risk <= 0:
                    continue
                # Short target: entry - RR × risk (price must fall this many points).
                target = entry - rr_ratio * risk

                exit_price, exit_time = _simulate_exit(
                    future_bars.iloc[1:], "short", stop, target, entry,
                    slippage_per_share=slippage_per_share,
                )

                trade = create_trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    direction="short",
                    entry_price=entry,
                    exit_price=exit_price,
                    stop_price=stop,
                    target_price=target,
                )
                if risk > 0 and commission_per_share > 0:
                    trade = _deduct_commission(trade, commission_per_share / risk)
                trades.append(trade)
                trade_taken = True

    if not trades:
        return pd.DataFrame()

    return trades_to_dataframe(trades)


def _deduct_commission(trade: Trade, commission_r: float) -> Trade:
    """Return a new Trade with commission_r subtracted from r_multiple and pnl_points."""
    from dataclasses import replace
    # Recompute pnl_points from the new R-multiple so the two fields stay in sync.
    # win is recalculated too — a trade that was a small winner may become a loser
    # after commission (common when exits are near breakeven).
    new_r   = trade.r_multiple - commission_r
    new_pnl = new_r * trade.risk_per_unit
    return replace(
        trade,
        r_multiple=new_r,
        pnl_points=new_pnl,
        win=int(new_pnl > 0),
    )


def _simulate_exit(
    future_bars: pd.DataFrame,
    direction: str,
    stop: float,
    target: float,
    entry: float,
    slippage_per_share: float = 0.02,
) -> tuple[float, pd.Timestamp]:
    """
    Walk forward through bars and return (exit_price, exit_time).

    Rules (applied in order, conservative):
      - If both stop and target are touched on the same bar → stop wins.
      - Stop fills with slippage (worse than the stop level).
      - Target fills at the exact target level (limit order, no slippage).
      - If session ends with no hit → exit at last bar's close.
    """
    for ts, row in future_bars.iterrows():
        if direction == "long":
            # A long stop is triggered when the bar's LOW dips to or below the stop level.
            # Using low (not close) is more realistic — the stop would fire intrabar.
            stop_hit   = row["low"]  <= stop
            # A long target is triggered when the bar's HIGH reaches or exceeds the target.
            target_hit = row["high"] >= target
        else:
            # Short: stop is triggered if HIGH touches the stop level (price moves against us).
            stop_hit   = row["high"] >= stop
            # Short: target is triggered if LOW touches or falls below the target.
            target_hit = row["low"]  <= target

        # When both are hit on the same bar we can't know the intrabar sequence —
        # assuming stop hit first is the conservative (pessimistic) choice.  This
        # avoids overfitting to optimistic scenarios and makes the backtest more robust.
        if stop_hit and target_hit:
            # Assume stop hit first (conservative)
            fill = stop - slippage_per_share if direction == "long" else stop + slippage_per_share
            return fill, ts
        if stop_hit:
            # Stop loss: fill is worse than the stop level because in fast markets
            # a market order to exit longs fills at bid (below stop), and shorts fill
            # at ask (above stop).
            fill = stop - slippage_per_share if direction == "long" else stop + slippage_per_share
            return fill, ts
        if target_hit:
            # Target is a limit order — it fills exactly at the target level with no
            # slippage (we are providing liquidity, not taking it).
            return target, ts   # Limit order — no slippage on targets

    # Session ended – exit at last close (no slippage, orderly close).
    # This is the time-stop: the ORB is an intraday strategy, and we never
    # carry open positions overnight to avoid gap risk.
    if future_bars.empty:
        # Edge case: entry bar was the last bar of the session — no exit bar exists.
        # Return NaT so downstream code can identify these degenerate trades.
        return entry, pd.Timestamp("NaT")

    last = future_bars.iloc[-1]
    return last["close"], future_bars.index[-1]


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_and_report(
    ticker: str,
    start: str = "",
    end: str = "",
    interval: str = "5m",
    or_minutes: int = 15,
    session_start: str = "09:30",
    session_end: str = "16:00",
    rr_ratio: float = 2.0,
    direction: str = "both",
    n_simulations: int = 500,
    risk_per_trade: float = 0.01,
    slippage_per_share: float = 0.02,
    commission_per_share: float = 0.0,
    source: str = "yfinance",
    alpaca_key: str = "",
    alpaca_secret: str = "",
    tv_csv_path: str = "",
    tv_exchange: str = "AMEX",
    tv_username: str = "",
    tv_password: str = "",
) -> pd.DataFrame:
    """
    Fetch data, run the ORB strategy, print performance stats vs buy-and-hold,
    and show plots.

    risk_per_trade converts R-multiples to % returns for fair comparison with
    buy-and-hold.  At 1% risk per trade, 1R = 1% of capital.

    Returns the trades DataFrame.
    """
    from core.performance import summary_stats
    from core.plots import plot_all
    from core.monte_carlo import run_monte_carlo, plot_monte_carlo_paths
    import matplotlib.pyplot as plt
    import yfinance as yf

    print(f"Fetching {interval} data for {ticker} ({start} → {end}) via {source} …")
    df = fetch_data(
        ticker, start=start, end=end, interval=interval, source=source,
        alpaca_key=alpaca_key, alpaca_secret=alpaca_secret,
        tv_csv_path=tv_csv_path, tv_exchange=tv_exchange,
        tv_username=tv_username, tv_password=tv_password,
    )
    if df.empty:
        print("No data returned. Check ticker, dates, or internet connection.")
        return pd.DataFrame()

    print(f"Running ORB strategy  (OR={or_minutes}m, RR={rr_ratio}, dir={direction}) …")
    trades = run_orb(
        df,
        or_minutes=or_minutes,
        session_start=session_start,
        session_end=session_end,
        rr_ratio=rr_ratio,
        direction=direction,
        slippage_per_share=slippage_per_share,
        commission_per_share=commission_per_share,
    )

    if trades.empty:
        print("No trades generated.")
        return trades

    # ── Buy-and-hold benchmark ─────────────────────────────────────────────
    # Always fetch daily data for the benchmark regardless of the strategy's
    # intraday interval — daily is sufficient for total/annualised return and drawdown.
    bnh_df = yf.download(ticker, start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False, multi_level_index=False)
    bnh_pct = float("nan")
    bnh_max_dd = float("nan")
    bnh_equity = None

    if not bnh_df.empty:
        bnh_df.columns = [c.lower() for c in bnh_df.columns]
        prices = bnh_df["close"].dropna()
        # Simple total return: (end_price / start_price - 1) × 100
        bnh_pct = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

        # Normalised equity curve (starts at 1.0) — dividing every price by the
        # first price puts buy-and-hold on the same scale as the ORB equity curve.
        bnh_equity = prices / prices.iloc[0]

        # Max drawdown for buy-and-hold: peak-to-trough % decline in the equity curve.
        rolling_max = prices.cummax()
        dd = (prices - rolling_max) / rolling_max * 100
        bnh_max_dd = dd.min()

    # ── Number of trading days in range (for annualisation) ───────────────
    # Use calendar days / 365.25 rather than trading days / 252 because buy-and-hold
    # returns are quoted on an annual basis; both approaches give the same CAGR
    # as long as we're consistent between strategy and benchmark.
    n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    years  = n_days / 365.25

    # ── Compounded equity curve ────────────────────────────────────────────
    # Each trade: equity *= (1 + r_multiple * risk_per_trade)
    # This means winners grow the pot; losers shrink it — realistic sizing.
    # np.cumprod is the vectorised equivalent of looping and multiplying sequentially.
    comp_equity = np.cumprod(1 + trades["r_multiple"].values * risk_per_trade)
    comp_total_pct  = (comp_equity[-1] - 1) * 100
    # CAGR formula: (ending_equity) ^ (1/years) - 1 — geometric annualisation.
    comp_ann_pct    = ((comp_equity[-1]) ** (1 / years) - 1) * 100 if years > 0 else float("nan")
    # Running maximum of the equity curve — used to measure how far equity has
    # fallen from its peak at each point in time.
    comp_roll_max   = np.maximum.accumulate(comp_equity)
    comp_max_dd_pct = ((comp_equity - comp_roll_max) / comp_roll_max * 100).min()

    # ── Flat (non-compounded) equity for comparison ────────────────────────
    # Flat sizing keeps position size constant in dollar terms (not % of equity).
    # It's easier to analyse — the equity curve slope directly reflects win rate
    # × avg RR — but it's less realistic for live trading.
    flat_total_pct = trades["r_multiple"].sum() * risk_per_trade * 100
    flat_ann_pct   = ((1 + flat_total_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else float("nan")
    # Flat drawdown is measured in cumulative R, then converted to % using risk_per_trade.
    flat_cum_r     = trades["r_multiple"].cumsum()
    flat_roll_max  = flat_cum_r.cummax()
    flat_max_dd_pct = ((flat_cum_r - flat_roll_max) * risk_per_trade * 100).min()

    bnh_ann_pct = ((1 + bnh_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else float("nan")

    # ── Print comparison table ─────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  {ticker} ORB vs Buy-and-Hold  ({start} → {end})")
    print(f"  Assumption: {risk_per_trade*100:.1f}% of capital risked per trade")
    print(f"{'='*64}")
    print(f"  {'Metric':<28} {'ORB (compound)':>14} {'ORB (flat)':>10} {'Buy & Hold':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Total return (%)':28} {comp_total_pct:>14.2f} {flat_total_pct:>10.2f} {bnh_pct:>10.2f}")
    print(f"  {'Annualised return (%)':28} {comp_ann_pct:>14.2f} {flat_ann_pct:>10.2f} {bnh_ann_pct:>10.2f}")
    print(f"  {'Max drawdown (%)':28} {comp_max_dd_pct:>14.2f} {flat_max_dd_pct:>10.2f} {bnh_max_dd:>10.2f}")
    print(f"  {'Trades':28} {len(trades):>14} {len(trades):>10} {'1':>10}")
    print(f"{'='*64}")

    print(f"\n{'='*56}")
    print(f"  {ticker} ORB — Strategy Stats")
    print(f"{'='*56}")
    stats = summary_stats(trades)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:>10.4f}")
        else:
            print(f"  {k:<28} {v:>10}")

    print(f"\nRunning Monte Carlo ({n_simulations} simulations) …")
    simulations = run_monte_carlo(trades, n_simulations=n_simulations)

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_all(trades)
    plot_monte_carlo_paths(simulations)

    # ── Overlay equity curves (both normalised to 1.0) ────────────────────
    # Using exit_time as the x-axis anchors each trade result to when the P&L
    # was realised — this is the correct timeline for an equity curve.
    orb_dates      = pd.to_datetime(trades["exit_time"])
    # Flat equity: start at 1.0 and add each trade's P&L as a fixed fraction of capital.
    orb_flat_equity = 1.0 + trades["r_multiple"].cumsum().values * risk_per_trade

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(orb_dates, comp_equity,       label=f"ORB compounded ({risk_per_trade*100:.0f}% risk/trade)", linewidth=1.5)
    ax.plot(orb_dates, orb_flat_equity,   label=f"ORB flat sizing",  linewidth=1.2, linestyle="-.", alpha=0.7)
    if bnh_equity is not None:
        ax.plot(bnh_equity.index, bnh_equity.values, label="Buy & Hold", linewidth=1.5, linestyle="--")
    # Horizontal line at 1.0 marks the starting capital level — anything below
    # means the strategy is underwater relative to cash at that point in time.
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(f"{ticker}: ORB vs Buy-and-Hold (normalised equity, slippage + commission included)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity multiple (1.0 = starting capital)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    return trades


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core.config import Config

    # ── Option A: Yahoo Finance (free, no setup, ~60 days of 5m data) ─────
    trades = run_and_report(
        ticker=Config.TICKER,
        start=Config.START_DATE,
        end=Config.END_DATE,
        interval=Config.INTERVAL,
        or_minutes=Config.OR_MINUTES,
        rr_ratio=Config.RR_RATIO,
        direction="both",
        source=Config.DATA_SOURCE,
        risk_per_trade=Config.RISK_PER_TRADE,
        slippage_per_share=Config.SLIPPAGE_PER_SHARE,
        n_simulations=Config.N_SIMULATIONS,
    )

    # ── Option B: TradingView CSV export (recommended for 5yr backtest) ────
    # 1. Open SPY on TradingView at the 5m timeframe
    # 2. Click the floppy-disk / export icon (top-right of chart) →
    #    "Export chart data" → save CSV anywhere
    # 3. Paste the path below:
    #
    # trades = run_and_report(
    #     ticker=Config.TICKER,
    #     or_minutes=Config.OR_MINUTES,
    #     rr_ratio=Config.RR_RATIO,
    #     direction="both",
    #     source="tv_csv",
    #     tv_csv_path="/Users/ryanraj/Downloads/SPY_5m.csv",
    # )

    # ── Option C: Alpaca (free account, 5+ years minute data) ─────────────
    # pip install alpaca-py  →  sign up at alpaca.markets  →  get API keys
    #
    # trades = run_and_report(
    #     ticker=Config.TICKER,
    #     start="2021-01-01",
    #     end="2026-01-01",
    #     interval=Config.INTERVAL,
    #     or_minutes=Config.OR_MINUTES,
    #     rr_ratio=Config.RR_RATIO,
    #     direction="both",
    #     source="alpaca",
    #     alpaca_key="YOUR_KEY_HERE",
    #     alpaca_secret="YOUR_SECRET_HERE",
    # )

    print(trades.head())
