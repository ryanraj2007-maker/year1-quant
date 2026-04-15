"""
Futures Opening Range Breakout (ORB) Strategy

Supports: ES, MES, NQ, MNQ (CME Globex)

Key differences from stock ORB:
  - P&L calculated in dollars using contract multiplier
  - Position sized in whole contracts based on dollar risk
  - Slippage measured in ticks
  - Margin-based account modelling
  - RTH session: 09:30–16:15 ET (futures close 15 min later than stocks)

Data sources:
  - yfinance  : ES=F, MES=F, NQ=F, MNQ=F  (last ~60 days of 5m data, free)
  - Databento : 5+ years of tick/bar data  (free tier, pip install databento)
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Allow running this file directly without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.trade_log import Trade, create_trade, trades_to_dataframe


# ---------------------------------------------------------------------------
# Contract specifications
# ---------------------------------------------------------------------------

@dataclass
class FuturesSpec:
    """
    Immutable spec for a single futures contract.
    All dollar values are per contract per point/tick.
    """
    multiplier:  float   # $ per point  (ES=50, MES=5, NQ=20, MNQ=2)
    tick_size:   float   # minimum price increment in points (all CME = 0.25)
    tick_value:  float   # $ per tick   (ES=12.50, MES=1.25, NQ=5.00, MNQ=0.50)
    margin:      float   # approx initial margin per contract in $
    yf_ticker:   str     # yfinance continuous contract symbol


# Micro contracts (MES/MNQ) are 1/10th the size of full contracts (ES/NQ).
# They're ideal for small accounts and precise position sizing.
FUTURES = {
    "ES":  FuturesSpec(50,   0.25, 12.50, 15_000, "ES=F"),
    "MES": FuturesSpec(5,    0.25,  1.25,  1_500, "MES=F"),
    "NQ":  FuturesSpec(20,   0.25,  5.00, 20_000, "NQ=F"),
    "MNQ": FuturesSpec(2,    0.25,  0.50,  2_000, "MNQ=F"),
}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_data_yfinance(
    symbol: str,
    start: str,
    end: str,
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Fetch futures OHLC data via yfinance.

    Uses continuous front-month contracts (e.g. ES=F).
    Limited to ~60 days of intraday history.

    symbol   : "ES", "MES", "NQ", or "MNQ"
    interval : "1m", "5m", "15m", "1h"
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    spec = _get_spec(symbol)

    # yfinance imposes hard per-request and total history limits by interval
    _chunk = {"1m": 7, "5m": 59, "15m": 59, "1h": 729}.get(interval, 59)
    _limit = {"1m": 30, "5m": 60, "15m": 60, "1h": 730}.get(interval, 60)

    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)

    # Clip the start date to the earliest date yfinance can serve for this interval
    earliest = pd.Timestamp.now().normalize() - pd.Timedelta(days=_limit - 1)
    if start_dt < earliest:
        print(f"  [yfinance] clipping start to {earliest.date()} "
              f"({interval} limit = {_limit} days)")
        start_dt = earliest

    # Download in chunks to stay within per-request day limits
    chunks, s = [], start_dt
    while s < end_dt:
        e = min(s + pd.Timedelta(days=_chunk), end_dt)
        chunk = yf.download(spec.yf_ticker,
                            start=s.strftime("%Y-%m-%d"),
                            end=e.strftime("%Y-%m-%d"),
                            interval=interval,
                            auto_adjust=True, progress=False,
                            multi_level_index=False)
        if not chunk.empty:
            chunks.append(chunk)
        s = e

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks)
    # Remove any duplicate timestamps from chunk boundaries
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    # Convert to tz-naive Eastern time for consistent session filtering
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df


def fetch_data_databento(
    symbol: str,
    start: str,
    end: str,
    interval: str = "5m",
    api_key: str = "",
) -> pd.DataFrame:
    """
    Fetch futures OHLC data via Databento (5+ years of history, free tier).

    Sign up free at https://databento.com — get your API key from the portal.
    Free tier includes 10 GB of data.

    Install:  pip install databento

    symbol   : "ES", "MES", "NQ", or "MNQ"
    interval : "1m", "5m", "15m", "1h", "1d"
    api_key  : Databento API key (starts with "db-")
    """
    try:
        import databento as db
    except ImportError:
        raise ImportError(
            "Databento is required for 5-year futures backtests:\n"
            "  pip install databento\n"
            "Sign up free at https://databento.com"
        )

    # Map our simple interval strings to Databento schema names
    _schema_map = {
        "1m":  "ohlcv-1m",
        "5m":  "ohlcv-5m",
        "15m": "ohlcv-15m" if hasattr(db.Schema, "OHLCV_15M") else "ohlcv-5m",
        "1h":  "ohlcv-1h",
        "1d":  "ohlcv-1d",
    }
    schema = _schema_map.get(interval, "ohlcv-5m")

    # Databento uses root symbol + exchange, e.g. "ES.c.0" for continuous front-month
    # ".c.0" means continuous (front-month roll adjusted)
    db_symbol = f"{symbol}.c.0"

    client = db.Historical(api_key)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",       # CME Globex dataset
        symbols=[db_symbol],
        schema=schema,
        start=start,
        end=end,
        stype_in="continuous",
    )
    df = data.to_df()

    if df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    df.columns = [c.lower() for c in df.columns]

    # Databento stores prices in fixed-point integers (divide by 1e9 to get dollars)
    # Only divide if values look like raw fixed-point (> 100,000 for ES at ~4000 pts)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns and df[col].max() > 100_000:
            df[col] = df[col] / 1e9

    return df[["open", "high", "low", "close", "volume"]
              if "volume" in df.columns else ["open", "high", "low", "close"]]


def fetch_data(
    symbol: str,
    start: str = "",
    end: str = "",
    interval: str = "5m",
    source: str = "yfinance",
    alpaca_key: str = "",
    alpaca_secret: str = "",
    databento_key: str = "",
) -> pd.DataFrame:
    """
    Unified futures data fetcher.

    source="yfinance"   — free, no setup, last ~60 days of intraday
    source="alpaca"     — free account, 5+ years of futures data
    source="databento"  — free account, 5+ years, pip install databento
    """
    if source == "databento":
        return fetch_data_databento(symbol, start, end, interval, databento_key)
    elif source == "alpaca":
        return fetch_data_alpaca_futures(symbol, start, end, interval,
                                         alpaca_key, alpaca_secret)
    else:
        # Default to yfinance — no setup required
        return fetch_data_yfinance(symbol, start, end, interval)


def fetch_data_alpaca_futures(
    symbol: str,
    start: str,
    end: str,
    interval: str = "5m",
    api_key: str = "",
    api_secret: str = "",
) -> pd.DataFrame:
    """Fetch futures data via Alpaca (crypto/futures endpoint)."""
    try:
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        raise ImportError("pip install alpaca-py")

    # Map simple interval strings to Alpaca TimeFrame objects
    _tf = {"1m": TimeFrame(1, TimeFrameUnit.Minute),
            "5m": TimeFrame(5, TimeFrameUnit.Minute),
           "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "1h": TimeFrame(1, TimeFrameUnit.Hour),
            "1d": TimeFrame(1, TimeFrameUnit.Day)}

    spec = _get_spec(symbol)
    tf   = _tf.get(interval, TimeFrame(5, TimeFrameUnit.Minute))

    client = StockHistoricalDataClient(api_key, api_secret)
    req = StockBarsRequest(
        symbol_or_symbols=spec.yf_ticker,
        timeframe=tf,
        start=pd.Timestamp(start, tz="America/New_York"),
        end=pd.Timestamp(end,   tz="America/New_York"),
        adjustment="all",   # apply split/dividend adjustments
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        return pd.DataFrame()

    # Alpaca returns a MultiIndex (symbol, timestamp) — flatten to just timestamp
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(spec.yf_ticker, level="symbol")

    bars.index = pd.to_datetime(bars.index)
    # Strip timezone after converting to Eastern — simpler for session math
    if bars.index.tz is not None:
        bars.index = bars.index.tz_convert("America/New_York").tz_localize(None)

    bars.columns = [c.lower() for c in bars.columns]
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in bars.columns]
    return bars[cols]


# ---------------------------------------------------------------------------
# Core strategy
# ---------------------------------------------------------------------------

def run_futures_orb(
    df: pd.DataFrame,
    symbol: str,
    or_minutes: int = 15,
    session_start: str = "09:30",
    session_end: str = "16:15",
    rr_ratio: float = 2.0,
    direction: str = "both",
    entry_slippage_ticks: int = 1,   # ticks of slippage on entry
    exit_slippage_ticks: int = 1,    # ticks of slippage on stop fills
) -> pd.DataFrame:
    """
    Run the futures ORB strategy.

    Parameters
    ----------
    df                    : OHLC DataFrame, tz-naive Eastern time index.
    symbol                : "ES", "MES", "NQ", or "MNQ"
    or_minutes            : Opening range window in minutes.
    session_start         : RTH open (default 09:30 ET).
    session_end           : RTH close (default 16:15 ET for futures).
    rr_ratio              : Reward-to-risk ratio.
    direction             : "long", "short", or "both".
    entry_slippage_ticks  : Ticks of entry slippage (market order fill).
    exit_slippage_ticks   : Ticks of exit slippage on stop fills only.
                            Target fills are limit orders — no slippage.

    Returns
    -------
    pd.DataFrame of trades with r_multiple column, compatible with the
    V2 core framework (performance, monte_carlo, plots).
    Also includes dollar_pnl column (P&L per contract in $).
    """
    spec = _get_spec(symbol)

    # Convert tick counts to point values using the contract's tick size
    slip_entry = entry_slippage_ticks * spec.tick_size
    slip_exit  = exit_slippage_ticks  * spec.tick_size

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)

    trades: list[Trade] = []

    # Process one trading day at a time — the ORB resets every day
    for date, day_df in df.groupby(df.index.date):
        # Restrict to regular trading hours only
        day_df = day_df.between_time(session_start, session_end)
        if day_df.empty:
            continue

        session_open = pd.Timestamp(str(date) + " " + session_start)
        or_end       = session_open + pd.Timedelta(minutes=or_minutes)

        # Opening range: bars strictly within the first or_minutes of the session
        or_df   = day_df[day_df.index < or_end]
        # Post-OR: bars available for entries and exits
        post_or = day_df[day_df.index >= or_end]

        if len(or_df) < 1 or post_or.empty:
            continue

        # The OR high and low define the breakout levels for the whole day
        or_high = or_df["high"].max()
        or_low  = or_df["low"].min()

        # Skip flat opens — no range means no breakout to trade
        if or_high == or_low:
            continue

        trade_taken = False

        for ts, row in post_or.iterrows():
            # One trade per day — once triggered, stop scanning
            if trade_taken:
                break

            # Look ahead to the next bar for a realistic entry
            future_bars = post_or[post_or.index > ts]
            if future_bars.empty:
                continue

            next_bar   = future_bars.iloc[0]
            entry_time = future_bars.index[0]

            # ── Long breakout ───────────────────────────────────────────────
            # Confirm on close above OR high — avoids wicks that briefly pierce
            if direction in ("long", "both") and row["close"] > or_high:
                # Enter at next bar's open + slippage (market order fills worse)
                entry  = next_bar["open"] + slip_entry
                # Stop pre-adjusted for exit slippage so risk calc is accurate
                stop   = or_low  - slip_exit
                risk   = entry - stop
                if risk <= 0:
                    continue
                # Target is a fixed multiple of our actual risk
                target = entry + rr_ratio * risk

                exit_price, exit_time = _simulate_exit(
                    future_bars.iloc[1:], "long", or_low, target,
                    entry, slip_exit,
                )

                trades.append(_make_trade(
                    entry_time, exit_time, "long",
                    entry, exit_price, stop, target, spec,
                ))
                trade_taken = True

            # ── Short breakout ──────────────────────────────────────────────
            elif direction in ("short", "both") and row["close"] < or_low:
                # Enter at next bar's open - slippage (worse for short = lower fill)
                entry  = next_bar["open"] - slip_entry
                # Stop above OR high + exit slippage
                stop   = or_high + slip_exit
                risk   = stop - entry
                if risk <= 0:
                    continue
                target = entry - rr_ratio * risk

                exit_price, exit_time = _simulate_exit(
                    future_bars.iloc[1:], "short", or_high, target,
                    entry, slip_exit,
                )

                trades.append(_make_trade(
                    entry_time, exit_time, "short",
                    entry, exit_price, stop, target, spec,
                ))
                trade_taken = True

    if not trades:
        return pd.DataFrame()

    df_trades = trades_to_dataframe(trades)
    # Dollar P&L = R points × multiplier — useful for sizing and reporting
    df_trades["dollar_pnl"] = df_trades["pnl_points"] * spec.multiplier
    return df_trades


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_spec(symbol: str) -> FuturesSpec:
    """Look up a FuturesSpec by symbol name (case-insensitive)."""
    s = symbol.upper()
    if s not in FUTURES:
        raise ValueError(f"Unknown symbol '{symbol}'. Choose from: {list(FUTURES)}")
    return FUTURES[s]


def _simulate_exit(
    future_bars: pd.DataFrame,
    direction: str,
    raw_stop: float,   # OR level (before exit slippage)
    target: float,
    entry: float,
    slip_exit: float,
) -> tuple[float, pd.Timestamp]:
    """
    Walk bars forward and return the first exit hit (stop, target, or session end).

    Stop detection uses the bar's low/high (intrabar), not the close —
    a stop can be triggered at any point within a bar even if price recovers.
    Same-bar ambiguity (stop AND target both touched) defaults to stop first,
    the more conservative assumption.
    """
    for ts, row in future_bars.iterrows():
        if direction == "long":
            # For a long: stop is below entry, target is above
            stop_hit   = row["low"]  <= raw_stop
            target_hit = row["high"] >= target
        else:
            # For a short: stop is above entry, target is below
            stop_hit   = row["high"] >= raw_stop
            target_hit = row["low"]  <= target

        if stop_hit and target_hit:
            # Both touched — assume stop fired first (worst case / conservative)
            fill = raw_stop - slip_exit if direction == "long" else raw_stop + slip_exit
            return fill, ts
        if stop_hit:
            # Stop fills worse than the stop level (market order, adverse slippage)
            fill = raw_stop - slip_exit if direction == "long" else raw_stop + slip_exit
            return fill, ts
        if target_hit:
            # Target fills exactly at the limit price — no slippage on limit orders
            return target, ts

    # No stop or target was hit — exit at the last bar's close (session end)
    if future_bars.empty:
        return entry, pd.Timestamp("NaT")
    last = future_bars.iloc[-1]
    return last["close"], future_bars.index[-1]


def _make_trade(
    entry_time, exit_time, direction,
    entry, exit_price, stop, target, spec: FuturesSpec,
) -> Trade:
    """Create a Trade using point-based risk (compatible with core framework)."""
    return create_trade(
        entry_time=entry_time,
        exit_time=exit_time,
        direction=direction,
        entry_price=entry,
        exit_price=exit_price,
        stop_price=stop,
        target_price=target,
    )


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def contracts_per_trade(
    account_equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
    spec: FuturesSpec,
) -> int:
    """
    How many contracts to trade given a fixed % risk of account equity.

    dollar_risk   = account_equity × risk_pct
    risk_per_contract = |entry - stop| × multiplier
    contracts     = floor(dollar_risk / risk_per_contract)

    Always returns at least 1 — we don't skip trades just because sizing
    rounds down to 0. The minimum position is 1 contract.
    """
    dollar_risk       = account_equity * risk_pct
    risk_per_contract = abs(entry - stop) * spec.multiplier
    if risk_per_contract == 0:
        return 0
    return max(1, int(dollar_risk / risk_per_contract))


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_and_report(
    symbol: str,
    start: str = "",
    end: str = "",
    interval: str = "5m",
    or_minutes: int = 15,
    session_start: str = "09:30",
    session_end: str = "16:15",
    rr_ratio: float = 2.0,
    direction: str = "both",
    account_size: float = 10_000,
    risk_per_trade: float = 0.01,
    entry_slippage_ticks: int = 1,
    exit_slippage_ticks: int = 1,
    n_simulations: int = 500,
    source: str = "yfinance",
    databento_key: str = "",
) -> pd.DataFrame:
    """
    Fetch data, run futures ORB, print stats vs buy-and-hold, show plots.

    account_size    : Starting capital in $.
    risk_per_trade  : Fraction of account risked per trade (e.g. 0.01 = 1%).
    """
    from core.performance import summary_stats
    from core.plots import plot_all
    from core.monte_carlo import run_monte_carlo, plot_monte_carlo_paths
    import matplotlib.pyplot as plt
    import yfinance as yf

    spec = _get_spec(symbol)

    print(f"Fetching {interval} data for {symbol} ({start} → {end}) via {source} …")
    df = fetch_data(symbol, start=start, end=end, interval=interval,
                    source=source, databento_key=databento_key)
    if df.empty:
        print("No data returned.")
        return pd.DataFrame()

    print(f"Running Futures ORB  (OR={or_minutes}m, RR={rr_ratio}, dir={direction}, "
          f"slip={entry_slippage_ticks}t entry/{exit_slippage_ticks}t exit) …")
    trades = run_futures_orb(
        df, symbol=symbol,
        or_minutes=or_minutes,
        session_start=session_start,
        session_end=session_end,
        rr_ratio=rr_ratio,
        direction=direction,
        entry_slippage_ticks=entry_slippage_ticks,
        exit_slippage_ticks=exit_slippage_ticks,
    )

    if trades.empty:
        print("No trades generated.")
        return trades

    # ── Compounded equity (% of account) ──────────────────────────────────
    # Each trade's return = r_multiple × risk_per_trade as a fraction of equity
    # cumprod compounds these: a +2R trade at 1% risk = +2% on that trade
    comp_equity     = np.cumprod(1 + trades["r_multiple"].values * risk_per_trade)
    comp_total_pct  = (comp_equity[-1] - 1) * 100
    comp_roll_max   = np.maximum.accumulate(comp_equity)
    comp_max_dd_pct = ((comp_equity - comp_roll_max) / comp_roll_max * 100).min()

    n_days  = (pd.Timestamp(end) - pd.Timestamp(start)).days
    years   = max(n_days / 365.25, 1 / 365.25)   # floor at 1 day to avoid div/0
    comp_ann_pct = ((comp_equity[-1]) ** (1 / years) - 1) * 100

    # Dollar P&L (flat 1 contract per trade for transparency)
    # Shows raw edge without position sizing effects
    total_dollar_pnl = trades["dollar_pnl"].sum()
    avg_dollar_pnl   = trades["dollar_pnl"].mean()

    # ── Buy-and-hold benchmark (continuous front month) ───────────────────
    # Use daily data for BnH — we don't need intraday precision for a passive hold
    bnh_pct = bnh_max_dd = float("nan")
    bnh_equity = None
    bnh_df = yf.download(spec.yf_ticker, start=start, end=end,
                         interval="1d", auto_adjust=True,
                         progress=False, multi_level_index=False)
    if not bnh_df.empty:
        bnh_df.columns = [c.lower() for c in bnh_df.columns]
        p = bnh_df["close"].dropna()
        bnh_pct    = (p.iloc[-1] / p.iloc[0] - 1) * 100
        # Normalise to 1.0 for direct comparison with comp_equity
        bnh_equity = p / p.iloc[0]
        bnh_max_dd = ((p - p.cummax()) / p.cummax() * 100).min()

    bnh_ann_pct = ((1 + bnh_pct / 100) ** (1 / years) - 1) * 100

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  {symbol} Futures ORB vs Buy-and-Hold  ({start} → {end})")
    print(f"  Account: ${account_size:,.0f}  |  Risk/trade: {risk_per_trade*100:.1f}%  "
          f"|  Slippage: {entry_slippage_ticks}t in / {exit_slippage_ticks}t out")
    print(f"{'='*64}")
    print(f"  {'Metric':<30} {'ORB (compound)':>14} {'Buy & Hold':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Total return (%)':30} {comp_total_pct:>14.2f} {bnh_pct:>12.2f}")
    print(f"  {'Annualised return (%)':30} {comp_ann_pct:>14.2f} {bnh_ann_pct:>12.2f}")
    print(f"  {'Max drawdown (%)':30} {comp_max_dd_pct:>14.2f} {bnh_max_dd:>12.2f}")
    print(f"  {'Trades':30} {len(trades):>14}")
    print(f"{'='*64}")

    # Dollar P&L section — educational: shows the raw contract value
    # without any position sizing, so you can see the intrinsic edge
    print(f"\n  Dollar P&L (1 contract, no sizing):")
    print(f"  {'Contract':20} {symbol} ({spec.yf_ticker})")
    print(f"  {'Multiplier':20} ${spec.multiplier}/point")
    print(f"  {'Tick value':20} ${spec.tick_value}/tick")
    print(f"  {'Total P&L':20} ${total_dollar_pnl:,.2f}")
    print(f"  {'Avg P&L / trade':20} ${avg_dollar_pnl:,.2f}")
    print(f"  {'Margin required':20} ~${spec.margin:,.0f}/contract")

    print(f"\n{'='*64}")
    print(f"  Strategy Stats")
    print(f"{'='*64}")
    stats = summary_stats(trades)
    for k, v in stats.items():
        print(f"  {k:<30} {v:>14.4f}" if isinstance(v, float)
              else f"  {k:<30} {v:>14}")

    print(f"\nRunning Monte Carlo ({n_simulations} simulations) …")
    simulations = run_monte_carlo(trades, n_simulations=n_simulations)

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_all(trades)
    plot_monte_carlo_paths(simulations)

    orb_dates = pd.to_datetime(trades["exit_time"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: normalised equity curves for direct visual comparison
    axes[0].plot(orb_dates, comp_equity, label=f"ORB compounded", linewidth=1.5)
    if bnh_equity is not None:
        axes[0].plot(bnh_equity.index, bnh_equity.values,
                     label="Buy & Hold", linewidth=1.5, linestyle="--")
    axes[0].axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    axes[0].set_title(f"{symbol} ORB vs Buy-and-Hold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Equity multiple")
    axes[0].legend()
    axes[0].grid(True)

    # Right: per-trade dollar P&L bar chart — green = winner, red = loser
    axes[1].bar(range(len(trades)),
                trades["dollar_pnl"].values,
                color=["#2ecc71" if x > 0 else "#e74c3c"
                       for x in trades["dollar_pnl"]],
                alpha=0.7, width=1.0)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"{symbol} Dollar P&L per trade (1 contract)")
    axes[1].set_xlabel("Trade #")
    axes[1].set_ylabel("P&L ($)")
    axes[1].grid(True, axis="y")

    plt.tight_layout()
    plt.show()

    return trades


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core.config import Config

    # ── MES (Micro E-mini S&P 500) via yfinance — last ~60 days ───────────
    # MES is ideal for small accounts: $5/point, ~$1,500 margin/contract
    trades = run_and_report(
        symbol=Config.SYMBOL,
        start=Config.START_DATE,
        end=Config.END_DATE,
        interval=Config.INTERVAL,
        or_minutes=Config.OR_MINUTES,
        rr_ratio=Config.RR_RATIO,
        direction="both",
        account_size=Config.ACCOUNT_SIZE,
        risk_per_trade=Config.RISK_PER_TRADE,
        entry_slippage_ticks=Config.ENTRY_SLIP_TICKS,
        exit_slippage_ticks=Config.EXIT_SLIP_TICKS,
        n_simulations=Config.N_SIMULATIONS,
        source=Config.DATA_SOURCE,
    )
    print(trades[["entry_time", "exit_time", "direction",
                  "entry_price", "exit_price", "r_multiple", "dollar_pnl"]].head(10))

    # ── ES (E-mini S&P 500) via Databento — 5 years ───────────────────────
    # Sign up free at https://databento.com — set DATA_SOURCE = "databento"
    # and SYMBOL = "ES" in Config, then uncomment below:
    #
    # trades = run_and_report(
    #     symbol="ES",
    #     start="2021-01-01",
    #     end="2026-01-01",
    #     interval=Config.INTERVAL,
    #     or_minutes=Config.OR_MINUTES,
    #     rr_ratio=Config.RR_RATIO,
    #     direction="both",
    #     account_size=Config.ACCOUNT_SIZE,
    #     risk_per_trade=Config.RISK_PER_TRADE,
    #     entry_slippage_ticks=Config.ENTRY_SLIP_TICKS,
    #     exit_slippage_ticks=Config.EXIT_SLIP_TICKS,
    #     source="databento",
    #     databento_key="db-YOUR-KEY-HERE",
    # )
