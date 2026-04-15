"""
8 AM Key Level ORB Strategy

Based on the Pine Script strategy by RP Profits / QuantPad.

Logic:
  1. Track Asia session (18:00–03:00 ET) and London session (03:30–11:30 ET) H/L.
  2. The 8:00–8:15 AM 15-minute candle defines the key zone (high, low, midpoint).
  3. At 9:30 AM:
       - If price closes ABOVE zone_high  → bullish bias
       - If price closes BELOW zone_low   → bearish bias
  4. Entry: limit order at zone midpoint, active 09:00–11:00 window only.
  5. Stop loss: nearest Asia or London session level within [sl_scan_low, sl_scan_high]
     points of entry. Fallback = sl_default if no level found.
  6. Take profit: entry ± (SL distance × rr_ratio).
  7. One trade per day. Invalidated if 8AM range > large_range_threshold.

Requires intraday data including overnight (Alpaca recommended for MES/ES).
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.trade_log import create_trade, trades_to_dataframe
# Reuse the contract specs and data fetchers already defined in futures_orb
from strategies.futures_orb import FUTURES, FuturesSpec, fetch_data, _get_spec


# ---------------------------------------------------------------------------
# Per-day state
# ---------------------------------------------------------------------------

@dataclass
class DayState:
    """
    Tracks everything we've accumulated for the current trading day.
    Reset fresh for every day in the main loop.

    Separating state into a dataclass makes the main loop easier to read —
    instead of tracking 10 loose variables, you just pass around one object.
    """
    # The 8AM zone (defined by the 8:00–8:15 candle)
    zone_high: float = np.nan
    zone_low:  float = np.nan
    zone_mid:  float = np.nan   # midpoint = (zone_high + zone_low) / 2
    zone_set:  bool  = False
    trade_banned: bool = False  # True if the zone range is too wide to trade

    # Session high/low levels used to anchor the stop loss
    asia_high:   float = np.nan
    asia_low:    float = np.nan
    london_high: float = np.nan
    london_low:  float = np.nan

    # Trade state for this day
    direction:    Optional[str] = None   # "bullish" | "bearish"
    trade_taken:  bool = False
    entry_price:  float = np.nan
    sl_price:     float = np.nan
    tp_price:     float = np.nan


# ---------------------------------------------------------------------------
# Core strategy
# ---------------------------------------------------------------------------

def run_key_level_orb(
    df: pd.DataFrame,
    symbol: str = "MES",
    # Zone / session times (ET)
    asia_start:   str = "18:00",
    asia_end:     str = "03:00",
    london_start: str = "03:30",
    london_end:   str = "11:30",
    zone_start:   str = "08:00",
    zone_end:     str = "08:15",
    direction_time: str = "09:30",
    trade_window_start: str = "09:00",
    trade_window_end:   str = "11:00",
    # Risk parameters
    sl_scan_low:  float = 5.0,    # min pts from entry for SL anchor
    sl_scan_high: float = 10.0,   # max pts from entry for SL anchor
    sl_default:   float = 8.0,    # fallback SL in pts if no level found
    rr_ratio:     float = 4.0,
    # Filters
    invalidate_large_range: bool  = True,
    large_range_threshold:  float = 20.0,
    entry_slippage_ticks: int = 1,
    exit_slippage_ticks:  int = 1,
) -> pd.DataFrame:
    """
    Run the 8AM Key Level ORB strategy.

    Parameters
    ----------
    df : OHLC DataFrame with tz-naive Eastern-time index.
         Must include overnight bars (Asia + London sessions).
    symbol : futures symbol for contract spec (tick size / multiplier).

    Returns
    -------
    pd.DataFrame of trades compatible with the V2 core framework.
    """
    spec = _get_spec(symbol)
    # Convert tick counts to point values
    slip_entry = entry_slippage_ticks * spec.tick_size
    slip_exit  = exit_slippage_ticks  * spec.tick_size

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    # Normalise to tz-naive Eastern time — needed for session time comparisons
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    df = df.sort_index()

    # ── Group into "trading days" that start at 18:00 ─────────────────────
    # Each futures day runs 18:00 (Sunday/prev day) → 17:00 next day.
    # We key everything off the calendar date of the RTH open (next morning).
    # So a bar at Monday 18:30 maps to Tuesday's trading date.
    def trading_date(ts: pd.Timestamp) -> pd.Timestamp:
        """Map a timestamp to the RTH date it belongs to."""
        if ts.hour >= 18:
            # Overnight bars (18:00+) belong to the next calendar day's session
            return (ts + pd.Timedelta(days=1)).normalize()
        return ts.normalize()

    df["_trade_date"] = df.index.map(trading_date)

    trades = []

    for trade_date, day_df in df.groupby("_trade_date"):
        # Fresh state object for each day — no carryover between days
        state = DayState()

        # ── Build zone from the 8:00–8:15 window ──────────────────────────
        # The zone is the price range traded in the 15 minutes before the
        # US pre-market becomes active. It acts as a key support/resistance
        # level that price often returns to after the NY open.
        zone_mask = (
            (day_df.index.time >= pd.Timestamp(f"{trade_date.date()} {zone_start}").time()) &
            (day_df.index.time <  pd.Timestamp(f"{trade_date.date()} {zone_end}").time())
        )
        zone_bars = day_df[zone_mask]
        if zone_bars.empty:
            continue

        state.zone_high = zone_bars["high"].max()
        state.zone_low  = zone_bars["low"].min()
        # Midpoint is the entry target — price often retraces here before continuing
        state.zone_mid  = (state.zone_high + state.zone_low) / 2.0
        state.zone_set  = True

        # If the zone is too wide, it suggests a volatile/choppy pre-market —
        # don't trade this day as the levels are less reliable
        if invalidate_large_range and (state.zone_high - state.zone_low) > large_range_threshold:
            state.trade_banned = True

        # ── Accumulate Asia session H/L ────────────────────────────────────
        # Asia runs 18:00 on prev calendar day → 03:00 on trade_date.
        # These levels are used to anchor the stop loss.
        prev_date = trade_date - pd.Timedelta(days=1)

        asia_bars = day_df[
            (day_df.index >= pd.Timestamp(f"{prev_date.date()} {asia_start}")) &
            (day_df.index <  pd.Timestamp(f"{trade_date.date()} {asia_end}"))
        ]
        if not asia_bars.empty:
            state.asia_high = asia_bars["high"].max()
            state.asia_low  = asia_bars["low"].min()

        # ── Accumulate London session H/L ──────────────────────────────────
        # London session: 03:30–11:30 ET. These levels often become intraday
        # support/resistance during the NY session.
        london_bars = day_df[
            (day_df.index.time >= pd.Timestamp(f"{trade_date.date()} {london_start}").time()) &
            (day_df.index.time <  pd.Timestamp(f"{trade_date.date()} {london_end}").time())
        ]
        if not london_bars.empty:
            state.london_high = london_bars["high"].max()
            state.london_low  = london_bars["low"].min()

        if state.trade_banned:
            continue

        # ── Direction at 9:30 ──────────────────────────────────────────────
        # Use the 9:30 bar's close to determine bias for the day.
        # Price above zone_high = bullish (zone becomes support).
        # Price below zone_low  = bearish (zone becomes resistance).
        # Price inside the zone = no trade — no clear bias.
        dir_bar = day_df[
            day_df.index == pd.Timestamp(f"{trade_date.date()} {direction_time}")
        ]
        if dir_bar.empty:
            # 9:30 bar might be missing on some days — use the next available bar
            after_930 = day_df[day_df.index.time >= pd.Timestamp(f"{trade_date.date()} {direction_time}").time()]
            if after_930.empty:
                continue
            dir_bar = after_930.iloc[[0]]

        dir_close = float(dir_bar["close"].iloc[0])

        if dir_close > state.zone_high:
            state.direction = "bullish"
        elif dir_close < state.zone_low:
            state.direction = "bearish"
        else:
            # Price is inside the zone — no directional bias, skip the day
            continue

        # ── Trade window bars ──────────────────────────────────────────────
        # Only look for entries between 09:00 and 11:00 ET.
        # Outside this window the edge diminishes as the market gets noisier.
        window_bars = day_df[
            (day_df.index.time >= pd.Timestamp(f"{trade_date.date()} {trade_window_start}").time()) &
            (day_df.index.time <= pd.Timestamp(f"{trade_date.date()} {trade_window_end}").time())
        ]
        if window_bars.empty:
            continue

        # ── Wait for price to touch the midpoint ──────────────────────────
        # Entry is triggered when price retraces back to the zone midpoint.
        # Bullish: price must dip down to the mid (low <= mid).
        # Bearish: price must rally up to the mid (high >= mid).
        for ts, row in window_bars.iterrows():
            if state.trade_taken:
                break

            touched = (
                (state.direction == "bullish" and row["low"]  <= state.zone_mid) or
                (state.direction == "bearish" and row["high"] >= state.zone_mid)
            )
            if not touched:
                continue

            # Entry: midpoint ± slippage (limit order fills at mid, slippage moves it)
            if state.direction == "bullish":
                entry = state.zone_mid + slip_entry   # long: fills slightly above mid
            else:
                entry = state.zone_mid - slip_entry   # short: fills slightly below mid

            # ── SL: nearest qualifying session level ───────────────────────
            # The stop is placed at the nearest Asia or London session extreme
            # that is within [sl_scan_low, sl_scan_high] points of entry.
            # Using a real market structure level as the stop is more robust
            # than a fixed ATR or fixed-point stop.
            sl_price, sl_dist = _find_sl(
                entry, state.direction,
                state.asia_high, state.asia_low,
                state.london_high, state.london_low,
                sl_scan_low, sl_scan_high, sl_default,
            )

            # Target = entry ± (stop distance × R:R ratio)
            tp_dist  = sl_dist * rr_ratio
            tp_price = entry + tp_dist if state.direction == "bullish" else entry - tp_dist

            # ── Simulate exit on subsequent bars ──────────────────────────
            # Allow the trade to run until session close (16:15), not just
            # the end of the trade window — the window only governs entries.
            eod_bars = day_df[
                (day_df.index > ts) &
                (day_df.index.time <= pd.Timestamp(f"{trade_date.date()} 16:15").time())
            ]

            exit_price, exit_time = _simulate_exit(
                eod_bars, state.direction,
                sl_price, tp_price, entry, slip_exit,
            )

            trades.append(create_trade(
                entry_time=ts,
                exit_time=exit_time,
                # Map strategy direction strings to the standard "long"/"short"
                direction="long" if state.direction == "bullish" else "short",
                entry_price=entry,
                exit_price=exit_price,
                stop_price=sl_price,
                target_price=tp_price,
            ))
            state.trade_taken = True

    if not trades:
        return pd.DataFrame()

    result = trades_to_dataframe(trades)
    # Add dollar P&L for reporting — pnl_points × multiplier (e.g. × $5 for MES)
    spec = _get_spec(symbol)
    result["dollar_pnl"] = result["pnl_points"] * spec.multiplier
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_sl(
    entry: float,
    direction: str,
    asia_high: float,
    asia_low: float,
    london_high: float,
    london_low: float,
    scan_low: float,
    scan_high: float,
    default_sl_pts: float,
) -> tuple[float, float]:
    """
    Find the best SL anchor from session levels.
    Returns (sl_price, sl_distance_pts).
    The "best" level is the one farthest from entry within [scan_low, scan_high].

    Why farthest? We want the stop beyond the most significant nearby level
    so that normal price noise doesn't trigger it — only a real structural break.
    """
    is_bull = direction == "bullish"
    best_price = np.nan
    best_dist  = 0.0

    # Check each of the four session levels
    for level in [asia_high, asia_low, london_high, london_low]:
        if np.isnan(level):
            continue
        # Distance from entry to the level (positive = level is below entry for bull)
        dist = (entry - level) if is_bull else (level - entry)
        # Only consider levels within the scan range — too close is dangerous,
        # too far means we're risking more than intended
        if scan_low <= dist <= scan_high and dist > best_dist:
            best_dist  = dist
            best_price = level

    # If no qualifying level was found, use a fixed fallback distance
    if np.isnan(best_price):
        best_dist  = default_sl_pts
        best_price = entry - default_sl_pts if is_bull else entry + default_sl_pts

    return best_price, best_dist


def _simulate_exit(
    bars: pd.DataFrame,
    direction: str,
    sl_price: float,
    tp_price: float,
    entry: float,
    slip_exit: float,
) -> tuple[float, pd.Timestamp]:
    """
    Walk forward through bars and return the first exit hit.

    Same conservative logic as the other strategies:
    - Stop detected using bar's low/high (intrabar)
    - Same-bar ambiguity → stop wins
    - Stop fills with slippage; target fills at limit (no slippage)
    - No hit → exit at last bar's close (session end)
    """
    for ts, row in bars.iterrows():
        if direction == "bullish":
            stop_hit   = row["low"]  <= sl_price
            target_hit = row["high"] >= tp_price
        else:
            stop_hit   = row["high"] >= sl_price
            target_hit = row["low"]  <= tp_price

        if stop_hit and target_hit:
            # Same-bar ambiguity — assume stop fired first (worst case)
            fill = sl_price - slip_exit if direction == "bullish" else sl_price + slip_exit
            return fill, ts
        if stop_hit:
            fill = sl_price - slip_exit if direction == "bullish" else sl_price + slip_exit
            return fill, ts
        if target_hit:
            return tp_price, ts   # Limit order — no slippage

    # Session ended without hitting stop or target
    if bars.empty:
        return entry, pd.Timestamp("NaT")
    last = bars.iloc[-1]
    return last["close"], bars.index[-1]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_and_report(
    symbol: str,
    start: str,
    end: str,
    interval: str = "5m",
    account_size: float = 50_000,
    risk_per_trade: float = 0.01,
    rr_ratio: float = 4.0,
    sl_scan_low: float = 5.0,
    sl_scan_high: float = 10.0,
    sl_default: float = 8.0,
    invalidate_large_range: bool = True,
    entry_slippage_ticks: int = 1,
    exit_slippage_ticks: int = 1,
    n_simulations: int = 500,
    source: str = "yfinance",
    alpaca_key: str = "",
    alpaca_secret: str = "",
    databento_key: str = "",
) -> pd.DataFrame:
    from core.performance import summary_stats
    from core.plots import plot_all
    from core.monte_carlo import run_monte_carlo, plot_monte_carlo_paths
    import matplotlib.pyplot as plt
    import yfinance as yf

    spec = _get_spec(symbol)

    print(f"Fetching {interval} data for {symbol} ({start} → {end}) via {source} …")
    df = fetch_data(symbol, start=start, end=end, interval=interval,
                    source=source, alpaca_key=alpaca_key,
                    alpaca_secret=alpaca_secret, databento_key=databento_key)
    if df.empty:
        print("No data returned.")
        return pd.DataFrame()

    print(f"Running 8AM Key Level ORB  (RR={rr_ratio}, SL scan={sl_scan_low}–{sl_scan_high}pts) …")
    trades = run_key_level_orb(
        df, symbol=symbol,
        rr_ratio=rr_ratio,
        sl_scan_low=sl_scan_low,
        sl_scan_high=sl_scan_high,
        sl_default=sl_default,
        invalidate_large_range=invalidate_large_range,
        entry_slippage_ticks=entry_slippage_ticks,
        exit_slippage_ticks=exit_slippage_ticks,
    )

    if trades.empty:
        print("No trades generated.")
        return trades

    # ── Compounded equity ──────────────────────────────────────────────────
    comp_equity     = np.cumprod(1 + trades["r_multiple"].values * risk_per_trade)
    comp_total_pct  = (comp_equity[-1] - 1) * 100
    comp_roll_max   = np.maximum.accumulate(comp_equity)
    comp_max_dd_pct = ((comp_equity - comp_roll_max) / comp_roll_max * 100).min()

    n_days      = (pd.Timestamp(end) - pd.Timestamp(start)).days
    years       = max(n_days / 365.25, 1 / 365.25)
    comp_ann    = ((comp_equity[-1]) ** (1 / years) - 1) * 100

    # Dollar P&L (1 contract)
    total_pnl = trades["dollar_pnl"].sum()
    avg_pnl   = trades["dollar_pnl"].mean()

    # Buy-and-hold benchmark — uses daily data for simplicity
    bnh_pct = bnh_dd = float("nan")
    bnh_equity = None
    bnh_df = yf.download(spec.yf_ticker, start=start, end=end,
                         interval="1d", auto_adjust=True,
                         progress=False, multi_level_index=False)
    if not bnh_df.empty:
        bnh_df.columns = [c.lower() for c in bnh_df.columns]
        p = bnh_df["close"].dropna()
        bnh_pct    = (p.iloc[-1] / p.iloc[0] - 1) * 100
        bnh_equity = p / p.iloc[0]
        bnh_dd     = ((p - p.cummax()) / p.cummax() * 100).min()
    bnh_ann = ((1 + bnh_pct / 100) ** (1 / years) - 1) * 100

    # ── Print ──────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  {symbol} 8AM Key Level ORB vs Buy-and-Hold  ({start} → {end})")
    print(f"  SL: {sl_scan_low}–{sl_scan_high}pts scan (fallback {sl_default}pts)  |  RR: {rr_ratio}:1")
    print(f"{'='*64}")
    print(f"  {'Metric':<30} {'ORB (compound)':>14} {'Buy & Hold':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Total return (%)':30} {comp_total_pct:>14.2f} {bnh_pct:>12.2f}")
    print(f"  {'Annualised return (%)':30} {comp_ann:>14.2f} {bnh_ann:>12.2f}")
    print(f"  {'Max drawdown (%)':30} {comp_max_dd_pct:>14.2f} {bnh_dd:>12.2f}")
    print(f"  {'Trades':30} {len(trades):>14}")
    print(f"{'='*64}")

    print(f"\n  Dollar P&L (1 contract):")
    print(f"  {'Total P&L':25} ${total_pnl:,.2f}")
    print(f"  {'Avg P&L / trade':25} ${avg_pnl:,.2f}")

    print(f"\n{'='*64}")
    print(f"  Strategy Stats")
    print(f"{'='*64}")
    stats = summary_stats(trades)
    for k, v in stats.items():
        print(f"  {k:<30} {v:>14.4f}" if isinstance(v, float)
              else f"  {k:<30} {v:>14}")

    simulations = run_monte_carlo(trades, n_simulations=n_simulations)

    plot_all(trades)
    plot_monte_carlo_paths(simulations)

    orb_dates = pd.to_datetime(trades["exit_time"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: normalised equity curve vs buy-and-hold
    axes[0].plot(orb_dates, comp_equity, label="8AM ORB compounded", linewidth=1.5)
    if bnh_equity is not None:
        axes[0].plot(bnh_equity.index, bnh_equity.values,
                     label="Buy & Hold", linewidth=1.5, linestyle="--")
    axes[0].axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    axes[0].set_title(f"{symbol} 8AM Key Level ORB vs Buy-and-Hold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Equity multiple")
    axes[0].legend()
    axes[0].grid(True)

    # Right: dollar P&L per trade — green = winner, red = loser
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
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core.config import Config

    # ── Quick demo: yfinance MES (last ~60 days, no API key needed) ───────
    # Strategy-specific params (sl_scan_low/high, sl_default) stay here
    # because they are unique to this strategy — not universal config values.
    trades = run_and_report(
        symbol=Config.SYMBOL,
        start=Config.START_DATE,
        end=Config.END_DATE,
        interval=Config.INTERVAL,
        account_size=Config.ACCOUNT_SIZE,
        risk_per_trade=Config.RISK_PER_TRADE,
        rr_ratio=4.0,          # Key Level ORB targets 4R — strategy-specific default
        sl_scan_low=5.0,       # points; specific to this strategy's SL logic
        sl_scan_high=10.0,
        sl_default=8.0,
        entry_slippage_ticks=Config.ENTRY_SLIP_TICKS,
        exit_slippage_ticks=Config.EXIT_SLIP_TICKS,
        n_simulations=Config.N_SIMULATIONS,
        source=Config.DATA_SOURCE,
    )

    # ── 5-year backtest via Alpaca ─────────────────────────────────────────
    # Set DATA_SOURCE = "alpaca" in Config, then uncomment:
    #
    # trades = run_and_report(
    #     symbol=Config.SYMBOL,
    #     start="2021-01-01",
    #     end="2026-01-01",
    #     interval=Config.INTERVAL,
    #     account_size=Config.ACCOUNT_SIZE,
    #     risk_per_trade=Config.RISK_PER_TRADE,
    #     rr_ratio=4.0,
    #     source="alpaca",
    #     alpaca_key="PASTE_KEY_HERE",
    #     alpaca_secret="PASTE_SECRET_HERE",
    # )

    if not trades.empty:
        print(trades[["entry_time", "exit_time", "direction",
                      "entry_price", "sl_price", "tp_price",
                      "r_multiple", "dollar_pnl"]].to_string())
