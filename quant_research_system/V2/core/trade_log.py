# =============================================================================
# V2 Core — Trade Log
# =============================================================================
# Defines the canonical Trade object and the functions to create, store, and
# export trades. Every strategy in V2 outputs a list of Trade objects, which
# get converted into a DataFrame for analysis.
#
# The key design principle here is R-normalisation: all P&L is expressed as
# a multiple of the initial risk (1R = 1× your risk amount). This makes
# strategies directly comparable regardless of dollar size or instrument.
#
#   +2R = won twice your risk
#   -1R = lost your full risk (hit stop)
#   +0.5R = partial win (exited before target)
# =============================================================================

import uuid
import math
from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd


# ── Trade schema ──────────────────────────────────────────────────────────────
# A frozen snapshot of everything about a single trade.
# Using a dataclass gives us __repr__, type hints, and easy dict conversion
# for free, without writing boilerplate.

@dataclass
class Trade:
    trade_id:     str            # unique identifier (UUID4 by default)
    entry_time:   pd.Timestamp   # when the trade was entered
    exit_time:    pd.Timestamp   # when the trade was closed (NaT if session-close fallback)
    duration:     pd.Timedelta   # how long the trade was open
    direction:    str            # "long" or "short"
    entry_price:  float          # fill price on entry (includes slippage)
    exit_price:   float          # fill price on exit (includes stop slippage if stopped out)
    stop_price:   float          # original stop level (before slippage)
    target_price: float          # profit target level
    risk_per_unit: float         # |entry - stop| in price units = 1R in points
    pnl_points:   float          # raw P&L in price points (positive = profit)
    r_multiple:   float          # pnl_points / risk_per_unit  (the key metric)
    win:          int            # 1 if trade was profitable, 0 if not


def create_trade(
    entry_time:   pd.Timestamp,
    exit_time:    pd.Timestamp,
    direction:    str,
    entry_price:  float,
    exit_price:   float,
    stop_price:   float,
    target_price: float,
    trade_id:     Optional[str] = None,
) -> Trade:
    """
    Construct a Trade object from raw entry/exit data, computing all derived
    fields automatically (risk, P&L, R-multiple, win flag, duration).

    Raises ValueError if any price is NaN/None, if direction is invalid,
    or if entry == stop (which would make R-multiple undefined).
    """
    # ── Validation ────────────────────────────────────────────────────────
    if direction not in ["long", "short"]:
        raise ValueError("direction must be 'long' or 'short'")

    # Check every price input for NaN — a NaN price would silently corrupt
    # the R-multiple calculation and produce meaningless results downstream
    for name, val in [
        ("entry_price",  entry_price),
        ("exit_price",   exit_price),
        ("stop_price",   stop_price),
        ("target_price", target_price),
    ]:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            raise ValueError(f"{name} cannot be NaN or None")

    # Risk is the distance from entry to stop — this is the denominator of R
    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit == 0:
        raise ValueError("risk_per_unit cannot be zero (entry_price == stop_price)")

    # ── P&L calculation ───────────────────────────────────────────────────
    # For a long: profit when exit > entry, loss when exit < entry
    # For a short: profit when exit < entry (price fell), loss when exit > entry
    if direction == "long":
        pnl_points = exit_price - entry_price
    else:
        pnl_points = entry_price - exit_price

    # R-multiple: how many times our initial risk did we gain or lose?
    # e.g. if risk = 2pts and we made 4pts → R = +2.0 (a 2R winner)
    r_multiple = pnl_points / risk_per_unit

    # Final sanity check — should not be possible with valid inputs, but guards
    # against edge cases like infinite prices
    if math.isnan(r_multiple):
        raise ValueError("r_multiple resolved to NaN — check entry/exit/stop prices")

    # ── Duration ──────────────────────────────────────────────────────────
    # How long was the trade open? NaT if exit_time is unknown (session close).
    duration = (
        (exit_time - entry_time)
        if pd.notna(exit_time) and pd.notna(entry_time)
        else pd.NaT
    )

    return Trade(
        trade_id=trade_id or str(uuid.uuid4()),
        entry_time=entry_time,
        exit_time=exit_time,
        duration=duration,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_per_unit=risk_per_unit,
        pnl_points=pnl_points,
        r_multiple=r_multiple,
        win=int(pnl_points > 0),
    )


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """
    Convert a list of Trade objects into a pandas DataFrame.

    Each row is one trade; columns match the Trade dataclass fields.
    This DataFrame is the standard input for performance.py, plots.py,
    and monte_carlo.py.
    """
    return pd.DataFrame([asdict(trade) for trade in trades])


def export_trades_csv(trades: pd.DataFrame, path: str) -> None:
    """
    Save the trades DataFrame to a CSV file.

    The CSV can be loaded back later, shared, or imported into Excel.
    The dashboard (Phase 3 / V4) will read from files like this.
    """
    trades.to_csv(path, index=False)


def export_trades_json(trades: pd.DataFrame, path: str) -> None:
    """
    Save the trades DataFrame to a JSON file (array of trade objects).

    Uses ISO 8601 timestamps and 2-space indentation for readability.
    Useful for the dashboard and the prop firm simulator (V3).
    """
    trades.to_json(path, orient="records", indent=2, date_format="iso")
