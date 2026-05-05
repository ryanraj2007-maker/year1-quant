# V2 Core — Trade Log
# Defines the Trade object and helpers for creating, storing, and exporting trades.
# All P&L is expressed as R-multiples so strategies are comparable regardless of size.

import uuid
import math
from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd


@dataclass
class Trade:
    trade_id:     str
    entry_time:   pd.Timestamp
    exit_time:    pd.Timestamp
    duration:     pd.Timedelta
    direction:    str
    entry_price:  float
    exit_price:   float
    stop_price:   float
    target_price: float
    risk_per_unit: float
    pnl_points:   float
    r_multiple:   float   # pnl / risk — the key metric
    win:          int     # 1 = profitable, 0 = loss


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
    Build a Trade from raw entry/exit data. Computes risk, P&L, R-multiple,
    win flag, and duration automatically.
    """
    if direction not in ["long", "short"]:
        raise ValueError("direction must be 'long' or 'short'")

    for name, val in [
        ("entry_price",  entry_price),
        ("exit_price",   exit_price),
        ("stop_price",   stop_price),
        ("target_price", target_price),
    ]:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            raise ValueError(f"{name} cannot be NaN or None")

    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit == 0:
        raise ValueError("risk_per_unit cannot be zero (entry_price == stop_price)")

    if direction == "long":
        pnl_points = exit_price - entry_price
    else:
        pnl_points = entry_price - exit_price

    r_multiple = pnl_points / risk_per_unit

    if math.isnan(r_multiple):
        raise ValueError("r_multiple resolved to NaN — check entry/exit/stop prices")

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
    """Convert a list of Trade objects to a DataFrame. Standard input for performance.py."""
    return pd.DataFrame([asdict(trade) for trade in trades])


def export_trades_csv(trades: pd.DataFrame, path: str) -> None:
    """Save trades to CSV."""
    trades.to_csv(path, index=False)


def export_trades_json(trades: pd.DataFrame, path: str) -> None:
    """Save trades to JSON."""
    trades.to_json(path, orient="records", indent=2, date_format="iso")
