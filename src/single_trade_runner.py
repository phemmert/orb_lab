# single_trade_runner.py
"""
Single Trade Runner
-------------------

Purpose:
    Execute ONE trade from entry to exit, bar-by-bar, using
    Pine-parity stop logic.

Scope:
    - Assumes entry signal already exists
    - Uses TradeStopManager to manage stops
    - Walks forward until exit

This module intentionally does NOT:
    - search for entries
    - manage multiple trades
    - calculate P&L
    - loop over symbols or days

This is a truth-validation layer, not a backtester.
"""

from __future__ import annotations

from typing import Dict, Optional
import pandas as pd

from stop_engine import StopState
from trade_stop_manager import TradeStopManager


def run_single_trade(
    *,
    df: pd.DataFrame,
    entry_bar_idx: int,
    direction: str,
    atr: float,
    rr_desired: float,
    break_even_rr: float = 0.5,
) -> Dict[str, Optional[object]]:
    """Run a single trade forward until exit."""

    entry_ts = df.index[entry_bar_idx]
    entry_price = float(df["close"].iloc[entry_bar_idx])
    vwap = float(df["vwap"].iloc[entry_bar_idx])

    manager = TradeStopManager(stop_state=StopState())

    # Initialize trade
    manager.enter_trade(
        direction=direction,
        entry_price=entry_price,
        atr=atr,
        rr_desired=rr_desired,
        day_df=df,
        bar_idx=entry_bar_idx,
        vwap=vwap,
    )

    # Walk forward bar-by-bar
    for i in range(entry_bar_idx + 1, len(df)):
        bar = df.iloc[i]

        should_exit, reason, exit_price = manager.process_bar(
            bar_high=float(bar["high"]),
            bar_low=float(bar["low"]),
            break_even_rr=break_even_rr,
        )

        if should_exit:
                        return {
                "entry_time": entry_ts,
                "entry_price": entry_price,
                "exit_time": df.index[i],
                "exit_price": exit_price,
                "exit_reason": reason,
                "stop_type": manager.stop_type,
                "achievable_rr": manager.achievable_rr,
            }

    # Trade still open at end of data
    return {
        "entry_time": entry_ts,
        "entry_price": entry_price,
        "exit_time": None,
        "exit_price": None,
        "exit_reason": None,
        "stop_type": manager.stop_type,
        "achievable_rr": manager.achievable_rr,
    }
