# src/trade_stop_manager.py
"""
Trade Stop Manager
------------------

Purpose:
    Glue layer that wires together:
      - stop selection (ATR / Swing / VWAP)
      - stop engine (auto-stop + break-even + exit detection)

Scope:
    - Assumes ENTRY has already occurred
    - Manages stops from entry until exit
    - Pine-parity ordering: Break-even checked BEFORE stop-hit

Dependencies:
    - stop_selector.py
    - stop_engine.py

This module intentionally does NOT:
    - calculate entries
    - calculate P&L
    - manage multiple positions
    - handle commissions or slippage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd

from stop_selector import select_best_stop
from stop_engine import StopState, enter_with_initial_stop, process_bar_for_exit


@dataclass
class TradeStopManager:
    """Manages stop lifecycle for a single trade."""

    stop_state: StopState

    stop_type: Optional[str] = None
    achievable_rr: Optional[float] = None

    def enter_trade(
        self,
        *,
        direction: str,
        entry_price: float,
        atr: float,
        rr_desired: float,
        day_df: pd.DataFrame,
        bar_idx: int,
        vwap: float,
        atr_mult: float = 2.0,
    ) -> None:
        """Called exactly once at trade entry."""

        stop_price, stop_type, rr = select_best_stop(
            entry=entry_price,
            atr=atr,
            direction=direction,
            rr_desired=rr_desired,
            day_df=day_df,
            bar_idx=bar_idx,
            vwap=vwap,
            atr_mult=atr_mult,
        )

        self.stop_type = stop_type
        self.achievable_rr = rr

        enter_with_initial_stop(
            self.stop_state,
            direction=direction,
            entry_price=entry_price,
            initial_stop=stop_price,
        )

    def process_bar(
        self,
        *,
        bar_high: float,
        bar_low: float,
        break_even_rr: float = 0.5,
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """Process a single bar AFTER entry."""
        return process_bar_for_exit(
            self.stop_state,
            bar_high=bar_high,
            bar_low=bar_low,
            break_even_rr=break_even_rr,
        )

    def reset(self) -> None:
        """Reset manager for next trade."""
        self.stop_state.reset()
        self.stop_type = None
        self.achievable_rr = None
