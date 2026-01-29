# src/stop_engine.py
"""
Stop Engine
-----------

Purpose:
    Pine-parity auto-stop + break-even logic.

Scope:
    - Manages stop state AFTER entry
    - Handles:
        * initial stop
        * break-even trigger
        * stop-hit detection
    - Enforces CRITICAL ORDER:
        1) check break-even FIRST
        2) then check stop-hit

This ordering matches Pine v73+ behavior and avoids the classic
"BE should have triggered but full stop was taken" bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StopState:
    """
    Minimal stop state to emulate Pine behavior.
    """
    in_position: bool = False
    direction: Optional[str] = None  # 'LONG' or 'SHORT'
    entry_price: Optional[float] = None

    initial_stop: Optional[float] = None
    current_stop: Optional[float] = None
    stop_moved_to_be: bool = False

    def reset(self) -> None:
        self.in_position = False
        self.direction = None
        self.entry_price = None
        self.initial_stop = None
        self.current_stop = None
        self.stop_moved_to_be = False


# -----------------------------------------------------------------------------
# Entry helpers
# -----------------------------------------------------------------------------

def calc_initial_stop_atr(entry_price: float, atr: float, atr_mult: float, direction: str) -> float:
    """
    Pine equivalent:
        atrStop = isLong ? entry - atr*mult : entry + atr*mult
    """
    if direction == "LONG":
        return entry_price - atr * atr_mult
    if direction == "SHORT":
        return entry_price + atr * atr_mult
    raise ValueError(f"direction must be 'LONG' or 'SHORT', got {direction!r}")


def enter_with_initial_stop(
    st: StopState,
    *,
    direction: str,
    entry_price: float,
    initial_stop: float,
) -> None:
    """
    Initialize stop state at trade entry.
    """
    st.in_position = True
    st.direction = direction
    st.entry_price = float(entry_price)
    st.initial_stop = float(initial_stop)
    st.current_stop = float(initial_stop)
    st.stop_moved_to_be = False


# -----------------------------------------------------------------------------
# Intrabar logic (Pine-parity)
# -----------------------------------------------------------------------------

def _check_break_even_intrabar(
    st: StopState,
    bar_high: float,
    bar_low: float,
    break_even_rr: float,
) -> bool:
    """
    Check if break-even should trigger on this bar.

    - Uses INITIAL risk (entry - initial_stop)
    - Uses bar HIGH / LOW for intrabar detection
    - Latches stop at entry when triggered
    """
    if (not st.in_position) or st.stop_moved_to_be:
        return False

    if st.entry_price is None or st.initial_stop is None:
        return False

    risk = abs(st.entry_price - st.initial_stop)
    if risk <= 0:
        return False

    if st.direction == "LONG":
        be_target = st.entry_price + (risk * break_even_rr)
        if bar_high >= be_target:
            st.stop_moved_to_be = True
            st.current_stop = st.entry_price
            return True

    elif st.direction == "SHORT":
        be_target = st.entry_price - (risk * break_even_rr)
        if bar_low <= be_target:
            st.stop_moved_to_be = True
            st.current_stop = st.entry_price
            return True

    else:
        raise ValueError(f"direction must be 'LONG' or 'SHORT', got {st.direction!r}")

    return False


def _check_stop_hit_intrabar(
    st: StopState,
    bar_high: float,
    bar_low: float,
) -> bool:
    """
    Intrabar stop-hit detection.

    LONG  -> stop hit if bar_low  <= current_stop
    SHORT -> stop hit if bar_high >= current_stop
    """
    if not st.in_position:
        return False
    if st.current_stop is None:
        return False

    if st.direction == "LONG":
        return bar_low <= st.current_stop
    if st.direction == "SHORT":
        return bar_high >= st.current_stop

    raise ValueError(f"direction must be 'LONG' or 'SHORT', got {st.direction!r}")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def process_bar_for_exit(
    st: StopState,
    *,
    bar_high: float,
    bar_low: float,
    break_even_rr: float = 0.5,
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Process a single bar AFTER entry.

    CRITICAL ORDER:
        1) check break-even FIRST
        2) then check stop-hit

    Returns:
        (should_exit, exit_reason, exit_price)
    """
    if not st.in_position:
        return False, None, None

    _check_break_even_intrabar(st, bar_high, bar_low, break_even_rr)

    if _check_stop_hit_intrabar(st, bar_high, bar_low):
        exit_price = float(st.current_stop)
        if st.stop_moved_to_be:
            return True, "BREAK-EVEN", exit_price
        return True, "STOP-LOSS", exit_price

    return False, None, None
