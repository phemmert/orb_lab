# src/stop_selector.py
"""
Stop Selector
-------------

Purpose:
    Pine-parity stop selection at entry time.

Given:
    - entry price
    - ATR
    - direction
    - VWAP
    - recent RTH bars (for swing stop)

Compute candidate stops:
    - ATR stop
    - Swing stop (RTH)
    - VWAP stop

Then:
    - enforce minimum stop distance from entry
    - choose stop that yields highest achievable RR (with max target ATR cap)

This module intentionally does NOT:
    - manage break-even or exits
    - calculate entries
    - calculate P&L
"""

from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd


def calc_atr_stop(entry: float, atr: float, atr_mult: float, direction: str) -> float:
    """ATR stop: entry +/- atr*mult"""
    if direction == "LONG":
        return entry - atr * atr_mult
    if direction == "SHORT":
        return entry + atr * atr_mult
    raise ValueError(f"Invalid direction: {direction!r}")


def calc_swing_stop(
    day_df: pd.DataFrame,
    bar_idx: int,
    direction: str,
    atr: float,
    swing_lookback: int = 5,
    swing_buffer: float = 0.02,
) -> float:
    """
    Swing stop using *RTH bars only*.

    LONG  -> lowest low over lookback
    SHORT -> highest high over lookback

    swing_buffer is expressed in ATR units (atr * swing_buffer).
    """

    def is_rth(ts) -> bool:
        mins = ts.hour * 60 + ts.minute
        return 9 * 60 + 30 <= mins < 16 * 60

    swing_val = None
    valid_bars = 0

    # Scan backwards; allow up to 3x lookback to find enough RTH bars
    for j in range(1, swing_lookback * 3 + 1):
        if bar_idx - j < 0:
            break

        ts = day_df.index[bar_idx - j]
        if not is_rth(ts):
            continue

        if direction == "LONG":
            val = float(day_df["low"].iloc[bar_idx - j])
            swing_val = val if swing_val is None else min(swing_val, val)
        elif direction == "SHORT":
            val = float(day_df["high"].iloc[bar_idx - j])
            swing_val = val if swing_val is None else max(swing_val, val)
        else:
            raise ValueError(f"Invalid direction: {direction!r}")

        valid_bars += 1
        if valid_bars >= swing_lookback:
            break

    # Fallback if no RTH history
    if swing_val is None:
        if direction == "LONG":
            swing_val = float(day_df["low"].iloc[max(0, bar_idx - 1)])
        else:
            swing_val = float(day_df["high"].iloc[max(0, bar_idx - 1)])

    buffer = atr * swing_buffer
    return swing_val - buffer if direction == "LONG" else swing_val + buffer


def calc_vwap_stop(
    vwap: float,
    atr: float,
    direction: str,
    vwap_atr_dist: float = 0.3,
) -> float:
    """VWAP stop: vwap +/- atr*dist"""
    dist = atr * vwap_atr_dist
    if direction == "LONG":
        return vwap - dist
    if direction == "SHORT":
        return vwap + dist
    raise ValueError(f"Invalid direction: {direction!r}")


def calc_achievable_rr(
    entry: float,
    stop: float,
    atr: float,
    rr_desired: float,
    max_target_atr: float = 3.0,
) -> float:
    """
    Achievable R:R with a max target cap in ATR units.

    If desired target implies > max_target_atr ATRs, reduce RR accordingly.
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return -1.0

    target_dist = risk * rr_desired
    target_atrs = target_dist / atr

    if target_atrs > max_target_atr:
        return (max_target_atr * atr) / risk

    return rr_desired


def select_best_stop(
    *,
    entry: float,
    atr: float,
    direction: str,
    rr_desired: float,
    day_df: pd.DataFrame,
    bar_idx: int,
    vwap: float,
    atr_mult: float = 2.0,
    swing_lookback: int = 5,
    swing_buffer: float = 0.02,
    vwap_atr_dist: float = 0.3,
    min_stop_atr: float = 0.15,
    max_target_atr: float = 3.0,
) -> Tuple[float, str, float]:
    """
    Compute candidate stops and choose the one with the highest achievable RR.

    Returns:
        (stop_price, stop_type, achievable_rr)
        stop_type in {'ATR','SWING','VWAP'}
    """
    min_dist = atr * min_stop_atr

    atr_stop = calc_atr_stop(entry, atr, atr_mult, direction)
    swing_stop = calc_swing_stop(day_df, bar_idx, direction, atr, swing_lookback, swing_buffer)
    vwap_stop = calc_vwap_stop(vwap, atr, direction, vwap_atr_dist)

    candidates: Dict[str, float] = {
        "ATR": atr_stop,
        "SWING": swing_stop,
        "VWAP": vwap_stop,
    }

    valid: Dict[str, float] = {}
    for name, stop in candidates.items():
        if direction == "LONG":
            if stop < entry - min_dist:
                valid[name] = stop
        elif direction == "SHORT":
            if stop > entry + min_dist:
                valid[name] = stop
        else:
            raise ValueError(f"Invalid direction: {direction!r}")

    # Fallback: ATR always exists
    if not valid:
        rr = calc_achievable_rr(entry, atr_stop, atr, rr_desired, max_target_atr)
        return atr_stop, "ATR", rr

    ranked = []
    for name, stop in valid.items():
        rr = calc_achievable_rr(entry, stop, atr, rr_desired, max_target_atr)
        if rr > 0:
            ranked.append((rr, name, stop))

    if not ranked:
        rr = calc_achievable_rr(entry, atr_stop, atr, rr_desired, max_target_atr)
        return atr_stop, "ATR", rr

    rr, name, stop = max(ranked, key=lambda x: x[0])
    return stop, name, rr
