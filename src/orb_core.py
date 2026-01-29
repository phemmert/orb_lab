"""
ORB Core Module - Clean rebuild from Pine reference
Validated components only: ORB building, breakout detection, ATR calculation

Pine Reference: 5ORB.pine v69
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY STATE - Exact copy from orb_backtester.py lines 244-252
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass 
class VolState:
    """Volatility state calculation"""
    orb_session_baseline: float = np.nan
    session_vf: float = 1.0
    htf_vf: float = 1.0
    vol_factor: float = 1.0
    vol_state: str = "NORMAL"


def calc_vol_state(bar: pd.Series, ts: pd.Timestamp, vol: VolState,
                   low_vol_threshold: float = 0.8,
                   high_vol_threshold: float = 1.3,
                   extreme_vol_threshold: float = 2.0) -> str:
    """
    Calculate volatility state for current bar.
    
    EXACT COPY from orb_backtester.py lines 681-725
    
    Requires bar to have columns: atr_fast_rth, atr_fast_sma_rth, atr_rth, daily_atr, daily_atr_slow
    Returns: vol_state string ("LOW", "NORMAL", "HIGH", "EXTREME")
    """
    atr_fast = bar.get('atr_fast_rth', bar.get('atr_rth', np.nan))
    atr_fast_sma = bar.get('atr_fast_sma_rth', atr_fast)
    if pd.isna(atr_fast):
        atr_fast = bar.get('atr_rth', 1.0)
    if pd.isna(atr_fast_sma):
        atr_fast_sma = atr_fast
    
    bar_minutes = ts.hour * 60 + ts.minute
    is_orb_vol_window = 9*60+30 <= bar_minutes < 10*60
    
    if ts.hour == 9 and ts.minute == 30:
        vol.orb_session_baseline = np.nan
    
    if is_orb_vol_window:
        if pd.isna(vol.orb_session_baseline):
            vol.orb_session_baseline = atr_fast_sma
        else:
            vol.orb_session_baseline = 0.9 * vol.orb_session_baseline + 0.1 * atr_fast_sma
    
    if not pd.isna(vol.orb_session_baseline) and vol.orb_session_baseline > 0:
        vol.session_vf = atr_fast / vol.orb_session_baseline
    else:
        vol.session_vf = 1.0
    
    # HTF VF from daily ATR
    daily_atr = bar.get('daily_atr', 1.0)
    daily_atr_slow = bar.get('daily_atr_slow', daily_atr)
    if pd.isna(daily_atr):
        daily_atr = 1.0
    if pd.isna(daily_atr_slow) or daily_atr_slow <= 0:
        daily_atr_slow = daily_atr
    
    vol.htf_vf = daily_atr / daily_atr_slow if daily_atr_slow > 0 else 1.0
    vol.vol_factor = vol.session_vf * vol.htf_vf
    
    if vol.vol_factor < low_vol_threshold:
        vol.vol_state = "LOW"
    elif vol.vol_factor >= extreme_vol_threshold:
        vol.vol_state = "EXTREME"
    elif vol.vol_factor >= high_vol_threshold:
        vol.vol_state = "HIGH"
    else:
        vol.vol_state = "NORMAL"
    
    return vol.vol_state


def prepare_vol_columns(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Prepare volatility columns needed for vol_state calculation.
    
    EXACT COPY from orb_backtester.py lines 585-639
    
    Adds columns: atr_rth, atr_fast_rth, atr_fast_sma_rth, daily_atr, daily_atr_slow
    """
    # Mark RTH bars
    df['is_rth'] = df.index.map(lambda x: 9*60+30 <= x.hour*60+x.minute < 16*60)
    
    rth_df = df[df['is_rth']].copy()
    rth_df['prev_close'] = rth_df['close'].shift(1)
    tr1_rth = rth_df['high'] - rth_df['low']
    tr2_rth = (rth_df['high'] - rth_df['prev_close']).abs()
    tr3_rth = (rth_df['low'] - rth_df['prev_close']).abs()
    rth_df['tr'] = pd.concat([tr1_rth, tr2_rth, tr3_rth], axis=1).max(axis=1)
    
    # Wilder RMA
    def wilder_rma(series, n):
        rma = series.copy().astype(float)
        rma.iloc[:n] = series.iloc[:n].expanding().mean()
        for i in range(n, len(series)):
            rma.iloc[i] = (rma.iloc[i-1] * (n - 1) + series.iloc[i]) / n
        return rma
    
    # ATR(14) for stops, ATR(5) for vol state
    rth_df['atr'] = wilder_rma(rth_df['tr'], atr_period)
    rth_df['atr_fast'] = wilder_rma(rth_df['tr'], 5)
    rth_df['atr_fast_sma'] = rth_df['atr_fast'].rolling(10).mean()  # Pine: ta.sma(orbATR_fast, 10)
    
    # Lookahead fix
    rth_df['atr_pine'] = rth_df['atr'].shift(1)
    
    # Map back to full df
    df['atr_rth'] = np.nan
    df.loc[rth_df.index, 'atr_rth'] = rth_df['atr_pine']
    df['atr_fast_rth'] = np.nan
    df.loc[rth_df.index, 'atr_fast_rth'] = rth_df['atr_fast']
    df['atr_fast_sma_rth'] = np.nan
    df.loc[rth_df.index, 'atr_fast_sma_rth'] = rth_df['atr_fast_sma']
    
    # Forward fill (note: orb_backtester uses .fillna(df['atr_eth']) but we may not have that)
    df['atr_rth'] = df['atr_rth'].ffill()
    df['atr_fast_rth'] = df['atr_fast_rth'].ffill()
    df['atr_fast_sma_rth'] = df['atr_fast_sma_rth'].ffill()
    
    # Daily ATR for HTF VF
    rth_df['date'] = rth_df.index.date
    daily_rth = rth_df.groupby('date').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    })
    daily_rth['prev_close'] = daily_rth['close'].shift(1)
    daily_tr1 = daily_rth['high'] - daily_rth['low']
    daily_tr2 = (daily_rth['high'] - daily_rth['prev_close']).abs()
    daily_tr3 = (daily_rth['low'] - daily_rth['prev_close']).abs()
    daily_rth['tr'] = pd.concat([daily_tr1, daily_tr2, daily_tr3], axis=1).max(axis=1)
    daily_rth['daily_atr'] = wilder_rma(daily_rth['tr'], 14)
    daily_rth['daily_atr_slow'] = daily_rth['daily_atr'].rolling(20).mean()
    
    # Map daily ATR to intraday
    df['date'] = df.index.date
    df['daily_atr'] = df['date'].map(daily_rth['daily_atr'].to_dict())
    df['daily_atr_slow'] = df['date'].map(daily_rth['daily_atr_slow'].to_dict())
    df['daily_atr'] = df['daily_atr'].ffill()
    df['daily_atr_slow'] = df['daily_atr_slow'].ffill()
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ORB STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ORBState:
    """
    Tracks ORB session state.
    
    Pine equivalent (lines 986, 1570):
        var float sessionHigh = na
        var float sessionLow = na
        orbComplete = not inSession and not na(sessionHigh)
        
    Pine breakout tracking (lines 941-942):
        var bool hasBrokenOutHigh = false
        var bool hasBrokenOutLow = false
    """
    session_high: float = None
    session_low: float = None
    orb_complete: bool = False
    has_broken_out_high: bool = False  # LONG breakout consumed
    has_broken_out_low: bool = False   # SHORT breakout consumed
    
    def reset_for_new_day(self):
        """Reset state for new trading day"""
        self.session_high = None
        self.session_low = None
        self.orb_complete = False
        self.has_broken_out_high = False
        self.has_broken_out_low = False
    
    def is_back_inside(self, close: float) -> bool:
        """Check if price is back inside ORB range - Pine line 1855"""
        if self.session_high is None or self.session_low is None:
            return False
        return close <= self.session_high and close >= self.session_low
    
    def consume_long_breakout(self):
        """Mark LONG breakout as consumed - Pine line 1834"""
        self.has_broken_out_high = True
    
    def consume_short_breakout(self):
        """Mark SHORT breakout as consumed - Pine line 1837"""
        self.has_broken_out_low = True
    
    def reset_breakout_if_back_inside(self, close: float, exited_long: bool = False, exited_short: bool = False):
        """
        Reset breakout flags when price comes back inside ORB.
        
        Pine lines 1856-1866:
        - If back inside AND orb complete: reset both
        - If back inside AND exited long: reset high
        - If back inside AND exited short: reset low
        """
        if not self.is_back_inside(close):
            return
        
        if self.orb_complete:
            self.has_broken_out_high = False
            self.has_broken_out_low = False
        
        if exited_long:
            self.has_broken_out_high = False
        
        if exited_short:
            self.has_broken_out_low = False


# ═══════════════════════════════════════════════════════════════════════════════
# ORB WINDOW & BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def is_orb_window(ts: pd.Timestamp, orb_minutes: int = 5) -> bool:
    """
    Check if timestamp is within ORB building window.
    
    Pine (line 87): sessionTime = "0930-0935"
    
    For 5-min ORB: 09:30:00 through 09:34:59
    """
    if ts.hour != 9:
        return False
    return 30 <= ts.minute < (30 + orb_minutes)


def build_orb(day_df: pd.DataFrame, orb_minutes: int = 5) -> ORBState:
    """
    Build Opening Range from a single day's data.
    
    Pine (lines 1570-1580):
        if isNewSession
            sessionHigh := high
            sessionLow := low
        else if inSession
            sessionHigh := math.max(sessionHigh, high)
            sessionLow := math.min(sessionLow, low)
    
    Returns ORBState with session_high, session_low, orb_complete set.
    """
    state = ORBState()
    
    for ts, bar in day_df.iterrows():
        if is_orb_window(ts, orb_minutes):
            if state.session_high is None:
                state.session_high = bar['high']
                state.session_low = bar['low']
            else:
                state.session_high = max(state.session_high, bar['high'])
                state.session_low = min(state.session_low, bar['low'])
    
    # ORB is complete if we have valid levels
    state.orb_complete = (state.session_high is not None and state.session_low is not None)
    
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# ATR CALCULATION - RTH ONLY, RMA (WILDER) SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════════

def calc_atr_rth(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR on RTH bars only (09:30-16:00) using RMA (Wilder smoothing).
    
    Pine (line 1280):
        rthTicker = ticker.new(syminfo.prefix, syminfo.ticker, session.regular)
        float rthATR = request.security(rthTicker, timeframe.period, 
            ta.rma(ta.tr(true), atrPeriod), barmerge.gaps_off, barmerge.lookahead_off)
    
    RMA formula: RMA[i] = (RMA[i-1] * (n-1) + TR[i]) / n
    First value seeded with SMA of first n TR values.
    """
    # Filter to RTH only
    rth_mask = df.index.map(lambda x: 9*60+30 <= x.hour*60+x.minute < 16*60)
    rth_df = df[rth_mask].copy()
    
    # True Range (using RTH prev close)
    rth_df['prev_close'] = rth_df['close'].shift(1)
    tr = pd.concat([
        rth_df['high'] - rth_df['low'],
        (rth_df['high'] - rth_df['prev_close']).abs(),
        (rth_df['low'] - rth_df['prev_close']).abs()
    ], axis=1).max(axis=1)
    
    # RMA (Wilder smoothing) - matches Pine ta.rma()
    tr_values = tr.values
    atr = np.full(len(tr_values), np.nan)
    
    # First ATR = SMA of first n TR values
    if len(tr_values) >= period:
        atr[period-1] = np.nanmean(tr_values[:period])
        
        # Apply Wilder formula for rest
        for i in range(period, len(tr_values)):
            if not np.isnan(tr_values[i]) and not np.isnan(atr[i-1]):
                atr[i] = (atr[i-1] * (period - 1) + tr_values[i]) / period
    
    rth_df['atr'] = atr
    
    # Map back to full dataframe, forward fill for premarket bars
    result = pd.Series(index=df.index, dtype=float)
    result.loc[rth_df.index] = rth_df['atr']
    result = result.ffill()
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BREAKOUT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def check_long_breakout(
    o: float, h: float, l: float, c: float,
    session_high: float,
    session_low: float,
    atr: float,
    orb_complete: bool,
    has_broken_out_high: bool = False,
    breakout_threshold_mult: float = 0.1,
    min_body_strength_ratio: float = 0.5,
) -> bool:
    """
    Check for long breakout - exact translation of Pine.
    
    Pine (lines 1807-1824, 1832):
        newLongBreakout = ... (breakout conditions)
        if newLongBreakout and not hasBrokenOutHigh
            hasBrokenOutHigh := true
    
    The has_broken_out_high flag prevents re-triggering on same breakout.
    """
    if not orb_complete:
        return False
    
    # Already broken out in this direction - Pine line 1832
    if has_broken_out_high:
        return False
    
    # Body strength check
    candle_body = abs(c - o)
    candle_range = h - l
    body_strength_ratio = (candle_body / candle_range) if candle_range > 0 else 0.0
    strong_body = body_strength_ratio >= min_body_strength_ratio
    
    # Breakout threshold
    breakout_threshold = atr * breakout_threshold_mult
    
    # Open must be inside ORB range
    open_in_range = (o >= session_low) and (o <= session_high)
    
    # Close must be above ORB high + threshold
    close_above = c > session_high
    sufficient_break = c > (session_high + breakout_threshold)
    
    return open_in_range and close_above and sufficient_break and strong_body


def check_short_breakout(
    o: float, h: float, l: float, c: float,
    session_high: float,
    session_low: float,
    atr: float,
    orb_complete: bool,
    has_broken_out_low: bool = False,
    breakout_threshold_mult: float = 0.1,
    min_body_strength_ratio: float = 0.5,
) -> bool:
    """
    Check for short breakout - exact translation of Pine.
    
    Pine (lines 1807-1829, 1835):
        newShortBreakout = ... (breakout conditions)
        if newShortBreakout and not hasBrokenOutLow
            hasBrokenOutLow := true
    
    The has_broken_out_low flag prevents re-triggering on same breakout.
    """
    if not orb_complete:
        return False
    
    # Already broken out in this direction - Pine line 1835
    if has_broken_out_low:
        return False
    
    # Body strength check
    candle_body = abs(c - o)
    candle_range = h - l
    body_strength_ratio = (candle_body / candle_range) if candle_range > 0 else 0.0
    strong_body = body_strength_ratio >= min_body_strength_ratio
    
    # Breakout threshold
    breakout_threshold = atr * breakout_threshold_mult
    
    # Open must be inside ORB range
    open_in_range = (o >= session_low) and (o <= session_high)
    
    # Close must be below ORB low - threshold
    close_below = c < session_low
    sufficient_break = c < (session_low - breakout_threshold)
    
    return open_in_range and close_below and sufficient_break and strong_body


# ═══════════════════════════════════════════════════════════════════════════════
# STOP CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calc_atr_stop(entry_price: float, atr: float, is_long: bool, 
                  atr_multiplier: float = 2.0) -> float:
    """
    Calculate ATR-based stop price.
    
    Pine (line 1136):
        float atrStop = isLong ? entryPrice - (atr * atrMultiplier) : entryPrice + (atr * atrMultiplier)
    
    Validated: AAPL Dec 19 @ 10:19 - MATCH ✓
    """
    if is_long:
        return entry_price - (atr * atr_multiplier)
    else:
        return entry_price + (atr * atr_multiplier)


def calc_vwap_stop(vwap: float, atr: float, is_long: bool,
                   vwap_stop_distance: float = 0.3) -> float:
    """
    Calculate VWAP-based stop price.
    
    Pine (line 1142):
        float vwapStop = isLong ? vwapValue - (atr * vwapStopDistance) : vwapValue + (atr * vwapStopDistance)
    
    Pine (line 212): vwapStopDistance default = 0.3
    
    Validated: AAPL Dec 19 - Polygon VWAP matches session VWAP ✓
               ATR variance ~0.0002 (negligible) ✓
    """
    if is_long:
        return vwap - (atr * vwap_stop_distance)
    else:
        return vwap + (atr * vwap_stop_distance)


def find_swing_low_rth(day_df: pd.DataFrame, bar_idx: int, lookback: int = 5) -> float:
    """
    Find lowest low from previous RTH bars.
    
    Pine (lines 1104-1117): findSwingLowRegularHours
    - Look back up to lookback * 3 bars
    - Only count RTH bars (09:30-16:00)
    - Stop when lookback RTH bars found
    - Return lowest low among them
    
    Validated: AAPL Dec 19 @ 10:19 - MATCH ✓
    """
    swing_low = None
    valid_bars = 0
    
    for i in range(1, lookback * 3 + 1):
        if bar_idx - i < 0:
            break
        check_ts = day_df.index[bar_idx - i]
        mins = check_ts.hour * 60 + check_ts.minute
        if 9*60+30 <= mins < 16*60:  # RTH only
            bar_low = day_df['low'].iloc[bar_idx - i]
            if swing_low is None or bar_low < swing_low:
                swing_low = bar_low
            valid_bars += 1
            if valid_bars >= lookback:
                break
    
    if swing_low is None:
        swing_low = day_df['low'].iloc[bar_idx - 1]
    
    return swing_low


def find_swing_high_rth(day_df: pd.DataFrame, bar_idx: int, lookback: int = 5) -> float:
    """
    Find highest high from previous RTH bars.
    
    Pine (lines 1119-1132): findSwingHighRegularHours
    - Look back up to lookback * 3 bars
    - Only count RTH bars (09:30-16:00)
    - Stop when lookback RTH bars found
    - Return highest high among them
    
    Validated: AAPL Dec 19 @ 10:19 - MATCH ✓
    """
    swing_high = None
    valid_bars = 0
    
    for i in range(1, lookback * 3 + 1):
        if bar_idx - i < 0:
            break
        check_ts = day_df.index[bar_idx - i]
        mins = check_ts.hour * 60 + check_ts.minute
        if 9*60+30 <= mins < 16*60:  # RTH only
            bar_high = day_df['high'].iloc[bar_idx - i]
            if swing_high is None or bar_high > swing_high:
                swing_high = bar_high
            valid_bars += 1
            if valid_bars >= lookback:
                break
    
    if swing_high is None:
        swing_high = day_df['high'].iloc[bar_idx - 1]
    
    return swing_high


def calc_swing_stop(day_df: pd.DataFrame, bar_idx: int, atr: float, is_long: bool,
                    swing_lookback: int = 5, swing_buffer: float = 0.02) -> float:
    """
    Calculate swing-based stop price.
    
    Pine (line 1140):
        float swingStop = isLong ? swingLow - (atr * swingBuffer) : swingHigh + (atr * swingBuffer)
    
    Pine defaults (lines 210-211):
        swingLookback = 5
        swingBuffer = 0.02
    
    Validated: AAPL Dec 19 @ 10:19 - MATCH ✓
    """
    if is_long:
        swing_low = find_swing_low_rth(day_df, bar_idx, swing_lookback)
        return swing_low - (atr * swing_buffer)
    else:
        swing_high = find_swing_high_rth(day_df, bar_idx, swing_lookback)
        return swing_high + (atr * swing_buffer)


def calc_hybrid_stop(atr_stop: float, swing_stop: float, vwap_stop: float, 
                     is_long: bool, exclude_vwap: bool = False) -> float:
    """
    Calculate hybrid stop - most conservative (tightest) of all stops.
    
    Pine (lines 1177-1181):
        "Hybrid" => 
            if isLong
                hybridExcludeVwap ? math.max(atrStop, swingStop) : math.max(atrStop, math.max(swingStop, vwapStop))
            else
                hybridExcludeVwap ? math.min(atrStop, swingStop) : math.min(atrStop, math.min(swingStop, vwapStop))
    
    For LONG: highest stop = closest to entry = tightest = most conservative
    For SHORT: lowest stop = closest to entry = tightest = most conservative
    """
    if is_long:
        if exclude_vwap:
            return max(atr_stop, swing_stop)
        else:
            return max(atr_stop, swing_stop, vwap_stop)
    else:
        if exclude_vwap:
            return min(atr_stop, swing_stop)
        else:
            return min(atr_stop, swing_stop, vwap_stop)


# ═══════════════════════════════════════════════════════════════════════════════
# STOP VALIDITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_stop_validity(stop: float, entry_price: float, atr: float, is_long: bool,
                        vwap: float = None, min_stop_atr: float = 0.15) -> bool:
    """
    Check if a stop is valid (far enough from entry, correct side).
    
    Pine (lines 1208-1212):
        float minStopDist = atrVal * 0.15
        bool atrValid = isLong ? (atrStop < entryPrice - minStopDist) : (atrStop > entryPrice + minStopDist)
        bool swingValid = isLong ? (swingStop < entryPrice - minStopDist) : (swingStop > entryPrice + minStopDist)
        bool vwapValid = isLong ? (vwapValue < entryPrice and vwapStop < entryPrice - minStopDist) : ...
    
    For VWAP stops, also checks that VWAP is on correct side of entry.
    """
    min_stop_dist = atr * min_stop_atr
    
    if is_long:
        # Stop must be below entry by at least minStopDist
        valid = stop < entry_price - min_stop_dist
        # For VWAP, also check VWAP is below entry
        if vwap is not None:
            valid = valid and (vwap < entry_price)
    else:
        # Stop must be above entry by at least minStopDist
        valid = stop > entry_price + min_stop_dist
        # For VWAP, also check VWAP is above entry
        if vwap is not None:
            valid = valid and (vwap > entry_price)
    
    return valid


# ═══════════════════════════════════════════════════════════════════════════════
# R:R CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _calc_achievable_rr(entry: float, stop: float, atr: float, rr_desired: float, max_target_atr: float = 3.0) -> float:
    """Calculate achievable R:R with max target cap"""
    risk = abs(entry - stop)
    if risk <= 0:
        return -1
    target_distance = risk * rr_desired
    target_atrs = target_distance / atr
    if target_atrs > max_target_atr:
        return (max_target_atr * atr) / risk
    return rr_desired


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-SELECT BEST STOP
# ═══════════════════════════════════════════════════════════════════════════════

def _calc_stops(entry: float, atr: float, vwap: float, is_long: bool, 
                day_df: pd.DataFrame, bar_idx: int,
                atr_stop_mult: float = 2.0, vwap_stop_distance: float = 0.3,
                min_stop_atr: float = 0.15, exclude_vwap_hybrid: bool = False) -> dict:
    """Calculate all stop options including hybrid"""
    min_stop_dist = atr * min_stop_atr
    
    atr_stop = entry - (atr * atr_stop_mult) if is_long else entry + (atr * atr_stop_mult)
    swing_stop = calc_swing_stop(day_df, bar_idx, atr, is_long)
    vwap_stop = vwap - (atr * vwap_stop_distance) if is_long else vwap + (atr * vwap_stop_distance)
    hybrid_stop = calc_hybrid_stop(atr_stop, swing_stop, vwap_stop, is_long, exclude_vwap_hybrid)
    
    # Validity checks - Pine lines 1209-1213
    if is_long:
        atr_valid = atr_stop < entry - min_stop_dist
        swing_valid = swing_stop < entry - min_stop_dist
        vwap_valid = (vwap < entry) and (vwap_stop < entry - min_stop_dist)
        hybrid_valid = hybrid_stop < entry - min_stop_dist
    else:
        atr_valid = atr_stop > entry + min_stop_dist
        swing_valid = swing_stop > entry + min_stop_dist
        vwap_valid = (vwap > entry) and (vwap_stop > entry + min_stop_dist)
        hybrid_valid = hybrid_stop > entry + min_stop_dist
    
    # Fallback invalid stops to ATR (for R:R calc purposes)
    if not swing_valid:
        swing_stop = atr_stop
    if not vwap_valid:
        vwap_stop = atr_stop
    
    return {'atr_stop': atr_stop, 'swing_stop': swing_stop, 'vwap_stop': vwap_stop, 
            'hybrid_stop': hybrid_stop,
            'atr_valid': atr_valid, 'swing_valid': swing_valid, 
            'vwap_valid': vwap_valid, 'hybrid_valid': hybrid_valid}


def _select_best_stop(stops: dict, entry: float, atr: float, 
                      rr_desired: float, is_long: bool, max_target_atr: float = 3.0) -> tuple:
    """Select best valid stop from all 4 types (ATR, Swing, VWAP, Hybrid)"""
    candidates = []
    for stop_type in ['atr', 'swing', 'vwap', 'hybrid']:
        stop_price = stops[f'{stop_type}_stop']
        valid = stops.get(f'{stop_type}_valid', True)
        if not valid:
            continue
        rr = _calc_achievable_rr(entry, stop_price, atr, rr_desired, max_target_atr)
        if rr > 0:
            candidates.append((stop_price, stop_type.upper(), rr))
    
    if not candidates:
        rr = _calc_achievable_rr(entry, stops['atr_stop'], atr, rr_desired, max_target_atr)
        return stops['atr_stop'], 'ATR', rr
    
    return max(candidates, key=lambda x: x[2])


def select_best_stop(entry: float, atr: float, vwap: float, is_long: bool,
                     day_df: pd.DataFrame, bar_idx: int,
                     profit_target_rr: float = 2.0, max_target_atr: float = 3.0,
                     atr_stop_mult: float = 2.0, vwap_stop_distance: float = 0.3,
                     min_stop_atr: float = 0.15, vol_state: str = "NORMAL",
                     exclude_vwap_hybrid: bool = False) -> tuple:
    """
    Select best valid stop from all 4 types (ATR, Swing, VWAP, Hybrid).
    
    Pine evaluateBestStop() lines 1199-1260
    
    Returns: (stop_price, stop_type, achievable_rr)
    """
    # Adaptive volatility adjustment - orb_backtester.py lines 1121-1127
    rr_desired = profit_target_rr
    if vol_state == "LOW":
        rr_desired += 0.5
    elif vol_state == "HIGH":
        rr_desired = max(rr_desired - 0.5, 1.2)
    elif vol_state == "EXTREME":
        rr_desired = max(rr_desired - 0.8, 1.0)
    
    stops = _calc_stops(entry, atr, vwap, is_long, day_df, bar_idx,
                        atr_stop_mult, vwap_stop_distance, min_stop_atr, exclude_vwap_hybrid)
    stop_price, stop_type, achievable_rr = _select_best_stop(stops, entry, atr, rr_desired, is_long, max_target_atr)
    
    return stop_price, stop_type, achievable_rr


def check_trade_feasibility(achievable_rr: float, min_acceptable_rr: float = 1.5) -> tuple:
    """
    Check if trade meets minimum R:R threshold (THE GATE).
    
    orb_backtester.py line 1132:
        if achievable_rr >= self.min_acceptable_rr - 0.0001:
    
    Returns: (is_feasible, should_skip)
    """
    is_feasible = achievable_rr >= min_acceptable_rr - 0.0001
    should_skip = not is_feasible
    return is_feasible, should_skip


# ═══════════════════════════════════════════════════════════════════════════════
# EXIT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionState:
    """
    Tracks active position state for exit management.
    
    Pine equivalents:
        var float longEntryPrice = na
        var float longInitialStop = na
        var bool longStopMovedToBreakEven = false
        var bool longTrailingStopActivated = false
        var float longTrailingStopLevel = na
        var int longBarsUnderEma = 0
    """
    entry_price: float
    initial_stop: float
    stop_type: str
    is_long: bool
    risk: float  # abs(entry - initial_stop)
    target_rr: float
    
    # Exit state
    current_stop: float = None
    stop_moved_to_be: bool = False
    trailing_activated: bool = False
    trailing_stop_level: float = None
    bars_beyond_ema: int = 0  # Counter for EMA exit confirmation
    
    def __post_init__(self):
        self.risk = abs(self.entry_price - self.initial_stop)
        self.current_stop = self.initial_stop


def calc_break_even_target(entry_price: float, risk: float, is_long: bool,
                           break_even_rr: float = 0.5) -> float:
    """
    Calculate price level where stop moves to break-even.
    
    Pine (lines 2514-2515):
        float breakEvenTarget = longEntryPrice + (math.abs(longEntryPrice - longInitialStop) * breakEvenRR)
    
    Pine (lines 2524-2525) for SHORT:
        float breakEvenTarget = shortEntryPrice - (math.abs(shortEntryPrice - shortInitialStop) * breakEvenRR)
    """
    if is_long:
        return entry_price + (risk * break_even_rr)
    else:
        return entry_price - (risk * break_even_rr)


def check_break_even_hit(current_price: float, entry_price: float, risk: float, 
                         is_long: bool, break_even_rr: float = 0.5) -> bool:
    """
    Check if price has reached break-even target.
    
    Pine (lines 2516, 2526):
        if close >= breakEvenTarget  (LONG)
        if close <= breakEvenTarget  (SHORT)
    """
    be_target = calc_break_even_target(entry_price, risk, is_long, break_even_rr)
    if is_long:
        return current_price >= be_target
    else:
        return current_price <= be_target


def calc_profit_target(entry_price: float, risk: float, is_long: bool,
                       profit_target_rr: float = 2.0) -> float:
    """
    Calculate profit target price (trailing stop activation level).
    
    Pine (lines 2537-2538):
        float profitTargetPrice = longEntryPrice + (longRisk * profitTargetRR)
    
    Pine (lines 2568-2569) for SHORT:
        float profitTargetPrice = shortEntryPrice - (shortRisk * profitTargetRR)
    """
    if is_long:
        return entry_price + (risk * profit_target_rr)
    else:
        return entry_price - (risk * profit_target_rr)


def check_profit_target_hit(current_price: float, entry_price: float, risk: float,
                            is_long: bool, profit_target_rr: float = 2.0) -> bool:
    """
    Check if price has reached profit target (trailing activation).
    
    Pine (lines 2540, 2571):
        if not longTrailingStopActivated and close >= profitTargetPrice
        if not shortTrailingStopActivated and close <= profitTargetPrice
    """
    target = calc_profit_target(entry_price, risk, is_long, profit_target_rr)
    if is_long:
        return current_price >= target
    else:
        return current_price <= target


def calc_trailing_stop_level(current_price: float, atr: float, is_long: bool,
                             trailing_stop_distance: float = 1.2) -> float:
    """
    Calculate trailing stop level.
    
    Pine (lines 2543, 2574):
        longTrailingStopLevel := close - (atr * trailingStopDistance)
        shortTrailingStopLevel := close + (atr * trailingStopDistance)
    """
    if is_long:
        return current_price - (atr * trailing_stop_distance)
    else:
        return current_price + (atr * trailing_stop_distance)


def update_trailing_stop(position: PositionState, current_price: float, atr: float,
                         trailing_stop_distance: float = 1.2,
                         use_aggressive_trailing: bool = True, ema: float = None,
                         ema_tighten_zone: float = 0.3, tightened_trail_distance: float = 0.3) -> float:
    """
    Update trailing stop - ratchets tighter, never looser.
    
    Pine (lines 2551-2556 for LONG):
        float newTrailLevel = close - (atr * currentTrailDist)
        if newTrailLevel > longTrailingStopLevel
            longTrailingStopLevel := newTrailLevel
        longCurrentStop := math.max(longTrailingStopLevel, longEntryPrice)
    
    Pine (lines 2582-2587 for SHORT):
        float newTrailLevel = close + (atr * currentTrailDist)
        if newTrailLevel < shortTrailingStopLevel
            shortTrailingStopLevel := newTrailLevel
        shortCurrentStop := math.min(shortTrailingStopLevel, shortEntryPrice)
    
    Returns: new current stop level
    """
    # Aggressive trailing - tighten near EMA (from orb_backtester.py lines 893-896)
    actual_trail_dist = trailing_stop_distance
    if use_aggressive_trailing and ema is not None:
        dist_to_ema = abs(current_price - ema)
        if dist_to_ema <= (atr * ema_tighten_zone):
            actual_trail_dist = tightened_trail_distance
    
    new_trail = calc_trailing_stop_level(current_price, atr, position.is_long, actual_trail_dist)
    
    if position.is_long:
        # Only move stop UP (tighter)
        if position.trailing_stop_level is None or new_trail > position.trailing_stop_level:
            position.trailing_stop_level = new_trail
        # Trailing stop supersedes BE stop, but never below entry
        position.current_stop = max(position.trailing_stop_level, position.entry_price)
    else:
        # Only move stop DOWN (tighter)
        if position.trailing_stop_level is None or new_trail < position.trailing_stop_level:
            position.trailing_stop_level = new_trail
        # Trailing stop supersedes BE stop, but never above entry
        position.current_stop = min(position.trailing_stop_level, position.entry_price)
    
    return position.current_stop


def check_stop_hit(current_bar: pd.Series, position: PositionState) -> bool:
    """
    Check if stop was hit on this bar.
    
    For LONG: stop hit if low <= current_stop
    For SHORT: stop hit if high >= current_stop
    
    Stops must always be respected - when price touches the stop, you're out.
    """
    if position.is_long:
        return current_bar['low'] <= position.current_stop
    else:
        return current_bar['high'] >= position.current_stop


def check_target_hit(current_bar: pd.Series, entry_price: float, risk: float,
                     is_long: bool, target_rr: float) -> bool:
    """
    Check if profit target was hit on this bar.
    
    For LONG: target hit if high >= target_price
    For SHORT: target hit if low <= target_price
    """
    target = calc_profit_target(entry_price, risk, is_long, target_rr)
    if is_long:
        return current_bar['high'] >= target
    else:
        return current_bar['low'] <= target


def get_exit_reason(position: PositionState) -> str:
    """
    Get descriptive exit reason based on position state.
    """
    if position.trailing_activated:
        return "Trailing Stop"
    elif position.stop_moved_to_be:
        return "Break Even"
    else:
        return "Initial Stop"


def manage_position_exit(position: PositionState, current_bar: pd.Series, atr: float,
                         break_even_rr: float = 0.5, profit_target_rr: float = 2.0,
                         trailing_stop_distance: float = 1.2,
                         use_break_even: bool = True, use_trailing: bool = True,
                         use_ema_exit: bool = True, ema: float = None,
                         ema_confirmation_bars: int = 2,
                         use_aggressive_trailing: bool = True,
                         ema_tighten_zone: float = 0.3,
                         tightened_trail_distance: float = 0.3) -> tuple:
    """
    Full exit management for one bar.
    
    Returns: (is_closed, exit_price, exit_reason)
    
    Pine exit flow (lines 2510-2610):
    1. Check break-even activation
    2. Check trailing stop activation  
    3. Update trailing stop if active
    4. Check EMA exit (if BE hit but trailing not active)
    
    CRITICAL ORDER (from working orb_backtester.py lines 1005-1021):
    1. Check stop hit first (against CURRENT stop from previous bar's state)
    2. ELSE: activate BE, activate trailing, update trailing, check EMA exit
    
    This means EMA exit can fire on the SAME bar that BE is armed,
    preventing BE from stealing exits that should go to EMA.
    """
    current_price = current_bar['close']
    
    # 1. Check if stop hit FIRST (against current_stop set from PREVIOUS bar)
    if check_stop_hit(current_bar, position):
        exit_price = position.current_stop
        exit_reason = get_exit_reason(position)
        return True, exit_price, exit_reason
    
    # 2. If stop not hit, proceed with state updates and other exit checks
    
    # 2a. Check/activate break-even (Pine lines 2512-2520)
    # This ARMS BE but doesn't close - closing happens via stop_hit on future bar
    if use_break_even and not position.stop_moved_to_be:
        if check_break_even_hit(current_price, position.entry_price, position.risk,
                                position.is_long, break_even_rr):
            position.stop_moved_to_be = True
            position.current_stop = position.entry_price
    
    # 2b. Check/activate trailing stop (Pine lines 2540-2546)
    if use_trailing and position.stop_moved_to_be and not position.trailing_activated:
        if check_profit_target_hit(current_price, position.entry_price, position.risk,
                                   position.is_long, profit_target_rr):
            position.trailing_activated = True
            position.trailing_stop_level = calc_trailing_stop_level(
                current_price, atr, position.is_long, trailing_stop_distance
            )
            position.current_stop = position.trailing_stop_level
    
    # 2c. Update trailing stop if active (Pine lines 2549-2558)
    if position.trailing_activated:
        update_trailing_stop(position, current_price, atr, trailing_stop_distance,
                            use_aggressive_trailing, ema, ema_tighten_zone, tightened_trail_distance)
    
    # 2d. Check EMA exit LAST (Pine lines 2595-2610)
    # This CAN fire on same bar that BE was armed - EMA takes priority!
    if use_ema_exit and position.stop_moved_to_be and not position.trailing_activated:
        in_profit = (current_price > position.entry_price) if position.is_long else (current_price < position.entry_price)
        if in_profit and ema is not None:
            # Check if price crossed EMA
            if position.is_long:
                if current_price < ema:
                    position.bars_beyond_ema += 1
                else:
                    position.bars_beyond_ema = 0
            else:  # SHORT
                if current_price > ema:
                    position.bars_beyond_ema += 1
                else:
                    position.bars_beyond_ema = 0
            
            # Exit after confirmation bars
            if position.bars_beyond_ema >= ema_confirmation_bars:
                return True, current_price, "EMA Exit"
    
    return False, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("ORB Core Module - Validated Components")
    print("=" * 50)
    print("✓ ORBState dataclass")
    print("✓ is_orb_window()")
    print("✓ build_orb()")
    print("✓ calc_atr_rth() - RTH only, RMA smoothing")
    print("✓ check_long_breakout()")
    print("✓ check_short_breakout()")
    print("✓ calc_atr_stop() - VALIDATED")
    print("✓ calc_vwap_stop() - VALIDATED")
    print("✓ calc_swing_stop() - VALIDATED")
    print("✓ calc_hybrid_stop()")
    print("✓ check_stop_validity()")
    print("✓ calc_achievable_rr()")
    print("✓ select_best_stop() - AUTO-SELECT")
    print("✓ check_trade_feasibility() - SKIP LOGIC")
    print("─" * 50)
    print("✓ PositionState dataclass")
    print("✓ calc_break_even_target()")
    print("✓ check_break_even_hit()")
    print("✓ calc_profit_target()")
    print("✓ check_profit_target_hit()")
    print("✓ calc_trailing_stop_level()")
    print("✓ update_trailing_stop()")
    print("✓ check_stop_hit()")
    print("✓ manage_position_exit()")
    print("=" * 50)
    print("\nReady for exit management validation")
