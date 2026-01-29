"""
Single Day Tracer v3 - Pine v69 Compatible
==========================================
Key Changes from v2:
1. DUAL ATR: RTH ATR(14) for stops, RTH ATR(5) for vol state
2. SKIP CONSUMPTION: barsSinceLastTrigger resets on skip too
3. LOOKAHEAD FIX: RTH ATR shifted by 1 bar to match Pine's barmerge.lookahead_off
4. VOL STATE FIX: Uses RTH-based fast ATR, not ETH-based
5. WILDER RMA: Uses Pine's ta.rma() formula, not EMA

Usage:
    python single_day_tracer_v3.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from dataclasses import dataclass, field
from typing import Optional, Tuple
import sys

sys.path.insert(0, r'C:\Users\phemm\orb_lab\src')


@dataclass
class ORBState:
    """ORB range and breakout state machine"""
    session_high: float = np.nan
    session_low: float = np.nan
    orb_complete: bool = False
    
    # State machine (v66 fix)
    long_breakout_pending: bool = False
    short_breakout_pending: bool = False
    has_broken_out_high: bool = False
    has_broken_out_low: bool = False
    
    # v68 fix: cooldown tracking
    bars_since_last_trigger: int = 5  # Start ready to trade
    min_bars_between_triggers: int = 5
    
    def reset_session(self):
        """Reset at new session (9:30)"""
        self.session_high = np.nan
        self.session_low = np.nan
        self.orb_complete = False
        self.long_breakout_pending = False
        self.short_breakout_pending = False
        self.has_broken_out_high = False
        self.has_broken_out_low = False
        self.bars_since_last_trigger = self.min_bars_between_triggers


@dataclass 
class VolState:
    """Volatility state calculation"""
    orb_session_baseline: float = np.nan
    session_vf: float = 1.0
    htf_vf: float = 1.0
    vol_factor: float = 1.0
    vol_state: str = "NORMAL"


@dataclass
class TradeEval:
    """Forensic trace for last evaluation (v68)"""
    last_eval_rr: float = np.nan
    last_eval_result: str = "NONE"  # "ALERT" or "SKIP"
    last_eval_bar: int = -1
    last_eval_dir: str = "NONE"  # "LONG" or "SHORT"


class SingleDayTracer:
    """
    Traces a single day bar-by-bar with full debug output.
    Matches Pine v69 logic exactly.
    """
    
    # AMD Presets
    PRESETS = {
        'AMD': {
            'ssl_baseline_length': 20,
            'ssl_length': 10,
            'wae_fast_ema': 14,
            'wae_slow_ema': 29,
            'wae_sensitivity': 300,
            'wae_bb_length': 20,
            'qqe_rsi1_length': 8,
            'qqe_rsi1_smoothing': 5,
            'qqe_rsi2_length': 4,
            'qqe_rsi2_smoothing': 4,
            'qqe_bb_length': 25,
            'vol_lookback_bars': 12,
            'low_vol_threshold': 0.8,
            'high_vol_threshold': 1.3,
            'extreme_vol_threshold': 2.0,
        }
    }
    
    # Fixed parameters
    ORB_MINUTES = 5  # 9:30-9:35
    BREAKOUT_THRESHOLD_MULT = 0.1
    MIN_BODY_STRENGTH = 0.5
    REQUIRE_OPEN_IN_RANGE = True
    ATR_PERIOD = 14
    ATR_STOP_MULT = 2.0
    SWING_LOOKBACK = 5
    SWING_BUFFER = 0.02
    VWAP_STOP_DISTANCE = 0.3
    MIN_STOP_ATR = 0.15
    MAX_TARGET_ATR = 3.0
    MIN_ACCEPTABLE_RR = 1.5
    PROFIT_TARGET_RR = 2.0
    MIN_CONFLUENCE = 3
    
    def __init__(self, symbol: str = 'AMD', date: str = '2025-12-03'):
        self.symbol = symbol
        self.target_date = date
        self.preset = self.PRESETS.get(symbol, self.PRESETS['AMD'])
        
        self.orb = ORBState()
        self.vol = VolState()
        self.trace = TradeEval()
        
        # Data
        self.df = None      # Full ETH data
        self.df_rth = None  # RTH-only for ATR
        self.day_df = None
        
    def load_data(self):
        """Load data with extended hours"""
        from data_collector import PolygonDataCollector
        
        collector = PolygonDataCollector()
        
        # Load ETH data (includes pre-market)
        self.df = collector.fetch_bars(self.symbol, days_back=120, bar_size=1, extended_hours=True)
        
        # Filter to target date
        self.day_df = self.df[self.df.index.date.astype(str) == self.target_date].copy()
        
        if len(self.day_df) == 0:
            raise ValueError(f"No data for {self.target_date}")
        
        print(f"✓ Loaded {len(self.day_df)} bars for {self.symbol} on {self.target_date}")
        print(f"  First bar: {self.day_df.index[0]}")
        print(f"  Last bar: {self.day_df.index[-1]}")
        
    def calc_indicators(self):
        """Calculate dual ATRs, VWAP, EMA on full dataset"""
        df = self.df.copy()
        
        # ═══════════════════════════════════════════════════════════
        # TRUE RANGE (used by both ATRs)
        # ═══════════════════════════════════════════════════════════
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ═══════════════════════════════════════════════════════════
        # ETH ATR - includes pre-market (NOT used for vol state anymore)
        # ═══════════════════════════════════════════════════════════
        df['atr_eth'] = df['tr'].ewm(alpha=1/self.ATR_PERIOD, adjust=False).mean()
        
        # ═══════════════════════════════════════════════════════════
        # RTH ATR - Regular Trading Hours only (for stops AND vol state)
        # ═══════════════════════════════════════════════════════════
        # Pine's request.security(rthTicker, ...) simply runs ta.atr() on RTH-only bars.
        # No magic - just a continuous RTH stream across all days.
        # Dec 2 16:00 → Dec 3 09:30 is naturally handled by shift(1) on RTH series.
        
        df['is_rth'] = df.index.map(lambda x: 9*60+30 <= x.hour*60+x.minute < 16*60)
        
        # Extract RTH-only bars across ALL days
        rth_df = df[df['is_rth']].copy()
        
        # TR on continuous RTH stream - shift(1) gives prior RTH bar's close
        rth_df['prev_close'] = rth_df['close'].shift(1)
        tr1 = rth_df['high'] - rth_df['low']
        tr2 = (rth_df['high'] - rth_df['prev_close']).abs()
        tr3 = (rth_df['low'] - rth_df['prev_close']).abs()
        rth_df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Wilder RMA function (Pine's ta.atr uses this, NOT EMA)
        def wilder_rma(series, n):
            rma = series.copy().astype(float)
            # Seed with SMA of first n values
            rma.iloc[:n] = series.iloc[:n].expanding().mean()
            # Wilder smoothing after seed
            for i in range(n, len(series)):
                rma.iloc[i] = (rma.iloc[i-1] * (n - 1) + series.iloc[i]) / n
            return rma
        
        # ATR(14) on RTH stream - for stops (Wilder RMA, not EMA!)
        rth_df['atr'] = wilder_rma(rth_df['tr'], self.ATR_PERIOD)
        
        # ATR(5) fast on RTH stream - for vol state (Wilder RMA, not EMA!)
        rth_df['atr_fast'] = wilder_rma(rth_df['tr'], 5)
        # Don't pre-average - let the ORB window baseline smoother handle it
        rth_df['atr_fast_sma'] = rth_df['atr_fast']
        
        # LOOKAHEAD FIX: Pine's request.security(..., barmerge.lookahead_off) 
        # returns PREVIOUS bar's ATR. Shift by 1 to match.
        rth_df['atr_pine'] = rth_df['atr'].shift(1)
        
        # Map back to full dataframe
        df['atr_rth'] = np.nan
        df.loc[rth_df.index, 'atr_rth'] = rth_df['atr_pine']
        
        # Map fast ATR (RTH-based) for vol state
        df['atr_fast_rth'] = np.nan
        df.loc[rth_df.index, 'atr_fast_rth'] = rth_df['atr_fast']
        df['atr_fast_sma_rth'] = np.nan
        df.loc[rth_df.index, 'atr_fast_sma_rth'] = rth_df['atr_fast_sma']
        
        # Forward fill into pre-market (so we have a value before 09:30)
        df['atr_rth'] = df['atr_rth'].ffill()
        df['atr_rth'] = df['atr_rth'].fillna(df['atr_eth'])
        df['atr_fast_rth'] = df['atr_fast_rth'].ffill()
        df['atr_fast_sma_rth'] = df['atr_fast_sma_rth'].ffill()
        
        # ═══════════════════════════════════════════════════════════
        # VWAP (resets at 9:30)
        # ═══════════════════════════════════════════════════════════
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']
        
        # Detect new RTH sessions (9:30)
        df['is_930'] = (df.index.hour == 9) & (df.index.minute == 30)
        df['session_id'] = df['is_930'].cumsum()
        
        # Cumulative for VWAP
        df['cum_tp_vol'] = df.groupby('session_id')['tp_volume'].cumsum()
        df['cum_vol'] = df.groupby('session_id')['volume'].cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        
        # ═══════════════════════════════════════════════════════════
        # EMA 9
        # ═══════════════════════════════════════════════════════════
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        
        self.df = df
        self.day_df = df[df.index.date.astype(str) == self.target_date].copy()
        
        # Verify ATRs at 09:35
        bar_935 = self.day_df[self.day_df.index.strftime('%H:%M') == '09:35']
        if len(bar_935) > 0:
            print(f"✓ ATR at 09:35:")
            print(f"    ETH ATR: {bar_935['atr_eth'].values[0]:.4f} (for vol state)")
            print(f"    RTH ATR: {bar_935['atr_rth'].values[0]:.4f} (for stops)")
        
    def is_orb_window(self, ts) -> bool:
        """Check if timestamp is in ORB window (9:30-9:35)"""
        bar_minutes = ts.hour * 60 + ts.minute
        orb_start = 9 * 60 + 30
        orb_end = orb_start + self.ORB_MINUTES
        return orb_start <= bar_minutes < orb_end
    
    def is_trading_window(self, ts) -> bool:
        """Check if in first hour (9:30-10:30)"""
        bar_minutes = ts.hour * 60 + ts.minute
        return 9 * 60 + 30 <= bar_minutes < 10 * 60 + 30
    
    def is_rth(self, ts) -> bool:
        """Check if regular trading hours (9:30-16:00)"""
        bar_minutes = ts.hour * 60 + ts.minute
        return 9 * 60 + 30 <= bar_minutes < 16 * 60
    
    def calc_vol_state(self, i: int) -> None:
        """
        Calculate volatility state - uses RTH ATR (not ETH!)
        """
        ts = self.day_df.index[i]
        bar = self.day_df.iloc[i]
        
        # Use RTH-based fast ATR for vol state (NOT ETH!)
        atr_fast = bar['atr_fast_rth'] if not pd.isna(bar.get('atr_fast_rth')) else bar['atr_rth']
        atr_fast_sma = bar['atr_fast_sma_rth'] if not pd.isna(bar.get('atr_fast_sma_rth')) else atr_fast
        
        # Check if in ORB window (9:30-10:00 for vol calc)
        bar_minutes = ts.hour * 60 + ts.minute
        is_orb_vol_window = 9 * 60 + 30 <= bar_minutes < 10 * 60
        
        # Reset on new day (9:30 bar)
        if ts.hour == 9 and ts.minute == 30:
            self.vol.orb_session_baseline = np.nan
        
        # Update baseline during ORB window
        if is_orb_vol_window:
            if pd.isna(self.vol.orb_session_baseline):
                self.vol.orb_session_baseline = atr_fast_sma
            else:
                # Smoothing: 0.9 * old + 0.1 * new
                self.vol.orb_session_baseline = 0.9 * self.vol.orb_session_baseline + 0.1 * atr_fast_sma
        
        # Session VF
        if not pd.isna(self.vol.orb_session_baseline) and self.vol.orb_session_baseline > 0:
            self.vol.session_vf = atr_fast / self.vol.orb_session_baseline
        else:
            self.vol.session_vf = 1.0
        
        # HTF VF (daily) - uses RTH ATR
        daily_atr = bar['atr_rth'] if not pd.isna(bar.get('atr_rth')) else 1.0
        # For daily ATR slow, use rolling mean of RTH ATR
        daily_atr_slow = self.df['atr_rth'].rolling(20).mean().loc[ts] if ts in self.df.index else daily_atr
        if not pd.isna(daily_atr_slow) and daily_atr_slow > 0:
            self.vol.htf_vf = daily_atr / daily_atr_slow
        else:
            self.vol.htf_vf = 1.0
        
        # Combined
        self.vol.vol_factor = self.vol.session_vf * self.vol.htf_vf
        
        # State thresholds
        if self.vol.vol_factor < self.preset['low_vol_threshold']:
            self.vol.vol_state = "LOW"
        elif self.vol.vol_factor >= self.preset['extreme_vol_threshold']:
            self.vol.vol_state = "EXTREME"
        elif self.vol.vol_factor >= self.preset['high_vol_threshold']:
            self.vol.vol_state = "HIGH"
        else:
            self.vol.vol_state = "NORMAL"
    
    def check_breakout(self, i: int, bar: pd.Series) -> Tuple[bool, bool]:
        """
        Check for new breakout - Pine v69 state machine
        Uses RTH ATR for threshold calculation
        Returns (new_long_breakout, new_short_breakout)
        """
        if not self.orb.orb_complete:
            return False, False
        
        # Use RTH ATR for breakout threshold (v69)
        atr = bar['atr_rth']
        threshold = atr * self.BREAKOUT_THRESHOLD_MULT
        
        candle_body = abs(bar['close'] - bar['open'])
        candle_range = bar['high'] - bar['low']
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        
        sufficient_high = bar['close'] > self.orb.session_high + threshold
        sufficient_low = bar['close'] < self.orb.session_low - threshold
        strong_body = body_ratio >= self.MIN_BODY_STRENGTH
        
        new_long = False
        new_short = False
        
        if self.REQUIRE_OPEN_IN_RANGE:
            open_in_range = (bar['open'] >= self.orb.session_low and 
                           bar['open'] <= self.orb.session_high)
            
            new_long = (open_in_range and 
                       bar['close'] > self.orb.session_high and 
                       sufficient_high and 
                       strong_body)
            
            new_short = (open_in_range and 
                        bar['close'] < self.orb.session_low and 
                        sufficient_low and 
                        strong_body)
        else:
            new_long = (bar['close'] > self.orb.session_high and 
                       sufficient_high and strong_body)
            new_short = (bar['close'] < self.orb.session_low and 
                        sufficient_low and strong_body)
        
        return new_long, new_short
    
    def calc_stops(self, entry: float, atr_rth: float, vwap: float, 
                   is_long: bool, i: int) -> dict:
        """
        Calculate stops with validity checks - uses RTH ATR (v69)
        """
        min_stop_dist = atr_rth * self.MIN_STOP_ATR
        
        # ATR stop - uses RTH ATR
        atr_stop = entry - (atr_rth * self.ATR_STOP_MULT) if is_long else \
                   entry + (atr_rth * self.ATR_STOP_MULT)
        
        # Swing stop (RTH bars only)
        swing_stop = self._calc_swing_stop(i, is_long, atr_rth)
        
        # VWAP stop - uses RTH ATR
        vwap_stop = vwap - (atr_rth * self.VWAP_STOP_DISTANCE) if is_long else \
                    vwap + (atr_rth * self.VWAP_STOP_DISTANCE)
        
        # Validity checks
        if is_long:
            atr_valid = atr_stop < entry - min_stop_dist
            swing_valid = swing_stop < entry - min_stop_dist
            vwap_valid = (vwap < entry) and (vwap_stop < entry - min_stop_dist)
        else:
            atr_valid = atr_stop > entry + min_stop_dist
            swing_valid = swing_stop > entry + min_stop_dist
            vwap_valid = (vwap > entry) and (vwap_stop > entry + min_stop_dist)
        
        # Fallback invalid to ATR
        if not swing_valid:
            swing_stop = atr_stop
        if not vwap_valid:
            vwap_stop = atr_stop
        
        return {
            'atr_stop': atr_stop,
            'swing_stop': swing_stop,
            'vwap_stop': vwap_stop,
            'atr_valid': atr_valid,
            'swing_valid': swing_valid,
            'vwap_valid': vwap_valid,
            'min_stop_dist': min_stop_dist
        }
    
    def _calc_swing_stop(self, i: int, is_long: bool, atr: float) -> float:
        """Find swing high/low from RTH bars only"""
        lookback = self.SWING_LOOKBACK
        valid_bars = 0
        swing_val = None
        
        ts = self.day_df.index[i]
        full_idx = self.df.index.get_loc(ts)
        
        for j in range(1, lookback * 3 + 1):
            if full_idx - j < 0:
                break
            bar_ts = self.df.index[full_idx - j]
            if self.is_rth(bar_ts):
                if is_long:
                    bar_low = self.df['low'].iloc[full_idx - j]
                    if swing_val is None or bar_low < swing_val:
                        swing_val = bar_low
                else:
                    bar_high = self.df['high'].iloc[full_idx - j]
                    if swing_val is None or bar_high > swing_val:
                        swing_val = bar_high
                valid_bars += 1
                if valid_bars >= lookback:
                    break
        
        if swing_val is None:
            swing_val = self.day_df['low'].iloc[max(0, i-1)] if is_long else \
                        self.day_df['high'].iloc[max(0, i-1)]
        
        if is_long:
            return swing_val - (atr * self.SWING_BUFFER)
        else:
            return swing_val + (atr * self.SWING_BUFFER)
    
    def calc_achievable_rr(self, entry: float, stop: float, 
                           atr: float, rr_desired: float) -> float:
        """Calculate achievable R:R with maxTargetATR cap"""
        risk = abs(entry - stop)
        if risk <= 0:
            return -1
        
        target_distance = risk * rr_desired
        target_atrs = target_distance / atr
        
        if target_atrs > self.MAX_TARGET_ATR:
            return (self.MAX_TARGET_ATR * atr) / risk
        else:
            return rr_desired
    
    def select_best_stop(self, stops: dict, entry: float, 
                         atr: float, rr_desired: float, is_long: bool) -> Tuple[float, str, float]:
        """Select best valid stop (highest achievable R:R)"""
        candidates = []
        
        for stop_type in ['atr', 'swing', 'vwap']:
            stop_price = stops[f'{stop_type}_stop']
            valid = stops[f'{stop_type}_valid'] if stop_type != 'atr' else True
            
            if not valid:
                continue
            
            rr = self.calc_achievable_rr(entry, stop_price, atr, rr_desired)
            if rr > 0:
                candidates.append((stop_price, stop_type.upper(), rr))
        
        if not candidates:
            rr = self.calc_achievable_rr(entry, stops['atr_stop'], atr, rr_desired)
            return stops['atr_stop'], 'ATR', rr
        
        best = max(candidates, key=lambda x: x[2])
        return best
    
    def can_trigger(self) -> bool:
        """Check if cooldown allows new trigger (v68)"""
        return self.orb.bars_since_last_trigger >= self.orb.min_bars_between_triggers
    
    def consume_trigger(self, result: str, direction: str, rr: float, bar_idx: int):
        """Consume trigger attempt - reset cooldown (v68 fix: applies to SKIP too)"""
        self.orb.bars_since_last_trigger = 0
        self.trace.last_eval_rr = rr
        self.trace.last_eval_result = result
        self.trace.last_eval_bar = bar_idx
        self.trace.last_eval_dir = direction
    
    def run(self):
        """Run the tracer"""
        print(f"\n{'='*60}")
        print(f"  SINGLE DAY TRACER v3: {self.symbol} on {self.target_date}")
        print(f"  (Pine v69: RTH ATR for stops, ETH ATR for vol state)")
        print(f"{'='*60}\n")
        
        # Load and prepare data
        self.load_data()
        self.calc_indicators()
        
        # Reset state
        self.orb.reset_session()
        
        print(f"\n{'─'*60}")
        print(f"  BAR-BY-BAR TRACE (RTH only)")
        print(f"{'─'*60}\n")
        
        bar_idx = 0
        for i in range(len(self.day_df)):
            ts = self.day_df.index[i]
            bar = self.day_df.iloc[i]
            
            # Skip pre-market for trace output (but data is used for ATR)
            if not self.is_rth(ts):
                continue
            
            time_str = ts.strftime('%H:%M')
            bar_idx += 1
            
            # Increment cooldown counter each RTH bar
            self.orb.bars_since_last_trigger += 1
            
            # ═══════════════════════════════════════════════════════════
            # PHASE 1: ORB Range Building
            # ═══════════════════════════════════════════════════════════
            
            if self.is_orb_window(ts):
                if ts.hour == 9 and ts.minute == 30:
                    self.orb.reset_session()
                    self.orb.session_high = bar['high']
                    self.orb.session_low = bar['low']
                    print(f"[{time_str}] 🟡 ORB START - H: {bar['high']:.2f}, L: {bar['low']:.2f}")
                else:
                    self.orb.session_high = max(self.orb.session_high, bar['high'])
                    self.orb.session_low = min(self.orb.session_low, bar['low'])
                    print(f"[{time_str}]    ORB building - H: {self.orb.session_high:.2f}, L: {self.orb.session_low:.2f}")
                continue
            
            # ORB complete after window
            if not self.orb.orb_complete and not pd.isna(self.orb.session_high):
                self.orb.orb_complete = True
                orb_range = self.orb.session_high - self.orb.session_low
                print(f"\n[{time_str}] 🟢 ORB COMPLETE")
                print(f"         High: {self.orb.session_high:.2f}")
                print(f"         Low:  {self.orb.session_low:.2f}")
                print(f"         Range: ${orb_range:.2f}")
                print(f"         ETH ATR: ${bar['atr_eth']:.4f} (vol state)")
                print(f"         RTH ATR: ${bar['atr_rth']:.4f} (stops)\n")
            
            # ═══════════════════════════════════════════════════════════
            # PHASE 2: Volatility State (uses ETH ATR)
            # ═══════════════════════════════════════════════════════════
            
            self.calc_vol_state(i)
            
            # ═══════════════════════════════════════════════════════════
            # PHASE 3: Breakout Detection
            # ═══════════════════════════════════════════════════════════
            
            new_long, new_short = self.check_breakout(i, bar)
            
            # Latch breakout
            if new_long and not self.orb.has_broken_out_high:
                self.orb.long_breakout_pending = True
                self.orb.has_broken_out_high = True
                
                print(f"[{time_str}] 🔺 NEW LONG BREAKOUT DETECTED")
                print(f"         Open:  {bar['open']:.2f} (in range: {self.orb.session_low:.2f}-{self.orb.session_high:.2f})")
                print(f"         Close: {bar['close']:.2f} > ORB High {self.orb.session_high:.2f}")
                print(f"         RTH ATR: {bar['atr_rth']:.4f}")
                print(f"         Vol State: {self.vol.vol_state} (factor: {self.vol.vol_factor:.2f})")
            
            if new_short and not self.orb.has_broken_out_low:
                self.orb.short_breakout_pending = True
                self.orb.has_broken_out_low = True
                
                print(f"[{time_str}] 🔻 NEW SHORT BREAKOUT DETECTED")
                print(f"         Open:  {bar['open']:.2f} (in range: {self.orb.session_low:.2f}-{self.orb.session_high:.2f})")
                print(f"         Close: {bar['close']:.2f} < ORB Low {self.orb.session_low:.2f}")
                print(f"         RTH ATR: {bar['atr_rth']:.4f}")
                print(f"         Vol State: {self.vol.vol_state} (factor: {self.vol.vol_factor:.2f})")
            
            # ═══════════════════════════════════════════════════════════
            # PHASE 4: Entry Decision (if pending AND cooldown allows)
            # ═══════════════════════════════════════════════════════════
            
            if self.orb.long_breakout_pending and self.is_trading_window(ts) and self.can_trigger():
                entry = bar['close']
                atr_rth = bar['atr_rth']  # Use RTH ATR for stops (v69)
                vwap = bar['vwap']
                
                rr_desired = self.PROFIT_TARGET_RR
                if self.vol.vol_state == "LOW":
                    rr_desired += 0.5
                elif self.vol.vol_state == "HIGH":
                    rr_desired = max(rr_desired - 0.5, 1.2)
                elif self.vol.vol_state == "EXTREME":
                    rr_desired = max(rr_desired - 0.8, 1.0)
                
                stops = self.calc_stops(entry, atr_rth, vwap, True, i)
                stop_price, stop_type, achievable_rr = self.select_best_stop(
                    stops, entry, atr_rth, rr_desired, True)
                
                print(f"\n         📊 LONG ENTRY EVALUATION:")
                print(f"         Entry: ${entry:.2f}")
                print(f"         RTH ATR: ${atr_rth:.4f}")
                print(f"         VWAP: ${vwap:.2f}")
                print(f"         R:R Desired: {rr_desired:.2f} (adj for {self.vol.vol_state})")
                print(f"         Stop Type: {stop_type} @ ${stop_price:.2f}")
                print(f"         Achievable R:R: {achievable_rr:.2f}")
                print(f"         Min Required: {self.MIN_ACCEPTABLE_RR}")
                
                if achievable_rr >= self.MIN_ACCEPTABLE_RR - 0.0001:  # tolerance for float comparison
                    risk = abs(entry - stop_price)
                    target = entry + (risk * achievable_rr)
                    print(f"\n         ✅ TRADE TAKEN")
                    print(f"         Target: ${target:.2f}")
                    print(f"         Risk: ${risk:.2f}")
                    self.consume_trigger("ALERT", "LONG", achievable_rr, bar_idx)
                else:
                    print(f"\n         ❌ SKIP: {achievable_rr:.2f} < {self.MIN_ACCEPTABLE_RR}")
                    # v68 fix: consume on skip too
                    self.consume_trigger("SKIP", "LONG", achievable_rr, bar_idx)
            
            if self.orb.short_breakout_pending and self.is_trading_window(ts) and self.can_trigger():
                entry = bar['close']
                atr_rth = bar['atr_rth']  # Use RTH ATR for stops (v69)
                vwap = bar['vwap']
                
                rr_desired = self.PROFIT_TARGET_RR
                if self.vol.vol_state == "LOW":
                    rr_desired += 0.5
                elif self.vol.vol_state == "HIGH":
                    rr_desired = max(rr_desired - 0.5, 1.2)
                elif self.vol.vol_state == "EXTREME":
                    rr_desired = max(rr_desired - 0.8, 1.0)
                
                stops = self.calc_stops(entry, atr_rth, vwap, False, i)
                stop_price, stop_type, achievable_rr = self.select_best_stop(
                    stops, entry, atr_rth, rr_desired, False)
                
                print(f"\n         📊 SHORT ENTRY EVALUATION:")
                print(f"         Entry: ${entry:.2f}")
                print(f"         RTH ATR: ${atr_rth:.4f}")
                print(f"         VWAP: ${vwap:.2f}")
                print(f"         R:R Desired: {rr_desired:.2f} (adj for {self.vol.vol_state})")
                print(f"         Stop Type: {stop_type} @ ${stop_price:.2f}")
                print(f"         Achievable R:R: {achievable_rr:.2f}")
                print(f"         Min Required: {self.MIN_ACCEPTABLE_RR}")
                
                if achievable_rr >= self.MIN_ACCEPTABLE_RR - 0.0001:  # tolerance for float comparison
                    risk = abs(entry - stop_price)
                    target = entry - (risk * achievable_rr)
                    print(f"\n         ✅ TRADE TAKEN")
                    print(f"         Target: ${target:.2f}")
                    print(f"         Risk: ${risk:.2f}")
                    self.consume_trigger("ALERT", "SHORT", achievable_rr, bar_idx)
                else:
                    print(f"\n         ❌ SKIP: {achievable_rr:.2f} < {self.MIN_ACCEPTABLE_RR}")
                    # v68 fix: consume on skip too
                    self.consume_trigger("SKIP", "SHORT", achievable_rr, bar_idx)
            
            # ═══════════════════════════════════════════════════════════
            # PHASE 5: State Machine Updates (end of bar)
            # ═══════════════════════════════════════════════════════════
            
            if self.orb.orb_complete:
                back_inside = (bar['close'] <= self.orb.session_high and 
                              bar['close'] >= self.orb.session_low)
                
                if back_inside:
                    if self.orb.has_broken_out_high:
                        self.orb.has_broken_out_high = False
                        print(f"[{time_str}]    ↩️ Price back inside - re-armed HIGH")
                    if self.orb.has_broken_out_low:
                        self.orb.has_broken_out_low = False
                        print(f"[{time_str}]    ↩️ Price back inside - re-armed LOW")
            
            # Consume pending (unconditional at bar close)
            if self.orb.long_breakout_pending:
                self.orb.long_breakout_pending = False
            if self.orb.short_breakout_pending:
                self.orb.short_breakout_pending = False
        
        print(f"\n{'='*60}")
        print(f"  TRACE COMPLETE")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    tracer = SingleDayTracer(symbol='AMD', date='2025-12-03')
    tracer.run()
