"""
Single Day Tracer v4 - Pine v69 Compatible with EXIT LOGIC
===========================================================
Key Features:
1. Entry logic from v3 (validated)
2. Break-even stop at 0.5R profit
3. Trailing stop activates at 2.0R (profitTargetRR)
4. Trailing distance: 1.2 ATR (tightens to 0.3 ATR near EMA)
5. EMA exit (only when NOT trailing)
6. Once trailing activates, it has SOLE priority
7. HTF VF uses DAILY RTH ATR (not rolling 1-minute)

Exit Priority:
- Before trailing: Initial stop → Break-even → EMA exit
- After trailing: Trailing stop ONLY (EMA exit disabled)

Usage:
    python single_day_tracer_v4.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
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
    bars_since_last_trigger: int = 5
    min_bars_between_triggers: int = 5
    
    def reset_session(self):
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
class Position:
    """Active position tracking with exit management"""
    direction: str = ""  # "LONG" or "SHORT"
    entry_price: float = np.nan
    entry_bar: int = -1
    entry_time: str = ""
    initial_stop: float = np.nan
    current_stop: float = np.nan
    stop_type: str = ""
    risk: float = np.nan
    target_rr: float = np.nan
    
    # Exit state
    stop_moved_to_be: bool = False
    trailing_activated: bool = False
    trailing_stop_level: float = np.nan
    bars_beyond_ema: int = 0
    
    # Result tracking
    exit_price: float = np.nan
    exit_bar: int = -1
    exit_time: str = ""
    exit_reason: str = ""
    pnl: float = np.nan
    r_multiple: float = np.nan
    
    def reset(self):
        self.direction = ""
        self.entry_price = np.nan
        self.entry_bar = -1
        self.entry_time = ""
        self.initial_stop = np.nan
        self.current_stop = np.nan
        self.stop_type = ""
        self.risk = np.nan
        self.target_rr = np.nan
        self.stop_moved_to_be = False
        self.trailing_activated = False
        self.trailing_stop_level = np.nan
        self.bars_beyond_ema = 0
        self.exit_price = np.nan
        self.exit_bar = -1
        self.exit_time = ""
        self.exit_reason = ""
        self.pnl = np.nan
        self.r_multiple = np.nan


@dataclass
class TradeResult:
    """Completed trade record"""
    direction: str
    entry_price: float
    entry_time: str
    exit_price: float
    exit_time: str
    exit_reason: str
    initial_stop: float
    risk: float
    pnl: float
    r_multiple: float


class SingleDayTracerV4:
    """
    Single day tracer with full exit management.
    Matches Pine v69 exit logic exactly.
    """
    
    # Default Presets (can be overridden)
    PRESETS = {
        'AMD': {
            'low_vol_threshold': 0.8,
            'high_vol_threshold': 1.3,
            'extreme_vol_threshold': 2.0,
        },
        'NVDA': {
            'low_vol_threshold': 0.8,
            'high_vol_threshold': 1.3,
            'extreme_vol_threshold': 2.0,
        },
        'TSLA': {
            'low_vol_threshold': 0.8,
            'high_vol_threshold': 1.3,
            'extreme_vol_threshold': 2.0,
        }
    }
    
    # Entry parameters
    ORB_MINUTES = 5
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
    
    # Exit parameters (Pine v69 defaults)
    BREAK_EVEN_RR = 0.5
    TRAILING_STOP_DISTANCE = 1.2  # ATR multiplier
    EMA_TIGHTEN_ZONE = 0.3  # ATR multiplier
    TIGHTENED_TRAIL_DISTANCE = 0.3  # ATR multiplier
    EMA_CONFIRMATION_BARS = 1
    USE_AGGRESSIVE_TRAILING = True
    
    def __init__(self, symbol: str = 'AMD', date: str = '2025-12-03'):
        self.symbol = symbol
        self.target_date = date
        self.preset = self.PRESETS.get(symbol, self.PRESETS['AMD'])
        
        self.orb = ORBState()
        self.vol = VolState()
        self.position = Position()
        self.trades: List[TradeResult] = []
        
        self.df = None
        self.day_df = None
        
    def load_data(self):
        """Load data with extended hours"""
        from data_collector import PolygonDataCollector
        
        collector = PolygonDataCollector()
        self.df = collector.fetch_bars(self.symbol, days_back=120, bar_size=1, extended_hours=True)
        self.day_df = self.df[self.df.index.date.astype(str) == self.target_date].copy()
        
        if len(self.day_df) == 0:
            raise ValueError(f"No data for {self.target_date}")
        
        print(f"✓ Loaded {len(self.day_df)} bars for {self.symbol} on {self.target_date}")
        
    def calc_indicators(self):
        """Calculate ATRs, VWAP, EMA on full dataset"""
        df = self.df.copy()
        
        # TR for ETH
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ETH ATR (fallback only)
        df['atr_eth'] = df['tr'].ewm(alpha=1/self.ATR_PERIOD, adjust=False).mean()
        
        # RTH mask
        df['is_rth'] = df.index.map(lambda x: 9*60+30 <= x.hour*60+x.minute < 16*60)
        
        # RTH-only calculations
        rth_df = df[df['is_rth']].copy()
        rth_df['prev_close'] = rth_df['close'].shift(1)
        tr1_rth = rth_df['high'] - rth_df['low']
        tr2_rth = (rth_df['high'] - rth_df['prev_close']).abs()
        tr3_rth = (rth_df['low'] - rth_df['prev_close']).abs()
        rth_df['tr'] = pd.concat([tr1_rth, tr2_rth, tr3_rth], axis=1).max(axis=1)
        
        # Wilder RMA function
        def wilder_rma(series, n):
            rma = series.copy().astype(float)
            rma.iloc[:n] = series.iloc[:n].expanding().mean()
            for i in range(n, len(series)):
                rma.iloc[i] = (rma.iloc[i-1] * (n - 1) + series.iloc[i]) / n
            return rma
        
        # ATR(14) for stops, ATR(5) for vol state
        rth_df['atr'] = wilder_rma(rth_df['tr'], self.ATR_PERIOD)
        rth_df['atr_fast'] = wilder_rma(rth_df['tr'], 5)
        rth_df['atr_fast_sma'] = rth_df['atr_fast']  # No cross-day rolling
        
        # Lookahead fix - shift by 1
        rth_df['atr_pine'] = rth_df['atr'].shift(1)
        
        # Map back to full df
        df['atr_rth'] = np.nan
        df.loc[rth_df.index, 'atr_rth'] = rth_df['atr_pine']
        df['atr_fast_rth'] = np.nan
        df.loc[rth_df.index, 'atr_fast_rth'] = rth_df['atr_fast']
        df['atr_fast_sma_rth'] = np.nan
        df.loc[rth_df.index, 'atr_fast_sma_rth'] = rth_df['atr_fast_sma']
        
        # Forward fill
        df['atr_rth'] = df['atr_rth'].ffill().fillna(df['atr_eth'])
        df['atr_fast_rth'] = df['atr_fast_rth'].ffill()
        df['atr_fast_sma_rth'] = df['atr_fast_sma_rth'].ffill()
        
        # ═══════════════════════════════════════════════════════════
        # DAILY ATR FOR HTF VF (Pine's request.security on "D" timeframe)
        # ═══════════════════════════════════════════════════════════
        # Build daily RTH bars from minute data
        rth_df['date'] = rth_df.index.date
        daily_rth = rth_df.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Calculate daily TR
        daily_rth['prev_close'] = daily_rth['close'].shift(1)
        daily_tr1 = daily_rth['high'] - daily_rth['low']
        daily_tr2 = (daily_rth['high'] - daily_rth['prev_close']).abs()
        daily_tr3 = (daily_rth['low'] - daily_rth['prev_close']).abs()
        daily_rth['tr'] = pd.concat([daily_tr1, daily_tr2, daily_tr3], axis=1).max(axis=1)
        
        # Daily ATR(14) with Wilder RMA
        daily_rth['daily_atr'] = wilder_rma(daily_rth['tr'], 14)
        
        # Daily ATR slow (20-day SMA)
        daily_rth['daily_atr_slow'] = daily_rth['daily_atr'].rolling(20).mean()
        
        # Map daily ATR back to intraday bars
        df['date'] = df.index.date
        df['daily_atr'] = df['date'].map(daily_rth['daily_atr'].to_dict())
        df['daily_atr_slow'] = df['date'].map(daily_rth['daily_atr_slow'].to_dict())
        
        # Forward fill for safety
        df['daily_atr'] = df['daily_atr'].ffill()
        df['daily_atr_slow'] = df['daily_atr_slow'].ffill()
        
        # VWAP (resets at 9:30)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']
        df['is_930'] = (df.index.hour == 9) & (df.index.minute == 30)
        df['session_id'] = df['is_930'].cumsum()
        df['cum_tp_vol'] = df.groupby('session_id')['tp_volume'].cumsum()
        df['cum_vol'] = df.groupby('session_id')['volume'].cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        
        # EMA 9
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        
        self.df = df
        self.day_df = df[df.index.date.astype(str) == self.target_date].copy()
        
    def is_orb_window(self, ts) -> bool:
        bar_minutes = ts.hour * 60 + ts.minute
        return 9*60+30 <= bar_minutes < 9*60+30 + self.ORB_MINUTES
    
    def is_trading_window(self, ts) -> bool:
        bar_minutes = ts.hour * 60 + ts.minute
        return 9*60+30 <= bar_minutes < 10*60+30
    
    def is_rth(self, ts) -> bool:
        bar_minutes = ts.hour * 60 + ts.minute
        return 9*60+30 <= bar_minutes < 16*60
    
    def calc_vol_state(self, i: int) -> None:
        ts = self.day_df.index[i]
        bar = self.day_df.iloc[i]
        
        atr_fast = bar.get('atr_fast_rth', bar['atr_rth'])
        atr_fast_sma = bar.get('atr_fast_sma_rth', atr_fast)
        if pd.isna(atr_fast):
            atr_fast = bar['atr_rth']
        if pd.isna(atr_fast_sma):
            atr_fast_sma = atr_fast
        
        bar_minutes = ts.hour * 60 + ts.minute
        is_orb_vol_window = 9*60+30 <= bar_minutes < 10*60
        
        if ts.hour == 9 and ts.minute == 30:
            self.vol.orb_session_baseline = np.nan
        
        if is_orb_vol_window:
            if pd.isna(self.vol.orb_session_baseline):
                self.vol.orb_session_baseline = atr_fast_sma
            else:
                self.vol.orb_session_baseline = 0.9 * self.vol.orb_session_baseline + 0.1 * atr_fast_sma
        
        if not pd.isna(self.vol.orb_session_baseline) and self.vol.orb_session_baseline > 0:
            self.vol.session_vf = atr_fast / self.vol.orb_session_baseline
        else:
            self.vol.session_vf = 1.0
        
        # HTF VF - use DAILY ATR (not rolling 1-minute!)
        daily_atr = bar.get('daily_atr', 1.0)
        daily_atr_slow = bar.get('daily_atr_slow', daily_atr)
        if pd.isna(daily_atr):
            daily_atr = 1.0
        if pd.isna(daily_atr_slow) or daily_atr_slow <= 0:
            daily_atr_slow = daily_atr
        
        if daily_atr_slow > 0:
            self.vol.htf_vf = daily_atr / daily_atr_slow
        else:
            self.vol.htf_vf = 1.0
        
        self.vol.vol_factor = self.vol.session_vf * self.vol.htf_vf
        
        if self.vol.vol_factor < self.preset['low_vol_threshold']:
            self.vol.vol_state = "LOW"
        elif self.vol.vol_factor >= self.preset['extreme_vol_threshold']:
            self.vol.vol_state = "EXTREME"
        elif self.vol.vol_factor >= self.preset['high_vol_threshold']:
            self.vol.vol_state = "HIGH"
        else:
            self.vol.vol_state = "NORMAL"
    
    def check_breakout(self, bar: pd.Series) -> Tuple[bool, bool]:
        if not self.orb.orb_complete:
            return False, False
        
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
            
            new_long = (open_in_range and bar['close'] > self.orb.session_high and 
                       sufficient_high and strong_body)
            new_short = (open_in_range and bar['close'] < self.orb.session_low and 
                        sufficient_low and strong_body)
        else:
            new_long = bar['close'] > self.orb.session_high and sufficient_high and strong_body
            new_short = bar['close'] < self.orb.session_low and sufficient_low and strong_body
        
        return new_long, new_short
    
    def calc_stops(self, entry: float, atr: float, vwap: float, is_long: bool, i: int) -> dict:
        min_stop_dist = atr * self.MIN_STOP_ATR
        
        atr_stop = entry - (atr * self.ATR_STOP_MULT) if is_long else entry + (atr * self.ATR_STOP_MULT)
        swing_stop = self._calc_swing_stop(i, is_long, atr)
        vwap_stop = vwap - (atr * self.VWAP_STOP_DISTANCE) if is_long else vwap + (atr * self.VWAP_STOP_DISTANCE)
        
        if is_long:
            swing_valid = swing_stop < entry - min_stop_dist
            vwap_valid = (vwap < entry) and (vwap_stop < entry - min_stop_dist)
        else:
            swing_valid = swing_stop > entry + min_stop_dist
            vwap_valid = (vwap > entry) and (vwap_stop > entry + min_stop_dist)
        
        if not swing_valid:
            swing_stop = atr_stop
        if not vwap_valid:
            vwap_stop = atr_stop
        
        return {
            'atr_stop': atr_stop, 'swing_stop': swing_stop, 'vwap_stop': vwap_stop,
            'swing_valid': swing_valid, 'vwap_valid': vwap_valid
        }
    
    def _calc_swing_stop(self, i: int, is_long: bool, atr: float) -> float:
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
            swing_val = self.day_df['low'].iloc[max(0, i-1)] if is_long else self.day_df['high'].iloc[max(0, i-1)]
        
        return swing_val - (atr * self.SWING_BUFFER) if is_long else swing_val + (atr * self.SWING_BUFFER)
    
    def calc_achievable_rr(self, entry: float, stop: float, atr: float, rr_desired: float) -> float:
        risk = abs(entry - stop)
        if risk <= 0:
            return -1
        target_distance = risk * rr_desired
        target_atrs = target_distance / atr
        if target_atrs > self.MAX_TARGET_ATR:
            return (self.MAX_TARGET_ATR * atr) / risk
        return rr_desired
    
    def select_best_stop(self, stops: dict, entry: float, atr: float, rr_desired: float, is_long: bool) -> Tuple[float, str, float]:
        candidates = []
        for stop_type in ['atr', 'swing', 'vwap']:
            stop_price = stops[f'{stop_type}_stop']
            valid = stops.get(f'{stop_type}_valid', True) if stop_type != 'atr' else True
            if not valid:
                continue
            rr = self.calc_achievable_rr(entry, stop_price, atr, rr_desired)
            if rr > 0:
                candidates.append((stop_price, stop_type.upper(), rr))
        
        if not candidates:
            rr = self.calc_achievable_rr(entry, stops['atr_stop'], atr, rr_desired)
            return stops['atr_stop'], 'ATR', rr
        
        return max(candidates, key=lambda x: x[2])
    
    def can_trigger(self) -> bool:
        return self.orb.bars_since_last_trigger >= self.orb.min_bars_between_triggers
    
    def in_position(self) -> bool:
        return self.position.direction != ""
    
    # ═══════════════════════════════════════════════════════════════════
    # EXIT LOGIC
    # ═══════════════════════════════════════════════════════════════════
    
    def check_stop_hit(self, bar: pd.Series) -> bool:
        """Check if current stop was hit"""
        if not self.in_position():
            return False
        
        if self.position.direction == "LONG":
            return bar['low'] <= self.position.current_stop
        else:  # SHORT
            return bar['high'] >= self.position.current_stop
    
    def check_break_even(self, bar: pd.Series) -> bool:
        """Check if break-even should be activated"""
        if not self.in_position() or self.position.stop_moved_to_be:
            return False
        
        be_target = self.position.risk * self.BREAK_EVEN_RR
        
        if self.position.direction == "LONG":
            profit = bar['close'] - self.position.entry_price
            if profit >= be_target:
                return True
        else:  # SHORT
            profit = self.position.entry_price - bar['close']
            if profit >= be_target:
                return True
        
        return False
    
    def activate_break_even(self, bar: pd.Series, time_str: str):
        """Move stop to entry price"""
        self.position.stop_moved_to_be = True
        self.position.current_stop = self.position.entry_price
        print(f"[{time_str}]    🟣 BREAK-EVEN activated - stop moved to ${self.position.entry_price:.2f}")
    
    def check_trailing_activation(self, bar: pd.Series, atr: float) -> bool:
        """Check if trailing stop should activate (at profitTargetRR)"""
        if not self.in_position() or not self.position.stop_moved_to_be or self.position.trailing_activated:
            return False
        
        profit_target = self.position.risk * self.PROFIT_TARGET_RR
        
        if self.position.direction == "LONG":
            profit = bar['close'] - self.position.entry_price
            return profit >= profit_target
        else:  # SHORT
            profit = self.position.entry_price - bar['close']
            return profit >= profit_target
    
    def activate_trailing(self, bar: pd.Series, atr: float, time_str: str):
        """Activate trailing stop"""
        self.position.trailing_activated = True
        
        if self.position.direction == "LONG":
            self.position.trailing_stop_level = bar['close'] - (atr * self.TRAILING_STOP_DISTANCE)
        else:  # SHORT
            self.position.trailing_stop_level = bar['close'] + (atr * self.TRAILING_STOP_DISTANCE)
        
        self.position.current_stop = self.position.trailing_stop_level
        print(f"[{time_str}]    🔵 TRAILING activated at {self.PROFIT_TARGET_RR}R - trail @ ${self.position.trailing_stop_level:.2f}")
    
    def update_trailing_stop(self, bar: pd.Series, atr: float, ema9: float):
        """Update trailing stop (ratchet only, never loosen)"""
        if not self.position.trailing_activated:
            return
        
        # Check for aggressive trailing near EMA
        trail_dist = self.TRAILING_STOP_DISTANCE
        if self.USE_AGGRESSIVE_TRAILING:
            dist_to_ema = abs(bar['close'] - ema9)
            if dist_to_ema <= (atr * self.EMA_TIGHTEN_ZONE):
                trail_dist = self.TIGHTENED_TRAIL_DISTANCE
        
        if self.position.direction == "LONG":
            new_trail = bar['close'] - (atr * trail_dist)
            # Only move UP (tighter)
            if new_trail > self.position.trailing_stop_level:
                self.position.trailing_stop_level = new_trail
            # Trailing supersedes BE
            self.position.current_stop = max(self.position.trailing_stop_level, self.position.entry_price)
        else:  # SHORT
            new_trail = bar['close'] + (atr * trail_dist)
            # Only move DOWN (tighter)
            if new_trail < self.position.trailing_stop_level:
                self.position.trailing_stop_level = new_trail
            # Trailing supersedes BE
            self.position.current_stop = min(self.position.trailing_stop_level, self.position.entry_price)
    
    def check_ema_exit(self, bar: pd.Series, ema9: float) -> bool:
        """Check for EMA exit (only when NOT trailing and in profit)"""
        if not self.in_position():
            return False
        if not self.position.stop_moved_to_be:
            return False
        if self.position.trailing_activated:
            return False  # EMA exit disabled when trailing
        
        if self.position.direction == "LONG":
            # Must be in profit
            if bar['close'] <= self.position.entry_price:
                return False
            # Check if below EMA
            if bar['close'] < ema9:
                self.position.bars_beyond_ema += 1
            else:
                self.position.bars_beyond_ema = 0
        else:  # SHORT
            # Must be in profit
            if bar['close'] >= self.position.entry_price:
                return False
            # Check if above EMA
            if bar['close'] > ema9:
                self.position.bars_beyond_ema += 1
            else:
                self.position.bars_beyond_ema = 0
        
        return self.position.bars_beyond_ema >= self.EMA_CONFIRMATION_BARS
    
    def close_position(self, exit_price: float, exit_reason: str, bar_idx: int, time_str: str):
        """Close position and record trade"""
        if self.position.direction == "LONG":
            pnl = exit_price - self.position.entry_price
        else:
            pnl = self.position.entry_price - exit_price
        
        r_multiple = pnl / self.position.risk if self.position.risk > 0 else 0
        
        trade = TradeResult(
            direction=self.position.direction,
            entry_price=self.position.entry_price,
            entry_time=self.position.entry_time,
            exit_price=exit_price,
            exit_time=time_str,
            exit_reason=exit_reason,
            initial_stop=self.position.initial_stop,
            risk=self.position.risk,
            pnl=pnl,
            r_multiple=r_multiple
        )
        self.trades.append(trade)
        
        emoji = "✅" if pnl > 0 else "❌"
        print(f"\n[{time_str}]    {emoji} POSITION CLOSED - {exit_reason}")
        print(f"         Entry: ${self.position.entry_price:.2f}")
        print(f"         Exit:  ${exit_price:.2f}")
        print(f"         P/L:   ${pnl:.2f} ({r_multiple:+.2f}R)")
        
        self.position.reset()
    
    def run(self):
        """Run the tracer with full exit management"""
        print(f"\n{'='*70}")
        print(f"  SINGLE DAY TRACER v4: {self.symbol} on {self.target_date}")
        print(f"  (Pine v69 with EXIT LOGIC)")
        print(f"{'='*70}\n")
        
        self.load_data()
        self.calc_indicators()
        self.orb.reset_session()
        
        print(f"Exit Parameters:")
        print(f"  Break-even: {self.BREAK_EVEN_RR}R")
        print(f"  Trailing activates: {self.PROFIT_TARGET_RR}R")
        print(f"  Trail distance: {self.TRAILING_STOP_DISTANCE} ATR")
        print(f"  EMA confirmation: {self.EMA_CONFIRMATION_BARS} bar(s)")
        
        print(f"\n{'─'*70}")
        print(f"  BAR-BY-BAR TRACE")
        print(f"{'─'*70}\n")
        
        bar_idx = 0
        for i in range(len(self.day_df)):
            ts = self.day_df.index[i]
            bar = self.day_df.iloc[i]
            
            if not self.is_rth(ts):
                continue
            
            time_str = ts.strftime('%H:%M')
            bar_idx += 1
            
            self.orb.bars_since_last_trigger += 1
            
            atr = bar['atr_rth']
            ema9 = bar['ema9']
            
            # ═══════════════════════════════════════════════════════════
            # EXIT CHECKS (before entry logic)
            # ═══════════════════════════════════════════════════════════
            
            if self.in_position():
                # 1. Check stop hit
                if self.check_stop_hit(bar):
                    exit_price = self.position.current_stop
                    reason = "TRAILING STOP" if self.position.trailing_activated else \
                             "BREAK-EVEN STOP" if self.position.stop_moved_to_be else "INITIAL STOP"
                    self.close_position(exit_price, reason, bar_idx, time_str)
                
                # If still in position, check exit management
                elif self.in_position():
                    # 2. Check break-even
                    if self.check_break_even(bar):
                        self.activate_break_even(bar, time_str)
                    
                    # 3. Check trailing activation
                    if self.check_trailing_activation(bar, atr):
                        self.activate_trailing(bar, atr, time_str)
                    
                    # 4. Update trailing stop
                    if self.position.trailing_activated:
                        old_stop = self.position.current_stop
                        self.update_trailing_stop(bar, atr, ema9)
                        if self.position.current_stop != old_stop:
                            print(f"[{time_str}]    📈 Trail updated: ${old_stop:.2f} → ${self.position.current_stop:.2f}")
                    
                    # 5. Check EMA exit (only if not trailing)
                    if self.check_ema_exit(bar, ema9):
                        self.close_position(bar['close'], "EMA EXIT", bar_idx, time_str)
            
            # ═══════════════════════════════════════════════════════════
            # ORB BUILDING
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
                continue
            
            # ORB complete
            if not self.orb.orb_complete and not pd.isna(self.orb.session_high):
                self.orb.orb_complete = True
                print(f"\n[{time_str}] 🟢 ORB COMPLETE - H: {self.orb.session_high:.2f}, L: {self.orb.session_low:.2f}, ATR: ${atr:.4f}\n")
            
            # Vol state
            self.calc_vol_state(i)
            
            # ═══════════════════════════════════════════════════════════
            # BREAKOUT DETECTION
            # ═══════════════════════════════════════════════════════════
            
            new_long, new_short = self.check_breakout(bar)
            
            # Latch breakouts
            if new_long and not self.orb.has_broken_out_high:
                self.orb.long_breakout_pending = True
                self.orb.has_broken_out_high = True
            
            if new_short and not self.orb.has_broken_out_low:
                self.orb.short_breakout_pending = True
                self.orb.has_broken_out_low = True
            
            # ═══════════════════════════════════════════════════════════
            # ENTRY EVALUATION (if not in position)
            # ═══════════════════════════════════════════════════════════
            
            if not self.in_position() and self.is_trading_window(ts) and self.can_trigger():
                
                if self.orb.long_breakout_pending:
                    entry = bar['close']
                    vwap = bar['vwap']
                    
                    rr_desired = self.PROFIT_TARGET_RR
                    if self.vol.vol_state == "LOW":
                        rr_desired += 0.5
                    elif self.vol.vol_state == "HIGH":
                        rr_desired = max(rr_desired - 0.5, 1.2)
                    elif self.vol.vol_state == "EXTREME":
                        rr_desired = max(rr_desired - 0.8, 1.0)
                    
                    stops = self.calc_stops(entry, atr, vwap, True, i)
                    stop_price, stop_type, achievable_rr = self.select_best_stop(stops, entry, atr, rr_desired, True)
                    
                    print(f"[{time_str}] 🔺 LONG BREAKOUT - Entry: ${entry:.2f}, Stop: ${stop_price:.2f}, R:R: {achievable_rr:.2f}, Vol: {self.vol.vol_state}")
                    
                    if achievable_rr >= self.MIN_ACCEPTABLE_RR - 0.0001:
                        risk = abs(entry - stop_price)
                        self.position.direction = "LONG"
                        self.position.entry_price = entry
                        self.position.entry_bar = bar_idx
                        self.position.entry_time = time_str
                        self.position.initial_stop = stop_price
                        self.position.current_stop = stop_price
                        self.position.stop_type = stop_type
                        self.position.risk = risk
                        self.position.target_rr = achievable_rr
                        
                        self.orb.bars_since_last_trigger = 0
                        print(f"         ✅ TRADE TAKEN - Risk: ${risk:.2f}")
                    else:
                        print(f"         ❌ SKIP: {achievable_rr:.2f} < {self.MIN_ACCEPTABLE_RR}")
                        self.orb.bars_since_last_trigger = 0
                
                if self.orb.short_breakout_pending and not self.in_position():
                    entry = bar['close']
                    vwap = bar['vwap']
                    
                    rr_desired = self.PROFIT_TARGET_RR
                    if self.vol.vol_state == "LOW":
                        rr_desired += 0.5
                    elif self.vol.vol_state == "HIGH":
                        rr_desired = max(rr_desired - 0.5, 1.2)
                    elif self.vol.vol_state == "EXTREME":
                        rr_desired = max(rr_desired - 0.8, 1.0)
                    
                    stops = self.calc_stops(entry, atr, vwap, False, i)
                    stop_price, stop_type, achievable_rr = self.select_best_stop(stops, entry, atr, rr_desired, False)
                    
                    print(f"[{time_str}] 🔻 SHORT BREAKOUT - Entry: ${entry:.2f}, Stop: ${stop_price:.2f}, R:R: {achievable_rr:.2f}, Vol: {self.vol.vol_state}")
                    
                    if achievable_rr >= self.MIN_ACCEPTABLE_RR - 0.0001:
                        risk = abs(entry - stop_price)
                        self.position.direction = "SHORT"
                        self.position.entry_price = entry
                        self.position.entry_bar = bar_idx
                        self.position.entry_time = time_str
                        self.position.initial_stop = stop_price
                        self.position.current_stop = stop_price
                        self.position.stop_type = stop_type
                        self.position.risk = risk
                        self.position.target_rr = achievable_rr
                        
                        self.orb.bars_since_last_trigger = 0
                        print(f"         ✅ TRADE TAKEN - Risk: ${risk:.2f}")
                    else:
                        print(f"         ❌ SKIP: {achievable_rr:.2f} < {self.MIN_ACCEPTABLE_RR}")
                        self.orb.bars_since_last_trigger = 0
            
            # ═══════════════════════════════════════════════════════════
            # STATE MACHINE UPDATES
            # ═══════════════════════════════════════════════════════════
            
            if self.orb.orb_complete:
                back_inside = (bar['close'] <= self.orb.session_high and 
                              bar['close'] >= self.orb.session_low)
                if back_inside:
                    if self.orb.has_broken_out_high:
                        self.orb.has_broken_out_high = False
                    if self.orb.has_broken_out_low:
                        self.orb.has_broken_out_low = False
            
            self.orb.long_breakout_pending = False
            self.orb.short_breakout_pending = False
        
        # ═══════════════════════════════════════════════════════════
        # END OF DAY - Close any open position
        # ═══════════════════════════════════════════════════════════
        
        if self.in_position():
            last_bar = self.day_df.iloc[-1]
            self.close_position(last_bar['close'], "END OF DAY", len(self.day_df), "16:00")
        
        # ═══════════════════════════════════════════════════════════
        # SUMMARY
        # ═══════════════════════════════════════════════════════════
        
        print(f"\n{'='*70}")
        print(f"  TRADE SUMMARY")
        print(f"{'='*70}\n")
        
        if not self.trades:
            print("No trades taken.")
        else:
            total_pnl = sum(t.pnl for t in self.trades)
            wins = sum(1 for t in self.trades if t.pnl > 0)
            losses = sum(1 for t in self.trades if t.pnl <= 0)
            total_r = sum(t.r_multiple for t in self.trades)
            
            for i, t in enumerate(self.trades, 1):
                emoji = "✅" if t.pnl > 0 else "❌"
                print(f"  {i}. {emoji} {t.direction} {t.entry_time}→{t.exit_time}: ${t.pnl:+.2f} ({t.r_multiple:+.2f}R) - {t.exit_reason}")
            
            print(f"\n  Total P/L: ${total_pnl:+.2f} ({total_r:+.2f}R)")
            print(f"  Record: {wins}W / {losses}L")
        
        print(f"\n{'='*70}\n")


if __name__ == '__main__':
    tracer = SingleDayTracerV4(symbol='TSLA', date='2025-12-03')
    tracer.run()
