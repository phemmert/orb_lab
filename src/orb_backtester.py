"""
ORB Backtester - Pine v76 Compatible (Confluence Enabled)
=========================================================
Full backtester built on validated v4 tracer logic.
NOW WITH CONFLUENCE SCORING - SSL, WAE, QQE, Volume filters.

v76 FIX (CRITICAL - January 27, 2026):
- ATR GAP CAPTURE: Pine's RTH-only ATR calculation captures overnight gaps.
  At 09:30, prev_close = previous session's 15:59 close (not premarket close).
  This creates high TR values after gap days, which feeds into Wilder RMA.
  Previous code HIDED gaps by using open at session start - WRONG.
- Removed unnecessary ATR shift(1) - Pine uses current bar ATR for breakout.
- STOP SELECTION TIE-BREAKER: When multiple stops achieve same R:R, Pine 
  selects the FIRST in order (ATRâ†’Swingâ†’VWAPâ†’Hybrid). Python was using max()
  which returned the LAST. Now uses explicit first-match logic.
- ATR_FAST_SMA FIX: Was set to raw atr_fast (no SMA). Pine uses ta.sma(orbATR_fast, 10).
  This affects vol_state baseline seeding â†’ wrong vol_state â†’ wrong rr_desired â†’ wrong stop selection.

v75 FIXES (CRITICAL - January 25, 2026):
1. EXIT LOGIC ORDER: Stop hit checked FIRST with PREVIOUS bar's stop value,
   THEN BE/trailing updates for NEXT bar. Matches Pine's strategy.exit() behavior.
2. SINGLE-BAR PENDING: Breakout pending flags enforced as single-bar only.
   - Entry requires BOTH breakout_pending AND pending_this_bar to be True
   - Prevents ghost trades from stale pending flags carrying across bars
   - Uses pending_this_bar pattern for explicit intent tracking

v74 FIXES:
- Trailing activation uses bar CLOSE - matches Pine exactly
- Skip tracking: logs trades skipped due to R:R below minimum
- Volume Z-score normalization support (update confluence_indicators.py)

v73 FIX (superseded by v74):
- BE check now uses bar high/low (not close) to detect if BE target was touched
  ^ THIS WAS WRONG - Pine uses CLOSE, not high/low

Features:
- Multi-day backtesting across date range
- All parameters configurable (Optuna-ready)
- CONFLUENCE SCORING matching Pine indicator logic
- Skip tracking with full details
- Trade log with full details
- Aggregate statistics for optimization

Usage:
    from orb_backtester import ORBBacktester
    
    bt = ORBBacktester(
        symbol='AMD',
        start_date='2025-10-01',
        end_date='2025-12-31',
        min_confluence=5  # Require all indicators to agree
    )
    results = bt.run()
    print(results['summary'])
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r'C:\Users\phemm\orb_lab\src')

# Import confluence indicators
try:
    from confluence_indicators import ConfluenceCalculator, ConfluenceScores, pine_round
except ImportError:
    # Fallback if not in path
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from confluence_indicators import ConfluenceCalculator, ConfluenceScores, pine_round


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SYMBOL PRESETS - Matching Pine v69
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

SYMBOL_PRESETS = {
    'AMD': {
        # SSL
        'ssl_baseline_length': 40,
        'ssl_length': 10,
        # WAE
        'wae_fast_ema': 13,
        'wae_slow_ema': 26,
        'wae_bb_length': 20,
        'wae_sensitivity': 200,
        # QQE
        'qqe_rsi1_length': 8,
        'qqe_rsi1_smoothing': 5,
        'qqe_rsi2_length': 4,
        'qqe_rsi2_smoothing': 4,
        'qqe_bb_length': 25,
        # Volume
        'vol_lookback': 12,
        # Vol thresholds
        'low_vol_threshold': 0.8,
        'high_vol_threshold': 1.3,
        'extreme_vol_threshold': 2.0,
    },
    'GOOGL': {
        # SSL
        'ssl_baseline_length': 50,
        'ssl_length': 10,
        # WAE
        'wae_fast_ema': 15,
        'wae_slow_ema': 30,
        'wae_bb_length': 20,
        'wae_sensitivity': 325,
        # QQE
        'qqe_rsi1_length': 8,
        'qqe_rsi1_smoothing': 5,
        'qqe_rsi2_length': 4,
        'qqe_rsi2_smoothing': 4,
        'qqe_bb_length': 25,
        # Volume
        'vol_lookback': 12,
        # Vol thresholds
        'low_vol_threshold': 0.8,
        'high_vol_threshold': 1.3,
        'extreme_vol_threshold': 2.0,
    },
}

# Default preset for unknown symbols
DEFAULT_PRESET = {
    'ssl_baseline_length': 45,
    'ssl_length': 5,
    'wae_fast_ema': 20,
    'wae_slow_ema': 40,
    'wae_bb_length': 20,
    'wae_sensitivity': 190,
    'qqe_rsi1_length': 6,
    'qqe_rsi1_smoothing': 5,
    'qqe_rsi2_length': 6,
    'qqe_rsi2_smoothing': 5,
    'qqe_bb_length': 50,
    'vol_lookback': 5,
    'low_vol_threshold': 0.8,
    'high_vol_threshold': 1.3,
    'extreme_vol_threshold': 2.0,
}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DATA CLASSES - Same as v4 tracer
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@dataclass
class ORBState:
    """ORB range and breakout state machine"""
    session_high: float = np.nan
    session_low: float = np.nan
    orb_complete: bool = False
    long_breakout_pending: bool = False
    short_breakout_pending: bool = False
    has_broken_out_high: bool = False
    has_broken_out_low: bool = False
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
    """Active position with exit management"""
    direction: str = ""
    entry_price: float = np.nan
    entry_time: str = ""
    entry_date: str = ""
    initial_stop: float = np.nan
    current_stop: float = np.nan
    stop_type: str = ""
    risk: float = np.nan
    target_rr: float = np.nan
    vol_state: str = ""
    
    # Confluence tracking (v70)
    confluence_score: int = 0
    ssl_score: float = 0.0
    wae_score: float = 0.0
    qqe_score: float = 0.0
    vol_score: float = 0.0
    
    # Exit state
    stop_moved_to_be: bool = False
    trailing_activated: bool = False
    trailing_stop_level: float = np.nan
    bars_beyond_ema: int = 0
    
    def reset(self):
        self.direction = ""
        self.entry_price = np.nan
        self.entry_time = ""
        self.entry_date = ""
        self.initial_stop = np.nan
        self.current_stop = np.nan
        self.stop_type = ""
        self.risk = np.nan
        self.target_rr = np.nan
        self.vol_state = ""
        self.confluence_score = 0
        self.ssl_score = 0.0
        self.wae_score = 0.0
        self.qqe_score = 0.0
        self.vol_score = 0.0
        self.stop_moved_to_be = False
        self.trailing_activated = False
        self.trailing_stop_level = np.nan
        self.bars_beyond_ema = 0


@dataclass
class TradeRecord:
    """Completed trade record"""
    date: str
    direction: str
    entry_price: float
    entry_time: str
    exit_price: float
    exit_time: str
    exit_reason: str
    initial_stop: float
    stop_type: str
    vol_state: str
    risk: float
    pnl: float
    r_multiple: float
    # Confluence tracking
    confluence_score: int = 0
    ssl_score: float = 0.0
    wae_score: float = 0.0
    qqe_score: float = 0.0
    vol_score: float = 0.0


@dataclass
class SkipRecord:
    """Skipped trade record - R:R didn't meet minimum"""
    date: str
    time: str
    direction: str
    entry_price: float
    stop_price: float
    stop_type: str
    achievable_rr: float
    min_rr: float
    reason: str
    # Confluence at skip time
    confluence_score: int = 0
    ssl_score: float = 0.0
    wae_score: float = 0.0
    qqe_score: float = 0.0
    vol_score: float = 0.0


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN BACKTESTER CLASS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class ORBBacktester:
    """
    Full ORB backtester with Pine v70 compatible logic.
    All parameters configurable for Optuna optimization.
    INCLUDES CONFLUENCE SCORING - SSL, WAE, QQE, Volume.
    """
    
    def __init__(
        self,
        symbol: str = 'AMD',
        start_date: str = '2025-10-01',
        end_date: str = '2025-12-31',
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # ENTRY PARAMETERS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        orb_minutes: int = 5,
        breakout_threshold_mult: float = 0.1,
        min_body_strength: float = 0.5,
        require_open_in_range: bool = True,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        swing_lookback: int = 5,
        swing_buffer: float = 0.02,
        vwap_stop_distance: float = 0.3,
        min_stop_atr: float = 0.15,
        max_target_atr: float = 3.0,
        min_acceptable_rr: float = 1.5,
        profit_target_rr: float = 2.0,
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # CONFLUENCE PARAMETERS (NEW - v70)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        enable_confluence: bool = True,
        min_confluence: int = 5,
        confluence_mode: str = 'all',  # 'all', 'ssl', 'wae', 'qqe', 'vol'
        
        # SSL Hybrid
        ssl_baseline_length: int = 45,
        ssl_length: int = 5,
        ssl_type: str = 'JMA',
        use_ssl_momentum: bool = True,
        
        # WAE
        wae_fast_ema: int = 20,
        wae_slow_ema: int = 40,
        wae_sensitivity: int = 190,
        wae_bb_length: int = 20,
        wae_bb_mult: float = 2.0,
        use_wae_acceleration: bool = True,
        
        # QQE
        qqe_rsi1_length: int = 6,
        qqe_rsi1_smoothing: int = 5,
        qqe_factor_primary: float = 3.0,
        qqe_rsi2_length: int = 6,
        qqe_rsi2_smoothing: int = 5,
        qqe_factor_secondary: float = 1.61,
        qqe_bb_length: int = 50,
        qqe_bb_mult: float = 0.35,
        qqe_threshold: int = 3,
        qqe_consecutive_bars: int = 3,
        use_qqe_momentum: bool = True,
        
        # Volume
        vol_lookback: int = 5,
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # EXIT PARAMETERS (the "dials" for Optuna)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        use_break_even: bool = True,
        break_even_rr: float = 0.5,
        use_adaptive_be: bool = False,
        use_trailing_stop: bool = True,
        trailing_stop_distance: float = 1.2,
        use_ema_exit: bool = True,
        ema_period: int = 9,
        ema_confirmation_bars: int = 1,
        use_aggressive_trailing: bool = True,
        ema_tighten_zone: float = 0.3,
        tightened_trail_distance: float = 0.3,
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # VOLATILITY THRESHOLDS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        low_vol_threshold: float = 0.8,
        high_vol_threshold: float = 1.3,
        extreme_vol_threshold: float = 2.0,
        skip_low_vol_c_grades: bool = True,
        low_vol_min_rr: float = 2.0,
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # TRADING WINDOW
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        trading_start_minutes: int = 570,  # 9:30 = 9*60+30
        trading_end_minutes: int = 630,    # 10:30 = 10*60+30
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # OUTPUT CONTROL
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        verbose: bool = False,  # Set True for debugging
    ):
        # Store all parameters
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
        # Entry params
        self.orb_minutes = orb_minutes
        self.breakout_threshold_mult = breakout_threshold_mult
        self.min_body_strength = min_body_strength
        self.require_open_in_range = require_open_in_range
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.swing_lookback = swing_lookback
        self.swing_buffer = swing_buffer
        self.vwap_stop_distance = vwap_stop_distance
        self.min_stop_atr = min_stop_atr
        self.max_target_atr = max_target_atr
        self.min_acceptable_rr = min_acceptable_rr
        self.profit_target_rr = profit_target_rr
        
        # Confluence params (v70) - AUTO-LOAD FROM SYMBOL PRESETS
        self.enable_confluence = enable_confluence
        self.min_confluence = min_confluence
        self.confluence_mode = confluence_mode  # 'all', 'ssl', 'wae', 'qqe', 'vol'
        
        # Get symbol preset (or default)
        preset = SYMBOL_PRESETS.get(symbol, DEFAULT_PRESET)
        
        # SSL params - use preset if param matches generic default
        self.ssl_baseline_length = preset['ssl_baseline_length'] if ssl_baseline_length == 45 else ssl_baseline_length
        self.ssl_length = preset['ssl_length'] if ssl_length == 5 else ssl_length
        self.ssl_type = ssl_type
        self.use_ssl_momentum = use_ssl_momentum
        
        # WAE params - use preset if param matches generic default
        self.wae_fast_ema = preset['wae_fast_ema'] if wae_fast_ema == 20 else wae_fast_ema
        self.wae_slow_ema = preset['wae_slow_ema'] if wae_slow_ema == 40 else wae_slow_ema
        self.wae_sensitivity = preset['wae_sensitivity'] if wae_sensitivity == 190 else wae_sensitivity
        self.wae_bb_length = preset['wae_bb_length'] if wae_bb_length == 20 else wae_bb_length
        self.wae_bb_mult = wae_bb_mult
        self.use_wae_acceleration = use_wae_acceleration
        
        # QQE params - use preset if param matches generic default
        self.qqe_rsi1_length = preset['qqe_rsi1_length'] if qqe_rsi1_length == 6 else qqe_rsi1_length
        self.qqe_rsi1_smoothing = preset['qqe_rsi1_smoothing'] if qqe_rsi1_smoothing == 5 else qqe_rsi1_smoothing
        self.qqe_factor_primary = qqe_factor_primary
        self.qqe_rsi2_length = preset['qqe_rsi2_length'] if qqe_rsi2_length == 6 else qqe_rsi2_length
        self.qqe_rsi2_smoothing = preset['qqe_rsi2_smoothing'] if qqe_rsi2_smoothing == 5 else qqe_rsi2_smoothing
        self.qqe_factor_secondary = qqe_factor_secondary
        self.qqe_bb_length = preset['qqe_bb_length'] if qqe_bb_length == 50 else qqe_bb_length
        self.qqe_bb_mult = qqe_bb_mult
        self.qqe_threshold = qqe_threshold
        self.qqe_consecutive_bars = qqe_consecutive_bars
        self.use_qqe_momentum = use_qqe_momentum
        
        # Volume params - use preset if param matches generic default
        self.vol_lookback = preset['vol_lookback'] if vol_lookback == 5 else vol_lookback
        
        # Vol thresholds - use preset
        self.low_vol_threshold = preset['low_vol_threshold'] if low_vol_threshold == 0.8 else low_vol_threshold
        self.high_vol_threshold = preset['high_vol_threshold'] if high_vol_threshold == 1.3 else high_vol_threshold
        self.extreme_vol_threshold = preset['extreme_vol_threshold'] if extreme_vol_threshold == 2.0 else extreme_vol_threshold
        self.skip_low_vol_c_grades = skip_low_vol_c_grades
        self.low_vol_min_rr = low_vol_min_rr
        
        # Exit params
        self.use_break_even = use_break_even
        self.break_even_rr = break_even_rr
        self.use_adaptive_be = use_adaptive_be
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_distance = trailing_stop_distance
        self.use_ema_exit = use_ema_exit
        self.ema_period = ema_period
        self.ema_confirmation_bars = ema_confirmation_bars
        self.use_aggressive_trailing = use_aggressive_trailing
        self.ema_tighten_zone = ema_tighten_zone
        self.tightened_trail_distance = tightened_trail_distance
        
        # Trading window
        self.trading_start_minutes = trading_start_minutes
        self.trading_end_minutes = trading_end_minutes
        
        # Output
        self.verbose = verbose
        
        if self.verbose:
            print(f"[Presets] Loaded {symbol}: SSL_base={self.ssl_baseline_length}, WAE_sens={self.wae_sensitivity}, QQE_rsi1={self.qqe_rsi1_length}")
        
        # State objects
        self.orb = ORBState()
        self.vol = VolState()
        self.position = Position()
        
        # Confluence calculator (v70)
        self.confluence_calc = ConfluenceCalculator({
            'ssl_baseline_length': self.ssl_baseline_length,
            'ssl_length': self.ssl_length,
            'ssl_type': self.ssl_type,
            'use_ssl_momentum': self.use_ssl_momentum,
            'wae_fast_ema': self.wae_fast_ema,
            'wae_slow_ema': self.wae_slow_ema,
            'wae_sensitivity': self.wae_sensitivity,
            'wae_bb_length': self.wae_bb_length,
            'wae_bb_mult': self.wae_bb_mult,
            'use_wae_acceleration': self.use_wae_acceleration,
            'qqe_rsi1_length': self.qqe_rsi1_length,
            'qqe_rsi1_smoothing': self.qqe_rsi1_smoothing,
            'qqe_factor_primary': self.qqe_factor_primary,
            'qqe_rsi2_length': self.qqe_rsi2_length,
            'qqe_rsi2_smoothing': self.qqe_rsi2_smoothing,
            'qqe_factor_secondary': self.qqe_factor_secondary,
            'qqe_bb_length': self.qqe_bb_length,
            'qqe_bb_mult': self.qqe_bb_mult,
            'qqe_threshold': self.qqe_threshold,
            'qqe_consecutive_bars': self.qqe_consecutive_bars,
            'use_qqe_momentum': self.use_qqe_momentum,
            'vol_lookback': self.vol_lookback,
            'min_confluence': self.min_confluence,
        })
        
        # Results
        self.trades: List[TradeRecord] = []
        self.skips: List[SkipRecord] = []
        self.df = None
        self.daily_atr_map = {}
        
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # DATA LOADING & INDICATOR CALCULATION
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    def load_data(self):
        """Load data and calculate all indicators"""
        from data_collector import PolygonDataCollector
        
        collector = PolygonDataCollector()
        self.df = collector.fetch_bars(self.symbol, days_back=180, bar_size=1, extended_hours=True)
        
        if self.verbose:
            print(f"âœ“ Loaded {len(self.df)} bars for {self.symbol}")
        
        self._calc_indicators()
    
    def _calc_indicators(self):
        """Calculate all indicators - same as v4 tracer"""
        df = self.df.copy()
        
        # TR for ETH
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ETH ATR (fallback)
        df['atr_eth'] = df['tr'].ewm(alpha=1/self.atr_period, adjust=False).mean()
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # ALL-BAR ATR for vol_state (Pine: orbATR_fast = ta.atr(5) on ALL bars)
        # 
        # CRITICAL: Pine captures the RTH GAP at 09:30!
        # Even on extended hours charts, Pine's ta.atr() at 09:30 uses the
        # previous RTH close (not the premarket close) for TR calculation.
        # This captures overnight/weekend gaps that drive vol_state to EXTREME.
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        
        # Create TR column that captures RTH gaps at 09:30
        df['tr_with_gap'] = df['tr'].copy()
        
        # Find previous RTH close for each 09:30 bar
        df['is_rth_close'] = (df.index.hour == 15) & (df.index.minute == 59)
        df['last_rth_close'] = df.loc[df['is_rth_close'], 'close']
        df['last_rth_close'] = df['last_rth_close'].ffill()
        
        # At 09:30 bars, recalculate TR using previous RTH close
        # DISABLED FOR TESTING - gap injection may overcook TR at 09:30
        # is_930 = (df.index.hour == 9) & (df.index.minute == 30)
        # for idx in df[is_930].index:
        #     prev_rth_close = df.loc[idx, 'last_rth_close']
        #     if pd.notna(prev_rth_close):
        #         h, l = df.loc[idx, 'high'], df.loc[idx, 'low']
        #         gap_tr = max(h - l, abs(h - prev_rth_close), abs(l - prev_rth_close))
        #         df.loc[idx, 'tr_with_gap'] = gap_tr
        
        # Now compute ATR(5) using gap-aware TR
        df['atr_fast_all'] = df['tr_with_gap'].ewm(alpha=1/5, adjust=False).mean()
        # SMA for vol_state baseline - runtime code will use prev bar at 09:30
        df['atr_fast_sma_all'] = df['atr_fast_all'].rolling(10).mean()
        df['atr_fast_sma_all'] = df['atr_fast_sma_all'].fillna(df['atr_fast_all'])
        
        # Cleanup temp columns
        df.drop(columns=['is_rth_close', 'last_rth_close', 'tr_with_gap'], inplace=True)
        
        # RTH mask
        df['is_rth'] = df.index.map(lambda x: 9*60+30 <= x.hour*60+x.minute < 16*60)
        
        # RTH-only calculations
        rth_df = df[df['is_rth']].copy()
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # PINE PARITY: True Range for RTH bars
        # Pine's request.security(rthTicker, ...) only sees RTH bars.
        # At 09:30, "previous close" = last RTH bar's close (15:59 yesterday)
        # This CAPTURES overnight gaps in the TR calculation.
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        
        # shift(1) on RTH-only dataframe automatically gives us:
        # - For 09:31+: previous RTH bar's close (correct)
        # - For 09:30: previous session's 15:59 close (CAPTURES THE GAP!)
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
        rth_df['atr'] = wilder_rma(rth_df['tr'], self.atr_period)
        rth_df['atr_fast'] = wilder_rma(rth_df['tr'], 5)
        # Pine uses: ta.sma(orbATR_fast, 10) for baseline seeding
        rth_df['atr_fast_sma'] = rth_df['atr_fast'].rolling(10).mean()
        
        # Map back to full df - NO SHIFT
        # Pine's breakoutThreshold = atr * 0.1 uses CURRENT bar's ATR
        df['atr_rth'] = np.nan
        df.loc[rth_df.index, 'atr_rth'] = rth_df['atr']
        df['atr_fast_rth'] = np.nan
        df.loc[rth_df.index, 'atr_fast_rth'] = rth_df['atr_fast']
        df['atr_fast_sma_rth'] = np.nan
        df.loc[rth_df.index, 'atr_fast_sma_rth'] = rth_df['atr_fast_sma']
        
        # Forward fill
        df['atr_rth'] = df['atr_rth'].ffill().fillna(df['atr_eth'])
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
        
        # VWAP (resets at 9:30)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']
        df['is_930'] = (df.index.hour == 9) & (df.index.minute == 30)
        df['session_id'] = df['is_930'].cumsum()
        df['cum_tp_vol'] = df.groupby('session_id')['tp_volume'].cumsum()
        df['cum_vol'] = df.groupby('session_id')['volume'].cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        
        # EMA
        df['ema9'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # CONFLUENCE INDICATORS (v70)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        if self.enable_confluence:
            if self.verbose:
                print("  [*] Calculating confluence indicators (SSL, WAE, QQE, Volume)...")
            df = self.confluence_calc.compute_indicators(df)
            if self.verbose:
                print("  [âœ“] Confluence indicators calculated")
        
        self.df = df
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # HELPER FUNCTIONS
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    def _is_orb_window(self, ts) -> bool:
        bar_minutes = ts.hour * 60 + ts.minute
        return 9*60+30 <= bar_minutes < 9*60+30 + self.orb_minutes
    
    def _is_trading_window(self, ts) -> bool:
        bar_minutes = ts.hour * 60 + ts.minute
        return self.trading_start_minutes <= bar_minutes < self.trading_end_minutes
    
    def _is_rth(self, ts) -> bool:
        bar_minutes = ts.hour * 60 + ts.minute
        return 9*60+30 <= bar_minutes < 16*60
    
    def _calc_vol_state(self, bar: pd.Series, ts) -> None:
        """
        Calculate volatility state for current bar.
        
        Pine (lines 1492-1516):
            orbATR_fast = ta.atr(5)              // On ALL bars (ETH+RTH)
            orbATR_slow = ta.sma(orbATR_fast, 10) // 10-bar SMA
            orbSessionBaseline seeded from orbATR_slow at 09:30
            sessionVF = orbATR_fast / orbSessionBaseline
        
        CRITICAL: At 09:30, Pine seeds baseline from the SMA BEFORE the gap spike.
        Python must use the PREVIOUS bar's SMA to match this behavior.
        """
        # Use RTH-only ATR values for vol_state (Pine runs on RTH chart)
        atr_fast = bar.get('atr_fast_rth', bar.get('atr_rth', 1.0))
        atr_fast_sma = bar.get('atr_fast_sma_rth', atr_fast)
        
        if pd.isna(atr_fast):
            atr_fast = bar.get('atr_rth', 1.0)
        if pd.isna(atr_fast_sma):
            atr_fast_sma = atr_fast
        
        bar_minutes = ts.hour * 60 + ts.minute
        is_orb_vol_window = 9*60+30 <= bar_minutes < 10*60
        
        # Reset baseline at start of day
        if ts.hour == 9 and ts.minute == 30:
            self.vol.orb_session_baseline = np.nan
        
        # Seed/update baseline during ORB window
        if is_orb_vol_window:
            if pd.isna(self.vol.orb_session_baseline):
                # On RTH chart, Pine seeds baseline with current bar's orbATR_slow
                # No previous-bar hack needed - RTH bars naturally include the gap
                self.vol.orb_session_baseline = atr_fast_sma
            else:
                # Pine: 0.9 * orbSessionBaseline + 0.1 * orbATR_slow
                self.vol.orb_session_baseline = 0.9 * self.vol.orb_session_baseline + 0.1 * atr_fast_sma
        
        # sessionVF = orbATR_fast / orbSessionBaseline
        if not pd.isna(self.vol.orb_session_baseline) and self.vol.orb_session_baseline > 0:
            self.vol.session_vf = atr_fast / self.vol.orb_session_baseline
        else:
            self.vol.session_vf = 1.0
        
        # HTF VF from daily ATR
        daily_atr = bar.get('daily_atr', 1.0)
        daily_atr_slow = bar.get('daily_atr_slow', daily_atr)
        if pd.isna(daily_atr):
            daily_atr = 1.0
        if pd.isna(daily_atr_slow) or daily_atr_slow <= 0:
            daily_atr_slow = daily_atr
        
        self.vol.htf_vf = daily_atr / daily_atr_slow if daily_atr_slow > 0 else 1.0
        self.vol.vol_factor = self.vol.session_vf * self.vol.htf_vf
        
        # Temporary deep VF debug
        if self.verbose and ts.hour == 9 and 35 <= ts.minute <= 45:
            daily_atr_val = bar.get('daily_atr', 0)
            daily_atr_slow_val = bar.get('daily_atr_slow', 0)
            print(f"    [VF DEBUG] {ts.strftime('%H:%M')} atr_fast={atr_fast:.4f} atr_fast_sma={atr_fast_sma:.4f} baseline={self.vol.orb_session_baseline:.4f} sessionVF={self.vol.session_vf:.4f} | daily_atr={daily_atr_val:.4f} daily_atr_slow={daily_atr_slow_val:.4f} htfVF={self.vol.htf_vf:.4f} | vf={self.vol.vol_factor:.4f} ({self.vol.vol_state})")
        
        if self.vol.vol_factor < self.low_vol_threshold:
            self.vol.vol_state = "LOW"
        elif self.vol.vol_factor >= self.extreme_vol_threshold:
            self.vol.vol_state = "EXTREME"
        elif self.vol.vol_factor >= self.high_vol_threshold:
            self.vol.vol_state = "HIGH"
        else:
            self.vol.vol_state = "NORMAL"
    
    def _check_breakout(self, bar: pd.Series) -> Tuple[bool, bool]:
        """Check for breakout"""
        if not self.orb.orb_complete:
            return False, False
        
        atr = bar['atr_rth']
        threshold = atr * self.breakout_threshold_mult
        
        candle_body = abs(bar['close'] - bar['open'])
        candle_range = bar['high'] - bar['low']
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        
        sufficient_high = bar['close'] > self.orb.session_high + threshold
        sufficient_low = bar['close'] < self.orb.session_low - threshold
        strong_body = body_ratio >= self.min_body_strength
        
        new_long = False
        new_short = False
        
        if self.require_open_in_range:
            open_in_range = (bar['open'] >= self.orb.session_low and 
                           bar['open'] <= self.orb.session_high)
            new_long = open_in_range and bar['close'] > self.orb.session_high and sufficient_high and strong_body
            new_short = open_in_range and bar['close'] < self.orb.session_low and sufficient_low and strong_body
        else:
            new_long = bar['close'] > self.orb.session_high and sufficient_high and strong_body
            new_short = bar['close'] < self.orb.session_low and sufficient_low and strong_body
        
        return new_long, new_short
    
    def _calc_swing_stop(self, day_df: pd.DataFrame, i: int, is_long: bool, atr: float) -> float:
        """Calculate swing stop from RTH bars"""
        swing_val = None
        valid_bars = 0
        
        for j in range(1, self.swing_lookback * 3 + 1):
            if i - j < 0:
                break
            bar_ts = day_df.index[i - j]
            if self._is_rth(bar_ts):
                if is_long:
                    bar_low = day_df['low'].iloc[i - j]
                    if swing_val is None or bar_low < swing_val:
                        swing_val = bar_low
                else:
                    bar_high = day_df['high'].iloc[i - j]
                    if swing_val is None or bar_high > swing_val:
                        swing_val = bar_high
                valid_bars += 1
                if valid_bars >= self.swing_lookback:
                    break
        
        if swing_val is None:
            swing_val = day_df['low'].iloc[max(0, i-1)] if is_long else day_df['high'].iloc[max(0, i-1)]
        
        return swing_val - (atr * self.swing_buffer) if is_long else swing_val + (atr * self.swing_buffer)
    
    def _calc_stops(self, entry: float, atr: float, vwap: float, is_long: bool, 
                    day_df: pd.DataFrame, i: int) -> dict:
        """Calculate all stop options"""
        min_stop_dist = atr * self.min_stop_atr
        
        atr_stop = entry - (atr * self.atr_stop_mult) if is_long else entry + (atr * self.atr_stop_mult)
        swing_stop = self._calc_swing_stop(day_df, i, is_long, atr)
        vwap_stop = vwap - (atr * self.vwap_stop_distance) if is_long else vwap + (atr * self.vwap_stop_distance)
        
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
        
        return {'atr_stop': atr_stop, 'swing_stop': swing_stop, 'vwap_stop': vwap_stop,
                'swing_valid': swing_valid, 'vwap_valid': vwap_valid}
    
    def _calc_achievable_rr(self, entry: float, stop: float, atr: float, rr_desired: float) -> float:
        """Calculate achievable R:R with max target cap"""
        risk = abs(entry - stop)
        if risk <= 0:
            return -1
        target_distance = risk * rr_desired
        target_atrs = target_distance / atr
        if target_atrs > self.max_target_atr + 1e-9:  # Epsilon tolerance for float boundary
            return (self.max_target_atr * atr) / risk
        return rr_desired
    
    def _select_best_stop(self, stops: dict, entry: float, atr: float, 
                          rr_desired: float, is_long: bool) -> Tuple[float, str, float]:
        """
        Select best valid stop.
        Pine tie-breaking order: ATR -> Swing -> VWAP -> Hybrid
        When R:Rs are equal, first in order wins.
        """
        candidates = []
        # Order matters for tie-breaking! ATR checked first, just like Pine
        for stop_type in ['atr', 'swing', 'vwap']:
            stop_price = stops[f'{stop_type}_stop']
            valid = stops.get(f'{stop_type}_valid', True) if stop_type != 'atr' else True
            if not valid:
                continue
            rr = self._calc_achievable_rr(entry, stop_price, atr, rr_desired)
            if rr > 0:
                candidates.append((stop_price, stop_type.upper(), rr))
        
        if not candidates:
            rr = self._calc_achievable_rr(entry, stops['atr_stop'], atr, rr_desired)
            return stops['atr_stop'], 'ATR', rr
        
        # Find best R:R, then return FIRST candidate with that R:R (Pine tie-breaker)
        best_rr = max(c[2] for c in candidates)
        
        if self.verbose:
            dir_str = "LONG" if is_long else "SHORT"
            print(f"  [STOP EVAL] {dir_str} entry=${entry:.2f} ATR={atr:.4f} VWAP={stops.get('vwap_stop',0):.2f} valid={stops.get('vwap_valid','?')}")
            for c in candidates:
                winner = " <-- BEST" if c[2] == best_rr else ""
                print(f"    {c[1]:6s}: stop=${c[0]:.2f}  RR={c[2]:.4f}{winner}")
        
        for c in candidates:
            if c[2] == best_rr:
                return c
        
        return candidates[0]  # Fallback
    
    def _in_position(self) -> bool:
        return self.position.direction != ""
    
    def _can_trigger(self) -> bool:
        return self.orb.bars_since_last_trigger >= self.orb.min_bars_between_triggers
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # EXIT LOGIC - v73 FIX: Intra-bar BE detection
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    def _check_stop_hit(self, bar: pd.Series) -> bool:
        if not self._in_position():
            return False
        if self.position.direction == "LONG":
            return bar['low'] <= self.position.current_stop
        else:
            return bar['high'] >= self.position.current_stop
    
    def _check_break_even(self, bar: pd.Series) -> bool:
        """
        v74 FIX: BE evaluated at bar CLOSE only - NOT wick/high/low.
        
        Pine logic: rr = (entryPrice - close) / risk  // for short
                    if rr >= breakEvenRR: stop := entryPrice
        
        This matches Pine's bar-close evaluation. Wicks touching BE do NOT trigger.
        Stop hit detection uses high/low (fills on touch), but BE uses close.
        """
        if not self._in_position() or not self.use_break_even or self.position.stop_moved_to_be:
            return False

        # Volatility-adaptive BE threshold (matches Pine useAdaptiveBE)
        be_rr = self.break_even_rr
        if self.use_adaptive_be:
            if self.position.vol_state == "LOW":
                be_rr = self.break_even_rr + 0.2      # Harder to trigger in low vol
            elif self.position.vol_state == "HIGH":
                be_rr = self.break_even_rr - 0.1     # Easier in high vol
            elif self.position.vol_state == "EXTREME":
                be_rr = self.break_even_rr - 0.2     # Easiest in extreme vol

        be_target_distance = self.position.risk * be_rr
        
        if self.position.direction == "LONG":
            # For longs, check if CLOSE reached BE target
            be_target_price = self.position.entry_price + be_target_distance
            return bar['close'] >= be_target_price
        else:
            # For shorts, check if CLOSE reached BE target
            be_target_price = self.position.entry_price - be_target_distance
            return bar['close'] <= be_target_price
    
    def _activate_break_even(self):
        self.position.stop_moved_to_be = True
        self.position.current_stop = self.position.entry_price
    
    def _check_trailing_activation(self, bar: pd.Series) -> bool:
        """
        v74: Trailing activation uses bar CLOSE - matching Pine.
        
        Pine: if not longTrailingStopActivated and close >= profitTargetPrice
        """
        if not self._in_position() or not self.use_trailing_stop:
            return False
        if not self.position.stop_moved_to_be or self.position.trailing_activated:
            return False
        
        profit_target_distance = self.position.risk * self.profit_target_rr
        
        if self.position.direction == "LONG":
            profit_target_price = self.position.entry_price + profit_target_distance
            return bar['close'] >= profit_target_price
        else:
            profit_target_price = self.position.entry_price - profit_target_distance
            return bar['close'] <= profit_target_price
    
    def _activate_trailing(self, bar: pd.Series, atr: float):
        self.position.trailing_activated = True
        if self.position.direction == "LONG":
            self.position.trailing_stop_level = bar['close'] - (atr * self.trailing_stop_distance)
        else:
            self.position.trailing_stop_level = bar['close'] + (atr * self.trailing_stop_distance)
        self.position.current_stop = self.position.trailing_stop_level
    
    def _update_trailing_stop(self, bar: pd.Series, atr: float, ema: float):
        if not self.position.trailing_activated:
            return
        
        trail_dist = self.trailing_stop_distance
        if self.use_aggressive_trailing:
            dist_to_ema = abs(bar['close'] - ema)
            if dist_to_ema <= (atr * self.ema_tighten_zone):
                trail_dist = self.tightened_trail_distance
        
        if self.position.direction == "LONG":
            new_trail = bar['close'] - (atr * trail_dist)
            if new_trail > self.position.trailing_stop_level:
                self.position.trailing_stop_level = new_trail
            self.position.current_stop = max(self.position.trailing_stop_level, self.position.entry_price)
        else:
            new_trail = bar['close'] + (atr * trail_dist)
            if new_trail < self.position.trailing_stop_level:
                self.position.trailing_stop_level = new_trail
            self.position.current_stop = min(self.position.trailing_stop_level, self.position.entry_price)
    
    def _check_ema_exit(self, bar: pd.Series, ema: float) -> bool:
        if not self._in_position() or not self.use_ema_exit:
            return False
        if not self.position.stop_moved_to_be or self.position.trailing_activated:
            return False
        
        if self.position.direction == "LONG":
            if bar['close'] <= self.position.entry_price:
                return False
            if bar['close'] < ema:
                self.position.bars_beyond_ema += 1
            else:
                self.position.bars_beyond_ema = 0
        else:
            if bar['close'] >= self.position.entry_price:
                return False
            if bar['close'] > ema:
                self.position.bars_beyond_ema += 1
            else:
                self.position.bars_beyond_ema = 0
        
        return self.position.bars_beyond_ema >= self.ema_confirmation_bars
    
    def _close_position(self, exit_price: float, exit_reason: str, time_str: str, date_str: str):
        """Close position and record trade"""
        if self.position.direction == "LONG":
            pnl = exit_price - self.position.entry_price
        else:
            pnl = self.position.entry_price - exit_price
        
        r_multiple = pnl / self.position.risk if self.position.risk > 0 else 0
        
        trade = TradeRecord(
            date=date_str,
            direction=self.position.direction,
            entry_price=self.position.entry_price,
            entry_time=self.position.entry_time,
            exit_price=exit_price,
            exit_time=time_str,
            exit_reason=exit_reason,
            initial_stop=self.position.initial_stop,
            stop_type=self.position.stop_type,
            vol_state=self.position.vol_state,
            risk=self.position.risk,
            pnl=pnl,
            r_multiple=r_multiple,
            confluence_score=self.position.confluence_score,
            ssl_score=self.position.ssl_score,
            wae_score=self.position.wae_score,
            qqe_score=self.position.qqe_score,
            vol_score=self.position.vol_score
        )
        self.trades.append(trade)
        
        if self.verbose:
            emoji = "âœ…" if pnl > 0 else "âŒ"
            conf_str = f" [C:{self.position.confluence_score}]" if self.enable_confluence else ""
            print(f"  {emoji} {self.position.direction} {date_str} {self.position.entry_time}â†’{time_str}: ${pnl:+.2f} ({r_multiple:+.2f}R) - {exit_reason}{conf_str}")
        
        self.position.reset()
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # MAIN PROCESSING LOOP
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    def _process_day(self, date_str: str):
        """Process a single trading day"""
        day_df = self.df[self.df.index.date.astype(str) == date_str].copy()
        
        if len(day_df) == 0:
            return
        
        # Reset session state
        self.orb.reset_session()
        self.vol.orb_session_baseline = np.nan
        
        for i in range(len(day_df)):
            ts = day_df.index[i]
            bar = day_df.iloc[i]
            
            if not self._is_rth(ts):
                continue
            
            time_str = ts.strftime('%H:%M')
            
            # Cooldown counter
            self.orb.bars_since_last_trigger += 1
            
            atr = bar['atr_rth']
            ema = bar['ema9']
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # EXIT CHECKS - CORRECTED: Stop checked FIRST with PREVIOUS bar's value
            # Pine: strategy.exit() runs first with previous bar's stop, THEN BE/trailing update for next bar
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            
            if self._in_position():
                # STEP 1: Check if stop was hit FIRST (using current_stop set on PREVIOUS bar)
                # This matches Pine's strategy.exit() which uses the stop value from when order was placed
                if self._check_stop_hit(bar):
                    exit_price = self.position.current_stop
                    reason = "TRAILING STOP" if self.position.trailing_activated else \
                             "BREAK-EVEN STOP" if self.position.stop_moved_to_be else "INITIAL STOP"
                    self._close_position(exit_price, reason, time_str, date_str)
                
                # STEP 2: Check EMA exit (only if not stopped out)
                elif self._check_ema_exit(bar, ema):
                    self._close_position(bar['close'], "EMA EXIT", time_str, date_str)
                
                # STEP 3: If still in position, update stops for NEXT bar
                # These updates take effect on the NEXT bar, matching Pine's execution order
                if self._in_position():
                    # Check if BE target was reached (updates stop for next bar)
                    if self._check_break_even(bar):
                        self._activate_break_even()
                    
                    # Check if trailing should activate (updates stop for next bar)
                    if self._check_trailing_activation(bar):
                        self._activate_trailing(bar, atr)
                    
                    # Update trailing stop level if active (updates stop for next bar)
                    if self.position.trailing_activated:
                        self._update_trailing_stop(bar, atr, ema)
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # VOL STATE - Must run BEFORE ORB window continue!
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            self._calc_vol_state(bar, ts)
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # ORB BUILDING
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            
            if self._is_orb_window(ts):
                if ts.hour == 9 and ts.minute == 30:
                    self.orb.reset_session()
                    self.orb.session_high = bar['high']
                    self.orb.session_low = bar['low']
                else:
                    self.orb.session_high = max(self.orb.session_high, bar['high'])
                    self.orb.session_low = min(self.orb.session_low, bar['low'])
                continue
            
            # ORB complete
            if not self.orb.orb_complete and not pd.isna(self.orb.session_high):
                self.orb.orb_complete = True
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # BREAKOUT DETECTION
            # Pine parity: breakout pending is SINGLE-BAR only
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            
            # Track if breakout detected THIS bar (for single-bar pending enforcement)
            pending_long_this_bar = False
            pending_short_this_bar = False
            
            new_long, new_short = self._check_breakout(bar)
            
            if new_long and not self.orb.has_broken_out_high:
                self.orb.long_breakout_pending = True
                pending_long_this_bar = True
                self.orb.has_broken_out_high = True
            
            if new_short and not self.orb.has_broken_out_low:
                self.orb.short_breakout_pending = True
                pending_short_this_bar = True
                self.orb.has_broken_out_low = True
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # ENTRY EVALUATION (with confluence check - v70)
            # Pine parity: Can only enter on the ACTUAL breakout bar
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            
            if not self._in_position() and self._is_trading_window(ts) and self._can_trigger():
                
                # CRITICAL: Require pending_this_bar - breakout must be THIS bar, not stale
                if self.orb.long_breakout_pending and pending_long_this_bar:
                    entry = bar['close']
                    vwap = bar['vwap']
                    
                    # Confluence check - Pine keeps pending alive until confluence met
                    if self.enable_confluence:
                        # Use current bar's confluence values
                        # Convert day_df index to full df index
                        full_idx = self.df.index.get_loc(ts)
                        passes_conf, conf_score, conf_details = self.confluence_calc.check_confluence(
                            full_idx, is_long=True, orb_breakout=True
                        )
                        
                        # Mode-specific scoring
                        if self.confluence_mode == 'ssl':
                            # ORB(1) + SSL only
                            mode_score = 1 + conf_details.ssl_score_bull
                            passes_conf = mode_score >= self.min_confluence
                        elif self.confluence_mode == 'wae':
                            # ORB(1) + WAE only
                            mode_score = 1 + conf_details.wae_score_bull
                            passes_conf = mode_score >= self.min_confluence
                        elif self.confluence_mode == 'qqe':
                            # ORB(1) + QQE only
                            mode_score = 1 + conf_details.qqe_score_bull
                            passes_conf = mode_score >= self.min_confluence
                        elif self.confluence_mode == 'vol':
                            # ORB(1) + Volume only
                            # pine_round imported at top
                            mode_score = 1 + pine_round(conf_details.vol_score)
                            passes_conf = mode_score >= self.min_confluence
                        # else 'all' - use passes_conf from check_confluence
                        
                        if passes_conf:
                            _conf_score = conf_score
                            _ssl_score = conf_details.ssl_score_bull
                            _wae_score = conf_details.wae_score_bull
                            _qqe_score = conf_details.qqe_score_bull
                            _vol_score = conf_details.vol_score
                        else:
                            # Confluence not met - consume pending anyway (Pine parity)
                            # breakout attempt is done, must return inside ORB for new one
                            self.orb.long_breakout_pending = False
                            self.orb.bars_since_last_trigger = 0
                            continue  # Skip to next bar
                    else:
                        passes_conf = True
                        _conf_score = 0
                        _ssl_score = 0.0
                        _wae_score = 0.0
                        _qqe_score = 0.0
                        _vol_score = 0.0
                    
                    # RR evaluation (confluence already passed if we get here)
                    rr_desired = self.profit_target_rr
                    if self.vol.vol_state == "LOW":
                        rr_desired += 0.5
                    elif self.vol.vol_state == "HIGH":
                        rr_desired = max(rr_desired - 0.5, 1.2)
                    elif self.vol.vol_state == "EXTREME":
                        rr_desired = max(rr_desired - 0.8, 1.0)
                    
                    if self.verbose:
                        print(f"  [VOL/RR] LONG vol_state={self.vol.vol_state} vf={self.vol.vol_factor:.4f} sessionVF={self.vol.session_vf:.4f} htfVF={self.vol.htf_vf:.4f} rr_desired={rr_desired:.2f} (base={self.profit_target_rr})")
                    stops = self._calc_stops(entry, atr, vwap, True, day_df, i)
                    stop_price, stop_type, achievable_rr = self._select_best_stop(stops, entry, atr, rr_desired, True)
                    
                    # LOW vol C-grade filter: require higher R:R in choppy conditions
                    min_rr_for_entry = self.min_acceptable_rr
                    if self.skip_low_vol_c_grades and self.vol.vol_state == "LOW":
                        min_rr_for_entry = max(self.min_acceptable_rr, self.low_vol_min_rr)
                    
                    if achievable_rr >= min_rr_for_entry - 0.0001:
                        risk = abs(entry - stop_price)
                        self.position.direction = "LONG"
                        self.position.entry_price = entry
                        self.position.entry_time = time_str
                        self.position.entry_date = date_str
                        self.position.initial_stop = stop_price
                        self.position.current_stop = stop_price
                        self.position.stop_type = stop_type
                        self.position.risk = risk
                        self.position.target_rr = achievable_rr
                        self.position.vol_state = self.vol.vol_state
                        self.position.confluence_score = _conf_score
                        self.position.ssl_score = _ssl_score
                        self.position.wae_score = _wae_score
                        self.position.qqe_score = _qqe_score
                        self.position.vol_score = _vol_score
                        self.orb.bars_since_last_trigger = 0
                        self.orb.long_breakout_pending = False  # Clear pending on entry
                    else:
                        skip = SkipRecord(
                            date=date_str, time=time_str, direction="LONG",
                            entry_price=entry, stop_price=stop_price, stop_type=stop_type,
                            achievable_rr=achievable_rr, min_rr=min_rr_for_entry,
                            reason=f"RR {achievable_rr:.2f} < {min_rr_for_entry:.1f}" + (" (LOW VOL C-GRADE)" if min_rr_for_entry > self.min_acceptable_rr else "")
)
                        self.skips.append(skip)
                        if self.verbose:
                            low_vol_tag = " [LOW VOL C-GRADE]" if min_rr_for_entry > self.min_acceptable_rr else ""
                            print(f"  [SKIP] LONG {date_str} {time_str} @ ${entry:.2f}: RR {achievable_rr:.2f} < {min_rr_for_entry:.2f} ({stop_type}){low_vol_tag}")
                        self.orb.bars_since_last_trigger = 0  # v68 skip consumption
                        self.orb.long_breakout_pending = False  # Pine consumes pending on SKIP too!
                
                # CRITICAL: Require pending_this_bar - breakout must be THIS bar, not stale
                if self.orb.short_breakout_pending and pending_short_this_bar and not self._in_position():
                    entry = bar['close']
                    vwap = bar['vwap']
                    
                    # Confluence check - Pine keeps pending alive until confluence met
                    if self.enable_confluence:
                        # Use current bar's confluence values
                        # Convert day_df index to full df index
                        full_idx = self.df.index.get_loc(ts)
                        passes_conf, conf_score, conf_details = self.confluence_calc.check_confluence(
                            full_idx, is_long=False, orb_breakout=True
                        )
                        
                        # Mode-specific scoring
                        if self.confluence_mode == 'ssl':
                            # ORB(1) + SSL only
                            mode_score = 1 + conf_details.ssl_score_bear
                            passes_conf = mode_score >= self.min_confluence
                        elif self.confluence_mode == 'wae':
                            # ORB(1) + WAE only
                            mode_score = 1 + conf_details.wae_score_bear
                            passes_conf = mode_score >= self.min_confluence
                        elif self.confluence_mode == 'qqe':
                            # ORB(1) + QQE only
                            mode_score = 1 + conf_details.qqe_score_bear
                            passes_conf = mode_score >= self.min_confluence
                        elif self.confluence_mode == 'vol':
                            # ORB(1) + Volume only
                            # pine_round imported at top
                            mode_score = 1 + pine_round(conf_details.vol_score)
                            passes_conf = mode_score >= self.min_confluence
                        # else 'all' - use passes_conf from check_confluence
                        
                        if passes_conf:
                            _conf_score = conf_score
                            _ssl_score = conf_details.ssl_score_bear
                            _wae_score = conf_details.wae_score_bear
                            _qqe_score = conf_details.qqe_score_bear
                            _vol_score = conf_details.vol_score
                        else:
                            # Confluence not met - consume pending anyway (Pine parity)
                            # breakout attempt is done, must return inside ORB for new one
                            self.orb.short_breakout_pending = False
                            self.orb.bars_since_last_trigger = 0
                            continue  # Skip to next bar
                    else:
                        passes_conf = True
                        _conf_score = 0
                        _ssl_score = 0.0
                        _wae_score = 0.0
                        _qqe_score = 0.0
                        _vol_score = 0.0
                    
                    # RR evaluation (confluence already passed if we get here)
                    rr_desired = self.profit_target_rr
                    if self.vol.vol_state == "LOW":
                        rr_desired += 0.5
                    elif self.vol.vol_state == "HIGH":
                        rr_desired = max(rr_desired - 0.5, 1.2)
                    elif self.vol.vol_state == "EXTREME":
                        rr_desired = max(rr_desired - 0.8, 1.0)
                    
                    if self.verbose:
                        print(f"  [VOL/RR] SHORT vol_state={self.vol.vol_state} vf={self.vol.vol_factor:.4f} sessionVF={self.vol.session_vf:.4f} htfVF={self.vol.htf_vf:.4f} rr_desired={rr_desired:.2f} (base={self.profit_target_rr})")
                    stops = self._calc_stops(entry, atr, vwap, False, day_df, i)
                    stop_price, stop_type, achievable_rr = self._select_best_stop(stops, entry, atr, rr_desired, False)
                    
                    # LOW vol C-grade filter: require higher R:R in choppy conditions
                    min_rr_for_entry = self.min_acceptable_rr
                    if self.skip_low_vol_c_grades and self.vol.vol_state == "LOW":
                        min_rr_for_entry = max(self.min_acceptable_rr, self.low_vol_min_rr)
                    
                    if achievable_rr >= min_rr_for_entry - 0.0001:
                        risk = abs(entry - stop_price)
                        self.position.direction = "SHORT"
                        self.position.entry_price = entry
                        self.position.entry_time = time_str
                        self.position.entry_date = date_str
                        self.position.initial_stop = stop_price
                        self.position.current_stop = stop_price
                        self.position.stop_type = stop_type
                        self.position.risk = risk
                        self.position.target_rr = achievable_rr
                        self.position.vol_state = self.vol.vol_state
                        self.position.confluence_score = _conf_score
                        self.position.ssl_score = _ssl_score
                        self.position.wae_score = _wae_score
                        self.position.qqe_score = _qqe_score
                        self.position.vol_score = _vol_score
                        self.orb.bars_since_last_trigger = 0
                        self.orb.short_breakout_pending = False  # Clear pending on entry
                    else:
                        skip = SkipRecord(
                            date=date_str, time=time_str, direction="SHORT",
                            entry_price=entry, stop_price=stop_price, stop_type=stop_type,
                            achievable_rr=achievable_rr, min_rr=min_rr_for_entry,
                            reason=f"RR {achievable_rr:.2f} < {min_rr_for_entry:.1f}" + (" (LOW VOL C-GRADE)" if min_rr_for_entry > self.min_acceptable_rr else "")
                        )
                        self.skips.append(skip)
                        if self.verbose:
                            low_vol_tag = " [LOW VOL C-GRADE]" if min_rr_for_entry > self.min_acceptable_rr else ""
                            print(f"  [SKIP] SHORT {date_str} {time_str} @ ${entry:.2f}: RR {achievable_rr:.2f} < {min_rr_for_entry:.2f} ({stop_type}){low_vol_tag}")
                        self.orb.bars_since_last_trigger = 0
                        self.orb.short_breakout_pending = False  # Pine consumes pending on SKIP too!
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # STATE MACHINE UPDATES
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            
            if self.orb.orb_complete and not self._in_position():
                # Pine: clear pending AND breakout flag when price returns inside ORB
                # This allows re-entry after BE stop when price returns to range
                # Long: reset when close <= sessionHigh
                # Short: reset when close >= sessionLow
                if bar['close'] <= self.orb.session_high:
                    self.orb.long_breakout_pending = False
                    self.orb.has_broken_out_high = False
                if bar['close'] >= self.orb.session_low:
                    self.orb.short_breakout_pending = False
                    self.orb.has_broken_out_low = False
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PINE PARITY: Pending signals are SINGLE-BAR only
            # A breakout must be acted upon on the breakout bar itself.
            # If not consumed this bar, clear pending for next bar.
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            if not pending_long_this_bar:
                self.orb.long_breakout_pending = False
            if not pending_short_this_bar:
                self.orb.short_breakout_pending = False
        
        # End of day - close any open position
        if self._in_position():
            last_bar = day_df[day_df.index.map(self._is_rth)].iloc[-1]
            self._close_position(last_bar['close'], "END OF DAY", "16:00", date_str)
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # MAIN RUN METHOD
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    def run(self) -> Dict[str, Any]:
        """
        Run backtest across date range.
        Returns dictionary with trades and summary stats.
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  ORB BACKTESTER: {self.symbol}")
            print(f"  {self.start_date} to {self.end_date}")
            print(f"{'='*70}\n")
        
        # Load data
        self.load_data()
        
        # Get trading days in range
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        
        all_dates = self.df.index.date
        unique_dates = sorted(set(all_dates))
        trading_dates = [d for d in unique_dates if start.date() <= d <= end.date()]
        
        if self.verbose:
            print(f"Processing {len(trading_dates)} trading days...\n")
        
        # Process each day
        for date in trading_dates:
            date_str = str(date)
            self._process_day(date_str)
        
        # Calculate summary statistics
        results = self._calc_summary()
        
        if self.verbose:
            self._print_summary(results)
        
        return results
    
    def _calc_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        if not self.trades:
            return {
                'symbol': self.symbol,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'breakevens': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_r': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown_r': 0,
                'avg_r_per_trade': 0,
                'trades': []
            }
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        breakevens = [t for t in self.trades if t.pnl == 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_r = sum(t.r_multiple for t in self.trades)
        
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        
        # Max drawdown in R
        cumulative_r = 0
        peak_r = 0
        max_dd_r = 0
        for t in self.trades:
            cumulative_r += t.r_multiple
            peak_r = max(peak_r, cumulative_r)
            dd = peak_r - cumulative_r
            max_dd_r = max(max_dd_r, dd)
        
        return {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_trades': len(self.trades),
            'total_skips': len(self.skips),
            'wins': len(wins),
            'losses': len(losses),
            'breakevens': len(breakevens),
            'win_rate': len(wins) / (len(wins) + len(losses)) * 100 if (len(wins) + len(losses)) > 0 else 0,
            'total_pnl': total_pnl,
            'total_r': total_r,
            'avg_win': sum(t.pnl for t in wins) / len(wins) if wins else 0,
            'avg_loss': sum(t.pnl for t in losses) / len(losses) if losses else 0,
            'avg_win_r': sum(t.r_multiple for t in wins) / len(wins) if wins else 0,
            'avg_loss_r': sum(t.r_multiple for t in losses) / len(losses) if losses else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'max_drawdown_r': max_dd_r,
            'avg_r_per_trade': total_r / len(self.trades) if self.trades else 0,
            'trades': [
                {
                    'date': t.date,
                    'direction': t.direction,
                    'entry_price': t.entry_price,
                    'entry_time': t.entry_time,
                    'exit_price': t.exit_price,
                    'exit_time': t.exit_time,
                    'exit_reason': t.exit_reason,
                    'stop_type': t.stop_type,
                    'vol_state': t.vol_state,
                    'pnl': t.pnl,
                    'r_multiple': t.r_multiple
                }
                for t in self.trades
            ],
            'skips': [
                {
                    'date': s.date,
                    'time': s.time,
                    'direction': s.direction,
                    'entry_price': s.entry_price,
                    'stop_price': s.stop_price,
                    'stop_type': s.stop_type,
                    'achievable_rr': s.achievable_rr,
                    'min_rr': s.min_rr,
                    'reason': s.reason
                }
                for s in self.skips
            ]
        }
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print formatted summary"""
        print(f"\n{'='*70}")
        print(f"  BACKTEST SUMMARY: {results['symbol']}")
        print(f"  {results['start_date']} to {results['end_date']}")
        print(f"{'='*70}\n")
        
        print(f"  Total Trades:    {results['total_trades']}")
        print(f"  Total Skips:     {results.get('total_skips', 0)} (R:R below {self.min_acceptable_rr})")
        print(f"  Wins/Losses/BE:  {results['wins']}W / {results['losses']}L / {results.get('breakevens', 0)}BE")
        print(f"  Win Rate:        {results['win_rate']:.1f}% (of decided trades)")
        print(f"")
        print(f"  Total P/L:       ${results['total_pnl']:+.2f}")
        print(f"  Total R:         {results['total_r']:+.2f}R")
        print(f"  Avg R/Trade:     {results['avg_r_per_trade']:+.3f}R")
        print(f"")
        print(f"  Avg Win:         ${results['avg_win']:.2f} ({results.get('avg_win_r', 0):.2f}R)")
        print(f"  Avg Loss:        ${results['avg_loss']:.2f} ({results.get('avg_loss_r', 0):.2f}R)")
        print(f"  Profit Factor:   {results['profit_factor']:.2f}")
        print(f"  Max Drawdown:    {results['max_drawdown_r']:.2f}R")
        print(f"\n{'='*70}\n")
        
        # Trade details table
        if results['trades']:
            print(f"\n{'='*70}")
            print("TRADE DETAILS")
            print(f"{'='*70}")
            for t in results['trades']:
                dir_str = "LONG " if t['direction'] == "LONG" else "SHORT"
                print(f"{t['date']} {dir_str} {t['entry_time']}->{t['exit_time']}  "
                      f"${t['entry_price']:.2f}->${t['exit_price']:.2f}  "
                      f"{t['exit_reason']:<18} {t['r_multiple']:+.2f}R")
            print(f"\nTotal: {len(results['trades'])} trades, {results['total_r']:+.2f}R")
        
        # Skip details (if any and verbose)
        if results.get('skips') and self.verbose:
            print(f"\n{'='*70}")
            print("SKIPPED TRADES (R:R below minimum)")
            print(f"{'='*70}")
            for s in results['skips']:
                dir_str = "LONG " if s['direction'] == "LONG" else "SHORT"
                print(f"{s['date']} {dir_str} {s['time']} @ ${s['entry_price']:.2f}  "
                      f"RR={s['achievable_rr']:.2f} < {s['min_rr']:.1f} ({s['stop_type']})")
            print(f"\nTotal: {len(results['skips'])} skipped")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# COMMAND LINE INTERFACE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

if __name__ == '__main__':
    # Example usage
    bt = ORBBacktester(
        symbol='AMD',
        start_date='2025-10-01',
        end_date='2025-12-31',
        verbose=True
    )
    results = bt.run()
