"""
HMM Backtester - Pine HHM Compatible (Confluence-Driven)
=========================================================
Forked from ORB Backtester v76. Replaces ORB breakout trigger with
HMM confluence scoring: SSL + WAE + QQE + Volume + IB Phase.

KEY DIFFERENCES FROM ORB:
1. NO ORB breakout trigger - entry is pure confluence-driven
2. IB (Initial Balance) formation: 30-60 min (symbol-dependent)
3. IB Phase Scoring: Edgeful retracement zones (IMP/PB25/PB50/FAIL)
4. Quality gates: HTF trend, market regime, volatility, lunch skip
5. Entry window: after IB formation (10:30+)
6. Signal re-evaluated each bar (not single-bar pending)

IDENTICAL TO ORB:
- Data loading (Polygon), Confluence indicators (SSL/WAE/QQE/Vol)
- Stop selection engine (ATR/Swing/VWAP with auto-select)
- Exit cascade (stop -> BE -> trailing -> EMA -> EOD)
- Volatility state engine, Position sizing, P&L tracking

Usage:
    from hmm_backtester import HMMBacktester
    
    bt = HMMBacktester(
        symbol='AMD',
        start_date='2025-10-01',
        end_date='2025-12-31',
        min_confluence=3
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

# Import confluence indicators (same as ORB)
try:
    from confluence_indicators import ConfluenceCalculator, ConfluenceScores, pine_round
except ImportError:
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from confluence_indicators import ConfluenceCalculator, ConfluenceScores, pine_round


# ═══════════════════════════════════════════════════════════════════════════
# HMM SYMBOL PRESETS - From HHM.pine switch blocks
# ═══════════════════════════════════════════════════════════════════════════

HMM_SYMBOL_PRESETS = {
    'AAPL': {
        'ssl_baseline_length': 20, 'ssl_length': 10,
        'wae_fast_ema': 22, 'wae_slow_ema': 44, 'wae_bb_length': 22, 'wae_sensitivity': 200,
        'qqe_rsi1_length': 10, 'qqe_rsi1_smoothing': 6,
        'qqe_rsi2_length': 5, 'qqe_rsi2_smoothing': 5, 'qqe_bb_length': 26,
        'vol_lookback': 16, 'ib_period_minutes': 60,
        'low_vol_threshold': 0.8, 'high_vol_threshold': 1.3, 'extreme_vol_threshold': 2.0,
    },
    'NVDA': {
        'ssl_baseline_length': 20, 'ssl_length': 10,
        'wae_fast_ema': 17, 'wae_slow_ema': 35, 'wae_bb_length': 20, 'wae_sensitivity': 325,
        'qqe_rsi1_length': 9, 'qqe_rsi1_smoothing': 5,
        'qqe_rsi2_length': 4, 'qqe_rsi2_smoothing': 4, 'qqe_bb_length': 25,
        'vol_lookback': 13, 'ib_period_minutes': 55,
        'low_vol_threshold': 0.8, 'high_vol_threshold': 1.3, 'extreme_vol_threshold': 2.0,
    },
    'AMD': {
        'ssl_baseline_length': 60, 'ssl_length': 10,
        'wae_fast_ema': 14, 'wae_slow_ema': 29, 'wae_bb_length': 20, 'wae_sensitivity': 300,
        'qqe_rsi1_length': 8, 'qqe_rsi1_smoothing': 5,
        'qqe_rsi2_length': 4, 'qqe_rsi2_smoothing': 4, 'qqe_bb_length': 25,
        'vol_lookback': 12, 'ib_period_minutes': 52,
        'low_vol_threshold': 0.8, 'high_vol_threshold': 1.3, 'extreme_vol_threshold': 2.0,
    },
    'GOOGL': {
        'ssl_baseline_length': 60, 'ssl_length': 10,
        'wae_fast_ema': 17, 'wae_slow_ema': 35, 'wae_bb_length': 20, 'wae_sensitivity': 200,
        'qqe_rsi1_length': 9, 'qqe_rsi1_smoothing': 5,
        'qqe_rsi2_length': 4, 'qqe_rsi2_smoothing': 4, 'qqe_bb_length': 25,
        'vol_lookback': 13, 'ib_period_minutes': 55,
        'low_vol_threshold': 0.8, 'high_vol_threshold': 1.3, 'extreme_vol_threshold': 2.0,
    },
    'AMZN': {
        'ssl_baseline_length': 50, 'ssl_length': 10,
        'wae_fast_ema': 21, 'wae_slow_ema': 43, 'wae_bb_length': 22, 'wae_sensitivity': 200,
        'qqe_rsi1_length': 10, 'qqe_rsi1_smoothing': 6,
        'qqe_rsi2_length': 5, 'qqe_rsi2_smoothing': 5, 'qqe_bb_length': 26,
        'vol_lookback': 16, 'ib_period_minutes': 60,
        'low_vol_threshold': 0.8, 'high_vol_threshold': 1.3, 'extreme_vol_threshold': 2.0,
    },
    'META': {
        'ssl_baseline_length': 60, 'ssl_length': 10,
        'wae_fast_ema': 17, 'wae_slow_ema': 35, 'wae_bb_length': 20, 'wae_sensitivity': 200,
        'qqe_rsi1_length': 9, 'qqe_rsi1_smoothing': 5,
        'qqe_rsi2_length': 4, 'qqe_rsi2_smoothing': 4, 'qqe_bb_length': 25,
        'vol_lookback': 13, 'ib_period_minutes': 55,
        'low_vol_threshold': 0.8, 'high_vol_threshold': 1.3, 'extreme_vol_threshold': 2.0,
    },
    'MSFT': {
        'ssl_baseline_length': 60, 'ssl_length': 10,
        'wae_fast_ema': 21, 'wae_slow_ema': 43, 'wae_bb_length': 22, 'wae_sensitivity': 200,
        'qqe_rsi1_length': 10, 'qqe_rsi1_smoothing': 6,
        'qqe_rsi2_length': 5, 'qqe_rsi2_smoothing': 5, 'qqe_bb_length': 26,
        'vol_lookback': 16, 'ib_period_minutes': 60,
        'low_vol_threshold': 0.8, 'high_vol_threshold': 1.3, 'extreme_vol_threshold': 2.0,
    },
    'TSLA': {
        'ssl_baseline_length': 40, 'ssl_length': 10,
        'wae_fast_ema': 14, 'wae_slow_ema': 29, 'wae_bb_length': 20, 'wae_sensitivity': 300,
        'qqe_rsi1_length': 8, 'qqe_rsi1_smoothing': 5,
        'qqe_rsi2_length': 4, 'qqe_rsi2_smoothing': 4, 'qqe_bb_length': 25,
        'vol_lookback': 12, 'ib_period_minutes': 52,
        'low_vol_threshold': 0.8, 'high_vol_threshold': 1.3, 'extreme_vol_threshold': 2.0,
    },
    'PLTR': {
        'ssl_baseline_length': 20, 'ssl_length': 10,
        'wae_fast_ema': 14, 'wae_slow_ema': 29, 'wae_bb_length': 20, 'wae_sensitivity': 200,
        'qqe_rsi1_length': 8, 'qqe_rsi1_smoothing': 5,
        'qqe_rsi2_length': 4, 'qqe_rsi2_smoothing': 4, 'qqe_bb_length': 25,
        'vol_lookback': 12, 'ib_period_minutes': 52,
        'low_vol_threshold': 0.8, 'high_vol_threshold': 1.3, 'extreme_vol_threshold': 2.0,
    },
    'NFLX': {
        'ssl_baseline_length': 60, 'ssl_length': 10,
        'wae_fast_ema': 17, 'wae_slow_ema': 35, 'wae_bb_length': 20, 'wae_sensitivity': 200,
        'qqe_rsi1_length': 9, 'qqe_rsi1_smoothing': 5,
        'qqe_rsi2_length': 4, 'qqe_rsi2_smoothing': 4, 'qqe_bb_length': 25,
        'vol_lookback': 13, 'ib_period_minutes': 55,
        'low_vol_threshold': 0.8, 'high_vol_threshold': 1.3, 'extreme_vol_threshold': 2.0,
    },
    'ORCL': {
        'ssl_baseline_length': 20, 'ssl_length': 10,
        'wae_fast_ema': 13, 'wae_slow_ema': 27, 'wae_bb_length': 20, 'wae_sensitivity': 200,
        'qqe_rsi1_length': 8, 'qqe_rsi1_smoothing': 5,
        'qqe_rsi2_length': 4, 'qqe_rsi2_smoothing': 4, 'qqe_bb_length': 25,
        'vol_lookback': 12, 'ib_period_minutes': 52,
        'low_vol_threshold': 0.8, 'high_vol_threshold': 1.3, 'extreme_vol_threshold': 2.0,
    },
}

HMM_DEFAULT_PRESET = {
    'ssl_baseline_length': 25, 'ssl_length': 10,
    'wae_fast_ema': 20, 'wae_slow_ema': 40, 'wae_bb_length': 20, 'wae_sensitivity': 200,
    'qqe_rsi1_length': 9, 'qqe_rsi1_smoothing': 4,
    'qqe_rsi2_length': 4, 'qqe_rsi2_smoothing': 4, 'qqe_bb_length': 25,
    'vol_lookback': 12, 'ib_period_minutes': 30,
    'low_vol_threshold': 0.8, 'high_vol_threshold': 1.3, 'extreme_vol_threshold': 2.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IBState:
    """Initial Balance range and breakout state"""
    ib_high: float = np.nan
    ib_low: float = np.nan
    ib_formed: bool = False
    # Persistent break tracking (Edgeful - does NOT reset on pullback)
    ever_broke_up: bool = False
    ever_broke_down: bool = False
    # Phase scoring
    phase_bull: str = "---"
    phase_bear: str = "---"
    score_bull: float = 0.0
    score_bear: float = 0.0
    
    def reset_session(self):
        self.ib_high = np.nan
        self.ib_low = np.nan
        self.ib_formed = False
        self.ever_broke_up = False
        self.ever_broke_down = False
        self.phase_bull = "---"
        self.phase_bear = "---"
        self.score_bull = 0.0
        self.score_bear = 0.0


@dataclass
class VolState:
    """Volatility state calculation (identical to ORB)"""
    orb_session_baseline: float = np.nan
    session_vf: float = 1.0
    htf_vf: float = 1.0
    vol_factor: float = 1.0
    vol_state: str = "NORMAL"


@dataclass
class Position:
    """Active position with exit management (identical to ORB)"""
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
    
    # Confluence tracking
    confluence_score: float = 0.0
    ssl_score: float = 0.0
    wae_score: float = 0.0
    qqe_score: float = 0.0
    vol_score: float = 0.0
    ib_score: float = 0.0
    ib_phase: str = "---"
    
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
        self.confluence_score = 0.0
        self.ssl_score = 0.0
        self.wae_score = 0.0
        self.qqe_score = 0.0
        self.vol_score = 0.0
        self.ib_score = 0.0
        self.ib_phase = "---"
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
    confluence_score: float = 0.0
    ssl_score: float = 0.0
    wae_score: float = 0.0
    qqe_score: float = 0.0
    vol_score: float = 0.0
    ib_score: float = 0.0
    ib_phase: str = "---"


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
    confluence_score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# MAIN BACKTESTER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class HMMBacktester:
    """
    HMM backtester with Pine HHM compatible logic.
    Confluence-driven entry with IB phase scoring.
    """
    
    def __init__(
        self,
        symbol: str = 'AMD',
        start_date: str = '2025-10-01',
        end_date: str = '2025-12-31',
        
        # ═══════════════════════════════════════════════════════════════
        # CONFLUENCE PARAMETERS
        # ═══════════════════════════════════════════════════════════════
        min_confluence: int = 3,
        include_ib_in_confluence: bool = True,
        use_ib_phase_scoring: bool = True,
        
        # SSL Hybrid
        ssl_baseline_length: int = 25,
        ssl_length: int = 10,
        ssl_type: str = 'JMA',
        use_ssl_momentum: bool = True,
        
        # WAE
        wae_fast_ema: int = 20,
        wae_slow_ema: int = 40,
        wae_sensitivity: int = 200,
        wae_bb_length: int = 20,
        wae_bb_mult: float = 2.0,
        use_wae_acceleration: bool = True,
        
        # QQE
        qqe_rsi1_length: int = 9,
        qqe_rsi1_smoothing: int = 4,
        qqe_factor_primary: float = 3.0,
        qqe_rsi2_length: int = 4,
        qqe_rsi2_smoothing: int = 4,
        qqe_factor_secondary: float = 1.61,
        qqe_bb_length: int = 25,
        qqe_bb_mult: float = 0.35,
        qqe_threshold: int = 3,
        qqe_consecutive_bars: int = 3,
        use_qqe_momentum: bool = True,
        
        # Volume
        vol_lookback: int = 12,
        
        # ═══════════════════════════════════════════════════════════════
        # IB (Initial Balance) PARAMETERS
        # ═══════════════════════════════════════════════════════════════
        ib_period_minutes: int = 30,
        ib_buffer_atr_mult: float = 0.1,  # Buffer for IB break detection
        
        # ═══════════════════════════════════════════════════════════════
        # QUALITY GATES
        # ═══════════════════════════════════════════════════════════════
        use_htf_trend_filter: bool = True,
        htf_trend_length: int = 5,
        use_market_regime_filter: bool = True,
        use_volatility_filter: bool = True,
        avoid_lunch_hour: bool = True,
        min_bars_between_entries: int = 1,
        
        # ═══════════════════════════════════════════════════════════════
        # STOP PARAMETERS (identical to ORB)
        # ═══════════════════════════════════════════════════════════════
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        swing_lookback: int = 5,
        swing_buffer: float = 0.02,
        vwap_stop_distance: float = 0.3,
        min_stop_atr: float = 0.15,
        max_target_atr: float = 3.0,
        min_acceptable_rr: float = 1.5,
        profit_target_rr: float = 2.0,
        
        # ═══════════════════════════════════════════════════════════════
        # EXIT PARAMETERS (identical to ORB)
        # ═══════════════════════════════════════════════════════════════
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
        
        # ═══════════════════════════════════════════════════════════════
        # VOLATILITY THRESHOLDS
        # ═══════════════════════════════════════════════════════════════
        low_vol_threshold: float = 0.8,
        high_vol_threshold: float = 1.3,
        extreme_vol_threshold: float = 2.0,
        skip_low_vol_c_grades: bool = True,
        low_vol_min_rr: float = 2.0,
        
        # ═══════════════════════════════════════════════════════════════
        # OUTPUT
        # ═══════════════════════════════════════════════════════════════
        verbose: bool = False,
        bar_size: int = 2,  # Chart timeframe in minutes (Pine HHM runs on 2-min charts)
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
        # Get HMM symbol preset
        preset = HMM_SYMBOL_PRESETS.get(symbol, HMM_DEFAULT_PRESET)
        
        # Confluence - apply presets where constructor has default values
        self.min_confluence = min_confluence
        self.include_ib_in_confluence = include_ib_in_confluence
        self.use_ib_phase_scoring = use_ib_phase_scoring
        
        self.ssl_baseline_length = preset['ssl_baseline_length'] if ssl_baseline_length == 25 else ssl_baseline_length
        self.ssl_length = preset['ssl_length'] if ssl_length == 10 else ssl_length
        self.ssl_type = ssl_type
        self.use_ssl_momentum = use_ssl_momentum
        
        self.wae_fast_ema = preset['wae_fast_ema'] if wae_fast_ema == 20 else wae_fast_ema
        self.wae_slow_ema = preset['wae_slow_ema'] if wae_slow_ema == 40 else wae_slow_ema
        self.wae_sensitivity = preset['wae_sensitivity'] if wae_sensitivity == 200 else wae_sensitivity
        self.wae_bb_length = preset['wae_bb_length'] if wae_bb_length == 20 else wae_bb_length
        self.wae_bb_mult = wae_bb_mult
        self.use_wae_acceleration = use_wae_acceleration
        
        self.qqe_rsi1_length = preset['qqe_rsi1_length'] if qqe_rsi1_length == 9 else qqe_rsi1_length
        self.qqe_rsi1_smoothing = preset['qqe_rsi1_smoothing'] if qqe_rsi1_smoothing == 4 else qqe_rsi1_smoothing
        self.qqe_factor_primary = qqe_factor_primary
        self.qqe_rsi2_length = preset['qqe_rsi2_length'] if qqe_rsi2_length == 4 else qqe_rsi2_length
        self.qqe_rsi2_smoothing = preset['qqe_rsi2_smoothing'] if qqe_rsi2_smoothing == 4 else qqe_rsi2_smoothing
        self.qqe_factor_secondary = qqe_factor_secondary
        self.qqe_bb_length = preset['qqe_bb_length'] if qqe_bb_length == 25 else qqe_bb_length
        self.qqe_bb_mult = qqe_bb_mult
        self.qqe_threshold = qqe_threshold
        self.qqe_consecutive_bars = qqe_consecutive_bars
        self.use_qqe_momentum = use_qqe_momentum
        
        self.vol_lookback = preset['vol_lookback'] if vol_lookback == 12 else vol_lookback
        
        # IB
        self.ib_period_minutes = preset.get('ib_period_minutes', ib_period_minutes) if ib_period_minutes == 30 else ib_period_minutes
        self.ib_buffer_atr_mult = ib_buffer_atr_mult
        
        # Quality gates
        self.use_htf_trend_filter = use_htf_trend_filter
        self.htf_trend_length = htf_trend_length
        self.use_market_regime_filter = use_market_regime_filter
        self.use_volatility_filter = use_volatility_filter
        self.avoid_lunch_hour = avoid_lunch_hour
        self.min_bars_between_entries = min_bars_between_entries
        
        # Stops (identical to ORB)
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.swing_lookback = swing_lookback
        self.swing_buffer = swing_buffer
        self.vwap_stop_distance = vwap_stop_distance
        self.min_stop_atr = min_stop_atr
        self.max_target_atr = max_target_atr
        self.min_acceptable_rr = min_acceptable_rr
        self.profit_target_rr = profit_target_rr
        
        # Exits (identical to ORB)
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
        
        # Vol thresholds
        self.low_vol_threshold = preset['low_vol_threshold'] if low_vol_threshold == 0.8 else low_vol_threshold
        self.high_vol_threshold = preset['high_vol_threshold'] if high_vol_threshold == 1.3 else high_vol_threshold
        self.extreme_vol_threshold = preset['extreme_vol_threshold'] if extreme_vol_threshold == 2.0 else extreme_vol_threshold
        self.skip_low_vol_c_grades = skip_low_vol_c_grades
        self.low_vol_min_rr = low_vol_min_rr
        
        self.verbose = verbose
        self.bar_size = bar_size
        
        if self.verbose:
            print(f"[HMM Presets] {symbol}: SSL_base={self.ssl_baseline_length}, WAE_sens={self.wae_sensitivity}, "
                  f"QQE_rsi1={self.qqe_rsi1_length}, IB_min={self.ib_period_minutes}")
        
        # State objects
        self.ib = IBState()
        self.vol = VolState()
        self.position = Position()
        
        # Confluence calculator (reuses ORB's engine)
        self.confluence_calc = ConfluenceCalculator({
            'ssl_baseline_length': self.ssl_baseline_length,
            'ssl_length': self.ssl_length,
            'ssl_type': self.ssl_type,
            'use_ssl_momentum': self.use_ssl_momentum,
            'ssl_smoothing_enabled': True,       # Pine default: true
            'ssl_smoothing_method': 'Super Smoother',  # Pine default
            'ssl_smoothing_factor': 2.0,         # Pine default
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
            'vol_lookback': 20,  # Pine: volLookback=20 (fixed), NOT volLookbackBarsEff
            'min_confluence': self.min_confluence,
        })
        
        # Results
        self.trades: List[TradeRecord] = []
        self.skips: List[SkipRecord] = []
        self.df = None
        self.bars_since_last_entry = 999
    
    # ═══════════════════════════════════════════════════════════════════════
    # DATA LOADING & INDICATORS (identical to ORB)
    # ═══════════════════════════════════════════════════════════════════════
    
    def load_data(self):
        """Load data and calculate all indicators"""
        from data_collector import PolygonDataCollector
        
        collector = PolygonDataCollector()
        df_1min = collector.fetch_bars(self.symbol, days_back=180, bar_size=1, extended_hours=True)
        
        if self.verbose:
            print(f"  Loaded {len(df_1min)} 1-min bars (ETH+RTH) for {self.symbol}")
        
        # Filter to RTH only (9:30-16:00 ET) — Pine on standard charts only sees RTH bars
        # All indicators must compute on the same data Pine sees
        df_rth = df_1min[df_1min.index.map(
            lambda x: 9*60+30 <= x.hour*60+x.minute < 16*60
        )].copy()
        
        if self.verbose:
            print(f"  Filtered to {len(df_rth)} RTH bars")
        
        # Resample to chart timeframe (Pine HHM runs on 2-min charts)
        if self.bar_size > 1:
            df_rth = self._resample_bars(df_rth, self.bar_size)
            if self.verbose:
                print(f"  Resampled to {len(df_rth)} {self.bar_size}-min bars")
        
        self.df = df_rth
        self._calc_indicators()
    
    def _resample_bars(self, df: pd.DataFrame, bar_size: int) -> pd.DataFrame:
        """
        Resample 1-min OHLCV to N-min bars for Pine chart parity.
        Pine indicators compute on the chart's timeframe bars.
        """
        resampled = df.resample(f'{bar_size}min', closed='left', label='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(subset=['open'])
        
        return resampled
    
    def _calc_indicators(self):
        """Calculate all indicators on RTH-only bars (matches Pine on standard chart)"""
        df = self.df.copy()
        
        # All bars are RTH — no ETH/RTH split needed
        df['is_rth'] = True
        
        # === TR and ATR (Wilder RMA, same as Pine ta.atr) ===
        df['prev_close'] = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['prev_close']).abs()
        tr3 = (df['low'] - df['prev_close']).abs()
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        def wilder_rma(series, n):
            rma = series.copy().astype(float)
            rma.iloc[:n] = series.iloc[:n].expanding().mean()
            for i in range(n, len(series)):
                rma.iloc[i] = (rma.iloc[i-1] * (n - 1) + series.iloc[i]) / n
            return rma
        
        df['atr_rth'] = wilder_rma(df['tr'], self.atr_period)
        df['atr_eth'] = df['atr_rth']  # same thing now, kept for compatibility
        df['atr_fast_rth'] = wilder_rma(df['tr'], 5)
        df['atr_fast_sma_rth'] = df['atr_fast_rth'].rolling(10).mean()
        df['atr_fast_all'] = df['atr_fast_rth']
        df['atr_fast_sma_all'] = df['atr_fast_sma_rth']
        
        # === Daily ATR for HTF VF ===
        df['date'] = df.index.date
        daily = df.groupby('date').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        })
        daily['prev_close'] = daily['close'].shift(1)
        daily_tr1 = daily['high'] - daily['low']
        daily_tr2 = (daily['high'] - daily['prev_close']).abs()
        daily_tr3 = (daily['low'] - daily['prev_close']).abs()
        daily['tr'] = pd.concat([daily_tr1, daily_tr2, daily_tr3], axis=1).max(axis=1)
        daily['daily_atr'] = wilder_rma(daily['tr'], 14)
        daily['daily_atr_slow'] = daily['daily_atr'].rolling(20).mean()
        
        df['daily_atr'] = df['date'].map(daily['daily_atr'].to_dict())
        df['daily_atr_slow'] = df['date'].map(daily['daily_atr_slow'].to_dict())
        df['daily_atr'] = df['daily_atr'].ffill()
        df['daily_atr_slow'] = df['daily_atr_slow'].ffill()
        
        # === VWAP (resets at 9:30) ===
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']
        df['is_930'] = (df.index.hour == 9) & (df.index.minute == 30)
        df['session_id'] = df['is_930'].cumsum()
        df['cum_tp_vol'] = df.groupby('session_id')['tp_volume'].cumsum()
        df['cum_vol'] = df.groupby('session_id')['volume'].cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        
        # === EMA ===
        df['ema9'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        
        # === HTF TREND (5-min EMA on RTH closes) ===
        # Pine: htfEma = request.security(syminfo.tickerid, "5", ta.ema(close, htfTrendLength))
        # All bars are RTH, resample to 5-min and compute EMA
        rth_5min = df['close'].resample('5min').last().dropna()
        htf_ema = rth_5min.ewm(span=self.htf_trend_length, adjust=False).mean()
        df['htf_ema'] = htf_ema.reindex(df.index, method='ffill')
        df['htf_ema'] = df['htf_ema'].ffill()
        
        # === Volatility condition (Pine: ATR acceleration or 80% of 20-period avg) ===
        df['atr_pct'] = df['atr_rth'] / df['close']
        df['atr_pct_avg'] = df['atr_pct'].rolling(20).mean()
        
        # === CONFLUENCE INDICATORS ===
        if self.verbose:
            print("  [*] Calculating confluence indicators (SSL, WAE, QQE, Volume)...")
        df = self.confluence_calc.compute_indicators(df)
        if self.verbose:
            print("  [OK] Confluence indicators calculated")
        
        self.df = df
    
    # ═══════════════════════════════════════════════════════════════════════
    # HELPER FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    def _is_rth(self, ts) -> bool:
        bar_minutes = ts.hour * 60 + ts.minute
        return 9*60+30 <= bar_minutes < 16*60
    
    def _is_ib_window(self, ts) -> bool:
        """IB window: 9:30 to 9:30 + ib_period_minutes"""
        bar_minutes = ts.hour * 60 + ts.minute
        return 9*60+30 <= bar_minutes < 9*60+30 + self.ib_period_minutes
    
    def _is_entry_window(self, ts) -> bool:
        """Entry window: after IB completes (10:30 default) through 15:55"""
        bar_minutes = ts.hour * 60 + ts.minute
        # Entry starts after IB period, ends at 15:55
        entry_start = 9*60+30 + self.ib_period_minutes
        return entry_start <= bar_minutes < 15*60+55
    
    def _is_lunch_hour(self, ts) -> bool:
        """Lunch skip: 11:30-13:30 ET"""
        bar_minutes = ts.hour * 60 + ts.minute
        return 11*60+30 <= bar_minutes < 13*60+30
    
    def _check_volatility_condition(self, bar: pd.Series) -> bool:
        """Pine: volatilityAccelerating OR minAcceptableVolatility"""
        if not self.use_volatility_filter:
            return True
        atr_pct = bar.get('atr_pct', 0)
        atr_pct_avg = bar.get('atr_pct_avg', atr_pct)
        if pd.isna(atr_pct) or pd.isna(atr_pct_avg) or atr_pct_avg <= 0:
            return True  # Don't filter when data insufficient
        return atr_pct >= atr_pct_avg * 0.8
    
    def _check_htf_trend(self, bar: pd.Series, is_long: bool) -> bool:
        """Pine: htfTrendBullish = close > htfEma (5-min)"""
        if not self.use_htf_trend_filter:
            return True
        htf_ema = bar.get('htf_ema', np.nan)
        if pd.isna(htf_ema):
            return True
        if is_long:
            return bar['close'] > htf_ema
        else:
            return bar['close'] < htf_ema
    
    def _calc_vol_state(self, bar: pd.Series, ts) -> None:
        """Volatility state calculation (identical to ORB)"""
        atr_fast = bar.get('atr_fast_rth', bar.get('atr_rth', 1.0))
        atr_fast_sma = bar.get('atr_fast_sma_rth', atr_fast)
        if pd.isna(atr_fast):
            atr_fast = bar.get('atr_rth', 1.0)
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
        
        daily_atr = bar.get('daily_atr', 1.0)
        daily_atr_slow = bar.get('daily_atr_slow', daily_atr)
        if pd.isna(daily_atr):
            daily_atr = 1.0
        if pd.isna(daily_atr_slow) or daily_atr_slow <= 0:
            daily_atr_slow = daily_atr
        
        self.vol.htf_vf = daily_atr / daily_atr_slow if daily_atr_slow > 0 else 1.0
        self.vol.vol_factor = self.vol.session_vf * self.vol.htf_vf
        
        if self.vol.vol_factor < self.low_vol_threshold:
            self.vol.vol_state = "LOW"
        elif self.vol.vol_factor >= self.extreme_vol_threshold:
            self.vol.vol_state = "EXTREME"
        elif self.vol.vol_factor >= self.high_vol_threshold:
            self.vol.vol_state = "HIGH"
        else:
            self.vol.vol_state = "NORMAL"
    
    # ═══════════════════════════════════════════════════════════════════════
    # IB PHASE SCORING (Edgeful retracement method)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _update_ib_phase(self, bar: pd.Series) -> None:
        """
        Calculate IB phase scores based on price position relative to
        25%/50% retracement zones after IB break.
        
        Pine amendment: IB PHASE-BASED SCORING
        """
        if not self.ib.ib_formed:
            return
        
        close = bar['close']
        atr = bar['atr_rth']
        ib_buffer = atr * self.ib_buffer_atr_mult if not pd.isna(atr) else 0
        
        ib_range = self.ib.ib_high - self.ib.ib_low
        if ib_range <= 0:
            return
        
        # Track persistent breaks
        if close > (self.ib.ib_high + ib_buffer) and not self.ib.ever_broke_up:
            self.ib.ever_broke_up = True
        if close < (self.ib.ib_low - ib_buffer) and not self.ib.ever_broke_down:
            self.ib.ever_broke_down = True
        
        # Reset scores each bar
        self.ib.score_bull = 0.0
        self.ib.score_bear = 0.0
        self.ib.phase_bull = "---"
        self.ib.phase_bear = "---"
        
        # Bull phase (after upside break)
        if self.ib.ever_broke_up:
            retrace25 = self.ib.ib_high - ib_range * 0.25
            retrace50 = self.ib.ib_high - ib_range * 0.50
            
            if self.use_ib_phase_scoring:
                if close > self.ib.ib_high:
                    self.ib.score_bull = 1.0
                    self.ib.phase_bull = "IMP"
                elif close >= retrace25:
                    self.ib.score_bull = 1.25
                    self.ib.phase_bull = "PB25"
                elif close >= retrace50:
                    self.ib.score_bull = 0.5
                    self.ib.phase_bull = "PB50"
                else:
                    self.ib.score_bull = 0.0
                    self.ib.phase_bull = "FAIL"
            else:
                # Legacy binary
                self.ib.score_bull = 1.0 if close > (self.ib.ib_high + ib_buffer) else 0.0
                self.ib.phase_bull = "BRK" if self.ib.score_bull > 0 else "---"
        
        # Bear phase (after downside break)
        if self.ib.ever_broke_down:
            retrace25 = self.ib.ib_low + ib_range * 0.25
            retrace50 = self.ib.ib_low + ib_range * 0.50
            
            if self.use_ib_phase_scoring:
                if close < self.ib.ib_low:
                    self.ib.score_bear = 1.0
                    self.ib.phase_bear = "IMP"
                elif close <= retrace25:
                    self.ib.score_bear = 1.25
                    self.ib.phase_bear = "PB25"
                elif close <= retrace50:
                    self.ib.score_bear = 0.5
                    self.ib.phase_bear = "PB50"
                else:
                    self.ib.score_bear = 0.0
                    self.ib.phase_bear = "FAIL"
            else:
                self.ib.score_bear = 1.0 if close < (self.ib.ib_low - ib_buffer) else 0.0
                self.ib.phase_bear = "BRK" if self.ib.score_bear > 0 else "---"
    
    # ═══════════════════════════════════════════════════════════════════════
    # STOP CALCULATIONS (identical to ORB)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _calc_swing_stop(self, day_df, i, is_long, atr):
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
    
    def _calc_stops(self, entry, atr, vwap, is_long, day_df, i):
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
    
    def _calc_achievable_rr(self, entry, stop, atr, rr_desired):
        risk = abs(entry - stop)
        if risk <= 0:
            return -1
        target_distance = risk * rr_desired
        target_atrs = target_distance / atr
        if target_atrs > self.max_target_atr + 1e-9:
            return (self.max_target_atr * atr) / risk
        return rr_desired
    
    def _select_best_stop(self, stops, entry, atr, rr_desired, is_long):
        candidates = []
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
        
        best_rr = max(c[2] for c in candidates)
        for c in candidates:
            if c[2] == best_rr:
                return c
        return candidates[0]
    
    # ═══════════════════════════════════════════════════════════════════════
    # EXIT LOGIC (identical to ORB)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _in_position(self) -> bool:
        return self.position.direction != ""
    
    def _check_stop_hit(self, bar):
        if not self._in_position():
            return False
        if self.position.direction == "LONG":
            return bar['low'] <= self.position.current_stop
        else:
            return bar['high'] >= self.position.current_stop
    
    def _check_break_even(self, bar):
        if not self._in_position() or not self.use_break_even or self.position.stop_moved_to_be:
            return False
        
        be_rr = self.break_even_rr
        if self.use_adaptive_be:
            if self.position.vol_state == "LOW":
                be_rr = self.break_even_rr + 0.2
            elif self.position.vol_state == "HIGH":
                be_rr = self.break_even_rr - 0.1
            elif self.position.vol_state == "EXTREME":
                be_rr = self.break_even_rr - 0.2
        
        be_target_distance = self.position.risk * be_rr
        if self.position.direction == "LONG":
            return bar['close'] >= self.position.entry_price + be_target_distance
        else:
            return bar['close'] <= self.position.entry_price - be_target_distance
    
    def _activate_break_even(self):
        self.position.stop_moved_to_be = True
        self.position.current_stop = self.position.entry_price
    
    def _check_trailing_activation(self, bar):
        if not self._in_position() or not self.use_trailing_stop:
            return False
        if not self.position.stop_moved_to_be or self.position.trailing_activated:
            return False
        
        profit_target_distance = self.position.risk * self.profit_target_rr
        if self.position.direction == "LONG":
            return bar['close'] >= self.position.entry_price + profit_target_distance
        else:
            return bar['close'] <= self.position.entry_price - profit_target_distance
    
    def _activate_trailing(self, bar, atr):
        self.position.trailing_activated = True
        if self.position.direction == "LONG":
            self.position.trailing_stop_level = bar['close'] - (atr * self.trailing_stop_distance)
        else:
            self.position.trailing_stop_level = bar['close'] + (atr * self.trailing_stop_distance)
        self.position.current_stop = self.position.trailing_stop_level
    
    def _update_trailing_stop(self, bar, atr, ema):
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
    
    def _check_ema_exit(self, bar, ema):
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
    
    def _close_position(self, exit_price, exit_reason, time_str, date_str):
        if self.position.direction == "LONG":
            pnl = exit_price - self.position.entry_price
        else:
            pnl = self.position.entry_price - exit_price
        
        r_multiple = pnl / self.position.risk if self.position.risk > 0 else 0
        
        trade = TradeRecord(
            date=date_str, direction=self.position.direction,
            entry_price=self.position.entry_price, entry_time=self.position.entry_time,
            exit_price=exit_price, exit_time=time_str, exit_reason=exit_reason,
            initial_stop=self.position.initial_stop, stop_type=self.position.stop_type,
            vol_state=self.position.vol_state, risk=self.position.risk,
            pnl=pnl, r_multiple=r_multiple,
            confluence_score=self.position.confluence_score,
            ssl_score=self.position.ssl_score, wae_score=self.position.wae_score,
            qqe_score=self.position.qqe_score, vol_score=self.position.vol_score,
            ib_score=self.position.ib_score, ib_phase=self.position.ib_phase,
        )
        self.trades.append(trade)
        
        if self.verbose:
            emoji = "WIN" if pnl > 0 else "LOSS"
            print(f"  [{emoji}] {self.position.direction} {date_str} {self.position.entry_time}->{time_str}: "
                  f"${pnl:+.2f} ({r_multiple:+.2f}R) - {exit_reason} "
                  f"[C:{self.position.confluence_score:.1f} IB:{self.position.ib_phase}]")
        
        self.position.reset()
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAIN PROCESSING LOOP (THE HMM DIFFERENCE)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _process_day(self, date_str: str):
        """Process a single trading day - HMM confluence-driven entry"""
        day_df = self.df[self.df.index.date.astype(str) == date_str].copy()
        
        if len(day_df) == 0:
            return
        
        # Reset session state
        self.ib.reset_session()
        self.vol.orb_session_baseline = np.nan
        
        if self.verbose:
            print(f"\n--- {date_str} ---")
        
        for i in range(len(day_df)):
            ts = day_df.index[i]
            bar = day_df.iloc[i]
            
            if not self._is_rth(ts):
                continue
            
            time_str = ts.strftime('%H:%M')
            self.bars_since_last_entry += 1
            
            atr = bar['atr_rth']
            ema = bar['ema9']
            
            # ═══════════════════════════════════════════════════════════
            # EXIT CHECKS (identical to ORB - stop first, then BE/trail)
            # ═══════════════════════════════════════════════════════════
            
            if self._in_position():
                if self._check_stop_hit(bar):
                    exit_price = self.position.current_stop
                    reason = "TRAILING STOP" if self.position.trailing_activated else \
                             "BREAK-EVEN STOP" if self.position.stop_moved_to_be else "INITIAL STOP"
                    self._close_position(exit_price, reason, time_str, date_str)
                elif self._check_ema_exit(bar, ema):
                    self._close_position(bar['close'], "EMA EXIT", time_str, date_str)
                
                if self._in_position():
                    if self._check_break_even(bar):
                        self._activate_break_even()
                    if self._check_trailing_activation(bar):
                        self._activate_trailing(bar, atr)
                    if self.position.trailing_activated:
                        self._update_trailing_stop(bar, atr, ema)
            
            # ═══════════════════════════════════════════════════════════
            # VOL STATE
            # ═══════════════════════════════════════════════════════════
            self._calc_vol_state(bar, ts)
            
            # ═══════════════════════════════════════════════════════════
            # IB BUILDING (replaces ORB building)
            # ═══════════════════════════════════════════════════════════
            
            if self._is_ib_window(ts):
                if ts.hour == 9 and ts.minute == 30:
                    self.ib.reset_session()
                    self.ib.ib_high = bar['high']
                    self.ib.ib_low = bar['low']
                else:
                    if not pd.isna(self.ib.ib_high):
                        self.ib.ib_high = max(self.ib.ib_high, bar['high'])
                        self.ib.ib_low = min(self.ib.ib_low, bar['low'])
                    else:
                        self.ib.ib_high = bar['high']
                        self.ib.ib_low = bar['low']
                continue  # Don't enter during IB formation
            
            # Mark IB as formed once we exit the window
            if not self.ib.ib_formed and not pd.isna(self.ib.ib_high):
                self.ib.ib_formed = True
                if self.verbose:
                    print(f"  [IB] Formed: H={self.ib.ib_high:.2f} L={self.ib.ib_low:.2f} "
                          f"Range={self.ib.ib_high - self.ib.ib_low:.2f}")
            
            # ═══════════════════════════════════════════════════════════
            # IB PHASE SCORING (Edgeful retracement zones)
            # ═══════════════════════════════════════════════════════════
            self._update_ib_phase(bar)
            
            # ═══════════════════════════════════════════════════════════
            # ENTRY EVALUATION (HMM confluence-driven)
            # No breakout trigger - signal fires when confluence met
            # ═══════════════════════════════════════════════════════════
            
            if (not self._in_position() 
                and self._is_entry_window(ts) 
                and self.bars_since_last_entry >= self.min_bars_between_entries):
                
                # Quality gates
                if self.avoid_lunch_hour and self._is_lunch_hour(ts):
                    continue
                if not self._check_volatility_condition(bar):
                    continue
                
                # Get confluence scores
                full_idx = self.df.index.get_loc(ts)
                scores = self.confluence_calc.get_scores(full_idx)
                
                # Build bull/bear total scores
                # SSL + WAE + QQE + Volume + IB (optional)
                bull_score = (scores.ssl_score_bull + scores.wae_score_bull + 
                             scores.qqe_score_bull + pine_round(scores.vol_score))
                bear_score = (scores.ssl_score_bear + scores.wae_score_bear + 
                             scores.qqe_score_bear + pine_round(scores.vol_score))
                
                if self.include_ib_in_confluence:
                    bull_score += self.ib.score_bull
                    bear_score += self.ib.score_bear
                
                # Check for LONG signal
                htf_bull = self._check_htf_trend(bar, is_long=True)
                if bull_score >= self.min_confluence and htf_bull:
                    self._try_enter(bar, day_df, i, "LONG", bull_score, scores, 
                                   date_str, time_str, atr)
                
                # Check for SHORT signal (only if not just entered long)
                if not self._in_position():
                    htf_bear = self._check_htf_trend(bar, is_long=False)
                    if bear_score >= self.min_confluence and htf_bear:
                        self._try_enter(bar, day_df, i, "SHORT", bear_score, scores,
                                       date_str, time_str, atr)
        
        # End of day - close any open position
        if self._in_position():
            rth_bars = day_df[day_df.index.map(self._is_rth)]
            if len(rth_bars) > 0:
                last_bar = rth_bars.iloc[-1]
                self._close_position(last_bar['close'], "END OF DAY", "16:00", date_str)
    
    def _try_enter(self, bar, day_df, i, direction, conf_score, scores, date_str, time_str, atr):
        """Evaluate stop/RR and enter if feasible"""
        is_long = direction == "LONG"
        entry = bar['close']
        vwap = bar['vwap']
        
        # Adaptive R:R by volatility
        rr_desired = self.profit_target_rr
        if self.vol.vol_state == "LOW":
            rr_desired += 0.5
        elif self.vol.vol_state == "HIGH":
            rr_desired = max(rr_desired - 0.5, 1.2)
        elif self.vol.vol_state == "EXTREME":
            rr_desired = max(rr_desired - 0.8, 1.0)
        
        stops = self._calc_stops(entry, atr, vwap, is_long, day_df, i)
        stop_price, stop_type, achievable_rr = self._select_best_stop(stops, entry, atr, rr_desired, is_long)
        
        # Low vol filter
        min_rr_for_entry = self.min_acceptable_rr
        if self.skip_low_vol_c_grades and self.vol.vol_state == "LOW":
            min_rr_for_entry = max(self.min_acceptable_rr, self.low_vol_min_rr)
        
        if achievable_rr >= min_rr_for_entry - 0.0001:
            # ENTER
            risk = abs(entry - stop_price)
            self.position.direction = direction
            self.position.entry_price = entry
            self.position.entry_time = time_str
            self.position.entry_date = date_str
            self.position.initial_stop = stop_price
            self.position.current_stop = stop_price
            self.position.stop_type = stop_type
            self.position.risk = risk
            self.position.target_rr = achievable_rr
            self.position.vol_state = self.vol.vol_state
            self.position.confluence_score = conf_score
            
            if is_long:
                self.position.ssl_score = scores.ssl_score_bull
                self.position.wae_score = scores.wae_score_bull
                self.position.qqe_score = scores.qqe_score_bull
                self.position.ib_score = self.ib.score_bull
                self.position.ib_phase = self.ib.phase_bull
            else:
                self.position.ssl_score = scores.ssl_score_bear
                self.position.wae_score = scores.wae_score_bear
                self.position.qqe_score = scores.qqe_score_bear
                self.position.ib_score = self.ib.score_bear
                self.position.ib_phase = self.ib.phase_bear
            self.position.vol_score = scores.vol_score
            
            self.bars_since_last_entry = 0
            
            if self.verbose:
                print(f"  [ENTRY] {direction} {time_str} @ ${entry:.2f} stop=${stop_price:.2f} ({stop_type}) "
                      f"RR={achievable_rr:.2f} vol={self.vol.vol_state} "
                      f"conf={conf_score:.1f} IB={self.position.ib_phase}")
        else:
            # SKIP
            skip = SkipRecord(
                date=date_str, time=time_str, direction=direction,
                entry_price=entry, stop_price=stop_price, stop_type=stop_type,
                achievable_rr=achievable_rr, min_rr=min_rr_for_entry,
                reason=f"RR {achievable_rr:.2f} < {min_rr_for_entry:.1f}",
                confluence_score=conf_score,
            )
            self.skips.append(skip)
            if self.verbose:
                print(f"  [SKIP] {direction} {time_str} @ ${entry:.2f}: "
                      f"RR {achievable_rr:.2f} < {min_rr_for_entry:.2f} ({stop_type})")
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAIN RUN METHOD
    # ═══════════════════════════════════════════════════════════════════════
    
    def run(self) -> Dict[str, Any]:
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  HMM BACKTESTER: {self.symbol}")
            print(f"  {self.start_date} to {self.end_date}")
            print(f"  Min Confluence: {self.min_confluence}  IB Phase: {self.use_ib_phase_scoring}")
            print(f"{'='*70}\n")
        
        self.load_data()
        
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        all_dates = self.df.index.date
        unique_dates = sorted(set(all_dates))
        
        # Data range guard
        if unique_dates:
            data_first = unique_dates[0]
            data_last = unique_dates[-1]
            if self.verbose:
                print(f"  Data range: {data_first} to {data_last} ({len(unique_dates)} days)")
            if start.date() < data_first or end.date() > data_last:
                print(f"\n  ⚠ WARNING: Requested {self.start_date} to {self.end_date} "
                      f"but data covers {data_first} to {data_last}")
                if start.date() > data_last or end.date() < data_first:
                    print(f"  ✖ NO OVERLAP — requested dates are entirely outside data range!")
                    print(f"    Data is 180 days back from today. Use recent dates.")
        
        trading_dates = [d for d in unique_dates if start.date() <= d <= end.date()]
        
        if self.verbose:
            print(f"Processing {len(trading_dates)} trading days...\n")
        
        for date in trading_dates:
            self._process_day(str(date))
        
        results = self._calc_summary()
        
        if self.verbose:
            self._print_summary(results)
        
        return results
    
    def _calc_summary(self) -> Dict[str, Any]:
        if not self.trades:
            return {
                'symbol': self.symbol, 'start_date': self.start_date, 'end_date': self.end_date,
                'total_trades': 0, 'wins': 0, 'losses': 0, 'breakevens': 0,
                'win_rate': 0, 'total_pnl': 0, 'total_r': 0, 'avg_win': 0, 'avg_loss': 0,
                'profit_factor': 0, 'max_drawdown_r': 0, 'avg_r_per_trade': 0,
                'trades': [], 'skips': []
            }
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        breakevens = [t for t in self.trades if t.pnl == 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_r = sum(t.r_multiple for t in self.trades)
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        
        cumulative_r = 0
        peak_r = 0
        max_dd_r = 0
        for t in self.trades:
            cumulative_r += t.r_multiple
            peak_r = max(peak_r, cumulative_r)
            max_dd_r = max(max_dd_r, peak_r - cumulative_r)
        
        return {
            'symbol': self.symbol, 'start_date': self.start_date, 'end_date': self.end_date,
            'total_trades': len(self.trades), 'total_skips': len(self.skips),
            'wins': len(wins), 'losses': len(losses), 'breakevens': len(breakevens),
            'win_rate': len(wins) / (len(wins) + len(losses)) * 100 if (len(wins) + len(losses)) > 0 else 0,
            'total_pnl': total_pnl, 'total_r': total_r,
            'avg_win': sum(t.pnl for t in wins) / len(wins) if wins else 0,
            'avg_loss': sum(t.pnl for t in losses) / len(losses) if losses else 0,
            'avg_win_r': sum(t.r_multiple for t in wins) / len(wins) if wins else 0,
            'avg_loss_r': sum(t.r_multiple for t in losses) / len(losses) if losses else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'max_drawdown_r': max_dd_r,
            'avg_r_per_trade': total_r / len(self.trades) if self.trades else 0,
            'trades': [
                {
                    'date': t.date, 'direction': t.direction,
                    'entry_price': t.entry_price, 'entry_time': t.entry_time,
                    'exit_price': t.exit_price, 'exit_time': t.exit_time,
                    'exit_reason': t.exit_reason, 'stop_type': t.stop_type,
                    'vol_state': t.vol_state, 'pnl': t.pnl, 'r_multiple': t.r_multiple,
                    'confluence': t.confluence_score, 'ib_phase': t.ib_phase,
                }
                for t in self.trades
            ],
            'skips': [
                {
                    'date': s.date, 'time': s.time, 'direction': s.direction,
                    'entry_price': s.entry_price, 'achievable_rr': s.achievable_rr,
                    'min_rr': s.min_rr, 'reason': s.reason,
                }
                for s in self.skips
            ]
        }
    
    def _print_summary(self, results):
        print(f"\n{'='*70}")
        print(f"  HMM BACKTEST SUMMARY: {results['symbol']}")
        print(f"  {results['start_date']} to {results['end_date']}")
        print(f"{'='*70}\n")
        
        print(f"  Total Trades:    {results['total_trades']}")
        print(f"  Total Skips:     {results.get('total_skips', 0)}")
        print(f"  Wins/Losses/BE:  {results['wins']}W / {results['losses']}L / {results.get('breakevens', 0)}BE")
        print(f"  Win Rate:        {results['win_rate']:.1f}%")
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
        
        if results['trades']:
            print(f"{'='*70}")
            print("TRADE DETAILS")
            print(f"{'='*70}")
            for t in results['trades']:
                dir_str = "LONG " if t['direction'] == "LONG" else "SHORT"
                print(f"{t['date']} {dir_str} {t['entry_time']}->{t['exit_time']}  "
                      f"${t['entry_price']:.2f}->${t['exit_price']:.2f}  "
                      f"{t['exit_reason']:<18} {t['r_multiple']:+.2f}R  "
                      f"C:{t['confluence']:.1f} IB:{t['ib_phase']}")
            print(f"\nTotal: {len(results['trades'])} trades, {results['total_r']:+.2f}R")


# ═══════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    bt = HMMBacktester(
        symbol='AMD',
        start_date='2025-10-01',
        end_date='2025-12-31',
        min_confluence=3,
        verbose=True
    )
    results = bt.run()
