"""
HMM Backtester V2 — Simplified Confluence-Driven Strategy
==========================================================
Mirrors HHM_simple.pine for backtesting parity.

KEPT from HHM: Confluence scoring (SSL, WAE, QQE, Volume Z-score),
    HTF trend filter, market regime filter, volatility condition,
    lunch hour filter, entry window, cooldown.
    Stop selection engine (ATR/Swing/VWAP auto-select-best-R:R).
    BE at fixed R:R, trailing at fixed profit target R:R,
    EMA exit after BE with confirmation bars + body percentage.

REMOVED: IB filter, SSL breach gate on EMA exit, adaptive BE,
    adaptive R:R by vol state, composite grade, per-symbol presets,
    9/20 EMA alternative exit, pre-signal feasibility engine.

Exit logic copied from ORB backtester (validated).
Entry is confluence-driven: signal on bar T close, fill on bar T+1 close.

Usage:
    from hmm_backtester_v2 import HMMBacktesterV2
    bt = HMMBacktesterV2(symbol='AMD', start_date='2025-10-01',
                          end_date='2025-12-31', verbose=True)
    results = bt.run()
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

try:
    from confluence_indicators import compute_confluence
except ImportError:
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from confluence_indicators import compute_confluence


# ═══════════════════════════════════════════════════════════════════════════
# PER-SYMBOL PRESETS (matches HHM_simple.pine switch blocks)
# ═══════════════════════════════════════════════════════════════════════════

HMM_V2_PRESETS = {
    'AAPL': {
        'ssl_baseline_length': 20, 'ssl_length': 10, 'wae_sensitivity': 200,
        'qqe_rsi1_length': 10, 'qqe_threshold': 3,
        'market_etf': 'QQQ', 'htf_trend_length': 5,
        'min_acceptable_rr': 1.5, 'break_even_rr': 0.5, 'profit_target_rr': 2.0,
        'trailing_stop_distance': 1.2, 'ema_confirmation_bars': 1, 'ema_body_percentage': 60.0,
        'min_bars_between_entries': 1,
    },
    'NVDA': {
        'ssl_baseline_length': 20, 'ssl_length': 10, 'wae_sensitivity': 325,
        'qqe_rsi1_length': 9, 'qqe_threshold': 3,
        'market_etf': 'QQQ', 'htf_trend_length': 5,
        'min_acceptable_rr': 1.5, 'break_even_rr': 0.5, 'profit_target_rr': 2.0,
        'trailing_stop_distance': 1.2, 'ema_confirmation_bars': 1, 'ema_body_percentage': 60.0,
        'min_bars_between_entries': 1,
    },
    'AMD': {
        'ssl_baseline_length': 60, 'ssl_length': 10, 'wae_sensitivity': 300,
        'qqe_rsi1_length': 8, 'qqe_threshold': 3,
        'market_etf': 'QQQ', 'htf_trend_length': 5,
        'min_acceptable_rr': 1.5, 'break_even_rr': 0.5, 'profit_target_rr': 2.0,
        'trailing_stop_distance': 1.2, 'ema_confirmation_bars': 1, 'ema_body_percentage': 60.0,
        'min_bars_between_entries': 1,
    },
    'GOOGL': {
        'ssl_baseline_length': 60, 'ssl_length': 10, 'wae_sensitivity': 200,
        'qqe_rsi1_length': 9, 'qqe_threshold': 3,
        'market_etf': 'QQQ', 'htf_trend_length': 5,
        'min_acceptable_rr': 1.5, 'break_even_rr': 0.5, 'profit_target_rr': 2.0,
        'trailing_stop_distance': 1.2, 'ema_confirmation_bars': 1, 'ema_body_percentage': 60.0,
        'min_bars_between_entries': 1,
    },
    'AMZN': {
        'ssl_baseline_length': 50, 'ssl_length': 10, 'wae_sensitivity': 225,
        'qqe_rsi1_length': 10, 'qqe_threshold': 3,
        'market_etf': 'QQQ', 'htf_trend_length': 5,
        'min_acceptable_rr': 1.5, 'break_even_rr': 0.5, 'profit_target_rr': 2.0,
        'trailing_stop_distance': 1.2, 'ema_confirmation_bars': 1, 'ema_body_percentage': 60.0,
        'min_bars_between_entries': 1,
    },
    'META': {
        'ssl_baseline_length': 60, 'ssl_length': 10, 'wae_sensitivity': 325,
        'qqe_rsi1_length': 9, 'qqe_threshold': 3,
        'market_etf': 'QQQ', 'htf_trend_length': 5,
        'min_acceptable_rr': 1.5, 'break_even_rr': 0.5, 'profit_target_rr': 2.0,
        'trailing_stop_distance': 1.2, 'ema_confirmation_bars': 1, 'ema_body_percentage': 60.0,
        'min_bars_between_entries': 1,
    },
    'MSFT': {
        'ssl_baseline_length': 60, 'ssl_length': 10, 'wae_sensitivity': 325,
        'qqe_rsi1_length': 10, 'qqe_threshold': 3,
        'market_etf': 'QQQ', 'htf_trend_length': 5,
        'min_acceptable_rr': 1.5, 'break_even_rr': 0.5, 'profit_target_rr': 2.0,
        'trailing_stop_distance': 1.2, 'ema_confirmation_bars': 1, 'ema_body_percentage': 60.0,
        'min_bars_between_entries': 1,
    },
    'NFLX': {
        'ssl_baseline_length': 60, 'ssl_length': 10, 'wae_sensitivity': 200,
        'qqe_rsi1_length': 9, 'qqe_threshold': 3,
        'market_etf': 'QQQ', 'htf_trend_length': 5,
        'min_acceptable_rr': 1.5, 'break_even_rr': 0.5, 'profit_target_rr': 2.0,
        'trailing_stop_distance': 1.2, 'ema_confirmation_bars': 1, 'ema_body_percentage': 60.0,
        'min_bars_between_entries': 1,
    },
    'ORCL': {
        'ssl_baseline_length': 20, 'ssl_length': 10, 'wae_sensitivity': 300,
        'qqe_rsi1_length': 8, 'qqe_threshold': 3,
        'market_etf': 'QQQ', 'htf_trend_length': 5,
        'min_acceptable_rr': 1.5, 'break_even_rr': 0.5, 'profit_target_rr': 2.0,
        'trailing_stop_distance': 1.2, 'ema_confirmation_bars': 1, 'ema_body_percentage': 60.0,
        'min_bars_between_entries': 1,
    },
    'PLTR': {
        'ssl_baseline_length': 20, 'ssl_length': 10, 'wae_sensitivity': 325,
        'qqe_rsi1_length': 8, 'qqe_threshold': 3,
        'market_etf': 'QQQ', 'htf_trend_length': 5,
        'min_acceptable_rr': 1.5, 'break_even_rr': 0.5, 'profit_target_rr': 2.0,
        'trailing_stop_distance': 1.2, 'ema_confirmation_bars': 1, 'ema_body_percentage': 60.0,
        'min_bars_between_entries': 1,
    },
    'TSLA': {
        'ssl_baseline_length': 40, 'ssl_length': 10, 'wae_sensitivity': 275,
        'qqe_rsi1_length': 8, 'qqe_threshold': 3,
        'market_etf': 'QQQ', 'htf_trend_length': 5,
        'min_acceptable_rr': 1.5, 'break_even_rr': 0.5, 'profit_target_rr': 2.0,
        'trailing_stop_distance': 1.2, 'ema_confirmation_bars': 1, 'ema_body_percentage': 60.0,
        'min_bars_between_entries': 1,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

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
    """Active position with exit management (ORB-compatible)"""
    direction: str = ""
    entry_price: float = np.nan
    entry_time: str = ""
    entry_date: str = ""
    entry_bar_idx: int = -1
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
        self.entry_bar_idx = -1
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
    confluence_score: float = 0.0
    ssl_score: float = 0.0
    wae_score: float = 0.0
    qqe_score: float = 0.0
    vol_score: float = 0.0


@dataclass
class SkipRecord:
    """Skipped trade record"""
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

class HMMBacktesterV2:
    """
    Simplified HMM backtester — mirrors HHM_simple.pine.
    Confluence-driven entry, ORB-compatible exit cascade.
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

        # SSL Hybrid
        ssl_baseline_length: int = 25,
        ssl_length: int = 5,
        ssl_type: str = 'JMA',
        use_ssl_momentum: bool = True,

        # WAE
        wae_fast_ema: int = 15,
        wae_slow_ema: int = 30,
        wae_sensitivity: int = 275,
        wae_bb_length: int = 20,
        wae_bb_mult: float = 2.0,
        use_wae_acceleration: bool = True,

        # QQE
        qqe_rsi1_length: int = 9,
        qqe_rsi1_smoothing: int = 5,
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
        vol_lookback: int = 20,

        # ═══════════════════════════════════════════════════════════════
        # QUALITY GATES
        # ═══════════════════════════════════════════════════════════════
        use_htf_trend_filter: bool = True,
        htf_trend_length: int = 5,
        use_market_regime_filter: bool = True,
        market_etf: str = 'SPY',
        market_trend_length: int = 10,
        use_volatility_filter: bool = True,
        avoid_lunch_hour: bool = True,
        min_bars_between_entries: int = 1,

        # ═══════════════════════════════════════════════════════════════
        # STOP PARAMETERS
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
        # EXIT PARAMETERS
        # ═══════════════════════════════════════════════════════════════
        use_break_even: bool = True,
        break_even_rr: float = 0.5,
        use_trailing_stop: bool = True,
        trailing_stop_distance: float = 1.2,
        use_ema_exit: bool = True,
        ema_period: int = 9,
        ema_confirmation_bars: int = 1,
        ema_body_percentage: float = 60.0,
        use_aggressive_trailing: bool = True,
        ema_tighten_zone: float = 0.3,
        tightened_trail_distance: float = 0.3,

        # ═══════════════════════════════════════════════════════════════
        # VOLATILITY THRESHOLDS
        # ═══════════════════════════════════════════════════════════════
        low_vol_threshold: float = 0.8,
        high_vol_threshold: float = 1.3,
        extreme_vol_threshold: float = 2.0,

        # ═══════════════════════════════════════════════════════════════
        # OUTPUT
        # ═══════════════════════════════════════════════════════════════
        verbose: bool = False,
        bar_size: int = 2,
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

        # Get per-symbol preset (empty dict if no preset for this symbol)
        preset = HMM_V2_PRESETS.get(symbol, {})

        # Helper: use preset value when constructor arg is at its default
        def p(key, arg, default):
            return preset.get(key, arg) if arg == default else arg

        # Confluence
        self.min_confluence = min_confluence

        self.ssl_baseline_length = p('ssl_baseline_length', ssl_baseline_length, 25)
        self.ssl_length = p('ssl_length', ssl_length, 5)
        self.ssl_type = ssl_type
        self.use_ssl_momentum = use_ssl_momentum

        self.wae_fast_ema = wae_fast_ema
        self.wae_slow_ema = wae_slow_ema
        self.wae_sensitivity = p('wae_sensitivity', wae_sensitivity, 275)
        self.wae_bb_length = wae_bb_length
        self.wae_bb_mult = wae_bb_mult
        self.use_wae_acceleration = use_wae_acceleration

        self.qqe_rsi1_length = p('qqe_rsi1_length', qqe_rsi1_length, 9)
        self.qqe_rsi1_smoothing = qqe_rsi1_smoothing
        self.qqe_factor_primary = qqe_factor_primary
        self.qqe_rsi2_length = qqe_rsi2_length
        self.qqe_rsi2_smoothing = qqe_rsi2_smoothing
        self.qqe_factor_secondary = qqe_factor_secondary
        self.qqe_bb_length = qqe_bb_length
        self.qqe_bb_mult = qqe_bb_mult
        self.qqe_threshold = p('qqe_threshold', qqe_threshold, 3)
        self.qqe_consecutive_bars = qqe_consecutive_bars
        self.use_qqe_momentum = use_qqe_momentum

        self.vol_lookback = vol_lookback

        # Quality gates
        self.use_htf_trend_filter = use_htf_trend_filter
        self.htf_trend_length = p('htf_trend_length', htf_trend_length, 5)
        self.use_market_regime_filter = use_market_regime_filter
        self.market_etf = p('market_etf', market_etf, 'SPY')
        self.market_trend_length = market_trend_length
        self.use_volatility_filter = use_volatility_filter
        self.avoid_lunch_hour = avoid_lunch_hour
        self.min_bars_between_entries = p('min_bars_between_entries', min_bars_between_entries, 1)

        # Stops
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.swing_lookback = swing_lookback
        self.swing_buffer = swing_buffer
        self.vwap_stop_distance = vwap_stop_distance
        self.min_stop_atr = min_stop_atr
        self.max_target_atr = max_target_atr
        self.min_acceptable_rr = p('min_acceptable_rr', min_acceptable_rr, 1.5)
        self.profit_target_rr = p('profit_target_rr', profit_target_rr, 2.0)

        # Exits (fixed — no adaptive BE or adaptive R:R)
        self.use_break_even = use_break_even
        self.break_even_rr = p('break_even_rr', break_even_rr, 0.5)
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_distance = p('trailing_stop_distance', trailing_stop_distance, 1.2)
        self.use_ema_exit = use_ema_exit
        self.ema_period = ema_period
        self.ema_confirmation_bars = p('ema_confirmation_bars', ema_confirmation_bars, 1)
        self.ema_body_percentage = p('ema_body_percentage', ema_body_percentage, 60.0)
        self.use_aggressive_trailing = use_aggressive_trailing
        self.ema_tighten_zone = ema_tighten_zone
        self.tightened_trail_distance = tightened_trail_distance

        # Vol thresholds
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.extreme_vol_threshold = extreme_vol_threshold

        self.verbose = verbose
        self.bar_size = bar_size

        if self.verbose and preset:
            print(f"[V2 Presets] {symbol}: SSL_base={self.ssl_baseline_length}, "
                  f"WAE_sens={self.wae_sensitivity}, QQE_rsi1={self.qqe_rsi1_length}, "
                  f"BE_rr={self.break_even_rr}, trail_dist={self.trailing_stop_distance}")

        # State objects
        self.vol = VolState()
        self.position = Position()

        # Confluence indicator kwargs
        self.ssl_kwargs = dict(
            baseline_type="HMA",
            baseline_length=self.ssl_baseline_length,
            ssl2_type=self.ssl_type,
            ssl2_length=self.ssl_length,
            use_momentum_filter=self.use_ssl_momentum,
            smoothing_enabled=True,
            smoothing_method='Super Smoother',
            smoothing_factor=2.0,
        )
        self.wae_kwargs = dict(
            fast_ema=self.wae_fast_ema,
            slow_ema=self.wae_slow_ema,
            sensitivity=self.wae_sensitivity,
            bb_length=self.wae_bb_length,
            bb_multiplier=self.wae_bb_mult,
            use_acceleration=self.use_wae_acceleration,
        )
        self.qqe_kwargs = dict(
            rsi1_length=self.qqe_rsi1_length,
            rsi1_smoothing=self.qqe_rsi1_smoothing,
            qqe_factor1=self.qqe_factor_primary,
            rsi2_length=self.qqe_rsi2_length,
            rsi2_smoothing=self.qqe_rsi2_smoothing,
            qqe_factor2=self.qqe_factor_secondary,
            bb_length=self.qqe_bb_length,
            bb_multiplier=self.qqe_bb_mult,
            threshold_secondary=self.qqe_threshold,
            consecutive_increasing_bars=self.qqe_consecutive_bars,
            use_momentum_filter=self.use_qqe_momentum,
        )
        self.vol_kwargs = dict(lookback=self.vol_lookback, binary=True)

        # Results
        self.trades: List[TradeRecord] = []
        self.skips: List[SkipRecord] = []
        self.df = None
        self.bars_since_last_entry = 999
        self._bar_counter = 0
        self._prev_long_qualified = False
        self._prev_short_qualified = False

    # ═══════════════════════════════════════════════════════════════════════
    # DATA LOADING & INDICATORS
    # ═══════════════════════════════════════════════════════════════════════

    def load_data(self):
        """Load data and calculate all indicators"""
        from data_collector import PolygonDataCollector

        collector = PolygonDataCollector()
        df_1min = collector.fetch_bars(self.symbol, days_back=180, bar_size=1, extended_hours=True)

        if self.verbose:
            print(f"  Loaded {len(df_1min)} 1-min bars (ETH+RTH) for {self.symbol}")

        # Filter to RTH only (9:30-16:00 ET)
        df_rth = df_1min[df_1min.index.map(
            lambda x: 9*60+30 <= x.hour*60+x.minute < 16*60
        )].copy()

        if self.verbose:
            print(f"  Filtered to {len(df_rth)} RTH bars")

        # Keep 1-min RTH closes for clean 5-min HTF resample
        self._rth_1min_close = df_rth['close'].copy()

        # Resample to chart timeframe
        if self.bar_size > 1:
            df_rth = self._resample_bars(df_rth, self.bar_size)
            if self.verbose:
                print(f"  Resampled to {len(df_rth)} {self.bar_size}-min bars")

        self.df = df_rth

        # Load market ETF data for regime filter
        if self.use_market_regime_filter:
            mkt_1min = collector.fetch_bars(self.market_etf, days_back=180, bar_size=1, extended_hours=True)
            mkt_rth = mkt_1min[mkt_1min.index.map(
                lambda x: 9*60+30 <= x.hour*60+x.minute < 16*60
            )].copy()
            if self.bar_size > 1:
                mkt_rth = self._resample_bars(mkt_rth, self.bar_size)
            mkt_ema = mkt_rth['close'].ewm(span=self.market_trend_length, adjust=False).mean()
            self.df['market_close'] = mkt_rth['close'].reindex(self.df.index, method='ffill')
            self.df['market_ema'] = mkt_ema.reindex(self.df.index, method='ffill')
            self.df['market_close'] = self.df['market_close'].ffill()
            self.df['market_ema'] = self.df['market_ema'].ffill()
            if self.verbose:
                print(f"  Loaded {self.market_etf} market regime data ({len(mkt_rth)} bars)")

        self._calc_indicators()

    def _resample_bars(self, df: pd.DataFrame, bar_size: int) -> pd.DataFrame:
        """Resample 1-min OHLCV to N-min bars"""
        resampled = df.resample(f'{bar_size}min', closed='left', label='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(subset=['open'])
        return resampled

    def _calc_indicators(self):
        """Calculate all indicators on RTH-only bars"""
        df = self.df.copy()
        df['is_rth'] = True

        # === TR and ATR (Wilder RMA) ===
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
        df['atr_fast_rth'] = wilder_rma(df['tr'], 5)
        df['atr_fast_sma_rth'] = df['atr_fast_rth'].rolling(10).mean()

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

        # === HTF TREND (5-min EMA from 1-min data) ===
        rth_5min = self._rth_1min_close.resample('5min').last().dropna()
        htf_ema = rth_5min.ewm(span=self.htf_trend_length, adjust=False).mean()
        df['htf_ema'] = htf_ema.reindex(df.index, method='ffill')
        df['htf_ema'] = df['htf_ema'].ffill()
        df['htf_close'] = df['close']

        # === Volatility condition ===
        df['atr_pct'] = df['atr_rth'] / df['close']
        df['atr_pct_avg'] = df['atr_pct'].rolling(20).mean()

        # === CONFLUENCE INDICATORS ===
        if self.verbose:
            print("  [*] Calculating confluence indicators (SSL, WAE, QQE, Volume)...")
        confluence_df = compute_confluence(
            df, ssl_kwargs=self.ssl_kwargs, wae_kwargs=self.wae_kwargs,
            qqe_kwargs=self.qqe_kwargs, vol_kwargs=self.vol_kwargs,
        )
        df = pd.concat([df, confluence_df], axis=1)
        if self.verbose:
            print("  [OK] Confluence indicators calculated")

        self.df = df

    # ═══════════════════════════════════════════════════════════════════════
    # HELPER FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════

    def _is_rth(self, ts) -> bool:
        bar_minutes = ts.hour * 60 + ts.minute
        return 9*60+30 <= bar_minutes < 16*60

    def _is_entry_window(self, ts) -> bool:
        """Entry window: 10:30-15:55"""
        bar_minutes = ts.hour * 60 + ts.minute
        return 10*60+30 <= bar_minutes < 15*60+55

    def _is_lunch_hour(self, ts) -> bool:
        """Lunch skip: 11:30-13:30 ET"""
        bar_minutes = ts.hour * 60 + ts.minute
        return 11*60+30 <= bar_minutes < 13*60+30

    def _check_volatility_condition(self, bar: pd.Series) -> bool:
        """Pine: ATR acceleration or 80% of 20-period avg"""
        if not self.use_volatility_filter:
            return True
        atr_pct = bar.get('atr_pct', 0)
        atr_pct_avg = bar.get('atr_pct_avg', atr_pct)
        if pd.isna(atr_pct) or pd.isna(atr_pct_avg) or atr_pct_avg <= 0:
            return True
        return atr_pct >= atr_pct_avg * 0.8

    def _check_htf_trend(self, bar: pd.Series, is_long: bool) -> bool:
        """Pine: htfTrendBullish = close > htfEma"""
        if not self.use_htf_trend_filter:
            return True
        htf_ema = bar.get('htf_ema', np.nan)
        htf_close = bar.get('htf_close', np.nan)
        if pd.isna(htf_ema) or pd.isna(htf_close):
            return True
        return htf_close > htf_ema if is_long else htf_close < htf_ema

    def _check_market_regime(self, bar: pd.Series, is_long: bool) -> bool:
        """Pine: marketBullish = marketClose > marketEma"""
        if not self.use_market_regime_filter:
            return True
        market_close = bar.get('market_close', np.nan)
        market_ema = bar.get('market_ema', np.nan)
        if pd.isna(market_close) or pd.isna(market_ema):
            return True
        return market_close > market_ema if is_long else market_close < market_ema

    def _calc_vol_state(self, bar: pd.Series, ts) -> None:
        """Volatility state calculation"""
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
    # STOP CALCULATIONS (copied from ORB)
    # ═══════════════════════════════════════════════════════════════════════

    def _calc_swing_stop(self, day_df, i, is_long, atr):
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

    def _calc_stops(self, entry, atr, vwap, is_long, day_df, i):
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

    def _calc_achievable_rr(self, entry, stop, atr, rr_desired):
        """Calculate achievable R:R with max target cap"""
        risk = abs(entry - stop)
        if risk <= 0:
            return -1
        target_distance = risk * rr_desired
        target_atrs = target_distance / atr
        if target_atrs > self.max_target_atr + 1e-9:
            return (self.max_target_atr * atr) / risk
        return rr_desired

    def _select_best_stop(self, stops, entry, atr, rr_desired, is_long):
        """Select best valid stop by R:R"""
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
    # EXIT LOGIC (ORB-compatible, no SSL breach gate, no adaptive BE)
    # ═══════════════════════════════════════════════════════════════════════

    def _in_position(self) -> bool:
        return self.position.direction != ""

    def _check_stop_hit(self, bar):
        if not self._in_position():
            return False
        # Guard: don't stop out on entry bar (Pine: bar_index > entryBar)
        if self._bar_counter <= self.position.entry_bar_idx:
            return False
        if self.position.direction == "LONG":
            return bar['low'] <= self.position.current_stop
        else:
            return bar['high'] >= self.position.current_stop

    def _check_break_even(self, bar):
        """BE activation uses HIGH/LOW (HHM_simple.pine parity)"""
        if not self._in_position() or not self.use_break_even or self.position.stop_moved_to_be:
            return False
        be_target_distance = self.position.risk * self.break_even_rr
        if self.position.direction == "LONG":
            return bar['high'] >= self.position.entry_price + be_target_distance
        else:
            return bar['low'] <= self.position.entry_price - be_target_distance

    def _activate_break_even(self):
        self.position.stop_moved_to_be = True
        self.position.current_stop = self.position.entry_price

    def _check_trailing_activation(self, bar):
        """Trailing activation uses CLOSE (Pine parity)"""
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
        """EMA exit: after BE, not trailing, with confirmation bars + body %
        No SSL breach gate (simplified — matches ORB pattern)."""
        if not self._in_position() or not self.use_ema_exit:
            return False
        if not self.position.stop_moved_to_be or self.position.trailing_activated:
            return False

        close = bar['close']
        open_ = bar['open']
        body_size = abs(close - open_)

        if self.position.direction == "LONG":
            if close <= self.position.entry_price:
                return False
            if close < ema:
                # Pine: open > ema9 ? 0.0 : ... (short-circuit)
                if open_ > ema:
                    body_beyond = 0.0
                elif body_size > 0:
                    body_beyond = min(body_size, ema - close) / body_size * 100
                else:
                    body_beyond = 0.0
                if body_beyond >= self.ema_body_percentage:
                    self.position.bars_beyond_ema += 1
                else:
                    self.position.bars_beyond_ema = 0
            else:
                self.position.bars_beyond_ema = 0
        else:
            if close >= self.position.entry_price:
                return False
            if close > ema:
                # Pine: open < ema9 ? 0.0 : ...
                if open_ < ema:
                    body_beyond = 0.0
                elif body_size > 0:
                    body_beyond = min(body_size, close - ema) / body_size * 100
                else:
                    body_beyond = 0.0
                if body_beyond >= self.ema_body_percentage:
                    self.position.bars_beyond_ema += 1
                else:
                    self.position.bars_beyond_ema = 0
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
        )
        self.trades.append(trade)

        if self.verbose:
            tag = "WIN" if pnl > 0 else "LOSS"
            print(f"  [{tag}] {self.position.direction} {date_str} {self.position.entry_time}->{time_str}: "
                  f"${pnl:+.2f} ({r_multiple:+.2f}R) - {exit_reason} "
                  f"[C:{self.position.confluence_score:.1f}]")

        self.position.reset()
        self.bars_since_last_entry = 0

    # ═══════════════════════════════════════════════════════════════════════
    # ENTRY LOGIC
    # ═══════════════════════════════════════════════════════════════════════

    def _try_enter(self, bar, day_df, i, direction, conf_score, full_idx,
                   date_str, time_str, atr, entry_price=None):
        """Evaluate stop/RR and enter if feasible. Fixed R:R (no vol adaptation)."""
        is_long = direction == "LONG"
        entry = entry_price if entry_price is not None else bar['close']
        vwap = bar['vwap']
        rr_desired = self.profit_target_rr  # Fixed — no adaptive R:R

        stops = self._calc_stops(entry, atr, vwap, is_long, day_df, i)
        stop_price, stop_type, achievable_rr = self._select_best_stop(stops, entry, atr, rr_desired, is_long)

        if achievable_rr >= self.min_acceptable_rr - 0.0001:
            risk = abs(entry - stop_price)
            self.position.direction = direction
            self.position.entry_price = entry
            self.position.entry_time = time_str
            self.position.entry_date = date_str
            self.position.entry_bar_idx = self._bar_counter
            self.position.initial_stop = stop_price
            self.position.current_stop = stop_price
            self.position.stop_type = stop_type
            self.position.risk = risk
            self.position.target_rr = achievable_rr
            self.position.vol_state = self.vol.vol_state
            self.position.confluence_score = conf_score

            if is_long:
                self.position.ssl_score = self.df['ssl_score_bull'].iloc[full_idx]
                self.position.wae_score = self.df['wae_score_bull'].iloc[full_idx]
                self.position.qqe_score = self.df['qqe_score_bull'].iloc[full_idx]
            else:
                self.position.ssl_score = self.df['ssl_score_bear'].iloc[full_idx]
                self.position.wae_score = self.df['wae_score_bear'].iloc[full_idx]
                self.position.qqe_score = self.df['qqe_score_bear'].iloc[full_idx]
            self.position.vol_score = self.df['vol_score'].iloc[full_idx]

            self.bars_since_last_entry = 0

            if self.verbose:
                print(f"  [ENTRY] {direction} {time_str} @ ${entry:.2f} stop=${stop_price:.2f} ({stop_type}) "
                      f"RR={achievable_rr:.2f} vol={self.vol.vol_state} conf={conf_score:.1f}")
        else:
            skip = SkipRecord(
                date=date_str, time=time_str, direction=direction,
                entry_price=entry, stop_price=stop_price, stop_type=stop_type,
                achievable_rr=achievable_rr, min_rr=self.min_acceptable_rr,
                reason=f"RR {achievable_rr:.2f} < {self.min_acceptable_rr:.1f}",
                confluence_score=conf_score,
            )
            self.skips.append(skip)
            if self.verbose:
                print(f"  [SKIP] {direction} {time_str} @ ${entry:.2f}: "
                      f"RR {achievable_rr:.2f} < {self.min_acceptable_rr:.2f} ({stop_type})")

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN PROCESSING LOOP
    # ═══════════════════════════════════════════════════════════════════════

    def _process_day(self, date_str: str):
        """Process a single trading day"""
        day_df = self.df[self.df.index.date.astype(str) == date_str].copy()
        if len(day_df) == 0:
            return

        self.vol.orb_session_baseline = np.nan

        # Pending signal state (signal at bar T close, entry at bar T+1 close)
        pending_direction = None
        pending_conf_score = 0.0
        pending_full_idx = 0

        # Pine parity: newLongSignal = longSignal and not longSignal[1]
        # Only fire on the transition from not-qualifying to qualifying
        self._prev_long_qualified = False
        self._prev_short_qualified = False

        if self.verbose:
            print(f"\n--- {date_str} ---")

        for i in range(len(day_df)):
            ts = day_df.index[i]
            bar = day_df.iloc[i]

            if not self._is_rth(ts):
                continue

            time_str = ts.strftime('%H:%M')
            self.bars_since_last_entry += 1
            self._bar_counter += 1
            closed_this_bar = False

            atr = bar['atr_rth']
            ema = bar['ema9']

            # ═══════════════════════════════════════════════════════════
            # PENDING ENTRY EXECUTION (signal bar T, fill at bar T+1 close)
            # ═══════════════════════════════════════════════════════════

            if (pending_direction is not None
                and not self._in_position()
                and self._is_entry_window(ts)):
                self._try_enter(bar, day_df, i, pending_direction,
                               pending_conf_score, pending_full_idx,
                               date_str, time_str, atr,
                               entry_price=bar['close'])
                pending_direction = None

            # ═══════════════════════════════════════════════════════════
            # EXIT CASCADE (stop → BE → trailing → EMA → EOD)
            # ═══════════════════════════════════════════════════════════

            if self._in_position():
                # 1. Stop hit
                if self._check_stop_hit(bar):
                    exit_price = self.position.current_stop
                    reason = "TRAILING STOP" if self.position.trailing_activated else \
                             "BREAK-EVEN STOP" if self.position.stop_moved_to_be else "INITIAL STOP"
                    self._close_position(exit_price, reason, time_str, date_str)
                    closed_this_bar = True

            if self._in_position():
                # 2. BE activation
                if self._check_break_even(bar):
                    self._activate_break_even()
                # 3. Trailing activation + update
                if self._check_trailing_activation(bar):
                    self._activate_trailing(bar, atr)
                if self.position.trailing_activated:
                    self._update_trailing_stop(bar, atr, ema)
                # 4. EMA exit
                if self._check_ema_exit(bar, ema):
                    self._close_position(bar['close'], "EMA EXIT", time_str, date_str)
                    closed_this_bar = True

            # ═══════════════════════════════════════════════════════════
            # VOL STATE
            # ═══════════════════════════════════════════════════════════
            self._calc_vol_state(bar, ts)

            # ═══════════════════════════════════════════════════════════
            # SIGNAL QUALIFICATION (runs EVERY bar — Pine parity)
            # Pine evaluates longSignal/shortSignal unconditionally;
            # only the entry action is gated by position/window/filters.
            # ═══════════════════════════════════════════════════════════

            # Clear unconsumed pending (single-bar only)
            pending_direction = None

            # Qualification runs on every bar regardless of position,
            # lunch, volatility, or entry window — so _prev flags
            # always reflect the true signal state.
            full_idx = self.df.index.get_loc(ts)
            row = self.df.iloc[full_idx]

            bull_score = float(row['bull_score'])
            bear_score = float(row['bear_score'])

            htf_bull = self._check_htf_trend(bar, is_long=True)
            mkt_bull = self._check_market_regime(bar, is_long=True)
            long_qualifies_now = bull_score >= self.min_confluence and htf_bull and mkt_bull

            htf_bear = self._check_htf_trend(bar, is_long=False)
            mkt_bear = self._check_market_regime(bar, is_long=False)
            short_qualifies_now = bear_score >= self.min_confluence and htf_bear and mkt_bear

            # ═══════════════════════════════════════════════════════════
            # PENDING SIGNAL (only fire on rising edge, gated by filters)
            # ═══════════════════════════════════════════════════════════

            can_enter = (not self._in_position()
                         and not closed_this_bar
                         and self._is_entry_window(ts)
                         and self.bars_since_last_entry >= self.min_bars_between_entries
                         and not (self.avoid_lunch_hour and self._is_lunch_hour(ts))
                         and self._check_volatility_condition(bar))

            if can_enter:
                if long_qualifies_now and not self._prev_long_qualified:
                    pending_direction = "LONG"
                    pending_conf_score = bull_score
                    pending_full_idx = full_idx

                if pending_direction is None and short_qualifies_now and not self._prev_short_qualified:
                    pending_direction = "SHORT"
                    pending_conf_score = bear_score
                    pending_full_idx = full_idx

            # Always update — flags track true signal state across
            # positions, lunch hours, and all other gates.
            self._prev_long_qualified = long_qualifies_now
            self._prev_short_qualified = short_qualifies_now

        # End of day — close any open position
        if self._in_position():
            rth_bars = day_df[day_df.index.map(self._is_rth)]
            if len(rth_bars) > 0:
                last_bar = rth_bars.iloc[-1]
                self._close_position(last_bar['close'], "END OF DAY", "16:00", date_str)

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN RUN METHOD
    # ═══════════════════════════════════════════════════════════════════════

    def run(self) -> Dict[str, Any]:
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  HMM BACKTESTER V2 (Simple): {self.symbol}")
            print(f"  {self.start_date} to {self.end_date}")
            print(f"  Min Confluence: {self.min_confluence}")
            print(f"{'='*70}\n")

        self.load_data()

        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        all_dates = self.df.index.date
        unique_dates = sorted(set(all_dates))

        if unique_dates:
            data_first = unique_dates[0]
            data_last = unique_dates[-1]
            if self.verbose:
                print(f"  Data range: {data_first} to {data_last} ({len(unique_dates)} days)")
            if start.date() < data_first or end.date() > data_last:
                print(f"\n  WARNING: Requested {self.start_date} to {self.end_date} "
                      f"but data covers {data_first} to {data_last}")

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
                    'confluence': t.confluence_score,
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
        print(f"  HMM V2 BACKTEST SUMMARY: {results['symbol']}")
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
                      f"C:{t['confluence']:.1f}")
            print(f"\nTotal: {len(results['trades'])} trades, {results['total_r']:+.2f}R")


# ═══════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    bt = HMMBacktesterV2(
        symbol='AMD',
        start_date='2025-10-01',
        end_date='2025-12-31',
        min_confluence=3,
        verbose=True
    )
    results = bt.run()
