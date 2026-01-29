"""
Confluence Indicators Module - Pine v69 Compatible
===================================================
Implements SSL Hybrid, WAE, QQE, and Volume scoring
to match the 5ORB Pine Script confluence system.

Each indicator returns a score of 0 or 1.
Total confluence score = ORB(1) + SSL(1) + WAE(1) + QQE(1) + Volume(1) = max 5
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from dataclasses import dataclass


def pine_round(x: float) -> int:
    """
    Pine-style rounding: rounds 0.5 UP (away from zero)
    Python's round() uses banker's rounding: round(0.5) = 0
    Pine's math.round(0.5) = 1
    
    This matters for volume score where 0.5 should become 1, not 0.
    """
    if x >= 0:
        return int(np.floor(x + 0.5))
    else:
        return int(np.ceil(x - 0.5))


@dataclass
class ConfluenceScores:
    """Container for all confluence scores"""
    orb_score: float = 0.0      # 1 if valid breakout
    ssl_score_bull: float = 0.0  # 1 if bullish confirmed + accelerating
    ssl_score_bear: float = 0.0  # 1 if bearish confirmed + accelerating
    wae_score_bull: float = 0.0  # 1 if bullish explosion + accelerating
    wae_score_bear: float = 0.0  # 1 if bearish explosion + accelerating
    qqe_score_bull: float = 0.0  # 1 if bullish signal + momentum rising
    qqe_score_bear: float = 0.0  # 1 if bearish signal + momentum falling
    vol_score: float = 0.0       # 0, 0.5, or 1.0 (Z-score based)
    
    def long_score(self) -> int:
        """Total confluence for long entry"""
        return int(self.orb_score + self.ssl_score_bull + 
                   self.wae_score_bull + self.qqe_score_bull + 
                   pine_round(self.vol_score))
    
    def short_score(self) -> int:
        """Total confluence for short entry"""
        return int(self.orb_score + self.ssl_score_bear + 
                   self.wae_score_bear + self.qqe_score_bear + 
                   pine_round(self.vol_score))


class ConfluenceCalculator:
    """
    Calculates all confluence indicators matching Pine v69 logic.
    
    Usage:
        calc = ConfluenceCalculator(params)
        calc.compute_indicators(df)  # Adds indicator columns to df
        scores = calc.get_scores(i)   # Get scores at bar i
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize with parameters.
        
        params dict can include:
            ssl_baseline_length: int (default 45)
            ssl_length: int (default 5)
            ssl_type: str (default 'JMA')
            wae_fast_ema: int (default 20)
            wae_slow_ema: int (default 40)
            wae_sensitivity: int (default 190)
            wae_bb_length: int (default 20)
            wae_bb_mult: float (default 2.0)
            qqe_rsi1_length: int (default 6)
            qqe_rsi1_smoothing: int (default 5)
            qqe_factor_primary: float (default 3.0)
            qqe_rsi2_length: int (default 6)
            qqe_rsi2_smoothing: int (default 5)
            qqe_factor_secondary: float (default 1.61)
            qqe_bb_length: int (default 50)
            qqe_bb_mult: float (default 0.35)
            qqe_threshold: int (default 3)
            qqe_consecutive_bars: int (default 3)
            vol_lookback: int (default 5)
            use_ssl_momentum: bool (default True)
            use_wae_acceleration: bool (default True)
            use_qqe_momentum: bool (default True)
        """
        self.params = params or {}
        
        # SSL Hybrid parameters
        self.ssl_baseline_length = self.params.get('ssl_baseline_length', 45)
        self.ssl_length = self.params.get('ssl_length', 5)
        self.ssl_type = self.params.get('ssl_type', 'JMA')
        self.use_ssl_momentum = self.params.get('use_ssl_momentum', True)
        
        # WAE parameters
        self.wae_fast_ema = self.params.get('wae_fast_ema', 20)
        self.wae_slow_ema = self.params.get('wae_slow_ema', 40)
        self.wae_sensitivity = self.params.get('wae_sensitivity', 190)
        self.wae_bb_length = self.params.get('wae_bb_length', 20)
        self.wae_bb_mult = self.params.get('wae_bb_mult', 2.0)
        self.use_wae_acceleration = self.params.get('use_wae_acceleration', True)
        
        # QQE parameters
        self.qqe_rsi1_length = self.params.get('qqe_rsi1_length', 6)
        self.qqe_rsi1_smoothing = self.params.get('qqe_rsi1_smoothing', 5)
        self.qqe_factor_primary = self.params.get('qqe_factor_primary', 3.0)
        self.qqe_rsi2_length = self.params.get('qqe_rsi2_length', 6)
        self.qqe_rsi2_smoothing = self.params.get('qqe_rsi2_smoothing', 5)
        self.qqe_factor_secondary = self.params.get('qqe_factor_secondary', 1.61)
        self.qqe_bb_length = self.params.get('qqe_bb_length', 50)
        self.qqe_bb_mult = self.params.get('qqe_bb_mult', 0.35)
        self.qqe_threshold = self.params.get('qqe_threshold', 3)
        self.qqe_consecutive_bars = self.params.get('qqe_consecutive_bars', 3)
        self.use_qqe_momentum = self.params.get('use_qqe_momentum', True)
        
        # Volume parameters (Z-score based)
        self.vol_lookback = self.params.get('vol_lookback', 20)
        
        # Minimum confluence score
        self.min_confluence = self.params.get('min_confluence', 5)
        
        # Internal state for QQE consecutive bars
        self._bull_qqe_count = 0
        self._bear_qqe_count = 0
        self._prev_bull_strength = 0.0
        self._prev_bear_strength = 0.0
        
    # =========================================================================
    # MOVING AVERAGE HELPERS
    # =========================================================================
    
    @staticmethod
    def _sma(series: pd.Series, length: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=length, min_periods=length).mean()
    
    @staticmethod
    def _ema(series: pd.Series, length: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=length, adjust=False).mean()
    
    @staticmethod
    def _rma(series: pd.Series, length: int) -> pd.Series:
        """Wilder's Moving Average (RMA)"""
        return series.ewm(alpha=1/length, adjust=False).mean()
    
    @staticmethod
    def _hma(series: pd.Series, length: int) -> pd.Series:
        """Hull Moving Average"""
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        
        wma_half = series.rolling(window=half_length, min_periods=1).apply(
            lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        wma_full = series.rolling(window=length, min_periods=1).apply(
            lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        
        raw_hma = 2 * wma_half - wma_full
        hma = raw_hma.rolling(window=sqrt_length, min_periods=1).apply(
            lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        
        return hma
    
    def _jma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Jurik Moving Average approximation
        Pine: momentum = change(src)
              volatility = atr(length)
              ratio = momentum / volatility
              smoothed = ema(src + ratio * volatility, length)
        """
        momentum = series.diff()
        
        # ATR calculation
        high = self.df['high'] if hasattr(self, 'df') else series
        low = self.df['low'] if hasattr(self, 'df') else series
        close = self.df['close'] if hasattr(self, 'df') else series
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        volatility = self._rma(tr, length)
        
        ratio = np.where(volatility != 0, momentum / volatility, 0)
        adjusted = series + ratio * volatility
        smoothed = self._ema(pd.Series(adjusted, index=series.index), length)
        
        return smoothed
    
    def _ma(self, series: pd.Series, length: int, ma_type: str) -> pd.Series:
        """Generic MA dispatcher"""
        if ma_type == 'SMA':
            return self._sma(series, length)
        elif ma_type == 'EMA':
            return self._ema(series, length)
        elif ma_type == 'HMA':
            return self._hma(series, length)
        elif ma_type == 'JMA':
            return self._jma(series, length)
        else:
            return self._sma(series, length)
    
    @staticmethod
    def _stdev(series: pd.Series, length: int) -> pd.Series:
        """Standard Deviation"""
        return series.rolling(window=length, min_periods=length).std(ddof=0)
    
    @staticmethod
    def _rsi(series: pd.Series, length: int) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # =========================================================================
    # SSL HYBRID CALCULATION
    # =========================================================================
    
    def _calc_ssl(self, df: pd.DataFrame) -> None:
        """
        Calculate SSL Hybrid indicators.
        
        Pine logic:
            sslBaseline = hma(close, 45)
            sslUp = ma(high, 5, JMA)
            sslDown = ma(low, 5, JMA)
            
            sslBullConfirmation = close > sslBaseline
            sslGapBull = close - sslBaseline
            sslAccelBull = sslGapBull > sslGapBull[1]
            
            sslScoreBull = 1 if (sslBullConfirmation AND sslAccelBull) else 0
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # SSL Baseline (always HMA in Pine)
        df['ssl_baseline'] = self._hma(close, self.ssl_baseline_length)
        
        # SSL Up/Down lines
        df['ssl_up'] = self._ma(high, self.ssl_length, self.ssl_type)
        df['ssl_down'] = self._ma(low, self.ssl_length, self.ssl_type)
        
        # Direction and confirmations
        df['ssl_bull_confirm'] = close > df['ssl_baseline']
        df['ssl_bear_confirm'] = close < df['ssl_baseline']
        
        # Gap calculations
        df['ssl_gap_bull'] = close - df['ssl_baseline']
        df['ssl_gap_bear'] = df['ssl_baseline'] - close
        
        # Acceleration (gap widening)
        df['ssl_accel_bull'] = df['ssl_gap_bull'] > df['ssl_gap_bull'].shift(1)
        df['ssl_accel_bear'] = df['ssl_gap_bear'] > df['ssl_gap_bear'].shift(1)
        
        # Final scores
        if self.use_ssl_momentum:
            df['ssl_score_bull'] = ((df['ssl_bull_confirm']) & 
                                    (df['ssl_accel_bull'])).astype(float)
            df['ssl_score_bear'] = ((df['ssl_bear_confirm']) & 
                                    (df['ssl_accel_bear'])).astype(float)
        else:
            df['ssl_score_bull'] = df['ssl_bull_confirm'].astype(float)
            df['ssl_score_bear'] = df['ssl_bear_confirm'].astype(float)
        
        # NOTE: Do NOT shift here. Shift is applied in backtester for Pine visual alignment.
    
    # =========================================================================
    # WAE (Waddah Attar Explosion) CALCULATION
    # =========================================================================
    
    def _calc_wae(self, df: pd.DataFrame) -> None:
        """
        Calculate WAE indicators.
        
        Pine logic:
            fastMA = ema(close, 20)
            slowMA = ema(close, 40)
            macd = fastMA - slowMA
            t1 = (macd - macd[1]) * 190
            
            basis = sma(close, 20)
            dev = 2 * stdev(close, 20)
            explosionLine = (basis + dev) - (basis - dev) = 2 * dev
            
            trendUp = t1 >= 0 ? t1 : 0
            trendDown = t1 < 0 ? -t1 : 0
            
            waeBullConfirmation = trendUp > explosionLine
            waeAccelBull = trendUp > trendUp[1]
            
            waeScoreBull = 1 if (waeBullConfirmation AND waeAccelBull) else 0
        """
        close = df['close']
        
        # MACD calculation (uses full session - Pine does this)
        fast_ma = self._ema(close, self.wae_fast_ema)
        slow_ma = self._ema(close, self.wae_slow_ema)
        macd = fast_ma - slow_ma
        
        # T1 (trend strength with sensitivity)
        t1 = (macd - macd.shift(1)) * self.wae_sensitivity
        
        # Bollinger Bands for explosion line - MUST USE RTH-ONLY (Pine parity)
        # Pine computes WAE explosion on RTH-only closes via request.security
        is_rth = df.index.map(lambda x: 9*60+30 <= x.hour*60+x.minute < 16*60)
        rth_close = close.where(is_rth)
        
        # Compute BB on RTH-only, then forward fill to all bars
        rth_basis = rth_close.rolling(window=self.wae_bb_length, min_periods=self.wae_bb_length).mean()
        rth_stdev = rth_close.rolling(window=self.wae_bb_length, min_periods=self.wae_bb_length).std(ddof=0)
        rth_dev = self.wae_bb_mult * rth_stdev
        rth_bb_upper = rth_basis + rth_dev
        rth_bb_lower = rth_basis - rth_dev
        rth_explosion = rth_bb_upper - rth_bb_lower
        
        # Forward fill to cover premarket bars of next day
        df['wae_explosion_line'] = rth_explosion.ffill()
        
        # Trend Up/Down (uses full session)
        df['wae_trend_up'] = t1.where(t1 >= 0, 0.0)
        df['wae_trend_down'] = (-t1).where(t1 < 0, 0.0)
        
        # Confirmations
        df['wae_bull_confirm'] = df['wae_trend_up'] > df['wae_explosion_line']
        df['wae_bear_confirm'] = df['wae_trend_down'] > df['wae_explosion_line']
        
        # Acceleration - Pine: trendUp > nz(trendUp[1]) AND trendUp > explosionLine
        # nz() in Pine replaces NaN with 0, so use fillna(0)
        prev_up = df['wae_trend_up'].shift(1).fillna(0.0)
        prev_dn = df['wae_trend_down'].shift(1).fillna(0.0)
        
        df['wae_accel_bull'] = ((df['wae_trend_up'] > prev_up) & 
                                (df['wae_trend_up'] > df['wae_explosion_line']))
        df['wae_accel_bear'] = ((df['wae_trend_down'] > prev_dn) & 
                                (df['wae_trend_down'] > df['wae_explosion_line']))
        
        # Final scores
        if self.use_wae_acceleration:
            df['wae_score_bull'] = ((df['wae_bull_confirm']) & 
                                    (df['wae_accel_bull'])).astype(float)
            df['wae_score_bear'] = ((df['wae_bear_confirm']) & 
                                    (df['wae_accel_bear'])).astype(float)
        else:
            df['wae_score_bull'] = df['wae_bull_confirm'].astype(float)
            df['wae_score_bear'] = df['wae_bear_confirm'].astype(float)
        
        # NOTE: Do NOT shift here. Shift is applied in backtester for Pine visual alignment.
    
    # =========================================================================
    # QQE (Quantitative Qualitative Estimation) CALCULATION
    # =========================================================================
    
    def _calc_qqe_line(self, smoothed_rsi: pd.Series, qqe_factor: float, rsi_length: int) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate QQE trend line and direction.
        
        Pine logic:
            wildersLength = rsiLength * 2 - 1
            atrRsi = abs(smoothedRsi[1] - smoothedRsi)
            smoothedAtrRsi = ema(atrRsi, wildersLength)
            dynamicAtrRsi = smoothedAtrRsi * qqeFactor
            
            newLongBand = smoothedRsi - dynamicAtrRsi
            newShortBand = smoothedRsi + dynamicAtrRsi
            
            (band logic with cross-based state tracking)
        """
        # Fix #1: Use passed rsi_length, not hardcoded primary length
        wilders_length = rsi_length * 2 - 1
        
        # ATR of RSI
        atr_rsi = abs(smoothed_rsi - smoothed_rsi.shift(1))
        smoothed_atr_rsi = self._ema(atr_rsi, wilders_length)
        dynamic_atr_rsi = smoothed_atr_rsi * qqe_factor
        
        # Initialize bands and trend
        long_band = pd.Series(index=smoothed_rsi.index, dtype=float)
        short_band = pd.Series(index=smoothed_rsi.index, dtype=float)
        trend = pd.Series(index=smoothed_rsi.index, dtype=int)
        
        # First values
        long_band.iloc[0] = 0.0
        short_band.iloc[0] = 0.0
        trend.iloc[0] = 0
        
        # Fix #2: Implement Pine-style cross() behavior for trend direction
        # Pine: if ta.cross(smoothedRsi, shortBand[1]) trendDirection := 1
        #       else if ta.cross(longBand[1], smoothedRsi) trendDirection := -1
        #       else trendDirection := trendDirection[1]
        
        for i in range(1, len(smoothed_rsi)):
            new_long_band = smoothed_rsi.iloc[i] - dynamic_atr_rsi.iloc[i]
            new_short_band = smoothed_rsi.iloc[i] + dynamic_atr_rsi.iloc[i]
            
            # Long band logic
            if (smoothed_rsi.iloc[i-1] > long_band.iloc[i-1] and 
                smoothed_rsi.iloc[i] > long_band.iloc[i-1]):
                long_band.iloc[i] = max(long_band.iloc[i-1], new_long_band)
            else:
                long_band.iloc[i] = new_long_band
            
            # Short band logic
            if (smoothed_rsi.iloc[i-1] < short_band.iloc[i-1] and 
                smoothed_rsi.iloc[i] < short_band.iloc[i-1]):
                short_band.iloc[i] = min(short_band.iloc[i-1], new_short_band)
            else:
                short_band.iloc[i] = new_short_band
            
            # Trend direction - Pine cross() logic
            # cross(a, b) = a crosses OVER b = (a[1] <= b[1]) and (a > b)
            # ta.cross(smoothedRsi, shortBand[1]) means RSI crosses OVER shortBand
            rsi_curr = smoothed_rsi.iloc[i]
            rsi_prev = smoothed_rsi.iloc[i-1]
            short_band_prev = short_band.iloc[i-1]
            long_band_prev = long_band.iloc[i-1]
            
            # Cross UP through short band -> bullish
            cross_up = (rsi_prev <= short_band_prev) and (rsi_curr > short_band_prev)
            # Cross DOWN through long band -> bearish  
            # ta.cross(longBand[1], smoothedRsi) means longBand crosses OVER RSI
            # which is equivalent to RSI crossing UNDER longBand
            cross_down = (rsi_prev >= long_band_prev) and (rsi_curr < long_band_prev)
            
            if cross_up:
                trend.iloc[i] = 1
            elif cross_down:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
        
        # QQE line
        qqe_line = pd.Series(index=smoothed_rsi.index, dtype=float)
        qqe_line = long_band.where(trend == 1, short_band)
        
        return qqe_line, trend
    
    def _calc_qqe(self, df: pd.DataFrame) -> None:
        """
        Calculate QQE indicators.
        
        Pine logic:
            Primary QQE: RSI(6) smoothed by EMA(5), factor=3
            Secondary QQE: RSI(6) smoothed by EMA(5), factor=1.61
            Bollinger on QQE: length=50, mult=0.35
            
            rawBullCondition = (secondaryRSI-50 > threshold) AND (primaryRSI-50 > BBupper)
            signalStrength increasing for 3+ bars
            momentum rising = RSI > RSI[1]
            
            qqeScoreBull = 1 if (rawCondition AND consecutive AND momentumRising) else 0
        """
        close = df['close']
        
        # Primary QQE - pass rsi_length for correct Wilder length
        primary_rsi = self._rsi(close, self.qqe_rsi1_length)
        primary_smoothed = self._ema(primary_rsi, self.qqe_rsi1_smoothing)
        primary_qqe, primary_trend = self._calc_qqe_line(primary_smoothed, self.qqe_factor_primary, self.qqe_rsi1_length)
        
        # Secondary QQE - pass rsi_length for correct Wilder length
        secondary_rsi = self._rsi(close, self.qqe_rsi2_length)
        secondary_smoothed = self._ema(secondary_rsi, self.qqe_rsi2_smoothing)
        secondary_qqe, secondary_trend = self._calc_qqe_line(secondary_smoothed, self.qqe_factor_secondary, self.qqe_rsi2_length)
        
        df['qqe_primary_rsi'] = primary_smoothed
        df['qqe_secondary_rsi'] = secondary_smoothed
        df['qqe_primary_line'] = primary_qqe  # Store for BB calculation
        
        # FIX #3: Bollinger Bands on primaryQQETrendLine, NOT primary_smoothed
        # Pine: bollingerBasis = ta.sma(primaryQQETrendLine - 50, qqeBBLengthEff)
        qqe_centered = primary_qqe - 50  # Use QQE line, not smoothed RSI
        bb_basis = self._sma(qqe_centered, self.qqe_bb_length)
        bb_dev = self.qqe_bb_mult * self._stdev(qqe_centered, self.qqe_bb_length)
        df['qqe_bb_upper'] = bb_basis + bb_dev
        df['qqe_bb_lower'] = bb_basis - bb_dev
        
        # Raw conditions
        df['qqe_raw_bull'] = ((secondary_smoothed - 50 > self.qqe_threshold) & 
                              (primary_smoothed - 50 > df['qqe_bb_upper']))
        df['qqe_raw_bear'] = ((secondary_smoothed - 50 < -self.qqe_threshold) & 
                              (primary_smoothed - 50 < df['qqe_bb_lower']))
        
        # Signal strength for consecutive bars tracking
        df['qqe_bull_strength'] = np.minimum(
            secondary_smoothed - 50 - self.qqe_threshold,
            primary_smoothed - 50 - df['qqe_bb_upper']
        ).where(df['qqe_raw_bull'], 0)
        
        df['qqe_bear_strength'] = np.minimum(
            -self.qqe_threshold - (secondary_smoothed - 50),
            df['qqe_bb_lower'] - (primary_smoothed - 50)
        ).where(df['qqe_raw_bear'], 0)
        
        # Count consecutive increasing bars
        df['qqe_bull_count'] = 0
        df['qqe_bear_count'] = 0
        
        bull_count = 0
        bear_count = 0
        prev_bull_strength = 0.0
        prev_bear_strength = 0.0
        
        for i in range(len(df)):
            # Bull counting
            if df['qqe_raw_bull'].iloc[i]:
                if df['qqe_bull_strength'].iloc[i] > prev_bull_strength:
                    bull_count += 1
                else:
                    bull_count = 0
                prev_bull_strength = df['qqe_bull_strength'].iloc[i]
            else:
                bull_count = 0
                prev_bull_strength = 0.0
            
            # Bear counting
            if df['qqe_raw_bear'].iloc[i]:
                if df['qqe_bear_strength'].iloc[i] > prev_bear_strength:
                    bear_count += 1
                else:
                    bear_count = 0
                prev_bear_strength = df['qqe_bear_strength'].iloc[i]
            else:
                bear_count = 0
                prev_bear_strength = 0.0
            
            df.iloc[i, df.columns.get_loc('qqe_bull_count')] = bull_count
            df.iloc[i, df.columns.get_loc('qqe_bear_count')] = bear_count
        
        # QQE signals - Pine uses: count >= consecutiveIncreasingBars - 1
        # For Pine parity, we match this exactly
        df['qqe_bull_signal'] = ((df['qqe_bull_count'] >= self.qqe_consecutive_bars - 1) & 
                                  df['qqe_raw_bull'])
        df['qqe_bear_signal'] = ((df['qqe_bear_count'] >= self.qqe_consecutive_bars - 1) & 
                                  df['qqe_raw_bear'])
        
        # Momentum direction - Pine uses primaryRSI which IS smoothedRsi from calculateQQE
        # Pine: [qqeTrendLine, smoothedRsi, trendDirection] = return from calculateQQE
        # Pine: [primaryQQETrendLine, primaryRSI, primaryTrend] = calculateQQE(...)
        # So primaryRSI = smoothedRsi = EMA(RSI), NOT raw RSI
        df['qqe_momentum_rising'] = primary_smoothed > primary_smoothed.shift(1)
        df['qqe_momentum_falling'] = primary_smoothed < primary_smoothed.shift(1)
        
        # Final scores
        if self.use_qqe_momentum:
            df['qqe_score_bull'] = ((df['qqe_bull_signal']) & 
                                    (df['qqe_momentum_rising'])).astype(float)
            df['qqe_score_bear'] = ((df['qqe_bear_signal']) & 
                                    (df['qqe_momentum_falling'])).astype(float)
        else:
            df['qqe_score_bull'] = df['qqe_bull_signal'].astype(float)
            df['qqe_score_bear'] = df['qqe_bear_signal'].astype(float)
    
    # =========================================================================
    # VOLUME ANALYSIS (Z-Score Normalized)
    # =========================================================================
    
    def _calc_volume(self, df: pd.DataFrame) -> None:
        """
        Calculate volume score using Z-score normalization.
        
        Pine logic (v70+):
            volMean = ta.sma(volume, volLookback)
            volStd = ta.stdev(volume, volLookback)
            volZ = volStd > 0 ? (volume - volMean) / volStd : 0.0
            
            if volZ >= 1.0: volScore = 1.0      // 1+ stdev above = Strong (+1 confluence)
            elif volZ >= 0.2: volScore = 0.5    // Slightly above = Avg (+0 confluence)
            else: volScore = 0.0                // Below normal = Low (+0 confluence)
        """
        vol_mean = self._sma(df['volume'], self.vol_lookback)
        vol_std = df['volume'].rolling(window=self.vol_lookback, min_periods=self.vol_lookback).std(ddof=0)
        
        # Z-score with safe divide
        df['vol_z'] = np.where(vol_std > 0, (df['volume'] - vol_mean) / vol_std, 0.0)
        
        # Score based on Z-score thresholds
        df['vol_score'] = np.where(
            df['vol_z'] >= 1.0, 1.0,
            np.where(df['vol_z'] >= 0.2, 0.5, 0.0)
        )
    
    # =========================================================================
    # MAIN COMPUTATION
    # =========================================================================
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all confluence indicators and add columns to dataframe.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            
        Returns:
            DataFrame with added indicator columns
        """
        self.df = df.copy()
        
        # Calculate all indicators
        self._calc_ssl(self.df)
        self._calc_wae(self.df)
        self._calc_qqe(self.df)
        self._calc_volume(self.df)
        
        return self.df
    
    def get_scores(self, i: int) -> ConfluenceScores:
        """
        Get confluence scores at bar index i.
        
        Args:
            i: Bar index
            
        Returns:
            ConfluenceScores dataclass with all scores
        """
        if not hasattr(self, 'df'):
            raise ValueError("Must call compute_indicators() first")
        
        return ConfluenceScores(
            orb_score=0.0,  # Set by caller based on breakout
            ssl_score_bull=self.df['ssl_score_bull'].iloc[i],
            ssl_score_bear=self.df['ssl_score_bear'].iloc[i],
            wae_score_bull=self.df['wae_score_bull'].iloc[i],
            wae_score_bear=self.df['wae_score_bear'].iloc[i],
            qqe_score_bull=self.df['qqe_score_bull'].iloc[i],
            qqe_score_bear=self.df['qqe_score_bear'].iloc[i],
            vol_score=self.df['vol_score'].iloc[i]
        )
    
    def check_confluence(self, i: int, is_long: bool, orb_breakout: bool = True) -> Tuple[bool, int, ConfluenceScores]:
        """
        Check if confluence requirements are met.
        
        Args:
            i: Bar index
            is_long: True for long, False for short
            orb_breakout: True if ORB breakout confirmed
            
        Returns:
            (passes_confluence, score, scores_detail)
        """
        scores = self.get_scores(i)
        scores.orb_score = 1.0 if orb_breakout else 0.0
        
        if is_long:
            total = scores.long_score()
        else:
            total = scores.short_score()
        
        passes = total >= self.min_confluence
        
        return passes, total, scores


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    print("Confluence Indicators Module")
    print("=" * 50)
    print("This module implements Pine v69 indicator logic for:")
    print("  - SSL Hybrid (baseline + gap widening)")
    print("  - WAE (explosion line + acceleration)")
    print("  - QQE (dual RSI + Bollinger + momentum)")
    print("  - Volume (ratio scoring)")
    print()
    print("Usage:")
    print("  calc = ConfluenceCalculator(params)")
    print("  df = calc.compute_indicators(df)")
    print("  passes, score, details = calc.check_confluence(i, is_long=True)")
