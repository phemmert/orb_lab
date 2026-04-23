"""
confluence_indicators.py
========================
Line-by-line faithful translation of the SSL Hybrid, WAE, QQE Mod, and
Volume Z-score confluence indicators from HHM.pine.

Expects RTH-only 2-minute bars as a pandas DataFrame with columns:
    open, high, low, close, volume   (lowercase, float64)
    index = DatetimeIndex (or any ordered index)

Pine reference lines are noted in comments (e.g. "# Pine L1951").
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT PARAMETERS  (mirror Pine input defaults — lines 152-220, 81-82)
# ═══════════════════════════════════════════════════════════════════════════════

SSL_DEFAULTS = dict(
    baseline_type="HMA",          # Pine L154: sslBaseline
    baseline_length=25,           # Pine L155: sslBaselineLength
    smoothing_enabled=True,       # Pine L156: sslSmoothingEnabled
    smoothing_method="Super Smoother",  # Pine L157
    smoothing_factor=2.0,         # Pine L158
    use_momentum_filter=True,     # Pine L160: useSSLMomentumFilter
    ssl2_type="JMA",              # Pine L161: sslType
    ssl2_length=5,                # Pine L162: sslLength
    exit_type="HMA",              # Pine L163: exitType
    exit_length=10,               # Pine L164: exitLength
)

WAE_DEFAULTS = dict(
    sensitivity=275,              # Pine L168: waeSensitivity
    fast_ema=15,                  # Pine L169: waeFastEMA
    slow_ema=30,                  # Pine L170: waeSlowEMA
    bb_length=20,                 # Pine L171: waeBBLength
    bb_multiplier=2.0,            # Pine L172: waeBBMultiplier
    use_acceleration=True,        # Pine L174: useWAEAcceleration
)

QQE_DEFAULTS = dict(
    rsi1_length=9,                # Pine L211: qqeRSI1Length
    rsi1_smoothing=5,             # Pine L212: qqeRSI1Smoothing
    qqe_factor1=3.0,              # Pine L213: qqeFactor1
    rsi2_length=4,                # Pine L214: qqeRSI2Length
    rsi2_smoothing=4,             # Pine L215: qqeRSI2Smoothing
    qqe_factor2=1.61,             # Pine L216: qqeFactor2
    bb_length=25,                 # Pine L217: qqeBBLength
    bb_multiplier=0.35,           # Pine L218: qqeBBMultiplier
    threshold_secondary=3.0,      # Pine L219: thresholdSecondary
    consecutive_increasing_bars=3,  # Pine L220
    use_momentum_filter=True,     # Pine L222: useQQEMomentumFilter
)

VOL_DEFAULTS = dict(
    lookback=20,                  # Pine L82: volLookback
    enabled=True,                 # Pine L81: enableVolume
)


# ═══════════════════════════════════════════════════════════════════════════════
# PINE BUILT-IN EQUIVALENTS
# ═══════════════════════════════════════════════════════════════════════════════

def _nz(series: pd.Series, replacement: float = 0.0) -> pd.Series:
    """Pine nz(): replace NaN with replacement."""
    return series.fillna(replacement)


def _pine_rsi(src: pd.Series, length: int) -> pd.Series:
    """Pine ta.rsi() — Wilder-smoothed RSI."""
    delta = src.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder's smoothing = EMA with alpha = 1/length
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _pine_ema(src: pd.Series, length: int) -> pd.Series:
    """Pine ta.ema() — standard EMA."""
    return src.ewm(span=length, min_periods=length, adjust=False).mean()


def _pine_sma(src: pd.Series, length: int) -> pd.Series:
    """Pine ta.sma()."""
    return src.rolling(window=length, min_periods=length).mean()


def _pine_stdev(src: pd.Series, length: int) -> pd.Series:
    """Pine ta.stdev() — population stdev (ddof=0) as Pine uses."""
    return src.rolling(window=length, min_periods=length).std(ddof=0)


def _pine_wma(src: pd.Series, length: int) -> pd.Series:
    """Pine ta.wma()."""
    weights = np.arange(1, length + 1, dtype=float)
    def _wma(x):
        return np.dot(x, weights) / weights.sum()
    return src.rolling(window=length, min_periods=length).apply(_wma, raw=True)


def _pine_hma(src: pd.Series, length: int) -> pd.Series:
    """Pine ta.hma() — Hull Moving Average."""
    half_len = max(int(length / 2), 1)
    sqrt_len = max(int(math.sqrt(length)), 1)
    wma_half = _pine_wma(src, half_len)
    wma_full = _pine_wma(src, length)
    return _pine_wma(2 * wma_half - wma_full, sqrt_len)


def _pine_atr(high: pd.Series, low: pd.Series, close: pd.Series,
              length: int) -> pd.Series:
    """Pine ta.atr() — Wilder-smoothed ATR."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()


def _pine_change(src: pd.Series) -> pd.Series:
    """Pine ta.change()."""
    return src.diff()


# ═══════════════════════════════════════════════════════════════════════════════
# MA / SMOOTHER HELPERS  (Pine L1356-L1428)
# ═══════════════════════════════════════════════════════════════════════════════

def _jma_ohlc(src: pd.Series, length: int,
              high: pd.Series, low: pd.Series,
              close: pd.Series) -> pd.Series:
    """Pine L1396-L1401: Enhanced JMA function (with OHLC for true ATR).
        jma(src, length) =>
            momentum = ta.change(src)
            volatility = ta.atr(length)
            ratio = volatility != 0 ? momentum / volatility : 0
            smoothed = ta.ema(src + ratio * volatility, length)
    """
    momentum = _pine_change(src)                       # Pine L1397
    volatility = _pine_atr(high, low, close, length)   # Pine L1398
    ratio = np.where(volatility != 0, momentum / volatility, 0.0)
    ratio = pd.Series(ratio, index=src.index)
    smoothed = _pine_ema(src + ratio * volatility, length)  # Pine L1400
    return smoothed


def _jma_series(src: pd.Series, length: int) -> pd.Series:
    """JMA fallback for non-OHLC series (smoothers applied to MA output)."""
    momentum = _pine_change(src)
    volatility = momentum.abs().ewm(
        alpha=1.0 / length, min_periods=length, adjust=False
    ).mean()
    ratio = np.where(volatility != 0, momentum / volatility, 0.0)
    ratio = pd.Series(ratio, index=src.index)
    smoothed = _pine_ema(src + ratio * volatility, length)
    return smoothed


def _dema(src: pd.Series, length: int) -> pd.Series:
    """Pine L1356-L1359: Double EMA."""
    ema1 = _pine_ema(src, length)
    ema2 = _pine_ema(ema1, length)
    return 2 * ema1 - ema2


def _tema(src: pd.Series, length: int) -> pd.Series:
    """Pine L1362-L1366: Triple EMA."""
    ema1 = _pine_ema(src, length)
    ema2 = _pine_ema(ema1, length)
    ema3 = _pine_ema(ema2, length)
    return 3 * (ema1 - ema2) + ema3


def _super_smoother(src: pd.Series, length: int) -> pd.Series:
    """Pine L1369-L1378: Ehlers Super Smoother Filter.
    Recursive IIR filter — must iterate bar by bar.
    """
    length = max(length, 1)
    cutoff = 1.414 * math.pi / length            # Pine L1370
    a1 = math.exp(-cutoff)                        # Pine L1371
    b1 = 2 * a1 * math.cos(1.414 * math.pi / length)  # Pine L1372
    c2 = b1                                       # Pine L1373
    c3 = -(a1 * a1)                               # Pine L1374
    c1 = 1 - c2 - c3                              # Pine L1375

    vals = src.values.astype(float)
    n = len(vals)
    ss = np.full(n, np.nan)
    for i in range(n):
        src_i    = vals[i]    if not np.isnan(vals[i])    else 0.0
        src_prev = vals[i-1]  if i >= 1 and not np.isnan(vals[i-1]) else 0.0
        ss_prev1 = ss[i-1]   if i >= 1 and not np.isnan(ss[i-1])   else 0.0
        ss_prev2 = ss[i-2]   if i >= 2 and not np.isnan(ss[i-2])   else 0.0
        # Pine L1377
        ss[i] = c1 * (src_i + src_prev) / 2 + c2 * ss_prev1 + c3 * ss_prev2
    return pd.Series(ss, index=src.index)


def _gaussian(src: pd.Series, length: int, sigma: float) -> pd.Series:
    """Pine L1381-L1393: Gaussian Filter."""
    length = max(int(length), 1)
    weights = np.zeros(length)
    for i in range(length):
        weights[i] = math.exp(
            -0.5 * ((i - length / 2) / (sigma * length / 4)) ** 2
        )
    sum_weights = weights.sum() or 1.0

    vals = src.values.astype(float)
    n = len(vals)
    out = np.full(n, np.nan)
    for bar in range(n):
        s = 0.0
        for i in range(length):
            if i < bar:                              # Pine L1391
                v = vals[bar - i] if not np.isnan(vals[bar - i]) else 0.0
                s += v * weights[i] / sum_weights    # Pine L1392
        out[bar] = s
    return pd.Series(out, index=src.index)


def _ma(src: pd.Series, length: int, ma_type: str,
        high: pd.Series | None = None,
        low: pd.Series | None = None,
        close: pd.Series | None = None) -> pd.Series:
    """Pine L1404-L1415: Main MA dispatcher."""
    _len = max(int(length), 1)                       # Pine L1405
    if ma_type == "SMA":
        return _pine_sma(src, _len)                  # Pine L1408
    elif ma_type == "EMA":
        return _pine_ema(src, length)                # Pine L1409
    elif ma_type == "HMA":
        return _pine_hma(src, _len)                  # Pine L1410
    elif ma_type == "WMA":
        return _pine_wma(src, _len)                  # Pine L1411
    elif ma_type == "JMA":
        if high is not None and low is not None and close is not None:
            return _jma_ohlc(src, length, high, low, close)  # Pine L1412
        return _jma_series(src, length)
    elif ma_type == "DEMA":
        return _dema(src, length)                    # Pine L1413
    elif ma_type == "TEMA":
        return _tema(src, length)                    # Pine L1414
    else:
        return _pine_sma(src, _len)                  # Pine L1415


def _apply_smoother(src: pd.Series, enabled: bool, method: str,
                    factor: float) -> pd.Series:
    """Pine L1418-L1428: applySmoother()."""
    if not enabled:                                  # Pine L1419-L1420
        return src
    smoothing_length = max(int(round(factor * 2)), 1)  # Pine L1422
    if method == "Super Smoother":
        return _super_smoother(src, smoothing_length)  # Pine L1424
    elif method == "Jurik Filter":
        return _jma_series(src, smoothing_length)      # Pine L1425
    elif method == "EMA Post-Smooth":
        return _pine_ema(src, smoothing_length)        # Pine L1426
    elif method == "Gaussian":
        return _gaussian(src, smoothing_length, factor)  # Pine L1427
    else:
        return src                                     # Pine L1428


# ═══════════════════════════════════════════════════════════════════════════════
# QQE CORE FUNCTION  (Pine L1986-L2015)
# ═══════════════════════════════════════════════════════════════════════════════

def _calculate_qqe(
    close: pd.Series, rsi_length: int, smoothing_factor: int,
    qqe_factor: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Pine L1986-L2015: calculateQQE().

    Returns (qqeTrendLine, smoothedRsi, trendDirection).

    Cross detection note:
        Pine L2004: crossUp  = ta.cross(smoothedRsi, shortBand[1])
        Pine L2005: crossDown = ta.cross(longBand[1], smoothedRsi)
        ta.cross(a, b) is bidirectional: true when a crosses b in either
        direction.  When the argument is already a [1] reference (e.g.
        shortBand[1]), Pine evaluates its "previous" value as [2].
    """
    # Pine L1987
    wilders_length = rsi_length * 2 - 1

    # Pine L1988-L1989
    rsi = _pine_rsi(close, rsi_length)
    smoothed_rsi = _pine_ema(rsi, smoothing_factor)

    # Pine L1990-L1992
    atr_rsi = (smoothed_rsi - smoothed_rsi.shift(1)).abs()
    smoothed_atr_rsi = _pine_ema(atr_rsi, wilders_length)
    dynamic_atr_rsi = smoothed_atr_rsi * qqe_factor

    # Pine L1994-L2014: recursive bands — bar-by-bar
    n = len(close)
    long_band  = np.zeros(n)
    short_band = np.zeros(n)
    trend_dir  = np.zeros(n, dtype=int)
    qqe_line   = np.zeros(n)

    sr  = smoothed_rsi.values.astype(float)
    dar = dynamic_atr_rsi.values.astype(float)

    for i in range(n):
        if np.isnan(sr[i]) or np.isnan(dar[i]):
            long_band[i]  = 0.0
            short_band[i] = 0.0
            trend_dir[i]  = trend_dir[i - 1] if i > 0 else 0
            qqe_line[i]   = long_band[i] if trend_dir[i] == 1 else short_band[i]
            continue

        new_long  = sr[i] - dar[i]                   # Pine L2000
        new_short = sr[i] + dar[i]                    # Pine L1999

        prev_sr = sr[i - 1] if i > 0 and not np.isnan(sr[i - 1]) else 0.0

        # Pine L2001: longBand update
        prev_lb = long_band[i - 1] if i > 0 else 0.0
        if prev_sr > prev_lb and sr[i] > prev_lb:
            long_band[i] = max(prev_lb, new_long)
        else:
            long_band[i] = new_long

        # Pine L2002: shortBand update
        prev_sb = short_band[i - 1] if i > 0 else 0.0
        if prev_sr < prev_sb and sr[i] < prev_sb:
            short_band[i] = min(prev_sb, new_short)
        else:
            short_band[i] = new_short

        # ── Cross detection ──
        # Pine L2004: crossUp = ta.cross(smoothedRsi, shortBand[1])
        #   At bar i:   a = sr[i],     b = shortBand[i-1]   (shortBand[1])
        #   At bar i-1: a = sr[i-1],   b = shortBand[i-2]   (shortBand[1] evaluated one bar earlier)
        # Pine L2005: crossDown = ta.cross(longBand[1], smoothedRsi)
        #   At bar i:   a = longBand[i-1],  b = sr[i]
        #   At bar i-1: a = longBand[i-2],  b = sr[i-1]
        cross_up = False
        cross_down = False
        if i >= 2:
            prev_sb_curr = short_band[i - 1]         # shortBand[1] at bar i
            prev_sb_prev = short_band[i - 2]         # shortBand[1] at bar i-1
            cross_up = ((sr[i] > prev_sb_curr and prev_sr <= prev_sb_prev) or
                        (sr[i] < prev_sb_curr and prev_sr >= prev_sb_prev))

            prev_lb_curr = long_band[i - 1]          # longBand[1] at bar i
            prev_lb_prev = long_band[i - 2]          # longBand[1] at bar i-1
            cross_down = ((prev_lb_curr > sr[i] and prev_lb_prev <= prev_sr) or
                          (prev_lb_curr < sr[i] and prev_lb_prev >= prev_sr))
        elif i == 1:
            prev_sb_curr = short_band[0]
            prev_sb_prev = 0.0
            cross_up = ((sr[i] > prev_sb_curr and prev_sr <= prev_sb_prev) or
                        (sr[i] < prev_sb_curr and prev_sr >= prev_sb_prev))

            prev_lb_curr = long_band[0]
            prev_lb_prev = 0.0
            cross_down = ((prev_lb_curr > sr[i] and prev_lb_prev <= prev_sr) or
                          (prev_lb_curr < sr[i] and prev_lb_prev >= prev_sr))

        # Pine L2007-L2012
        if cross_up:
            trend_dir[i] = 1
        elif cross_down:
            trend_dir[i] = -1
        else:
            trend_dir[i] = trend_dir[i - 1] if i > 0 else 0

        # Pine L2014
        qqe_line[i] = long_band[i] if trend_dir[i] == 1 else short_band[i]

    return (
        pd.Series(qqe_line, index=close.index),
        smoothed_rsi,
        pd.Series(trend_dir, index=close.index, dtype=float),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SSL HYBRID  (Pine L1827-L1896)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ssl(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute SSL Hybrid scores.

    Returns DataFrame with columns:
        ssl_score_bull, ssl_score_bear,
        ssl_bullish_simple, ssl_bearish_simple,
        ssl_accel_bull, ssl_accel_bear,
        baseline, ssl_exit
    """
    p = {**SSL_DEFAULTS, **kwargs}
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # Pine L1831: baselineRaw = ma(close, sslBaselineLengthEff, sslBaseline)
    baseline_raw = _ma(close, p["baseline_length"], p["baseline_type"],
                       high=high, low=low, close=close)

    # Pine L1834: baseline = applySmoother(baselineRaw, ...)
    baseline = _apply_smoother(baseline_raw, p["smoothing_enabled"],
                               p["smoothing_method"], p["smoothing_factor"])

    # Pine L1837-L1838: sslUp/sslDown raw
    ssl_up_raw = _ma(high, p["ssl2_length"], p["ssl2_type"],
                     high=high, low=low, close=close)
    ssl_down_raw = _ma(low, p["ssl2_length"], p["ssl2_type"],
                       high=high, low=low, close=close)

    # Pine L1841-L1842: smooth SSL bands (lighter, skip Super Smoother)
    smooth_bands = (p["smoothing_enabled"]
                    and p["smoothing_method"] != "Super Smoother")
    ssl_up = _apply_smoother(ssl_up_raw, smooth_bands,
                             "EMA Post-Smooth", p["smoothing_factor"] * 0.5)
    ssl_down = _apply_smoother(ssl_down_raw, smooth_bands,
                               "EMA Post-Smooth", p["smoothing_factor"] * 0.5)

    # Pine L1845-L1850: exit bands
    exit_high_raw = _ma(high, p["exit_length"], p["exit_type"],
                        high=high, low=low, close=close)
    exit_low_raw  = _ma(low, p["exit_length"], p["exit_type"],
                        high=high, low=low, close=close)
    exit_high = _apply_smoother(exit_high_raw, p["smoothing_enabled"],
                                "EMA Post-Smooth", p["smoothing_factor"] * 0.3)
    exit_low  = _apply_smoother(exit_low_raw, p["smoothing_enabled"],
                                "EMA Post-Smooth", p["smoothing_factor"] * 0.3)

    # ── Pine L1853-L1855: exitDirection (var = stateful across bars) ──
    n = len(df)
    exit_direction = np.zeros(n, dtype=int)
    c  = close.values.astype(float)
    eh = exit_high.values.astype(float)
    el = exit_low.values.astype(float)
    for i in range(n):
        if np.isnan(c[i]) or np.isnan(eh[i]) or np.isnan(el[i]):
            exit_direction[i] = exit_direction[i - 1] if i > 0 else 0
            continue
        if c[i] > eh[i]:                             # Pine L1854
            exit_direction[i] = 1
        elif c[i] < el[i]:
            exit_direction[i] = -1
        else:
            exit_direction[i] = exit_direction[i - 1] if i > 0 else 0

    # Pine L1855: sslExit = exitDirection < 0 ? exitHigh : exitLow
    ssl_exit = pd.Series(
        np.where(exit_direction < 0, eh, el), index=df.index
    )

    # ── Pine L1857-L1858: sslDirection (not used in scoring) ──
    ssl_direction = np.zeros(n, dtype=int)
    su = ssl_up.values.astype(float)
    sd = ssl_down.values.astype(float)
    for i in range(n):
        if np.isnan(c[i]) or np.isnan(su[i]) or np.isnan(sd[i]):
            ssl_direction[i] = ssl_direction[i - 1] if i > 0 else 0
            continue
        if c[i] > su[i]:
            ssl_direction[i] = 1
        elif c[i] < sd[i]:
            ssl_direction[i] = -1
        else:
            ssl_direction[i] = ssl_direction[i - 1] if i > 0 else 0

    # ── Pine L1860-L1863 ──
    ssl_bullish_simple = close > baseline
    ssl_bearish_simple = close < baseline

    # ── Pine L1868-L1872: gap widening (nz-safe) ──
    ssl_gap_bull  = _nz(close - baseline)
    ssl_gap_bear  = _nz(baseline - close)
    ssl_accel_bull = _nz(ssl_gap_bull) > _nz(ssl_gap_bull.shift(1))
    ssl_accel_bear = _nz(ssl_gap_bear) > _nz(ssl_gap_bear.shift(1))

    # ── Pine L1874-L1893: score calculation ──
    if p["use_momentum_filter"]:
        # Pine L1884-L1889
        ssl_score_bull = (ssl_bullish_simple & ssl_accel_bull).astype(float)
        ssl_score_bear = (ssl_bearish_simple & ssl_accel_bear).astype(float)
    else:
        # Pine L1892-L1893
        ssl_score_bull = ssl_bullish_simple.astype(float)
        ssl_score_bear = ssl_bearish_simple.astype(float)

    return pd.DataFrame({
        "ssl_score_bull":    ssl_score_bull,
        "ssl_score_bear":    ssl_score_bear,
        "ssl_bullish_simple": ssl_bullish_simple,
        "ssl_bearish_simple": ssl_bearish_simple,
        "ssl_accel_bull":    ssl_accel_bull,
        "ssl_accel_bear":    ssl_accel_bear,
        "baseline":          baseline,
        "ssl_exit":          ssl_exit,
        "ssl_up":            ssl_up,
        "ssl_down":          ssl_down,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# WADDAH ATTAR EXPLOSION  (Pine L1946-L1983)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_wae(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute WAE scores.

    Returns DataFrame with columns:
        wae_score_bull, wae_score_bear,
        wae_bull_confirmation, wae_bear_confirmation,
        wae_accel_bull, wae_accel_bear,
        trend_up, trend_down, explosion_line
    """
    p = {**WAE_DEFAULTS, **kwargs}
    close = df["close"]

    # Pine L1947-L1948
    fast_ma = _pine_ema(close, p["fast_ema"])
    slow_ma = _pine_ema(close, p["slow_ema"])

    # Pine L1949-L1951
    macd      = fast_ma - slow_ma
    macd_prev = fast_ma.shift(1) - slow_ma.shift(1)  # Pine L1950: fastMA[1] - slowMA[1]
    t1 = (macd - macd_prev) * p["sensitivity"]       # Pine L1951

    # Pine L1953-L1957: Bollinger Band explosion line
    basis = _pine_sma(close, p["bb_length"])
    dev   = p["bb_multiplier"] * _pine_stdev(close, p["bb_length"])
    bb_upper = basis + dev
    bb_lower = basis - dev
    explosion_line = bb_upper - bb_lower              # Pine L1957

    # Pine L1959-L1960
    trend_up   = t1.clip(lower=0.0)                   # t1 >= 0 ? t1 : 0
    trend_down = (-t1).clip(lower=0.0)                # t1 < 0  ? -1*t1 : 0

    # Pine L1963-L1964
    wae_bull_confirmation = trend_up   > explosion_line
    wae_bear_confirmation = trend_down > explosion_line

    # Pine L1967-L1968: acceleration = growing AND above explosion line
    trend_up_prev   = _nz(trend_up.shift(1))
    trend_down_prev = _nz(trend_down.shift(1))
    wae_accel_bull = (trend_up   > trend_up_prev)   & (trend_up   > explosion_line)
    wae_accel_bear = (trend_down > trend_down_prev) & (trend_down > explosion_line)

    # Pine L1970-L1983: score with acceleration filter
    if p["use_acceleration"]:
        # Pine L1977-L1980
        wae_score_bull = (wae_bull_confirmation & wae_accel_bull).astype(float)
        wae_score_bear = (wae_bear_confirmation & wae_accel_bear).astype(float)
    else:
        # Pine L1982-L1983
        wae_score_bull = wae_bull_confirmation.astype(float)
        wae_score_bear = wae_bear_confirmation.astype(float)

    return pd.DataFrame({
        "wae_score_bull":         wae_score_bull,
        "wae_score_bear":         wae_score_bear,
        "wae_bull_confirmation":  wae_bull_confirmation,
        "wae_bear_confirmation":  wae_bear_confirmation,
        "wae_accel_bull":         wae_accel_bull,
        "wae_accel_bear":         wae_accel_bear,
        "trend_up":               trend_up,
        "trend_down":             trend_down,
        "explosion_line":         explosion_line,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# QQE MOD  (Pine L1985-L2080)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_qqe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute QQE Mod scores.

    Returns DataFrame with columns:
        qqe_score_bull, qqe_score_bear,
        qqe_bull_signal, qqe_bear_signal,
        qqe_momentum_rising, qqe_momentum_falling,
        primary_rsi, secondary_rsi,
        primary_qqe_trend_line, bollinger_upper, bollinger_lower
    """
    p = {**QQE_DEFAULTS, **kwargs}
    close = df["close"]

    # Pine L2017-L2018: two QQE calculations
    primary_qqe_line, primary_rsi, primary_trend = _calculate_qqe(
        close, p["rsi1_length"], p["rsi1_smoothing"], p["qqe_factor1"])
    _secondary_qqe_line, secondary_rsi, _secondary_trend = _calculate_qqe(
        close, p["rsi2_length"], p["rsi2_smoothing"], p["qqe_factor2"])

    # Pine L2020-L2024: Bollinger on (primaryQQETrendLine - 50)
    qqe_centered = primary_qqe_line - 50
    bb_basis = _pine_sma(qqe_centered, p["bb_length"])
    bb_dev   = p["bb_multiplier"] * _pine_stdev(qqe_centered, p["bb_length"])
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev

    # Pine L2027-L2028: raw conditions
    pri_centered = primary_rsi   - 50
    sec_centered = secondary_rsi - 50
    raw_bull = (sec_centered > p["threshold_secondary"])  & (pri_centered > bb_upper)
    raw_bear = (sec_centered < -p["threshold_secondary"]) & (pri_centered < bb_lower)

    # Pine L2031-L2032: signal strength
    bull_strength = pd.concat([
        sec_centered - p["threshold_secondary"],
        pri_centered - bb_upper,
    ], axis=1).min(axis=1)
    bear_strength = pd.concat([
        -p["threshold_secondary"] - sec_centered,
        bb_lower - pri_centered,
    ], axis=1).min(axis=1)

    # Pine L2034-L2056: consecutive increasing bars (stateful loop)
    n = len(df)
    bull_inc_count   = np.zeros(n, dtype=int)
    bear_inc_count   = np.zeros(n, dtype=int)
    prev_bull_str    = np.zeros(n)
    prev_bear_str    = np.zeros(n)
    qqe_bull_signal  = np.zeros(n, dtype=bool)
    qqe_bear_signal  = np.zeros(n, dtype=bool)

    rb  = raw_bull.values
    rbe = raw_bear.values
    bs  = bull_strength.values.astype(float)
    bes = bear_strength.values.astype(float)

    for i in range(n):
        # Previous bar's strength (Pine var, updated at end of each bar)
        pbs  = prev_bull_str[i - 1] if i > 0 else 0.0
        pbes = prev_bear_str[i - 1] if i > 0 else 0.0

        # Helper: safely coerce raw_bull/raw_bear to bool
        rb_i  = bool(rb[i])  if not (isinstance(rb[i], float) and np.isnan(rb[i]))  else False
        rbe_i = bool(rbe[i]) if not (isinstance(rbe[i], float) and np.isnan(rbe[i])) else False

        # ── Pine L2035-L2041: bull increasing count ──
        if rb_i:
            if bs[i] > pbs:                           # Pine L2036
                bull_inc_count[i] = (bull_inc_count[i - 1] if i > 0 else 0) + 1
            else:                                     # Pine L2039
                bull_inc_count[i] = 0
        else:                                         # Pine L2041
            bull_inc_count[i] = 0

        # ── Pine L2043-L2049: bear increasing count ──
        if rbe_i:
            if bes[i] > pbes:                         # Pine L2044
                bear_inc_count[i] = (bear_inc_count[i - 1] if i > 0 else 0) + 1
            else:                                     # Pine L2047
                bear_inc_count[i] = 0
        else:                                         # Pine L2049
            bear_inc_count[i] = 0

        # Pine L2051-L2052: update prevStrength for next bar
        prev_bull_str[i] = bs[i]  if rb_i  else 0.0
        prev_bear_str[i] = bes[i] if rbe_i else 0.0

        # Pine L2055-L2056
        consec = p["consecutive_increasing_bars"]
        qqe_bull_signal[i] = (bull_inc_count[i] >= consec - 1) and rb_i
        qqe_bear_signal[i] = (bear_inc_count[i] >= consec - 1) and rbe_i

    qqe_bull_signal = pd.Series(qqe_bull_signal, index=df.index)
    qqe_bear_signal = pd.Series(qqe_bear_signal, index=df.index)

    # Pine L2061-L2062: momentum direction (nz-safe)
    pri_rsi_nz = _nz(primary_rsi)
    qqe_momentum_rising  = pri_rsi_nz > pri_rsi_nz.shift(1)
    qqe_momentum_falling = pri_rsi_nz < pri_rsi_nz.shift(1)

    # Pine L2064-L2080: score with momentum filter
    if p["use_momentum_filter"]:
        # Pine L2072-L2076
        qqe_score_bull = (qqe_bull_signal & qqe_momentum_rising).astype(float)
        qqe_score_bear = (qqe_bear_signal & qqe_momentum_falling).astype(float)
    else:
        # Pine L2079-L2080
        qqe_score_bull = qqe_bull_signal.astype(float)
        qqe_score_bear = qqe_bear_signal.astype(float)

    return pd.DataFrame({
        "qqe_score_bull":         qqe_score_bull,
        "qqe_score_bear":         qqe_score_bear,
        "qqe_bull_signal":        qqe_bull_signal,
        "qqe_bear_signal":        qqe_bear_signal,
        "qqe_momentum_rising":    qqe_momentum_rising,
        "qqe_momentum_falling":   qqe_momentum_falling,
        "primary_rsi":            primary_rsi,
        "secondary_rsi":          secondary_rsi,
        "primary_qqe_trend_line": primary_qqe_line,
        "bollinger_upper":        bb_upper,
        "bollinger_lower":        bb_lower,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME Z-SCORE  (Pine L2082-L2096)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_volume(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute volume Z-score confluence.

    Returns DataFrame with columns:
        vol_z, vol_score,
        vol_bull_confirmation, vol_bear_confirmation
    """
    p = {**VOL_DEFAULTS, **kwargs}
    volume = df["volume"].astype(float)

    # Pine L2083-L2085
    vol_mean = _pine_sma(volume, p["lookback"])
    vol_std  = _pine_stdev(volume, p["lookback"])
    vol_z = pd.Series(
        np.where(vol_std > 0, (volume - vol_mean) / vol_std, 0.0),
        index=df.index,
    )

    # Pine L2087-L2093: tiered scoring (binary=True uses HHM_simple threshold)
    if p.get("binary", False):
        vol_score = pd.Series(
            np.where(vol_z >= 1.0, 1.0, 0.0), index=df.index)
    else:
        vol_score = pd.Series(np.where(
            vol_z >= 1.0, 1.0,
            np.where(vol_z >= 0.2, 0.5, 0.0),
        ), index=df.index)

    # Pine L2095-L2096
    vol_bull_confirmation = vol_score >= 0.25
    vol_bear_confirmation = vol_score >= 0.25

    return pd.DataFrame({
        "vol_z":                  vol_z,
        "vol_score":              vol_score,
        "vol_bull_confirmation":  vol_bull_confirmation,
        "vol_bear_confirmation":  vol_bear_confirmation,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFLUENCE SCORING  (Pine L2105-L2109)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_confluence(
    df: pd.DataFrame,
    ssl_kwargs: dict | None = None,
    wae_kwargs: dict | None = None,
    qqe_kwargs: dict | None = None,
    vol_kwargs: dict | None = None,
    enable_volume: bool = True,
) -> pd.DataFrame:
    """Run all four indicators and produce bull/bear confluence scores.

    Returns a wide DataFrame with every per-indicator column plus:
        bull_score  — total bull confluence (0..4 with volume, 0..3 without)
        bear_score  — total bear confluence
        max_score   — denominator (3 or 4)

    Pine L2106:
        bullScore = sslScoreBull + waeScoreBull + qqeScoreBull
                    + (enableVolume ? int(math.round(volScore)) : 0)
    Pine L2109:
        maxScore = 3 + (enableVolume ? 1 : 0)
    """
    ssl = compute_ssl(df, **(ssl_kwargs or {}))
    wae = compute_wae(df, **(wae_kwargs or {}))
    qqe = compute_qqe(df, **(qqe_kwargs or {}))
    vol = compute_volume(df, **(vol_kwargs or {}))

    # Pine: int(math.round(volScore))  — Pine rounds 0.5 → 1 (away from zero)
    if enable_volume:
        vol_contrib = np.floor(vol["vol_score"].values + 0.5).astype(int)
    else:
        vol_contrib = 0

    bull_score = (ssl["ssl_score_bull"] + wae["wae_score_bull"]
                  + qqe["qqe_score_bull"] + vol_contrib)
    bear_score = (ssl["ssl_score_bear"] + wae["wae_score_bear"]
                  + qqe["qqe_score_bear"] + vol_contrib)

    max_score = 3 + (1 if enable_volume else 0)

    result = pd.concat([ssl, wae, qqe, vol], axis=1)
    result["bull_score"] = bull_score
    result["bear_score"] = bear_score
    result["max_score"]  = max_score
    return result
