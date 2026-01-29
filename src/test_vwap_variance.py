"""
VWAP Variance Check - Compare Polygon VWAP vs calculated session VWAP
across multiple data points
"""
import sys
sys.path.insert(0, r'C:\Users\phemm\orb_lab\src')
import pandas as pd
import numpy as np
from orb_backtester import ORBBacktester

# Load data
bt = ORBBacktester(symbol='AAPL', start_date='2025-12-19', end_date='2025-12-19', verbose=False)
bt.load_data()
df = bt.df

# Get Dec 19 data
dec19 = df[df.index.date == pd.Timestamp('2025-12-19').date()].copy()

# Calculate session VWAP like Pine: ta.vwap(hlc3)
# Resets at session start (09:30), uses hlc3 = (high + low + close) / 3
def calc_session_vwap(day_df):
    """Calculate session VWAP matching Pine's ta.vwap(hlc3)"""
    result = pd.Series(index=day_df.index, dtype=float)
    
    hlc3 = (day_df['high'] + day_df['low'] + day_df['close']) / 3
    
    cum_vol = 0.0
    cum_vol_price = 0.0
    
    for ts, row in day_df.iterrows():
        # Reset at RTH open
        if ts.hour == 9 and ts.minute == 30:
            cum_vol = 0.0
            cum_vol_price = 0.0
        
        cum_vol += row['volume']
        cum_vol_price += row['volume'] * hlc3.loc[ts]
        
        if cum_vol > 0:
            result.loc[ts] = cum_vol_price / cum_vol
        else:
            result.loc[ts] = np.nan
    
    return result

# Calculate our session VWAP
session_vwap = calc_session_vwap(dec19)

# Compare at multiple RTH times
print("═══ VWAP VARIANCE CHECK ═══")
print("AAPL Dec 19, 2025")
print("=" * 60)
print(f"{'Time':<8} {'Polygon':>10} {'Session':>10} {'Diff':>10} {'Diff %':>10}")
print("=" * 60)

check_times = ['09:35', '09:45', '10:00', '10:15', '10:19', '10:30', '11:00', '12:00', '13:00', '14:00', '15:00', '15:30']

diffs = []
for t in check_times:
    ts = pd.Timestamp(f'2025-12-19 {t}', tz='US/Eastern')
    if ts in dec19.index:
        poly_vwap = dec19.loc[ts, 'vwap']
        sess_vwap = session_vwap.loc[ts]
        diff = poly_vwap - sess_vwap
        diff_pct = (diff / sess_vwap) * 100 if sess_vwap > 0 else 0
        diffs.append(abs(diff))
        print(f"{t:<8} {poly_vwap:>10.2f} {sess_vwap:>10.2f} {diff:>10.2f} {diff_pct:>9.3f}%")

print("=" * 60)
print(f"Max absolute diff: {max(diffs):.4f}")
print(f"Avg absolute diff: {sum(diffs)/len(diffs):.4f}")
print(f"Min absolute diff: {min(diffs):.4f}")
