"""
VWAP Stop Validation - Compare Python to Pine
Target: AAPL Dec 19 @ 10:19
"""
import sys
sys.path.insert(0, r'C:\Users\phemm\orb_lab\src')
import pandas as pd
from orb_backtester import ORBBacktester
from orb_core import calc_atr_rth

# Load data
bt = ORBBacktester(symbol='AAPL', start_date='2025-12-19', end_date='2025-12-19', verbose=False)
bt.load_data()
df = bt.df

# Get 10:19 bar
ts = pd.Timestamp('2025-12-19 10:19', tz='US/Eastern')
bar = df.loc[ts]

# Entry = close
entry_price = bar['close']

# VWAP from data (Polygon provides this)
vwap_value = bar['vwap']

# ATR from our calc (current, not lagged - matching Pine debug)
atr_series = calc_atr_rth(df, period=14)
rth_atr = atr_series.loc[ts]

# VWAP stop calc - Pine line 1142
vwap_stop_distance = 0.3
vwap_stop_long = vwap_value - (rth_atr * vwap_stop_distance)
vwap_stop_short = vwap_value + (rth_atr * vwap_stop_distance)

print("═══ VWAP STOP DEBUG (Python) ═══")
print(f"Time: 10:19")
print("═══ INPUTS ═══")
print(f"Entry: {entry_price:.2f}")
print(f"rthATR: {rth_atr:.4f}")
print(f"VWAP: {vwap_value:.2f}")
print(f"Distance mult: {vwap_stop_distance}")
print("═══ VWAP STOP ═══")
print(f"LONG stop: {vwap_stop_long:.2f}")
print(f"SHORT stop: {vwap_stop_short:.2f}")
print()
print("═══ PINE VALUES ═══")
print("Entry: 271.59")
print("rthATR: 0.3377")
print("VWAP: 272.16")
print("LONG stop: 272.06")
print("SHORT stop: 272.26")
print()
print("═══ MATCH? ═══")
print(f"Entry: {'✓' if abs(entry_price - 271.59) < 0.01 else '✗'}")
print(f"ATR: {'✓' if abs(rth_atr - 0.3377) < 0.01 else '✗'} (Python: {rth_atr:.4f}, Pine: 0.3377, diff: {abs(rth_atr - 0.3377):.4f})")
print(f"VWAP: {'✓' if abs(vwap_value - 272.16) < 0.01 else '✗'} (Python: {vwap_value:.2f})")
print(f"LONG stop: {'✓' if abs(vwap_stop_long - 272.06) < 0.01 else '✗'} (Python: {vwap_stop_long:.2f})")
print(f"SHORT stop: {'✓' if abs(vwap_stop_short - 272.26) < 0.01 else '✗'} (Python: {vwap_stop_short:.2f})")
