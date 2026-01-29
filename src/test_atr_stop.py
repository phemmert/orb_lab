"""
ATR Stop Validation - Compare Python to Pine
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

# ATR from our calc
atr_series = calc_atr_rth(df, period=14)
rth_atr = atr_series.loc[ts]

# ATR stop calc - Pine line 1136
atr_multiplier = 2.0
atr_stop_long = entry_price - (rth_atr * atr_multiplier)
atr_stop_short = entry_price + (rth_atr * atr_multiplier)

print("═══ ATR STOP DEBUG (Python) ═══")
print(f"Time: 10:19")
print("═══ INPUTS ═══")
print(f"Entry: {entry_price:.2f}")
print(f"rthATR: {rth_atr:.4f}")
print(f"Multiplier: {atr_multiplier:.0f}")
print("═══ ATR STOP ═══")
print(f"LONG stop: {atr_stop_long:.2f}")
print(f"SHORT stop: {atr_stop_short:.2f}")
print()
print("═══ PINE VALUES ═══")
print("Entry: 271.59")
print("rthATR: 0.3155")
print("LONG stop: 270.96")
print("SHORT stop: 272.22")
print()
print("═══ MATCH? ═══")
print(f"Entry: {'✓' if abs(entry_price - 271.59) < 0.01 else '✗'}")
print(f"ATR: {'✓' if abs(rth_atr - 0.3155) < 0.001 else '✗'} (diff: {abs(rth_atr - 0.3155):.4f})")
print(f"LONG stop: {'✓' if abs(atr_stop_long - 270.96) < 0.01 else '✗'}")
print(f"SHORT stop: {'✓' if abs(atr_stop_short - 272.22) < 0.01 else '✗'}")
