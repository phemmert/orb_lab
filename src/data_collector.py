# data_collector.py
import os
import pandas as pd
from polygon import RESTClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

load_dotenv()

class PolygonDataCollector:
    def __init__(self):
        self.client = RESTClient(api_key=os.getenv("POLYGON_API_KEY"))
        self.cache_dir = r"C:\Users\phemm\orb_lab\data\cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"[DataCollector] Using cache: {self.cache_dir}")
        
    def fetch_bars(self, symbol, days_back=60, bar_size=1, extended_hours=True):
        """Fetch minute bars for a symbol
        
        Args:
            symbol: Ticker symbol
            days_back: Days of history
            bar_size: Minutes per bar (1, 2, 5, etc.)
            extended_hours: If True, include pre/post market (4:00-20:00)
                           If False, RTH only (9:30-16:00)
        """
        suffix = "eth" if extended_hours else "rth"
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{days_back}d_{bar_size}m_{suffix}.parquet")
        
        print(f"📊 Fetching {symbol} ({bar_size}-min bars, {days_back} days, {'extended' if extended_hours else 'RTH'})...")
        
        # Check cache first (valid for 8 hours)
        if os.path.exists(cache_file):
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age < timedelta(hours=8):
                print(f"  ✓ Loading from cache ({cache_age.seconds // 3600}.{(cache_age.seconds % 3600) // 360} hours old)")
                return pd.read_parquet(cache_file)
            else:
                print(f"  ⚠ Cache stale ({cache_age.seconds // 3600} hours), refreshing...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Fetch from Polygon
        bars = []
        for agg in self.client.list_aggs(
            symbol,
            bar_size,
            "minute",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            limit=50000
        ):
            bars.append({
                'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='US/Eastern'),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume,
                'vwap': agg.vwap,
                'trades': agg.transactions
            })
        
        df = pd.DataFrame(bars)
        df.set_index('timestamp', inplace=True)
        
        # Filter to trading hours
        if extended_hours:
            df = df.between_time('04:00', '20:00')
        else:
            df = df.between_time('09:30', '16:00')
        
        # Cache it
        df.to_parquet(cache_file)
        print(f"  ✓ Fetched {len(df)} bars, cached for next time")
        
        time.sleep(0.5)  # Be nice to API
        return df
    
    def fetch_multiple(self, symbols, days_back=60, extended_hours=True):
        """Fetch data for multiple symbols"""
        data = {}
        for symbol in symbols:
            data[symbol] = self.fetch_bars(symbol, days_back, extended_hours=extended_hours)
        return data

# Test it
if __name__ == "__main__":
    collector = PolygonDataCollector()
    
    # Test with AMD
    df = collector.fetch_bars('AMD', days_back=120, bar_size=1, extended_hours=True)
    
    print(f"\n✅ Data Collection Complete!")
    print(f"  AMD: {len(df)} bars")
    print(f"  First: {df.index[0]}")
    print(f"  Last: {df.index[-1]}")
    
    # Check Dec 3
    dec3 = df[df.index.strftime('%Y-%m-%d') == '2025-12-03']
    print(f"\n  Dec 3 bars: {len(dec3)}")
    print(f"  Dec 3 first: {dec3.index[0]}")
    print(f"  Dec 3 last: {dec3.index[-1]}")
