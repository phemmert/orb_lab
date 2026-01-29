# ORB Optimizer Project
## Location: `C:\Users\phemm\orb_lab\`

---

## Project Structure

```
orb_lab/
├── src/                    # Python source code
│   ├── data_collector.py   # Polygon API data fetching (with cache)
│   ├── orb_backtester.py   # Main backtester (replicates Pine logic)
│   ├── walk_forward.py     # Walk-forward validation (coming soon)
│   └── optuna_optimizer.py # Parameter optimization (coming soon)
│
├── notebooks/              # Jupyter notebooks for testing/analysis
│   └── 01_test_backtester.py  # Test script (copy to notebook)
│
├── data/                   # Data directory
│   └── cache/              # Cached Polygon data (parquet files)
│
├── pine/                   # Pine Script source files
│   └── 5ORB.pine           # Source of truth for trading logic
│
├── docs/                   # Documentation
│   └── PARAMETER_SPACE.md  # Complete parameter documentation
│
└── README.md               # This file
```

---

## Setup Instructions

### 1. Copy Files to Your Machine

Copy these files to `C:\Users\phemm\orb_lab\`:

```
orb_lab/
├── src/
│   ├── data_collector.py
│   └── orb_backtester.py
├── notebooks/
│   └── 01_test_backtester.py
├── data/
│   └── cache/              (create empty folder)
├── pine/
│   └── 5ORB.pine
└── docs/
    └── PARAMETER_SPACE.md
```

### 2. Environment Setup

Make sure your `.env` file has your Polygon API key:

```
# In C:\Users\phemm\orb_lab\.env (or your existing .env location)
POLYGON_API_KEY=your_api_key_here
```

Or set the environment variable directly in Anaconda.

### 3. Test the Backtester

1. Open JupyterLab (your orb_lab environment)
2. Create a new notebook
3. Copy/paste cells from `notebooks/01_test_backtester.py`
4. Run each cell

You should see:
- Data fetching from Polygon
- Backtest results for NVDA
- Equity curve plot
- Parameter sensitivity test

---

## Architecture Overview

```
Polygon API
    ↓
data_collector.py (fetch + cache)
    ↓
orb_backtester.py (simulate trades)
    ↓
walk_forward.py (prevent overfitting)
    ↓
optuna_optimizer.py (find best params)
    ↓
Export to Pine preset format
```

---

## Parameter Organization

### Preset-Controlled (from enhanced_preset_generator.py)
- SSL baseline/length
- WAE EMA lengths
- QQE RSI lengths
- Volatility thresholds per symbol

### Optuna-Optimizable (backtester knobs)
- ORB period (5/10/15/30 min)
- Exit timeframe (1-min vs 2-min)
- Breakout threshold
- Confluence score minimum
- Exit strategy parameters
- Risk management settings
- All other user-adjustable inputs

See `docs/PARAMETER_SPACE.md` for complete list.

---

## Current Status

- [x] Parameter space documented
- [x] Backtester core built
- [ ] Walk-forward validation
- [ ] Optuna integration
- [ ] Multi-symbol optimization
- [ ] Preset export

---

## Notes

- Cache files expire after 16 hours automatically
- The backtester matches 5ORB.pine v65 logic
- All indicator calculations (SSL, WAE, QQE) are replicated in Python
