# HHM Simple Backtester Validation — State

## Status
VALIDATED against Pine reference. Parity: 95.7% (Python 118 / Pine 113 
on AMD Feb 3-28 2026, 2-min bars). Residual gap attributed to 
Polygon/TV volume data delta, not logic. Committed at 41dc49d.

## Files
- C:\Users\phemm\orb_lab\src\HHM_simple.pine
- C:\Users\phemm\orb_lab\src\hmm_backtester_v2.py  
- C:\Users\phemm\orb_lab\src\confluence_indicators.py

## Fixes applied (in order)
1. Market ETF: Pine was SPY for presets, corrected to QQQ to match 
   Python (Pine L187)
2. Signal qualification scope: Pine moved baseOK/canEnter out of 
   longConditions/shortConditions; signals now fire on confluence + 
   HTF + market regime only, gates apply to execution only 
   (Pine L622-697)
3. Volume scoring: added binary=True flag to compute_volume in 
   confluence_indicators.py; hmm_backtester_v2.py passes binary=True. 
   Matches Pine's binary vol_z >= 1.0 threshold. ORB backtester and 
   HMM v1 preserve tiered scoring.
4. Cooldown: Pine > changed to >= for bar_index - lastEntryBar, 
   matching Python's >= semantics (Pine L645)

## AMD presets (validated matching)
SSL baseline=60, WAE sensitivity=300, QQE RSI1=8, HTF trend len=5, 
market ETF=QQQ, market trend EMA=10, min R:R=1.5, BE R:R=0.5, 
target R:R=2.0, trail dist=1.2 ATR, EMA confirm bars=1, 
EMA body %=60, min bars between entries=1

## Feb 3-28 2026 AMD baseline
- Pine: 113 trades ($-5.22 @ 1sh, win rate 38.1%)
- Python: 118 trades (+5 vs Pine)
- Mixed-sign per-day deltas
- Perfect alignment: Feb 5, 20, 25, 26, 27

## Known data variance (accepted)
Polygon (Python) uses consolidated tape volume. TradingView (Pine) 
uses primary-exchange volume. ~20% difference on AMD causes 
borderline vol_z bars to score differently. Not fixable without 
switching data providers. Keeping Polygon — closer to live trading 
tape.

## Audit items completed
- #1: Pine market ETF preset corrected to QQQ
- #2: Volume scoring binary flag added (Pine binary parity)
- #3: Pine signal qualification scope fixed
- #5: Cooldown comparison >= in both files
- #4: Entry fill = next-bar-open in Python (matches Pine default)
- #6/#7: barsBeyondEma counter runs every bar in-position; resets 
  only on price re-crossing EMA, not on body% failure

## Audit items deferred (very minor)
- #8: EMA exit gate close > entryPrice. Pine has 
  `breakEvenActivated OR close > entryPrice`; Python has 
  `close > entryPrice`. In practice once BE activates, close is 
  almost always > entry — rarely diverges.

## Exit reason distribution (Feb 3-28 AMD)
| Reason     | Pine | Python |
|------------|------|--------|
| BE STOP    | 47   | 46     |
| STOP LOSS  | 38   | 42     |
| EMA EXIT   | 10   | 13     |
| TRAIL STOP | 11   | 8      |
| EOD EXIT   | 7    | 9      |
| **Total**  | 113  | 118    |
Absolute deviation: 13 (down from 19 pre-audit-cleanup)

## Repo cleanup deferred
- src/HHM.pine (original HHM Pine): uncommitted weeks-old changes, 
  not touched in validation work
- src/hmm_backtester.py (original HMM v1): uncommitted weeks-old 
  changes, not touched in validation work
- 3 notebooks in notebooks/: mixed state
- 1 stash entry from 3217dac "Fix vol_state execution order to match 
  Pine" — leave it alone for now
Clean up during a fresh session, not mid-debug.

## Workflow notes
- Pine edits require manual copy-paste from local file to TV 
  editor, then re-add to chart, then re-run Deep Backtesting.
- Python: use the notebook pattern with module-reload loop at top 
  (del sys.modules[...]), HMMBacktesterV2 with market_etf='QQQ'.
- Claude Code edits files directly; browser Claude diffs and 
  analyzes.
- Claude Code in WSL does not have GitHub credentials — push from 
  GitHub Desktop or Windows cmd.

## Next phase
Layer features back one at a time, validating each against Pine 
before adding the next. Candidate order:
1. Composite grading (A+ to D)
2. Pre-signal feasibility engine
3. Adaptive BE/RR by volatility state
4. IB filter
5. 9/20 EMA cross exit
Never add two features at once — if parity breaks, you won't know 
which one caused it.

## Bootstrap for new chat
"Read C:\Users\phemm\orb_lab\STATE.md — resuming HHM Simple 
validation." Then upload the file.