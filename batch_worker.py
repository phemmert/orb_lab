"""
ORB Batch Worker
================
Standalone process that runs all 4 optimizer phases for a single symbol.
Writes status updates to a JSON file that Streamlit polls.

Usage:
    python batch_worker.py --symbol AMD --trials-p1 200 --trials-p2 150 \
        --trials-p3 75 --trials-p4 50 \
        --train-start 2025-02-01 --train-end 2025-09-30 \
        --test-start 2025-10-01 --test-end 2026-01-27
"""

import argparse
import json
import os
import sys
import time
import threading
import optuna
from datetime import datetime

# Fix Windows encoding for Unicode characters in output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from Orb_Optimizer_Final import PhasedOptimizer, PHASE_CONFIG

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def get_storage_path(symbol=None):
    """Shared Optuna DB for all symbols — studies have unique names so no conflicts."""
    return os.path.join(PROJECT_ROOT, 'orb_optuna_shared.db')


def init_shared_db():
    """Set WAL mode on shared DB for better concurrent write performance."""
    import sqlite3
    db_path = get_storage_path()
    try:
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=30000')  # 30s timeout for locked DB
        conn.close()
    except Exception:
        pass  # DB may not exist yet, will be created by Optuna


def get_status_path(symbol):
    return os.path.join(RESULTS_DIR, f'{symbol}_batch_status.json')


def write_status(symbol, status):
    """Write status JSON for Streamlit to poll."""
    status['updated_at'] = datetime.now().isoformat()
    path = get_status_path(symbol)
    tmp_path = path + '.tmp'

    with open(tmp_path, 'w') as f:
        json.dump(status, f, indent=2)

    # Retry os.replace — Streamlit may have the file open for reading
    for attempt in range(10):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            time.sleep(0.5)

    # Last resort — write directly (not atomic but won't crash)
    try:
        with open(path, 'w') as f:
            json.dump(status, f, indent=2)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass  # Status update lost but worker continues


def _archive_validation(symbol):
    """Save a timestamped copy of FINAL_validation to history."""
    history_dir = os.path.join(RESULTS_DIR, 'history')
    os.makedirs(history_dir, exist_ok=True)

    src = os.path.join(RESULTS_DIR, f'{symbol}_FINAL_validation.json')
    if not os.path.exists(src):
        return
    with open(src, 'r') as f:
        data = json.load(f)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    data['archive_id'] = ts
    dst = os.path.join(history_dir, f'{symbol}_validation_{ts}.json')
    with open(dst, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Archived validation: {dst}")


class ProgressMonitor:
    """Background thread that polls Optuna DB for trial progress."""

    def __init__(self, symbol, phase, n_trials, storage_path, status, write_fn):
        self.symbol = symbol
        self.phase = phase
        self.n_trials = n_trials
        self.storage_path = storage_path
        self.status = status
        self.write_fn = write_fn
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _monitor(self):
        study_name = f"ORB_{self.symbol}_phase{self.phase}_v3"
        storage = f"sqlite:///{self.storage_path}"
        last_count = 0

        while not self._stop.is_set():
            try:
                study = optuna.load_study(study_name=study_name, storage=storage)
                n_complete = len(study.trials)
                best_val = study.best_value if study.best_trial else None

                if n_complete != last_count:
                    last_count = n_complete
                    phase_status = self.status['phases'].get(str(self.phase), {})
                    phase_status['trials_complete'] = n_complete
                    phase_status['trials_total'] = self.n_trials
                    phase_status['best_so_far'] = round(best_val, 4) if best_val else None
                    self.write_fn(self.symbol, self.status)

            except Exception:
                pass  # Study may not exist yet

            self._stop.wait(3)  # Update every 3 seconds


def cleanup_study(symbol, phase):
    """Delete existing Optuna study for clean re-run."""
    storage = f'sqlite:///{get_storage_path(symbol)}'
    study_name = f'ORB_{symbol}_phase{phase}_v3'
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except KeyError:
        pass  # Study doesn't exist, that's fine


def run_symbol(symbol, phase_trials, train_start, train_end, test_start, test_end):
    """Run all 4 phases for a symbol."""

    status = {
        'symbol': symbol,
        'state': 'running',
        'started_at': datetime.now().isoformat(),
        'current_phase': 0,
        'phases': {},
        'error': None,
    }

    for phase in range(1, 5):
        phase_name = PHASE_CONFIG[phase]['name']
        status['current_phase'] = phase
        status['phases'][str(phase)] = {
            'name': phase_name,
            'state': 'running',
            'started_at': datetime.now().isoformat(),
            'score': None,
            'train': None,
            'test': None,
            'trials_complete': 0,
            'trials_total': phase_trials[phase],
        }
        write_status(symbol, status)

        try:
            # Clean up old study
            cleanup_study(symbol, phase)

            # Start progress monitor
            monitor = ProgressMonitor(
                symbol=symbol,
                phase=phase,
                n_trials=phase_trials[phase],
                storage_path=get_storage_path(symbol),
                status=status,
                write_fn=write_status,
            )
            monitor.start()

            # Run optimizer
            opt = PhasedOptimizer(
                symbol=symbol,
                phase=phase,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_trials=phase_trials[phase],
                results_dir=RESULTS_DIR,
                storage_path=get_storage_path(symbol),
                verbose=True,
            )
            result = opt.run()

            # Stop progress monitor
            monitor.stop()

            # Update status with results
            status['phases'][str(phase)].update({
                'state': 'complete',
                'completed_at': datetime.now().isoformat(),
                'score': result.get('best_score'),
                'params': result.get('best_params'),
                'train': result.get('train_results'),
                'test': result.get('test_results'),
            })
            write_status(symbol, status)

            print(f"\n{'='*50}")
            print(f"  {symbol} Phase {phase} ({phase_name}) COMPLETE")
            print(f"  Score: {result.get('best_score', 'N/A')}")
            print(f"{'='*50}\n")

            # --- CIRCUIT BREAKER ---
            # Stop wasting compute if this symbol clearly has no edge
            best_score = result.get('best_score')
            test_r = result.get('test_results', {}).get('total_r', 0)
            train_r = result.get('train_results', {}).get('total_r', 0)

            if best_score is not None and (best_score == float('-inf') or best_score < 0):
                reason = f"Phase {phase} score: {best_score}"
                if test_r < 0:
                    reason += f", Test R: {test_r:+.1f}"
                print(f"\n{'!'*50}")
                print(f"  CIRCUIT BREAKER: {symbol}")
                print(f"  {reason}")
                print(f"  No edge detected — skipping remaining phases.")
                print(f"{'!'*50}\n")

                status['state'] = 'no_edge'
                status['error'] = f"Circuit breaker: {reason}"
                write_status(symbol, status)
                return status

        except Exception as e:
            status['phases'][str(phase)].update({
                'state': 'error',
                'error': str(e),
            })
            status['state'] = 'error'
            status['error'] = f"Phase {phase} ({phase_name}): {str(e)}"
            write_status(symbol, status)
            print(f"\nERROR: {symbol} Phase {phase}: {e}\n")
            return status

    # All phases complete — run full validation
    status['current_phase'] = 'validation'
    write_status(symbol, status)

    try:
        _run_full_validation(symbol, train_start, train_end, test_start, test_end)
        _archive_validation(symbol)
        status['state'] = 'complete'
        status['completed_at'] = datetime.now().isoformat()
    except Exception as e:
        status['state'] = 'complete_no_validation'
        status['error'] = f"Validation failed: {str(e)}"

    write_status(symbol, status)
    print(f"\n{'='*50}")
    print(f"  {symbol} ALL PHASES COMPLETE")
    print(f"{'='*50}\n")
    return status


def _run_full_validation(symbol, train_start, train_end, test_start, test_end):
    """Run the full combined validation (FCF) and save FINAL file."""
    from orb_backtester import ORBBacktester

    # Collect all params from all 4 phases
    all_params = {}
    for phase in range(1, 5):
        path = os.path.join(RESULTS_DIR, f'{symbol}_phase{phase}.json')
        with open(path, 'r') as f:
            data = json.load(f)
        all_params.update(data.get('best_params', {}))

    # Add always-fixed params
    from Orb_Optimizer_Final import ALWAYS_FIXED
    full_params = ALWAYS_FIXED.copy()
    full_params.update(all_params)
    # Phase 3+ has enable_confluence via forced_params
    full_params['enable_confluence'] = True

    # Run train
    bt_train = ORBBacktester(
        symbol=symbol,
        start_date=train_start,
        end_date=train_end,
        verbose=False,
        **full_params
    )
    train_results = bt_train.run()

    # Run test
    bt_test = ORBBacktester(
        symbol=symbol,
        start_date=test_start,
        end_date=test_end,
        verbose=False,
        **full_params
    )
    test_results = bt_test.run()

    # Save FINAL validation
    output = {
        'symbol': symbol,
        'validation_date': datetime.now().isoformat(),
        'all_params': full_params,
        'train_period': {'start': train_start, 'end': train_end},
        'test_period': {'start': test_start, 'end': test_end},
        'train_results': {
            'total_trades': train_results['total_trades'],
            'win_rate': train_results['win_rate'],
            'total_r': train_results['total_r'],
            'avg_r_per_trade': train_results['avg_r_per_trade'],
            'profit_factor': train_results['profit_factor'],
            'max_drawdown_r': train_results['max_drawdown_r'],
        },
        'test_results': {
            'total_trades': test_results['total_trades'],
            'win_rate': test_results['win_rate'],
            'total_r': test_results['total_r'],
            'avg_r_per_trade': test_results['avg_r_per_trade'],
            'profit_factor': test_results['profit_factor'],
            'max_drawdown_r': test_results['max_drawdown_r'],
        },
    }

    path = os.path.join(RESULTS_DIR, f'{symbol}_FINAL_validation.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  FINAL validation saved: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORB Batch Worker')
    parser.add_argument('--symbol', required=True, help='Ticker symbol')
    parser.add_argument('--trials-p1', type=int, default=200, help='Trials for Phase 1')
    parser.add_argument('--trials-p2', type=int, default=150, help='Trials for Phase 2')
    parser.add_argument('--trials-p3', type=int, default=75, help='Trials for Phase 3')
    parser.add_argument('--trials-p4', type=int, default=50, help='Trials for Phase 4')
    parser.add_argument('--train-start', default='2025-02-01')
    parser.add_argument('--train-end', default='2025-09-30')
    parser.add_argument('--test-start', default='2025-10-01')
    parser.add_argument('--test-end', default='2026-01-27')

    args = parser.parse_args()

    phase_trials = {
        1: args.trials_p1,
        2: args.trials_p2,
        3: args.trials_p3,
        4: args.trials_p4,
    }

    print(f"\n{'='*60}")
    print(f"  ORB BATCH WORKER: {args.symbol}")
    print(f"  Train: {args.train_start} -> {args.train_end}")
    print(f"  Test:  {args.test_start} -> {args.test_end}")
    print(f"  Trials: P1={phase_trials[1]} P2={phase_trials[2]} P3={phase_trials[3]} P4={phase_trials[4]}")
    print(f"  Storage: {get_storage_path()} (shared)")
    print(f"{'='*60}\n")

    # Initialize shared DB with WAL mode for concurrent access
    init_shared_db()

    run_symbol(
        symbol=args.symbol,
        phase_trials=phase_trials,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
