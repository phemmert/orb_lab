"""
ORB Batch Worker
================
Standalone process that runs all 4 optimizer phases for a single symbol.
Writes status updates to a JSON file that Streamlit polls.

Usage:
    python batch_worker.py --symbol AMD --trials 200 \
        --train-start 2025-02-01 --train-end 2025-09-30 \
        --test-start 2025-10-01 --test-end 2026-01-27
"""

import argparse
import json
import os
import sys
import time
import optuna
from datetime import datetime

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from orb_optimizer_v3 import PhasedOptimizer, PHASE_CONFIG

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def get_storage_path(symbol):
    """Per-symbol Optuna DB to avoid SQLite contention in parallel runs."""
    return os.path.join(PROJECT_ROOT, f'orb_optuna_{symbol}.db')


def get_status_path(symbol):
    return os.path.join(RESULTS_DIR, f'{symbol}_batch_status.json')


def write_status(symbol, status):
    """Write status JSON for Streamlit to poll."""
    status['updated_at'] = datetime.now().isoformat()
    path = get_status_path(symbol)
    # Write to temp file then rename (atomic on Windows)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(status, f, indent=2)
    os.replace(tmp_path, path)


def cleanup_study(symbol, phase):
    """Delete existing Optuna study for clean re-run."""
    storage = f'sqlite:///{get_storage_path(symbol)}'
    study_name = f'ORB_{symbol}_phase{phase}_v3'
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except KeyError:
        pass  # Study doesn't exist, that's fine


def run_symbol(symbol, trials, train_start, train_end, test_start, test_end):
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
        }
        write_status(symbol, status)

        try:
            # Clean up old study
            cleanup_study(symbol, phase)

            # Run optimizer
            opt = PhasedOptimizer(
                symbol=symbol,
                phase=phase,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_trials=trials,
                results_dir=RESULTS_DIR,
                storage_path=get_storage_path(symbol),
                verbose=True,
            )
            result = opt.run()

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
    from orb_optimizer_v3 import ALWAYS_FIXED
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
    parser.add_argument('--trials', type=int, default=200, help='Trials per phase')
    parser.add_argument('--train-start', default='2025-02-01')
    parser.add_argument('--train-end', default='2025-09-30')
    parser.add_argument('--test-start', default='2025-10-01')
    parser.add_argument('--test-end', default='2026-01-27')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ORB BATCH WORKER: {args.symbol}")
    print(f"  Train: {args.train_start} -> {args.train_end}")
    print(f"  Test:  {args.test_start} -> {args.test_end}")
    print(f"  Trials: {args.trials}")
    print(f"{'='*60}\n")

    run_symbol(
        symbol=args.symbol,
        trials=args.trials,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
