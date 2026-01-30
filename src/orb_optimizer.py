"""
ORB Optimizer v2 - Robust Parameter Search
==========================================
Layer 2 of the optimization architecture.

Improvements over v1:
- Robust scoring: median_monthly_R - DD_penalty - variance_penalty
- Narrow search option: constrains params around proven defaults
- Penalizes fragility (low win rate, high variance)

Usage:
    from orb_optimizer import ORBOptimizer
    
    optimizer = ORBOptimizer(
        symbol='AMD',
        train_start='2025-04-01',
        train_end='2025-09-30',
        test_start='2025-10-01',
        test_end='2025-12-31',
        n_trials=100,
        objective_metric='robust',  # NEW: penalizes overfitting
        narrow_search=True          # NEW: constrains around defaults
    )
    best_params = optimizer.run()
"""

import optuna
from optuna.samplers import TPESampler
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orb_backtester import ORBBacktester

# Project root for default paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ORBOptimizer:
    """
    Optuna-powered optimizer for ORB strategy parameters.
    Finds optimal settings for each symbol.
    """
    
    def __init__(
        self,
        symbol: str = 'AMD',
        train_start: str = '2025-04-01',
        train_end: str = '2025-09-30',
        test_start: str = '2025-10-01',
        test_end: str = '2025-12-31',
        n_trials: int = 100,
        
        # Optimization target
        objective_metric: str = 'robust',  # 'robust', 'total_r', 'profit_factor', 'sharpe'
        min_trades: int = 20,
        
        # Search space control
        narrow_search: bool = True,  # Constrain around defaults

        # Output & Storage
        results_dir: Optional[str] = None,
        storage_path: Optional[str] = None,  # SQLite path for Optuna dashboard
        verbose: bool = True,
    ):
        self.symbol = symbol
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.n_trials = n_trials
        self.objective_metric = objective_metric
        self.min_trades = min_trades
        self.narrow_search = narrow_search
        self.results_dir = results_dir or os.path.join(PROJECT_ROOT, 'results')
        self.storage_path = storage_path or os.path.join(PROJECT_ROOT, 'orb_optuna.db')
        self.verbose = verbose

        os.makedirs(self.results_dir, exist_ok=True)
        
        self.best_params = None
        self.best_score = float('-inf')
        self.study = None
        
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        
        # ═══════════════════════════════════════════════════════════════════
        # SEARCH SPACE - Narrow vs Wide
        # ═══════════════════════════════════════════════════════════════════
        
        if self.narrow_search:
            # NARROW SEARCH - ±20-30% around proven defaults
            params = {
                # Exit params (defaults: BE=0.5, PT=2.0, Trail=1.2)
                'break_even_rr': trial.suggest_float('break_even_rr', 0.4, 0.7, step=0.1),
                'profit_target_rr': trial.suggest_float('profit_target_rr', 1.75, 2.5, step=0.25),
                'trailing_stop_distance': trial.suggest_float('trailing_stop_distance', 1.0, 1.5, step=0.1),
                'ema_tighten_zone': trial.suggest_float('ema_tighten_zone', 0.25, 0.4, step=0.05),
                'tightened_trail_distance': trial.suggest_float('tightened_trail_distance', 0.25, 0.4, step=0.05),
                'ema_confirmation_bars': trial.suggest_int('ema_confirmation_bars', 1, 2),
                
                # Stop params (defaults: ATR=2.0, minATR=0.15)
                'atr_stop_mult': trial.suggest_float('atr_stop_mult', 1.75, 2.25, step=0.25),
                'min_stop_atr': trial.suggest_float('min_stop_atr', 0.10, 0.20, step=0.05),
                'swing_lookback': trial.suggest_int('swing_lookback', 4, 6),
                
                # R:R filter (default: 1.5)
                'min_acceptable_rr': trial.suggest_float('min_acceptable_rr', 1.25, 1.75, step=0.25),
                
                # Vol thresholds (defaults: 0.8/1.3/2.0)
                'low_vol_threshold': trial.suggest_float('low_vol_threshold', 0.7, 0.9, step=0.1),
                'high_vol_threshold': trial.suggest_float('high_vol_threshold', 1.2, 1.4, step=0.1),
                'extreme_vol_threshold': trial.suggest_float('extreme_vol_threshold', 1.8, 2.2, step=0.1),
                
                # Trading window (default: 630 = 10:30)
                'trading_start_minutes': 570,
                'trading_end_minutes': trial.suggest_int('trading_end_minutes', 615, 660, step=15),
                
                # Feature toggles - keep proven defaults ON
                'use_break_even': True,
                'use_trailing_stop': True,
                'use_ema_exit': trial.suggest_categorical('use_ema_exit', [True, False]),
                'use_aggressive_trailing': trial.suggest_categorical('use_aggressive_trailing', [True, False]),
            }
        else:
            # WIDE SEARCH - full exploration
            params = {
                'break_even_rr': trial.suggest_float('break_even_rr', 0.3, 1.0, step=0.1),
                'profit_target_rr': trial.suggest_float('profit_target_rr', 1.5, 3.0, step=0.25),
                'trailing_stop_distance': trial.suggest_float('trailing_stop_distance', 0.8, 2.0, step=0.1),
                'ema_tighten_zone': trial.suggest_float('ema_tighten_zone', 0.2, 0.5, step=0.05),
                'tightened_trail_distance': trial.suggest_float('tightened_trail_distance', 0.2, 0.5, step=0.05),
                'ema_confirmation_bars': trial.suggest_int('ema_confirmation_bars', 1, 3),
                'atr_stop_mult': trial.suggest_float('atr_stop_mult', 1.5, 3.0, step=0.25),
                'min_stop_atr': trial.suggest_float('min_stop_atr', 0.10, 0.25, step=0.05),
                'swing_lookback': trial.suggest_int('swing_lookback', 3, 7),
                'min_acceptable_rr': trial.suggest_float('min_acceptable_rr', 1.0, 2.0, step=0.25),
                'low_vol_threshold': trial.suggest_float('low_vol_threshold', 0.6, 0.9, step=0.1),
                'high_vol_threshold': trial.suggest_float('high_vol_threshold', 1.2, 1.6, step=0.1),
                'extreme_vol_threshold': trial.suggest_float('extreme_vol_threshold', 1.8, 2.5, step=0.1),
                'trading_start_minutes': 570,
                'trading_end_minutes': trial.suggest_int('trading_end_minutes', 600, 690, step=15),
                'use_break_even': trial.suggest_categorical('use_break_even', [True, False]),
                'use_trailing_stop': trial.suggest_categorical('use_trailing_stop', [True, False]),
                'use_ema_exit': trial.suggest_categorical('use_ema_exit', [True, False]),
                'use_aggressive_trailing': trial.suggest_categorical('use_aggressive_trailing', [True, False]),
            }
        
        # ═══════════════════════════════════════════════════════════════════
        # RUN BACKTEST
        # ═══════════════════════════════════════════════════════════════════
        
        try:
            bt = ORBBacktester(
                symbol=self.symbol,
                start_date=self.train_start,
                end_date=self.train_end,
                verbose=False,
                **params
            )
            results = bt.run()
        except Exception as e:
            if self.verbose:
                print(f"  Trial {trial.number} failed: {e}")
            return float('-inf')
        
        # ═══════════════════════════════════════════════════════════════════
        # CALCULATE SCORE
        # ═══════════════════════════════════════════════════════════════════
        
        if results['total_trades'] < self.min_trades:
            return float('-inf')
        
        if self.objective_metric == 'robust':
            # ═══════════════════════════════════════════════════════════════
            # ROBUST SCORING - Surgeon's formula
            # score = median_monthly_R - 0.5*max_DD - 0.2*stdev_monthly
            # ═══════════════════════════════════════════════════════════════
            
            if not results['trades']:
                return float('-inf')
            
            # Group trades by month
            monthly_r = defaultdict(float)
            for t in results['trades']:
                month_key = t['date'][:7]  # "2025-04"
                monthly_r[month_key] += t['r_multiple']
            
            monthly_values = list(monthly_r.values())
            
            if len(monthly_values) < 2:
                return float('-inf')
            
            median_monthly = np.median(monthly_values)
            stdev_monthly = np.std(monthly_values)
            max_dd = results['max_drawdown_r']
            
            # Robust score
            score = median_monthly - (0.5 * max_dd) - (0.2 * stdev_monthly)
            
            # Penalty for negative months (Surgeon's addition)
            # Systems that lose 4/6 months but win big in 2 blow up live
            neg_months = sum(1 for m in monthly_values if m < 0)
            score -= 0.5 * neg_months
            
            # Penalty for fragility
            if results['win_rate'] < 30:
                score -= 1.0
                
        elif self.objective_metric == 'total_r':
            score = results['total_r']
            
        elif self.objective_metric == 'profit_factor':
            score = results['profit_factor'] if results['profit_factor'] != float('inf') else 10.0
            
        elif self.objective_metric == 'avg_r_per_trade':
            score = results['avg_r_per_trade']
            
        elif self.objective_metric == 'sharpe':
            if results['trades']:
                r_values = [t['r_multiple'] for t in results['trades']]
                std_r = np.std(r_values) if len(r_values) > 1 else 1.0
                score = results['avg_r_per_trade'] / std_r if std_r > 0 else 0
            else:
                score = 0
        else:
            score = results['total_r']
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
        
        return score
    
    def run(self) -> Dict[str, Any]:
        """Run optimization and validation."""
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  ORB OPTIMIZER v2: {self.symbol}")
            print(f"{'='*70}")
            print(f"\n  Training: {self.train_start} to {self.train_end}")
            print(f"  Testing:  {self.test_start} to {self.test_end}")
            print(f"  Trials:   {self.n_trials}")
            print(f"  Metric:   {self.objective_metric}")
            print(f"  Search:   {'NARROW (around defaults)' if self.narrow_search else 'WIDE (full exploration)'}")
            print(f"\n{'─'*70}")
            print(f"  OPTIMIZATION IN PROGRESS...")
            print(f"{'─'*70}\n")
        
        # Create study with SQLite storage for dashboard visualization
        sampler = TPESampler(seed=42)
        storage = f'sqlite:///{self.storage_path}'
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'ORB_{self.symbol}_v2',
            storage=storage,
            load_if_exists=True
        )
        
        # Suppress Optuna logging unless verbose
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )
        
        # Get best
        best_params = self.study.best_params
        best_trial = self.study.best_trial
        
        # Add fixed params back
        if self.narrow_search:
            best_params['use_break_even'] = True
            best_params['use_trailing_stop'] = True
            best_params['trading_start_minutes'] = 570
        
        if self.verbose:
            print(f"\n{'─'*70}")
            print(f"  OPTIMIZATION COMPLETE")
            print(f"{'─'*70}\n")
            print(f"  Best trial: #{best_trial.number}")
            print(f"  Best {self.objective_metric}: {best_trial.value:.3f}")
            print(f"\n  Best parameters:")
            for k, v in sorted(best_params.items()):
                print(f"    {k}: {v}")
        
        # ═══════════════════════════════════════════════════════════════════
        # VALIDATION
        # ═══════════════════════════════════════════════════════════════════
        
        if self.verbose:
            print(f"\n{'─'*70}")
            print(f"  VALIDATION ON HELD-OUT DATA")
            print(f"{'─'*70}\n")
        
        # Training results
        bt_train = ORBBacktester(
            symbol=self.symbol,
            start_date=self.train_start,
            end_date=self.train_end,
            verbose=False,
            **best_params
        )
        train_results = bt_train.run()
        
        # Test results
        bt_test = ORBBacktester(
            symbol=self.symbol,
            start_date=self.test_start,
            end_date=self.test_end,
            verbose=False,
            **best_params
        )
        test_results = bt_test.run()
        
        if self.verbose:
            print(f"  {'Metric':<20} {'Training':>12} {'Testing':>12}")
            print(f"  {'-'*44}")
            print(f"  {'Total Trades':<20} {train_results['total_trades']:>12} {test_results['total_trades']:>12}")
            print(f"  {'Win Rate':<20} {train_results['win_rate']:>11.1f}% {test_results['win_rate']:>11.1f}%")
            print(f"  {'Total R':<20} {train_results['total_r']:>+12.2f} {test_results['total_r']:>+12.2f}")
            print(f"  {'Avg R/Trade':<20} {train_results['avg_r_per_trade']:>+12.3f} {test_results['avg_r_per_trade']:>+12.3f}")
            print(f"  {'Profit Factor':<20} {train_results['profit_factor']:>12.2f} {test_results['profit_factor']:>12.2f}")
            print(f"  {'Max Drawdown':<20} {train_results['max_drawdown_r']:>12.2f} {test_results['max_drawdown_r']:>12.2f}")
        
        # Save results
        output = {
            'symbol': self.symbol,
            'optimization_date': datetime.now().isoformat(),
            'optimizer_version': 'v2_robust',
            'objective_metric': self.objective_metric,
            'narrow_search': self.narrow_search,
            'train_period': {'start': self.train_start, 'end': self.train_end},
            'test_period': {'start': self.test_start, 'end': self.test_end},
            'n_trials': self.n_trials,
            'best_score': best_trial.value,
            'best_params': best_params,
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
            }
        }
        
        output_path = os.path.join(self.results_dir, f'{self.symbol}_optimal_v2.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        if self.verbose:
            print(f"\n  ✓ Results saved to: {output_path}")
            print(f"\n{'='*70}\n")
        
        return output


# ═══════════════════════════════════════════════════════════════════════════
# BATCH OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

class BatchOptimizer:
    """Run optimization across multiple symbols."""

    def __init__(
        self,
        symbols: list = ['AMD', 'NVDA', 'TSLA', 'AAPL'],
        train_start: str = '2025-04-01',
        train_end: str = '2025-09-30',
        test_start: str = '2025-10-01',
        test_end: str = '2025-12-31',
        n_trials: int = 100,
        objective_metric: str = 'robust',
        narrow_search: bool = True,
        results_dir: Optional[str] = None,
        storage_path: Optional[str] = None,
    ):
        self.symbols = symbols
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.n_trials = n_trials
        self.objective_metric = objective_metric
        self.narrow_search = narrow_search
        self.results_dir = results_dir or os.path.join(PROJECT_ROOT, 'results')
        self.storage_path = storage_path or os.path.join(PROJECT_ROOT, 'orb_optuna.db')
        self.results = {}
    
    def run(self) -> Dict[str, Any]:
        """Run optimization for all symbols."""
        
        print(f"\n{'='*70}")
        print(f"  BATCH OPTIMIZATION v2: {len(self.symbols)} symbols")
        print(f"  {', '.join(self.symbols)}")
        print(f"  Metric: {self.objective_metric} | Search: {'NARROW' if self.narrow_search else 'WIDE'}")
        print(f"{'='*70}\n")
        
        for i, symbol in enumerate(self.symbols, 1):
            print(f"\n[{i}/{len(self.symbols)}] Optimizing {symbol}...")
            
            optimizer = ORBOptimizer(
                symbol=symbol,
                train_start=self.train_start,
                train_end=self.train_end,
                test_start=self.test_start,
                test_end=self.test_end,
                n_trials=self.n_trials,
                objective_metric=self.objective_metric,
                narrow_search=self.narrow_search,
                results_dir=self.results_dir,
                storage_path=self.storage_path,
                verbose=True
            )
            
            self.results[symbol] = optimizer.run()
        
        self._print_summary()
        return self.results
    
    def _print_summary(self):
        """Print comparison across all symbols."""
        
        print(f"\n{'='*70}")
        print(f"  BATCH OPTIMIZATION SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"  {'Symbol':<8} {'Train R':>10} {'Test R':>10} {'Train WR':>10} {'Test WR':>10} {'PF':>8}")
        print(f"  {'-'*56}")
        
        for symbol, result in self.results.items():
            train = result['train_results']
            test = result['test_results']
            print(f"  {symbol:<8} {train['total_r']:>+10.2f} {test['total_r']:>+10.2f} "
                  f"{train['win_rate']:>9.1f}% {test['win_rate']:>9.1f}% "
                  f"{test['profit_factor']:>8.2f}")
        
        print(f"\n{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    optimizer = ORBOptimizer(
        symbol='AMD',
        train_start='2025-04-01',
        train_end='2025-09-30',
        test_start='2025-10-01',
        test_end='2025-12-31',
        n_trials=100,
        objective_metric='robust',
        narrow_search=True,
        verbose=True
    )
    results = optimizer.run()
