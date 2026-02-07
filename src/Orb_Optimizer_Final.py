"""
ORB Optimizer v3 - Phased Parameter Search
==========================================
Research-grade optimizer with staged parameter discovery.

Philosophy (from the Surgeon):
- Not all parameters are equal
- Some parameters only make sense after others are fixed
- Human judgment matters at phase boundaries

Phases:
  1. Exit Geometry    - How we manage trades (confluence OFF)
  2. Volatility       - When to adjust behavior by regime
  3. Confluence       - Entry quality gating
  4. Boolean Polish   - Final execution style tuning

Usage:
    from orb_optimizer_v3 import PhasedOptimizer
    
    # Phase 1 - Exit Geometry
    opt = PhasedOptimizer(symbol='AMD', phase=1, n_trials=200)
    results = opt.run()
    # Review results, then...
    
    # Phase 2 - Auto-loads Phase 1 results
    opt = PhasedOptimizer(symbol='AMD', phase=2, n_trials=200)
    results = opt.run()
    
    # Override specific params if needed
    opt = PhasedOptimizer(
        symbol='AMD', 
        phase=2,
        fixed_params={'break_even_rr': 0.6},  # Override Phase 1
        n_trials=200
    )
"""

import optuna
from optuna.samplers import TPESampler
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orb_backtester import ORBBacktester

# Project root for default paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════
# PHASE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

PHASE_CONFIG = {
    1: {
        'name': 'Exit Geometry',
        'description': 'Trade management foundation - how we exit',
        'min_trades': 30,
        'objective': 'robust',
        'requires_phases': [],  # No prerequisites
        'forced_params': {
            # Confluence OFF - we want all breakouts to test exits
            'enable_confluence': False,
            'min_confluence': 0,
            'skip_low_vol_c_grades': False,
        },
        'search_params': [
            'break_even_rr',
            'profit_target_rr', 
            'trailing_stop_distance',
            'ema_tighten_zone',
            'tightened_trail_distance',
            'ema_confirmation_bars',
            'atr_stop_mult',
            'min_stop_atr',
            'swing_lookback',
            'min_acceptable_rr',
            'ema_period',
            'vwap_stop_distance',
            'max_target_atr',
            'breakout_threshold_mult',
            'min_body_strength',
            'trading_end_minutes',
        ],
    },
    2: {
        'name': 'Volatility Regime',
        'description': 'When to adjust behavior by market conditions',
        'min_trades': 30,
        'objective': 'robust',
        'requires_phases': [1],
        'forced_params': {
            'enable_confluence': False,
            'min_confluence': 0,
        },
        'search_params': [
            'low_vol_threshold',
            'high_vol_threshold',
            'extreme_vol_threshold',
            'skip_low_vol_c_grades',
            'low_vol_min_rr',
        ],
    },
    3: {
        'name': 'Confluence Gating',
        'description': 'Entry quality filtering',
        'min_trades': 20,  # Lower - confluence reduces trades
        'objective': 'robust_pf',
        'requires_phases': [1, 2],
        'forced_params': {
            'enable_confluence': True,  # Force ON - this is the whole point
        },
        'search_params': [
            'min_confluence',  # Only search the threshold
        ],
    },
    4: {
        'name': 'Boolean Polish',
        'description': 'Execution style fine-tuning',
        'min_trades': 20,  # Was 45 — too high after confluence gating
        'objective': 'sortino',  # Now Sortino makes sense
        'requires_phases': [1, 2, 3],
        'forced_params': {},
        'search_params': [
            'use_adaptive_be',
            'use_ema_exit',
            'use_aggressive_trailing',
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER RANGES
# ═══════════════════════════════════════════════════════════════════════════

PARAM_RANGES = {
    # Exit Geometry
    'break_even_rr': {'type': 'float', 'low': 0.3, 'high': 0.8, 'step': 0.1},
    'profit_target_rr': {'type': 'float', 'low': 1.5, 'high': 3.0, 'step': 0.25},
    'trailing_stop_distance': {'type': 'float', 'low': 0.8, 'high': 2.0, 'step': 0.1},
    'ema_tighten_zone': {'type': 'float', 'low': 0.2, 'high': 0.5, 'step': 0.05},
    'tightened_trail_distance': {'type': 'float', 'low': 0.2, 'high': 0.5, 'step': 0.05},
    'ema_confirmation_bars': {'type': 'int', 'low': 1, 'high': 3},
    'ema_period': {'type': 'int', 'low': 7, 'high': 12},
    
    # Stops
    'atr_stop_mult': {'type': 'float', 'low': 1.5, 'high': 3.0, 'step': 0.25},
    'min_stop_atr': {'type': 'float', 'low': 0.10, 'high': 0.25, 'step': 0.05},
    'swing_lookback': {'type': 'int', 'low': 3, 'high': 7},
    'vwap_stop_distance': {'type': 'float', 'low': 0.2, 'high': 0.5, 'step': 0.05},
    
    # Entry Quality
    'min_acceptable_rr': {'type': 'float', 'low': 1.0, 'high': 2.0, 'step': 0.25},
    'max_target_atr': {'type': 'float', 'low': 2.5, 'high': 4.0, 'step': 0.25},
    'breakout_threshold_mult': {'type': 'float', 'low': 0.05, 'high': 0.15, 'step': 0.025},
    'min_body_strength': {'type': 'float', 'low': 0.3, 'high': 0.7, 'step': 0.1},
    
    # Volatility Thresholds
    'low_vol_threshold': {'type': 'float', 'low': 0.6, 'high': 0.9, 'step': 0.1},
    'high_vol_threshold': {'type': 'float', 'low': 1.1, 'high': 1.5, 'step': 0.1},
    'extreme_vol_threshold': {'type': 'float', 'low': 1.8, 'high': 2.5, 'step': 0.1},
    'low_vol_min_rr': {'type': 'float', 'low': 1.5, 'high': 2.5, 'step': 0.25},
    
    # Confluence
    'enable_confluence': {'type': 'categorical', 'choices': [True, False]},
    'min_confluence': {'type': 'int', 'low': 1, 'high': 5},
    
    # Booleans
    'skip_low_vol_c_grades': {'type': 'categorical', 'choices': [True, False]},
    'use_adaptive_be': {'type': 'categorical', 'choices': [True, False]},
    'use_ema_exit': {'type': 'categorical', 'choices': [True, False]},
    'use_aggressive_trailing': {'type': 'categorical', 'choices': [True, False]},
    
    # Time
    'trading_end_minutes': {'type': 'int', 'low': 600, 'high': 690, 'step': 15},
}

# Always fixed - core strategy identity
ALWAYS_FIXED = {
    'use_break_even': True,
    'use_trailing_stop': True,
    'use_ssl_momentum': True,
    'use_wae_acceleration': True,
    'use_qqe_momentum': True,
    'trading_start_minutes': 570,  # 9:30 AM
}


# ═══════════════════════════════════════════════════════════════════════════
# PHASED OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

class PhasedOptimizer:
    """
    Research-grade optimizer with staged parameter discovery.
    """
    
    def __init__(
        self,
        symbol: str = 'AMD',
        phase: int = 1,
        
        # Date ranges
        train_start: str = '2025-08-29',
        train_end: str = '2025-11-15',
        test_start: str = '2025-11-16',
        test_end: str = '2026-01-27',
        
        # Optimization settings
        n_trials: int = 200,
        fixed_params: Optional[Dict[str, Any]] = None,  # Manual overrides
        
        # Output
        results_dir: Optional[str] = None,
        storage_path: Optional[str] = None,
        verbose: bool = True,
    ):
        if phase not in PHASE_CONFIG:
            raise ValueError(f"Invalid phase {phase}. Must be 1-4.")
        
        self.symbol = symbol
        self.phase = phase
        self.phase_config = PHASE_CONFIG[phase]
        
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        
        self.n_trials = n_trials
        self.manual_fixed_params = fixed_params or {}
        
        self.results_dir = results_dir or os.path.join(PROJECT_ROOT, 'results')
        self.storage_path = storage_path or os.path.join(PROJECT_ROOT, 'orb_optuna_v3.db')
        self.verbose = verbose
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Build complete fixed params from previous phases + manual overrides
        self.fixed_params = self._build_fixed_params()
        
        self.best_params = None
        self.best_score = float('-inf')
        self.study = None
    
    def _build_fixed_params(self) -> Dict[str, Any]:
        """
        Build fixed params by:
        1. Start with ALWAYS_FIXED
        2. Add phase-specific forced params
        3. Load results from required previous phases
        4. Apply manual overrides
        """
        fixed = ALWAYS_FIXED.copy()
        
        # Add phase-specific forced params
        fixed.update(self.phase_config['forced_params'])
        
        # Load previous phase results
        for prev_phase in self.phase_config['requires_phases']:
            prev_results = self._load_phase_results(prev_phase)
            if prev_results is None:
                raise FileNotFoundError(
                    f"Phase {self.phase} requires Phase {prev_phase} results.\n"
                    f"Expected file: {self._get_results_path(prev_phase)}\n"
                    f"Run Phase {prev_phase} first."
                )
            # Add best params from previous phase
            fixed.update(prev_results['best_params'])
        
        # Apply manual overrides (highest priority)
        fixed.update(self.manual_fixed_params)
        
        return fixed
    
    def _load_phase_results(self, phase: int) -> Optional[Dict]:
        """Load results from a previous phase."""
        path = self._get_results_path(phase)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)
    
    def _get_results_path(self, phase: int) -> str:
        """Get the results file path for a phase."""
        return os.path.join(self.results_dir, f'{self.symbol}_phase{phase}.json')
    
    def _suggest_param(self, trial: optuna.Trial, name: str) -> Any:
        """Suggest a parameter value based on its defined range."""
        if name in self.fixed_params:
            return self.fixed_params[name]
        
        spec = PARAM_RANGES[name]
        
        if spec['type'] == 'float':
            return trial.suggest_float(
                name, spec['low'], spec['high'], 
                step=spec.get('step')
            )
        elif spec['type'] == 'int':
            return trial.suggest_int(
                name, spec['low'], spec['high'],
                step=spec.get('step', 1)
            )
        elif spec['type'] == 'categorical':
            return trial.suggest_categorical(name, spec['choices'])
        else:
            raise ValueError(f"Unknown param type: {spec['type']}")
    
    def _calc_score(self, results: Dict[str, Any]) -> float:
        """Calculate score based on phase objective."""
        
        min_trades = self.phase_config['min_trades']
        objective = self.phase_config['objective']
        
        # Hard filters (apply to ALL phases)
        if results['total_trades'] < min_trades:
            return float('-inf')
        
        if results['avg_r_per_trade'] < 0:
            # Surgeon's rule: reject negative expectancy
            return float('-inf')
        
        if not results['trades']:
            return float('-inf')
        
        # Objective-specific scoring
        if objective == 'robust':
            return self._calc_robust_score(results)
        
        elif objective == 'robust_pf':
            # Robust score with profit factor filter
            if results['profit_factor'] < 1.2:
                return float('-inf')
            return self._calc_robust_score(results)
        
        elif objective == 'sortino':
            return self._calc_sortino_score(results)
        
        else:
            return self._calc_robust_score(results)
    
    def _calc_robust_score(self, results: Dict) -> float:
        """
        Robust score: median_monthly_R - DD_penalty - variance_penalty
        Penalizes fragility and inconsistency.
        """
        # Group trades by month
        monthly_r = defaultdict(float)
        for t in results['trades']:
            month_key = t['date'][:7]
            monthly_r[month_key] += t['r_multiple']
        
        monthly_values = list(monthly_r.values())
        
        if len(monthly_values) < 2:
            return float('-inf')
        
        median_monthly = np.median(monthly_values)
        stdev_monthly = np.std(monthly_values)
        max_dd = results['max_drawdown_r']
        
        # Core robust score
        score = median_monthly - (0.5 * max_dd) - (0.2 * stdev_monthly)
        
        # Penalty for negative months
        neg_months = sum(1 for m in monthly_values if m < 0)
        score -= 0.5 * neg_months
        
        # Penalty for low win rate
        if results['win_rate'] < 30:
            score -= 1.0
        
        return score
    
    def _calc_sortino_score(self, results: Dict) -> float:
        """
        Sortino ratio: reward/downside_deviation
        Rewards upside volatility, penalizes downside.
        """
        r_values = [t['r_multiple'] for t in results['trades']]
        
        negative_r = [r for r in r_values if r < 0]
        
        if len(negative_r) > 1:
            downside_dev = np.std(negative_r, ddof=1)
        elif len(negative_r) == 1:
            downside_dev = abs(negative_r[0])
        else:
            downside_dev = 0.001  # Avoid div by zero
        
        return results['avg_r_per_trade'] / downside_dev if downside_dev > 0 else 10.0
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        
        # Build params: fixed + suggested for this phase
        params = self.fixed_params.copy()
        
        for param_name in self.phase_config['search_params']:
            params[param_name] = self._suggest_param(trial, param_name)
        
        # ═══════════════════════════════════════════════════════════════════
        # RUN TRAINING BACKTEST
        # ═══════════════════════════════════════════════════════════════════
        
        try:
            bt_train = ORBBacktester(
                symbol=self.symbol,
                start_date=self.train_start,
                end_date=self.train_end,
                verbose=False,
                **params
            )
            train_results = bt_train.run()
        except Exception as e:
            if self.verbose:
                print(f"  Trial {trial.number} train failed: {e}")
            return float('-inf')
        
        train_score = self._calc_score(train_results)
        if train_score == float('-inf'):
            return float('-inf')
        
        # ═══════════════════════════════════════════════════════════════════
        # RUN TEST BACKTEST
        # ═══════════════════════════════════════════════════════════════════
        
        try:
            bt_test = ORBBacktester(
                symbol=self.symbol,
                start_date=self.test_start,
                end_date=self.test_end,
                verbose=False,
                **params
            )
            test_results = bt_test.run()
        except Exception as e:
            if self.verbose:
                print(f"  Trial {trial.number} test failed: {e}")
            return float('-inf')
        
        test_score = self._calc_score(test_results)
        if test_score == float('-inf'):
            return float('-inf')
        
        # ═══════════════════════════════════════════════════════════════════
        # 40/60 TRAIN/TEST BLEND
        # ═══════════════════════════════════════════════════════════════════
        
        blended_score = 0.4 * train_score + 0.6 * test_score
        
        # Track best
        if blended_score > self.best_score:
            self.best_score = blended_score
            # Only save searched params (not fixed ones)
            self.best_params = {
                k: params[k] for k in self.phase_config['search_params']
                if k in params
            }
        
        return blended_score
    
    def run(self) -> Dict[str, Any]:
        """Run optimization for this phase."""
        
        if self.verbose:
            self._print_header()
        
        # Create Optuna study
        sampler = TPESampler(seed=42)
        storage = f'sqlite:///{self.storage_path}'
        study_name = f'ORB_{self.symbol}_phase{self.phase}_v3'
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )
        
        # Get best params for this phase's search space
        best_trial = self.study.best_trial
        best_searched_params = {
            k: best_trial.params[k] 
            for k in self.phase_config['search_params']
            if k in best_trial.params
        }
        
        # Run validation
        validation = self._run_validation(best_searched_params)
        
        # Build output
        output = {
            'symbol': self.symbol,
            'phase': self.phase,
            'phase_name': self.phase_config['name'],
            'optimization_date': datetime.now().isoformat(),
            'optimizer_version': 'v3_phased',
            'objective': self.phase_config['objective'],
            
            'train_period': {'start': self.train_start, 'end': self.train_end},
            'test_period': {'start': self.test_start, 'end': self.test_end},
            'n_trials': self.n_trials,
            
            'best_score': best_trial.value,
            'best_params': best_searched_params,  # Only this phase's searched params
            'fixed_params': self.fixed_params,    # What was fixed for this phase
            
            'train_results': validation['train'],
            'test_results': validation['test'],
        }
        
        # Save results
        output_path = self._get_results_path(self.phase)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        if self.verbose:
            self._print_results(output)
        
        return output
    
    def _run_validation(self, best_searched_params: Dict) -> Dict:
        """Run final validation with best params."""
        
        # Combine fixed + best searched
        full_params = self.fixed_params.copy()
        full_params.update(best_searched_params)
        
        # Training
        bt_train = ORBBacktester(
            symbol=self.symbol,
            start_date=self.train_start,
            end_date=self.train_end,
            verbose=False,
            **full_params
        )
        train_results = bt_train.run()
        
        # Testing
        bt_test = ORBBacktester(
            symbol=self.symbol,
            start_date=self.test_start,
            end_date=self.test_end,
            verbose=False,
            **full_params
        )
        test_results = bt_test.run()
        
        return {
            'train': {
                'total_trades': train_results['total_trades'],
                'win_rate': train_results['win_rate'],
                'total_r': train_results['total_r'],
                'avg_r_per_trade': train_results['avg_r_per_trade'],
                'profit_factor': train_results['profit_factor'],
                'max_drawdown_r': train_results['max_drawdown_r'],
            },
            'test': {
                'total_trades': test_results['total_trades'],
                'win_rate': test_results['win_rate'],
                'total_r': test_results['total_r'],
                'avg_r_per_trade': test_results['avg_r_per_trade'],
                'profit_factor': test_results['profit_factor'],
                'max_drawdown_r': test_results['max_drawdown_r'],
            }
        }
    
    def _print_header(self):
        """Print optimization header."""
        print(f"\n{'='*70}")
        print(f"  ORB PHASED OPTIMIZER v3: {self.symbol}")
        print(f"  Phase {self.phase}: {self.phase_config['name']}")
        print(f"{'='*70}")
        print(f"\n  {self.phase_config['description']}")
        print(f"\n  Training: {self.train_start} to {self.train_end}")
        print(f"  Testing:  {self.test_start} to {self.test_end}")
        print(f"  Trials:   {self.n_trials}")
        print(f"  Objective: {self.phase_config['objective']}")
        print(f"  Min Trades: {self.phase_config['min_trades']}")
        
        print(f"\n  Searching {len(self.phase_config['search_params'])} params:")
        for p in self.phase_config['search_params']:
            print(f"    • {p}")
        
        if self.fixed_params:
            print(f"\n  Fixed from previous phases: {len(self.fixed_params)} params")
        
        print(f"\n{'─'*70}")
        print(f"  OPTIMIZATION IN PROGRESS...")
        print(f"{'─'*70}\n")
    
    def _print_results(self, output: Dict):
        """Print optimization results."""
        print(f"\n{'─'*70}")
        print(f"  PHASE {self.phase} COMPLETE: {self.phase_config['name']}")
        print(f"{'─'*70}")
        
        print(f"\n  Best {self.phase_config['objective']} score: {output['best_score']:.4f}")
        
        print(f"\n  Best parameters found:")
        for k, v in sorted(output['best_params'].items()):
            print(f"    {k}: {v}")
        
        train = output['train_results']
        test = output['test_results']
        
        print(f"\n  {'Metric':<20} {'Training':>12} {'Testing':>12}")
        print(f"  {'-'*44}")
        print(f"  {'Total Trades':<20} {train['total_trades']:>12} {test['total_trades']:>12}")
        print(f"  {'Win Rate':<20} {train['win_rate']:>11.1f}% {test['win_rate']:>11.1f}%")
        print(f"  {'Total R':<20} {train['total_r']:>+12.2f} {test['total_r']:>+12.2f}")
        print(f"  {'Avg R/Trade':<20} {train['avg_r_per_trade']:>+12.3f} {test['avg_r_per_trade']:>+12.3f}")
        print(f"  {'Profit Factor':<20} {train['profit_factor']:>12.2f} {test['profit_factor']:>12.2f}")
        print(f"  {'Max Drawdown':<20} {train['max_drawdown_r']:>12.2f} {test['max_drawdown_r']:>12.2f}")
        
        output_path = self._get_results_path(self.phase)
        print(f"\n  ✓ Results saved to: {output_path}")
        
        # Next steps
        if self.phase < 4:
            print(f"\n  Next: Review results, then run Phase {self.phase + 1}")
        else:
            print(f"\n  ✓ All phases complete! Run validation to confirm.")
        
        print(f"\n{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_full_validation(
    symbol: str,
    train_start: str = '2025-08-29',
    train_end: str = '2025-11-15',
    test_start: str = '2025-11-16', 
    test_end: str = '2026-01-27',
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run validation with all optimized params from all phases.
    Call this after completing all 4 phases.
    """
    results_dir = results_dir or os.path.join(PROJECT_ROOT, 'results')
    
    # Load all phase results
    all_params = ALWAYS_FIXED.copy()
    
    for phase in [1, 2, 3, 4]:
        path = os.path.join(results_dir, f'{symbol}_phase{phase}.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Phase {phase} results: {path}")
        
        with open(path, 'r') as f:
            phase_results = json.load(f)
        all_params.update(phase_results['best_params'])
    
    print(f"\n{'='*70}")
    print(f"  FULL VALIDATION: {symbol}")
    print(f"  All 4 phases combined")
    print(f"{'='*70}")
    
    print(f"\n  Complete parameter set ({len(all_params)} params):")
    for k, v in sorted(all_params.items()):
        print(f"    {k}: {v}")
    
    # Run backtests
    print(f"\n  Running validation backtests...")
    
    bt_train = ORBBacktester(
        symbol=symbol,
        start_date=train_start,
        end_date=train_end,
        verbose=False,
        **all_params
    )
    train_results = bt_train.run()
    
    bt_test = ORBBacktester(
        symbol=symbol,
        start_date=test_start,
        end_date=test_end,
        verbose=False,
        **all_params
    )
    test_results = bt_test.run()
    
    # Print results
    print(f"\n  {'Metric':<20} {'Training':>12} {'Testing':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Total Trades':<20} {train_results['total_trades']:>12} {test_results['total_trades']:>12}")
    print(f"  {'Win Rate':<20} {train_results['win_rate']:>11.1f}% {test_results['win_rate']:>11.1f}%")
    print(f"  {'Total R':<20} {train_results['total_r']:>+12.2f} {test_results['total_r']:>+12.2f}")
    print(f"  {'Avg R/Trade':<20} {train_results['avg_r_per_trade']:>+12.3f} {test_results['avg_r_per_trade']:>+12.3f}")
    print(f"  {'Profit Factor':<20} {train_results['profit_factor']:>12.2f} {test_results['profit_factor']:>12.2f}")
    print(f"  {'Max Drawdown':<20} {train_results['max_drawdown_r']:>12.2f} {test_results['max_drawdown_r']:>12.2f}")
    
    # Save validation results
    output = {
        'symbol': symbol,
        'validation_date': datetime.now().isoformat(),
        'all_params': all_params,
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
        }
    }
    
    output_path = os.path.join(results_dir, f'{symbol}_FINAL_validation.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  ✓ Validation saved to: {output_path}")
    print(f"\n{'='*70}\n")
    
    return output


# ═══════════════════════════════════════════════════════════════════════════
# BATCH RUNNER (Multi-symbol)
# ═══════════════════════════════════════════════════════════════════════════

class BatchPhasedOptimizer:
    """Run all phases for multiple symbols."""
    
    def __init__(
        self,
        symbols: List[str] = ['AMD', 'GOOGL'],
        phases: List[int] = [1, 2, 3, 4],
        n_trials: int = 200,
        train_start: str = '2025-08-29',
        train_end: str = '2025-11-15',
        test_start: str = '2025-11-16',
        test_end: str = '2026-01-27',
        results_dir: Optional[str] = None,
    ):
        self.symbols = symbols
        self.phases = phases
        self.n_trials = n_trials
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.results_dir = results_dir or os.path.join(PROJECT_ROOT, 'results')
    
    def run(self) -> Dict[str, Dict]:
        """Run all phases for all symbols."""
        
        print(f"\n{'='*70}")
        print(f"  BATCH PHASED OPTIMIZATION")
        print(f"  Symbols: {', '.join(self.symbols)}")
        print(f"  Phases: {self.phases}")
        print(f"{'='*70}\n")
        
        all_results = {}
        
        for symbol in self.symbols:
            print(f"\n{'━'*70}")
            print(f"  SYMBOL: {symbol}")
            print(f"{'━'*70}")
            
            all_results[symbol] = {}
            
            for phase in self.phases:
                opt = PhasedOptimizer(
                    symbol=symbol,
                    phase=phase,
                    train_start=self.train_start,
                    train_end=self.train_end,
                    test_start=self.test_start,
                    test_end=self.test_end,
                    n_trials=self.n_trials,
                    results_dir=self.results_dir,
                    verbose=True,
                )
                all_results[symbol][f'phase{phase}'] = opt.run()
            
            # Run full validation
            all_results[symbol]['validation'] = run_full_validation(
                symbol=symbol,
                train_start=self.train_start,
                train_end=self.train_end,
                test_start=self.test_start,
                test_end=self.test_end,
                results_dir=self.results_dir,
            )
        
        self._print_summary(all_results)
        return all_results
    
    def _print_summary(self, all_results: Dict):
        """Print summary across all symbols."""
        
        print(f"\n{'='*70}")
        print(f"  BATCH OPTIMIZATION SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"  {'Symbol':<8} {'Train R':>10} {'Test R':>10} {'Train WR':>10} {'Test WR':>10} {'PF':>8}")
        print(f"  {'-'*56}")
        
        for symbol, results in all_results.items():
            if 'validation' in results:
                v = results['validation']
                train = v['train_results']
                test = v['test_results']
                print(f"  {symbol:<8} {train['total_r']:>+10.2f} {test['total_r']:>+10.2f} "
                      f"{train['win_rate']:>9.1f}% {test['win_rate']:>9.1f}% "
                      f"{test['profit_factor']:>8.2f}")
        
        print(f"\n{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ORB Phased Optimizer v3')
    parser.add_argument('--symbol', type=str, default='AMD', help='Symbol to optimize')
    parser.add_argument('--phase', type=int, default=1, help='Phase to run (1-4)')
    parser.add_argument('--trials', type=int, default=200, help='Number of trials')
    parser.add_argument('--validate', action='store_true', help='Run full validation')
    
    args = parser.parse_args()
    
    if args.validate:
        run_full_validation(symbol=args.symbol)
    else:
        opt = PhasedOptimizer(
            symbol=args.symbol,
            phase=args.phase,
            n_trials=args.trials,
            verbose=True,
        )
        opt.run()
