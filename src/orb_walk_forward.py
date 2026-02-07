"""
ORB Walk-Forward Analysis
=========================
Post-validation robustness testing for optimized ORB parameters.
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, Optional, List
from collections import defaultdict
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orb_backtester import ORBBacktester

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WalkForwardAnalysis:
    """
    Rolling-window walk-forward analysis.
    Tests locked params across sequential train/test folds.
    """

    def __init__(
        self,
        symbol='AMD',
        data_start='2025-08-29',
        data_end='2026-01-27',
        train_months=2,
        test_months=1,
        step_months=1,
        anchored=False,
        params_file=None,
        results_dir=None,
        verbose=True,
    ):
        self.symbol = symbol
        self.data_start = datetime.strptime(data_start, '%Y-%m-%d')
        self.data_end = datetime.strptime(data_end, '%Y-%m-%d')
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.anchored = anchored
        self.results_dir = results_dir or os.path.join(PROJECT_ROOT, 'results')
        self.verbose = verbose
        self.params = self._load_params(params_file)
        self.folds = self._generate_folds()

    def _load_params(self, params_file=None):
        if params_file is None:
            params_file = os.path.join(
                self.results_dir, f'{self.symbol}_FINAL_validation.json'
            )
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Validation file not found: {params_file}")
        with open(params_file, 'r') as f:
            data = json.load(f)
        return data['all_params']

    # Find and replace the _generate_folds method in orb_walk_forward.py

    def _generate_folds(self):
        folds = []
        fold_num = 1
        cursor = self.data_start

        while True:
            if self.anchored:
                fold_train_start = self.data_start
                fold_train_end = cursor + relativedelta(months=self.train_months) - timedelta(days=1)
            else:
                fold_train_start = cursor
                fold_train_end = cursor + relativedelta(months=self.train_months) - timedelta(days=1)

            fold_test_start = fold_train_end + timedelta(days=1)
            fold_test_end = fold_test_start + relativedelta(months=self.test_months) - timedelta(days=1)

            if fold_test_start >= self.data_end:
                break
            if fold_test_end > self.data_end:
                fold_test_end = self.data_end

            folds.append({
                'fold': fold_num,
                'train_start': fold_train_start.strftime('%Y-%m-%d'),
                'train_end': fold_train_end.strftime('%Y-%m-%d'),
                'test_start': fold_test_start.strftime('%Y-%m-%d'),
                'test_end': fold_test_end.strftime('%Y-%m-%d'),
            })

            fold_num += 1
            cursor += relativedelta(months=self.step_months)

        return folds

    def run_fixed(self):
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  WALK-FORWARD ANALYSIS: {self.symbol}")
            print(f"  Mode: {'Anchored' if self.anchored else 'Rolling'} | Folds: {len(self.folds)}")
            print(f"{'='*70}")

        fold_results = []

        for fold in self.folds:
            if self.verbose:
                print(f"\n  Fold {fold['fold']}: Train {fold['train_start']} to {fold['train_end']}")
                print(f"          Test  {fold['test_start']} to {fold['test_end']}")

            try:
                bt_train = ORBBacktester(
                    symbol=self.symbol,
                    start_date=fold['train_start'],
                    end_date=fold['train_end'],
                    verbose=False,
                    **self.params
                )
                train_results = bt_train.run()
            except Exception as e:
                print(f"    Train failed: {e}")
                continue

            try:
                bt_test = ORBBacktester(
                    symbol=self.symbol,
                    start_date=fold['test_start'],
                    end_date=fold['test_end'],
                    verbose=False,
                    **self.params
                )
                test_results = bt_test.run()
            except Exception as e:
                print(f"    Test failed: {e}")
                continue

            fold_data = {
                'fold': fold['fold'],
                'windows': fold,
                'train': {
                    'total_trades': train_results['total_trades'],
                    'win_rate': train_results['win_rate'],
                    'total_r': float(train_results['total_r']),
                    'avg_r_per_trade': float(train_results['avg_r_per_trade']),
                    'profit_factor': float(train_results['profit_factor']),
                    'max_drawdown_r': float(train_results['max_drawdown_r']),
                },
                'test': {
                    'total_trades': test_results['total_trades'],
                    'win_rate': test_results['win_rate'],
                    'total_r': float(test_results['total_r']),
                    'avg_r_per_trade': float(test_results['avg_r_per_trade']),
                    'profit_factor': float(test_results['profit_factor']),
                    'max_drawdown_r': float(test_results['max_drawdown_r']),
                }
            }
            fold_results.append(fold_data)

            if self.verbose:
                t = fold_data['test']
                print(f"    OOS: {t['total_trades']} trades | "
                      f"WR {t['win_rate']:.1f}% | "
                      f"R {t['total_r']:+.2f} | "
                      f"PF {t['profit_factor']:.2f} | "
                      f"DD {t['max_drawdown_r']:.2f}")

        summary = self._compute_summary(fold_results)

        output = {
            'symbol': self.symbol,
            'analysis_date': datetime.now().isoformat(),
            'mode': 'anchored' if self.anchored else 'rolling',
            'n_folds': len(fold_results),
            'params_used': self.params,
            'folds': fold_results,
            'summary': summary,
        }

        output_path = os.path.join(self.results_dir, f'{self.symbol}_walk_forward.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        if self.verbose:
            self._print_summary(fold_results, summary)
            print(f"\n  Saved to: {output_path}")

        return output

    def _compute_summary(self, fold_results):
        if not fold_results:
            return {}

        oos_total_r = [f['test']['total_r'] for f in fold_results]
        oos_avg_r = [f['test']['avg_r_per_trade'] for f in fold_results]
        oos_wr = [f['test']['win_rate'] for f in fold_results]
        oos_pf = [f['test']['profit_factor'] for f in fold_results]
        oos_dd = [f['test']['max_drawdown_r'] for f in fold_results]
        oos_trades = [f['test']['total_trades'] for f in fold_results]

        profitable_folds = sum(1 for r in oos_total_r if r > 0)

        degradation_ratios = []
        for f in fold_results:
            if f['train']['total_r'] > 0:
                degradation_ratios.append(f['test']['total_r'] / f['train']['total_r'])

        pct = 100 * profitable_folds / len(fold_results)
        avg_r_mean = float(np.mean(oos_avg_r))
        pf_mean = float(np.mean(oos_pf))

        if pct >= 80 and avg_r_mean > 0.2 and pf_mean > 1.5:
            verdict = "STRONG PASS - System is robust across time periods"
        elif pct >= 60 and avg_r_mean > 0.1 and pf_mean > 1.2:
            verdict = "PASS - System shows consistent edge, some variance"
        elif pct >= 50 and avg_r_mean > 0:
            verdict = "MARGINAL - Edge exists but inconsistent"
        else:
            verdict = "FAIL - System does not generalize, likely overfit"

        summary = {
            'oos_total_r_sum': float(sum(oos_total_r)),
            'oos_total_r_mean': float(np.mean(oos_total_r)),
            'oos_total_r_std': float(np.std(oos_total_r)),
            'oos_avg_r_mean': avg_r_mean,
            'oos_win_rate_mean': float(np.mean(oos_wr)),
            'oos_pf_mean': pf_mean,
            'oos_worst_dd': float(max(oos_dd)),
            'oos_total_trades': sum(oos_trades),
            'profitable_folds': profitable_folds,
            'total_folds': len(fold_results),
            'profitable_pct': pct,
            'degradation_ratio': float(np.mean(degradation_ratios)) if degradation_ratios else None,
            'verdict': verdict,
        }
        return summary

    def _print_summary(self, fold_results, summary):
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD SUMMARY: {self.symbol}")
        print(f"{'='*70}")

        print(f"\n  {'Fold':<6} {'Period':<24} {'Trades':>7} {'WR':>7} "
              f"{'Total R':>9} {'PF':>7} {'DD':>7}")
        print(f"  {'-'*68}")

        for f in fold_results:
            t = f['test']
            period = f"{f['windows']['test_start']} to {f['windows']['test_end']}"
            print(f"  {f['fold']:<6} {period:<24} {t['total_trades']:>7} "
                  f"{t['win_rate']:>6.1f}% {t['total_r']:>+9.2f} "
                  f"{t['profit_factor']:>7.2f} {t['max_drawdown_r']:>7.2f}")

        s = summary
        print(f"\n  AGGREGATE OOS:")
        print(f"    Total Trades:       {s['oos_total_trades']}")
        print(f"    Combined R:         {s['oos_total_r_sum']:+.2f}")
        print(f"    Mean R/Fold:        {s['oos_total_r_mean']:+.2f} (std {s['oos_total_r_std']:.2f})")
        print(f"    Mean Avg R/Trade:   {s['oos_avg_r_mean']:+.3f}")
        print(f"    Mean Win Rate:      {s['oos_win_rate_mean']:.1f}%")
        print(f"    Mean PF:            {s['oos_pf_mean']:.2f}")
        print(f"    Worst DD:           {s['oos_worst_dd']:.2f}R")
        print(f"    Profitable Folds:   {s['profitable_folds']}/{s['total_folds']} ({s['profitable_pct']:.0f}%)")
        if s['degradation_ratio'] is not None:
            print(f"    Degradation Ratio:  {s['degradation_ratio']:.2f}x")
        print(f"\n  VERDICT: {s['verdict']}")
        print(f"\n{'='*70}")


class MonteCarloAnalysis:
    """
    Monte Carlo stress test using trade-level resampling.
    """

    def __init__(
        self,
        symbol='AMD',
        start_date='2025-08-29',
        end_date='2026-01-27',
        params_file=None,
        results_dir=None,
        verbose=True,
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.results_dir = results_dir or os.path.join(PROJECT_ROOT, 'results')
        self.verbose = verbose
        self.params = self._load_params(params_file)
        self.trades = self._get_trades()

    def _load_params(self, params_file=None):
        if params_file is None:
            params_file = os.path.join(
                self.results_dir, f'{self.symbol}_FINAL_validation.json'
            )
        with open(params_file, 'r') as f:
            data = json.load(f)
        return data['all_params']

    def _get_trades(self):
        bt = ORBBacktester(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            verbose=False,
            **self.params
        )
        results = bt.run()
        return [t['r_multiple'] for t in results['trades']]

    def run(self, n_simulations=5000, n_trades=None, seed=42):
        if not self.trades:
            raise ValueError("No trades to simulate.")

        rng = np.random.default_rng(seed)
        trade_array = np.array(self.trades)
        sim_length = n_trades or len(trade_array)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  MONTE CARLO ANALYSIS: {self.symbol}")
            print(f"{'='*70}")
            print(f"  Source: {len(trade_array)} trades | "
                  f"Avg R: {np.mean(trade_array):+.3f} | "
                  f"WR: {100*np.sum(trade_array>0)/len(trade_array):.1f}%")
            print(f"  Running {n_simulations:,} simulations of {sim_length} trades...")

        sim_trades = rng.choice(trade_array, size=(n_simulations, sim_length), replace=True)
        equity_curves = np.cumsum(sim_trades, axis=1)
        final_equity = equity_curves[:, -1]

        running_max = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = running_max - equity_curves
        max_drawdowns = np.max(drawdowns, axis=1)

        results = {
            'equity': {
                'mean': float(np.mean(final_equity)),
                'median': float(np.median(final_equity)),
                'std': float(np.std(final_equity)),
                'p5': float(np.percentile(final_equity, 5)),
                'p10': float(np.percentile(final_equity, 10)),
                'p25': float(np.percentile(final_equity, 25)),
                'p75': float(np.percentile(final_equity, 75)),
                'p90': float(np.percentile(final_equity, 90)),
                'p95': float(np.percentile(final_equity, 95)),
                'worst': float(np.min(final_equity)),
                'best': float(np.max(final_equity)),
            },
            'drawdown': {
                'mean': float(np.mean(max_drawdowns)),
                'median': float(np.median(max_drawdowns)),
                'p75': float(np.percentile(max_drawdowns, 75)),
                'p90': float(np.percentile(max_drawdowns, 90)),
                'p95': float(np.percentile(max_drawdowns, 95)),
                'p99': float(np.percentile(max_drawdowns, 99)),
                'worst': float(np.max(max_drawdowns)),
            },
            'risk': {
                'prob_losing': float(100 * np.mean(final_equity < 0)),
                'prob_dd_5r': float(100 * np.mean(max_drawdowns > 5)),
                'prob_dd_8r': float(100 * np.mean(max_drawdowns > 8)),
                'prob_dd_10r': float(100 * np.mean(max_drawdowns > 10)),
                'prob_dd_15r': float(100 * np.mean(max_drawdowns > 15)),
            },
        }

        output = {
            'symbol': self.symbol,
            'analysis_date': datetime.now().isoformat(),
            'source_trades': len(trade_array),
            **results,
        }

        output_path = os.path.join(self.results_dir, f'{self.symbol}_monte_carlo.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        if self.verbose:
            self._print_results(results, sim_length)
            print(f"\n  Saved to: {output_path}")

        return output

    def _print_results(self, results, sim_length):
        eq = results['equity']
        dd = results['drawdown']
        risk = results['risk']

        print(f"\n  EQUITY DISTRIBUTION ({sim_length} trades)")
        print(f"  {'-'*50}")
        print(f"    Best:           {eq['best']:>+8.2f} R")
        print(f"    95th pct:       {eq['p95']:>+8.2f} R")
        print(f"    75th pct:       {eq['p75']:>+8.2f} R")
        print(f"    Median:         {eq['median']:>+8.2f} R")
        print(f"    Mean:           {eq['mean']:>+8.2f} R")
        print(f"    25th pct:       {eq['p25']:>+8.2f} R")
        print(f"    10th pct:       {eq['p10']:>+8.2f} R")
        print(f"    5th pct:        {eq['p5']:>+8.2f} R")
        print(f"    Worst:          {eq['worst']:>+8.2f} R")

        print(f"\n  MAX DRAWDOWN DISTRIBUTION")
        print(f"  {'-'*50}")
        print(f"    Median:         {dd['median']:>8.2f} R")
        print(f"    Mean:           {dd['mean']:>8.2f} R")
        print(f"    90th pct:       {dd['p90']:>8.2f} R")
        print(f"    95th pct:       {dd['p95']:>8.2f} R")
        print(f"    99th pct:       {dd['p99']:>8.2f} R")
        print(f"    Worst:          {dd['worst']:>8.2f} R")

        print(f"\n  RISK PROBABILITIES")
        print(f"  {'-'*50}")
        print(f"    P(losing money):  {risk['prob_losing']:>6.1f}%")
        print(f"    P(DD > 5R):       {risk['prob_dd_5r']:>6.1f}%")
        print(f"    P(DD > 8R):       {risk['prob_dd_8r']:>6.1f}%")
        print(f"    P(DD > 10R):      {risk['prob_dd_10r']:>6.1f}%")
        print(f"    P(DD > 15R):      {risk['prob_dd_15r']:>6.1f}%")
        print(f"\n{'='*70}")


def run_full_robustness_suite(symbol='AMD', results_dir=None):
    results_dir = results_dir or os.path.join(PROJECT_ROOT, 'results')
    all_results = {}

    print(f"\n{'='*70}")
    print(f"  FULL ROBUSTNESS SUITE: {symbol}")
    print(f"{'='*70}")

    print(f"\n  Stage 1: Rolling Walk-Forward")
    wfa_rolling = WalkForwardAnalysis(symbol=symbol, anchored=False, results_dir=results_dir)
    all_results['rolling'] = wfa_rolling.run_fixed()

    print(f"\n  Stage 2: Anchored Walk-Forward")
    wfa_anchored = WalkForwardAnalysis(symbol=symbol, anchored=True, results_dir=results_dir)
    all_results['anchored'] = wfa_anchored.run_fixed()

    print(f"\n  Stage 3: Monte Carlo")
    mc = MonteCarloAnalysis(symbol=symbol, results_dir=results_dir)
    all_results['monte_carlo'] = mc.run(n_simulations=10000)

    print(f"\n{'='*70}")
    print(f"  SUITE COMPLETE: {symbol}")
    print(f"  Rolling:  {all_results['rolling']['summary']['verdict']}")
    print(f"  Anchored: {all_results['anchored']['summary']['verdict']}")
    print(f"  MC P(loss): {all_results['monte_carlo']['risk']['prob_losing']:.1f}%")
    print(f"{'='*70}\n")

    return all_results
