"""
ORB Lab - Phased Optimizer UI
==============================
Streamlit interface for running phased ORB strategy optimizations,
validation, robustness testing, and cross-symbol comparison.

Launch with: streamlit run optimizer_app.py

Tabs:
  1. Optimize   - Run all 4 phases with review/approve between each
  2. Validate   - Run full validation + EMA exit A/B comparison
  3. Robustness - Walk-forward analysis + Monte Carlo simulation
  4. Compare    - Cross-symbol results comparison
"""

import streamlit as st
import sys
import os
import json
import time
import subprocess
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import optuna
from datetime import datetime, date

# Import our modules
from Orb_Optimizer_Final import (
    PhasedOptimizer, PHASE_CONFIG, ALWAYS_FIXED,
    run_full_validation, PROJECT_ROOT
)
from orb_backtester import ORBBacktester
from orb_walk_forward import WalkForwardAnalysis, MonteCarloAnalysis
from orb_settings_export import (
    SettingsExporter, generate_preset_block, write_preset_file,
    generate_master_config, generate_map_declarations,
    generate_resolution_lines, minutes_to_time,
    PRESET_FILE_PATH
)

# =====================================================================
# PAGE CONFIG
# =====================================================================

st.set_page_config(
    page_title="ORB Lab",
    page_icon="🔬",
    layout="wide"
)

# Preset symbols
PRESET_SYMBOLS = ['AMD', 'NVDA', 'TSLA', 'AAPL', 'GOOGL', 'PLTR', 'META', 'MSFT']
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================================
# HELPERS
# =====================================================================

def get_phase_status(symbol):
    """Check which phases have been completed for a symbol."""
    status = {}
    for phase in range(1, 5):
        path = os.path.join(RESULTS_DIR, f'{symbol}_phase{phase}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            status[phase] = {
                'complete': True,
                'date': data.get('optimization_date', 'unknown'),
                'score': data.get('best_score', 0),
                'params': data.get('best_params', {}),
                'train': data.get('train_results', {}),
                'test': data.get('test_results', {}),
            }
        else:
            status[phase] = {'complete': False}
    return status


def get_validation_status(symbol):
    """Check if full validation exists."""
    path = os.path.join(RESULTS_DIR, f'{symbol}_FINAL_validation.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def get_robustness_status(symbol):
    """Check if robustness results exist."""
    wf_path = os.path.join(RESULTS_DIR, f'{symbol}_walk_forward.json')
    mc_path = os.path.join(RESULTS_DIR, f'{symbol}_monte_carlo.json')
    results = {}
    if os.path.exists(wf_path):
        with open(wf_path, 'r') as f:
            results['walk_forward'] = json.load(f)
    if os.path.exists(mc_path):
        with open(mc_path, 'r') as f:
            results['monte_carlo'] = json.load(f)
    return results if results else None


def display_metrics_comparison(train, test, label_train="Training", label_test="Testing"):
    """Display train vs test metrics side by side."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{label_train}**")
        st.metric("Total Trades", train.get('total_trades', 0))
        st.metric("Win Rate", f"{train.get('win_rate', 0):.1f}%")
        st.metric("Total R", f"{train.get('total_r', 0):+.2f}")
        st.metric("Avg R/Trade", f"{train.get('avg_r_per_trade', 0):+.3f}")
        st.metric("Profit Factor", f"{train.get('profit_factor', 0):.2f}")
        st.metric("Max Drawdown", f"{train.get('max_drawdown_r', 0):.2f}R")

    with col2:
        st.markdown(f"**{label_test}**")
        st.metric("Total Trades", test.get('total_trades', 0))
        r_delta = test.get('total_r', 0) - train.get('total_r', 0)
        wr_delta = test.get('win_rate', 0) - train.get('win_rate', 0)
        st.metric("Win Rate", f"{test.get('win_rate', 0):.1f}%", delta=f"{wr_delta:+.1f}%")
        st.metric("Total R", f"{test.get('total_r', 0):+.2f}", delta=f"{r_delta:+.2f}")
        st.metric("Avg R/Trade", f"{test.get('avg_r_per_trade', 0):+.3f}")
        pf_delta = test.get('profit_factor', 0) - train.get('profit_factor', 0)
        st.metric("Profit Factor", f"{test.get('profit_factor', 0):.2f}", delta=f"{pf_delta:+.2f}")
        st.metric("Max Drawdown", f"{test.get('max_drawdown_r', 0):.2f}R")


def delete_optuna_study(symbol, phase):
    """Delete an Optuna study if it exists."""
    storage_path = os.path.join(PROJECT_ROOT, 'orb_optuna_v3.db')
    storage = f'sqlite:///{storage_path}'
    study_name = f'ORB_{symbol}_phase{phase}_v3'
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
        return True
    except KeyError:
        return False


# =====================================================================
# SIDEBAR
# =====================================================================

with st.sidebar:
    st.header("🔬 ORB Lab v2")
    st.divider()

    # Symbol selection
    symbol_option = st.selectbox(
        "Symbol",
        options=PRESET_SYMBOLS + ['Custom...'],
        index=0
    )
    if symbol_option == 'Custom...':
        symbol = st.text_input("Enter symbol", value="SPY").upper()
    else:
        symbol = symbol_option

    st.divider()

    # Date ranges
    st.subheader("Date Ranges")
    train_start = st.date_input("Train Start", value=date(2025, 2, 1))
    train_end = st.date_input("Train End", value=date(2025, 9, 30))
    test_start = st.date_input("Test Start", value=date(2025, 10, 1))
    test_end = st.date_input("Test End", value=date(2026, 1, 27))

    # Date validation
    date_error = None
    if train_start >= train_end:
        date_error = "Train start must be before train end"
    elif test_start >= test_end:
        date_error = "Test start must be before test end"
    elif train_end >= test_start:
        date_error = "Training must end before testing starts"
    if date_error:
        st.error(date_error)

    st.divider()

    # Status overview
    st.subheader("Phase Status")
    phase_status = get_phase_status(symbol)
    for phase in range(1, 5):
        config = PHASE_CONFIG[phase]
        if phase_status[phase]['complete']:
            st.success(f"Phase {phase}: {config['name']} ✅")
        else:
            st.warning(f"Phase {phase}: {config['name']} ⬜")

    validation = get_validation_status(symbol)
    if validation:
        st.success("Validation ✅")
    else:
        st.warning("Validation ⬜")

    st.divider()

    # Optuna dashboard
    st.subheader("Optuna Dashboard")
    db_path = os.path.join(PROJECT_ROOT, 'orb_optuna_v3.db')
    if os.path.exists(db_path):
        st.code(f"optuna-dashboard sqlite:///{db_path}", language="bash")
    else:
        st.caption("No database yet")


# =====================================================================
# MAIN CONTENT - TABS
# =====================================================================

st.title(f"🔬 ORB Lab: {symbol}")

tab_optimize, tab_validate, tab_robustness, tab_compare, tab_settings, tab_batch = st.tabs([
    "⚡ Optimize", "✅ Validate", "🛡️ Robustness", "📊 Compare", "📋 Settings", "🚀 Batch"
])


# =====================================================================
# TAB 1: OPTIMIZE (Phased)
# =====================================================================

with tab_optimize:
    st.header("Phased Optimization")
    st.caption("Run each phase, review results, then approve to continue.")

    # Trial count
    n_trials = st.slider(
        "Trials per phase",
        min_value=10, max_value=500, value=200, step=10,
        help="Phase 1-2: 200 recommended. Phase 3-4: 50-100 is fine.",
        key="opt_trials"
    )

    # Refresh status
    phase_status = get_phase_status(symbol)

    for phase in range(1, 5):
        config = PHASE_CONFIG[phase]

        st.divider()
        col_header, col_action = st.columns([3, 1])

        with col_header:
            status_icon = "✅" if phase_status[phase]['complete'] else "⬜"
            st.subheader(f"Phase {phase}: {config['name']} {status_icon}")
            st.caption(config['description'])
            st.caption(f"Objective: `{config['objective']}` | Min trades: {config['min_trades']}")
            st.caption(f"Search params: {', '.join(config['search_params'])}")

        # Check prerequisites
        prereqs_met = all(
            phase_status[p]['complete'] for p in config['requires_phases']
        )

        # Show existing results if complete
        if phase_status[phase]['complete']:
            ps = phase_status[phase]
            with st.expander(f"Phase {phase} Results (score: {ps['score']:.4f})", expanded=False):
                st.markdown("**Best Parameters:**")
                for k, v in sorted(ps['params'].items()):
                    st.text(f"  {k}: {v}")

                if ps.get('train') and ps.get('test'):
                    display_metrics_comparison(ps['train'], ps['test'])

        # Run / Re-run button
        with col_action:
            if not prereqs_met:
                st.warning(f"Needs Phase {config['requires_phases']}")
                can_run = False
            else:
                can_run = date_error is None

            button_label = "🔄 Re-run" if phase_status[phase]['complete'] else "▶️ Run"
            run_key = f"run_phase_{phase}_{symbol}"

            if st.button(button_label, key=run_key, disabled=not can_run, type="primary"):
                # Delete stale Optuna study
                with st.spinner("Clearing previous study..."):
                    delete_optuna_study(symbol, phase)

                # Run optimization
                progress_bar = st.progress(0)
                status_text = st.empty()

                def phase_callback(study, trial):
                    pct = (trial.number + 1) / n_trials
                    progress_bar.progress(pct)
                    best = study.best_value if study.best_trial else float('-inf')
                    score_str = f"{best:.4f}" if best != float('-inf') else "-inf"
                    status_text.text(
                        f"Trial {trial.number + 1}/{n_trials} | Best: {score_str}"
                    )

                try:
                    status_text.text(f"Starting Phase {phase}...")

                    opt = PhasedOptimizer(
                        symbol=symbol,
                        phase=phase,
                        train_start=str(train_start),
                        train_end=str(train_end),
                        test_start=str(test_start),
                        test_end=str(test_end),
                        n_trials=n_trials,
                        results_dir=RESULTS_DIR,
                        verbose=False,
                    )

                    # Suppress Optuna logging
                    optuna.logging.set_verbosity(optuna.logging.WARNING)

                    # Hook into study with callback
                    from optuna.samplers import TPESampler
                    sampler = TPESampler(seed=42)
                    storage = f'sqlite:///{opt.storage_path}'
                    study_name = f'ORB_{symbol}_phase{phase}_v3'

                    study = optuna.create_study(
                        direction='maximize',
                        sampler=sampler,
                        study_name=study_name,
                        storage=storage,
                        load_if_exists=True,
                    )
                    opt.study = study

                    study.optimize(
                        opt._objective,
                        n_trials=n_trials,
                        callbacks=[phase_callback],
                        show_progress_bar=False,
                    )

                    progress_bar.progress(1.0)
                    status_text.text("Running validation...")

                    # Get best params and save
                    best_trial = study.best_trial
                    best_searched_params = {
                        k: best_trial.params[k]
                        for k in config['search_params']
                        if k in best_trial.params
                    }

                    validation = opt._run_validation(best_searched_params)

                    output = {
                        'symbol': symbol,
                        'phase': phase,
                        'phase_name': config['name'],
                        'optimization_date': datetime.now().isoformat(),
                        'optimizer_version': 'v3_phased',
                        'objective': config['objective'],
                        'train_period': {
                            'start': str(train_start),
                            'end': str(train_end)
                        },
                        'test_period': {
                            'start': str(test_start),
                            'end': str(test_end)
                        },
                        'n_trials': n_trials,
                        'best_score': best_trial.value,
                        'best_params': best_searched_params,
                        'fixed_params': opt.fixed_params,
                        'train_results': validation['train'],
                        'test_results': validation['test'],
                    }

                    output_path = os.path.join(
                        RESULTS_DIR, f'{symbol}_phase{phase}.json'
                    )
                    with open(output_path, 'w') as f:
                        json.dump(output, f, indent=2)

                    status_text.text(f"✅ Phase {phase} complete!")
                    st.success(
                        f"Phase {phase} done! Score: {best_trial.value:.4f}"
                    )

                    # Show results inline
                    display_metrics_comparison(
                        validation['train'], validation['test']
                    )

                    st.rerun()

                except Exception as e:
                    st.error(f"Phase {phase} failed: {str(e)}")
                    raise e


# =====================================================================
# TAB 2: VALIDATE
# =====================================================================

with tab_validate:
    st.header("Validation & A/B Testing")

    phase_status = get_phase_status(symbol)
    all_phases_done = all(phase_status[p]['complete'] for p in range(1, 5))

    if not all_phases_done:
        missing = [p for p in range(1, 5) if not phase_status[p]['complete']]
        st.warning(f"Complete all 4 phases first. Missing: Phase {missing}")
    else:
        # Full Validation
        st.subheader("Full Validation (FCF)")
        st.caption("Combines all 4 phase results and runs a clean backtest.")

        validation = get_validation_status(symbol)

        if validation:
            st.success("Validation complete")
            display_metrics_comparison(
                validation['train_results'],
                validation['test_results']
            )

            with st.expander("Complete Parameter Set"):
                for k, v in sorted(validation['all_params'].items()):
                    st.text(f"  {k}: {v}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🔄 Re-run Validation" if validation else "▶️ Run Validation",
                type="primary", key="run_validation"
            ):
                with st.spinner("Running full validation..."):
                    try:
                        result = run_full_validation(
                            symbol=symbol,
                            train_start=str(train_start),
                            train_end=str(train_end),
                            test_start=str(test_start),
                            test_end=str(test_end),
                            results_dir=RESULTS_DIR,
                        )
                        st.success("Validation complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Validation failed: {e}")

        # A/B Testing
        st.divider()
        st.subheader("A/B Comparison: EMA Exit ON vs OFF")
        st.caption(
            "Compare your validated config against an override "
            "to measure the exact cost/benefit of EMA exit."
        )

        if validation:
            ab_param = st.selectbox(
                "Parameter to toggle",
                options=[
                    'use_ema_exit', 'use_adaptive_be',
                    'use_aggressive_trailing'
                ],
                key="ab_param"
            )

            current_val = validation['all_params'].get(ab_param, False)
            st.caption(
                f"Current value: `{ab_param} = {current_val}` "
                f"| A/B will test: `{ab_param} = {not current_val}`"
            )

            if st.button("🧪 Run A/B Test", type="primary", key="run_ab"):
                with st.spinner("Running A/B comparison..."):
                    try:
                        # Run with toggled param
                        params_toggled = validation['all_params'].copy()
                        params_toggled[ab_param] = not current_val

                        bt_train = ORBBacktester(
                            symbol=symbol,
                            start_date=str(train_start),
                            end_date=str(train_end),
                            verbose=False,
                            **params_toggled
                        )
                        train_toggled = bt_train.run()

                        bt_test = ORBBacktester(
                            symbol=symbol,
                            start_date=str(test_start),
                            end_date=str(test_end),
                            verbose=False,
                            **params_toggled
                        )
                        test_toggled = bt_test.run()

                        # Display comparison
                        st.divider()
                        st.subheader("A/B Results")

                        # Build comparison dataframe
                        metrics = [
                            'total_trades', 'win_rate', 'total_r',
                            'avg_r_per_trade', 'profit_factor', 'max_drawdown_r'
                        ]
                        labels = [
                            'Total Trades', 'Win Rate', 'Total R',
                            'Avg R/Trade', 'Profit Factor', 'Max Drawdown'
                        ]

                        for period, orig, toggled, period_name in [
                            (
                                'train',
                                validation['train_results'],
                                {
                                    'total_trades': train_toggled['total_trades'],
                                    'win_rate': train_toggled['win_rate'],
                                    'total_r': float(train_toggled['total_r']),
                                    'avg_r_per_trade': float(train_toggled['avg_r_per_trade']),
                                    'profit_factor': float(train_toggled['profit_factor']),
                                    'max_drawdown_r': float(train_toggled['max_drawdown_r']),
                                },
                                "Training"
                            ),
                            (
                                'test',
                                validation['test_results'],
                                {
                                    'total_trades': test_toggled['total_trades'],
                                    'win_rate': test_toggled['win_rate'],
                                    'total_r': float(test_toggled['total_r']),
                                    'avg_r_per_trade': float(test_toggled['avg_r_per_trade']),
                                    'profit_factor': float(test_toggled['profit_factor']),
                                    'max_drawdown_r': float(test_toggled['max_drawdown_r']),
                                },
                                "Testing"
                            ),
                        ]:
                            st.markdown(f"**{period_name} Period**")
                            rows = []
                            for m, label in zip(metrics, labels):
                                orig_val = orig.get(m, 0)
                                tog_val = toggled.get(m, 0)
                                delta = tog_val - orig_val
                                fmt = '.1f' if m == 'win_rate' else '.2f'
                                if m == 'total_trades':
                                    rows.append({
                                        'Metric': label,
                                        f'{ab_param}={current_val}': f"{orig_val}",
                                        f'{ab_param}={not current_val}': f"{tog_val}",
                                        'Delta': f"{delta:+.0f}",
                                    })
                                else:
                                    rows.append({
                                        'Metric': label,
                                        f'{ab_param}={current_val}': f"{orig_val:{fmt}}",
                                        f'{ab_param}={not current_val}': f"{tog_val:{fmt}}",
                                        'Delta': f"{delta:+{fmt}}",
                                    })

                            df = pd.DataFrame(rows)
                            st.dataframe(df, use_container_width=True, hide_index=True)

                    except Exception as e:
                        st.error(f"A/B test failed: {e}")
        else:
            st.info("Run validation first to enable A/B testing.")


# =====================================================================
# TAB 3: ROBUSTNESS
# =====================================================================

with tab_robustness:
    st.header("Robustness Testing")

    validation = get_validation_status(symbol)

    if not validation:
        st.warning("Run full validation first (Validate tab).")
    else:
        robustness = get_robustness_status(symbol)

        # Walk-Forward
        st.subheader("Walk-Forward Analysis")
        st.caption(
            "Tests your locked parameters across rolling time windows. "
            "Each fold trains and tests on different unseen periods."
        )

        col_wf1, col_wf2 = st.columns(2)
        with col_wf1:
            wf_mode = st.selectbox(
                "Mode",
                options=['Rolling', 'Anchored'],
                help="Rolling: sliding window. Anchored: always starts from beginning.",
                key="wf_mode"
            )
        with col_wf2:
            wf_train_months = st.number_input(
                "Train window (months)", min_value=1, max_value=6, value=2,
                key="wf_train"
            )

        if st.button("▶️ Run Walk-Forward", type="primary", key="run_wf"):
            with st.spinner("Running walk-forward analysis..."):
                try:
                    wfa = WalkForwardAnalysis(
                        symbol=symbol,
                        data_start=str(train_start),
                        data_end=str(test_end),
                        train_months=wf_train_months,
                        test_months=1,
                        step_months=1,
                        anchored=(wf_mode == 'Anchored'),
                        results_dir=RESULTS_DIR,
                        verbose=False,
                    )
                    wf_result = wfa.run_fixed()
                    st.success("Walk-forward complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Walk-forward failed: {e}")

        # Display WF results
        if robustness and 'walk_forward' in robustness:
            wf = robustness['walk_forward']
            summary = wf.get('summary', {})

            verdict = summary.get('verdict', 'N/A')
            if 'STRONG PASS' in verdict:
                st.success(f"**{verdict}**")
            elif 'PASS' in verdict:
                st.info(f"**{verdict}**")
            elif 'MARGINAL' in verdict:
                st.warning(f"**{verdict}**")
            else:
                st.error(f"**{verdict}**")

            # Fold table
            fold_rows = []
            for f in wf.get('folds', []):
                t = f['test']
                fold_rows.append({
                    'Fold': f['fold'],
                    'Period': f"{f['windows']['test_start']} to {f['windows']['test_end']}",
                    'Trades': t['total_trades'],
                    'Win Rate': f"{t['win_rate']:.1f}%",
                    'Total R': f"{t['total_r']:+.2f}",
                    'PF': f"{t['profit_factor']:.2f}",
                    'Max DD': f"{t['max_drawdown_r']:.2f}R",
                })
            if fold_rows:
                st.dataframe(
                    pd.DataFrame(fold_rows),
                    use_container_width=True, hide_index=True
                )

            # Aggregate metrics
            with st.expander("Aggregate OOS Metrics"):
                acol1, acol2, acol3, acol4 = st.columns(4)
                acol1.metric("Combined R", f"{summary.get('oos_total_r_sum', 0):+.2f}")
                acol2.metric("Mean PF", f"{summary.get('oos_pf_mean', 0):.2f}")
                acol3.metric("Worst DD", f"{summary.get('oos_worst_dd', 0):.2f}R")
                pct = summary.get('profitable_pct', 0)
                total = summary.get('total_folds', 0)
                prof = summary.get('profitable_folds', 0)
                acol4.metric("Profitable Folds", f"{prof}/{total} ({pct:.0f}%)")

        # Monte Carlo
        st.divider()
        st.subheader("Monte Carlo Simulation")
        st.caption(
            "Randomly reorders your trades thousands of times to build "
            "a distribution of possible equity curves and drawdowns."
        )

        mc_sims = st.slider(
            "Simulations",
            min_value=1000, max_value=50000, value=10000, step=1000,
            key="mc_sims"
        )

        if st.button("▶️ Run Monte Carlo", type="primary", key="run_mc"):
            with st.spinner(f"Running {mc_sims:,} simulations..."):
                try:
                    mc = MonteCarloAnalysis(
                        symbol=symbol,
                        start_date=str(train_start),
                        end_date=str(test_end),
                        results_dir=RESULTS_DIR,
                        verbose=False,
                    )
                    mc_result = mc.run(n_simulations=mc_sims)
                    st.success("Monte Carlo complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Monte Carlo failed: {e}")

        # Display MC results
        if robustness and 'monte_carlo' in robustness:
            mc = robustness['monte_carlo']

            eq = mc.get('equity', {})
            dd = mc.get('drawdown', {})
            risk = mc.get('risk', {})

            # Key risk metrics
            rcol1, rcol2, rcol3, rcol4 = st.columns(4)
            rcol1.metric("P(Losing Money)", f"{risk.get('prob_losing', 0):.1f}%")
            rcol2.metric("Median Equity", f"{eq.get('median', 0):+.1f}R")
            rcol3.metric("Median Max DD", f"{dd.get('median', 0):.1f}R")
            rcol4.metric("P(DD > 10R)", f"{risk.get('prob_dd_10r', 0):.1f}%")

            col_eq, col_dd = st.columns(2)

            with col_eq:
                st.markdown("**Equity Distribution**")
                eq_data = {
                    'Percentile': [
                        '95th', '75th', 'Median', 'Mean',
                        '25th', '10th', '5th', 'Worst'
                    ],
                    'R': [
                        eq.get('p95', 0), eq.get('p75', 0),
                        eq.get('median', 0), eq.get('mean', 0),
                        eq.get('p25', 0), eq.get('p10', 0),
                        eq.get('p5', 0), eq.get('worst', 0),
                    ]
                }
                df_eq = pd.DataFrame(eq_data)
                df_eq['R'] = df_eq['R'].apply(lambda x: f"{x:+.2f}")
                st.dataframe(df_eq, use_container_width=True, hide_index=True)

            with col_dd:
                st.markdown("**Drawdown Distribution**")
                dd_data = {
                    'Percentile': [
                        'Median', 'Mean', '75th',
                        '90th', '95th', '99th', 'Worst'
                    ],
                    'R': [
                        dd.get('median', 0), dd.get('mean', 0),
                        dd.get('p75', 0), dd.get('p90', 0),
                        dd.get('p95', 0), dd.get('p99', 0),
                        dd.get('worst', 0),
                    ]
                }
                df_dd = pd.DataFrame(dd_data)
                df_dd['R'] = df_dd['R'].apply(lambda x: f"{x:.2f}")
                st.dataframe(df_dd, use_container_width=True, hide_index=True)

            # Risk table
            st.markdown("**Risk Probabilities**")
            risk_rows = [
                {'Scenario': 'P(losing money)', 'Probability': f"{risk.get('prob_losing', 0):.1f}%"},
                {'Scenario': 'P(DD > 5R)', 'Probability': f"{risk.get('prob_dd_5r', 0):.1f}%"},
                {'Scenario': 'P(DD > 8R)', 'Probability': f"{risk.get('prob_dd_8r', 0):.1f}%"},
                {'Scenario': 'P(DD > 10R)', 'Probability': f"{risk.get('prob_dd_10r', 0):.1f}%"},
                {'Scenario': 'P(DD > 15R)', 'Probability': f"{risk.get('prob_dd_15r', 0):.1f}%"},
            ]
            st.dataframe(
                pd.DataFrame(risk_rows),
                use_container_width=True, hide_index=True
            )


# =====================================================================
# TAB 4: COMPARE
# =====================================================================

with tab_compare:
    st.header("Cross-Symbol Comparison")
    st.caption("Compare validated results across all optimized symbols.")

    # Scan results directory for validated symbols
    validated_symbols = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith('_FINAL_validation.json'):
            sym = filename.replace('_FINAL_validation.json', '')
            validated_symbols.append(sym)

    if not validated_symbols:
        st.info("No validated symbols yet. Complete optimization and validation first.")
    else:
        st.success(f"Validated symbols: {', '.join(sorted(validated_symbols))}")

        # Build comparison table
        rows = []
        for sym in sorted(validated_symbols):
            val = get_validation_status(sym)
            if not val:
                continue

            train = val.get('train_results', {})
            test = val.get('test_results', {})

            row = {
                'Symbol': sym,
                'Train R': f"{train.get('total_r', 0):+.2f}",
                'Test R': f"{test.get('total_r', 0):+.2f}",
                'Train WR': f"{train.get('win_rate', 0):.1f}%",
                'Test WR': f"{test.get('win_rate', 0):.1f}%",
                'Train PF': f"{train.get('profit_factor', 0):.2f}",
                'Test PF': f"{test.get('profit_factor', 0):.2f}",
                'Test DD': f"{test.get('max_drawdown_r', 0):.2f}R",
                'Trades': test.get('total_trades', 0),
            }
            rows.append(row)

            # Add robustness data if available
            rob = get_robustness_status(sym)
            if rob and 'monte_carlo' in rob:
                mc = rob['monte_carlo']
                row['MC P(loss)'] = f"{mc['risk']['prob_losing']:.1f}%"
                row['MC Median R'] = f"{mc['equity']['median']:+.1f}"
            if rob and 'walk_forward' in rob:
                wf = rob['walk_forward']
                row['WF Verdict'] = wf.get('summary', {}).get('verdict', 'N/A')

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True, hide_index=True
            )

        # Detailed comparison
        if len(validated_symbols) >= 2:
            st.divider()
            st.subheader("Head-to-Head")
            col1, col2 = st.columns(2)
            with col1:
                sym_a = st.selectbox("Symbol A", validated_symbols, key="sym_a")
            with col2:
                sym_b = st.selectbox(
                    "Symbol B",
                    [s for s in validated_symbols if s != sym_a],
                    key="sym_b"
                )

            val_a = get_validation_status(sym_a)
            val_b = get_validation_status(sym_b)

            if val_a and val_b:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{sym_a}**")
                    test_a = val_a['test_results']
                    st.metric("Test R", f"{test_a['total_r']:+.2f}")
                    st.metric("Win Rate", f"{test_a['win_rate']:.1f}%")
                    st.metric("Profit Factor", f"{test_a['profit_factor']:.2f}")
                    st.metric("Max DD", f"{test_a['max_drawdown_r']:.2f}R")

                with col2:
                    st.markdown(f"**{sym_b}**")
                    test_b = val_b['test_results']
                    st.metric("Test R", f"{test_b['total_r']:+.2f}")
                    st.metric("Win Rate", f"{test_b['win_rate']:.1f}%")
                    st.metric("Profit Factor", f"{test_b['profit_factor']:.2f}")
                    st.metric("Max DD", f"{test_b['max_drawdown_r']:.2f}R")


# =====================================================================
# TAB 5: SETTINGS EXPORT
# =====================================================================

with tab_settings:
    st.header("Settings Export")
    st.caption(
        "Map optimized parameters to Pine Script inputs. "
        "Generate cheat sheets and preset blocks."
    )

    validation = get_validation_status(symbol)

    if not validation:
        st.warning(
            f"No validated results for {symbol}. "
            f"Complete optimization and validation first."
        )
    else:
        # Cheat Sheet
        st.subheader(f"📋 {symbol} Cheat Sheet")
        st.caption("Use this to manually update TradingView strategy settings.")

        try:
            exporter = SettingsExporter(symbol=symbol, results_dir=RESULTS_DIR)
            grouped = exporter.get_cheat_sheet_grouped()

            for group, params in grouped.items():
                with st.expander(f"**{group}**", expanded=True):
                    for p in params:
                        col_label, col_val = st.columns([3, 1])
                        with col_label:
                            st.text(p['label'])
                        with col_val:
                            if isinstance(p['value'], bool):
                                if p['value']:
                                    st.success("ON")
                                else:
                                    st.error("OFF")
                            else:
                                st.info(p['display'])

            # Time conversion note
            start_min = validation['all_params'].get('trading_start_minutes', 570)
            end_min = validation['all_params'].get('trading_end_minutes', 690)
            st.info(
                f"**Trading Window:** {minutes_to_time(start_min)} - "
                f"{minutes_to_time(end_min)}"
            )

            # Copy-pasteable text version
            st.divider()
            st.subheader("📄 Text Version (copy-paste)")
            cheat_text = exporter.get_cheat_sheet_text()
            st.code(cheat_text, language="text")

        except Exception as e:
            st.error(f"Error generating cheat sheet: {e}")

        # Pine Script Preset Block + AHK Injection
        st.divider()
        st.subheader("🌲 Pine Script Preset Block")
        st.caption(
            "Generates the preset file for AHK injection. "
            "Hit the button below, then press **F2** in TradingView."
        )

        # Find all validated symbols
        validated_symbols = []
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith('_FINAL_validation.json'):
                sym = filename.replace('_FINAL_validation.json', '')
                validated_symbols.append(sym)

        if validated_symbols:
            selected_syms = st.multiselect(
                "Include symbols in preset block",
                options=sorted(validated_symbols),
                default=sorted(validated_symbols),
                key="preset_symbols"
            )

            col_write, col_preview = st.columns(2)

            with col_write:
                if selected_syms and st.button(
                    "🚀 Write Preset File (F2 Ready)",
                    type="primary", key="write_preset"
                ):
                    try:
                        result = write_preset_file(
                            selected_syms, results_dir=RESULTS_DIR
                        )
                        st.success(
                            f"✅ Written to: `{result['path']}`\n\n"
                            f"Symbols: {', '.join(result['symbols'])}\n\n"
                            f"**Now press F2 in TradingView to inject!**"
                        )
                        st.session_state['last_preset_block'] = result['block_text']
                    except Exception as e:
                        st.error(f"Preset write failed: {e}")

            with col_preview:
                if selected_syms and st.button(
                    "👀 Preview Only", key="preview_preset"
                ):
                    try:
                        block = generate_preset_block(
                            selected_syms, results_dir=RESULTS_DIR
                        )
                        st.session_state['last_preset_block'] = block
                    except Exception as e:
                        st.error(f"Preview failed: {e}")

            # Show the block if generated
            if 'last_preset_block' in st.session_state:
                with st.expander("Generated Preset Block", expanded=False):
                    st.code(
                        st.session_state['last_preset_block'],
                        language="javascript"
                    )

            # File status
            if os.path.exists(PRESET_FILE_PATH):
                mod_time = datetime.fromtimestamp(
                    os.path.getmtime(PRESET_FILE_PATH)
                ).strftime('%Y-%m-%d %H:%M')
                size_kb = os.path.getsize(PRESET_FILE_PATH) / 1024
                st.caption(
                    f"📁 `{PRESET_FILE_PATH}` — "
                    f"Last modified: {mod_time} — "
                    f"Size: {size_kb:.1f} KB"
                )
            else:
                st.caption(f"📁 `{PRESET_FILE_PATH}` — not yet created")

        else:
            st.info("No validated symbols found.")

        # Pine Script Scaffolding (one-time setup)
        st.divider()
        st.subheader("🔩 Pine Script Scaffolding (one-time)")
        st.caption(
            "New map declarations and resolution lines needed in your "
            "Pine Script to support optimizer-tuned parameters. "
            "Add these **once** when first integrating ORB Lab."
        )

        with st.expander("Map Declarations (before preset block markers)"):
            st.code(generate_map_declarations(), language="javascript")

        with st.expander("Resolution Lines (after preset block markers)"):
            st.code(generate_resolution_lines(), language="javascript")

        # Master Config
        st.divider()
        st.subheader("💾 Master Config")
        st.caption(
            "JSON file with all symbol settings for quick reference."
        )

        if st.button("💾 Generate Master Config", key="gen_master"):
            try:
                master = generate_master_config(results_dir=RESULTS_DIR)
                st.success(
                    f"Master config saved with "
                    f"{len(master['symbols'])} symbols."
                )
                for sym, data in master['symbols'].items():
                    with st.expander(f"{sym} - Optimizer Params"):
                        for k, v in sorted(data.items()):
                            st.text(f"  {k}: {v}")
            except Exception as e:
                st.error(f"Master config failed: {e}")


# =====================================================================
# TAB 6: BATCH OPTIMIZER
# =====================================================================

with tab_batch:
    st.header("Batch Optimizer")
    st.caption(
        "Run multiple symbols through all 4 phases in parallel. "
        "Each symbol gets its own background process — your i9 can handle it."
    )

    # --- Symbol Selection ---
    all_symbols = ['AMD', 'NVDA', 'TSLA', 'AAPL', 'GOOGL', 'PLTR', 'META',
                   'MSFT', 'AMZN', 'NFLX', 'SPY', 'QQQ', 'SOFI', 'COIN', 'MARA']

    batch_symbols = st.multiselect(
        "Symbols to optimize",
        options=all_symbols,
        default=['NVDA', 'GOOGL', 'TSLA', 'META'],
        key="batch_symbols"
    )

    col_trials, col_workers = st.columns(2)
    with col_trials:
        batch_trials = st.slider(
            "Trials per phase",
            min_value=50, max_value=500, value=200, step=50,
            key="batch_trials"
        )
    with col_workers:
        max_workers = st.slider(
            "Max parallel workers",
            min_value=1, max_value=6, value=4, step=1,
            help="i9-13980HX: 4-6 is ideal",
            key="batch_workers"
        )

    # Date ranges (use sidebar values)
    st.caption(
        f"**Date Range:** Train {train_start} → {train_end} | "
        f"Test {test_start} → {test_end} (from sidebar)"
    )

    st.divider()

    # --- Launch Button ---
    if batch_symbols and st.button(
        f"🚀 Launch {len(batch_symbols)} Workers",
        type="primary", key="launch_batch"
    ):
        # Clear old status files
        for sym in batch_symbols:
            status_path = os.path.join(RESULTS_DIR, f'{sym}_batch_status.json')
            if os.path.exists(status_path):
                os.remove(status_path)

        # Build the worker command
        worker_script = os.path.join(PROJECT_ROOT, 'batch_worker.py')

        # Launch processes in batches of max_workers
        launched = []
        for i, sym in enumerate(batch_symbols):
            cmd = [
                sys.executable, worker_script,
                '--symbol', sym,
                '--trials', str(batch_trials),
                '--train-start', str(train_start),
                '--train-end', str(train_end),
                '--test-start', str(test_start),
                '--test-end', str(test_end),
            ]

            # Create log file for each worker
            log_dir = os.path.join(RESULTS_DIR, 'batch_logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = open(os.path.join(log_dir, f'{sym}_batch.log'), 'w')

            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
            launched.append((sym, proc, log_file))

            # Stagger launches slightly to avoid Optuna DB contention
            if i < len(batch_symbols) - 1:
                time.sleep(1)

        # Store PIDs in session state
        st.session_state['batch_pids'] = {
            sym: proc.pid for sym, proc, _ in launched
        }
        st.session_state['batch_launched'] = True
        st.session_state['batch_symbols'] = batch_symbols

        st.success(
            f"Launched {len(launched)} workers: "
            f"{', '.join(batch_symbols)}"
        )

    # --- Status Dashboard ---
    st.divider()
    st.subheader("📊 Batch Status")

    # Determine which symbols to monitor
    monitor_symbols = st.session_state.get('batch_symbols', batch_symbols)

    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (every 5s)", value=True, key="batch_refresh")

    # Status cards
    status_cols = st.columns(min(len(monitor_symbols), 4)) if monitor_symbols else []

    all_complete = True
    any_running = False

    for i, sym in enumerate(monitor_symbols):
        col = status_cols[i % len(status_cols)] if status_cols else st

        with col:
            status_path = os.path.join(RESULTS_DIR, f'{sym}_batch_status.json')

            if os.path.exists(status_path):
                try:
                    with open(status_path, 'r') as f:
                        status = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    st.info(f"**{sym}** — Starting...")
                    all_complete = False
                    any_running = True
                    continue

                state = status.get('state', 'unknown')
                current_phase = status.get('current_phase', '?')
                phases = status.get('phases', {})

                # Header with state indicator
                if state == 'complete':
                    st.success(f"**{sym}** ✅ Complete")
                elif state == 'error':
                    st.error(f"**{sym}** ❌ Error")
                    all_complete = False
                else:
                    phase_name = PHASE_CONFIG.get(current_phase, {}).get('name', 'Validation') if isinstance(current_phase, int) else 'Validation'
                    st.warning(f"**{sym}** ⏳ Phase {current_phase}: {phase_name}")
                    all_complete = False
                    any_running = True

                # Phase summary
                for p in ['1', '2', '3', '4']:
                    if p in phases:
                        ph = phases[p]
                        ph_state = ph.get('state', 'pending')
                        ph_score = ph.get('score')

                        if ph_state == 'complete':
                            score_str = f"{ph_score:.2f}" if ph_score else "—"
                            train_r = ph.get('train', {}).get('total_r', 0)
                            test_r = ph.get('test', {}).get('total_r', 0)
                            st.caption(
                                f"P{p} ✅ Score: {score_str} | "
                                f"Train: {train_r:.1f}R | Test: {test_r:.1f}R"
                            )
                        elif ph_state == 'running':
                            st.caption(f"P{p} ⏳ Running...")
                        elif ph_state == 'error':
                            st.caption(f"P{p} ❌ {ph.get('error', 'unknown')}")

                # Show elapsed time
                started = status.get('started_at')
                if started and state != 'complete':
                    start_dt = datetime.fromisoformat(started)
                    elapsed = datetime.now() - start_dt
                    mins = int(elapsed.total_seconds() // 60)
                    secs = int(elapsed.total_seconds() % 60)
                    st.caption(f"⏱️ {mins}m {secs}s elapsed")
                elif state == 'complete':
                    completed = status.get('completed_at')
                    if completed and started:
                        start_dt = datetime.fromisoformat(started)
                        end_dt = datetime.fromisoformat(completed)
                        elapsed = end_dt - start_dt
                        mins = int(elapsed.total_seconds() // 60)
                        secs = int(elapsed.total_seconds() % 60)
                        st.caption(f"⏱️ Finished in {mins}m {secs}s")

            else:
                st.info(f"**{sym}** — Waiting to start...")
                all_complete = False

    # Auto-refresh mechanism
    if auto_refresh and any_running:
        time.sleep(5)
        st.rerun()

    # Summary when complete
    if all_complete and monitor_symbols and st.session_state.get('batch_launched'):
        st.divider()
        st.success(f"🎉 All {len(monitor_symbols)} symbols complete!")
        st.caption("Head to the **Compare** tab to see cross-symbol results, "
                   "or **Settings** tab to generate preset blocks.")

        # Show summary table
        summary_rows = []
        for sym in monitor_symbols:
            status_path = os.path.join(RESULTS_DIR, f'{sym}_batch_status.json')
            if os.path.exists(status_path):
                with open(status_path, 'r') as f:
                    status = json.load(f)
                p4 = status.get('phases', {}).get('4', {})
                summary_rows.append({
                    'Symbol': sym,
                    'P1 Score': status.get('phases', {}).get('1', {}).get('score', 0),
                    'P4 Score': p4.get('score', 0),
                    'Train R': p4.get('train', {}).get('total_r', 0),
                    'Test R': p4.get('test', {}).get('total_r', 0),
                    'Train WR%': p4.get('train', {}).get('win_rate', 0),
                    'Test WR%': p4.get('test', {}).get('win_rate', 0),
                })

        if summary_rows:
            df = pd.DataFrame(summary_rows)
            for col in ['P1 Score', 'P4 Score', 'Train R', 'Test R']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: f"{x:.2f}" if x else "—")
            for col in ['Train WR%', 'Test WR%']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: f"{x:.1f}%" if x else "—")
            st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Batch Log Viewer ---
    st.divider()
    st.subheader("📝 Worker Logs")
    log_dir = os.path.join(RESULTS_DIR, 'batch_logs')
    if os.path.exists(log_dir):
        log_symbol = st.selectbox(
            "View log for",
            options=monitor_symbols,
            key="log_viewer_sym"
        )
        log_path = os.path.join(log_dir, f'{log_symbol}_batch.log')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log_content = f.read()
            if log_content:
                with st.expander(f"{log_symbol} log (last 100 lines)", expanded=False):
                    lines = log_content.strip().split('\n')
                    st.code('\n'.join(lines[-100:]), language="text")
            else:
                st.caption("Log is empty — worker may still be starting.")
        else:
            st.caption("No log file yet.")
    else:
        st.caption("No batch runs yet.")


# =====================================================================
# FOOTER
# =====================================================================

st.divider()
st.caption("ORB Lab v2.0 | Phased Optimizer + Robustness Suite + Batch Runner | Built with Streamlit")
