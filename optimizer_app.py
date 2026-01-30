"""
ORB Optimizer UI - Streamlit App
================================
Simple web interface for running ORB strategy optimizations.

Launch with: streamlit run optimizer_app.py
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import optuna
from datetime import datetime, date
from orb_optimizer import ORBOptimizer, PROJECT_ROOT
import subprocess
import threading

# Page config
st.set_page_config(
    page_title="ORB Optimizer",
    page_icon="📈",
    layout="centered"
)

# Title
st.title("📈 ORB Optimizer")
st.markdown("*Opening Range Breakout Strategy Optimization*")
st.divider()

# Symbols with presets
PRESET_SYMBOLS = ['AMD', 'NVDA', 'TSLA', 'AAPL', 'GOOGL', 'PLTR']

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")

    # Objective metric
    objective_metric = st.selectbox(
        "Objective Metric",
        options=['robust', 'total_r', 'profit_factor', 'sharpe', 'avg_r_per_trade'],
        index=0,
        help="'robust' penalizes overfitting and fragility"
    )

    # Search mode
    narrow_search = st.toggle(
        "Narrow Search",
        value=True,
        help="Constrain search around proven defaults (recommended)"
    )

    # Min trades filter
    min_trades = st.number_input(
        "Min Trades Required",
        min_value=5,
        max_value=100,
        value=20,
        help="Minimum trades required for valid backtest"
    )

    st.divider()

    # Dashboard launcher
    st.header("📊 Optuna Dashboard")
    db_path = os.path.join(PROJECT_ROOT, 'orb_optuna.db')

    if os.path.exists(db_path):
        st.success("Database found")
        st.code(f"optuna-dashboard sqlite:///{db_path}", language="bash")
        st.caption("Run this command in terminal to launch dashboard")
    else:
        st.warning("No database yet - run an optimization first")

# Main content
col1, col2 = st.columns(2)

with col1:
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

with col2:
    # Number of trials
    n_trials = st.slider(
        "Number of Trials",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        help="More trials = better results but longer runtime"
    )

st.divider()

# Date range selection
st.subheader("📅 Date Ranges")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Training Period**")
    train_start = st.date_input(
        "Train Start",
        value=date(2025, 4, 1),
        key="train_start"
    )
    train_end = st.date_input(
        "Train End",
        value=date(2025, 9, 30),
        key="train_end"
    )

with col2:
    st.markdown("**Testing Period**")
    test_start = st.date_input(
        "Test Start",
        value=date(2025, 10, 1),
        key="test_start"
    )
    test_end = st.date_input(
        "Test End",
        value=date(2025, 12, 31),
        key="test_end"
    )

# Validation
date_error = None
if train_start >= train_end:
    date_error = "Train start must be before train end"
elif test_start >= test_end:
    date_error = "Test start must be before test end"
elif train_end >= test_start:
    date_error = "Training period must end before test period starts"

if date_error:
    st.error(date_error)

st.divider()

# Run button
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    run_button = st.button(
        "▶️ Run Optimization",
        type="primary",
        use_container_width=True,
        disabled=date_error is not None
    )

# Initialize session state
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Run optimization
if run_button:
    st.session_state.optimization_complete = False
    st.session_state.results = None

    st.divider()
    st.subheader(f"🔄 Optimizing {symbol}...")

    # Progress placeholder
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Create optimizer
    optimizer = ORBOptimizer(
        symbol=symbol,
        train_start=str(train_start),
        train_end=str(train_end),
        test_start=str(test_start),
        test_end=str(test_end),
        n_trials=n_trials,
        objective_metric=objective_metric,
        narrow_search=narrow_search,
        min_trades=min_trades,
        verbose=False
    )

    # Custom callback for progress
    def progress_callback(study, trial):
        progress = (trial.number + 1) / n_trials
        progress_bar.progress(progress)

        best_value = study.best_value if study.best_trial else 0
        status_text.text(f"Trial {trial.number + 1}/{n_trials} | Best score: {best_value:.3f}")

    # Run with progress tracking
    try:
        # Load data first (this takes time)
        status_text.text("Loading market data...")

        # We need to run the optimization with callbacks
        # Recreate the study setup to add our callback
        from optuna.samplers import TPESampler

        sampler = TPESampler(seed=42)
        storage = f'sqlite:///{optimizer.storage_path}'

        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'ORB_{symbol}_v2',
            storage=storage,
            load_if_exists=True
        )

        optimizer.study = study

        # Pre-load data by running a dummy check
        status_text.text("Loading market data (this may take a moment)...")

        # Run optimization with callback
        study.optimize(
            optimizer._objective,
            n_trials=n_trials,
            callbacks=[progress_callback],
            show_progress_bar=False
        )

        progress_bar.progress(1.0)
        status_text.text("Running validation on test data...")

        # Get best params and run validation
        best_params = study.best_params

        # Add fixed params back
        if narrow_search:
            best_params['use_break_even'] = True
            best_params['use_trailing_stop'] = True
            best_params['trading_start_minutes'] = 570

        # Validation runs
        from orb_backtester import ORBBacktester

        bt_train = ORBBacktester(
            symbol=symbol,
            start_date=str(train_start),
            end_date=str(train_end),
            verbose=False,
            **best_params
        )
        train_results = bt_train.run()

        bt_test = ORBBacktester(
            symbol=symbol,
            start_date=str(test_start),
            end_date=str(test_end),
            verbose=False,
            **best_params
        )
        test_results = bt_test.run()

        # Store results
        st.session_state.results = {
            'symbol': symbol,
            'best_params': best_params,
            'best_score': study.best_value,
            'train': train_results,
            'test': test_results
        }
        st.session_state.optimization_complete = True

        status_text.text("✅ Optimization complete!")

    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        raise e

# Display results
if st.session_state.optimization_complete and st.session_state.results:
    results = st.session_state.results

    st.divider()
    st.subheader(f"📊 Results for {results['symbol']}")

    # Score
    st.metric("Best Optimization Score", f"{results['best_score']:.3f}")

    # Train vs Test comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Training Performance**")
        train = results['train']
        st.metric("Total R", f"{train['total_r']:+.2f}")
        st.metric("Win Rate", f"{train['win_rate']:.1f}%")
        st.metric("Trades", train['total_trades'])
        st.metric("Profit Factor", f"{train['profit_factor']:.2f}")
        st.metric("Max Drawdown", f"{train['max_drawdown_r']:.2f}R")

    with col2:
        st.markdown("**Test Performance**")
        test = results['test']

        # Color code based on comparison to training
        r_delta = test['total_r'] - train['total_r']
        wr_delta = test['win_rate'] - train['win_rate']

        st.metric("Total R", f"{test['total_r']:+.2f}", delta=f"{r_delta:+.2f}")
        st.metric("Win Rate", f"{test['win_rate']:.1f}%", delta=f"{wr_delta:+.1f}%")
        st.metric("Trades", test['total_trades'])
        st.metric("Profit Factor", f"{test['profit_factor']:.2f}")
        st.metric("Max Drawdown", f"{test['max_drawdown_r']:.2f}R")

    # Best parameters
    with st.expander("🔧 Best Parameters"):
        for key, value in sorted(results['best_params'].items()):
            st.text(f"{key}: {value}")

    # Health check
    st.divider()

    if test['total_r'] > 0 and test['win_rate'] > 35:
        st.success("✅ Strategy shows positive out-of-sample performance")
    elif test['total_r'] > 0:
        st.warning("⚠️ Positive R but low win rate - may be fragile")
    else:
        st.error("❌ Negative out-of-sample performance - likely overfit")

# Footer
st.divider()
st.caption("ORB Lab v1.0 | Built with Streamlit")
