"""
ORB Optimizer History & Recovery Module
========================================
Scans results directory for all previous optimization runs,
displays them in a table, and can regenerate preset files
from any historical run.

Can be run standalone (CLI recovery) or imported by optimizer_app.py
as a Streamlit tab.

Results directory structure:
  results/
    {SYM}_phase1.json          # Phase results (best_params, train/test)
    {SYM}_phase2.json
    {SYM}_phase3.json
    {SYM}_phase4.json
    {SYM}_FINAL_validation.json  # Combined validation
    {SYM}_batch_status.json      # Batch runner status
    {SYM}_optimal.json           # Legacy single-phase results
    history/
      {SYM}_validation_{timestamp}.json  # Archived copies
    *.db                         # Optuna SQLite databases

Usage:
  python history_recovery.py                    # Scan and report
  python history_recovery.py --export AMD       # Export AMD presets from latest results
  python history_recovery.py --export-all       # Export all validated symbols
  python history_recovery.py --scan-optuna      # Deep scan Optuna DBs for recoverable data
"""

import json
import os
import sys
import glob
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
HISTORY_DIR = os.path.join(RESULTS_DIR, 'history')
PRESET_OUTPUT = r'C:\hmm_trading\pine_presets_orb.txt'

PHASE_NAMES = {
    1: 'Core Exits (BE/EMA/Trail)',
    2: 'Breakout Filters + Confluence',
    3: 'Stop Loss Selection',
    4: 'Volatility Engine + Time',
}

# Map from optimizer param names to Pine Script map names
PINE_MAP_NAMES = {
    'break_even_rr':              ('mapBERR', 'float'),
    'profit_target_rr':           ('mapProfitTarget', 'float'),
    'trailing_stop_distance':     ('mapTrailDist', 'float'),
    'ema_tighten_zone':           ('mapEmaTightenZone', 'float'),
    'tightened_trail_distance':   ('mapTightTrailDist', 'float'),
    'atr_stop_multiplier':       ('mapATRStopMult', 'float'),
    'min_stop_atr':              ('mapMinStopATR', 'float'),
    'vwap_stop_distance':        ('mapVWAPStopDist', 'float'),
    'min_acceptable_rr':         ('mapMinRR', 'float'),
    'max_target_atr':            ('mapMaxTargetATR', 'float'),
    'breakout_threshold_mult':   ('mapBreakoutThresh', 'float'),
    'min_body_strength':         ('mapMinBodyStr', 'float'),
    'skip_low_vol_c':            ('mapSkipLowVolC', 'bool'),
    'low_vol_min_rr':            ('mapLowVolMinRR', 'float'),
    'use_ema_cross_exit':        ('mapEmaExit', 'bool'),
    'use_adaptive_be':           ('mapAdaptiveBE', 'bool'),
    'aggressive_trailing':       ('mapAggressiveTrail', 'bool'),
    'enable_confluence':         ('mapConfluence', 'bool'),
}


# ============================================================
# SCANNING FUNCTIONS
# ============================================================

def scan_results_directory(results_dir=None):
    """Scan the results directory and return all available data."""
    results_dir = results_dir or RESULTS_DIR
    
    if not os.path.exists(results_dir):
        return {'error': f'Results directory not found: {results_dir}'}
    
    findings = {
        'results_dir': results_dir,
        'symbols': {},
        'history': [],
        'optuna_dbs': [],
        'other_files': [],
    }
    
    # Scan for phase results and validations
    for f in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        fname = os.path.basename(f)
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        fsize = os.path.getsize(f)
        
        # Parse symbol from filename
        # Match only exact pattern: {SYMBOL}_phase{1-4}.json
        import re as _re
        phase_match = _re.match(r'^([A-Z]{1,5})_phase([1-4])\.json$', fname)
        if phase_match:
            sym = phase_match.group(1)
            phase = int(phase_match.group(2))
            
            if sym not in findings['symbols']:
                findings['symbols'][sym] = {'phases': {}, 'validation': None, 'batch_status': None, 'optimal': None}
            
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                findings['symbols'][sym]['phases'][phase] = {
                    'file': f,
                    'modified': mtime.isoformat(),
                    'best_score': data.get('best_score'),
                    'best_params': data.get('best_params', {}),
                    'fixed_params': data.get('fixed_params', {}),
                    'train': data.get('train_results', {}),
                    'test': data.get('test_results', {}),
                    'n_trials': data.get('n_trials'),
                    'train_period': data.get('train_period', {}),
                    'test_period': data.get('test_period', {}),
                }
            except (json.JSONDecodeError, Exception) as e:
                findings['symbols'][sym]['phases'][phase] = {
                    'file': f, 'modified': mtime.isoformat(), 'error': str(e)
                }
        
        elif '_FINAL_validation' in fname:
            # Must be {SYMBOL}_FINAL_validation.json with clean ticker
            val_match = _re.match(r'^([A-Z]{1,5})_FINAL_validation\.json$', fname)
            if not val_match:
                findings['other_files'].append({'file': f, 'modified': mtime.isoformat(), 'size': fsize})
                continue
            sym = val_match.group(1)
            if sym not in findings['symbols']:
                findings['symbols'][sym] = {'phases': {}, 'validation': None, 'batch_status': None, 'optimal': None}
            
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                findings['symbols'][sym]['validation'] = {
                    'file': f,
                    'modified': mtime.isoformat(),
                    'all_params': data.get('all_params', data.get('best_params', {})),
                    'train': data.get('train_results', data.get('train', {})),
                    'test': data.get('test_results', data.get('test', {})),
                    'raw_data': data,
                }
            except (json.JSONDecodeError, Exception) as e:
                findings['symbols'][sym]['validation'] = {
                    'file': f, 'modified': mtime.isoformat(), 'error': str(e)
                }
        
        elif '_batch_status' in fname:
            sym = fname.replace('_batch_status.json', '')
            if sym not in findings['symbols']:
                findings['symbols'][sym] = {'phases': {}, 'validation': None, 'batch_status': None, 'optimal': None}
            
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                findings['symbols'][sym]['batch_status'] = {
                    'file': f,
                    'modified': mtime.isoformat(),
                    'state': data.get('state'),
                    'started_at': data.get('started_at'),
                    'completed_at': data.get('completed_at'),
                    'phases': data.get('phases', {}),
                }
            except Exception as e:
                findings['symbols'][sym]['batch_status'] = {
                    'file': f, 'modified': mtime.isoformat(), 'error': str(e)
                }
        
        elif '_optimal' in fname:
            sym = fname.replace('_optimal.json', '')
            if sym not in findings['symbols']:
                findings['symbols'][sym] = {'phases': {}, 'validation': None, 'batch_status': None, 'optimal': None}
            
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                findings['symbols'][sym]['optimal'] = {
                    'file': f,
                    'modified': mtime.isoformat(),
                    'data': data,
                }
            except Exception as e:
                pass
        
        else:
            findings['other_files'].append({'file': f, 'modified': mtime.isoformat(), 'size': fsize})
    
    # Scan history directory
    history_dir = os.path.join(results_dir, 'history')
    if os.path.exists(history_dir):
        for f in sorted(glob.glob(os.path.join(history_dir, '*.json')), reverse=True):
            fname = os.path.basename(f)
            mtime = datetime.fromtimestamp(os.path.getmtime(f))
            
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                
                # Parse symbol and timestamp from filename
                # Format: {SYM}_validation_{YYYYMMDD_HHMMSS}.json
                parts = fname.replace('.json', '').split('_validation_')
                sym = parts[0] if parts else 'unknown'
                ts = parts[1] if len(parts) > 1 else 'unknown'
                
                findings['history'].append({
                    'file': f,
                    'symbol': sym,
                    'timestamp': ts,
                    'modified': mtime.isoformat(),
                    'archive_id': data.get('archive_id', ts),
                    'all_params': data.get('all_params', data.get('best_params', {})),
                    'train': data.get('train_results', data.get('train', {})),
                    'test': data.get('test_results', data.get('test', {})),
                    'raw_data': data,
                })
            except Exception as e:
                findings['history'].append({
                    'file': f, 'symbol': 'unknown', 'error': str(e)
                })
    
    # Scan for Optuna databases
    for f in glob.glob(os.path.join(results_dir, '*.db')):
        fsize = os.path.getsize(f)
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        findings['optuna_dbs'].append({
            'file': f,
            'modified': mtime.isoformat(),
            'size_mb': round(fsize / 1024 / 1024, 2),
        })
    
    # Also check parent directory for DBs
    parent_db = os.path.join(os.path.dirname(results_dir), '*.db')
    for f in glob.glob(parent_db):
        fsize = os.path.getsize(f)
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        findings['optuna_dbs'].append({
            'file': f,
            'modified': mtime.isoformat(),
            'size_mb': round(fsize / 1024 / 1024, 2),
        })
    
    return findings


def reconstruct_params_from_phases(symbol_data):
    """
    Merge best_params from all 4 phases into a single parameter dict.
    Later phases override earlier ones for any overlapping keys.
    """
    merged = {}
    
    for phase_num in sorted(symbol_data.get('phases', {}).keys()):
        phase = symbol_data['phases'][phase_num]
        if 'error' in phase:
            continue
        
        # Fixed params are carried forward from earlier phases
        fixed = phase.get('fixed_params', {})
        merged.update(fixed)
        
        # Best params are this phase's optimized values
        best = phase.get('best_params', {})
        merged.update(best)
    
    return merged


def get_best_available_params(symbol_data):
    """
    Get the best available parameters for a symbol, preferring
    FINAL_validation > reconstructed from phases > batch_status.
    Returns (params_dict, source_description).
    """
    # Priority 1: FINAL validation
    if symbol_data.get('validation') and 'error' not in symbol_data['validation']:
        params = symbol_data['validation'].get('all_params', {})
        if params:
            return params, 'FINAL_validation.json'
    
    # Priority 2: Reconstruct from phase files
    if symbol_data.get('phases'):
        params = reconstruct_params_from_phases(symbol_data)
        if params:
            n_phases = len([p for p in symbol_data['phases'].values() if 'error' not in p])
            return params, f'reconstructed from {n_phases} phase files'
    
    # Priority 3: Batch status (has params embedded in phase data)
    if symbol_data.get('batch_status') and 'error' not in symbol_data['batch_status']:
        status = symbol_data['batch_status']
        merged = {}
        for p_num in sorted(status.get('phases', {}).keys()):
            phase = status['phases'][p_num]
            if phase.get('params'):
                merged.update(phase['params'])
        if merged:
            return merged, 'batch_status.json'
    
    # Priority 4: Legacy optimal.json
    if symbol_data.get('optimal') and 'error' not in symbol_data.get('optimal', {}):
        data = symbol_data['optimal'].get('data', {})
        params = data.get('best_params', {})
        if params:
            return params, 'optimal.json (legacy)'
    
    return {}, 'no data found'


def get_best_available_metrics(symbol_data):
    """Get train/test metrics from the best available source."""
    # Priority: validation > last phase > batch_status
    if symbol_data.get('validation') and 'error' not in symbol_data['validation']:
        return {
            'train': symbol_data['validation'].get('train', {}),
            'test': symbol_data['validation'].get('test', {}),
            'source': 'FINAL_validation',
        }
    
    # Last completed phase
    phases = symbol_data.get('phases', {})
    for p in [4, 3, 2, 1]:
        if p in phases and 'error' not in phases[p]:
            return {
                'train': phases[p].get('train', {}),
                'test': phases[p].get('test', {}),
                'source': f'phase{p}',
            }
    
    return {'train': {}, 'test': {}, 'source': 'none'}


# ============================================================
# PRESET GENERATION FROM RECOVERED DATA
# ============================================================

def generate_pine_presets(symbols_params, output_file=None):
    """
    Generate Pine Script preset map.put() statements from recovered params.
    
    symbols_params: dict of {symbol: params_dict}
    output_file: path to write (default: pine_presets_orb.txt)
    """
    output_file = output_file or PRESET_OUTPUT
    
    lines = []
    lines.append(f"// ===== PRESET_BLOCK_START_7X9Q2 =====")
    lines.append(f"// AUTO-GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"// Source: History Recovery Module")
    lines.append(f"// Symbols: {', '.join(sorted(symbols_params.keys()))}")
    lines.append(f"// DO NOT EDIT BETWEEN MARKERS\n")
    lines.append(f"if barstate.isfirst")
    
    for sym, params in sorted(symbols_params.items()):
        lines.append(f"    // {sym} Optimizer Presets")
        
        for param_name, value in sorted(params.items()):
            if param_name in PINE_MAP_NAMES:
                map_name, val_type = PINE_MAP_NAMES[param_name]
                
                if val_type == 'bool':
                    pine_val = 'true' if value else 'false'
                elif val_type == 'float':
                    pine_val = f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)
                else:
                    pine_val = str(value)
                
                lines.append(f'    {map_name}.put("{sym}", {pine_val})')
        
        lines.append("")
    
    lines.append("// ===== PRESET_BLOCK_END_7X9Q2 =====")
    
    content = "\n".join(lines)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(content)
    
    return content


# ============================================================
# OPTUNA DB RECOVERY
# ============================================================

def scan_optuna_db(db_path):
    """Deep scan an Optuna database for recoverable study data."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return {'error': 'optuna not installed'}
    
    storage = f"sqlite:///{db_path}"
    results = []
    
    try:
        study_summaries = optuna.get_all_study_summaries(storage=storage)
        
        for summary in study_summaries:
            study = optuna.load_study(study_name=summary.study_name, storage=storage)
            
            best_trial = None
            try:
                best_trial = study.best_trial
            except ValueError:
                pass
            
            results.append({
                'study_name': summary.study_name,
                'n_trials': len(study.trials),
                'best_value': best_trial.value if best_trial else None,
                'best_params': best_trial.params if best_trial else {},
                'datetime_start': str(summary.datetime_start) if summary.datetime_start else None,
            })
    except Exception as e:
        return {'error': str(e)}
    
    return results


# ============================================================
# STREAMLIT TAB MODULE
# ============================================================

def render_history_tab(results_dir=None):
    """Render the History & Recovery tab in Streamlit."""
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        print("Streamlit not available - use CLI mode instead")
        return
    
    st.header("📜 Optimization History & Recovery")
    
    results_dir = results_dir or RESULTS_DIR
    
    # Scan button
    if st.button("🔍 Scan Results Directory", type="primary"):
        st.session_state['history_scan'] = scan_results_directory(results_dir)
    
    # Auto-scan on first load
    if 'history_scan' not in st.session_state:
        st.session_state['history_scan'] = scan_results_directory(results_dir)
    
    findings = st.session_state['history_scan']
    
    if 'error' in findings:
        st.error(findings['error'])
        return
    
    # ---- Summary Stats ----
    n_symbols = len(findings['symbols'])
    n_history = len(findings['history'])
    n_dbs = len(findings['optuna_dbs'])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbols Found", n_symbols)
    c2.metric("History Archives", n_history)
    c3.metric("Optuna DBs", n_dbs)
    c4.metric("Results Dir", os.path.basename(results_dir))
    
    # ---- Current Results Table ----
    st.subheader("📊 Current Results (Latest Run)")
    
    if findings['symbols']:
        rows = []
        for sym, data in sorted(findings['symbols'].items()):
            params, source = get_best_available_params(data)
            metrics = get_best_available_metrics(data)
            
            # Count completed phases
            n_phases = len([p for p in data.get('phases', {}).values() 
                          if isinstance(p, dict) and 'error' not in p])
            
            # Get status
            status = '❓'
            if data.get('batch_status') and isinstance(data['batch_status'], dict):
                state = data['batch_status'].get('state', '')
                if state == 'complete':
                    status = '✅ Complete'
                elif state == 'error':
                    status = '❌ Error'
                elif state == 'running':
                    status = '⏳ Running'
            elif data.get('validation') and isinstance(data['validation'], dict) and 'error' not in data['validation']:
                status = '✅ Validated'
            elif n_phases > 0:
                status = f'⚠️ {n_phases}/4 Phases'
            
            # Get timestamps
            modified = ''
            if data.get('validation') and isinstance(data['validation'], dict):
                modified = data['validation'].get('modified', '')[:16]
            elif data.get('phases'):
                latest = max(
                    (p.get('modified', '') for p in data['phases'].values() 
                     if isinstance(p, dict) and 'modified' in p),
                    default=''
                )
                modified = latest[:16]
            
            train_r = metrics.get('train', {}).get('total_r', 0)
            test_r = metrics.get('test', {}).get('total_r', 0)
            train_wr = metrics.get('train', {}).get('win_rate', 0)
            test_wr = metrics.get('test', {}).get('win_rate', 0)
            train_trades = metrics.get('train', {}).get('total_trades', 0)
            test_trades = metrics.get('test', {}).get('total_trades', 0)
            
            rows.append({
                'Symbol': sym,
                'Status': status,
                'Phases': f"{n_phases}/4",
                'Train R': f"{train_r:.1f}" if train_r else '—',
                'Test R': f"{test_r:.1f}" if test_r else '—',
                'Train WR': f"{train_wr:.0%}" if train_wr else '—',
                'Test WR': f"{test_wr:.0%}" if test_wr else '—',
                'Train #': train_trades or '—',
                'Test #': test_trades or '—',
                'Source': source,
                'Last Modified': modified,
            })
        
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # ---- Parameter Detail Expander ----
        st.subheader("🔧 Recovered Parameters")
        
        detail_sym = st.selectbox(
            "Select symbol for details",
            sorted(findings['symbols'].keys()),
            key="history_detail_sym"
        )
        
        if detail_sym:
            sym_data = findings['symbols'][detail_sym]
            params, source = get_best_available_params(sym_data)
            
            if params:
                st.success(f"**{detail_sym}** — {len(params)} parameters recovered from {source}")
                
                # Show params in two columns
                col1, col2 = st.columns(2)
                param_items = sorted(params.items())
                mid = len(param_items) // 2
                
                with col1:
                    for k, v in param_items[:mid]:
                        if isinstance(v, float):
                            st.text(f"{k}: {v:.4f}")
                        else:
                            st.text(f"{k}: {v}")
                
                with col2:
                    for k, v in param_items[mid:]:
                        if isinstance(v, float):
                            st.text(f"{k}: {v:.4f}")
                        else:
                            st.text(f"{k}: {v}")
                
                # Phase-by-phase breakdown
                if sym_data.get('phases'):
                    with st.expander("Phase-by-Phase Breakdown"):
                        for p_num in sorted(sym_data['phases'].keys()):
                            phase = sym_data['phases'][p_num]
                            if 'error' in phase:
                                st.warning(f"Phase {p_num}: Error - {phase['error']}")
                                continue
                            
                            p_name = PHASE_NAMES.get(p_num, f'Phase {p_num}')
                            score = phase.get('best_score', '—')
                            train_r = phase.get('train', {}).get('total_r', '—')
                            test_r = phase.get('test', {}).get('total_r', '—')
                            
                            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                            train_r_str = f"{train_r:.1f}" if isinstance(train_r, (int, float)) else str(train_r)
                            test_r_str = f"{test_r:.1f}" if isinstance(test_r, (int, float)) else str(test_r)
                            st.markdown(
                                f"**P{p_num}: {p_name}** — "
                                f"Score: {score_str} | "
                                f"Train: {train_r_str}R | "
                                f"Test: {test_r_str}R"
                            )
                            st.json(phase.get('best_params', {}))
            else:
                st.warning(f"No parameters found for {detail_sym}")
        
        # ---- Export Section ----
        st.subheader("📤 Export Presets from History")
        
        available_syms = [
            sym for sym, data in findings['symbols'].items()
            if get_best_available_params(data)[0]
        ]
        
        export_syms = st.multiselect(
            "Select symbols to export",
            available_syms,
            default=available_syms,
            key="history_export_syms"
        )
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("📋 Preview Preset Block", disabled=not export_syms):
                syms_params = {}
                for sym in export_syms:
                    params, _ = get_best_available_params(findings['symbols'][sym])
                    if params:
                        syms_params[sym] = params
                
                preset_text = generate_pine_presets(syms_params, output_file=None)
                st.code(preset_text, language='javascript')
        
        with export_col2:
            if st.button("💾 Write Preset File (F2 Ready)", type="primary", disabled=not export_syms):
                syms_params = {}
                for sym in export_syms:
                    params, _ = get_best_available_params(findings['symbols'][sym])
                    if params:
                        syms_params[sym] = params
                
                preset_text = generate_pine_presets(syms_params, output_file=PRESET_OUTPUT)
                st.success(f"✅ Written to {PRESET_OUTPUT} — Press F2 to inject!")
                st.caption(f"Contains presets for: {', '.join(sorted(syms_params.keys()))}")
    else:
        st.info("No result files found in the results directory.")
    
    # ---- History Archives ----
    st.subheader("🗂️ Archived Validation History")
    
    if findings['history']:
        hist_rows = []
        for h in findings['history']:
            if 'error' in h:
                continue
            
            train = h.get('train', {})
            test = h.get('test', {})
            
            hist_rows.append({
                'Symbol': h.get('symbol', '?'),
                'Timestamp': h.get('timestamp', '?'),
                'Train R': f"{train.get('total_r', 0):.1f}" if train.get('total_r') else '—',
                'Test R': f"{test.get('total_r', 0):.1f}" if test.get('total_r') else '—',
                'Train WR': f"{train.get('win_rate', 0):.0%}" if train.get('win_rate') else '—',
                'Test WR': f"{test.get('win_rate', 0):.0%}" if test.get('win_rate') else '—',
                'File': os.path.basename(h.get('file', '')),
            })
        
        if hist_rows:
            df_hist = pd.DataFrame(hist_rows)
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
            
            # Load from history
            hist_options = [
                f"{h['symbol']} @ {h['timestamp']}"
                for h in findings['history'] if 'error' not in h
            ]
            
            if hist_options:
                selected_hist = st.selectbox(
                    "Load parameters from archive",
                    hist_options,
                    key="history_load_select"
                )
                
                if st.button("📥 Load Archived Parameters"):
                    idx = hist_options.index(selected_hist)
                    arch = findings['history'][idx]
                    params = arch.get('all_params', {})
                    
                    if params:
                        st.success(f"Loaded {len(params)} params from {arch['file']}")
                        st.json(params)
                    else:
                        st.warning("No parameters found in this archive")
        else:
            st.info("No readable history archives found.")
    else:
        st.info("No history directory or archives found.")
    
    # ---- Optuna DB Recovery ----
    if findings['optuna_dbs']:
        st.subheader("🗄️ Optuna Database Recovery")
        st.caption("Deep scan Optuna SQLite databases for trial data")
        
        for db in findings['optuna_dbs']:
            st.text(f"📁 {os.path.basename(db['file'])} — {db['size_mb']} MB — {db['modified'][:16]}")
        
        if st.button("🔬 Deep Scan Optuna DBs (may take a moment)"):
            for db in findings['optuna_dbs']:
                with st.expander(f"📊 {os.path.basename(db['file'])}"):
                    studies = scan_optuna_db(db['file'])
                    
                    if isinstance(studies, dict) and 'error' in studies:
                        st.error(studies['error'])
                    elif studies:
                        for study in studies:
                            st.markdown(
                                f"**{study['study_name']}** — "
                                f"{study['n_trials']} trials | "
                                f"Best: {study['best_value']:.4f if study['best_value'] else '—'}"
                            )
                            if study['best_params']:
                                st.json(study['best_params'])
                    else:
                        st.info("No studies found in this database")


# ============================================================
# CLI MODE
# ============================================================

def cli_report():
    """Print a CLI recovery report."""
    print("=" * 70)
    print("ORB OPTIMIZER HISTORY & RECOVERY REPORT")
    print(f"Scan time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    findings = scan_results_directory()
    
    if 'error' in findings:
        print(f"\n❌ {findings['error']}")
        return
    
    print(f"\n📁 Results directory: {findings['results_dir']}")
    print(f"   Symbols found: {len(findings['symbols'])}")
    print(f"   History archives: {len(findings['history'])}")
    print(f"   Optuna databases: {len(findings['optuna_dbs'])}")
    
    # Symbol details
    if findings['symbols']:
        print(f"\n{'─' * 70}")
        print("SYMBOL RESULTS")
        print(f"{'─' * 70}")
        
        for sym, data in sorted(findings['symbols'].items()):
            params, source = get_best_available_params(data)
            metrics = get_best_available_metrics(data)
            
            n_phases = len([p for p in data.get('phases', {}).values()
                          if isinstance(p, dict) and 'error' not in p])
            
            train_r = metrics.get('train', {}).get('total_r', 0)
            test_r = metrics.get('test', {}).get('total_r', 0)
            
            status_icon = '✅' if data.get('validation') else f'⚠️ {n_phases}/4'
            
            print(f"\n  {status_icon} {sym}")
            print(f"     Phases: {n_phases}/4 | Source: {source}")
            print(f"     Train: {train_r:.1f}R | Test: {test_r:.1f}R")
            print(f"     Parameters recovered: {len(params)}")
            
            if params:
                for k, v in sorted(params.items()):
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    print(f"       {k}: {val_str}")
    
    # History
    if findings['history']:
        print(f"\n{'─' * 70}")
        print("ARCHIVED HISTORY")
        print(f"{'─' * 70}")
        
        for h in findings['history'][:10]:
            if 'error' in h:
                continue
            sym = h.get('symbol', '?')
            ts = h.get('timestamp', '?')
            train_r = h.get('train', {}).get('total_r', 0)
            test_r = h.get('test', {}).get('total_r', 0)
            print(f"  {sym} @ {ts} — Train: {train_r:.1f}R | Test: {test_r:.1f}R")
        
        if len(findings['history']) > 10:
            print(f"  ... and {len(findings['history']) - 10} more")
    
    # Optuna DBs
    if findings['optuna_dbs']:
        print(f"\n{'─' * 70}")
        print("OPTUNA DATABASES")
        print(f"{'─' * 70}")
        
        for db in findings['optuna_dbs']:
            print(f"  📁 {os.path.basename(db['file'])} — {db['size_mb']} MB")
    
    print(f"\n{'=' * 70}")
    print("RECOVERY OPTIONS:")
    print(f"{'=' * 70}")
    print("  python history_recovery.py --export AMD        # Export AMD presets")
    print("  python history_recovery.py --export-all        # Export all symbols")
    print("  python history_recovery.py --scan-optuna       # Deep scan Optuna DBs")
    print(f"{'=' * 70}")


def cli_export(symbols=None):
    """Export presets for specified symbols (or all)."""
    findings = scan_results_directory()
    
    if 'error' in findings:
        print(f"❌ {findings['error']}")
        return
    
    if symbols is None:
        symbols = list(findings['symbols'].keys())
    
    syms_params = {}
    for sym in symbols:
        if sym not in findings['symbols']:
            print(f"⚠️ {sym} not found in results")
            continue
        
        params, source = get_best_available_params(findings['symbols'][sym])
        if params:
            syms_params[sym] = params
            print(f"✅ {sym}: {len(params)} params from {source}")
        else:
            print(f"❌ {sym}: no recoverable parameters")
    
    if syms_params:
        preset_text = generate_pine_presets(syms_params)
        print(f"\n✅ Written to {PRESET_OUTPUT}")
        print(f"   Symbols: {', '.join(sorted(syms_params.keys()))}")
        print(f"   Press F2 in TradingView to inject!")
    else:
        print("\n❌ No parameters to export")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ORB Optimizer History & Recovery')
    parser.add_argument('--export', nargs='*', metavar='SYM',
                       help='Export presets for symbol(s)')
    parser.add_argument('--export-all', action='store_true',
                       help='Export presets for all validated symbols')
    parser.add_argument('--scan-optuna', action='store_true',
                       help='Deep scan Optuna databases')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Override results directory path')
    
    args = parser.parse_args()
    
    if args.results_dir:
        RESULTS_DIR = args.results_dir
        HISTORY_DIR = os.path.join(RESULTS_DIR, 'history')
    
    if args.export is not None:
        syms = args.export if args.export else None
        cli_export(syms)
    elif args.export_all:
        cli_export()
    elif args.scan_optuna:
        findings = scan_results_directory()
        if findings.get('optuna_dbs'):
            for db in findings['optuna_dbs']:
                print(f"\n📊 {os.path.basename(db['file'])}")
                studies = scan_optuna_db(db['file'])
                if isinstance(studies, dict) and 'error' in studies:
                    print(f"  ❌ {studies['error']}")
                else:
                    for s in studies:
                        print(f"  {s['study_name']}: {s['n_trials']} trials, best={s['best_value']}")
        else:
            print("No Optuna databases found")
    else:
        cli_report()
