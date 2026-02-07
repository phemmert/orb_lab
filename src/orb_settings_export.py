"""
ORB Settings Export v2
======================
Generates Pine Script preset blocks compatible with the AHK injector pipeline.

Merges two data sources:
  1. ORB Optimizer results (exit geometry, vol thresholds, booleans)
  2. Existing preset values (indicator tuning from HMM generator)

Output: C:\\hmm_trading\\pine_presets_orb.txt
Markers: // ===== PRESET_BLOCK_START_7X9Q2 =====
         // ===== PRESET_BLOCK_END_7X9Q2 =====

Pipeline:
  Streamlit Settings tab -> writes pine_presets_orb.txt -> F2 in TradingView -> done

Usage:
    from orb_settings_export import SettingsExporter, write_preset_file
    write_preset_file(['AMD', 'GOOGL'])
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================================
# FILE PATHS
# =====================================================================

PRESET_FILE_PATH = r"C:\hmm_trading\pine_presets_orb.txt"
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

START_MARKER = "// ===== PRESET_BLOCK_START_7X9Q2 ====="
END_MARKER = "// ===== PRESET_BLOCK_END_7X9Q2 ====="


# =====================================================================
# PARAMETER MAPPING: Optimizer param -> Pine Script preset infrastructure
# =====================================================================

# --- SWITCH-BASED params (int lengths, vary by symbol) ---
# These use:  paramEff = switch / syminfo.ticker == "XXX" => value
# The Pine input variable name is the fallback.
SWITCH_PARAMS = {
    # From HMM generator (indicator tuning)
    'ssl_baseline_length':  {'eff': 'sslBaselineLengthEff', 'input': 'sslBaselineLength'},
    'wae_fast_ema':         {'eff': 'waeFastEMAEff',        'input': 'waeFastEMA'},
    'wae_slow_ema':         {'eff': 'waeSlowEMAEff',        'input': 'waeSlowEMA'},
    # From optimizer (integer params that vary by symbol)
    'ema_period':           {'eff': 'emaPeriodEff',          'input': 'emaPeriod'},
    'ema_confirmation_bars':{'eff': 'emaConfirmationBarsEff','input': 'emaConfirmationBars'},
    'swing_lookback':       {'eff': 'swingLookbackEff',      'input': 'swingLookback'},
    'min_confluence':       {'eff': 'minConfluenceScoreEff', 'input': 'minConfluenceScore'},
}

# --- UNIFORM params (same value across all preset symbols) ---
# These use:  paramEff = _isPresetSymbol ? value : inputParam
UNIFORM_PARAMS = {
    # From HMM generator
    'ssl_length':           {'eff': 'sslLengthEff',          'input': 'sslLength'},
    'exit_length':          {'eff': 'exitLengthEff',          'input': 'exitLength'},
    'wae_bb_length':        {'eff': 'waeBBLengthEff',         'input': 'waeBBLength'},
    'qqe_rsi1_length':      {'eff': 'qqeRSI1LengthEff',      'input': 'qqeRSI1Length'},
    'qqe_rsi1_smoothing':   {'eff': 'qqeRSI1SmoothingEff',   'input': 'qqeRSI1Smoothing'},
    'qqe_rsi2_length':      {'eff': 'qqeRSI2LengthEff',      'input': 'qqeRSI2Length'},
    'qqe_rsi2_smoothing':   {'eff': 'qqeRSI2SmoothingEff',   'input': 'qqeRSI2Smoothing'},
    'qqe_bb_length':        {'eff': 'qqeBBLengthEff',         'input': 'qqeBBLength'},
    'vol_lookback_bars':    {'eff': 'volLookbackBarsEff',     'input': 'volLookbackBars'},
}

# --- MAP-BASED params (float/bool/int via map.put()) ---
# Each entry: internal_key -> { map_name, pine_type }
MAP_PARAMS = {
    # Volatility thresholds (existing maps - from optimizer Phase 2)
    'low_vol_threshold':     {'map': 'mapVFLow',           'type': 'float'},
    'high_vol_threshold':    {'map': 'mapVFHigh',          'type': 'float'},
    'extreme_vol_threshold': {'map': 'mapVFExtreme',       'type': 'float'},

    # Exit geometry (NEW maps - from optimizer Phase 1)
    'break_even_rr':          {'map': 'mapBERR',            'type': 'float'},
    'profit_target_rr':       {'map': 'mapProfitTarget',    'type': 'float'},
    'trailing_stop_distance': {'map': 'mapTrailDist',       'type': 'float'},
    'ema_tighten_zone':       {'map': 'mapEmaTightenZone',  'type': 'float'},
    'tightened_trail_distance':{'map': 'mapTightTrailDist', 'type': 'float'},
    'atr_stop_mult':          {'map': 'mapATRStopMult',     'type': 'float'},
    'min_stop_atr':           {'map': 'mapMinStopATR',      'type': 'float'},
    'vwap_stop_distance':     {'map': 'mapVWAPStopDist',    'type': 'float'},
    'min_acceptable_rr':      {'map': 'mapMinRR',           'type': 'float'},
    'max_target_atr':         {'map': 'mapMaxTargetATR',    'type': 'float'},
    'breakout_threshold_mult':{'map': 'mapBreakoutThresh',  'type': 'float'},
    'min_body_strength':      {'map': 'mapMinBodyStr',      'type': 'float'},

    # Volatility behavior (from optimizer Phase 2)
    'skip_low_vol_c_grades':  {'map': 'mapSkipLowVolC',    'type': 'bool'},
    'low_vol_min_rr':         {'map': 'mapLowVolMinRR',    'type': 'float'},

    # Booleans (from optimizer Phase 4)
    'use_ema_exit':           {'map': 'mapEmaExit',         'type': 'bool'},
    'use_adaptive_be':        {'map': 'mapAdaptiveBE',      'type': 'bool'},
    'use_aggressive_trailing':{'map': 'mapAggressiveTrail', 'type': 'bool'},
    'enable_confluence':      {'map': 'mapConfluence',      'type': 'bool'},

    # Existing HMM-sourced maps (preserved when merging)
    'be_low':                 {'map': 'mapBELow',           'type': 'float'},
    'be_normal':              {'map': 'mapBENormal',        'type': 'float'},
    'be_high':                {'map': 'mapBEHigh',          'type': 'float'},
    'ssl_exit':               {'map': 'mapSSLexit',         'type': 'bool'},
    'vol_thresh':             {'map': 'mapVolThresh',       'type': 'float'},
    'wae_sens':               {'map': 'mapWAEsens',         'type': 'int'},
    'vwap_weight':            {'map': 'mapVWAPweight',      'type': 'float'},
    'hod_weight':             {'map': 'mapHODweight',       'type': 'float'},
    'cam_weight':             {'map': 'mapCamWeight',       'type': 'float'},
    'weekly_weight':          {'map': 'mapWeeklyWeight',    'type': 'float'},
    'prox_touch':             {'map': 'mapProxTouch',       'type': 'float'},
    'prox_near':              {'map': 'mapProxNear',        'type': 'float'},
    'orb_extension':          {'map': 'mapORBextension',    'type': 'float'},
    'first_hour_mult':        {'map': 'mapFirstHourMult',   'type': 'float'},
    'power_hour_mult':        {'map': 'mapPowerHourMult',   'type': 'float'},
}


def minutes_to_time(minutes):
    """Convert minutes-from-midnight to HH:MM AM/PM string."""
    h = minutes // 60
    m = minutes % 60
    period = "AM" if h < 12 else "PM"
    display_h = h if h <= 12 else h - 12
    return f"{display_h}:{m:02d} {period}"


def _fmt_pine_value(val, pine_type='float'):
    """Format a Python value for Pine Script."""
    if isinstance(val, bool) or pine_type == 'bool':
        return str(bool(val)).lower()
    elif pine_type == 'int' or (isinstance(val, int) and not isinstance(val, bool)):
        return str(int(val))
    else:
        s = f"{float(val):.4f}".rstrip('0')
        if s.endswith('.'):
            s += '0'
        return s


# =====================================================================
# PARSE EXISTING PRESET FILE
# =====================================================================

def parse_existing_presets(filepath=None):
    """
    Parse the existing pine_presets_orb.txt to extract current values.
    Returns dict with keys: switch, uniform, map
    """
    filepath = filepath or PRESET_FILE_PATH
    if not os.path.exists(filepath):
        return {'switch': {}, 'uniform': {}, 'map': {}}

    content = open(filepath, 'r').read()
    result = {'switch': {}, 'uniform': {}, 'map': {}}

    # Parse map-based: mapName.put("SYMBOL", value)
    map_pattern = re.compile(r'(\w+)\.put\("(\w+)",\s*(.+?)\)')
    for match in map_pattern.finditer(content):
        map_name, sym, raw_val = match.groups()
        raw_val = raw_val.strip()
        if raw_val == 'true':
            val = True
        elif raw_val == 'false':
            val = False
        elif '.' in raw_val:
            val = float(raw_val)
        else:
            try:
                val = int(raw_val)
            except ValueError:
                val = raw_val
        if sym not in result['map']:
            result['map'][sym] = {}
        result['map'][sym][map_name] = val

    # Parse switch-based: syminfo.ticker == "XXX" => value
    switch_blocks = re.findall(
        r'(\w+Eff)\s*=\s*switch\n(.*?)(?=\n\w|\n//|\n\n)',
        content, re.DOTALL
    )
    for eff_name, block in switch_blocks:
        ticker_pattern = re.compile(r'syminfo\.ticker\s*==\s*"(\w+)"\s*=>\s*(.+)')
        for match in ticker_pattern.finditer(block):
            sym, raw_val = match.groups()
            raw_val = raw_val.strip()
            if '.' in raw_val:
                val = float(raw_val)
            else:
                try:
                    val = int(raw_val)
                except ValueError:
                    val = raw_val
            if sym not in result['switch']:
                result['switch'][sym] = {}
            result['switch'][sym][eff_name] = val

    # Parse uniform: paramEff = _isPresetSymbol ? value : fallback
    uniform_pattern = re.compile(r'(\w+Eff)\s*=\s*_isPresetSymbol\s*\?\s*(.+?)\s*:\s*\w+')
    for match in uniform_pattern.finditer(content):
        eff_name, raw_val = match.groups()
        raw_val = raw_val.strip()
        if '.' in raw_val:
            val = float(raw_val)
        else:
            try:
                val = int(raw_val)
            except ValueError:
                val = raw_val
        result['uniform'][eff_name] = val

    return result


# =====================================================================
# LOAD OPTIMIZER RESULTS
# =====================================================================

def load_optimizer_params(symbol, results_dir=None):
    """Load the full validated parameter set for a symbol."""
    results_dir = results_dir or RESULTS_DIR
    path = os.path.join(results_dir, f'{symbol}_FINAL_validation.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('all_params', {})


# =====================================================================
# GENERATE PRESET BLOCK
# =====================================================================

def generate_preset_block(
    symbols,
    results_dir=None,
    existing_preset_path=None,
):
    """
    Generate a complete Pine Script preset block that merges:
      1. Optimizer results (exit geometry, vol thresholds, booleans)
      2. Existing HMM-sourced values (indicator tuning, weights)

    Returns the block text including markers.
    """
    results_dir = results_dir or RESULTS_DIR
    existing = parse_existing_presets(existing_preset_path)

    # Load optimizer results
    optimizer_data = {}
    for sym in symbols:
        params = load_optimizer_params(sym, results_dir)
        if params:
            optimizer_data[sym] = params

    if not optimizer_data:
        return f"{START_MARKER}\n// No validated symbols found\n{END_MARKER}"

    all_symbols = sorted(optimizer_data.keys())

    # ---- Merge switch params ----
    merged_switch = {}
    for internal_key, info in SWITCH_PARAMS.items():
        eff_name = info['eff']
        merged_switch[internal_key] = {}
        for sym in all_symbols:
            if internal_key in optimizer_data.get(sym, {}):
                merged_switch[internal_key][sym] = optimizer_data[sym][internal_key]
            elif sym in existing.get('switch', {}) and eff_name in existing['switch'][sym]:
                merged_switch[internal_key][sym] = existing['switch'][sym][eff_name]

    # ---- Merge uniform params ----
    merged_uniform = {}
    for internal_key, info in UNIFORM_PARAMS.items():
        eff_name = info['eff']
        if eff_name in existing.get('uniform', {}):
            merged_uniform[internal_key] = existing['uniform'][eff_name]

    # ---- Merge map params ----
    merged_map = {}
    for internal_key, info in MAP_PARAMS.items():
        map_name = info['map']
        merged_map[internal_key] = {}
        for sym in all_symbols:
            if internal_key in optimizer_data.get(sym, {}):
                merged_map[internal_key][sym] = optimizer_data[sym][internal_key]
            elif sym in existing.get('map', {}) and map_name in existing['map'][sym]:
                merged_map[internal_key][sym] = existing['map'][sym][map_name]

    # ---- Generate output ----
    lines = []
    lines.append(START_MARKER)
    lines.append(f"// AUTO-GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"// Generator: ORB Lab Phased Optimizer v3 + HMM Presets (merged)")
    lines.append(f"// Script Type: ORB (5-Min Opening Range)")
    lines.append(f"// Symbols: {', '.join(all_symbols)}")
    lines.append(f"// DO NOT EDIT BETWEEN MARKERS")
    lines.append("")

    # ---- SWITCH-BASED PARAMETERS ----
    switch_varying = {}
    switch_uniform_extra = {}

    for internal_key, sym_vals in merged_switch.items():
        if not sym_vals:
            continue
        info = SWITCH_PARAMS[internal_key]
        unique_vals = set(str(v) for v in sym_vals.values())
        if len(unique_vals) > 1 and len(sym_vals) > 1:
            switch_varying[internal_key] = sym_vals
        elif len(unique_vals) == 1 and len(sym_vals) >= 1:
            switch_uniform_extra[internal_key] = list(sym_vals.values())[0]

    if switch_varying:
        lines.append("// ========== SWITCH-BASED LENGTH PARAMETERS ==========")
        lines.append("")
        for internal_key, sym_vals in sorted(switch_varying.items()):
            info = SWITCH_PARAMS[internal_key]
            lines.append(f"{info['eff']} = switch")
            lines.append(f"    not useSymbolPresets => {info['input']}")
            for sym in all_symbols:
                if sym in sym_vals:
                    val = sym_vals[sym]
                    lines.append(f'    syminfo.ticker == "{sym}" => {_fmt_pine_value(val, "int")}')
            lines.append(f"    => {info['input']}")
            lines.append("")

    # ---- UNIFORM PARAMETERS ----
    has_uniform = merged_uniform or switch_uniform_extra

    if has_uniform:
        lines.append("// ========== UNIFORM PARAMETERS (SCOPE-OPTIMIZED) ==========")
        lines.append("// These values are identical across all preset symbols")
        lines.append("// Using ternaries instead of switches saves ~27 scopes")
        lines.append("")

        sym_check = " or ".join(f'syminfo.ticker == "{sym}"' for sym in all_symbols)
        lines.append(f"_isPresetSymbol = useSymbolPresets and ({sym_check})")
        lines.append("")

        for internal_key, val in sorted(merged_uniform.items()):
            info = UNIFORM_PARAMS[internal_key]
            lines.append(
                f"{info['eff']} = _isPresetSymbol ? "
                f"{_fmt_pine_value(val, 'int')} : {info['input']}"
            )

        for internal_key, val in sorted(switch_uniform_extra.items()):
            info = SWITCH_PARAMS[internal_key]
            lines.append(
                f"{info['eff']} = _isPresetSymbol ? "
                f"{_fmt_pine_value(val, 'int')} : {info['input']}"
            )

        lines.append("")

    # ---- MAP-BASED PARAMETERS ----
    active_maps = {}
    for internal_key, sym_vals in merged_map.items():
        if sym_vals:
            for sym, val in sym_vals.items():
                if sym not in active_maps:
                    active_maps[sym] = {}
                active_maps[sym][internal_key] = val

    if active_maps:
        lines.append("// ========== MAP-BASED MULTIPLIER PARAMETERS ==========")
        lines.append("")
        lines.append("if barstate.isfirst")

        # Define output order
        hmm_order = [
            'low_vol_threshold', 'high_vol_threshold', 'extreme_vol_threshold',
            'be_low', 'be_normal', 'be_high',
            'ssl_exit', 'vol_thresh', 'wae_sens',
            'vwap_weight', 'hod_weight', 'cam_weight', 'weekly_weight',
            'prox_touch', 'prox_near',
            'orb_extension', 'first_hour_mult', 'power_hour_mult',
        ]
        optimizer_order = [
            'break_even_rr', 'profit_target_rr', 'trailing_stop_distance',
            'ema_tighten_zone', 'tightened_trail_distance',
            'atr_stop_mult', 'min_stop_atr', 'vwap_stop_distance',
            'min_acceptable_rr', 'max_target_atr',
            'breakout_threshold_mult', 'min_body_strength',
            'skip_low_vol_c_grades', 'low_vol_min_rr',
            'use_ema_exit', 'use_adaptive_be', 'use_aggressive_trailing',
            'enable_confluence',
        ]

        for sym in all_symbols:
            if sym not in active_maps:
                continue

            sym_data = active_maps[sym]
            lines.append(f"    // {sym} Presets")

            # HMM-sourced maps
            for key in hmm_order:
                if key in sym_data:
                    info = MAP_PARAMS[key]
                    val = _fmt_pine_value(sym_data[key], info['type'])
                    lines.append(f'    {info["map"]}.put("{sym}", {val})')

            # Optimizer-sourced maps
            has_optimizer = any(key in sym_data for key in optimizer_order)
            if has_optimizer:
                lines.append(f"    // {sym} Optimizer (ORB Lab v3)")
                for key in optimizer_order:
                    if key in sym_data:
                        info = MAP_PARAMS[key]
                        val = _fmt_pine_value(sym_data[key], info['type'])
                        lines.append(f'    {info["map"]}.put("{sym}", {val})')

            lines.append("")

    lines.append(END_MARKER)
    return "\n".join(lines)


# =====================================================================
# WRITE PRESET FILE (for AHK injection)
# =====================================================================

def write_preset_file(
    symbols,
    results_dir=None,
    output_path=None,
    existing_preset_path=None,
):
    """
    Generate preset block and write to the AHK injection file.
    Returns dict with: path, symbols, block_text
    """
    output_path = output_path or PRESET_FILE_PATH
    existing_preset_path = existing_preset_path or output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    block = generate_preset_block(
        symbols,
        results_dir=results_dir,
        existing_preset_path=existing_preset_path,
    )

    with open(output_path, 'w') as f:
        f.write(block)

    return {
        'path': output_path,
        'symbols': symbols,
        'block_text': block,
    }


# =====================================================================
# CHEAT SHEET
# =====================================================================

CHEAT_SHEET_MAP = {
    'use_break_even':           {'label': '1. Move to Break Even when target reached',   'group': 'Exit Strategy Sequence'},
    'break_even_rr':            {'label': '   Break Even at R:R',                        'group': 'Exit Strategy Sequence'},
    'use_adaptive_be':          {'label': '   Adaptive BE (Vol-Based)',                  'group': 'Exit Strategy Sequence'},
    'use_ema_exit':             {'label': '2. Use EMA Exit after Break Even',            'group': 'Exit Strategy Sequence'},
    'ema_period':               {'label': '   EMA Period',                               'group': 'Exit Strategy Sequence'},
    'ema_confirmation_bars':    {'label': '   Confirmation Bars',                        'group': 'Exit Strategy Sequence'},
    'use_trailing_stop':        {'label': '3. Use Trailing Stop at Profit Target',       'group': 'Exit Strategy Sequence'},
    'profit_target_rr':         {'label': '   Profit Target R:R',                        'group': 'Exit Strategy Sequence'},
    'trailing_stop_distance':   {'label': '   Trailing Stop Distance (ATR Mult)',        'group': 'Exit Strategy Sequence'},
    'use_aggressive_trailing':  {'label': 'Use Aggressive Trailing Near EMA',            'group': 'Advanced Trailing Settings'},
    'ema_tighten_zone':         {'label': 'EMA Zone for Tighter Trail (% of ATR)',       'group': 'Advanced Trailing Settings'},
    'tightened_trail_distance': {'label': 'Tightened Trail Distance (ATR Mult)',          'group': 'Advanced Trailing Settings'},
    'breakout_threshold_mult':  {'label': 'Breakout Threshold Multiplier (ATR)',         'group': 'Breakout Filter Settings'},
    'min_body_strength':        {'label': 'Minimum Body Strength Ratio',                 'group': 'Breakout Filter Settings'},
    'enable_confluence':        {'label': 'Enable Confluence Scoring',                   'group': 'Confluence Settings'},
    'min_confluence':           {'label': 'Minimum Confluence Score',                    'group': 'Confluence Settings'},
    'atr_stop_mult':            {'label': 'ATR Stop Loss Multiplier',                    'group': 'Risk Management'},
    'min_stop_atr':             {'label': 'Min Stop Distance (ATR fraction)',            'group': 'Stop Loss Selection'},
    'swing_lookback':           {'label': 'Swing Lookback Bars',                         'group': 'Stop Loss Selection'},
    'vwap_stop_distance':       {'label': 'VWAP Stop Distance (ATR multiplier)',         'group': 'Stop Loss Selection'},
    'low_vol_threshold':        {'label': 'Low Volatility Threshold',                    'group': 'Adaptive Volatility Engine'},
    'high_vol_threshold':       {'label': 'High Volatility Threshold',                   'group': 'Adaptive Volatility Engine'},
    'extreme_vol_threshold':    {'label': 'Extreme Volatility Threshold',                'group': 'Adaptive Volatility Engine'},
    'max_target_atr':           {'label': 'Max Target Distance (ATR)',                   'group': 'Adaptive Volatility Engine'},
    'min_acceptable_rr':        {'label': 'Minimum Acceptable R:R',                      'group': 'Adaptive Volatility Engine'},
    'skip_low_vol_c_grades':    {'label': 'Skip C-Grade Trades in LOW Vol',              'group': 'Adaptive Volatility Engine'},
    'low_vol_min_rr':           {'label': 'LOW Vol Minimum R:R',                         'group': 'Adaptive Volatility Engine'},
    'trading_start_minutes':    {'label': 'Trading Start',                               'group': 'Trading Window'},
    'trading_end_minutes':      {'label': 'Trading End',                                 'group': 'Trading Window'},
}


class SettingsExporter:
    """Generate cheat sheets and preset data for a single symbol."""

    def __init__(self, symbol='AMD', results_dir=None):
        self.symbol = symbol
        self.results_dir = results_dir or RESULTS_DIR
        self.params = self._load_params()

    def _load_params(self):
        path = os.path.join(self.results_dir, f'{self.symbol}_FINAL_validation.json')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No validation file for {self.symbol}. Run full validation first."
            )
        with open(path, 'r') as f:
            data = json.load(f)
        return data['all_params']

    def get_cheat_sheet_grouped(self):
        """Return params grouped by Pine Script settings section."""
        grouped = {}
        for param, value in self.params.items():
            if param in CHEAT_SHEET_MAP:
                info = CHEAT_SHEET_MAP[param]
                group = info['group']
                if group not in grouped:
                    grouped[group] = []
                if isinstance(value, bool):
                    display = str(value).upper()
                elif isinstance(value, float):
                    display = f"{value:.2f}".rstrip('0').rstrip('.')
                    if '.' not in display:
                        display += '.0'
                else:
                    display = str(value)
                grouped[group].append({
                    'param': param,
                    'label': info['label'],
                    'value': value,
                    'display': display,
                })
        return grouped

    def get_cheat_sheet_text(self):
        """Generate plain-text cheat sheet for manual entry."""
        grouped = self.get_cheat_sheet_grouped()
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"  ORB SETTINGS CHEAT SHEET: {self.symbol}")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"  Source: ORB Lab Phased Optimizer v3")
        lines.append(f"{'='*60}")
        lines.append("")
        lines.append("  Open TradingView Strategy Settings and update:")
        lines.append("")
        for group, params in grouped.items():
            lines.append(f"  --- {group} ---")
            for p in params:
                if p['param'] in ('trading_start_minutes', 'trading_end_minutes'):
                    lines.append(f"    {p['label']}: {minutes_to_time(p['value'])}")
                else:
                    lines.append(f"    {p['label']}: {p['display']}")
            lines.append("")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def print_cheat_sheet(self):
        print(self.get_cheat_sheet_text())


# =====================================================================
# PINE SCRIPT SCAFFOLDING (one-time additions)
# =====================================================================

def generate_map_declarations():
    """
    Generate Pine Script map declarations for NEW optimizer params.
    One-time paste into Pine Script BEFORE the preset block markers.
    """
    optimizer_keys = [
        'break_even_rr', 'profit_target_rr', 'trailing_stop_distance',
        'ema_tighten_zone', 'tightened_trail_distance',
        'atr_stop_mult', 'min_stop_atr', 'vwap_stop_distance',
        'min_acceptable_rr', 'max_target_atr',
        'breakout_threshold_mult', 'min_body_strength',
        'skip_low_vol_c_grades', 'low_vol_min_rr',
        'use_ema_exit', 'use_adaptive_be', 'use_aggressive_trailing',
        'enable_confluence',
    ]
    lines = []
    lines.append("// ========== NEW MAP DECLARATIONS (ORB Lab Optimizer) ==========")
    lines.append("// Add these alongside the existing map declarations")
    lines.append("")
    for key in optimizer_keys:
        if key in MAP_PARAMS:
            info = MAP_PARAMS[key]
            m = info['map']
            t = info['type']
            if t == 'float':
                lines.append(f'var map<string, float>  {m:20s} = map.new<string, float>()')
            elif t == 'int':
                lines.append(f'var map<string, int>    {m:20s} = map.new<string, int>()')
            elif t == 'bool':
                lines.append(f'var map<string, bool>   {m:20s} = map.new<string, bool>()')
    return "\n".join(lines)


def generate_resolution_lines():
    """
    Generate Pine Script resolution lines for NEW optimizer params.
    One-time paste AFTER the preset block markers.
    """
    resolutions = [
        ('break_even_rr',          'mapBERR',            'breakEvenRR',                'breakEvenRREff'),
        ('profit_target_rr',       'mapProfitTarget',    'profitTargetRR',             'profitTargetRREff'),
        ('trailing_stop_distance', 'mapTrailDist',       'trailingStopDistance',       'trailingStopDistanceEff'),
        ('ema_tighten_zone',       'mapEmaTightenZone',  'emaTightenZone',             'emaTightenZoneEff'),
        ('tightened_trail_distance','mapTightTrailDist', 'tightenedTrailDistance',     'tightenedTrailDistanceEff'),
        ('atr_stop_mult',          'mapATRStopMult',     'atrMultiplier',              'atrMultiplierEff'),
        ('min_stop_atr',           'mapMinStopATR',      'minStopATR',                 'minStopATREff'),
        ('vwap_stop_distance',     'mapVWAPStopDist',    'vwapStopDistance',           'vwapStopDistanceEff'),
        ('min_acceptable_rr',      'mapMinRR',           'minAcceptableRR',            'minAcceptableRREff'),
        ('max_target_atr',         'mapMaxTargetATR',    'maxTargetATR',               'maxTargetATREff'),
        ('breakout_threshold_mult','mapBreakoutThresh',  'breakoutThresholdMultiplier','breakoutThresholdMultiplierEff'),
        ('min_body_strength',      'mapMinBodyStr',      'minBodyStrengthRatio',       'minBodyStrengthRatioEff'),
        ('skip_low_vol_c_grades',  'mapSkipLowVolC',     'skipLowVolCGrades',          'skipLowVolCGradesEff'),
        ('low_vol_min_rr',         'mapLowVolMinRR',     'lowVolMinRR',                'lowVolMinRREff'),
        ('use_ema_exit',           'mapEmaExit',         'useEmaCrossExit',            'useEmaCrossExitEff'),
        ('use_adaptive_be',        'mapAdaptiveBE',      'useAdaptiveBE',              'useAdaptiveBEEff'),
        ('use_aggressive_trailing','mapAggressiveTrail', 'aggressiveTrailing',         'aggressiveTrailingEff'),
        ('enable_confluence',      'mapConfluence',      'enableConfluenceScore',      'enableConfluenceScoreEff'),
    ]
    lines = []
    lines.append("// ========== NEW MAP RESOLUTION (ORB Lab Optimizer) ==========")
    lines.append("// Add these alongside the existing resolution lines")
    lines.append("")
    for _, map_name, input_name, eff_name in resolutions:
        lines.append(
            f'{eff_name} = hasPreset and {map_name}.contains(presetSymbolSimple) '
            f'? {map_name}.get(presetSymbolSimple) : {input_name}'
        )
    return "\n".join(lines)


# =====================================================================
# MASTER CONFIG
# =====================================================================

def generate_master_config(symbols=None, results_dir=None):
    """Generate master JSON config with all symbol settings."""
    results_dir = results_dir or RESULTS_DIR
    if symbols is None:
        symbols = []
        for f in os.listdir(results_dir):
            if f.endswith('_FINAL_validation.json'):
                symbols.append(f.replace('_FINAL_validation.json', ''))
    master = {
        'generated': datetime.now().isoformat(),
        'generator': 'ORB Lab Phased Optimizer v3',
        'symbols': {},
    }
    for sym in sorted(symbols):
        params = load_optimizer_params(sym, results_dir)
        if params:
            master['symbols'][sym] = params
    output_path = os.path.join(results_dir, 'MASTER_CONFIG.json')
    with open(output_path, 'w') as f:
        json.dump(master, f, indent=2)
    return master


# =====================================================================
# CLI
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ORB Settings Export v2')
    parser.add_argument('--symbol', type=str, default='AMD',
                        help='Symbol for cheat sheet')
    parser.add_argument('--write-presets', nargs='+',
                        help='Generate and write preset file for symbols')
    parser.add_argument('--scaffold', action='store_true',
                        help='Generate Pine Script scaffolding')
    parser.add_argument('--master', action='store_true',
                        help='Generate master config JSON')
    args = parser.parse_args()

    if args.write_presets:
        result = write_preset_file(args.write_presets)
        print(f"Written to: {result['path']}")
        print(f"Symbols: {', '.join(result['symbols'])}")
        print()
        print(result['block_text'])
    elif args.scaffold:
        print(generate_map_declarations())
        print()
        print(generate_resolution_lines())
    elif args.master:
        generate_master_config()
    else:
        exporter = SettingsExporter(symbol=args.symbol)
        exporter.print_cheat_sheet()
