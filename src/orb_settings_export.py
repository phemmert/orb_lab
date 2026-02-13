"""
ORB Settings Export v3 - Two-Block Architecture
=================================================
Generates Pine Script preset block for OPTIMIZER params only.

This writes to a SEPARATE file with SEPARATE markers from the HMM presets.
The two blocks are completely independent and can be injected in any order.

Block 1 (_7X9Q2):  HMM presets  -> pine_presets_orb.txt     -> F2
Block 2 (_OPT_4R8W): Optimizer  -> pine_presets_orb_optimizer.txt -> F7

Output: C:\\hmm_trading\\pine_presets_orb_optimizer.txt
Markers: // ===== PRESET_BLOCK_START_OPT_4R8W =====
         // ===== PRESET_BLOCK_END_OPT_4R8W =====

Usage:
    from orb_settings_export import write_optimizer_preset_file
    write_optimizer_preset_file(['AMD', 'PLTR', 'TSLA'])
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

OPTIMIZER_PRESET_FILE = r"C:\hmm_trading\pine_presets_orb_optimizer.txt"
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

OPT_START_MARKER = "// ===== PRESET_BLOCK_START_OPT_4R8W ====="
OPT_END_MARKER = "// ===== PRESET_BLOCK_END_OPT_4R8W ====="

# Legacy merged file (kept for backward compat if needed)
LEGACY_PRESET_FILE = r"C:\hmm_trading\pine_presets_orb.txt"
LEGACY_START_MARKER = "// ===== PRESET_BLOCK_START_7X9Q2 ====="
LEGACY_END_MARKER = "// ===== PRESET_BLOCK_END_7X9Q2 ====="
# Backward compat alias for Streamlit import
PRESET_FILE_PATH = OPTIMIZER_PRESET_FILE


# =====================================================================
# OPTIMIZER-OWNED PARAMETERS
# =====================================================================

# --- Eff variables owned by the optimizer (switch-based, vary by symbol) ---
# These use := reassignment since defaults are declared above both blocks.
OPTIMIZER_EFF_PARAMS = {
    'ema_period':            {'eff': 'emaPeriodEff',           'input': 'emaPeriod'},
    'ema_confirmation_bars': {'eff': 'emaConfirmationBarsEff', 'input': 'emaConfirmationBars'},
    'swing_lookback':        {'eff': 'swingLookbackEff',       'input': 'swingLookback'},
    'min_confluence':        {'eff': 'minConfluenceScoreEff',  'input': 'minConfluenceScore'},
}

# --- MAP-BASED params owned by the optimizer ---
OPTIMIZER_MAP_PARAMS = {
    # Volatility thresholds (Phase 2)
    'low_vol_threshold':      {'map': 'mapVFLow',           'type': 'float'},
    'high_vol_threshold':     {'map': 'mapVFHigh',          'type': 'float'},
    'extreme_vol_threshold':  {'map': 'mapVFExtreme',       'type': 'float'},

    # Exit geometry (Phase 1)
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

    # Volatility behavior (Phase 2)
    'skip_low_vol_c_grades':  {'map': 'mapSkipLowVolC',    'type': 'bool'},
    'low_vol_min_rr':         {'map': 'mapLowVolMinRR',    'type': 'float'},

    # Booleans (Phase 4)
    'use_ema_exit':           {'map': 'mapEmaExit',         'type': 'bool'},
    'use_adaptive_be':        {'map': 'mapAdaptiveBE',      'type': 'bool'},
    'use_aggressive_trailing':{'map': 'mapAggressiveTrail', 'type': 'bool'},
    'enable_confluence':      {'map': 'mapConfluence',      'type': 'bool'},
}

# Output order for map.put() calls
MAP_OUTPUT_ORDER = [
    'low_vol_threshold', 'high_vol_threshold', 'extreme_vol_threshold',
    'break_even_rr', 'profit_target_rr', 'trailing_stop_distance',
    'ema_tighten_zone', 'tightened_trail_distance',
    'atr_stop_mult', 'min_stop_atr', 'vwap_stop_distance',
    'min_acceptable_rr', 'max_target_atr',
    'breakout_threshold_mult', 'min_body_strength',
    'skip_low_vol_c_grades', 'low_vol_min_rr',
    'use_ema_exit', 'use_adaptive_be', 'use_aggressive_trailing',
    'enable_confluence',
]


# =====================================================================
# FORMATTING
# =====================================================================

def _fmt_pine_value(val, pine_type='float'):
    """Format a Python value for Pine Script."""
    if isinstance(val, bool) or pine_type == 'bool':
        return str(bool(val)).lower()
    elif isinstance(val, str):
        return val
    elif pine_type == 'int' or (isinstance(val, int) and not isinstance(val, bool)):
        return str(int(val))
    else:
        s = f"{float(val):.4f}".rstrip('0')
        if s.endswith('.'):
            s += '0'
        return s


def minutes_to_time(minutes):
    """Convert minutes-from-midnight to HH:MM AM/PM string."""
    h = minutes // 60
    m = minutes % 60
    period = "AM" if h < 12 else "PM"
    display_h = h if h <= 12 else h - 12
    return f"{display_h}:{m:02d} {period}"


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
# GENERATE OPTIMIZER PRESET BLOCK
# =====================================================================

def generate_optimizer_block(symbols, results_dir=None):
    """
    Generate a Pine Script preset block containing ONLY optimizer params.
    
    This block targets the _OPT_4R8W markers and contains:
      1. Optimizer-owned Eff variable overrides (switch statements)
      2. Optimizer map.put() calls
    
    No HMM data, no merge logic.
    """
    results_dir = results_dir or RESULTS_DIR

    # Load optimizer results
    optimizer_data = {}
    for sym in symbols:
        params = load_optimizer_params(sym, results_dir)
        if params:
            optimizer_data[sym] = params

    if not optimizer_data:
        return (
            f"{OPT_START_MARKER}\n"
            f"// No validated optimizer symbols found\n"
            f"if barstate.isfirst\n"
            f"    noop = 0\n"
            f"{OPT_END_MARKER}"
        )

    all_symbols = sorted(optimizer_data.keys())
    lines = []
    lines.append(OPT_START_MARKER)
    lines.append(f"// AUTO-GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"// Generator: ORB Lab Optimizer v3 (standalone)")
    lines.append(f"// Symbols: {', '.join(all_symbols)}")
    lines.append(f"// DO NOT EDIT BETWEEN MARKERS")
    lines.append("")

    # ---- OPTIMIZER Eff VARIABLES (always switch-based, := reassignment) ----
    # We always use switch statements here (not ternaries) to avoid
    # depending on _isPresetSymbol which is owned by the HMM block.
    eff_data = {}
    for internal_key, info in OPTIMIZER_EFF_PARAMS.items():
        eff_data[internal_key] = {}
        for sym in all_symbols:
            if internal_key in optimizer_data.get(sym, {}):
                eff_data[internal_key][sym] = optimizer_data[sym][internal_key]

    has_eff = any(sv for sv in eff_data.values())
    if has_eff:
        lines.append("// ========== OPTIMIZER LENGTH PARAMETERS ==========")
        lines.append("")
        for internal_key, sym_vals in sorted(eff_data.items()):
            if not sym_vals:
                continue
            info = OPTIMIZER_EFF_PARAMS[internal_key]
            lines.append(f"{info['eff']} := switch")
            lines.append(f"    not useSymbolPresets => {info['input']}")
            for sym in all_symbols:
                if sym in sym_vals:
                    val = sym_vals[sym]
                    lines.append(f'    syminfo.ticker == "{sym}" => {_fmt_pine_value(val, "int")}')
            lines.append(f"    => {info['input']}")
            lines.append("")

    # ---- OPTIMIZER MAP.PUT() CALLS ----
    lines.append("// ========== OPTIMIZER MAP PARAMETERS ==========")
    lines.append("")
    lines.append("if barstate.isfirst")

    for sym in all_symbols:
        sym_params = optimizer_data[sym]
        lines.append(f"    // {sym} Optimizer Presets")

        for key in MAP_OUTPUT_ORDER:
            if key in sym_params and key in OPTIMIZER_MAP_PARAMS:
                info = OPTIMIZER_MAP_PARAMS[key]
                val = _fmt_pine_value(sym_params[key], info['type'])
                lines.append(f'    {info["map"]}.put("{sym}", {val})')

        lines.append("")

    lines.append(OPT_END_MARKER)
    return "\n".join(lines)


def write_optimizer_preset_file(symbols, results_dir=None, output_path=None):
    """
    Generate optimizer preset block and write to the F7 injection file.
    """
    output_path = output_path or OPTIMIZER_PRESET_FILE
    block_text = generate_optimizer_block(symbols, results_dir)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(block_text)

    # Count symbols in output
    sym_count = block_text.count("Optimizer Presets")

    return {
        'path': output_path,
        'symbols': [s for s in sorted(symbols) if load_optimizer_params(s, results_dir)],
        'block_text': block_text,
        'sym_count': sym_count,
    }


# =====================================================================
# SETTINGS EXPORTER (cheat sheet for manual entry)
# =====================================================================

CHEAT_SHEET_PARAMS = {
    'break_even_rr':           {'label': 'Break-Even R:R',          'group': 'Exit Geometry',     'fmt': '.2f'},
    'profit_target_rr':        {'label': 'Profit Target R:R',       'group': 'Exit Geometry',     'fmt': '.2f'},
    'trailing_stop_distance':  {'label': 'Trailing Stop Distance',  'group': 'Exit Geometry',     'fmt': '.2f'},
    'ema_tighten_zone':        {'label': 'EMA Tighten Zone',        'group': 'Exit Geometry',     'fmt': '.2f'},
    'tightened_trail_distance':{'label': 'Tightened Trail Dist',    'group': 'Exit Geometry',     'fmt': '.2f'},
    'atr_stop_mult':           {'label': 'ATR Stop Multiplier',     'group': 'Stop Placement',    'fmt': '.2f'},
    'min_stop_atr':            {'label': 'Min Stop (ATR)',          'group': 'Stop Placement',    'fmt': '.2f'},
    'vwap_stop_distance':      {'label': 'VWAP Stop Distance',      'group': 'Stop Placement',    'fmt': '.2f'},
    'min_acceptable_rr':       {'label': 'Min Acceptable R:R',      'group': 'Target/Quality',    'fmt': '.2f'},
    'max_target_atr':          {'label': 'Max Target (ATR)',         'group': 'Target/Quality',    'fmt': '.2f'},
    'breakout_threshold_mult': {'label': 'Breakout Threshold',      'group': 'Entry Filters',     'fmt': '.3f'},
    'min_body_strength':       {'label': 'Min Body Strength',       'group': 'Entry Filters',     'fmt': '.2f'},
    'low_vol_threshold':       {'label': 'Low Vol Threshold',       'group': 'Volatility',        'fmt': '.2f'},
    'high_vol_threshold':      {'label': 'High Vol Threshold',      'group': 'Volatility',        'fmt': '.2f'},
    'extreme_vol_threshold':   {'label': 'Extreme Vol Threshold',   'group': 'Volatility',        'fmt': '.2f'},
    'skip_low_vol_c_grades':   {'label': 'Skip Low-Vol C Grades',   'group': 'Volatility',        'fmt': 'bool'},
    'low_vol_min_rr':          {'label': 'Low Vol Min R:R',          'group': 'Volatility',        'fmt': '.2f'},
    'ema_period':              {'label': 'EMA Period',               'group': 'Indicator Lengths', 'fmt': 'd'},
    'ema_confirmation_bars':   {'label': 'EMA Confirm Bars',         'group': 'Indicator Lengths', 'fmt': 'd'},
    'swing_lookback':          {'label': 'Swing Lookback',           'group': 'Indicator Lengths', 'fmt': 'd'},
    'min_confluence':          {'label': 'Min Confluence Score',     'group': 'Indicator Lengths', 'fmt': 'd'},
    'use_ema_exit':            {'label': 'Use EMA Cross Exit',       'group': 'Behavior Flags',   'fmt': 'bool'},
    'use_adaptive_be':         {'label': 'Adaptive Break-Even',      'group': 'Behavior Flags',   'fmt': 'bool'},
    'use_aggressive_trailing': {'label': 'Aggressive Trailing',      'group': 'Behavior Flags',   'fmt': 'bool'},
    'enable_confluence':       {'label': 'Enable Confluence',        'group': 'Behavior Flags',   'fmt': 'bool'},
}


class SettingsExporter:
    """Generate cheat sheets for manual settings entry."""

    def __init__(self, symbol, results_dir=None):
        self.symbol = symbol.upper()
        self.results_dir = results_dir or RESULTS_DIR
        self.params = load_optimizer_params(self.symbol, self.results_dir) or {}

    def get_cheat_sheet_grouped(self):
        from collections import OrderedDict
        grouped = OrderedDict()
        for param, info in CHEAT_SHEET_PARAMS.items():
            group = info['group']
            if group not in grouped:
                grouped[group] = []
            value = self.params.get(param)
            if value is None:
                continue
            fmt = info['fmt']
            if fmt == 'bool':
                display = str(bool(value))
            elif fmt == 'd':
                display = str(int(value))
            else:
                display = f"{float(value):{fmt}}"
            grouped[group].append({
                'param': param,
                'label': info['label'],
                'value': value,
                'display': display,
            })
        return grouped

    def get_cheat_sheet_text(self):
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
                lines.append(f"    {p['label']}: {p['display']}")
            lines.append("")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def print_cheat_sheet(self):
        print(self.get_cheat_sheet_text())


# =====================================================================
# SCAFFOLDING (one-time Pine additions)
# =====================================================================

def generate_map_declarations():
    """Generate Pine Script map declarations for optimizer params."""
    lines = []
    lines.append("// ========== MAP DECLARATIONS (ORB Lab Optimizer) ==========")
    lines.append("")
    for key in MAP_OUTPUT_ORDER:
        if key in OPTIMIZER_MAP_PARAMS:
            info = OPTIMIZER_MAP_PARAMS[key]
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
    """Generate Pine Script resolution lines for optimizer map params."""
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
    lines.append("// ========== MAP RESOLUTION (ORB Lab Optimizer) ==========")
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
# BACKWARD COMPATIBILITY (Streamlit Settings tab imports these names)
# =====================================================================

def write_preset_file(symbols, results_dir=None, output_path=None, existing_preset_path=None):
    """Legacy wrapper - now writes optimizer-only block to new file."""
    return write_optimizer_preset_file(symbols, results_dir, output_path)

def generate_preset_block(symbols, results_dir=None, existing_preset_path=None):
    """Legacy wrapper - now generates optimizer-only block."""
    return generate_optimizer_block(symbols, results_dir)


# =====================================================================
# CLI
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ORB Settings Export v3 (Two-Block)')
    parser.add_argument('--symbol', type=str, default='AMD',
                        help='Symbol for cheat sheet')
    parser.add_argument('--write-presets', nargs='+',
                        help='Generate optimizer preset file for symbols')
    parser.add_argument('--scaffold', action='store_true',
                        help='Generate Pine Script scaffolding')
    parser.add_argument('--master', action='store_true',
                        help='Generate master config JSON')
    args = parser.parse_args()

    if args.write_presets:
        result = write_optimizer_preset_file(args.write_presets)
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
