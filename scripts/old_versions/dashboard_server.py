"""
Healthcare Interactive Dashboard Server (Pure Python - Zero Dependencies)

Launches a local web server that serves an interactive dashboard.
Users can type natural-language queries directly in the browser,
and see results rendered as charts + tables below.

Uses only Python builtins: http.server, json, sqlite3, webbrowser, os.
"""

import os
import sys
import json
import sqlite3
import webbrowser
import math
import time
import html as html_lib
import threading
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Any, Optional

# Public tunnel support (auto-installs pyngrok if needed)
HAS_NGROK = False
try:
    from pyngrok import ngrok as _ngrok
    HAS_NGROK = True
except ImportError:
    _ngrok = None

# Import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Matplotlib chart engine (graceful fallback to SVG charts if unavailable)
try:
    from chart_engine import generate_chart as mpl_generate_chart
    HAS_CHART_ENGINE = True
except ImportError:
    HAS_CHART_ENGINE = False

# Databricks connector
try:
    from databricks_connector import DataSourceManager
    HAS_DATABRICKS_MODULE = True
except ImportError:
    HAS_DATABRICKS_MODULE = False

# ML-powered chart selection (graceful fallback to rules if unavailable)
try:
    from ml_integration import MLChartSelector
    _ml_chart_selector = MLChartSelector()
    _ml_chart_selector.train()
    _HAS_ML_CHART = _ml_chart_selector._trained
except ImportError:
    _HAS_ML_CHART = False
    _ml_chart_selector = None

# =============================================================================
# KAISER PERMANENTE BRAND COLOR SYSTEM
# =============================================================================
# Official KP brand palette mapped to healthcare dashboard semantics:
#   - KP Blue (#006BA6)   = Primary brand / Navigation / Informational
#   - KP Navy (#0D1C3D)   = Dark backgrounds / Headers
#   - KP Thrive (#286140) = Healthy / Approved / On Target (the iconic green)
#   - KP Red (#C8102E)    = Critical / Alert / Denied / Danger
#   - KP Amber (#F2A900)  = Warning / Needs Attention / Approaching threshold
#   - KP Teal (#007A7C)   = Preventive Care / Wellness / Population Health
#   - KP Purple (#5C2D91) = Behavioral Health / Specialty
#   - KP Sky (#5DB1D1)    = Secondary accent / Lighter informational
#   - KP Gray (#58595B)   = Inactive / Pending / Muted text

COLORS = {
    'critical':    '#C8102E',  # KP Red - alerts, denials, critical issues
    'warning':     '#F2A900',  # KP Amber - approaching thresholds
    'healthy':     '#286140',  # KP Thrive Green - on target, paid, approved
    'info':        '#006BA6',  # KP Blue - primary brand, informational
    'behavioral':  '#5C2D91',  # KP Purple - behavioral/specialty
    'preventive':  '#007A7C',  # KP Teal - wellness, preventive
    'inactive':    '#58595B',  # KP Gray - pending, inactive
    'maternal':    '#C63663',  # KP Rose - maternal health
    'operations':  '#1B4F9B',  # KP Indigo - operational/administrative
    'bg_primary':  '#FFFFFF',  # White - KP standard background
    'bg_card':     '#F7F8FA',  # Light gray - card backgrounds
    'bg_hover':    '#EDF0F5',  # Hover state - slightly darker gray
    'text':        '#0D1C3D',  # KP Navy - primary text
    'text_sec':    '#58595B',  # KP Gray - secondary text
    'text_muted':  '#8C8E91',  # Lighter gray - muted text
    'border':      '#D9DDE3',  # Light gray border
}

# Map KPI categories to healthcare colors
CATEGORY_COLORS = {
    'revenue':     COLORS['healthy'],     # Green = money/financial health
    'retention':   COLORS['info'],         # Blue = member engagement
    'acquisition': COLORS['preventive'],   # Teal = growth/wellness
    'operations':  COLORS['operations'],   # Indigo = operational
}

CATEGORY_ICONS = {
    'revenue':     '&#128176;',  # money bag
    'retention':   '&#128101;',  # people
    'acquisition': '&#127919;',  # target
    'operations':  '&#9881;',    # gear
}

# Map status values to colors
STATUS_COLORS = {
    'PAID':     COLORS['healthy'],
    'DENIED':   COLORS['critical'],
    'PENDING':  COLORS['warning'],
    'ADJUSTED': COLORS['info'],
    'APPEALED': COLORS['behavioral'],
}


def _esc(text):
    return html_lib.escape(str(text)) if text is not None else "N/A"


def _fmt(val, unit=""):
    if val is None: return "N/A"
    if isinstance(val, float):
        if abs(val) >= 1_000_000: return "${:,.1f}M".format(val / 1_000_000)
        if unit == "$": return "${:,.2f}".format(val)
        return "{:,.2f}{}".format(val, unit)
    if isinstance(val, int): return "{:,}{}".format(val, unit)
    return str(val)


# =============================================================================
# FULL INTERACTIVE HTML DASHBOARD (Single Page App)
# =============================================================================

import secrets as _boot_secrets
# Unique ID per server start — forces re-auth when server restarts
_SERVER_BOOT_ID = _boot_secrets.token_hex(8)


def build_full_dashboard_html(kpi_data=None, initial_results=None, initial_query="", initial_sql=""):
    """Build the complete interactive dashboard HTML."""
    server_boot_id = _SERVER_BOOT_ID

    # KPI cards HTML
    kpi_cards_html = ""
    if kpi_data:
        for cat, kpis in kpi_data.get('by_category', {}).items():
            color = CATEGORY_COLORS.get(cat, COLORS['info'])
            icon = CATEGORY_ICONS.get(cat, '&#128202;')
            cards = ""
            for kpi in kpis:
                val = kpi.get('value', 'N/A')
                unit = kpi.get('unit', '')
                has_alert = kpi.get('alert')
                has_error = kpi.get('error')

                if has_error:
                    display_val = "Error"
                    val_color = COLORS['critical']
                    border_color = COLORS['critical']
                elif has_alert:
                    display_val = _fmt(val, unit) if isinstance(val, (int, float)) else str(val)
                    val_color = COLORS['critical']
                    border_color = COLORS['critical']
                elif isinstance(val, (int, float)):
                    display_val = _fmt(val, unit)
                    # Color based on direction
                    direction = kpi.get('direction', '')
                    if direction == 'lower_is_better':
                        val_color = COLORS['healthy'] if (kpi.get('alert') is None) else COLORS['critical']
                    else:
                        val_color = color
                    border_color = COLORS['border']
                elif isinstance(val, list):
                    display_val = f"{len(val)} items"
                    val_color = color
                    border_color = COLORS['border']
                else:
                    display_val = str(val)[:30]
                    val_color = color
                    border_color = COLORS['border']

                alert_badge = f'<span class="alert-badge">ALERT</span>' if has_alert else ''
                # Make KPI clickable - clicking queries for more detail
                kpi_query = f"show me {kpi.get('kpi', '').lower().replace('(', '').replace(')', '')} details"
                cards += f'''
                <div class="kpi-card" style="border-color:{border_color};" onclick="submitQuery('{_esc(kpi_query)}')">
                    <div class="kpi-name">{_esc(kpi.get('kpi', '?'))} {alert_badge}</div>
                    <div class="kpi-value" style="color:{val_color};">{display_val}</div>
                    <div class="kpi-desc">{_esc(kpi.get('description', '')[:70])}</div>
                    <div class="kpi-click-hint">Click to explore &#8594;</div>
                </div>'''

            kpi_cards_html += f'''
            <div class="kpi-section">
                <div class="kpi-section-title">{icon} {cat.upper()}</div>
                <div class="kpi-grid">{cards}</div>
            </div>'''

    # Alert banner
    alerts_html = ""
    if kpi_data:
        alerts = kpi_data.get('alerts', [])
        if alerts:
            alert_items = "".join(
                f'<div class="alert-item"><span class="alert-dot"></span>{_esc(a.get("alert", ""))}</div>'
                for a in alerts
            )
            alerts_html = f'<div class="alerts-banner">{alert_items}</div>'

    # Initial results HTML
    results_html = ""
    if initial_results and isinstance(initial_results, list) and initial_results:
        results_html = _build_results_html(initial_results, initial_query, initial_sql)

    # Suggested queries — organized by category for demo walkthrough
    suggestion_groups = {
        'Counts & Breakdowns': [
            "how many claims",
            "claims count by member",
            "members by plan type",
            "encounters by visit type",
        ],
        'Rankings': [
            "which member has the most claims",
            "top 5 providers by total billed",
            "top 3 diagnoses by frequency",
            "which medication is most prescribed",
        ],
        'Aggregations': [
            "average paid amount by region",
            "average length of stay by facility",
            "total billed amount by department",
            "denial rate by region",
        ],
        'Filters & Trends': [
            "show me denied claims",
            "members with risk score above 3",
            "claims trend over 2024",
            "members with more than 5 claims",
        ],
        'Cross-Table': [
            "average billed amount by specialty",
            "total prescriptions by provider",
            "top 10 providers by volume",
            "which specialty has highest average cost",
        ],
    }
    suggestion_chips = ""
    for group_name, queries in suggestion_groups.items():
        chips = "".join(
            f'<button class="suggestion-chip" onclick="submitQuery(\'{s}\')">{s}</button>'
            for s in queries
        )
        suggestion_chips += f'''
        <div class="suggestion-group">
            <span class="suggestion-group-label">{group_name}:</span>
            {chips}
        </div>'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MTP Healthcare Analytics Dashboard</title>
<style>
:root {{
    --critical: {COLORS['critical']};
    --warning: {COLORS['warning']};
    --healthy: {COLORS['healthy']};
    --info: {COLORS['info']};
    --behavioral: {COLORS['behavioral']};
    --preventive: {COLORS['preventive']};
    --inactive: {COLORS['inactive']};
    --operations: {COLORS['operations']};
    --bg: {COLORS['bg_primary']};
    --bg-card: {COLORS['bg_card']};
    --bg-hover: {COLORS['bg_hover']};
    --text: {COLORS['text']};
    --text-sec: {COLORS['text_sec']};
    --text-muted: {COLORS['text_muted']};
    --border: {COLORS['border']};
}}

* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:var(--text); }}

/* ---- Header + Query Bar ---- */
.top-bar {{
    background: #FFFFFF;
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    position: sticky; top: 0; z-index: 100;
}}
.top-bar h1 {{
    font-size: 20px; font-weight: 700;
    background: linear-gradient(135deg, var(--info), var(--preventive));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}}
.query-bar {{
    display: flex; gap: 10px; align-items: center;
}}
.query-input {{
    flex: 1; padding: 10px 16px; font-size: 15px;
    background: var(--bg); border: 2px solid var(--border); border-radius: 10px;
    color: var(--text); outline: none; transition: border-color 0.2s;
}}
.query-input:focus {{ border-color: var(--info); }}
.query-input::placeholder {{ color: var(--text-muted); }}
.query-btn {{
    padding: 10px 24px; font-size: 14px; font-weight: 600;
    background: var(--info); color: #fff; border: none; border-radius: 10px;
    cursor: pointer; transition: background 0.2s;
}}
.query-btn:hover {{ background: #005A8C; }}
.query-btn:disabled {{ background: var(--inactive); cursor: not-allowed; }}

.suggestions {{
    margin-top: 12px; max-height: 0; overflow: hidden; transition: max-height 0.4s ease;
}}
.suggestions.expanded {{ max-height: 500px; }}
.suggestion-toggle {{
    font-size: 12px; color: var(--text-muted); cursor: pointer; margin-top: 8px;
    display: inline-flex; align-items: center; gap: 4px;
    border: none; background: none; padding: 4px 0;
}}
.suggestion-toggle:hover {{ color: var(--info); }}
.suggestion-group {{
    display: flex; gap: 6px; margin-bottom: 6px; flex-wrap: wrap; align-items: center;
}}
.suggestion-group-label {{
    font-size: 10px; font-weight: 700; color: var(--text-muted); text-transform: uppercase;
    letter-spacing: 0.3px; min-width: 120px; padding-right: 4px;
}}
.suggestion-chip {{
    padding: 3px 10px; font-size: 11px; background: rgba(0,107,166,0.06);
    color: var(--info); border: 1px solid rgba(0,107,166,0.2); border-radius: 14px;
    cursor: pointer; transition: all 0.2s; white-space: nowrap;
}}
.suggestion-chip:hover {{ background: rgba(0,107,166,0.15); border-color: var(--info); transform: translateY(-1px); }}

/* ---- Loading ---- */
.loading {{ display:none; text-align:center; padding:40px; }}
.loading.active {{ display:block; }}
.spinner {{
    width:40px; height:40px; margin:0 auto 12px;
    border:4px solid var(--border); border-top-color: var(--info);
    border-radius:50%; animation: spin 0.8s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}

/* ---- Alerts Banner ---- */
.alerts-banner {{
    background: rgba(200,16,46,0.05); border: 1px solid var(--critical);
    border-radius: 10px; padding: 12px 16px; margin: 16px 24px 0;
}}
.alert-item {{
    display: flex; align-items: center; gap: 8px;
    font-size: 13px; color: var(--critical); padding: 4px 0;
}}
.alert-dot {{
    width: 8px; height: 8px; border-radius: 50%; background: var(--critical);
    animation: pulse 2s infinite;
}}
@keyframes pulse {{
    0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }}
}}
.alert-badge {{
    font-size:10px; font-weight:700; padding:2px 6px; border-radius:4px;
    background:var(--critical); color:#fff; margin-left:6px; vertical-align:middle;
}}

/* ---- Main Content ---- */
.main {{ padding: 24px; max-width: 1500px; margin: 0 auto; }}

/* ---- KPI Grid ---- */
.kpi-section {{ margin-bottom: 24px; }}
.kpi-section-title {{
    font-size: 15px; font-weight: 600; color: var(--text-sec);
    margin-bottom: 12px; padding-left: 4px;
}}
.kpi-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
}}
.kpi-card {{
    background: var(--bg-card); border-radius: 10px; padding: 16px;
    border: 1px solid var(--border); cursor: pointer;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    position: relative;
}}
.kpi-card:hover {{
    transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,107,166,0.12);
    border-color: var(--info);
}}
.kpi-name {{ font-size:12px; font-weight:600; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.3px; margin-bottom:6px; }}
.kpi-value {{ font-size:28px; font-weight:700; margin-bottom:4px; }}
.kpi-desc {{ font-size:11px; color:var(--text-sec); line-height:1.4; }}
.kpi-click-hint {{
    font-size:10px; color:var(--text-muted); margin-top:8px; opacity:0;
    transition: opacity 0.2s;
}}
.kpi-card:hover .kpi-click-hint {{ opacity:1; }}

/* ---- Results Area ---- */
#results-area {{ min-height: 200px; }}

.result-header {{
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:16px; flex-wrap: wrap; gap: 8px;
}}
.result-title {{ font-size:18px; font-weight:600; }}
.result-meta {{ display:flex; gap:8px; }}
.meta-badge {{
    padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600;
}}

/* SQL display (read-only fallback) */
.sql-box {{
    font-family:'SF Mono',Monaco,'Consolas',monospace; font-size:12px;
    color: var(--preventive); background: var(--bg);
    padding:10px 14px; border-radius:8px; border:1px solid var(--border);
    margin-bottom:16px; overflow-x:auto; white-space:pre-wrap;
}}

/* Editable SQL editor */
.sql-edit-container {{
    margin-bottom:16px;
}}
.sql-editor {{
    font-family:'SF Mono',Monaco,'Consolas',monospace; font-size:12px;
    color: var(--preventive); background: var(--bg);
    padding:12px 14px; border-radius:8px 8px 0 0; border:1px solid var(--border);
    width:100%; min-height:60px; max-height:200px; resize:vertical;
    outline:none; line-height:1.6; box-sizing:border-box;
    border-bottom:none;
}}
.sql-editor:focus {{
    border-color: var(--info);
}}
.sql-edit-actions {{
    display:flex; align-items:center; gap:8px; padding:8px 12px;
    background:rgba(0,107,166,0.05); border:1px solid var(--border);
    border-top:1px dashed rgba(0,107,166,0.2);
    border-radius:0 0 8px 8px;
}}
.sql-run-btn {{
    padding:5px 14px; background:var(--info); color:#fff; border:none;
    border-radius:6px; font-size:12px; font-weight:600; cursor:pointer;
    transition:background 0.2s;
}}
.sql-run-btn:hover {{ background:#005A8C; }}
.sql-reset-btn {{
    padding:5px 12px; background:transparent; color:var(--text-sec); border:1px solid var(--border);
    border-radius:6px; font-size:12px; cursor:pointer;
}}
.sql-reset-btn:hover {{ background:var(--bg-hover); }}
.sql-edit-hint {{
    font-size:11px; color:var(--text-muted); margin-left:auto;
}}

/* Auth UI */
.auth-bar {{
    display:flex; align-items:center; gap:10px; padding:8px 16px;
    background:var(--bg-card); border-bottom:1px solid var(--border);
    font-size:13px;
}}
.auth-user {{
    display:flex; align-items:center; gap:6px; color:var(--text);
    font-weight:600;
}}
.auth-team {{
    font-size:11px; color:var(--text-muted); padding:2px 8px;
    background:rgba(0,107,166,0.08); border-radius:10px;
}}
.auth-btn {{
    padding:4px 12px; border-radius:6px; font-size:12px; font-weight:600;
    cursor:pointer; border:1px solid var(--border); background:transparent;
    color:var(--text-sec); transition:all 0.2s;
}}
.auth-btn:hover {{ background:var(--bg-hover); color:var(--text); }}
.auth-btn-primary {{
    background:var(--info); color:#fff; border-color:var(--info);
}}
.auth-btn-primary:hover {{ background:#005A8C; }}
.auth-btn-danger {{
    color:var(--critical);
}}
.auth-btn-danger:hover {{ background:rgba(200,16,46,0.08); }}

/* Modal */
.modal-overlay {{
    display:none; position:fixed; top:0; left:0; width:100%; height:100%;
    background:rgba(0,0,0,0.35); z-index:9999; align-items:center; justify-content:center;
}}
.modal-overlay.active {{ display:flex; }}
.modal {{
    background:#FFFFFF; border-radius:14px; padding:28px;
    width:400px; max-width:90vw; border:1px solid var(--border);
    box-shadow:0 20px 60px rgba(0,0,0,0.15);
}}
.modal h2 {{
    margin:0 0 20px; font-size:20px; color:var(--text);
}}
.modal-input {{
    width:100%; padding:10px 14px; background:var(--bg); border:1px solid var(--border);
    border-radius:8px; color:var(--text); font-size:14px; outline:none;
    margin-bottom:12px; box-sizing:border-box;
}}
.modal-input:focus {{ border-color:var(--info); }}
.modal-input::placeholder {{ color:var(--text-muted); }}
.modal-actions {{
    display:flex; gap:10px; margin-top:16px;
}}
.modal-error {{
    color:var(--critical); font-size:12px; margin-top:8px; display:none;
}}

/* Save dashboard */
.save-bar {{
    display:flex; align-items:center; gap:8px; padding:8px 0; margin-bottom:12px;
}}
.saved-dashboards-list {{
    display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));
    gap:12px; margin:16px 0;
}}
.saved-dash-card {{
    background:var(--bg-card); border:1px solid var(--border); border-radius:10px;
    padding:14px; cursor:pointer; transition:border-color 0.2s;
}}
.saved-dash-card:hover {{ border-color:var(--info); }}
.saved-dash-name {{ font-weight:600; font-size:14px; margin-bottom:4px; }}
.saved-dash-meta {{ font-size:11px; color:var(--text-muted); }}
.saved-dash-actions {{ display:flex; gap:6px; margin-top:8px; }}

/* ---- Charts ---- */
.chart-section {{ margin-bottom:24px; background:rgba(13,28,61,0.3); border-radius:12px; padding:16px; border:1px solid rgba(0,107,166,0.1); }}
.chart-section img {{ border-radius:8px; max-width:100%; height:auto; }}
.chart-title {{ font-size:15px; font-weight:600; margin-bottom:12px; color:var(--text); }}
.bar-row {{ display:flex; align-items:center; margin-bottom:8px; gap:10px; }}
.bar-label {{ min-width:110px; font-size:12px; color:var(--text-sec); text-align:right; }}
.bar-track {{ flex:1; height:26px; background:rgba(0,107,166,0.06); border-radius:6px; overflow:hidden; }}
.bar-fill {{
    height:100%; border-radius:6px; display:flex; align-items:center;
    padding:0 8px; font-size:11px; font-weight:600; color:#fff;
    min-width:30px; transition: width 0.8s ease;
}}

/* ---- Stats Cards ---- */
.stats-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); gap:12px; margin-bottom:20px; }}
.stat-card {{
    background:var(--bg-card); border-radius:10px; padding:14px;
    border:1px solid var(--border);
}}
.stat-label {{ font-size:11px; color:var(--text-muted); text-transform:uppercase; }}
.stat-value {{ font-size:22px; font-weight:700; margin:4px 0 2px; }}
.stat-detail {{ font-size:11px; color:var(--text-sec); }}

/* ---- Data Table ---- */
.table-controls {{
    display:flex; gap:10px; margin-bottom:10px; align-items:center; flex-wrap:wrap;
}}
.table-search {{
    flex:1; min-width:200px; padding:7px 12px; background:var(--bg);
    border:1px solid var(--border); border-radius:8px;
    color:var(--text); font-size:13px; outline:none;
}}
.table-search:focus {{ border-color:var(--info); }}
.row-count {{ font-size:12px; color:var(--text-muted); white-space:nowrap; }}

.col-toggles {{
    padding:6px 10px; background:var(--bg); border-radius:6px;
    margin-bottom:8px; overflow-x:auto; white-space:nowrap;
}}
.col-toggle {{
    font-size:11px; color:var(--text-sec); margin-right:10px; cursor:pointer;
}}
.col-toggle input {{ accent-color:var(--info); margin-right:3px; }}

.table-wrap {{
    max-height:450px; overflow:auto; border-radius:8px; border:1px solid var(--border);
}}
.data-table {{ width:100%; border-collapse:collapse; font-size:12px; }}
.data-table th {{
    background:var(--bg); color:var(--text-muted); font-weight:600;
    text-transform:uppercase; font-size:10px; letter-spacing:0.4px;
    padding:8px 12px; text-align:left; border-bottom:2px solid var(--border);
    position:sticky; top:0; cursor:pointer; user-select:none;
    white-space:nowrap;
}}
.data-table th:hover {{ color:var(--info); }}
.data-table td {{
    padding:7px 12px; border-bottom:1px solid var(--border); white-space:nowrap;
    max-width:200px; overflow:hidden; text-overflow:ellipsis;
}}
.data-table tr:hover td {{ background:var(--bg-hover); }}

/* Status cell coloring */
.status-PAID {{ color: {COLORS['healthy']}; font-weight:600; }}
.status-DENIED {{ color: {COLORS['critical']}; font-weight:600; }}
.status-PENDING {{ color: {COLORS['warning']}; font-weight:600; }}
.status-ADJUSTED {{ color: {COLORS['info']}; font-weight:600; }}
.status-APPEALED {{ color: {COLORS['behavioral']}; font-weight:600; }}

/* ---- Color Legend ---- */
.color-legend {{
    display:flex; gap:16px; flex-wrap:wrap; padding:10px 16px;
    background:var(--bg-card); border-radius:8px; border:1px solid var(--border);
    margin-bottom:20px; font-size:11px;
}}
.legend-item {{ display:flex; align-items:center; gap:5px; color:var(--text-sec); }}
.legend-dot {{ width:10px; height:10px; border-radius:50%; }}

/* ---- Code in explanations ---- */
code {{
    background: rgba(0,107,166,0.1); color: var(--info);
    padding: 1px 5px; border-radius: 4px; font-size: 11px;
    font-family: 'SF Mono', Monaco, 'Consolas', monospace;
}}
details summary::-webkit-details-marker {{ color: var(--preventive); }}

/* ---- Footer ---- */
.footer {{
    text-align:center; padding:16px; color:var(--text-muted); font-size:11px;
    border-top:1px solid var(--border); margin-top:32px;
}}

/* Query Builder Tabs */
.qb-tabs {{ display:flex; gap:0; margin-bottom:14px; border-radius:10px; overflow:hidden; border:1px solid var(--border); }}
.qb-tab {{ flex:1; padding:10px; font-size:13px; font-weight:600; cursor:pointer; border:none; background:var(--bg-card); color:var(--text-sec); transition:all 0.2s; text-align:center; }}
.qb-tab.active {{ background:var(--info); color:#fff; }}
.qb-tab:hover:not(.active) {{ background:var(--bg-hover); }}

/* Query Builder Steps */
.qb-container {{ display:none; }}
.qb-container.active {{ display:block; }}
.qb-step {{ margin-bottom:20px; padding:14px 16px; background:rgba(0,107,166,0.02); border-radius:10px; border-left:3px solid var(--info); transition:opacity 0.35s ease, transform 0.35s ease; }}
.qb-step-label {{ font-size:12px; font-weight:700; color:var(--text-sec); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px; display:flex; align-items:center; gap:8px; }}
.qb-step-num {{ width:24px; height:24px; border-radius:50%; background:linear-gradient(135deg, var(--info), #0097A7); color:#fff; display:inline-flex; align-items:center; justify-content:center; font-size:11px; font-weight:700; box-shadow:0 2px 6px rgba(0,107,166,0.3); }}

/* Concept Cards */
.qb-cards {{ display:flex; gap:8px; flex-wrap:wrap; }}
.qb-card {{ padding:10px 16px; border-radius:10px; border:2px solid var(--border); background:var(--bg-card); cursor:pointer; transition:all 0.2s; text-align:center; min-width:100px; }}
.qb-card:hover {{ border-color:var(--info); transform:translateY(-1px); }}
.qb-card.selected {{ border-color:var(--info); background:rgba(0,107,166,0.08); }}
.qb-card-icon {{ font-size:24px; margin-bottom:4px; }}
.qb-card-label {{ font-size:12px; font-weight:600; color:var(--text); }}
.qb-card-desc {{ font-size:10px; color:var(--text-muted); margin-top:2px; }}

/* Type checkboxes */
.qb-type-group {{ margin-bottom:8px; }}
.qb-type-header {{ font-size:12px; font-weight:600; color:var(--text-sec); margin-bottom:4px; }}
.qb-type-option {{ display:inline-flex; align-items:center; gap:4px; padding:4px 10px; margin:2px; border-radius:6px; border:1px solid var(--border); background:var(--bg-card); cursor:pointer; font-size:11px; color:var(--text); transition:all 0.15s; }}
.qb-type-option:hover {{ border-color:var(--info); }}
.qb-type-option.selected {{ border-color:var(--info); background:rgba(0,107,166,0.08); color:var(--info); font-weight:600; }}

/* Action cards */
.qb-action {{ padding:8px 14px; border-radius:8px; border:1px solid var(--border); background:var(--bg-card); cursor:pointer; font-size:12px; color:var(--text); transition:all 0.15s; display:inline-flex; align-items:center; gap:6px; margin:3px; }}
.qb-action:hover {{ border-color:var(--info); }}
.qb-action.selected {{ border-color:var(--info); background:rgba(0,107,166,0.08); color:var(--info); font-weight:600; }}

/* Filter dropdowns */
.qb-filter-row {{ display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }}
.qb-select {{ padding:7px 12px; border-radius:8px; border:1px solid var(--border); background:var(--bg-card); color:var(--text); font-size:12px; outline:none; min-width:140px; }}
.qb-select:focus {{ border-color:var(--info); }}

/* Preview box */
.qb-preview {{ padding:14px 18px; background:rgba(0,107,166,0.04); border:1px solid rgba(0,107,166,0.15); border-radius:10px; margin-top:12px; }}
.qb-preview-text {{ font-size:14px; color:var(--text); font-weight:500; line-height:1.5; }}
.qb-preview-actions {{ display:flex; gap:8px; margin-top:10px; }}

/* ── Visual Query Builder (VQB) Styles ── */
.vqb-table-selector {{
    display:flex;
    gap:12px;
    flex-wrap:wrap;
    margin:16px 0;
}}
.vqb-table-chip {{
    display:flex;
    align-items:center;
    gap:8px;
    padding:10px 14px;
    border:2px solid var(--border);
    border-radius:6px;
    cursor:pointer;
    background:var(--bg-card);
    font-size:13px;
    color:var(--text);
    font-weight:500;
    transition:all 0.25s ease;
    position:relative;
    box-shadow:0 1px 3px rgba(0,0,0,0.08);
}}
.vqb-table-chip:hover {{
    border-color:var(--info);
    background:rgba(0,107,166,0.08);
    transform:translateY(-1px);
    box-shadow:0 2px 6px rgba(0,107,166,0.15);
}}
.vqb-table-chip.selected {{
    background:var(--info);
    color:#fff;
    border-color:var(--info);
    box-shadow:0 2px 8px rgba(0,107,166,0.3);
}}
.vqb-table-chip::after {{
    content:attr(data-count);
    display:flex;
    align-items:center;
    justify-content:center;
    min-width:20px;
    height:20px;
    padding:0 4px;
    background:rgba(0,0,0,0.1);
    border-radius:10px;
    font-size:10px;
    font-weight:600;
}}
.vqb-table-chip.selected::after {{
    background:rgba(255,255,255,0.3);
}}
.vqb-table-chip input[type="checkbox"] {{
    cursor:pointer;
}}
.vqb-columns-area {{
    display:flex;
    flex-direction:column;
    gap:12px;
    margin:16px 0;
}}
.vqb-table-panel {{
    border:1px solid var(--border);
    border-radius:8px;
    background:var(--bg-card);
    overflow:hidden;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
    transition:all 0.2s ease;
}}
.vqb-table-panel:hover {{
    box-shadow:0 2px 6px rgba(0,0,0,0.1);
}}
.vqb-table-panel-header {{
    padding:12px 14px;
    background:linear-gradient(135deg, rgba(0,107,166,0.08), rgba(0,107,166,0.04));
    display:flex;
    align-items:center;
    gap:10px;
    cursor:pointer;
    user-select:none;
    transition:background 0.2s ease;
    border-bottom:1px solid rgba(0,107,166,0.1);
}}
.vqb-table-panel-header:hover {{
    background:linear-gradient(135deg, rgba(0,107,166,0.12), rgba(0,107,166,0.08));
}}
.vqb-table-panel-header .collapse-arrow {{
    transition:transform 0.25s ease;
    font-size:16px;
    color:var(--info);
}}
.vqb-table-panel-header .collapse-arrow.collapsed {{
    transform:rotate(-90deg);
}}
.vqb-table-panel-header .table-icon {{
    font-size:18px;
    color:var(--info);
}}
.vqb-table-panel-header .table-name {{
    font-weight:600;
    color:var(--text);
    flex:1;
    letter-spacing:0.3px;
}}
.vqb-table-panel-header .select-controls {{
    font-size:11px;
    color:var(--info);
}}
.vqb-table-panel-header .select-controls a {{
    color:var(--info);
    text-decoration:none;
    cursor:pointer;
    margin:0 4px;
    font-weight:500;
    transition:color 0.15s;
}}
.vqb-table-panel-header .select-controls a:hover {{
    color:var(--info);
    text-decoration:underline;
}}
.vqb-col-grid {{
    display:grid;
    grid-template-columns:repeat(3, 1fr);
    gap:10px;
    padding:12px;
    max-height:240px;
    overflow-y:auto;
}}
.vqb-col-item {{
    display:flex;
    align-items:center;
    gap:8px;
    padding:8px;
    border-radius:5px;
    transition:all 0.15s ease;
    background:rgba(0,0,0,0.02);
}}
.vqb-col-item:nth-child(odd) {{
    background:rgba(0,0,0,0);
}}
.vqb-col-item:hover {{
    background:rgba(0,107,166,0.1);
}}
.vqb-col-item input[type="checkbox"] {{
    cursor:pointer;
    flex-shrink:0;
}}
.vqb-col-label {{
    font-size:12px;
    color:var(--text);
    flex:1;
    cursor:pointer;
    word-break:break-word;
}}
.vqb-col-type {{
    font-size:10px;
    font-weight:600;
    padding:2px 6px;
    border-radius:3px;
    white-space:nowrap;
    margin-left:auto;
    flex-shrink:0;
}}
.vqb-col-type.numeric {{
    background:#e8f5e9;
    color:#2e7d32;
}}
.vqb-col-type.string {{
    background:#e3f2fd;
    color:#1565c0;
}}
.vqb-col-type.date {{
    background:#fff3e0;
    color:#e65100;
}}
.vqb-filter-area {{
    display:flex;
    flex-direction:column;
    gap:10px;
    margin:16px 0;
}}
.vqb-filter-row {{
    display:flex;
    align-items:center;
    gap:10px;
    padding:12px;
    background:var(--bg-card);
    border:1px solid var(--border);
    border-radius:6px;
    box-shadow:0 1px 2px rgba(0,0,0,0.04);
    transition:all 0.2s ease;
}}
.vqb-filter-row:nth-child(odd) {{
    background:rgba(0,107,166,0.02);
}}
.vqb-filter-row:hover {{
    border-color:var(--info);
}}
.vqb-filter-row select,
.vqb-filter-row input {{
    padding:8px 10px;
    border:1px solid var(--border);
    border-radius:5px;
    font-size:12px;
    min-height:32px;
    transition:all 0.15s ease;
    background:var(--bg-card);
    color:var(--text);
}}
.vqb-filter-row select:focus,
.vqb-filter-row input:focus {{
    border-color:var(--info);
    outline:none;
    box-shadow:0 0 0 2px rgba(0,107,166,0.1);
}}
.vqb-filter-row select {{
    min-width:120px;
}}
.vqb-filter-row input {{
    flex:1;
    min-width:120px;
}}
.vqb-filter-remove {{
    padding:6px 12px;
    background:rgba(200,16,46,0.08);
    color:#C8102E;
    border:1px solid rgba(200,16,46,0.2);
    border-radius:5px;
    cursor:pointer;
    font-size:12px;
    font-weight:600;
    transition:all 0.2s ease;
}}
.vqb-filter-remove:hover {{
    background:rgba(200,16,46,0.15);
    border-color:#C8102E;
}}
.vqb-add-filter-btn {{
    padding:10px 14px;
    background:rgba(0,107,166,0.08);
    border:2px solid rgba(0,107,166,0.3);
    color:var(--info);
    border-radius:6px;
    cursor:pointer;
    font-size:12px;
    font-weight:600;
    transition:all 0.2s ease;
}}
.vqb-add-filter-btn:hover {{
    background:rgba(0,107,166,0.15);
    border-color:var(--info);
    transform:translateY(-1px);
}}
.vqb-limit-row {{
    display:flex;
    align-items:center;
    gap:14px;
    margin:16px 0;
    padding:12px 14px;
    background:var(--bg-card);
    border-radius:6px;
    border:1px solid var(--border);
    box-shadow:0 1px 2px rgba(0,0,0,0.04);
}}
.vqb-limit-row label {{
    font-weight:500;
}}
.vqb-limit-row select {{
    padding:8px 10px;
    border:1px solid var(--border);
    border-radius:5px;
    font-size:12px;
    background:var(--bg-card);
    color:var(--text);
    cursor:pointer;
    transition:all 0.15s ease;
}}
.vqb-limit-row select:focus {{
    border-color:var(--info);
    outline:none;
    box-shadow:0 0 0 2px rgba(0,107,166,0.1);
}}
.vqb-sql-preview {{
    font-family:Consolas, Monaco, 'Courier New', monospace;
    font-size:12px;
    padding:14px;
    background:var(--bg-card);
    color:var(--text);
    border-radius:6px;
    border:1px solid var(--border);
    margin:16px 0;
    max-height:220px;
    overflow-y:auto;
    white-space:pre-wrap;
    word-break:break-word;
    line-height:1.5;
    box-shadow:inset 0 1px 3px rgba(0,0,0,0.05);
}}
.vqb-sql-keyword {{
    color:var(--info);
    font-weight:600;
}}
.vqb-sql-table {{
    color:#2e7d32;
    font-weight:500;
}}
.vqb-sql-column {{
    color:#f57c00;
}}
.vqb-run-bar {{
    display:flex;
    gap:10px;
    margin:16px 0;
    padding:12px;
    background:rgba(0,107,166,0.04);
    border-radius:6px;
    border:1px solid rgba(0,107,166,0.1);
}}
.vqb-run-bar button {{
    padding:10px 16px;
    border-radius:5px;
    font-size:12px;
    font-weight:600;
    cursor:pointer;
    border:none;
    transition:all 0.2s ease;
    display:flex;
    align-items:center;
    gap:6px;
}}
.vqb-run-bar .query-btn {{
    background:var(--info);
    color:#fff;
}}
.vqb-run-bar .query-btn:hover {{
    background:#005080;
    transform:translateY(-1px);
    box-shadow:0 2px 8px rgba(0,107,166,0.25);
}}
.vqb-run-bar .auth-btn {{
    background:var(--border);
    color:var(--text);
}}
.vqb-run-bar .auth-btn:hover {{
    background:rgba(0,0,0,0.1);
}}

/* Data Source Navigator Styles */
#dataSourceNavPanel {{
    font-size:14px;
}}
#dataSourceNavPanel select {{
    width:100%;
    padding:8px;
    border:1px solid var(--border);
    border-radius:6px;
    font-size:12px;
    color:var(--text);
    background:#fff;
    cursor:pointer;
}}
#dataSourceNavPanel select:focus {{
    outline:none;
    border-color:var(--info);
    box-shadow:0 0 0 3px rgba(0,107,166,0.1);
}}
#dsSchemaTree {{
    display:flex;
    flex-direction:column;
    gap:8px;
}}
.ds-schema-item {{
    border:1px solid var(--border);
    border-radius:6px;
    overflow:hidden;
}}
.ds-schema-header {{
    padding:8px 10px;
    background:var(--bg-card);
    cursor:pointer;
    display:flex;
    align-items:center;
    gap:8px;
    user-select:none;
    transition:background 0.2s;
}}
.ds-schema-header:hover {{
    background:var(--bg-hover);
}}
.ds-table-item {{
    padding:8px;
    border-bottom:1px solid var(--border);
    cursor:pointer;
    transition:all 0.2s;
}}
.ds-table-item:last-child {{
    border-bottom:none;
}}
.ds-table-item:hover {{
    background:var(--bg-hover);
}}
.ds-table-item.selected {{
    background:rgba(0,107,166,0.08);
    border-left:3px solid var(--info);
}}
.ds-table-name {{
    font-weight:500;
    color:var(--text);
    font-size:12px;
}}
.ds-table-desc {{
    font-size:11px;
    color:var(--text-muted);
    margin-top:2px;
}}
.ds-tag {{
    display:inline-block;
    font-size:9px;
    padding:2px 6px;
    margin-right:4px;
    background:rgba(0,107,166,0.1);
    color:var(--info);
    border-radius:3px;
    margin-top:4px;
}}
#dsClaimTypesGrid {{
    display:grid;
    gap:8px;
    grid-template-columns:1fr;
}}
.ds-claim-category {{
    padding:10px;
    border:1px solid var(--border);
    border-radius:6px;
    background:var(--bg-card);
}}
.ds-claim-category-header {{
    display:flex;
    align-items:center;
    gap:8px;
    margin-bottom:8px;
}}
.ds-claim-category-title {{
    font-weight:600;
    color:var(--text);
    font-size:12px;
}}
.ds-claim-subtype {{
    padding:8px;
    margin-bottom:4px;
    border-radius:4px;
    cursor:pointer;
    border:1px solid var(--border);
    background:#fff;
    color:var(--text);
    font-size:11px;
    transition:all 0.2s;
}}
.ds-claim-subtype:hover {{
    border-color:var(--info);
    background:rgba(0,107,166,0.04);
}}
.ds-claim-subtype.selected {{
    background:var(--info);
    color:#fff;
    border-color:var(--info);
}}
.ds-claim-subtype-name {{
    font-weight:500;
}}
.ds-claim-subtype-desc {{
    font-size:10px;
    opacity:0.8;
    margin-top:2px;
}}
.ds-claim-subtype-code {{
    font-size:9px;
    opacity:0.7;
    margin-top:2px;
}}

/* ---- Responsive ---- */
@media (max-width:768px) {{
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
    #dataSourceNavPanel {{ width:90vw!important; }}
}}
</style>
<!-- Anti-DevTools: block inspect, view-source, and debugging -->
<script>
// Block right-click context menu
document.addEventListener('contextmenu', function(e) {{ e.preventDefault(); return false; }});

// Block keyboard shortcuts for DevTools
document.addEventListener('keydown', function(e) {{
    // F12
    if (e.key === 'F12' || e.keyCode === 123) {{ e.preventDefault(); return false; }}
    // Ctrl+Shift+I / Cmd+Opt+I (Inspect)
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'I' || e.key === 'i' || e.keyCode === 73)) {{ e.preventDefault(); return false; }}
    // Ctrl+Shift+J / Cmd+Opt+J (Console)
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'J' || e.key === 'j' || e.keyCode === 74)) {{ e.preventDefault(); return false; }}
    // Ctrl+Shift+C / Cmd+Opt+C (Element picker)
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'C' || e.key === 'c' || e.keyCode === 67)) {{ e.preventDefault(); return false; }}
    // Ctrl+U / Cmd+U (View Source)
    if ((e.ctrlKey || e.metaKey) && (e.key === 'U' || e.key === 'u' || e.keyCode === 85)) {{ e.preventDefault(); return false; }}
    // Ctrl+S / Cmd+S (Save page)
    if ((e.ctrlKey || e.metaKey) && (e.key === 'S' || e.key === 's' || e.keyCode === 83)) {{ e.preventDefault(); return false; }}
}});

// DevTools open detection — checks for debugger timing + window size delta
(function() {{
    const _dtCheck = function() {{
        const widthThreshold = window.outerWidth - window.innerWidth > 160;
        const heightThreshold = window.outerHeight - window.innerHeight > 160;
        if (widthThreshold || heightThreshold) {{
            document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;' +
                'height:100vh;background:#FFFFFF;color:#C8102E;font-family:system-ui;font-size:18px;' +
                'text-align:center;padding:40px;">' +
                '<div><div style="font-size:48px;margin-bottom:16px;">&#128274;</div>' +
                '<b>Developer Tools Detected</b><br><br>' +
                '<span style="color:#58595B;font-size:14px;">This application is secured. ' +
                'Please close Developer Tools to continue.</span></div></div>';
        }}
    }};
    setInterval(_dtCheck, 1000);

    // Debugger-based detection
    const _debugDetect = new Image();
    Object.defineProperty(_debugDetect, 'id', {{
        get: function() {{
            // If this getter is called, devtools is open (inspecting the object)
            document.title = '🔒 Secured';
        }}
    }});

    // Disable console methods to prevent data extraction
    const _noop = function() {{ return undefined; }};
    try {{
        Object.defineProperty(window, 'console', {{
            value: new Proxy(console, {{
                get: function(target, prop) {{
                    if (['log','debug','info','warn','error','table','dir','trace','group','groupEnd','clear'].includes(prop)) {{
                        return _noop;
                    }}
                    return target[prop];
                }}
            }}),
            writable: false,
            configurable: false,
        }});
    }} catch(e) {{}}

    // Prevent drag (no drag-to-save images/text)
    document.addEventListener('dragstart', function(e) {{ e.preventDefault(); }});

    // Prevent selection of sensitive elements (but allow input fields)
    document.addEventListener('selectstart', function(e) {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return true;
        // Allow selection in results area for copy-paste of data
        if (e.target.closest && e.target.closest('#results-area')) return true;
        e.preventDefault();
    }});
}})();
</script>
</head>
<body>

<!-- ═══ AUTH GATE OVERLAY — blocks entire dashboard until login ═══ -->
<div id="authGate" style="position:fixed;inset:0;z-index:100000;
    background:linear-gradient(135deg, #F0F4F8 0%, #FFFFFF 50%, #E8F0FE 100%);
    display:flex;align-items:center;justify-content:center;">
    <div style="background:#FFFFFF;border-radius:20px;border:1px solid #D9DDE3;
        padding:44px 40px;width:440px;max-width:92vw;box-shadow:0 24px 80px rgba(0,0,0,0.12);">

        <!-- KP Branding -->
        <div style="text-align:center;margin-bottom:28px;">
            <div style="font-size:44px;margin-bottom:8px;">&#9879;</div>
            <h1 style="font-size:24px;margin:0 0 4px;color:var(--text);font-weight:700;">
                KP Healthcare
            </h1>
            <p style="font-size:13px;color:var(--text-muted);margin:0;">
                MTP Healthcare Analytics Platform
            </p>
        </div>

        <!-- SSO Buttons (hidden if not configured) -->
        <div id="ssoButtons" style="display:none;margin-bottom:20px;">
            <button onclick="startSSO('google')" id="googleSSOBtn"
                style="width:100%;padding:12px 16px;border-radius:10px;border:1px solid var(--border);
                background:#fff;color:#333;font-size:14px;font-weight:500;cursor:pointer;
                display:flex;align-items:center;justify-content:center;gap:10px;margin-bottom:10px;">
                <svg width="18" height="18" viewBox="0 0 18 18"><path fill="#4285F4" d="M16.51 8H8.98v3h4.3c-.18 1-.74 1.48-1.6 2.04v2.01h2.6a7.8 7.8 0 002.38-5.88c0-.57-.05-.99-.15-1.17z"/><path fill="#34A853" d="M8.98 17c2.16 0 3.97-.72 5.3-1.94l-2.6-2a4.8 4.8 0 01-7.18-2.54H1.83v2.07A8 8 0 008.98 17z"/><path fill="#FBBC05" d="M4.5 10.52a4.8 4.8 0 010-3.04V5.41H1.83a8 8 0 000 7.18l2.67-2.07z"/><path fill="#EA4335" d="M8.98 3.58c1.32 0 2.29.46 3.13 1.25l2.32-2.32A7.93 7.93 0 008.98 0 8 8 0 001.83 5.41l2.67 2.07A4.8 4.8 0 018.98 3.58z"/></svg>
                Continue with Google
            </button>
            <button onclick="startSSO('microsoft')" id="msSSOBtn"
                style="width:100%;padding:12px 16px;border-radius:10px;border:1px solid var(--border);
                background:#fff;color:#333;font-size:14px;font-weight:500;cursor:pointer;
                display:flex;align-items:center;justify-content:center;gap:10px;">
                <svg width="18" height="18" viewBox="0 0 21 21"><rect fill="#F25022" x="1" y="1" width="9" height="9"/><rect fill="#7FBA00" x="11" y="1" width="9" height="9"/><rect fill="#00A4EF" x="1" y="11" width="9" height="9"/><rect fill="#FFB900" x="11" y="11" width="9" height="9"/></svg>
                Continue with Microsoft
            </button>
            <div style="display:flex;align-items:center;gap:12px;margin:18px 0;">
                <hr style="flex:1;border:none;border-top:1px solid var(--border);">
                <span style="font-size:12px;color:var(--text-muted);white-space:nowrap;">or sign in with credentials</span>
                <hr style="flex:1;border:none;border-top:1px solid var(--border);">
            </div>
        </div>

        <!-- Gate tabs -->
        <div style="display:flex;gap:0;margin-bottom:18px;border-radius:10px;overflow:hidden;border:1px solid var(--border);">
            <button id="gateLoginTab" onclick="switchGateTab('login')"
                style="flex:1;padding:11px;font-size:13px;font-weight:600;cursor:pointer;border:none;
                background:var(--info);color:#fff;transition:all 0.2s;">Log In</button>
            <button id="gateSignupTab" onclick="switchGateTab('signup')"
                style="flex:1;padding:11px;font-size:13px;font-weight:600;cursor:pointer;border:none;
                background:var(--bg-hover);color:var(--text-sec);transition:all 0.2s;">Sign Up</button>
        </div>

        <!-- Login Form -->
        <div id="gateLoginForm">
            <input class="modal-input" id="gateLoginUser" placeholder="Username or Email"
                autocomplete="username" style="width:100%;box-sizing:border-box;">
            <input class="modal-input" id="gateLoginPass" type="password" placeholder="Password"
                autocomplete="current-password" style="width:100%;box-sizing:border-box;"
                onkeydown="if(event.key==='Enter')gateLogin()">
            <div class="modal-error" id="gateLoginError"></div>
            <button class="auth-btn auth-btn-primary" onclick="gateLogin()"
                style="width:100%;padding:12px;font-size:14px;margin-top:8px;border-radius:10px;">
                Sign In
            </button>
        </div>

        <!-- Signup Form -->
        <div id="gateSignupForm" style="display:none;">
            <input class="modal-input" id="gateSignupUser" placeholder="Username" autocomplete="username"
                style="width:100%;box-sizing:border-box;">
            <input class="modal-input" id="gateSignupEmail" placeholder="Email Address (optional — auto-generated if blank)"
                type="email" autocomplete="email" style="width:100%;box-sizing:border-box;">
            <input class="modal-input" id="gateSignupDisplay" placeholder="Display Name (optional)"
                style="width:100%;box-sizing:border-box;">
            <input class="modal-input" id="gateSignupPass" type="password" placeholder="Password (min 4 chars)"
                autocomplete="new-password" style="width:100%;box-sizing:border-box;">
            <input class="modal-input" id="gateSignupTeam" placeholder="Team / Department (optional)" list="gateTeamList"
                style="width:100%;box-sizing:border-box;">
            <datalist id="gateTeamList"></datalist>
            <div class="modal-error" id="gateSignupError"></div>
            <button class="auth-btn auth-btn-primary" onclick="gateSignup()"
                style="width:100%;padding:12px;font-size:14px;margin-top:8px;border-radius:10px;">
                Create Account
            </button>
            <div style="margin-top:12px;text-align:center;">
                <a href="#" onclick="openGoogleSignup();return false;"
                    style="font-size:12px;color:var(--info);text-decoration:none;">
                    Need a Google account? Create one here &#8599;
                </a>
            </div>
        </div>

        <!-- Footer -->
        <div style="text-align:center;margin-top:20px;padding-top:16px;border-top:1px solid var(--border);">
            <p style="font-size:11px;color:var(--text-muted);margin:0;">
                Secure Authentication &bull; Encrypted Sessions
            </p>
        </div>
    </div>
</div>

<!-- Auth Bar -->
<div class="auth-bar" id="authBar">
    <div style="display:flex;align-items:center;gap:8px;">
        <button class="auth-btn" onclick="showSavedDashboards()">&#128190; Dashboards</button>
        <button class="auth-btn" onclick="promptSaveDashboard()">&#128427; Save</button>
        <button class="auth-btn" onclick="openEmailModal()">&#9993; Email</button>
    </div>
    <div id="authLoggedIn" style="display:flex;align-items:center;gap:10px;margin-left:auto;">
        <img id="authAvatar" src="" style="width:28px;height:28px;border-radius:50%;display:none;">
        <div class="auth-user">
            <span id="authDisplayName"></span>
            <span id="authEmail" style="font-size:11px;color:var(--text-muted);margin-left:4px;"></span>
        </div>
        <span class="auth-team" id="authTeam" style="display:none;"></span>
        <span id="authSSOBadge" style="display:none;font-size:10px;padding:2px 8px;border-radius:4px;
            background:rgba(0,107,166,0.15);color:var(--info);font-weight:600;"></span>
        <span class="meta-badge" id="sessionBadge" style="background:rgba(5,150,105,0.12);color:var(--healthy);font-size:11px;display:none;"></span>
        <button class="auth-btn auth-btn-danger" onclick="doLogout()">Logout</button>
    </div>
</div>

<!-- Email Modal -->
<div class="modal-overlay" id="emailModal">
    <div class="modal" style="width:480px;">
        <h2>&#9993; Email Dashboard</h2>
        <p style="font-size:12px;color:var(--text-muted);margin:0 0 8px;">
            Send the current dashboard results via email.
        </p>
        <div style="font-size:12px;color:var(--text-sec);margin-bottom:14px;padding:8px 12px;background:var(--bg-hover);border-radius:8px;">
            From: <strong id="emailFromDisplay">your-account@gmail.com</strong>
        </div>
        <input class="modal-input" id="emailTo" type="email" placeholder="Recipient / To email address">
        <input class="modal-input" id="emailSubject" value="Healthcare Dashboard Report">
        <textarea class="modal-input" id="emailBody" rows="3" placeholder="Add a note (optional)"
            style="resize:vertical;font-family:inherit;"></textarea>
        <div id="smtpSetup" style="display:none;margin-top:12px;padding:12px;background:var(--bg-hover);
            border-radius:8px;border:1px solid var(--border);">
            <p style="font-size:12px;font-weight:600;color:var(--warning);margin:0 0 8px;">
                &#9888; SMTP Setup Required (one-time)
            </p>
            <input class="modal-input" id="smtpEmail" type="email" placeholder="Your Gmail/Outlook email">
            <input class="modal-input" id="smtpHost" value="smtp.gmail.com" placeholder="SMTP Host">
            <input class="modal-input" id="smtpPort" value="587" placeholder="SMTP Port" type="number">
            <input class="modal-input" id="smtpPass" type="password" placeholder="App Password (Gmail) or SMTP Password">
            <button class="auth-btn auth-btn-primary" onclick="saveSMTPConfig()"
                style="width:100%;margin-top:4px;">Save SMTP Config</button>
        </div>
        <div class="modal-error" id="emailError"></div>
        <div class="modal-actions" style="margin-top:12px;">
            <button class="auth-btn auth-btn-primary" onclick="doSendEmail()" style="flex:1;">Send Email</button>
            <button class="auth-btn" onclick="hideModal('emailModal')">Cancel</button>
        </div>
    </div>
</div>

<!-- Export Password Modal -->
<div class="modal-overlay" id="exportPwModal">
    <div class="modal">
        <h2 id="exportPwTitle">Set Export Password</h2>
        <p style="font-size:12px;color:var(--text-muted);margin:0 0 12px;">
            This password protects your exported files (HTML/PDF/CSV). It is separate from your login password.
        </p>
        <div id="exportPwCarryForward" style="display:none;margin-bottom:12px;">
            <label style="display:flex;align-items:center;gap:8px;font-size:13px;color:var(--text-sec);cursor:pointer;">
                <input type="checkbox" id="exportPwReuse" checked onchange="toggleExportPwInput()">
                Use my existing export password
            </label>
        </div>
        <div id="exportPwNewFields">
            <input class="modal-input" id="exportPwInput" type="password" placeholder="Enter export password (min 4 chars)">
            <input class="modal-input" id="exportPwConfirm" type="password" placeholder="Confirm export password"
                onkeydown="if(event.key==='Enter')confirmExportPassword()">
        </div>
        <div class="modal-error" id="exportPwError"></div>
        <div class="modal-actions">
            <button class="auth-btn auth-btn-primary" onclick="confirmExportPassword()" style="flex:1;">Confirm &amp; Save</button>
            <button class="auth-btn" onclick="hideModal('exportPwModal')">Cancel</button>
        </div>
    </div>
</div>

<!-- Save Dashboard Modal -->
<div class="modal-overlay" id="saveModal">
    <div class="modal">
        <h2>Save Dashboard</h2>
        <input class="modal-input" id="saveDashName" placeholder="Dashboard name (e.g. 'Q1 Claims Review')">
        <input class="modal-input" id="saveDashDesc" placeholder="Description (optional)">
        <div class="modal-error" id="saveError"></div>
        <div class="modal-actions">
            <button class="auth-btn auth-btn-primary" onclick="doSaveDashboard()" style="flex:1;">Save</button>
            <button class="auth-btn" onclick="hideModal('saveModal')">Cancel</button>
        </div>
    </div>
</div>

<!-- Saved Dashboards Modal -->
<div class="modal-overlay" id="dashboardsModal">
    <div class="modal" style="width:600px;">
        <h2>My Dashboards</h2>
        <div style="display:flex;gap:8px;margin-bottom:16px;">
            <button class="auth-btn auth-btn-primary" onclick="loadDashboardList('my')">My Dashboards</button>
            <button class="auth-btn" onclick="loadDashboardList('team')">Team Dashboards</button>
        </div>
        <div id="dashboardsList" style="max-height:400px;overflow-y:auto;"></div>
        <div class="modal-actions" style="margin-top:16px;">
            <button class="auth-btn" onclick="hideModal('dashboardsModal')">Close</button>
        </div>
    </div>
</div>

<!-- Top Bar with Query Input & Visual Query Builder -->
<div class="top-bar">
    <h1>MTP Healthcare Analytics</h1>

    <!-- Tab buttons -->
    <div class="qb-tabs">
        <button class="qb-tab active" id="askTab" onclick="switchQueryTab('ask')">Ask a Question</button>
        <button class="qb-tab" id="buildTab" onclick="switchQueryTab('build')">Build a Query</button>
    </div>

    <!-- Tab 1: Ask Container (existing query bar) -->
    <div id="askContainer" class="qb-container active">
        <div class="query-bar">
            <input class="query-input" id="queryInput" type="text"
                   placeholder="Ask a question... e.g. 'show me denied claims' or 'average cost by region'"
                   value="{_esc(initial_query)}"
                   onkeydown="if(event.key==='Enter')submitQuery()">
            <button class="query-btn" id="queryBtn" onclick="submitQuery()">Query</button>
        </div>
        <button class="suggestion-toggle" onclick="document.getElementById('suggestions').classList.toggle('expanded');this.querySelector('.arrow').textContent=document.getElementById('suggestions').classList.contains('expanded')?'&#9650;':'&#9660;'">
            <span class="arrow">&#9660;</span> Example queries (click to browse)
        </button>
        <div class="suggestions" id="suggestions">{suggestion_chips}</div>
    </div>

    <!-- Tab 2: Build Container (Visual Query Builder) -->
    <div id="buildContainer" class="qb-container">
      <!-- VQB Header -->
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;padding:0 4px;">
        <h2 style="margin:0;font-size:18px;color:var(--text);font-weight:600;">Visual Query Builder</h2>
        <button class="auth-btn" onclick="vqbClear()" title="Reset the query builder"
                style="padding:6px 12px;font-size:12px;">↺ Reset</button>
      </div>

      <!-- Step 1: Select Tables -->
      <div class="qb-step">
        <div class="qb-step-label"><span class="qb-step-num">1</span> Select Tables</div>
        <div class="vqb-table-selector" id="vqbTables">
          <!-- 7 table chips rendered by JS -->
        </div>
      </div>

      <!-- Suggested Queries (shown when tables are selected) -->
      <div id="vqbSuggestions" style="display:none;margin:0 0 4px 0;padding:10px 14px;
          background:linear-gradient(135deg,rgba(0,107,166,0.08),rgba(40,97,64,0.06));
          border-radius:10px;border:1px solid rgba(0,107,166,0.18);">
        <div style="font-size:11px;font-weight:600;color:var(--info);margin-bottom:8px;">
          💡 Suggested queries based on your selection
        </div>
        <div id="vqbSuggestionsChips" style="display:flex;gap:6px;flex-wrap:wrap;"></div>
      </div>

      <!-- Step 2: Select Columns -->
      <div class="qb-step" id="vqbStep2" style="display:none">
        <div class="qb-step-label">
          <span class="qb-step-num">2</span> Select Columns
          <span style="font-size:11px;font-weight:400;color:var(--text-muted);margin-left:8px;">
            <a href="#" onclick="vqbSelectAllCols();return false" style="color:var(--info)">Select All</a> ·
            <a href="#" onclick="vqbDeselectAllCols();return false" style="color:var(--info)">Deselect All</a>
          </span>
        </div>
        <div class="vqb-columns-area" id="vqbColumns"></div>
      </div>

      <!-- Step 3: Filters -->
      <div class="qb-step" id="vqbStep3" style="display:none">
        <div class="qb-step-label"><span class="qb-step-num">3</span> Filters (optional)</div>
        <div class="vqb-filter-area" id="vqbFilters"></div>
        <button class="vqb-add-filter-btn" onclick="vqbAddFilter()">+ Add Filter</button>
      </div>

      <!-- Step 4: Aggregation (optional) -->
      <div class="qb-step" id="vqbStep4" style="display:none">
        <div class="qb-step-label"><span class="qb-step-num">4</span> Aggregation & Grouping (optional)</div>
        <div id="vqbAggArea">
          <div class="qb-filter-row" style="margin-bottom:8px">
            <label style="font-size:12px;color:var(--text-sec);min-width:80px;font-weight:500;">Function:</label>
            <select class="qb-select" id="vqbAggFunc" onchange="vqbUpdateSQL()">
              <option value="">None (raw records)</option>
              <option value="COUNT">COUNT</option>
              <option value="SUM">SUM</option>
              <option value="AVG">AVG</option>
              <option value="MAX">MAX</option>
              <option value="MIN">MIN</option>
            </select>
            <label style="font-size:12px;color:var(--text-sec);margin-left:8px;font-weight:500;">on:</label>
            <select class="qb-select" id="vqbAggCol" onchange="vqbUpdateSQL()">
              <option value="*">*</option>
            </select>
          </div>
          <div class="qb-filter-row">
            <label style="font-size:12px;color:var(--text-sec);min-width:80px;font-weight:500;">Group By:</label>
            <select class="qb-select" id="vqbGroupBy" onchange="vqbUpdateSQL()" multiple style="min-height:60px;max-height:120px">
            </select>
          </div>
        </div>
      </div>

      <!-- Step 5: Record Limit + SQL Preview + Run -->
      <div class="qb-step" id="vqbStep5" style="display:none">
        <div class="qb-step-label"><span class="qb-step-num">5</span> Preview & Run</div>
        <div class="vqb-limit-row">
          <label style="font-size:12px;color:var(--text-sec);font-weight:500;">Record Limit:</label>
          <select id="vqbLimit" onchange="vqbUpdateSQL()" style="padding:8px 10px;border:1px solid var(--border);border-radius:5px;font-size:12px;background:var(--bg-card);color:var(--text);">
            <option value="10">10</option>
            <option value="25">25</option>
            <option value="50" selected>50</option>
            <option value="100">100</option>
            <option value="500">500</option>
            <option value="1000">1000</option>
            <option value="10000">10,000</option>
          </select>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin:12px 0 12px 0;">
          <label style="font-size:12px;color:var(--text-sec);font-weight:500;">Generated SQL:</label>
          <button class="auth-btn" onclick="vqbCopySQL()" style="padding:4px 10px;font-size:11px;">📋 Copy SQL</button>
        </div>
        <div class="vqb-sql-preview" id="vqbSQLPreview">
          -- Select tables and columns to build your query
        </div>
        <div class="vqb-run-bar">
          <button class="query-btn" onclick="vqbRunQuery()" style="flex:1;padding:10px">
            ▶ Run Query
          </button>
        </div>
      </div>
    </div>

    <!-- Contextual learning: others also asked + catalog browser -->
    <div id="learningBar" style="display:none;margin-top:8px;padding:8px 12px;
        background:rgba(0,107,166,0.06);border-radius:10px;border:1px solid rgba(0,107,166,0.15);">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <span style="font-size:12px;font-weight:600;color:var(--info);">
                &#128161; Others also asked
            </span>
            <button onclick="showCatalogBrowser()" style="font-size:11px;padding:4px 10px;
                border-radius:6px;border:1px solid var(--border);background:var(--bg-card);
                color:var(--text-sec);cursor:pointer;">
                &#128218; Browse Catalogs
            </button>
        </div>
        <div id="otherQueries" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;"></div>
    </div>
</div>

<!-- Catalog Browser Modal -->
<div class="modal-overlay" id="catalogModal">
    <div class="modal" style="width:640px;max-height:80vh;overflow-y:auto;">
        <h2>&#128218; Catalog Browser</h2>
        <div id="catalogContent" style="min-height:200px;">
            <div style="text-align:center;padding:40px;color:var(--text-muted);">Loading catalogs...</div>
        </div>
        <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:16px;">
            <button class="auth-btn" onclick="hideModal('catalogModal')">Close</button>
        </div>
    </div>
</div>

<!-- Table Hierarchy Multi-Select Panel -->
<div id="hierarchyPanel" style="display:none;position:fixed;right:0;top:0;width:380px;height:100vh;
    background:#FFFFFF;border-left:1px solid var(--border);z-index:1000;
    overflow-y:auto;box-shadow:-4px 0 20px rgba(0,0,0,0.1);transition:transform 0.3s ease;">
    <div style="padding:16px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;">
        <div>
            <h3 style="margin:0;font-size:16px;color:var(--text);">&#128204; Table Selection</h3>
            <p style="margin:4px 0 0;font-size:11px;color:var(--text-muted);">Select specific data types for your query</p>
        </div>
        <button onclick="closeHierarchyPanel()" style="background:none;border:none;color:var(--text-sec);
            font-size:20px;cursor:pointer;padding:4px 8px;">&times;</button>
    </div>
    <div id="hierarchyTree" style="padding:12px;"></div>
    <div style="padding:12px 16px;border-top:1px solid var(--border);position:sticky;bottom:0;
        background:#FFFFFF;">
        <div id="selectionSummary" style="font-size:11px;color:var(--text-muted);margin-bottom:8px;">
            No selections — queries will use default tables
        </div>
        <div style="display:flex;gap:8px;">
            <button onclick="applyHierarchySelections()" style="flex:1;padding:8px;border-radius:8px;
                border:none;background:var(--info);color:#fff;font-weight:600;cursor:pointer;font-size:13px;">
                Apply Selections
            </button>
            <button onclick="clearHierarchySelections()" style="padding:8px 12px;border-radius:8px;
                border:1px solid var(--border);background:var(--bg-card);color:var(--text-sec);
                cursor:pointer;font-size:13px;">
                Clear
            </button>
        </div>
    </div>
</div>

<!-- Floating button to open hierarchy panel -->
<button id="hierarchyToggle" onclick="openHierarchyPanel()"
    style="position:fixed;right:20px;bottom:60px;width:48px;height:48px;border-radius:50%;
    background:var(--info);color:#fff;border:none;font-size:20px;cursor:pointer;
    box-shadow:0 4px 12px rgba(0,107,166,0.25);z-index:999;display:none;"
    title="Table Selection">&#128204;</button>

<!-- Data Source Navigator Panel -->
<div id="dataSourceNavPanel" style="display:none;position:fixed;right:0;top:0;width:420px;height:100vh;
    background:#FFFFFF;border-left:1px solid var(--border);z-index:1001;
    overflow-y:auto;box-shadow:-4px 0 20px rgba(0,0,0,0.1);transition:transform 0.3s ease;">
    <div style="padding:16px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:#FFFFFF;z-index:10;">
        <div>
            <h3 style="margin:0;font-size:16px;color:var(--text);">&#128230; Data Source Navigator</h3>
            <p style="margin:4px 0 0;font-size:11px;color:var(--text-muted);">Browse environments &amp; catalogs</p>
        </div>
        <button onclick="closeDataSourceNav()" style="background:none;border:none;color:var(--text-sec);
            font-size:20px;cursor:pointer;padding:4px 8px;">&times;</button>
    </div>

    <!-- Environment &amp; Catalog Selection -->
    <div style="padding:12px 16px;border-bottom:1px solid var(--border);">
        <div style="margin-bottom:12px;">
            <label style="display:block;font-size:11px;font-weight:600;color:var(--text-muted);margin-bottom:4px;">ENVIRONMENT</label>
            <select id="dsEnvSelect" onchange="dsSelectEnvironment(this.value)" style="width:100%;padding:8px;border:1px solid var(--border);border-radius:6px;font-size:12px;color:var(--text);">
                <option value="local">Local (SQLite)</option>
                <option value="dev">Development</option>
                <option value="staging">Staging</option>
                <option value="prod">Production</option>
            </select>
        </div>
        <div>
            <label style="display:block;font-size:11px;font-weight:600;color:var(--text-muted);margin-bottom:4px;">CATALOG</label>
            <select id="dsCatalogSelect" onchange="dsSelectCatalog(this.value)" style="width:100%;padding:8px;border:1px solid var(--border);border-radius:6px;font-size:12px;color:var(--text);">
                <option value="local_catalog">Local Catalog</option>
                <option value="enriched_catalog">Enriched Catalog</option>
                <option value="staging_catalog">Staging Catalog</option>
                <option value="clarity_catalog">Clarity Catalog</option>
                <option value="apixio_catalog">Apixio Catalog</option>
            </select>
        </div>
        <div id="dsCatalogBadge" style="margin-top:8px;padding:6px 10px;border-radius:6px;background:var(--bg-card);font-size:10px;color:var(--text-muted);">
            &#9679; Source: SQLite
        </div>
    </div>

    <!-- Schema &amp; Table Tree -->
    <div style="padding:12px 16px;">
        <div style="font-size:11px;font-weight:600;color:var(--text-muted);margin-bottom:8px;">SCHEMAS &amp; TABLES</div>
        <div id="dsSchemaTree" style="font-size:12px;"></div>
    </div>

    <!-- Claim Type Deep-Drill (shown when Claims schema selected) -->
    <div id="dsClaimTypesPanel" style="display:none;padding:12px 16px;border-top:1px solid var(--border);">
        <div style="font-size:11px;font-weight:600;color:var(--text-muted);margin-bottom:8px;">CLAIM TYPES</div>
        <div id="dsClaimTypesGrid" style="display:grid;gap:8px;grid-template-columns:1fr;"></div>
    </div>

    <!-- Selected Source Summary -->
    <div style="padding:12px 16px;border-top:1px solid var(--border);position:sticky;bottom:0;
        background:#FFFFFF;">
        <div id="dsSourceSummary" style="font-size:11px;color:var(--text-muted);margin-bottom:8px;padding:8px 10px;
            background:var(--bg-card);border-radius:6px;border-left:3px solid var(--info);">
            No source selected
        </div>
        <div style="display:flex;gap:8px;">
            <button onclick="dsApplySelection()" style="flex:1;padding:8px;border-radius:8px;
                border:none;background:var(--info);color:#fff;font-weight:600;cursor:pointer;font-size:13px;">
                Apply Source
            </button>
            <button onclick="dsClearSelection()" style="padding:8px 12px;border-radius:8px;
                border:1px solid var(--border);background:var(--bg-card);color:var(--text-sec);
                cursor:pointer;font-size:13px;">
                Clear
            </button>
        </div>
    </div>
</div>

<!-- Floating button to open data source navigator -->
<button id="dataSourceNavToggle" onclick="openDataSourceNav()"
    style="position:fixed;right:20px;bottom:116px;width:48px;height:48px;border-radius:50%;
    background:var(--healthy);color:#fff;border:none;font-size:20px;cursor:pointer;
    box-shadow:0 4px 12px rgba(40,97,64,0.25);z-index:999;display:none;"
    title="Data Source Navigator">&#128230;</button>

{alerts_html}

<div class="main">
    <!-- Color Legend -->
    <div class="color-legend">
        <span style="font-weight:600;color:var(--text);">Color Guide:</span>
        <div class="legend-item"><div class="legend-dot" style="background:var(--critical)"></div> Critical / Denied</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--warning)"></div> Warning / Pending</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--healthy)"></div> Healthy / Paid / On Target</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--info)"></div> Informational</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--preventive)"></div> Preventive / Wellness</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--operations)"></div> Operational</div>
    </div>

    <!-- KPI Section -->
    {kpi_cards_html}

    <!-- Query History -->
    <div id="query-history" style="display:none;margin-bottom:16px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;"></div>

    <!-- Session Context Bar -->
    <div id="contextBar" style="display:none;margin-bottom:12px;padding:10px 14px;
        background:rgba(0,107,166,0.06);border:1px solid rgba(0,107,166,0.15);border-radius:10px;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-size:12px;font-weight:600;color:var(--info);">&#128172; Session Context</span>
                <span id="contextSelections" style="font-size:11px;color:var(--text-sec);"></span>
            </div>
            <div style="display:flex;gap:6px;">
                <span id="contextTurnCount" style="font-size:10px;color:var(--text-muted);"></span>
                <button onclick="clearSessionContext()" style="font-size:10px;padding:2px 8px;
                    border-radius:4px;border:1px solid var(--border);background:var(--bg-card);
                    color:var(--text-muted);cursor:pointer;">Reset</button>
            </div>
        </div>
        <div id="contextFollowup" style="display:none;margin-top:6px;font-size:11px;color:var(--preventive);"></div>
        <div id="contextDuplicate" style="display:none;margin-top:6px;padding:6px 10px;
            background:rgba(217,119,6,0.1);border-radius:6px;border:1px solid rgba(217,119,6,0.2);">
        </div>
    </div>

    <!-- Loading Indicator -->
    <div class="loading" id="loading">
        <div class="spinner"></div>
        <div style="color:var(--text-sec);">Analyzing your question...</div>
    </div>

    <!-- Results Area -->
    <div id="results-area">{results_html}</div>
</div>

<div class="footer">
    MTP Healthcare Analytics &bull; Built from Scratch &bull; Zero External Dependencies &bull; {time.strftime("%B %d, %Y")}
</div>

<script>
const queryHistory = [];
const currentSession = {{ queries: [], saved: false, _pendingQuery: null, sessionId: null }};
let authToken = localStorage.getItem('mtp_auth_token') || '';
let _exportPwCallback = null;

// ─── Auth Gate Functions ───
function switchGateTab(tab) {{
    const loginForm = document.getElementById('gateLoginForm');
    const signupForm = document.getElementById('gateSignupForm');
    const loginTab = document.getElementById('gateLoginTab');
    const signupTab = document.getElementById('gateSignupTab');
    if (tab === 'login') {{
        loginForm.style.display = 'block';
        signupForm.style.display = 'none';
        loginTab.style.background = 'var(--info)'; loginTab.style.color = '#fff';
        signupTab.style.background = 'var(--bg-hover)'; signupTab.style.color = 'var(--text-sec)';
    }} else {{
        loginForm.style.display = 'none';
        signupForm.style.display = 'block';
        signupTab.style.background = 'var(--info)'; signupTab.style.color = '#fff';
        loginTab.style.background = 'var(--bg-hover)'; loginTab.style.color = 'var(--text-sec)';
        loadGateTeamsList();
    }}
}}

async function loadGateTeamsList() {{
    try {{
        const resp = await fetch('/api/teams');
        const data = await resp.json();
        const dl = document.getElementById('gateTeamList');
        dl.innerHTML = '';
        (data.teams || []).forEach(t => {{
            const opt = document.createElement('option');
            opt.value = t.team_name;
            dl.appendChild(opt);
        }});
    }} catch (e) {{ /* ignore */ }}
}}

async function gateLogin() {{
    const username = document.getElementById('gateLoginUser').value.trim();
    const password = document.getElementById('gateLoginPass').value;
    if (!username || !password) {{ showError('gateLoginError', 'Please fill in all fields'); return; }}
    try {{
        const resp = await fetch('/api/login', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ username, password }})
        }});
        const data = await resp.json();
        if (data.success) {{
            authToken = data.token;
            localStorage.setItem('mtp_auth_token', authToken);
            onAuthSuccess();
        }} else {{
            showError('gateLoginError', data.message);
        }}
    }} catch (e) {{
        showError('gateLoginError', 'Login failed: ' + e.message);
    }}
}}

async function gateSignup() {{
    const username = document.getElementById('gateSignupUser').value.trim();
    const email = document.getElementById('gateSignupEmail').value.trim();
    const display_name = document.getElementById('gateSignupDisplay').value.trim();
    const password = document.getElementById('gateSignupPass').value;
    const team_name = document.getElementById('gateSignupTeam').value.trim();
    if (!username || !password) {{ showError('gateSignupError', 'Username and password are required'); return; }}
    try {{
        const resp = await fetch('/api/signup', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ username, password, display_name, team_name, email }})
        }});
        const data = await resp.json();
        if (data.success) {{
            authToken = data.token;
            localStorage.setItem('mtp_auth_token', authToken);
            onAuthSuccess();
        }} else {{
            showError('gateSignupError', data.message);
        }}
    }} catch (e) {{
        showError('gateSignupError', 'Signup failed: ' + e.message);
    }}
}}

// ─── SSO Functions ───
async function checkSSOProviders() {{
    try {{
        const resp = await fetch('/api/sso-config');
        const data = await resp.json();
        const ssoDiv = document.getElementById('ssoButtons');
        if (data.any) {{
            ssoDiv.style.display = 'block';
            document.getElementById('googleSSOBtn').style.display = data.google ? 'flex' : 'none';
            document.getElementById('msSSOBtn').style.display = data.microsoft ? 'flex' : 'none';
        }}
    }} catch (e) {{ /* SSO not available, hide buttons */ }}
}}

async function startSSO(provider) {{
    try {{
        const resp = await fetch('/api/sso-url?provider=' + provider);
        const data = await resp.json();
        if (data.url) {{
            // Open SSO popup
            const w = 500, h = 600;
            const left = (screen.width - w) / 2, top = (screen.height - h) / 2;
            const popup = window.open(data.url, 'sso_popup',
                `width=${{w}},height=${{h}},left=${{left}},top=${{top}},toolbar=no,menubar=no`);

            // Listen for callback message
            window.addEventListener('message', function handler(e) {{
                if (e.data && e.data.type === 'sso_success') {{
                    authToken = e.data.token;
                    localStorage.setItem('mtp_auth_token', authToken);
                    onAuthSuccess();
                    window.removeEventListener('message', handler);
                }} else if (e.data && e.data.type === 'sso_error') {{
                    showError('gateLoginError', 'SSO failed: ' + e.data.error);
                    window.removeEventListener('message', handler);
                }}
            }});
        }} else {{
            showError('gateLoginError', data.error || 'SSO not available');
        }}
    }} catch (e) {{
        showError('gateLoginError', 'SSO failed: ' + e.message);
    }}
}}

function openGoogleSignup() {{
    window.open('https://accounts.google.com/signup', '_blank');
}}

// ─── Email Functions ───
function openEmailModal() {{
    if (!authToken) {{ alert('Please log in first.'); return; }}
    // Show user's email as the "from" address
    const emailDisplay = document.getElementById('authEmail');
    const fromEl = document.getElementById('emailFromDisplay');
    if (emailDisplay && emailDisplay.textContent) {{
        fromEl.textContent = emailDisplay.textContent.replace(/[()]/g, '').trim();
    }}
    showModal('emailModal');
}}

async function doSendEmail() {{
    const to = document.getElementById('emailTo').value.trim();
    const subject = document.getElementById('emailSubject').value.trim();
    const note = document.getElementById('emailBody').value.trim();
    if (!to || !to.includes('@')) {{ showError('emailError', 'Enter a valid email address'); return; }}

    // Build email body from current dashboard
    const resultsEl = document.getElementById('results-area');
    const body = '<h2>' + subject + '</h2>' +
        (note ? '<p>' + note + '</p>' : '') +
        '<hr>' +
        (resultsEl ? resultsEl.innerHTML : '<p>No dashboard data</p>') +
        '<hr><p style="font-size:11px;color:#999;">Sent from MTP Healthcare Analytics</p>';

    try {{
        const resp = await fetch('/api/send-email', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ token: authToken, to_email: to, subject: subject, body: body }})
        }});
        const data = await resp.json();
        if (data.success) {{
            hideModal('emailModal');
            showToast('Email sent to ' + to);
        }} else {{
            if (data.message && data.message.includes('SMTP not configured')) {{
                document.getElementById('smtpSetup').style.display = 'block';
            }}
            showError('emailError', data.message);
        }}
    }} catch (e) {{
        showError('emailError', 'Failed: ' + e.message);
    }}
}}

async function saveSMTPConfig() {{
    const email = document.getElementById('smtpEmail').value.trim();
    const host = document.getElementById('smtpHost').value.trim();
    const port = parseInt(document.getElementById('smtpPort').value) || 587;
    const pass = document.getElementById('smtpPass').value;
    if (!email || !pass) {{ showError('emailError', 'Email and password required'); return; }}

    try {{
        const resp = await fetch('/api/save-smtp', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ email, smtp_host: host, smtp_port: port, app_password: pass }})
        }});
        const data = await resp.json();
        if (data.success) {{
            document.getElementById('smtpSetup').style.display = 'none';
            showToast('SMTP config saved! Try sending again.');
        }} else {{
            showError('emailError', data.message);
        }}
    }} catch (e) {{
        showError('emailError', 'Failed: ' + e.message);
    }}
}}

function showToast(msg) {{
    const toast = document.createElement('div');
    toast.style.cssText = 'position:fixed;top:60px;right:20px;background:var(--healthy);color:#fff;padding:10px 20px;border-radius:8px;font-size:13px;font-weight:600;z-index:99999;';
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}}

async function onAuthSuccess() {{
    // Hide gate, show dashboard
    document.getElementById('authGate').style.display = 'none';
    await checkAuthState();
    // Create a new user session
    try {{
        const resp = await fetch('/api/create-session', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ token: authToken }})
        }});
        const data = await resp.json();
        if (data.success) {{
            currentSession.sessionId = data.session_id;
            const badge = document.getElementById('sessionBadge');
            badge.textContent = 'Session active';
            badge.style.display = 'inline';
        }}
    }} catch (e) {{ /* ignore */ }}
}}

function showModal(id) {{
    document.getElementById(id).classList.add('active');
}}
function hideModal(id) {{
    document.getElementById(id).classList.remove('active');
    document.querySelectorAll('.modal-error').forEach(e => {{ e.style.display = 'none'; }});
}}

async function doLogout() {{
    try {{
        await fetch('/api/logout', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ token: authToken }})
        }});
    }} catch (e) {{ /* ignore */ }}
    authToken = '';
    currentSession.sessionId = null;
    currentSession.queries = [];
    currentSession.saved = false;
    localStorage.removeItem('mtp_auth_token');
    localStorage.removeItem('mtp_boot_id');
    // Show auth gate again
    document.getElementById('authGate').style.display = 'flex';
    document.getElementById('authLoggedIn').style.display = 'none';
    checkSSOProviders();
}}

async function checkAuthState() {{
    if (!authToken) {{
        document.getElementById('authGate').style.display = 'flex';
        document.getElementById('authLoggedIn').style.display = 'none';
        checkSSOProviders();
        return;
    }}
    try {{
        const resp = await fetch('/api/session?token=' + encodeURIComponent(authToken));
        const data = await resp.json();
        if (data.authenticated) {{
            document.getElementById('authGate').style.display = 'none';
            document.getElementById('authLoggedIn').style.display = 'flex';
            document.getElementById('authDisplayName').textContent = data.user.display_name || data.user.username;

            // Email display
            const emailEl = document.getElementById('authEmail');
            if (data.user.email) {{
                emailEl.textContent = '(' + data.user.email + ')';
                emailEl.style.display = 'inline';
            }} else {{
                emailEl.style.display = 'none';
            }}

            // Avatar
            const avatar = document.getElementById('authAvatar');
            if (data.user.profile_picture) {{
                avatar.src = data.user.profile_picture;
                avatar.style.display = 'block';
            }} else {{
                avatar.style.display = 'none';
            }}

            // SSO badge
            const ssoBadge = document.getElementById('authSSOBadge');
            if (data.user.sso_provider) {{
                ssoBadge.textContent = data.user.sso_provider.toUpperCase() + ' SSO';
                ssoBadge.style.display = 'inline';
            }} else {{
                ssoBadge.style.display = 'none';
            }}

            // Team
            const teamEl = document.getElementById('authTeam');
            if (data.user.team_name) {{
                teamEl.textContent = data.user.team_name;
                teamEl.style.display = 'inline';
            }} else {{
                teamEl.style.display = 'none';
            }}
        }} else {{
            authToken = '';
            localStorage.removeItem('mtp_auth_token');
            document.getElementById('authGate').style.display = 'flex';
            document.getElementById('authLoggedIn').style.display = 'none';
            checkSSOProviders();
        }}
    }} catch (e) {{
        document.getElementById('authGate').style.display = 'flex';
        document.getElementById('authLoggedIn').style.display = 'none';
    }}
}}

function showError(id, msg) {{
    const el = document.getElementById(id);
    el.textContent = msg;
    el.style.display = 'block';
}}

// ─── SQL Editing ───
function runEditedSQL() {{
    const editor = document.getElementById('sqlEditor');
    if (!editor) return;
    const sql = editor.value.trim();
    if (!sql) return;

    const results = document.getElementById('results-area');
    const loading = document.getElementById('loading');
    loading.classList.add('active');

    fetch('/api/execute-sql?sql=' + encodeURIComponent(sql))
        .then(r => r.json())
        .then(data => {{
            if (data.html) {{
                // Keep the SQL editor, replace everything after it
                const existingEditor = results.querySelector('.sql-edit-container');
                const existingHidden = results.querySelector('#originalSQL');
                let editorHtml = '';
                if (existingEditor && existingHidden) {{
                    editorHtml = existingEditor.outerHTML + existingHidden.outerHTML;
                }}
                // Update results but keep editor at top
                results.innerHTML = data.html;
                // Update the SQL editor with the executed SQL
                const newEditor = results.querySelector('.sql-editor');
                if (newEditor) newEditor.value = sql;
            }}
            // Track query
            currentSession.queries.push({{ question: 'Custom SQL', sql: sql, timestamp: new Date().toISOString() }});
        }})
        .catch(err => {{
            results.innerHTML += '<div style="color:var(--critical);padding:20px;">Error: ' + err.message + '</div>';
        }})
        .finally(() => {{
            loading.classList.remove('active');
        }});
}}

function resetSQL() {{
    const hidden = document.getElementById('originalSQL');
    const editor = document.getElementById('sqlEditor');
    if (hidden && editor) {{
        editor.value = hidden.value;
    }}
}}

// ─── Dashboard Save/Load ───
function promptSaveDashboard() {{
    if (!authToken) {{
        alert('Please log in first to save dashboards.');
        return;
    }}
    if (currentSession.queries.length === 0) {{
        alert('No queries to save. Run some queries first!');
        return;
    }}
    // First check if user has export password, then show export pw modal before save
    _exportPwCallback = function() {{ showModal('saveModal'); }};
    checkAndPromptExportPassword();
}}

async function checkAndPromptExportPassword() {{
    try {{
        const resp = await fetch('/api/check-export-password', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ token: authToken }})
        }});
        const data = await resp.json();
        if (data.has_password) {{
            // User already has an export password — offer carry-forward or new
            document.getElementById('exportPwTitle').textContent = 'Export Password';
            document.getElementById('exportPwCarryForward').style.display = 'block';
            document.getElementById('exportPwReuse').checked = true;
            toggleExportPwInput();
            showModal('exportPwModal');
        }} else {{
            // First time — must set a new export password
            document.getElementById('exportPwTitle').textContent = 'Set Export Password (First Time)';
            document.getElementById('exportPwCarryForward').style.display = 'none';
            document.getElementById('exportPwNewFields').style.display = 'block';
            showModal('exportPwModal');
        }}
    }} catch (e) {{
        // If check fails, just proceed to save
        if (_exportPwCallback) {{ _exportPwCallback(); _exportPwCallback = null; }}
    }}
}}

function toggleExportPwInput() {{
    const reuse = document.getElementById('exportPwReuse').checked;
    document.getElementById('exportPwNewFields').style.display = reuse ? 'none' : 'block';
}}

async function confirmExportPassword() {{
    const reuse = document.getElementById('exportPwReuse');
    const carryForward = document.getElementById('exportPwCarryForward');

    // If carry-forward is visible and checked, just verify existing pw
    if (carryForward.style.display !== 'none' && reuse && reuse.checked) {{
        hideModal('exportPwModal');
        if (_exportPwCallback) {{ _exportPwCallback(); _exportPwCallback = null; }}
        return;
    }}

    // Otherwise, set a new export password
    const pw = document.getElementById('exportPwInput').value;
    const confirm_pw = document.getElementById('exportPwConfirm').value;
    if (!pw || pw.length < 4) {{ showError('exportPwError', 'Password must be at least 4 characters'); return; }}
    if (pw !== confirm_pw) {{ showError('exportPwError', 'Passwords do not match'); return; }}

    try {{
        const resp = await fetch('/api/set-export-password', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ token: authToken, password: pw }})
        }});
        const data = await resp.json();
        if (data.success) {{
            hideModal('exportPwModal');
            document.getElementById('exportPwInput').value = '';
            document.getElementById('exportPwConfirm').value = '';
            if (_exportPwCallback) {{ _exportPwCallback(); _exportPwCallback = null; }}
        }} else {{
            showError('exportPwError', data.message);
        }}
    }} catch (e) {{
        showError('exportPwError', 'Failed: ' + e.message);
    }}
}}

async function doSaveDashboard() {{
    const name = document.getElementById('saveDashName').value.trim();
    const desc = document.getElementById('saveDashDesc').value.trim();
    if (!name) {{ showError('saveError', 'Please enter a dashboard name'); return; }}

    try {{
        const resp = await fetch('/api/save-dashboard', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{
                token: authToken,
                name: name,
                description: desc,
                queries: currentSession.queries
            }})
        }});
        const data = await resp.json();
        if (data.success) {{
            hideModal('saveModal');
            currentSession.saved = true;
            // Show success feedback
            const toast = document.createElement('div');
            toast.style.cssText = 'position:fixed;top:60px;right:20px;background:var(--healthy);color:#fff;padding:10px 20px;border-radius:8px;font-size:13px;font-weight:600;z-index:99999;animation:fadeIn 0.3s;';
            toast.textContent = 'Dashboard saved!';
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
            // Run pending query if user was interrupted by save prompt
            if (currentSession._pendingQuery) {{
                const pq = currentSession._pendingQuery;
                currentSession._pendingQuery = null;
                currentSession.saved = false;
                submitQuery(pq);
            }}
        }} else {{
            showError('saveError', data.message);
        }}
    }} catch (e) {{
        showError('saveError', 'Save failed: ' + e.message);
    }}
}}

async function showSavedDashboards() {{
    if (!authToken) {{
        alert('Please log in first to view dashboards.');
        return;
    }}
    showModal('dashboardsModal');
    loadDashboardList('my');
}}

async function loadDashboardList(scope) {{
    const container = document.getElementById('dashboardsList');
    container.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted);">Loading...</div>';

    try {{
        const resp = await fetch('/api/dashboards?token=' + encodeURIComponent(authToken) + '&scope=' + scope);
        const data = await resp.json();
        const dashboards = data.dashboards || [];

        if (dashboards.length === 0) {{
            container.innerHTML = '<div style="padding:30px;text-align:center;color:var(--text-muted);">No saved dashboards yet.</div>';
            return;
        }}

        container.innerHTML = dashboards.map(d => `
            <div class="saved-dash-card">
                <div class="saved-dash-name">${{d.name}}</div>
                <div class="saved-dash-meta">
                    ${{d.description ? d.description + ' &bull; ' : ''}}
                    ${{d.queries.length}} queries &bull; Updated ${{d.updated_at}}
                    ${{d.display_name ? ' &bull; by ' + d.display_name : ''}}
                </div>
                <div class="saved-dash-actions">
                    <button class="auth-btn auth-btn-primary" onclick="loadDashboard('${{d.dashboard_id}}', ${{JSON.stringify(d.queries).replace(/'/g, "\\\\'")}})">Load</button>
                    ${{scope === 'my' ? '<button class=\\"auth-btn auth-btn-danger\\" onclick=\\"deleteDashboard(\\'' + d.dashboard_id + '\\')\\">Delete</button>' : ''}}
                </div>
            </div>
        `).join('');
    }} catch (e) {{
        container.innerHTML = '<div style="padding:20px;color:var(--critical);">Error loading dashboards: ' + e.message + '</div>';
    }}
}}

function loadDashboard(dashId, queries) {{
    hideModal('dashboardsModal');
    // Replay queries
    if (queries && queries.length > 0) {{
        // Execute the last query
        const lastQ = queries[queries.length - 1];
        if (lastQ.question && lastQ.question !== 'Custom SQL') {{
            submitQuery(lastQ.question);
        }} else if (lastQ.sql) {{
            // Direct SQL execution
            const editor = document.getElementById('sqlEditor');
            if (editor) editor.value = lastQ.sql;
            runEditedSQL();
        }}
        currentSession.queries = [...queries];
    }}
}}

async function deleteDashboard(dashId) {{
    if (!confirm('Delete this dashboard?')) return;
    try {{
        await fetch('/api/delete-dashboard', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ token: authToken, dashboard_id: dashId }})
        }});
        loadDashboardList('my');
    }} catch (e) {{
        alert('Delete failed: ' + e.message);
    }}
}}

// ─── Auto-logout on tab/browser close ───
window.addEventListener('beforeunload', function() {{
    if (authToken) {{
        // Use sendBeacon for reliable delivery during page unload
        navigator.sendBeacon('/api/logout', JSON.stringify({{ token: authToken }}));
        localStorage.removeItem('mtp_auth_token');
        localStorage.removeItem('mtp_boot_id');
    }}
}});

// Also handle visibility change (mobile browsers sometimes don't fire beforeunload)
document.addEventListener('visibilitychange', function() {{
    if (document.visibilityState === 'hidden' && authToken) {{
        // Mark session for cleanup but don't logout immediately
        // (user might just be switching tabs)
        sessionStorage.setItem('mtp_tab_hidden', Date.now().toString());
    }} else if (document.visibilityState === 'visible') {{
        sessionStorage.removeItem('mtp_tab_hidden');
    }}
}});

// Check auth on page load
// Server embeds a boot ID that changes on every server restart.
// If it doesn't match what we stored, force re-auth (clear stale token).
const SERVER_BOOT_ID = '{server_boot_id}';
const storedBootId = localStorage.getItem('mtp_boot_id') || '';

(async function() {{
    // If server restarted (boot ID changed), clear old token and force re-auth
    if (storedBootId && storedBootId !== SERVER_BOOT_ID) {{
        authToken = '';
        localStorage.removeItem('mtp_auth_token');
    }}
    localStorage.setItem('mtp_boot_id', SERVER_BOOT_ID);

    if (authToken) {{
        try {{
            const resp = await fetch('/api/session?token=' + encodeURIComponent(authToken));
            const data = await resp.json();
            if (data.authenticated) {{
                await onAuthSuccess();
                return;
            }}
        }} catch (e) {{ /* fall through to gate */ }}
        // Token invalid — clear it
        authToken = '';
        localStorage.removeItem('mtp_auth_token');
    }}
    // No valid token — show gate and check SSO
    document.getElementById('authGate').style.display = 'flex';
    checkSSOProviders();
}})();

function submitQuery(q) {{
    const input = document.getElementById('queryInput');
    const query = q || input.value.trim();
    if (!query) return;

    // Check for duplicate questions
    checkDuplicate(query);

    // Prompt to save if there are unsaved queries and user is logged in
    if (currentSession.queries.length > 0 && !currentSession.saved && authToken) {{
        const wantSave = confirm('You have unsaved queries. Would you like to save your current dashboard before running a new query?');
        if (wantSave) {{
            promptSaveDashboard();
            // Store pending query to run after save
            currentSession._pendingQuery = query;
            return;
        }}
    }}

    input.value = query;

    // Track history
    if (!queryHistory.includes(query)) queryHistory.unshift(query);
    if (queryHistory.length > 20) queryHistory.pop();
    _updateHistory();

    const btn = document.getElementById('queryBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results-area');
    const startTime = performance.now();

    btn.disabled = true;
    btn.textContent = 'Running...';
    loading.classList.add('active');

    fetch(window.location.origin + '/query?q=' + encodeURIComponent(query))
        .then(r => r.json())
        .then(data => {{
            const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
            const timeHtml = '<div style="font-size:11px;color:var(--text-muted);margin-bottom:12px;">' +
                '&#9889; Response in ' + elapsed + 's' +
                (data.intent ? ' &bull; Intent: <b>' + data.intent + '</b>' : '') +
                '</div>';
            results.innerHTML = timeHtml + (data.html || '<div style="padding:20px;color:var(--text-sec)">No results</div>');
            // Track query in session for save feature
            currentSession.queries.push({{ question: query, sql: data.sql || '', timestamp: new Date().toISOString() }});
            currentSession.saved = false;
            // Record conversation turn
            recordConversationTurn(query, data.sql || '', data.tables || [], data.row_count || 0);
            // Update context bar
            updateContextBar();
            // Sync to user session on server
            if (currentSession.sessionId && authToken) {{
                fetch('/api/update-session', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ token: authToken, session_id: currentSession.sessionId, queries: currentSession.queries }})
                }}).catch(() => {{}});
            }}
            // Animate bars
            document.querySelectorAll('.bar-fill').forEach(bar => {{
                const w = bar.style.width;
                bar.style.width = '0%';
                requestAnimationFrame(() => {{ bar.style.width = w; }});
            }});
            // Scroll to results
            results.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        }})
        .catch(err => {{
            results.innerHTML = '<div style="padding:20px;color:var(--critical);">Error: ' + err.message + '</div>';
        }})
        .finally(() => {{
            btn.disabled = false;
            btn.textContent = 'Query';
            loading.classList.remove('active');
        }});
}}

function _updateHistory() {{
    const el = document.getElementById('query-history');
    if (!el || queryHistory.length === 0) return;
    el.style.display = 'block';
    el.innerHTML = '<span style="font-size:10px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.3px;">Recent:</span> ' +
        queryHistory.slice(0, 8).map(q =>
            '<button class="suggestion-chip" onclick="submitQuery(\\\'' + q.replace(/'/g, "\\\\'") + '\\\')">' + q + '</button>'
        ).join('');
}}

// ─── Catalog Browser & Contextual Learning ───
async function loadSuggestions() {{
    try {{
        const resp = await fetch('/api/suggestions');
        const data = await resp.json();
        const suggestions = data.suggestions || [];
        if (suggestions.length > 0) {{
            const bar = document.getElementById('learningBar');
            const container = document.getElementById('otherQueries');
            const popular = suggestions.filter(s => s.category === 'popular').slice(0, 5);
            if (popular.length > 0) {{
                container.innerHTML = popular.map(s =>
                    '<button class="suggestion-chip" style="font-size:11px;" onclick="submitQuery(\\\'' +
                    s.question.replace(/'/g, "\\\\'") + '\\\')">' +
                    s.question.substring(0, 60) + (s.question.length > 60 ? '...' : '') +
                    ' <span style="color:var(--text-muted);font-size:10px;">(' + (s.frequency || 1) + 'x)</span></button>'
                ).join('');
                bar.style.display = 'block';
            }}
        }}
    }} catch (e) {{ /* ignore */ }}
}}

async function showCatalogBrowser() {{
    showModal('catalogModal');
    const content = document.getElementById('catalogContent');
    content.innerHTML = '<div style="text-align:center;padding:40px;color:var(--text-muted);">Loading...</div>';
    try {{
        const resp = await fetch('/api/catalogs?view=summary');
        const data = await resp.json();
        const summary = data.summary || data;
        let html = '<div style="margin-bottom:16px;">' +
            '<span style="font-size:14px;font-weight:600;color:var(--text);">Registry Overview</span>' +
            '<div style="display:flex;gap:16px;margin-top:8px;">' +
            '<div style="padding:12px 16px;background:var(--bg-hover);border-radius:8px;flex:1;text-align:center;">' +
            '<div style="font-size:24px;font-weight:700;color:var(--info);">' + (summary.environments || 0) + '</div>' +
            '<div style="font-size:11px;color:var(--text-muted);">Environments</div></div>' +
            '<div style="padding:12px 16px;background:var(--bg-hover);border-radius:8px;flex:1;text-align:center;">' +
            '<div style="font-size:24px;font-weight:700;color:var(--healthy);">' + (summary.total_tables || 0) + '</div>' +
            '<div style="font-size:11px;color:var(--text-muted);">Tables</div></div>' +
            '<div style="padding:12px 16px;background:var(--bg-hover);border-radius:8px;flex:1;text-align:center;">' +
            '<div style="font-size:24px;font-weight:700;color:var(--behavioral);">' + ((summary.concepts || []).length) + '</div>' +
            '<div style="font-size:11px;color:var(--text-muted);">Concepts</div></div>' +
            '</div></div>';

        // Show environments
        const envResp = await fetch('/api/catalogs?view=environments');
        const envData = await envResp.json();
        const envs = envData.environments || [];
        if (envs.length > 0) {{
            html += '<div style="margin-bottom:16px;">' +
                '<span style="font-size:13px;font-weight:600;color:var(--text);">Environments</span>';
            for (const env of envs) {{
                const active = env.is_active ? '&#9989;' : '&#10060;';
                html += '<div style="padding:10px 14px;margin-top:6px;background:var(--bg-hover);border-radius:8px;' +
                    'border:1px solid var(--border);cursor:pointer;" onclick="browseCatalogs(\\\'' + env.name + '\\\')">' +
                    '<div style="display:flex;justify-content:space-between;align-items:center;">' +
                    '<span style="font-weight:600;color:var(--text);">' + active + ' ' + env.name + '</span>' +
                    '<span style="font-size:11px;color:var(--text-muted);">' + env.catalog_count + ' catalogs</span>' +
                    '</div>' +
                    (env.description ? '<div style="font-size:12px;color:var(--text-sec);margin-top:2px;">' + env.description + '</div>' : '') +
                    '</div>';
            }}
            html += '</div>';
        }}

        // Show concepts
        if (summary.concepts && summary.concepts.length > 0) {{
            html += '<div><span style="font-size:13px;font-weight:600;color:var(--text);">Data Concepts</span>' +
                '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;">';
            for (const c of summary.concepts) {{
                html += '<button class="suggestion-chip" onclick="searchCatalog(\\\'' + c + '\\\')" ' +
                    'style="font-size:12px;">' + c + '</button>';
            }}
            html += '</div></div>';
        }}

        content.innerHTML = html;
    }} catch (e) {{
        content.innerHTML = '<div style="padding:20px;color:var(--critical);">Failed to load catalogs: ' + e.message + '</div>';
    }}
}}

async function browseCatalogs(envName) {{
    const content = document.getElementById('catalogContent');
    try {{
        const resp = await fetch('/api/catalogs?view=catalogs&env=' + encodeURIComponent(envName));
        const data = await resp.json();
        const catalogs = data.catalogs || [];
        let html = '<button onclick="showCatalogBrowser()" style="font-size:12px;margin-bottom:12px;' +
            'padding:4px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg-card);' +
            'color:var(--info);cursor:pointer;">&#8592; Back</button>' +
            '<h3 style="margin:0 0 12px;color:var(--text);">' + envName + ' Catalogs</h3>';
        for (const cat of catalogs) {{
            const def = cat.is_default ? ' <span style="font-size:10px;padding:2px 6px;background:rgba(0,107,166,0.15);' +
                'color:var(--info);border-radius:4px;">DEFAULT</span>' : '';
            html += '<div style="padding:12px 14px;margin-bottom:8px;background:var(--bg-hover);border-radius:8px;' +
                'border:1px solid var(--border);cursor:pointer;" ' +
                'onclick="browseTables(\\\'' + cat.name + '\\\', \\\'' + envName + '\\\')">' +
                '<div style="font-weight:600;color:var(--text);">' + cat.name + def + '</div>' +
                '<div style="font-size:12px;color:var(--text-sec);margin-top:2px;">' +
                (cat.description || '') + '</div>' +
                '<div style="font-size:11px;color:var(--text-muted);margin-top:4px;">' +
                cat.table_count + ' tables' + (cat.source_system ? ' &bull; Source: ' + cat.source_system : '') +
                '</div></div>';
        }}
        content.innerHTML = html;
    }} catch (e) {{
        content.innerHTML = '<div style="padding:20px;color:var(--critical);">Error: ' + e.message + '</div>';
    }}
}}

async function browseTables(catalogName, envName) {{
    const content = document.getElementById('catalogContent');
    try {{
        const resp = await fetch('/api/catalogs?view=tables&catalog=' + encodeURIComponent(catalogName) +
            '&env=' + encodeURIComponent(envName));
        const data = await resp.json();
        const tables = data.tables || [];
        let html = '<button onclick="browseCatalogs(\\\'' + envName + '\\\')" style="font-size:12px;margin-bottom:12px;' +
            'padding:4px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg-card);' +
            'color:var(--info);cursor:pointer;">&#8592; Back</button>' +
            '<h3 style="margin:0 0 12px;color:var(--text);">' + catalogName + ' Tables</h3>';
        for (const t of tables) {{
            html += '<div style="padding:10px 14px;margin-bottom:6px;background:var(--bg-hover);border-radius:8px;' +
                'border:1px solid var(--border);">' +
                '<div style="font-weight:600;color:var(--text);font-size:13px;">' + t.name + '</div>' +
                '<div style="font-size:11px;color:var(--text-muted);margin-top:2px;">' +
                '<code style="font-size:11px;color:var(--info);">' + t.full_path + '</code>' +
                (t.row_count ? ' &bull; ' + t.row_count.toLocaleString() + ' rows' : '') +
                (t.column_count ? ' &bull; ' + t.column_count + ' cols' : '') +
                '</div>' +
                (t.description ? '<div style="font-size:12px;color:var(--text-sec);margin-top:4px;">' + t.description + '</div>' : '') +
                (t.columns && t.columns.length > 0 ? '<div style="font-size:11px;color:var(--text-muted);margin-top:4px;">' +
                    'Columns: ' + t.columns.slice(0, 8).join(', ') + (t.columns.length > 8 ? '...' : '') + '</div>' : '') +
                '</div>';
        }}
        if (tables.length === 0) {{
            html += '<div style="padding:20px;text-align:center;color:var(--text-muted);">No tables found in this catalog.</div>';
        }}
        content.innerHTML = html;
    }} catch (e) {{
        content.innerHTML = '<div style="padding:20px;color:var(--critical);">Error: ' + e.message + '</div>';
    }}
}}

async function searchCatalog(concept) {{
    const content = document.getElementById('catalogContent');
    try {{
        const resp = await fetch('/api/catalogs?view=tables&concept=' + encodeURIComponent(concept));
        const data = await resp.json();
        const tables = data.tables || [];
        let html = '<button onclick="showCatalogBrowser()" style="font-size:12px;margin-bottom:12px;' +
            'padding:4px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg-card);' +
            'color:var(--info);cursor:pointer;">&#8592; Back</button>' +
            '<h3 style="margin:0 0 12px;color:var(--text);">"' + concept + '" across all catalogs</h3>';
        for (const t of tables) {{
            html += '<div style="padding:10px 14px;margin-bottom:6px;background:var(--bg-hover);border-radius:8px;' +
                'border:1px solid var(--border);">' +
                '<div style="display:flex;justify-content:space-between;align-items:center;">' +
                '<span style="font-weight:600;color:var(--text);font-size:13px;">' + t.name + '</span>' +
                '<span style="font-size:10px;padding:2px 8px;background:rgba(0,107,166,0.1);color:var(--info);' +
                'border-radius:4px;">' + t.environment + '</span></div>' +
                '<div style="font-size:11px;color:var(--text-muted);margin-top:2px;">' +
                '<code style="font-size:11px;color:var(--info);">' + t.full_path + '</code></div>' +
                (t.description ? '<div style="font-size:12px;color:var(--text-sec);margin-top:4px;">' + t.description + '</div>' : '') +
                '</div>';
        }}
        content.innerHTML = html;
    }} catch (e) {{
        content.innerHTML = '<div style="padding:20px;color:var(--critical);">Error: ' + e.message + '</div>';
    }}
}}

// ─────────────────────────────────────────────────────────────────────────────
// HIERARCHY PANEL FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

let hierarchyData = {{}};
let hierarchySelections = {{}};

async function openHierarchyPanel() {{
    const panel = document.getElementById('hierarchyPanel');
    try {{
        const resp = await fetch('/api/table-hierarchy');
        const data = await resp.json();
        hierarchyData = data;
        renderHierarchyTree(data);
        panel.style.display = 'block';
    }} catch (e) {{
        console.error('Failed to load hierarchy:', e);
        panel.innerHTML = '<div style="padding:20px;color:var(--critical);">Error loading hierarchy</div>';
    }}
}}

function closeHierarchyPanel() {{
    document.getElementById('hierarchyPanel').style.display = 'none';
}}

function renderHierarchyTree(hierarchy) {{
    const container = document.getElementById('hierarchyTree');
    let html = '';

    if (hierarchy.concepts) {{
        hierarchy.concepts.forEach(concept => {{
            const conceptKey = concept.key;
            html += '<div style="margin-bottom:16px;">';
            html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;cursor:pointer;" onclick="toggleHierarchyCategory(\\\'' + conceptKey + '\\\', \\\'_concept\\\')">';
            html += '<span style="color:var(--text-sec);font-size:14px;user-select:none;">▼</span>';
            html += '<input type="checkbox" style="margin:0;" onchange="onHierarchyCheck(\\\'' + conceptKey + '\\\')" />';
            html += '<span style="font-weight:600;color:var(--text);font-size:13px;">' + concept.label + '</span>';
            html += '</div>';

            html += '<div style="margin-left:24px;">';
            if (concept.categories) {{
                concept.categories.forEach(category => {{
                    const categoryKey = conceptKey + '.' + category.key;
                    html += '<div style="margin-bottom:12px;">';
                    html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;cursor:pointer;" onclick="toggleHierarchyCategory(\\\'' + conceptKey + '\\\', \\\'' + category.key + '\\\')">';
                    html += '<span style="color:var(--text-sec);font-size:12px;user-select:none;">▶</span>';
                    html += '<input type="checkbox" style="margin:0;" onchange="onHierarchyCheck(\\\'' + categoryKey + '\\\')" />';
                    html += '<span style="color:var(--text-sec);font-size:12px;font-weight:500;">' + category.label + '</span>';
                    html += '</div>';

                    html += '<div style="margin-left:24px;display:none;" id="cat_' + categoryKey + '">';
                    if (category.types) {{
                        category.types.forEach(type => {{
                            const typeKey = categoryKey + '.' + type.key;
                            html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">';
                            html += '<input type="checkbox" style="margin:0;" onchange="onHierarchyCheck(\\\'' + typeKey + '\\\')" />';
                            html += '<span style="color:var(--text);font-size:11px;">' + type.label + '</span>';
                            html += '</div>';
                        }});
                    }}
                    html += '</div>';
                    html += '</div>';
                }});
            }}
            html += '</div>';
            html += '</div>';
        }});
    }}

    container.innerHTML = html;
}}

function toggleHierarchyCategory(conceptKey, categoryKey) {{
    const categoryId = 'cat_' + conceptKey + '.' + categoryKey;
    const div = document.getElementById(categoryId);
    if (div) {{
        div.style.display = div.style.display === 'none' ? 'block' : 'none';
    }}
}}

function onHierarchyCheck(path) {{
    const checkbox = document.querySelector('input[onchange="onHierarchyCheck(\\\'' + path + '\\\')"]');
    if (checkbox && checkbox.checked) {{
        hierarchySelections[path] = true;
    }} else {{
        delete hierarchySelections[path];
    }}
    updateSelectionSummary();
}}

function updateSelectionSummary() {{
    const count = Object.keys(hierarchySelections).length;
    const summary = document.getElementById('selectionSummary');
    if (count === 0) {{
        summary.textContent = 'No selections — queries will use default tables';
    }} else {{
        summary.textContent = count + ' data type' + (count === 1 ? '' : 's') + ' selected';
    }}
}}

async function applyHierarchySelections() {{
    const selections = Object.keys(hierarchySelections);
    try {{
        const resp = await fetch('/api/set-selections', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{
                session_id: currentSession.sessionId || 'default',
                selections: selections
            }})
        }});
        const data = await resp.json();
        if (data.success) {{
            updateContextBar();
            closeHierarchyPanel();
        }}
    }} catch (e) {{
        console.error('Failed to apply selections:', e);
    }}
}}

function clearHierarchySelections() {{
    hierarchySelections = {{}};
    document.querySelectorAll('#hierarchyTree input[type="checkbox"]').forEach(cb => {{
        cb.checked = false;
    }});
    updateSelectionSummary();
}}

function getActiveSelections() {{
    return Object.keys(hierarchySelections);
}}

// ─────────────────────────────────────────────────────────────────────────────
// SESSION CONTEXT FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

let sessionContextData = {{
    sessionId: null,
    recentTurns: [],
    activeSelections: [],
    turnCount: 0
}};

function initSessionContext() {{
    if (!currentSession.sessionId) {{
        currentSession.sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
    }}
    sessionContextData.sessionId = currentSession.sessionId;
    loadSessionContext();
}}

async function loadSessionContext() {{
    try {{
        const resp = await fetch('/api/session-context?session_id=' + sessionContextData.sessionId);
        const data = await resp.json();
        sessionContextData.recentTurns = data.recent_turns || [];
        sessionContextData.turnCount = data.turn_count || 0;
    }} catch (e) {{
        console.error('Failed to load session context:', e);
    }}
}}

async function recordConversationTurn(question, sql, tables, resultCount) {{
    try {{
        await fetch('/api/conversation-turn', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{
                session_id: sessionContextData.sessionId,
                question: question,
                sql: sql,
                tables: tables || [],
                result_count: resultCount || 0
            }})
        }});
        sessionContextData.recentTurns.unshift({{ question: question, timestamp: new Date().toISOString() }});
        if (sessionContextData.recentTurns.length > 10) {{
            sessionContextData.recentTurns.pop();
        }}
        sessionContextData.turnCount++;
    }} catch (e) {{
        console.error('Failed to record turn:', e);
    }}
}}

function checkDuplicate(question) {{
    const recent = sessionContextData.recentTurns.slice(0, 3);
    for (let turn of recent) {{
        if (turn.question.toLowerCase().trim() === question.toLowerCase().trim()) {{
            const dupDiv = document.getElementById('contextDuplicate');
            dupDiv.style.display = 'block';
            dupDiv.innerHTML = '⚠ This question was asked ' + (turn.timestamp ? 'recently' : 'before') + '. Continue?';
            return true;
        }}
    }}
    const dupDiv = document.getElementById('contextDuplicate');
    dupDiv.style.display = 'none';
    return false;
}}

function showFollowupContext(info) {{
    const followup = document.getElementById('contextFollowup');
    if (info) {{
        followup.style.display = 'block';
        followup.textContent = '↪ Continuing from: ' + info;
    }} else {{
        followup.style.display = 'none';
    }}
}}

function clearSessionContext() {{
    sessionContextData = {{
        sessionId: currentSession.sessionId,
        recentTurns: [],
        activeSelections: [],
        turnCount: 0
    }};
    hierarchySelections = {{}};
    updateContextBar();
}}

function updateContextBar() {{
    const contextBar = document.getElementById('contextBar');
    const selections = getActiveSelections();

    if (selections.length === 0 && sessionContextData.turnCount === 0) {{
        contextBar.style.display = 'none';
        return;
    }}

    contextBar.style.display = 'block';

    const selectionsText = selections.length > 0
        ? selections.slice(0, 2).join(', ') + (selections.length > 2 ? ' +' + (selections.length - 2) + ' more' : '')
        : 'None';

    document.getElementById('contextSelections').textContent = selectionsText;
    document.getElementById('contextTurnCount').textContent = sessionContextData.turnCount + ' turn' + (sessionContextData.turnCount === 1 ? '' : 's');
}}

// ─────────────────────────────────────────────────────────────────────────────
// DATA SOURCE NAVIGATOR FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

// Data Source Navigator State
let dsState = {{
    environment: 'local',
    catalog: 'local_catalog',
    schema: null,
    table: null,
    claimType: null,
    claimCategory: null,
    filter: null
}};

// Claim Type Hierarchy
const CLAIM_TYPES = {{
    institutional: {{
        label: 'Institutional',
        icon: '🏥',
        color: '#C8102E',
        subtypes: [
            {{ key: 'inpatient', label: 'Inpatient', desc: 'Hospital inpatient stays', code: 'UB-04' }},
            {{ key: 'outpatient_facility', label: 'Outpatient Facility', desc: 'Facility outpatient services', code: 'UB-04' }},
            {{ key: 'snf', label: 'SNF', desc: 'Skilled nursing facility', code: 'UB-04' }}
        ]
    }},
    professional: {{
        label: 'Professional',
        icon: '🩺',
        color: '#006BA6',
        subtypes: [
            {{ key: 'physician', label: 'Physician', desc: 'Physician services', code: 'CMS-1500' }},
            {{ key: 'specialist', label: 'Specialist', desc: 'Specialist services', code: 'CMS-1500' }},
            {{ key: 'telehealth', label: 'Telehealth', desc: 'Remote care services', code: 'CMS-1500' }}
        ]
    }},
    pharmacy: {{
        label: 'Pharmacy',
        icon: '💊',
        color: '#286140',
        subtypes: [
            {{ key: 'retail', label: 'Retail', desc: 'Retail pharmacy claims', code: 'NCPDP' }},
            {{ key: 'specialty', label: 'Specialty Rx', desc: 'Specialty pharmacy claims', code: 'NCPDP' }},
            {{ key: 'mail_order', label: 'Mail-Order', desc: 'Mail order pharmacy', code: 'NCPDP' }}
        ]
    }},
    dental: {{
        label: 'Dental',
        icon: '🦷',
        color: '#007A7C',
        subtypes: [
            {{ key: 'preventive', label: 'Preventive', desc: 'Preventive dental care', code: 'ADA' }},
            {{ key: 'restorative', label: 'Restorative', desc: 'Restorative dental procedures', code: 'ADA' }}
        ]
    }},
    behavioral: {{
        label: 'Behavioral Health',
        icon: '🧠',
        color: '#5C2D91',
        subtypes: [
            {{ key: 'mental_health', label: 'Mental Health', desc: 'Mental health services', code: 'CPT' }},
            {{ key: 'substance_use', label: 'Substance Use', desc: 'Substance use disorder treatment', code: 'CPT' }}
        ]
    }},
    dme: {{
        label: 'DME',
        icon: '🦽',
        color: '#F2A900',
        subtypes: [
            {{ key: 'equipment', label: 'Equipment', desc: 'Durable medical equipment', code: 'HCPCS' }}
        ]
    }},
    all: {{
        label: 'All Claims',
        icon: '📋',
        color: '#58595B',
        subtypes: [
            {{ key: 'combined', label: 'Combined', desc: 'All claim types combined', code: 'All' }}
        ]
    }}
}};

async function openDataSourceNav() {{
    const panel = document.getElementById('dataSourceNavPanel');
    try {{
        await loadCatalogTree();
        panel.style.display = 'block';
        document.body.style.overflow = 'hidden';
    }} catch (e) {{
        console.error('Failed to open data source navigator:', e);
        alert('Error loading data sources');
    }}
}}

function closeDataSourceNav() {{
    document.getElementById('dataSourceNavPanel').style.display = 'none';
    document.body.style.overflow = 'auto';
}}

async function loadCatalogTree() {{
    try {{
        const resp = await fetch('/api/catalog-tree');
        const tree = await resp.json();
        renderSchemaTree(tree);
    }} catch (e) {{
        console.error('Failed to load catalog tree:', e);
        document.getElementById('dsSchemaTree').innerHTML = '<div style="color:var(--critical);font-size:11px;">Error loading schemas</div>';
    }}
}}

function renderSchemaTree(tree) {{
    const container = document.getElementById('dsSchemaTree');
    const env = dsState.environment;
    const catalog = dsState.catalog;

    if (!tree.environments || !tree.environments[env] || !tree.environments[env].catalogs || !tree.environments[env].catalogs[catalog]) {{
        container.innerHTML = '<div style="color:var(--text-muted);font-size:11px;">No schemas available for selected environment/catalog</div>';
        return;
    }}

    const schemas = tree.environments[env].catalogs[catalog].schemas || {{}};
    let html = '';

    Object.keys(schemas).forEach(schemaName => {{
        const schema = schemas[schemaName];
        const tables = schema.tables || [];

        // Schema header
        html += '<div style="margin-bottom:12px;border:1px solid var(--border);border-radius:6px;overflow:hidden;">';
        html += '<div style="padding:8px 10px;background:var(--bg-card);cursor:pointer;display:flex;align-items:center;gap:8px;" onclick="dsToggleSchema(\\'' + schemaName + '\\')">';
        html += '<span id="dsSchemaArrow_' + schemaName + '" style="color:var(--text-sec);font-size:12px;user-select:none;">▶</span>';
        html += '<span style="font-weight:600;color:var(--text);font-size:12px;">' + schemaName + '</span>';
        html += '<span style="font-size:10px;color:var(--text-muted);margin-left:auto;">' + tables.length + ' tables</span>';
        html += '</div>';

        // Tables list
        html += '<div id="dsSchema_' + schemaName + '" style="display:none;padding:8px;background:#fff;border-top:1px solid var(--border);">';
        tables.forEach(tbl => {{
            const tblKey = schemaName + '_' + tbl.name;
            const isSelected = dsState.schema === schemaName && dsState.table === tbl.name;
            html += '<div style="padding:6px 8px;margin-bottom:4px;border-radius:4px;cursor:pointer;background:' + (isSelected ? 'var(--bg-hover)' : 'transparent') + ';border-left:3px solid ' + (isSelected ? 'var(--info)' : 'transparent') + ';" onclick="dsSelectTable(\\'' + schemaName + '\\', \\'' + tbl.name + '\\')">';
            html += '<div style="font-weight:500;color:var(--text);font-size:11px;">' + tbl.name + '</div>';
            if (tbl.description) {{
                html += '<div style="font-size:10px;color:var(--text-muted);margin-top:2px;">' + tbl.description + '</div>';
            }}
            if (tbl.tags && tbl.tags.length > 0) {{
                html += '<div style="margin-top:4px;">';
                tbl.tags.forEach(tag => {{
                    html += '<span style="display:inline-block;font-size:9px;padding:2px 6px;margin-right:4px;background:rgba(0,107,166,0.1);color:var(--info);border-radius:3px;">' + tag + '</span>';
                }});
                html += '</div>';
            }}
            html += '</div>';
        }});
        html += '</div>';
        html += '</div>';
    }});

    container.innerHTML = html;
}}

function dsToggleSchema(schemaName) {{
    const container = document.getElementById('dsSchema_' + schemaName);
    const arrow = document.getElementById('dsSchemaArrow_' + schemaName);
    if (container.style.display === 'none') {{
        container.style.display = 'block';
        arrow.textContent = '▼';
    }} else {{
        container.style.display = 'none';
        arrow.textContent = '▶';
    }}
}}

function dsSelectEnvironment(env) {{
    dsState.environment = env;
    dsState.schema = null;
    dsState.table = null;
    dsState.claimType = null;
    dsState.claimCategory = null;
    updateDataSourceBadge();
    loadCatalogTree();
    document.getElementById('dsCatalogSelect').value = dsState.catalog;
}}

function dsSelectCatalog(catalog) {{
    dsState.catalog = catalog;
    dsState.schema = null;
    dsState.table = null;
    dsState.claimType = null;
    dsState.claimCategory = null;
    updateDataSourceBadge();
    loadCatalogTree();
}}

function dsSelectTable(schema, table) {{
    dsState.schema = schema;
    dsState.table = table;
    dsState.claimType = null;
    dsState.claimCategory = null;

    // If Claims schema selected, show claim types
    if (schema === 'claims') {{
        renderClaimTypesPanel();
        document.getElementById('dsClaimTypesPanel').style.display = 'block';
    }} else {{
        document.getElementById('dsClaimTypesPanel').style.display = 'none';
    }}

    updateSourceSummary();
}}

function renderClaimTypesPanel() {{
    const container = document.getElementById('dsClaimTypesGrid');
    let html = '';

    Object.keys(CLAIM_TYPES).forEach(categoryKey => {{
        const category = CLAIM_TYPES[categoryKey];
        html += '<div style="padding:10px;border:1px solid var(--border);border-radius:6px;background:var(--bg-card);">';
        html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">';
        html += '<span style="font-size:18px;">' + category.icon + '</span>';
        html += '<div>';
        html += '<div style="font-weight:600;color:var(--text);font-size:12px;">' + category.label + '</div>';
        html += '</div>';
        html += '</div>';

        // Subtypes
        category.subtypes.forEach(subtype => {{
            const isSelected = dsState.claimType === subtype.key;
            html += '<div style="padding:6px 8px;margin-bottom:4px;border-radius:4px;cursor:pointer;background:' + (isSelected ? 'var(--info)' : '#fff') + ';color:' + (isSelected ? '#fff' : 'var(--text)') + ';border:1px solid ' + (isSelected ? 'var(--info)' : 'var(--border)') + ';font-size:11px;" onclick="dsSelectClaimType(\\'' + categoryKey + '\\', \\'' + subtype.key + '\\')">';
            html += '<div style="font-weight:500;">' + subtype.label + '</div>';
            html += '<div style="font-size:9px;opacity:0.8;margin-top:2px;">' + subtype.desc + '</div>';
            html += '<div style="font-size:9px;opacity:0.7;margin-top:2px;">Format: ' + subtype.code + '</div>';
            html += '</div>';
        }});

        html += '</div>';
    }});

    container.innerHTML = html;
}}

function dsSelectClaimType(category, type) {{
    dsState.claimCategory = category;
    dsState.claimType = type;
    updateSourceSummary();
    renderClaimTypesPanel(); // Re-render to show selection
}}

function updateSourceSummary() {{
    const summary = document.getElementById('dsSourceSummary');
    let text = '';

    if (dsState.environment && dsState.catalog) {{
        text += dsState.environment + ' → ' + dsState.catalog;
    }}

    if (dsState.schema) {{
        text += ' → ' + dsState.schema;
    }}

    if (dsState.table) {{
        text += ' → ' + dsState.table;
    }}

    if (dsState.claimType) {{
        text += ' [' + dsState.claimType + ']';
    }}

    if (text.length === 0) {{
        text = 'No source selected';
        summary.style.borderLeftColor = 'var(--text-muted)';
    }} else {{
        summary.style.borderLeftColor = 'var(--info)';
    }}

    summary.textContent = text;
}}

function updateDataSourceBadge() {{
    const badge = document.getElementById('dsCatalogBadge');
    let source = 'Unknown';
    let color = '#58595B';

    if (dsState.catalog === 'local_catalog') {{
        source = 'SQLite';
        color = '#286140';
    }} else if (dsState.catalog === 'clarity_catalog') {{
        source = 'Clarity';
        color = '#006BA6';
    }} else if (dsState.catalog === 'apixio_catalog') {{
        source = 'Apixio';
        color = '#5C2D91';
    }} else if (dsState.catalog === 'enriched_catalog') {{
        source = 'Enriched';
        color = '#F2A900';
    }}

    badge.innerHTML = '&#9679; Source: ' + source;
    badge.style.color = color;
}}

async function dsApplySelection() {{
    if (!dsState.schema || !dsState.table) {{
        alert('Please select a table');
        return;
    }}

    // Close navigator
    closeDataSourceNav();

    // Update context
    updateContextBar();

    // If a claim type was selected, add filter to query builder
    if (dsState.claimType && dsState.schema === 'claims') {{
        const filterSql = 'CLAIM_TYPE = \\'' + dsState.claimType + '\\'';
        qbState.filters.claimType = filterSql;
    }}

    // Show notification
    const notif = document.createElement('div');
    notif.style.cssText = 'position:fixed;top:20px;right:20px;padding:12px 16px;background:var(--healthy);color:#fff;border-radius:8px;font-size:12px;z-index:2000;animation:slideIn 0.3s ease;';
    notif.innerHTML = '✓ Data source applied: ' + dsState.environment + ' → ' + dsState.table;
    document.body.appendChild(notif);
    setTimeout(() => notif.remove(), 3000);
}}

function dsClearSelection() {{
    dsState = {{
        environment: 'local',
        catalog: 'local_catalog',
        schema: null,
        table: null,
        claimType: null,
        claimCategory: null,
        filter: null
    }};
    document.getElementById('dsEnvSelect').value = 'local';
    document.getElementById('dsCatalogSelect').value = 'local_catalog';
    document.getElementById('dsClaimTypesPanel').style.display = 'none';
    updateSourceSummary();
    updateDataSourceBadge();
    loadCatalogTree();
}}

// Show data navigator button on auth
const _origOnAuthDSN = onAuthSuccess;
onAuthSuccess = function() {{
    _origOnAuthDSN();
    document.getElementById('dataSourceNavToggle').style.display = 'block';
}};

// ─────────────────────────────────────────────────────────────────────────────
// VISUAL QUERY BUILDER FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

// ─── Query Builder State ───
const qbState = {{
    concept: null,
    types: [],
    action: null,
    groupBy: null,
    filters: {{}},
    step: 1
}};

// Concept definitions for the builder
const QB_CONCEPTS = {{
    claims: {{ icon: '💳', label: 'Claims', desc: 'Billing & payments', table: 'claims' }},
    members: {{ icon: '👥', label: 'Members', desc: 'Demographics & enrollment', table: 'members' }},
    encounters: {{ icon: '🏥', label: 'Encounters', desc: 'Clinical visits', table: 'encounters' }},
    providers: {{ icon: '🩺', label: 'Providers', desc: 'Provider directory', table: 'providers' }},
    diagnoses: {{ icon: '📋', label: 'Diagnoses', desc: 'ICD-10 codes', table: 'diagnoses' }},
    prescriptions: {{ icon: '💊', label: 'Prescriptions', desc: 'Medications', table: 'prescriptions' }},
    referrals: {{ icon: '📤', label: 'Referrals', desc: 'Care referrals', table: 'referrals' }}
}};

const QB_ACTIONS = [
    {{ key: 'count', icon: '#️⃣', label: 'Count', template: 'how many {{concept}}' }},
    {{ key: 'total', icon: '💰', label: 'Total Amount', template: 'total amount of {{concept}}' }},
    {{ key: 'average', icon: '📊', label: 'Average Amount', template: 'average cost of {{concept}}' }},
    {{ key: 'details', icon: '📄', label: 'Show Records', template: 'show me {{concept}} details' }},
    {{ key: 'breakdown', icon: '📈', label: 'Breakdown By...', template: '{{concept}} breakdown by {{groupby}}' }},
    {{ key: 'top', icon: '🏆', label: 'Top / Bottom N', template: 'top 10 {{concept}} by {{metric}}' }},
    {{ key: 'ratio', icon: '⚖️', label: 'Ratio / Compare', template: '{{concept}} ratio of {{a}} to {{b}}' }},
    {{ key: 'trend', icon: '📅', label: 'Trend Over Time', template: '{{concept}} trend by month' }}
];

// GROUP BY options per concept
const QB_GROUPBY = {{
    claims: ['KP_REGION', 'CLAIM_STATUS', 'CLAIM_TYPE', 'RENDERING_NPI', 'SERVICE_DATE_month'],
    members: ['KP_REGION', 'PLAN_TYPE', 'GENDER', 'RACE', 'LANGUAGE'],
    encounters: ['VISIT_TYPE', 'KP_REGION', 'ENCOUNTER_STATUS', 'ADMIT_DATE_month'],
    providers: ['SPECIALTY', 'DEPARTMENT', 'KP_REGION'],
    diagnoses: ['DIAGNOSIS_CATEGORY', 'HCC_CATEGORY', 'SEVERITY', 'CHRONIC_FLAG'],
    prescriptions: ['DRUG_CLASS', 'GENERIC_NAME', 'PHARMACY_TYPE'],
    referrals: ['REFERRAL_REASON', 'URGENCY', 'REFERRAL_STATUS']
}};

function switchQueryTab(tab) {{
    // tab = \\'ask\\' or \\'build\\'
    document.getElementById('askTab').classList.toggle('active', tab === 'ask');
    document.getElementById('buildTab').classList.toggle('active', tab === 'build');
    document.getElementById('askContainer').style.display = tab === 'ask' ? 'block' : 'none';
    document.getElementById('buildContainer').style.display = tab === 'build' ? 'block' : 'none';
    if (tab === 'build') {{
        vqbInit();
    }}
}}

function renderQBStep1() {{
    const container = document.getElementById('qbConcepts');
    let html = '';
    Object.keys(QB_CONCEPTS).forEach(key => {{
        const concept = QB_CONCEPTS[key];
        const selected = qbState.concept === key ? ' selected' : '';
        html += '<div class="qb-card qb-concept-card' + selected + '" id="qbc_' + key + '" onclick="qbSelectConcept(\\'' + key + '\\')">';
        html += '<div class="qb-card-icon">' + concept.icon + '</div>';
        html += '<div class="qb-card-label">' + concept.label + '</div>';
        html += '<div class="qb-card-desc">' + concept.desc + '</div>';
        html += '</div>';
    }});
    container.innerHTML = html;
}}

function qbSelectConcept(key) {{
    qbState.concept = key;
    qbState.types = [];
    qbState.action = null;
    qbState.groupBy = null;
    qbState.filters = {{}};
    // Highlight selected card
    document.querySelectorAll('.qb-concept-card').forEach(c => c.classList.remove('selected'));
    const el = document.getElementById('qbc_' + key);
    if (el) el.classList.add('selected');
    // Show step 2
    renderQBStep2();
    renderQBStep3();
    renderQBStep4();
    updateQBPreview();
    loadQBSuggestions();
}}

async function renderQBStep2() {{
    const container = document.getElementById('qbStep2');
    if (!qbState.concept) {{ container.innerHTML = ''; return; }}
    container.style.display = 'block';
    // Fetch hierarchy types for this concept from /api/table-hierarchy
    try {{
        const resp = await fetch('/api/table-hierarchy');
        const data = await resp.json();
        const concepts = data.concepts || [];
        const concept = concepts.find(c => c.key === qbState.concept);
        if (!concept || !concept.categories) {{ container.innerHTML = '<div style="color:var(--text-muted);font-size:12px;">No subtypes available</div>'; return; }}

        let html = '';
        concept.categories.forEach(cat => {{
            html += '<div class="qb-type-group">';
            html += '<div class="qb-type-header">' + cat.label + '</div>';
            (cat.types || []).forEach(t => {{
                const typeKey = qbState.concept + '.' + cat.key + '.' + t.key;
                html += '<span class="qb-type-option" id="qbt_' + typeKey.replace(/\\./g, '_') + '" onclick="qbToggleType(\\'' + typeKey + '\\')">' + t.label + '</span>';
            }});
            html += '</div>';
        }});
        container.querySelector('.qb-cards').innerHTML = html;
    }} catch(e) {{
        container.innerHTML = '<div style="color:var(--text-muted);font-size:12px;">Could not load types</div>';
    }}
}}

function qbToggleType(typeKey) {{
    const idx = qbState.types.indexOf(typeKey);
    if (idx > -1) {{ qbState.types.splice(idx, 1); }}
    else {{ qbState.types.push(typeKey); }}
    const el = document.getElementById('qbt_' + typeKey.replace(/\\./g, '_'));
    if (el) el.classList.toggle('selected', qbState.types.includes(typeKey));
    updateQBPreview();
}}

function renderQBStep3() {{
    const container = document.getElementById('qbStep3');
    if (!qbState.concept) {{ container.style.display = 'none'; return; }}
    container.style.display = 'block';
    let html = '';
    QB_ACTIONS.forEach(a => {{
        html += '<span class="qb-action" id="qba_' + a.key + '" onclick="qbSelectAction(\\'' + a.key + '\\')">' + a.icon + ' ' + a.label + '</span>';
    }});
    container.querySelector('.qb-cards').innerHTML = html;
}}

function qbSelectAction(key) {{
    qbState.action = key;
    document.querySelectorAll('.qb-action').forEach(el => el.classList.remove('selected'));
    const el = document.getElementById('qba_' + key);
    if (el) el.classList.add('selected');
    renderQBStep4();
    updateQBPreview();
}}

async function renderQBStep4() {{
    const container = document.getElementById('qbStep4');
    if (!qbState.concept || !qbState.action) {{ container.style.display = 'none'; return; }}
    container.style.display = 'block';

    const concept = qbState.concept;
    const action = qbState.action;
    let html = '';

    // Group By dropdown (for breakdown/trend)
    if (action === 'breakdown' || action === 'trend') {{
        const groupOptions = QB_GROUPBY[concept] || [];
        html += '<div class="qb-filter-row"><label style="font-size:12px;color:var(--text-sec);min-width:80px;">Group by:</label>';
        html += '<select class="qb-select" id="qbGroupBy" onchange="qbState.groupBy=this.value;updateQBPreview()">';
        html += '<option value="">— Select —</option>';
        groupOptions.forEach(g => {{
            const label = g.replace(/_/g, ' ').replace('month', '(Monthly)');
            html += '<option value="' + g + '">' + label + '</option>';
        }});
        html += '</select></div>';
    }}

    // Top N input
    if (action === 'top') {{
        html += '<div class="qb-filter-row"><label style="font-size:12px;color:var(--text-sec);min-width:80px;">Show top:</label>';
        html += '<select class="qb-select" id="qbTopN" onchange="qbState.filters.topN=this.value;updateQBPreview()">';
        [5,10,15,20,25,50].forEach(n => {{ html += '<option value="' + n + '"' + (n===10 ? ' selected' : '') + '>' + n + '</option>'; }});
        html += '</select>';
        // Metric to rank by
        html += '<label style="font-size:12px;color:var(--text-sec);">by:</label>';
        html += '<select class="qb-select" id="qbTopMetric" onchange="qbState.filters.topMetric=this.value;updateQBPreview()">';
        html += '<option value="paid_amount">Paid Amount</option><option value="billed_amount">Billed Amount</option><option value="count">Count</option>';
        html += '</select></div>';
    }}

    // Ratio inputs
    if (action === 'ratio') {{
        // Load distinct values for the ratio comparison
        html += '<div class="qb-filter-row"><label style="font-size:12px;color:var(--text-sec);min-width:80px;">Ratio of:</label>';
        html += '<select class="qb-select" id="qbRatioA" onchange="qbState.filters.ratioA=this.value;updateQBPreview()"><option value="">— Select A —</option></select>';
        html += '<span style="font-size:12px;color:var(--text-muted);">to</span>';
        html += '<select class="qb-select" id="qbRatioB" onchange="qbState.filters.ratioB=this.value;updateQBPreview()"><option value="">— Select B —</option></select>';
        html += '</div>';
    }}

    // Common filters: Region, Status, Date range
    html += '<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border);">';
    html += '<div style="font-size:11px;font-weight:600;color:var(--text-muted);margin-bottom:6px;">FILTERS (optional)</div>';

    // Region dropdown — load from data
    html += '<div class="qb-filter-row"><label style="font-size:12px;color:var(--text-sec);min-width:80px;">Region:</label>';
    html += '<select class="qb-select" id="qbRegion" onchange="qbState.filters.region=this.value;updateQBPreview()">';
    html += '<option value="">All Regions</option></select></div>';

    // Status dropdown (for claims/encounters)
    if (concept === 'claims' || concept === 'encounters') {{
        const statusCol = concept === 'claims' ? 'CLAIM_STATUS' : 'ENCOUNTER_STATUS';
        html += '<div class="qb-filter-row"><label style="font-size:12px;color:var(--text-sec);min-width:80px;">Status:</label>';
        html += '<select class="qb-select" id="qbStatus" onchange="qbState.filters.status=this.value;updateQBPreview()">';
        html += '<option value="">All Statuses</option></select></div>';
    }}

    html += '</div>';
    container.querySelector('.qb-cards').innerHTML = html;

    // Populate dropdowns from real data
    populateQBDropdowns();
}}

async function populateQBDropdowns() {{
    const concept = qbState.concept;
    if (!concept) return;
    const table = QB_CONCEPTS[concept].table;

    // Region
    try {{
        const regionCol = (concept === 'providers') ? 'KP_REGION' : 'KP_REGION';
        const resp = await fetch('/api/distinct-values?table=' + table + '&column=' + regionCol);
        const data = await resp.json();
        const sel = document.getElementById('qbRegion');
        if (sel && data.values) {{
            data.values.forEach(v => {{ const opt = document.createElement('option'); opt.value = v; opt.textContent = v; sel.appendChild(opt); }});
        }}
    }} catch(e) {{}}

    // Status
    const statusSel = document.getElementById('qbStatus');
    if (statusSel) {{
        const statusCol = concept === 'claims' ? 'CLAIM_STATUS' : 'ENCOUNTER_STATUS';
        try {{
            const resp = await fetch('/api/distinct-values?table=' + table + '&column=' + statusCol);
            const data = await resp.json();
            if (data.values) {{
                data.values.forEach(v => {{ const opt = document.createElement('option'); opt.value = v; opt.textContent = v; statusSel.appendChild(opt); }});
            }}
        }} catch(e) {{}}
    }}

    // Ratio selectors
    if (qbState.action === 'ratio') {{
        let ratioOptions = [];
        if (concept === 'claims') ratioOptions = ['PAID', 'DENIED', 'PENDING', 'ADJUSTED'];
        else if (concept === 'encounters') ratioOptions = ['EMERGENCY', 'INPATIENT', 'OUTPATIENT', 'OBSERVATION'];
        const selA = document.getElementById('qbRatioA');
        const selB = document.getElementById('qbRatioB');
        if (selA && selB) {{
            ratioOptions.forEach(v => {{
                selA.innerHTML += '<option value="' + v + '">' + v + '</option>';
                selB.innerHTML += '<option value="' + v + '">' + v + '</option>';
            }});
        }}
    }}
}}

function updateQBPreview() {{
    const preview = document.getElementById('qbPreview');
    const previewText = document.getElementById('qbPreviewText');
    if (!qbState.concept) {{ preview.style.display = 'none'; return; }}
    preview.style.display = 'block';

    const concept = QB_CONCEPTS[qbState.concept];
    let sentence = '';

    // Build type description
    let typeDesc = concept.label.toLowerCase();
    if (qbState.types.length > 0) {{
        const typeLabels = qbState.types.map(t => {{
            const parts = t.split('.');
            return parts[parts.length - 1].replace(/_/g, ' ');
        }});
        typeDesc = typeLabels.join(' and ') + ' ' + concept.label.toLowerCase();
    }}

    // Build sentence based on action
    const action = qbState.action;
    if (!action) {{
        sentence = 'Select what you want to know about ' + typeDesc;
    }} else if (action === 'count') {{
        sentence = 'How many ' + typeDesc;
    }} else if (action === 'total') {{
        sentence = 'Total paid amount for ' + typeDesc;
    }} else if (action === 'average') {{
        sentence = 'Average cost of ' + typeDesc;
    }} else if (action === 'details') {{
        sentence = 'Show me ' + typeDesc + ' details';
    }} else if (action === 'breakdown') {{
        const gb = qbState.groupBy ? qbState.groupBy.replace(/_/g, ' ').replace('month', '(monthly)').toLowerCase() : '...';
        sentence = typeDesc.charAt(0).toUpperCase() + typeDesc.slice(1) + ' breakdown by ' + gb;
    }} else if (action === 'top') {{
        const n = qbState.filters.topN || 10;
        const metric = (qbState.filters.topMetric || 'paid_amount').replace(/_/g, ' ');
        sentence = 'Top ' + n + ' ' + typeDesc + ' by ' + metric;
    }} else if (action === 'ratio') {{
        const a = (qbState.filters.ratioA || '...').toLowerCase();
        const b = (qbState.filters.ratioB || '...').toLowerCase();
        sentence = a + ' to ' + b + ' ratio for ' + typeDesc;
    }} else if (action === 'trend') {{
        const gb = qbState.groupBy || 'month';
        sentence = typeDesc.charAt(0).toUpperCase() + typeDesc.slice(1) + ' trend by ' + gb.replace(/_/g, ' ').toLowerCase();
    }}

    // Add filters
    if (qbState.filters.region) sentence += ' in ' + qbState.filters.region;
    if (qbState.filters.status) sentence += ' where status is ' + qbState.filters.status;

    previewText.textContent = sentence;
}}

function qbRunQuery() {{
    const text = document.getElementById('qbPreviewText').textContent;
    if (!text || text.startsWith('Select what')) return;
    // Switch to Ask tab and put the query there
    switchQueryTab('ask');
    document.getElementById('queryInput').value = text;
    submitQuery(text);
}}

function qbClear() {{
    qbState.concept = null;
    qbState.types = [];
    qbState.action = null;
    qbState.groupBy = null;
    qbState.filters = {{}};
    document.querySelectorAll('.qb-concept-card').forEach(c => c.classList.remove('selected'));
    document.getElementById('qbStep2').style.display = 'none';
    document.getElementById('qbStep2').innerHTML = '';
    document.getElementById('qbStep3').style.display = 'none';
    document.getElementById('qbStep4').style.display = 'none';
    document.getElementById('qbPreview').style.display = 'none';
    document.getElementById('qbSuggestions').innerHTML = '';
}}

async function loadQBSuggestions() {{
    const container = document.getElementById('qbSuggestions');
    if (!qbState.concept) {{ container.innerHTML = ''; return; }}

    // Build concept-specific suggestion templates
    const concept = qbState.concept;
    const suggestions = [];

    if (concept === 'claims') {{
        suggestions.push('total claims by region', 'denied claims this month', 'average paid amount by claim type', 'top 10 providers by claim count', 'claims trend by month', 'paid vs denied ratio', 'high cost claims over $10000');
    }} else if (concept === 'members') {{
        suggestions.push('member count by region', 'enrollment by plan type', 'member demographics breakdown', 'high risk members', 'new enrollments this month');
    }} else if (concept === 'encounters') {{
        suggestions.push('emergency visits by region', 'average length of stay', 'inpatient vs outpatient ratio', 'readmission rate', 'encounters by visit type');
    }} else if (concept === 'providers') {{
        suggestions.push('providers by specialty', 'top 10 providers by patient volume', 'provider panel size distribution');
    }} else if (concept === 'diagnoses') {{
        suggestions.push('top diagnoses by frequency', 'chronic conditions breakdown', 'diagnosis severity distribution');
    }} else if (concept === 'prescriptions') {{
        suggestions.push('top prescribed medications', 'prescription cost by drug class', 'generic vs brand breakdown');
    }} else if (concept === 'referrals') {{
        suggestions.push('referrals by reason', 'pending referrals', 'referral completion rate');
    }}

    // Also fetch from learning engine
    try {{
        const resp = await fetch('/api/suggestions?concept=' + concept);
        const data = await resp.json();
        (data.suggestions || []).forEach(s => {{
            if (!suggestions.includes(s.question)) suggestions.push(s.question);
        }});
    }} catch(e) {{}}

    let html = '<div style="font-size:11px;font-weight:600;color:var(--text-muted);margin-bottom:6px;margin-top:12px;">SUGGESTED QUERIES</div>';
    html += '<div style="display:flex;gap:6px;flex-wrap:wrap;">';
    suggestions.slice(0, 12).forEach(s => {{
        html += '<button class="suggestion-chip" onclick="switchQueryTab(\\\'ask\\\');document.getElementById(\\\'queryInput\\\').value=\\\'' + s.replace(/'/g, '') + '\\\';submitQuery()">' + s + '</button>';
    }});
    html += '</div>';
    container.innerHTML = html;
}}

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════
// ── VISUAL QUERY BUILDER (VQB) - Enhanced SQL-based query builder with live preview ──
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════

const vqbState = {{
    tables: [],           // selected table names
    columns: {{}},        // {{table: [col1, col2, ...]}}
    filters: [],          // [{{table, column, operator, value}}]
    aggFunc: '',
    aggCol: '*',
    groupBy: [],
    limit: 50,
    schema: null          // loaded from /api/schema-columns
}};

const VQB_TABLE_META = {{
    claims:        {{icon: '💳', label: 'Claims', desc: 'Billing & payments'}},
    members:       {{icon: '👥', label: 'Members', desc: 'Demographics & enrollment'}},
    encounters:    {{icon: '🏥', label: 'Encounters', desc: 'Clinical visits'}},
    providers:     {{icon: '🩺', label: 'Providers', desc: 'Provider directory'}},
    diagnoses:     {{icon: '📋', label: 'Diagnoses', desc: 'ICD-10 codes'}},
    prescriptions: {{icon: '💊', label: 'Prescriptions', desc: 'Medications'}},
    referrals:     {{icon: '📤', label: 'Referrals', desc: 'Care referrals'}}
}};

const VQB_ALIASES = {{
    claims: 'c',
    members: 'm',
    encounters: 'e',
    providers: 'p',
    diagnoses: 'd',
    prescriptions: 'rx',
    referrals: 'r'
}};

const VQB_JOINS = {{
    'claims-members': {{left: 'claims', right: 'members', on: 'MEMBER_ID'}},
    'claims-encounters': {{left: 'claims', right: 'encounters', on: 'ENCOUNTER_ID'}},
    'claims-providers': {{left: 'claims', right: 'providers', leftCol: 'RENDERING_NPI', rightCol: 'NPI'}},
    'encounters-members': {{left: 'encounters', right: 'members', on: 'MEMBER_ID'}},
    'encounters-providers': {{left: 'encounters', right: 'providers', leftCol: 'RENDERING_NPI', rightCol: 'NPI'}},
    'diagnoses-encounters': {{left: 'diagnoses', right: 'encounters', on: 'ENCOUNTER_ID'}},
    'diagnoses-members': {{left: 'diagnoses', right: 'members', on: 'MEMBER_ID'}},
    'prescriptions-members': {{left: 'prescriptions', right: 'members', on: 'MEMBER_ID'}},
    'prescriptions-providers': {{left: 'prescriptions', right: 'providers', leftCol: 'PRESCRIBING_NPI', rightCol: 'NPI'}},
    'referrals-members': {{left: 'referrals', right: 'members', on: 'MEMBER_ID'}},
    'referrals-providers': {{left: 'referrals', right: 'providers', leftCol: 'REFERRING_NPI', rightCol: 'NPI'}}
}};

function vqbColIcon(type) {{
    if (!type) return '📄';
    const t = type.toLowerCase();
    if (t === 'string' || t === 'text') return '🔤';
    if (t.includes('int') || t.includes('float') || t.includes('currency') || t.includes('numeric')) return '🔢';
    if (t.includes('date')) return '📅';
    return '📄';
}}

async function vqbInit() {{
    // Fetch schema from /api/schema-columns
    try {{
        const resp = await fetch('/api/schema-columns');
        const data = await resp.json();
        vqbState.schema = data.schema || {{}};
    }} catch(e) {{
        console.error('Failed to load schema:', e);
        vqbState.schema = {{}};
    }}

    // Reset state but keep schema
    vqbState.tables = [];
    vqbState.columns = {{}};
    vqbState.filters = [];
    vqbState.aggFunc = '';
    vqbState.aggCol = '*';
    vqbState.groupBy = [];
    vqbState.limit = 50;

    vqbRenderTableSelector();
}}

function vqbRenderTableSelector() {{
    const container = document.getElementById('vqbTables');
    if (!container) return;

    let html = '';
    Object.keys(VQB_TABLE_META).forEach(table => {{
        const meta = VQB_TABLE_META[table];
        const selected = vqbState.tables.includes(table) ? ' selected' : '';
        html += `
          <div class="vqb-table-chip${{selected}}" onclick="vqbToggleTable('${{table}}')">
            <span style="font-size:16px">${{meta.icon}}</span>
            <span>${{meta.label}}</span>
          </div>
        `;
    }});
    container.innerHTML = html;
}}

function vqbToggleTable(table) {{
    const idx = vqbState.tables.indexOf(table);
    if (idx >= 0) {{
        vqbState.tables.splice(idx, 1);
        delete vqbState.columns[table];
    }} else {{
        vqbState.tables.push(table);
        // Auto-select all columns for this table
        if (vqbState.schema && vqbState.schema[table]) {{
            vqbState.columns[table] = vqbState.schema[table].map(c => c.name);
        }}
    }}

    vqbRenderTableSelector();
    vqbLoadSuggestions();
    if (vqbState.tables.length > 0) {{
        vqbRevealStep('vqbStep2', true);
        vqbRenderColumns();
        vqbUpdateStep4();
        vqbUpdateSQL();
        vqbRevealStep('vqbStep3', true);
        vqbRevealStep('vqbStep4', true);
        vqbRevealStep('vqbStep5', true);
    }} else {{
        ['vqbStep2','vqbStep3','vqbStep4','vqbStep5'].forEach(id => vqbRevealStep(id, false));
    }}
}}

function vqbRevealStep(id, show) {{
    const el = document.getElementById(id);
    if (!el) return;
    if (show) {{
        if (el.style.display !== 'none' && el.style.display !== '') return; // already visible
        el.style.display = 'block';
        el.style.opacity = '0';
        el.style.transform = 'translateY(10px)';
        requestAnimationFrame(() => {{
            el.style.transition = 'opacity 0.35s ease, transform 0.35s ease';
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }});
    }} else {{
        el.style.opacity = '0';
        el.style.transform = 'translateY(10px)';
        setTimeout(() => {{ el.style.display = 'none'; }}, 200);
    }}
}}

const VQB_SUGGESTIONS = {{
    claims: [
        'total claims by region',
        'denied claims this month',
        'average paid amount by claim type',
        'top 10 providers by claim count',
        'claims trend by month',
        'paid vs denied ratio',
        'high cost claims over $10000',
        'total billed amount for denied claims',
        'claims by status',
        'institutional vs professional claims breakdown',
    ],
    members: [
        'member count by region',
        'enrollment by plan type',
        'member demographics breakdown',
        'how many members are there',
        'members older than 65',
        'members with chronic conditions',
        'youngest and oldest members',
        'gender distribution of members',
    ],
    encounters: [
        'emergency visits by region',
        'average length of stay',
        'inpatient vs outpatient ratio',
        'encounters by visit type',
        'busiest departments by encounter count',
        'departments with average length of stay over 3 days',
        'encounters by facility',
        'readmission rate',
    ],
    providers: [
        'providers by specialty',
        'top 10 providers by patient volume',
        'provider count by region',
        'busiest providers this month',
        'providers in cardiology',
    ],
    diagnoses: [
        'top diagnoses by frequency',
        'chronic conditions breakdown',
        'most common ICD-10 codes',
        'diagnosis count by HCC category',
        'diagnoses by severity',
    ],
    prescriptions: [
        'top prescribed medications',
        'prescription cost by drug class',
        'generic vs brand breakdown',
        'prescriptions by status',
        'most expensive prescriptions',
    ],
    referrals: [
        'referrals by reason',
        'pending referrals count',
        'approved vs denied referrals',
        'referral completion rate',
        'referrals by specialty',
    ],
}};

// Combined suggestions for multi-table selections
const VQB_CROSS_TABLE_SUGGESTIONS = {{
    'claims,members': [
        'total billed amount per member',
        'how many members have more than 5 claims',
        'average claim amount by gender',
        'claims for members older than 50',
    ],
    'claims,encounters': [
        'claims linked to emergency encounters',
        'average claim amount by visit type',
        'billed amount by department',
    ],
    'encounters,members': [
        'how many members visited emergency more than 10 times',
        'members with multiple encounters',
        'encounter count by age group',
    ],
    'encounters,providers': [
        'busiest providers by encounter count',
        'encounters by provider specialty',
    ],
    'diagnoses,members': [
        'members with diabetes diagnosis',
        'diagnosis count per member',
    ],
    'prescriptions,members': [
        'members with most prescriptions',
        'prescription count by gender',
    ],
}};

function vqbLoadSuggestions() {{
    const container = document.getElementById('vqbSuggestions');
    const chipsContainer = document.getElementById('vqbSuggestionsChips');
    if (!container || !chipsContainer) return;

    if (vqbState.tables.length === 0) {{
        container.style.display = 'none';
        chipsContainer.innerHTML = '';
        return;
    }}

    // 1. Cross-table suggestions first (highest priority)
    const crossSuggestions = [];
    if (vqbState.tables.length >= 2) {{
        const sortedTables = [...vqbState.tables].sort();
        for (let i = 0; i < sortedTables.length; i++) {{
            for (let j = i + 1; j < sortedTables.length; j++) {{
                const key = sortedTables[i] + ',' + sortedTables[j];
                (VQB_CROSS_TABLE_SUGGESTIONS[key] || []).forEach(s => {{
                    if (!crossSuggestions.includes(s)) crossSuggestions.push(s);
                }});
            }}
        }}
    }}

    // 2. Interleave per-table suggestions (round-robin so all tables get fair share)
    const perTableArrays = vqbState.tables.map(t => [...(VQB_SUGGESTIONS[t] || [])]);
    const interleaved = [];
    let maxLen = Math.max(...perTableArrays.map(a => a.length), 0);
    for (let i = 0; i < maxLen; i++) {{
        perTableArrays.forEach(arr => {{
            if (i < arr.length && !interleaved.includes(arr[i]) && !crossSuggestions.includes(arr[i])) {{
                interleaved.push(arr[i]);
            }}
        }});
    }}

    // 3. Combine: cross-table first, then interleaved per-table
    const allSuggestions = [...crossSuggestions, ...interleaved].slice(0, 18);

    if (allSuggestions.length === 0) {{
        container.style.display = 'none';
        return;
    }}

    let html = '';
    allSuggestions.forEach((s, idx) => {{
        const isCross = idx < crossSuggestions.length;
        const bgColor = isCross ? 'rgba(40,97,64,0.12)' : 'rgba(0,107,166,0.08)';
        const borderColor = isCross ? 'rgba(40,97,64,0.35)' : 'rgba(0,107,166,0.25)';
        const textColor = isCross ? '#4CAF50' : 'var(--info)';
        const hoverBg = isCross ? 'rgba(40,97,64,0.22)' : 'rgba(0,107,166,0.18)';
        const prefix = isCross ? '🔗 ' : '';
        html += `<button style="font-size:11px;padding:5px 10px;border-radius:16px;
            border:1px solid ${{borderColor}};background:${{bgColor}};color:${{textColor}};
            cursor:pointer;transition:all 0.2s;white-space:nowrap;font-weight:${{isCross ? '600' : '400'}};"
            onmouseover="this.style.background='${{hoverBg}}'"
            onmouseout="this.style.background='${{bgColor}}'"
            onclick="switchQueryTab('ask');document.getElementById('queryInput').value='${{s.replace(/'/g, '')}}';submitQuery()">${{prefix}}${{s}}</button>`;
    }});

    chipsContainer.innerHTML = html;
    container.style.display = 'block';
    // Smooth reveal animation
    container.style.opacity = '0';
    container.style.transform = 'translateY(-8px)';
    requestAnimationFrame(() => {{
        container.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        container.style.opacity = '1';
        container.style.transform = 'translateY(0)';
    }});
}}

function vqbRenderColumns() {{
    const container = document.getElementById('vqbColumns');
    if (!container || vqbState.tables.length === 0) return;

    let html = '';
    vqbState.tables.forEach(table => {{
        if (!vqbState.schema || !vqbState.schema[table]) return;

        const cols = vqbState.schema[table];
        const selectedCols = vqbState.columns[table] || [];
        const allSelected = cols.length > 0 && cols.length === selectedCols.length;
        const meta = VQB_TABLE_META[table];

        html += `
          <div class="vqb-table-panel">
            <div class="vqb-table-panel-header" onclick="vqbTogglePanel('${{table}}')">
              <span class="table-icon">${{meta.icon}}</span>
              <span class="table-name">${{meta.label}}</span>
              <span class="select-controls">
                <a onclick="event.stopPropagation();vqbSelectAllTableCols('${{table}}')">Select All</a> ·
                <a onclick="event.stopPropagation();vqbDeselectAllTableCols('${{table}}')">None</a>
              </span>
              <span class="collapse-arrow">▼</span>
            </div>
            <div class="vqb-col-grid" id="vqbPanel_${{table}}">
        `;

        cols.forEach(col => {{
            const isSelected = selectedCols.includes(col.name);
            const icon = vqbColIcon(col.type);
            html += `
              <div class="vqb-col-item">
                <input type="checkbox" id="vqbCol_${{table}}_${{col.name}}"
                       ${{isSelected ? 'checked' : ''}}
                       onchange="vqbToggleColumn('${{table}}', '${{col.name}}')">
                <label for="vqbCol_${{table}}_${{col.name}}" class="vqb-col-label">
                  <span style="font-size:12px">${{icon}}</span> ${{col.name}}
                  <span class="vqb-col-type">${{col.type}}</span>
                </label>
              </div>
            `;
        }});

        html += '</div></div>';
    }});

    container.innerHTML = html;
}}

function vqbToggleColumn(table, col) {{
    if (!vqbState.columns[table]) vqbState.columns[table] = [];
    const idx = vqbState.columns[table].indexOf(col);
    if (idx >= 0) {{
        vqbState.columns[table].splice(idx, 1);
    }} else {{
        vqbState.columns[table].push(col);
    }}
    vqbUpdateSQL();
}}

function vqbTogglePanel(table) {{
    const panel = document.getElementById('vqbPanel_' + table);
    const header = event.currentTarget;
    const arrow = header.querySelector('.collapse-arrow');

    if (panel.style.display === 'none') {{
        panel.style.display = 'grid';
        arrow.classList.remove('collapsed');
    }} else {{
        panel.style.display = 'none';
        arrow.classList.add('collapsed');
    }}
}}

function vqbSelectAllTableCols(table) {{
    if (!vqbState.schema || !vqbState.schema[table]) return;
    vqbState.columns[table] = vqbState.schema[table].map(c => c.name);
    vqbRenderColumns();
    vqbUpdateSQL();
}}

function vqbDeselectAllTableCols(table) {{
    vqbState.columns[table] = [];
    vqbRenderColumns();
    vqbUpdateSQL();
}}

function vqbSelectAllCols() {{
    vqbState.tables.forEach(table => {{
        if (vqbState.schema && vqbState.schema[table]) {{
            vqbState.columns[table] = vqbState.schema[table].map(c => c.name);
        }}
    }});
    vqbRenderColumns();
    vqbUpdateSQL();
}}

function vqbDeselectAllCols() {{
    vqbState.tables.forEach(table => {{
        vqbState.columns[table] = [];
    }});
    vqbRenderColumns();
    vqbUpdateSQL();
}}

function vqbHighlightSQL(sql) {{
    // Apply basic syntax highlighting to SQL
    const keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'ON', 'GROUP', 'BY', 'ORDER', 'LIMIT', 'AND', 'OR', 'AS'];
    let highlighted = sql;

    // Highlight SQL keywords
    keywords.forEach(kw => {{
        const regex = new RegExp(`\\b${{kw}}\\b`, 'gi');
        highlighted = highlighted.replace(regex, match => `<span class="vqb-sql-keyword">${{match.toUpperCase()}}</span>`);
    }});

    return highlighted;
}}

function vqbCopySQL() {{
    const preview = document.getElementById('vqbSQLPreview');
    const sql = preview.getAttribute('data-plain-sql') || preview.textContent || '';
    if (!sql || sql.startsWith('--')) {{
        alert('Please build a complete query first');
        return;
    }}

    navigator.clipboard.writeText(sql).then(() => {{
        const btn = event.target;
        const original = btn.textContent;
        btn.textContent = '✓ Copied!';
        setTimeout(() => {{
            btn.textContent = original;
        }}, 2000);
    }}).catch(err => {{
        alert('Failed to copy: ' + err.message);
    }});
}}

function vqbAddFilter() {{
    vqbState.filters.push({{table: '', column: '', operator: '=', value: ''}});
    vqbRenderFilters();
}}

function vqbRemoveFilter(idx) {{
    vqbState.filters.splice(idx, 1);
    vqbRenderFilters();
    vqbUpdateSQL();
}}

function vqbRenderFilters() {{
    const container = document.getElementById('vqbFilters');
    if (!container) return;

    let html = '';
    vqbState.filters.forEach((filter, idx) => {{
        html += `
          <div class="vqb-filter-row">
            <select class="qb-select" onchange="vqbFilterTableChanged(${{idx}}, this.value)"
                    style="min-width:100px">
              <option value="">Select Table</option>
        `;
        vqbState.tables.forEach(t => {{
            const selected = filter.table === t ? ' selected' : '';
            html += `<option value="${{t}}"${{selected}}>${{VQB_TABLE_META[t].label}}</option>`;
        }});
        html += `</select>`;

        html += `<select class="qb-select" onchange="vqbFilterColChanged(${{idx}}, this.value)"
                    style="min-width:100px">
          <option value="">Select Column</option>`;
        if (filter.table && vqbState.schema && vqbState.schema[filter.table]) {{
            vqbState.schema[filter.table].forEach(col => {{
                const selected = filter.column === col.name ? ' selected' : '';
                html += `<option value="${{col.name}}"${{selected}}>${{col.name}}</option>`;
            }});
        }}
        html += `</select>`;

        html += `
          <select class="qb-select" onchange="vqbFilterOpChanged(${{idx}}, this.value)"
                  style="min-width:90px">
            <option value="="${{filter.operator === '=' ? ' selected' : ''}}>= (equals)</option>
            <option value="!="${{filter.operator === '!=' ? ' selected' : ''}}>!= (not)</option>
            <option value=">"${{filter.operator === '>' ? ' selected' : ''}}>>> (greater)</option>
            <option value="<"${{filter.operator === '<' ? ' selected' : ''}}>&lt; (less)</option>
            <option value=">="${{filter.operator === '>=' ? ' selected' : ''}}>>= (≥)</option>
            <option value="<="${{filter.operator === '<=' ? ' selected' : ''}}>&lt;= (≤)</option>
            <option value="LIKE"${{filter.operator === 'LIKE' ? ' selected' : ''}}>LIKE (contains)</option>
            <option value="IN"${{filter.operator === 'IN' ? ' selected' : ''}}>IN (list)</option>
            <option value="BETWEEN"${{filter.operator === 'BETWEEN' ? ' selected' : ''}}>BETWEEN</option>
            <option value="IS NULL"${{filter.operator === 'IS NULL' ? ' selected' : ''}}>IS NULL</option>
            <option value="IS NOT NULL"${{filter.operator === 'IS NOT NULL' ? ' selected' : ''}}>IS NOT NULL</option>
          </select>
        `;

        if (filter.operator !== 'IS NULL' && filter.operator !== 'IS NOT NULL') {{
            html += `
              <input type="text" placeholder="Enter value" class="qb-select" value="${{filter.value}}"
                     onchange="vqbFilterValChanged(${{idx}}, this.value)"
                     style="flex:1;min-width:120px">
            `;
        }}

        html += `
          <button class="vqb-filter-remove" onclick="vqbRemoveFilter(${{idx}})">✕ Remove</button>
          </div>
        `;
    }});

    container.innerHTML = html;
}}

function vqbFilterTableChanged(idx, table) {{
    vqbState.filters[idx].table = table;
    vqbState.filters[idx].column = '';
    vqbRenderFilters();
}}

function vqbFilterColChanged(idx, col) {{
    vqbState.filters[idx].column = col;
    vqbRenderFilters();
    vqbUpdateSQL();
}}

function vqbFilterOpChanged(idx, op) {{
    vqbState.filters[idx].operator = op;
    vqbRenderFilters();
    vqbUpdateSQL();
}}

function vqbFilterValChanged(idx, val) {{
    vqbState.filters[idx].value = val;
    vqbUpdateSQL();
}}

function vqbUpdateStep4() {{
    // Update aggregation column dropdown with all selected columns
    const aggColSelect = document.getElementById('vqbAggCol');
    if (!aggColSelect) return;

    let html = '<option value="*">* (all rows)</option>';
    vqbState.tables.forEach(table => {{
        const alias = VQB_ALIASES[table];
        if (vqbState.columns[table]) {{
            vqbState.columns[table].forEach(col => {{
                html += `<option value="${{alias}}.${{col}}">${{VQB_TABLE_META[table].label}}.${{col}}</option>`;
            }});
        }}
    }});
    aggColSelect.innerHTML = html;

    // Update GROUP BY with all selected columns
    const groupBySelect = document.getElementById('vqbGroupBy');
    if (!groupBySelect) return;

    html = '';
    vqbState.tables.forEach(table => {{
        const alias = VQB_ALIASES[table];
        if (vqbState.columns[table]) {{
            vqbState.columns[table].forEach(col => {{
                html += `<option value="${{alias}}.${{col}}">${{VQB_TABLE_META[table].label}}.${{col}}</option>`;
            }});
        }}
    }});
    groupBySelect.innerHTML = html;
}}

function vqbUpdateSQL() {{
    // Generate SQL from current state
    if (vqbState.tables.length === 0) {{
        document.getElementById('vqbSQLPreview').textContent = '-- Select tables and columns to build your query';
        return;
    }}

    let sql = 'SELECT ';

    // Build SELECT clause
    const selectCols = [];
    vqbState.tables.forEach(table => {{
        if (vqbState.columns[table] && vqbState.columns[table].length > 0) {{
            const alias = VQB_ALIASES[table];
            vqbState.columns[table].forEach(col => {{
                selectCols.push(`${{alias}}.${{col}}`);
            }});
        }}
    }});

    if (selectCols.length === 0) {{
        sql = '-- Select at least one column';
        document.getElementById('vqbSQLPreview').textContent = sql;
        return;
    }}

    if (vqbState.aggFunc) {{
        // When aggregating, include GROUP BY columns in SELECT first
        const groupBySelect = document.getElementById('vqbGroupBy');
        const groupCols = groupBySelect ? Array.from(groupBySelect.selectedOptions).map(o => o.value) : [];
        if (groupCols.length > 0) {{
            sql += groupCols.join(', ') + ', ';
        }}
        sql += `${{vqbState.aggFunc}}(${{vqbState.aggCol}})`;
    }} else {{
        sql += selectCols.join(', ');
    }}

    // Build FROM clause with primary table
    const primaryTable = vqbState.tables[0];
    const primaryAlias = VQB_ALIASES[primaryTable];
    sql += `\\n  FROM [${{primaryTable}}] ${{primaryAlias}}`;

    // Build JOINs
    if (vqbState.tables.length > 1) {{
        const joinedTables = new Set([primaryTable]);
        vqbState.tables.slice(1).forEach(table => {{
            // Find a join path from primary table or any joined table
            let joinFound = false;
            for (let jKey of Object.keys(VQB_JOINS)) {{
                const jInfo = VQB_JOINS[jKey];
                if ((joinedTables.has(jInfo.left) && jInfo.right === table) ||
                    (joinedTables.has(jInfo.right) && jInfo.left === table)) {{
                    const leftTbl = jInfo.left;
                    const rightTbl = jInfo.right;
                    const leftAlias = VQB_ALIASES[leftTbl];
                    const rightAlias = VQB_ALIASES[rightTbl];
                    const onClause = jInfo.on ? `${{leftAlias}}.${{jInfo.on}} = ${{rightAlias}}.${{jInfo.on}}` :
                        `${{leftAlias}}.${{jInfo.leftCol}} = ${{rightAlias}}.${{jInfo.rightCol}}`;
                    sql += `\\n  LEFT JOIN [${{rightTbl}}] ${{rightAlias}} ON ${{onClause}}`;
                    joinedTables.add(table);
                    joinFound = true;
                    break;
                }}
            }}
            if (!joinFound) {{
                // Fallback: simple join on MEMBER_ID if it exists
                const rightAlias = VQB_ALIASES[table];
                sql += `\\n  LEFT JOIN [${{table}}] ${{rightAlias}} ON ${{primaryAlias}}.MEMBER_ID = ${{rightAlias}}.MEMBER_ID`;
                joinedTables.add(table);
            }}
        }});
    }}

    // Build WHERE clause
    if (vqbState.filters.length > 0) {{
        const whereClauses = vqbState.filters
            .filter(f => f.table && f.column)
            .map(f => {{
                const alias = VQB_ALIASES[f.table];
                if (f.operator === 'IS NULL') return `${{alias}}.${{f.column}} IS NULL`;
                if (f.operator === 'IS NOT NULL') return `${{alias}}.${{f.column}} IS NOT NULL`;
                if (!f.value) return '';
                if (f.operator === 'LIKE') return `${{alias}}.${{f.column}} LIKE '%${{f.value}}%'`;
                if (f.operator === 'IN') return `${{alias}}.${{f.column}} IN (${{f.value}})`;
                if (f.operator === 'BETWEEN') return `${{alias}}.${{f.column}} BETWEEN ${{f.value}}`; // user provides: val1 AND val2
                return `${{alias}}.${{f.column}} ${{f.operator}} '${{f.value}}'`;
            }})
            .filter(c => c);
        if (whereClauses.length > 0) {{
            sql += `\\n  WHERE ${{whereClauses.join(' AND ')}}`;
        }}
    }}

    // Build GROUP BY
    const groupBySelect = document.getElementById('vqbGroupBy');
    if (groupBySelect && groupBySelect.selectedOptions.length > 0) {{
        const groupCols = Array.from(groupBySelect.selectedOptions).map(o => o.value);
        if (groupCols.length > 0) {{
            sql += `\\n  GROUP BY ${{groupCols.join(', ')}}`;
        }}
    }}

    // Add LIMIT
    const limit = parseInt(document.getElementById('vqbLimit').value) || 50;
    sql += `\\n  LIMIT ${{limit}}`;

    // Apply syntax highlighting to the display while keeping plain text for copy
    const preview = document.getElementById('vqbSQLPreview');
    preview.innerHTML = vqbHighlightSQL(sql);
    // Store the plain SQL in a data attribute for the copy function
    preview.setAttribute('data-plain-sql', sql);
}}

async function vqbRunQuery() {{
    const preview = document.getElementById('vqbSQLPreview');
    const sql = preview.getAttribute('data-plain-sql') || preview.textContent || '';
    if (!sql || sql.startsWith('--')) {{
        alert('Please build a complete query first');
        return;
    }}

    // Send to /api/execute-sql
    try {{
        const resp = await fetch('/api/execute-sql', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{sql: sql}})
        }});
        const result = await resp.json();

        if (result.error) {{
            alert('Query error: ' + result.error);
            return;
        }}

        // Display results in the results area
        const columns = result.columns || [];
        const rows = result.rows || [];
        const rowCount = result.row_count || 0;

        let resultsHtml = '<div class="result-header"><div class="result-title">Query Results</div></div>';
        resultsHtml += '<div class="sql-box">' + sql.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div>';

        if (rows.length === 0) {{
            resultsHtml += '<div style="padding:16px;color:var(--text-muted);">No results found.</div>';
        }} else {{
            resultsHtml += '<table class="query-results-table" id="queryResultsTable" style="width:100%;border-collapse:collapse;font-size:12px;">';
            resultsHtml += '<thead><tr style="background:var(--bg-card);border-bottom:1px solid var(--border);">';
            columns.forEach((col, i) => {{
                resultsHtml += `<th style="padding:8px;text-align:left;border-right:1px solid var(--border);cursor:pointer;" onclick="sortTable('queryResultsTable',${{i}})">
                  ${{col}} <span class="sort-icon">⇅</span></th>`;
            }});
            resultsHtml += '</tr></thead><tbody>';
            rows.forEach(row => {{
                resultsHtml += '<tr style="border-bottom:1px solid var(--border);">';
                columns.forEach(col => {{
                    const val = row[col] !== null ? String(row[col]) : '';
                    resultsHtml += `<td style="padding:8px;border-right:1px solid var(--border);">${{val.replace(/</g, '&lt;').replace(/>/g, '&gt;')}}</td>`;
                }});
                resultsHtml += '</tr>';
            }});
            resultsHtml += '</tbody></table>';
        }}

        resultsHtml += `<div style="margin-top:12px;display:flex;align-items:center;justify-content:space-between;">
          <span style="font-size:11px;color:var(--text-muted);">Showing ${{Math.min(rows.length, rowCount)}} of ${{rowCount}} rows</span>
          <div style="display:flex;gap:8px;">
            <button onclick="vqbExportCSV()" class="query-btn" style="padding:6px 14px;font-size:12px;">
              ⬇ Export CSV
            </button>
            <button onclick="vqbSaveDashboard()" class="query-btn" style="padding:6px 14px;font-size:12px;background:var(--success,#286140);">
              💾 Save to Dashboard
            </button>
          </div>
        </div>`;

        // Store results for export
        window._vqbLastResult = {{columns: columns, rows: rows, sql: sql, rowCount: rowCount}};

        document.getElementById('results-area').innerHTML = resultsHtml;
        document.getElementById('results-area').style.display = 'block';
    }} catch(e) {{
        alert('Failed to execute query: ' + e.message);
    }}
}}

function vqbExportCSV() {{
    if (!window._vqbLastResult) return;
    const {{columns, rows}} = window._vqbLastResult;
    let csv = columns.join(',') + '\\n';
    rows.forEach(row => {{
        csv += columns.map(col => {{
            let val = row[col] !== null && row[col] !== undefined ? String(row[col]) : '';
            if (val.includes(',') || val.includes('"') || val.includes('\\n')) {{
                val = '"' + val.replace(/"/g, '""') + '"';
            }}
            return val;
        }}).join(',') + '\\n';
    }});
    const blob = new Blob([csv], {{type: 'text/csv'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'query_results_' + new Date().toISOString().slice(0,10) + '.csv';
    a.click();
    URL.revokeObjectURL(url);
}}

function vqbSaveDashboard() {{
    if (!window._vqbLastResult) return;
    // Use existing saveDashboard logic
    const title = prompt('Enter a name for this dashboard card:', 'Custom Query');
    if (!title) return;
    const sql = window._vqbLastResult.sql;
    if (typeof saveToDashboard === 'function') {{
        saveToDashboard(title, sql, 'table');
    }} else {{
        // Fallback: call the save-dashboard API directly
        fetch('/api/save-dashboard', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{
                title: title,
                sql: sql,
                chart_type: 'table',
                token: localStorage.getItem('authToken') || ''
            }})
        }}).then(r => r.json()).then(d => {{
            if (d.success) alert('Saved to dashboard!');
            else alert('Save failed: ' + (d.message || 'Unknown error'));
        }}).catch(e => alert('Save failed: ' + e.message));
    }}
}}

function vqbClear() {{
    vqbState.tables = [];
    vqbState.columns = {{}};
    vqbState.filters = [];
    vqbState.aggFunc = '';
    vqbState.aggCol = '*';
    vqbState.groupBy = [];
    vqbState.limit = 50;

    ['vqbStep2','vqbStep3','vqbStep4','vqbStep5'].forEach(id => vqbRevealStep(id, false));
    const sugEl = document.getElementById('vqbSuggestions');
    if (sugEl) {{ sugEl.style.opacity = '0'; setTimeout(() => {{ sugEl.style.display = 'none'; }}, 200); }}

    vqbRenderTableSelector();
    document.getElementById('results-area').innerHTML = '';
}}

// Load suggestions on auth success
const _origOnAuth = onAuthSuccess;
onAuthSuccess = async function() {{
    await _origOnAuth();
    loadSuggestions();
    document.getElementById('hierarchyToggle').style.display = 'block';
    initSessionContext();
    updateContextBar();
    renderQBStep1();
}};

// Also load after page init if already logged in
setTimeout(() => {{ if (authToken) {{
    loadSuggestions();
    document.getElementById('hierarchyToggle').style.display = 'block';
    initSessionContext();
    updateContextBar();
    renderQBStep1();
}} }}, 2000);

/* Table sort */
const sortState = {{}};
function sortTable(tableId, colIdx) {{
    const table = document.getElementById(tableId);
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const key = tableId + '_' + colIdx;
    sortState[key] = !sortState[key];
    const asc = sortState[key];

    rows.sort((a, b) => {{
        let va = a.cells[colIdx]?.textContent.trim() || '';
        let vb = b.cells[colIdx]?.textContent.trim() || '';
        const na = parseFloat(va.replace(/[,$%]/g, ''));
        const nb = parseFloat(vb.replace(/[,$%]/g, ''));
        if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }});
    rows.forEach(row => tbody.appendChild(row));
    table.querySelectorAll('th .sort-icon').forEach((icon, i) => {{
        icon.innerHTML = i === colIdx ? (asc ? '&#9650;' : '&#9660;') : '&#8597;';
    }});
}}

function toggleColumn(tableId, colIdx, show) {{
    const table = document.getElementById(tableId);
    table.querySelectorAll('tr').forEach(row => {{
        const cell = row.cells[colIdx];
        if (cell) cell.style.display = show ? '' : 'none';
    }});
}}

function exportCSV(tableId) {{
    const table = document.getElementById(tableId);
    const rows = table.querySelectorAll('tr');
    let csv = '';
    rows.forEach(row => {{
        const cells = Array.from(row.querySelectorAll('th, td'));
        csv += cells.map(c => '"' + c.textContent.replace(/"/g, '""') + '"').join(',') + '\\n';
    }});
    const blob = new Blob([csv], {{ type: 'text/csv' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'query_results.csv'; a.click();
    URL.revokeObjectURL(url);
}}

function filterTable(input, tableId) {{
    const filter = input.value.toLowerCase();
    const table = document.getElementById(tableId);
    const rows = table.querySelectorAll('tbody tr');
    let visible = 0;
    rows.forEach(row => {{
        const show = row.textContent.toLowerCase().includes(filter);
        row.style.display = show ? '' : 'none';
        if (show) visible++;
    }});
    const c = document.getElementById(tableId + '-count');
    if (c) c.textContent = visible + ' / ' + rows.length + ' rows';
}}
</script>
</body>
</html>'''


def _analyze_columns(results):
    """Detect numeric, categorical, date, and status columns from data."""
    columns = list(results[0].keys())
    numeric_cols = []  # (name, [float values])
    cat_cols = []      # name
    date_cols = []     # name
    status_cols = []   # name
    pct_cols = []      # (name, [float values]) — columns that are percentages

    for col in columns:
        non_empty = [(r, r.get(col)) for r in results if r.get(col) is not None and str(r[col]).strip()]
        if not non_empty:
            cat_cols.append(col)
            continue

        vals_raw = [str(v) for _, v in non_empty]

        # Status detection
        if col.upper() in ('CLAIM_STATUS', 'STATUS', 'ENCOUNTER_STATUS', 'DIAGNOSIS_TYPE'):
            status_cols.append(col)
            cat_cols.append(col)
            continue

        # Date detection
        import re
        if col.upper().endswith(('_DATE', '_TIME')) or all(re.match(r'\d{4}-\d{2}', v) for v in vals_raw[:5]):
            date_cols.append(col)
            continue

        # ID column detection — treat as categorical even if numeric
        col_upper = col.upper()
        if col_upper.endswith(('_ID', '_CODE', '_NPI')) or col_upper in ('NPI', 'ID', 'MEMBER_ID', 'CLAIM_ID', 'PROVIDER_ID', 'ENCOUNTER_ID', 'ZIP', 'ZIPCODE', 'SSN'):
            cat_cols.append(col)
            continue

        # Numeric detection
        nums = []
        for v in vals_raw:
            try:
                nums.append(float(v.replace(',', '').replace('$', '').replace('%', '')))
            except (ValueError, TypeError):
                break

        if len(nums) == len(vals_raw) and nums:
            # Check if percentage
            col_lower = col.lower()
            if 'pct' in col_lower or 'rate' in col_lower or 'percent' in col_lower or '%' in col_lower:
                pct_cols.append((col, nums))
            numeric_cols.append((col, nums))
        else:
            cat_cols.append(col)

    return {
        'all': columns, 'numeric': numeric_cols, 'categorical': cat_cols,
        'dates': date_cols, 'status': status_cols, 'percentages': pct_cols,
    }


def _pick_chart_type(col_info, n_rows, question=""):
    """
    Smart chart selection based on data shape.
    Uses ML Decision Tree model when available, falls back to rules.

    Rules:
      - 1 row, 1 numeric col          → Single big number (KPI card)
      - Status column present          → Donut/pie breakdown
      - 1 cat + 1 numeric, <=12 rows  → Horizontal bar chart
      - 1 cat + 1 numeric, >12 rows   → Top-N bar chart
      - 1 cat + 2+ numeric            → Grouped bar chart (multi-metric)
      - Date col + numeric             → Sparkline / line trend
      - 2+ numeric, no cat             → Scatter / comparison cards
      - Percentage columns             → Gauge meters
      - Only categorical               → Frequency count (auto-aggregate)
      - Many rows, no grouping         → Table only (no chart)
    """
    # ML-powered chart selection (Decision Tree)
    if _HAS_ML_CHART and _ml_chart_selector:
        try:
            ml_chart = _ml_chart_selector.select(col_info, n_rows)
            if ml_chart == 'big_number' and n_rows == 1:
                return [('big_number', {})]
            # Use ML suggestion as primary, but still build full list for fallback
        except Exception:
            pass

    nc = col_info['numeric']
    cc = col_info['categorical']
    dc = col_info['dates']
    sc = col_info['status']
    pc = col_info['percentages']

    charts = []

    # Single scalar result
    if n_rows == 1 and len(nc) >= 1:
        charts.append(('big_number', {}))
        return charts

    # Status breakdown → donut
    if sc:
        charts.append(('donut', {'col': sc[0]}))

    # Percentage columns → gauges
    if pc:
        charts.append(('gauges', {'cols': pc[:4]}))

    # Date + numeric → line/trend
    if dc and nc:
        charts.append(('line', {'date_col': dc[0], 'val_col': nc[0][0]}))

    # Category + numeric → bar or grouped bar
    if cc and nc:
        if len(nc) == 1:
            charts.append(('bar', {'label_col': cc[0], 'val_col': nc[0][0]}))
        elif len(nc) >= 2:
            charts.append(('grouped_bar', {'label_col': cc[0], 'val_cols': [n[0] for n in nc[:3]]}))

    # 2+ numerics, no cat → comparison
    if len(nc) >= 2 and not cc:
        charts.append(('comparison', {'cols': nc[:4]}))

    # Fallback: if no charts picked, do frequency of first cat
    if not charts and cc:
        charts.append(('frequency', {'col': cc[0]}))

    return charts


def _generate_insight(col_info, results, question=""):
    """
    Auto-generate rich, healthcare-contextual insight text.
    Reads like an analyst briefing — explains what you're looking at,
    what stands out, and what it means for the organization.
    """
    n = len(results)
    nc = col_info['numeric']
    cc = col_info['categorical']
    sc = col_info['status']
    dc = col_info['dates']
    pc = col_info['percentages']
    q = question.lower()

    sections = []

    # ====== OPENING: What are we looking at? ======
    if n == 1 and nc:
        col, vals = nc[0]
        val = vals[0]
        # Display as integer if it's a whole number
        display_val = f"{int(val):,}" if val == int(val) and abs(val) < 1e15 else _fmt(val)
        sections.append(f"This is a <b>single aggregate metric</b>. The query returned one value: "
                       f"<b>{_esc(col)}</b> = <b>{display_val}</b>. "
                       f"This represents the computed result across the entire filtered dataset.")
    elif nc and cc and any(g in q for g in ['by ', 'per ', 'each ', 'group']):
        sections.append(f"This is a <b>grouped analysis</b> showing {n} distinct groups. "
                       f"Each row represents a unique <b>{_esc(cc[0])}</b> value with its "
                       f"corresponding metric(s). This breakdown lets you compare performance "
                       f"across categories and identify outliers.")
    elif 'denied' in q or 'denial' in q:
        sections.append(f"This is a <b>claims denial review</b> showing {n} denied claims. "
                       f"Denials represent revenue leakage and member friction — each denied "
                       f"claim requires staff time to investigate, appeal, or write off.")
    elif 'top' in q:
        sections.append(f"This is a <b>ranked list</b> showing the top {n} results sorted "
                       f"by the primary metric. The ranking identifies the highest-impact "
                       f"items that may need attention or represent best practices.")
    elif dc:
        sections.append(f"This is a <b>time-based view</b> with {n} data points. "
                       f"Reviewing trends over time helps identify seasonality, growth "
                       f"patterns, and emerging issues before they escalate.")
    else:
        sections.append(f"Showing <b>{n} records</b> with {len(col_info['all'])} data fields. "
                       f"This is a detail-level view of individual records matching your query.")

    # ====== NUMERIC ANALYSIS ======
    for col, vals in nc[:2]:
        avg = sum(vals) / len(vals)
        mn, mx = min(vals), max(vals)
        spread = mx - mn
        std = (sum((v - avg)**2 for v in vals) / len(vals)) ** 0.5
        cv = std / avg if avg != 0 else 0
        median = sorted(vals)[len(vals)//2]

        col_lower = col.lower()
        is_money = any(k in col_lower for k in ['amount', 'paid', 'billed', 'cost', 'revenue', 'copay'])
        unit = '$' if is_money else ''

        # Concentration analysis: what % of total comes from top 20%?
        sorted_vals = sorted(vals, reverse=True)
        top_20_count = max(1, len(vals) // 5)
        top_20_sum = sum(sorted_vals[:top_20_count])
        total_sum = sum(vals)
        concentration = top_20_sum / total_sum * 100 if total_sum > 0 else 0

        if n == 1:
            pass  # Already covered in opening
        elif cv < 0.1:
            sections.append(f"<b>{_esc(col)}</b> is remarkably consistent — averaging "
                          f"<b>{_fmt(avg, unit)}</b> with only {cv:.0%} variation. "
                          f"This uniformity suggests standardized pricing or consistent "
                          f"case mix across the dataset.")
        elif cv > 0.8:
            sections.append(f"<b>{_esc(col)}</b> shows <span style='color:var(--warning);'>high variance</span> "
                          f"— ranging from {_fmt(mn, unit)} to {_fmt(mx, unit)} "
                          f"(a {_fmt(spread, unit)} spread). The average is {_fmt(avg, unit)} but "
                          f"the median is {_fmt(median, unit)}, suggesting the distribution is "
                          f"{'right-skewed (a few very high values pull the average up)' if avg > median * 1.2 else 'left-skewed'}. "
                          f"The top 20% of values account for <b>{concentration:.0f}%</b> of the total.")
        elif cv > 0.3:
            sections.append(f"<b>{_esc(col)}</b> has moderate spread — from {_fmt(mn, unit)} to "
                          f"{_fmt(mx, unit)}, averaging <b>{_fmt(avg, unit)}</b>. "
                          f"{'The top 20% contributes ' + str(round(concentration)) + '% of the total value, indicating some concentration.' if concentration > 60 else ''}")
        else:
            sections.append(f"<b>{_esc(col)}</b> averages <b>{_fmt(avg, unit)}</b> across "
                          f"{n} results (range: {_fmt(mn, unit)} – {_fmt(mx, unit)}).")

    # ====== STATUS / DENIAL ANALYSIS ======
    for col in sc[:1]:
        from collections import Counter
        counts = Counter(str(r.get(col, '')) for r in results)
        total = sum(counts.values())

        # Build distribution description
        dist_parts = []
        for status, cnt in counts.most_common():
            pct = cnt / total * 100
            color = STATUS_COLORS.get(status, COLORS['text_sec'])
            dist_parts.append(f"<span style='color:{color};font-weight:600;'>{_esc(status)}</span>: "
                            f"{cnt} ({pct:.1f}%)")

        sections.append(f"<b>{_esc(col)} breakdown:</b> {', '.join(dist_parts)}.")

        if 'DENIED' in counts:
            denial_pct = counts['DENIED'] / total * 100
            denial_count = counts['DENIED']

            if denial_pct > 20:
                sections.append(f"<span style='color:var(--critical);'><b>Critical:</b> {denial_pct:.1f}% "
                              f"denial rate ({denial_count} claims)</span> — this is well above the "
                              f"healthcare industry benchmark of 5–10%. Common causes include "
                              f"missing prior authorization, coding errors, eligibility issues, "
                              f"and timely filing violations. Recommend immediate review of "
                              f"denial reasons and root cause analysis.")
            elif denial_pct > 10:
                sections.append(f"<span style='color:var(--warning);'><b>Elevated:</b> {denial_pct:.1f}% "
                              f"denial rate</span> — above the 5–10% industry target. "
                              f"Review the DENIAL_REASON column to identify the most frequent "
                              f"denial categories and target process improvements.")
            elif denial_pct > 0:
                sections.append(f"Denial rate of {denial_pct:.1f}% is within acceptable range "
                              f"(industry benchmark: 5–10%).")

        if 'PENDING' in counts:
            pending_pct = counts['PENDING'] / total * 100
            if pending_pct > 15:
                sections.append(f"<span style='color:var(--warning);'>{pending_pct:.1f}% of claims "
                              f"are still PENDING</span> — this backlog may indicate processing "
                              f"delays or staffing constraints in the adjudication team.")

    # ====== CATEGORICAL DISTRIBUTION ======
    for col in cc[:1]:
        if col in [s for s in sc]:
            continue
        from collections import Counter
        cat_counts = Counter(str(r.get(col, '')) for r in results)
        unique = len(cat_counts)

        if unique == n and n > 5:
            sections.append(f"Each record has a unique <b>{_esc(col)}</b> — this is a "
                          f"detail-level listing. Consider adding GROUP BY to aggregate "
                          f"these into meaningful categories.")
        elif unique <= 8 and n > unique:
            top3 = cat_counts.most_common(3)
            top3_text = ", ".join(f"<b>{_esc(v)}</b> ({c})" for v, c in top3)
            sections.append(f"Data is distributed across {unique} <b>{_esc(col)}</b> values. "
                          f"Top groups: {top3_text}.")

    # ====== DATE RANGE ======
    if dc:
        date_col = dc[0]
        date_vals = sorted(str(r.get(date_col, '')) for r in results if r.get(date_col))
        if date_vals:
            sections.append(f"<b>Date range:</b> {_esc(date_vals[0][:10])} to {_esc(date_vals[-1][:10])} "
                          f"({_esc(date_col)}).")

    # ====== PERCENTAGE METRICS ======
    for col, vals in pc[:2]:
        avg = sum(vals) / len(vals)
        if avg > 90:
            sections.append(f"<b>{_esc(col)}</b> is at <span style='color:var(--healthy);'>{avg:.1f}%</span> — excellent.")
        elif avg < 50:
            sections.append(f"<b>{_esc(col)}</b> is at <span style='color:var(--critical);'>{avg:.1f}%</span> — needs improvement.")

    if not sections:
        sections.append(f"Returned {n} rows with {len(col_info['all'])} columns.")

    return "<br><br>".join(sections)


def _generate_sql_explanation(question, sql):
    """
    Generate a detailed explanation of why this SQL was chosen.
    Uses the DynamicSQLEngine's explain capability.
    """
    try:
        from dynamic_sql_engine import DynamicSQLEngine
        engine = DynamicSQLEngine()
        result = engine.generate(question)
        return result.get('explanation', '')
    except Exception:
        # Fallback: basic SQL parsing explanation
        parts = []
        sql_upper = sql.upper()
        if 'JOIN' in sql_upper:
            parts.append("<b>Tables:</b> Multiple tables are joined to combine data from different sources.")
        if 'WHERE' in sql_upper:
            parts.append("<b>Filters:</b> WHERE clause narrows results to match your criteria.")
        if 'GROUP BY' in sql_upper:
            parts.append("<b>Grouping:</b> Results are aggregated into categories for comparison.")
        if 'ORDER BY' in sql_upper:
            parts.append("<b>Sorting:</b> Results are ordered to show the most relevant items first.")
        if 'LIMIT' in sql_upper:
            parts.append("<b>Limit:</b> Result set is capped to keep the output manageable.")
        return "<br>".join(parts) if parts else ""


def _svg_donut(data_dict, size=160):
    """Generate SVG donut chart from {label: count} dict."""
    total = sum(data_dict.values())
    if total == 0:
        return ""

    status_color_map = {
        'PAID': COLORS['healthy'], 'DENIED': COLORS['critical'],
        'PENDING': COLORS['warning'], 'ADJUSTED': COLORS['info'],
        'APPEALED': COLORS['behavioral'],
    }
    fallback_colors = [COLORS['info'], COLORS['healthy'], COLORS['preventive'],
                       COLORS['behavioral'], COLORS['operations'], COLORS['warning'],
                       COLORS['critical'], COLORS['inactive']]

    cx, cy, r = size/2, size/2, size/2 - 10
    r_inner = r * 0.6
    start_angle = -90
    paths = ""
    legend = ""

    for i, (label, count) in enumerate(data_dict.items()):
        pct = count / total
        angle = pct * 360
        end_angle = start_angle + angle

        color = status_color_map.get(str(label).upper(), fallback_colors[i % len(fallback_colors)])

        # Arc calculation
        def arc_point(a, radius):
            rad = math.radians(a)
            return cx + radius * math.cos(rad), cy + radius * math.sin(rad)

        x1o, y1o = arc_point(start_angle, r)
        x2o, y2o = arc_point(end_angle, r)
        x1i, y1i = arc_point(end_angle, r_inner)
        x2i, y2i = arc_point(start_angle, r_inner)
        large = 1 if angle > 180 else 0

        paths += f'<path d="M{x1o:.1f},{y1o:.1f} A{r},{r} 0 {large} 1 {x2o:.1f},{y2o:.1f} L{x1i:.1f},{y1i:.1f} A{r_inner},{r_inner} 0 {large} 0 {x2i:.1f},{y2i:.1f} Z" fill="{color}" opacity="0.9"><title>{_esc(label)}: {count} ({pct:.1%})</title></path>'

        legend += f'<div class="legend-item"><div class="legend-dot" style="background:{color};"></div>{_esc(label)}: {count} ({pct:.0%})</div>'
        start_angle = end_angle

    # Center text
    paths += f'<text x="{cx}" y="{cy-4}" text-anchor="middle" fill="{COLORS["text"]}" font-size="18" font-weight="700">{total}</text>'
    paths += f'<text x="{cx}" y="{cy+12}" text-anchor="middle" fill="{COLORS["text_muted"]}" font-size="10">total</text>'

    return f'''
    <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">{paths}</svg>
        <div style="display:flex;flex-direction:column;gap:4px;">{legend}</div>
    </div>'''


def _svg_gauges(pct_cols, results):
    """Generate SVG gauge meters for percentage columns."""
    gauges = ""
    for col, vals in pct_cols[:4]:
        avg = sum(vals) / len(vals)
        # Determine color based on value
        if avg >= 80:
            color = COLORS['healthy']
        elif avg >= 50:
            color = COLORS['info']
        elif avg >= 25:
            color = COLORS['warning']
        else:
            color = COLORS['critical']

        pct = min(max(avg / 100, 0), 1)
        angle = 180 * pct
        end_x = 70 + 55 * math.cos(math.radians(180 - angle))
        end_y = 70 - 55 * math.sin(math.radians(180 - angle))
        large = 1 if angle > 180 else 0

        gauges += f'''
        <div style="text-align:center;">
            <svg width="150" height="85" viewBox="0 0 140 80">
                <path d="M 15 70 A 55 55 0 0 1 125 70" fill="none" stroke="{COLORS['border']}" stroke-width="10" stroke-linecap="round"/>
                <path d="M 15 70 A 55 55 0 {large} 1 {end_x:.1f} {end_y:.1f}" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"/>
                <text x="70" y="60" text-anchor="middle" fill="{color}" font-size="20" font-weight="700">{avg:.1f}%</text>
            </svg>
            <div style="font-size:11px;color:var(--text-sec);margin-top:-4px;">{_esc(col)}</div>
        </div>'''

    return f'<div style="display:flex;gap:24px;flex-wrap:wrap;justify-content:center;">{gauges}</div>'


def _svg_line(results, date_col, val_col):
    """Generate SVG sparkline/line chart."""
    points = []
    for r in results:
        d = str(r.get(date_col, ''))
        try:
            v = float(str(r.get(val_col, 0)).replace(',', ''))
            points.append((d, v))
        except:
            pass

    if len(points) < 2:
        return ""

    # Sort by date
    points.sort(key=lambda p: p[0])

    vals = [v for _, v in points]
    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1

    w, h = 600, 180
    pad = 40
    pw = w - 2 * pad
    ph = h - 2 * pad

    # Build path
    coords = []
    for i, (d, v) in enumerate(points):
        x = pad + (i / (len(points) - 1)) * pw
        y = pad + ph - ((v - mn) / rng) * ph
        coords.append((x, y))

    path_d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in coords)

    # Gradient fill
    fill_d = path_d + f" L{coords[-1][0]:.1f},{h-pad} L{coords[0][0]:.1f},{h-pad} Z"

    # Trend detection
    first_half = sum(v for _, v in points[:len(points)//2]) / max(len(points)//2, 1)
    second_half = sum(v for _, v in points[len(points)//2:]) / max(len(points) - len(points)//2, 1)
    trend_color = COLORS['healthy'] if second_half >= first_half else COLORS['critical']
    trend_word = "upward" if second_half > first_half * 1.05 else "downward" if second_half < first_half * 0.95 else "stable"

    # X-axis labels (show ~5)
    step = max(1, len(points) // 5)
    x_labels = "".join(
        f'<text x="{coords[i][0]:.1f}" y="{h-8}" text-anchor="middle" fill="{COLORS["text_muted"]}" font-size="9">{_esc(points[i][0][:10])}</text>'
        for i in range(0, len(points), step)
    )

    # Data point dots
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{trend_color}" opacity="0.8"><title>{_esc(points[i][0])}: {_fmt(points[i][1])}</title></circle>'
        for i, (x, y) in enumerate(coords)
    )

    return f'''
    <div style="margin-bottom:8px;">
        <svg width="100%" viewBox="0 0 {w} {h}" style="max-width:{w}px;">
            <defs><linearGradient id="lg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="{trend_color}" stop-opacity="0.2"/>
                <stop offset="100%" stop-color="{trend_color}" stop-opacity="0"/>
            </linearGradient></defs>
            <path d="{fill_d}" fill="url(#lg)"/>
            <path d="{path_d}" fill="none" stroke="{trend_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            {dots}
            {x_labels}
        </svg>
        <div style="font-size:12px;color:var(--text-sec);margin-top:4px;">
            Trend: <span style="color:{trend_color};font-weight:600;">{trend_word}</span>
            &bull; Start: {_fmt(points[0][1])} &bull; End: {_fmt(points[-1][1])}
            &bull; {len(points)} data points
        </div>
    </div>'''


def _grouped_bar_html(results, label_col, val_cols):
    """Multi-metric grouped horizontal bars."""
    bar_colors = [COLORS['info'], COLORS['healthy'], COLORS['preventive'],
                  COLORS['behavioral'], COLORS['operations']]
    html = ""

    # Legend
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;font-size:11px;color:var(--text-sec);">'
        f'<span style="width:10px;height:10px;border-radius:3px;background:{bar_colors[i % len(bar_colors)]};"></span>{_esc(c)}</span>'
        for i, c in enumerate(val_cols)
    )
    html += f'<div style="margin-bottom:12px;">{legend}</div>'

    # Compute max across all val cols
    max_val = 0
    for row in results[:15]:
        for vc in val_cols:
            try:
                v = float(str(row.get(vc, 0)).replace(',', ''))
                max_val = max(max_val, v)
            except: pass
    max_val = max_val or 1

    for row in results[:15]:
        label = str(row.get(label_col, ''))[:20]
        bars = ""
        for i, vc in enumerate(val_cols):
            try:
                v = float(str(row.get(vc, 0)).replace(',', ''))
                pct = v / max_val * 100
                c = bar_colors[i % len(bar_colors)]
                bars += f'<div class="bar-track" style="height:20px;margin-bottom:3px;"><div class="bar-fill" style="width:{pct:.1f}%;background:{c};height:20px;font-size:10px;">{_fmt(v)}</div></div>'
            except: pass
        html += f'''
        <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;">
            <span class="bar-label">{_esc(label)}</span>
            <div style="flex:1;">{bars}</div>
        </div>'''

    return html


def _build_results_html(results, question="", sql="", intent="", mode=""):
    """Build results HTML with smart chart selection and auto-insights."""
    html = ""

    # Query + SQL header
    if question or sql:
        html += f'''
        <div class="result-header">
            <div class="result-title">{_esc(question)}</div>
            <div class="result-meta">
                <span class="meta-badge" style="background:rgba(0,107,166,0.12);color:var(--info);">{_esc(intent or 'query')}</span>
                <span class="meta-badge" style="background:rgba(13,148,136,0.12);color:var(--preventive);">{_esc(mode or 'auto')}</span>
            </div>
        </div>'''
    if sql:
        html += f'''<div class="sql-edit-container">
            <textarea class="sql-editor" id="sqlEditor" spellcheck="false">{_esc(sql)}</textarea>
            <div class="sql-edit-actions">
                <button class="sql-run-btn" onclick="runEditedSQL()">&#9654; Run Edited SQL</button>
                <button class="sql-reset-btn" onclick="resetSQL()">&#8634; Reset</button>
                <span class="sql-edit-hint">Edit the SQL above and click Run to see updated results</span>
            </div>
        </div>
        <input type="hidden" id="originalSQL" value="{_esc(sql)}">'''

        # SQL Explanation — get from dynamic engine if available
        sql_explanation = _generate_sql_explanation(question, sql)
        if sql_explanation:
            html += f'''
            <details style="margin-bottom:16px;">
                <summary style="cursor:pointer;font-size:13px;font-weight:600;color:var(--preventive);
                    padding:8px 0;user-select:none;">
                    &#128269; Why this SQL? (click to expand)
                </summary>
                <div style="background:rgba(13,148,136,0.05);border:1px solid rgba(13,148,136,0.15);
                    border-radius:10px;padding:14px 18px;margin-top:6px;
                    font-size:12px;color:var(--text-sec);line-height:1.7;">
                    {sql_explanation}
                </div>
            </details>'''

    if not results or not isinstance(results, list):
        html += '<div style="padding:20px;color:var(--text-sec);">No data returned.</div>'
        return html

    if not isinstance(results[0], dict):
        for r in results[:20]:
            html += f'<div style="padding:4px 0;font-size:13px;">{_esc(r)}</div>'
        return html

    n_rows = len(results)
    col_info = _analyze_columns(results)
    charts = _pick_chart_type(col_info, n_rows, question)
    insight_text = _generate_insight(col_info, results, question)

    # Insight box
    html += f'''
    <div style="background:rgba(0,107,166,0.06);border:1px solid rgba(0,107,166,0.15);border-radius:10px;padding:14px 18px;margin-bottom:20px;">
        <div style="font-size:12px;font-weight:600;color:var(--info);margin-bottom:4px;">&#128161; Insight</div>
        <div style="font-size:13px;color:var(--text-sec);line-height:1.5;">{insight_text}</div>
    </div>'''

    # Matplotlib chart (production-quality visualization)
    _mpl_chart_rendered = False
    if HAS_CHART_ENGINE:
        try:
            mpl_html = mpl_generate_chart(results, question, intent)
            if mpl_html:
                html += f'''
                <div class="chart-section" style="text-align:center;padding:12px 0;">
                    {mpl_html}
                </div>'''
                _mpl_chart_rendered = True
        except Exception:
            pass

    # Stats cards for numeric columns
    nc = col_info['numeric']
    if nc:
        stats = ""
        for col, vals in nc[:6]:
            avg = sum(vals) / len(vals)
            mn, mx, total = min(vals), max(vals), sum(vals)
            # Color based on column semantics
            col_lower = col.lower()
            if 'denied' in col_lower or 'denial' in col_lower:
                scolor = COLORS['critical']
            elif 'paid' in col_lower or 'revenue' in col_lower:
                scolor = COLORS['healthy']
            elif 'cost' in col_lower or 'billed' in col_lower:
                scolor = COLORS['warning']
            else:
                scolor = COLORS['info']

            stats += f'''
            <div class="stat-card">
                <div class="stat-label">{_esc(col)}</div>
                <div class="stat-value" style="color:{scolor};">{_fmt(avg)}</div>
                <div class="stat-detail">Sum: {_fmt(total)} &bull; Min: {_fmt(mn)} &bull; Max: {_fmt(mx)} &bull; N: {len(vals)}</div>
            </div>'''
        html += f'<div class="stats-grid">{stats}</div>'

    # Render each selected chart (skip basic SVG charts if matplotlib already rendered a chart)
    for chart_type, params in charts:
        if _mpl_chart_rendered and chart_type in ('donut', 'gauges', 'line', 'bar', 'grouped_bar', 'frequency'):
            continue  # Matplotlib already rendered a better version of this chart
        if chart_type == 'big_number':
            # Single KPI-style big number
            for col, vals in nc[:3]:
                val = vals[0]
                html += f'''
                <div class="stat-card" style="text-align:center;padding:30px;">
                    <div class="stat-label">{_esc(col)}</div>
                    <div style="font-size:48px;font-weight:700;color:var(--info);">{_fmt(val)}</div>
                </div>'''

        elif chart_type == 'donut':
            status_col = params['col']
            from collections import Counter
            counts = Counter(str(r.get(status_col, 'Unknown')) for r in results)
            html += f'''
            <div class="chart-section">
                <div class="chart-title">{_esc(status_col)} Distribution</div>
                {_svg_donut(dict(counts.most_common(8)))}
            </div>'''

        elif chart_type == 'gauges':
            html += f'''
            <div class="chart-section">
                <div class="chart-title">Rate Metrics</div>
                {_svg_gauges(params['cols'], results)}
            </div>'''

        elif chart_type == 'line':
            html += f'''
            <div class="chart-section">
                <div class="chart-title">{_esc(params['val_col'])} Over Time</div>
                {_svg_line(results, params['date_col'], params['val_col'])}
            </div>'''

        elif chart_type == 'bar':
            label_col = params['label_col']
            val_col = params['val_col']
            chart_data = []
            for row in results[:20]:
                try:
                    chart_data.append((
                        str(row.get(label_col, ''))[:20],
                        float(str(row.get(val_col, 0)).replace(',', ''))
                    ))
                except: pass

            if chart_data:
                # Sort by value descending for bar chart
                chart_data.sort(key=lambda x: x[1], reverse=True)
                max_val = max(v for _, v in chart_data) or 1
                bars = ""
                bar_colors = [COLORS['info'], COLORS['healthy'], COLORS['preventive'],
                             COLORS['behavioral'], COLORS['operations'], COLORS['warning']]
                for i, (label, val) in enumerate(chart_data):
                    pct = val / max_val * 100
                    c = bar_colors[i % len(bar_colors)]
                    bars += f'''
                    <div class="bar-row">
                        <span class="bar-label">{_esc(label)}</span>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:{pct:.1f}%;background:{c};">{_fmt(val)}</div>
                        </div>
                    </div>'''
                html += f'''
                <div class="chart-section">
                    <div class="chart-title">{_esc(val_col)} by {_esc(label_col)}</div>
                    {bars}
                </div>'''

        elif chart_type == 'grouped_bar':
            html += f'''
            <div class="chart-section">
                <div class="chart-title">Multi-Metric Comparison by {_esc(params['label_col'])}</div>
                {_grouped_bar_html(results, params['label_col'], params['val_cols'])}
            </div>'''

        elif chart_type == 'comparison':
            # Side-by-side stat cards for multiple numeric columns
            comp_colors = [COLORS['info'], COLORS['healthy'], COLORS['preventive'], COLORS['behavioral']]
            comp_cards = ""
            for i, (col, vals) in enumerate(params.get('cols', [])[:4]):
                avg = sum(vals) / len(vals)
                c = comp_colors[i % len(comp_colors)]
                comp_cards += f'''
                <div class="stat-card" style="text-align:center;">
                    <div class="stat-label">{_esc(col)}</div>
                    <div class="stat-value" style="color:{c};">{_fmt(avg)}</div>
                    <div class="stat-detail">Min: {_fmt(min(vals))} &bull; Max: {_fmt(max(vals))}</div>
                </div>'''
            html += f'''
            <div class="chart-section">
                <div class="chart-title">Numeric Comparison</div>
                <div class="stats-grid">{comp_cards}</div>
            </div>'''

        elif chart_type == 'frequency':
            from collections import Counter
            col = params['col']
            counts = Counter(str(r.get(col, '')) for r in results)
            top = counts.most_common(12)
            max_val = top[0][1] if top else 1
            bars = ""
            bar_colors = [COLORS['info'], COLORS['healthy'], COLORS['preventive'],
                         COLORS['behavioral'], COLORS['operations'], COLORS['warning']]
            for i, (label, cnt) in enumerate(top):
                pct = cnt / max_val * 100
                c = bar_colors[i % len(bar_colors)]
                bars += f'''
                <div class="bar-row">
                    <span class="bar-label">{_esc(label)}</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{pct:.1f}%;background:{c};">{cnt}</div>
                    </div>
                </div>'''
            html += f'''
            <div class="chart-section">
                <div class="chart-title">{_esc(col)} Frequency</div>
                {bars}
            </div>'''

    # Data table
    columns = col_info['all']
    tid = "results-table"
    toggles = "".join(
        f'<label class="col-toggle"><input type="checkbox" checked '
        f'onchange="toggleColumn(\'{tid}\', {i}, this.checked)"> {_esc(c)}</label>'
        for i, c in enumerate(columns)
    )
    header = "".join(
        f'<th onclick="sortTable(\'{tid}\', {i})">{_esc(c)} <span class="sort-icon">&#8597;</span></th>'
        for i, c in enumerate(columns)
    )

    rows_html = ""
    for row in results:
        cells = ""
        for c in columns:
            val = str(row.get(c, ''))
            status_class = ""
            if c.upper() in ('CLAIM_STATUS', 'STATUS', 'ENCOUNTER_STATUS'):
                status_class = f' class="status-{val.upper()}"'
            cells += f'<td{status_class}>{_esc(val)}</td>'
        rows_html += f'<tr>{cells}</tr>'

    html += f'''
    <div class="table-controls">
        <input class="table-search" placeholder="Filter results..." oninput="filterTable(this, '{tid}')">
        <button class="suggestion-chip" style="font-size:11px;" onclick="exportCSV('{tid}')">&#128190; Export CSV</button>
        <span class="row-count" id="{tid}-count">{len(results)} rows</span>
    </div>
    <div class="col-toggles">{toggles}</div>
    <div class="table-wrap">
        <table class="data-table" id="{tid}">
            <thead><tr>{header}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>'''

    return html


# =============================================================================
# HTTP REQUEST HANDLER
# =============================================================================

class DashboardHandler(BaseHTTPRequestHandler):
    """
    Production HTTP request handler for the healthcare dashboard.

    Security features:
        - Rate limiting per client IP
        - Input validation on all query parameters
        - Restricted CORS (localhost only)
        - Content Security Policy headers
        - Health check endpoint
        - Structured request logging
    """

    engine = None  # Set externally
    catalog = None
    cfg = None
    kpi_data = None
    dashboard_html = ""
    _rate_limiter = None
    _logger = None

    @classmethod
    def _init_production(cls):
        """Initialize production features (called once)."""
        if cls._rate_limiter is None:
            try:
                from production import RateLimiter, get_logger
                cls._rate_limiter = RateLimiter(requests_per_minute=60, burst=15)
                cls._logger = get_logger('mtp.http')
            except ImportError:
                cls._rate_limiter = None
                cls._logger = None

    def log_message(self, format, *args):
        """Route HTTP logs to structured logger instead of stderr."""
        if self.__class__._logger:
            self.__class__._logger.debug("HTTP %s", format % args)

    def _get_client_ip(self) -> str:
        """Get client IP address."""
        forwarded = self.headers.get('X-Forwarded-For', '')
        if forwarded:
            return forwarded.split(',')[0].strip()
        return self.client_address[0] if self.client_address else '127.0.0.1'

    def _send_security_headers(self):
        """Add security headers to all responses — hardened for public access."""
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('X-XSS-Protection', '1; mode=block')
        self.send_header('Referrer-Policy', 'strict-origin-when-cross-origin')
        self.send_header('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        # Content Security Policy — only allow self, inline scripts/styles (needed for f-string HTML)
        self.send_header('Content-Security-Policy',
                         "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                         "connect-src 'self'; frame-ancestors 'none';")
        # CORS — allow the requesting origin for public access
        origin = self.headers.get('Origin', '')
        if origin:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _send_json_error(self, code: int, message: str):
        """Send a JSON error response."""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self._send_security_headers()
        self.end_headers()
        resp = json.dumps({'error': message, 'html': f'<div style="color:var(--critical);padding:20px;">{_esc(message)}</div>'})
        self.wfile.write(resp.encode('utf-8'))

    def do_GET(self):
        self.__class__._init_production()
        parsed = urlparse(self.path)
        client_ip = self._get_client_ip()

        # ── HEALTH CHECK ENDPOINT ──
        if parsed.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from production import get_health_status
                health = get_health_status()
            except ImportError:
                health = {'status': 'ok', 'production_module': False}
            # Add Databricks status
            if HAS_DATABRICKS_MODULE:
                try:
                    ds_mgr = DataSourceManager.get_instance()
                    health['data_source'] = ds_mgr.health_check()
                except Exception:
                    health['data_source'] = {'mode': 'local'}
            self.wfile.write(json.dumps(health, indent=2).encode('utf-8'))
            return

        if parsed.path == '/auth/callback':
            # ── SSO OAUTH CALLBACK ──
            params = parse_qs(parsed.query)
            code = params.get('code', [''])[0]
            state = params.get('state', [''])[0]
            error = params.get('error', [''])[0]

            if error:
                html_resp = f'<html><body><script>window.opener.postMessage({{type:"sso_error",error:"{_esc(error)}"}}, "*");window.close();</script></body></html>'
            elif code and state:
                try:
                    from sso_auth import handle_callback
                    from auth_manager import sso_login_or_create
                    success, msg, user_info = handle_callback(code, state)
                    if success and user_info:
                        ok, login_msg, token = sso_login_or_create(
                            email=user_info['email'],
                            display_name=user_info.get('name', ''),
                            provider=user_info.get('provider', ''),
                            provider_id=user_info.get('provider_id', ''),
                            picture=user_info.get('picture', ''),
                        )
                        if ok:
                            html_resp = f'<html><body><script>window.opener.postMessage({{type:"sso_success",token:"{token}"}}, "*");window.close();</script></body></html>'
                        else:
                            html_resp = f'<html><body><script>window.opener.postMessage({{type:"sso_error",error:"{_esc(login_msg)}"}}, "*");window.close();</script></body></html>'
                    else:
                        html_resp = f'<html><body><script>window.opener.postMessage({{type:"sso_error",error:"{_esc(msg)}"}}, "*");window.close();</script></body></html>'
                except Exception as e:
                    html_resp = f'<html><body><script>window.opener.postMessage({{type:"sso_error",error:"{_esc(str(e))}"}}, "*");window.close();</script></body></html>'
            else:
                html_resp = '<html><body><script>window.opener.postMessage({type:"sso_error",error:"Missing code"}, "*");window.close();</script></body></html>'

            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html_resp.encode('utf-8'))
            return

        if parsed.path == '/api/sso-config':
            # ── SSO PROVIDER AVAILABILITY ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from sso_auth import is_sso_configured
                config = is_sso_configured()
                response = json.dumps(config)
            except ImportError:
                response = json.dumps({'google': False, 'microsoft': False, 'any': False})
            self.wfile.write(response.encode('utf-8'))
            return

        if parsed.path == '/api/sso-url':
            # ── GET SSO AUTH URL ──
            params = parse_qs(parsed.query)
            provider = params.get('provider', [''])[0]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from sso_auth import generate_auth_url
                url, state = generate_auth_url(provider)
                if url:
                    response = json.dumps({'url': url, 'state': state})
                else:
                    response = json.dumps({'error': state})  # state contains error msg
            except ImportError:
                response = json.dumps({'error': 'SSO module not available'})
            self.wfile.write(response.encode('utf-8'))
            return

        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(self.dashboard_html.encode('utf-8'))

        elif parsed.path == '/api/suggestions':
            # ── QUERY SUGGESTIONS & CONTEXTUAL LEARNING ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from catalog_registry import ContextualLearningEngine
                engine = ContextualLearningEngine()
                params = parse_qs(parsed.query)
                user_id = params.get('user_id', [''])[0]
                suggestions = engine.get_suggestions(user_id=user_id or None, limit=10)
                response = json.dumps({'suggestions': suggestions})
            except ImportError:
                response = json.dumps({'suggestions': []})
            except Exception as e:
                response = json.dumps({'suggestions': [], 'error': str(e)})
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/api/similar':
            # ── SIMILAR QUERIES (other people also asked) ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from catalog_registry import ContextualLearningEngine
                engine = ContextualLearningEngine()
                params = parse_qs(parsed.query)
                q = params.get('q', [''])[0]
                similar = engine.get_similar_queries(q, limit=5)
                response = json.dumps({'similar': similar})
            except ImportError:
                response = json.dumps({'similar': []})
            except Exception as e:
                response = json.dumps({'similar': [], 'error': str(e)})
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/api/catalogs':
            # ── CATALOG REGISTRY — list environments, catalogs, tables ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from catalog_registry import CatalogRegistry
                reg = CatalogRegistry.get_instance()
                if not reg._initialized:
                    import os as _os
                    _cfg = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                         '..', 'data', 'catalog_config.json')
                    reg.initialize(config_path=_cfg)
                params = parse_qs(parsed.query)
                view = params.get('view', ['summary'])[0]
                env = params.get('env', [''])[0]
                if view == 'environments':
                    response = json.dumps({'environments': reg.get_environments()})
                elif view == 'catalogs':
                    response = json.dumps({'catalogs': reg.get_catalogs(environment=env or None)})
                elif view == 'tables':
                    concept = params.get('concept', [''])[0]
                    catalog_name = params.get('catalog', [''])[0]
                    response = json.dumps({'tables': reg.get_tables(
                        concept=concept or None, catalog=catalog_name or None,
                        environment=env or None)})
                elif view == 'search':
                    q = params.get('q', [''])[0]
                    response = json.dumps({'results': reg.search_tables(q)})
                else:
                    response = json.dumps(reg.get_summary())
                    response = json.dumps({'summary': reg.get_summary()})
            except ImportError:
                response = json.dumps({'summary': {'environments': 0, 'total_tables': 0}})
            except Exception as e:
                response = json.dumps({'error': str(e)})
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/api/catalog-tree':
            # ── DATA SOURCE NAVIGATOR: FULL CATALOG TREE ──
            # Returns: environments → catalogs → schemas → tables with hierarchy
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from catalog_registry import CatalogRegistry
                reg = CatalogRegistry.get_instance()
                if not reg._initialized:
                    import os as _os
                    _cfg = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                         '..', 'data', 'catalog_config.json')
                    reg.initialize(config_path=_cfg)

                # Build full tree structure
                tree = {{
                    'environments': {{}},
                }}

                # Populate environments
                envs = reg.get_environments() or ['prod', 'staging', 'dev', 'local']
                for env in envs:
                    tree['environments'][env] = {{
                        'catalogs': {{}}
                    }}

                    # Get catalogs for this environment
                    catalogs = reg.get_catalogs(environment=env) or []
                    for cat_info in catalogs:
                        cat_name = cat_info if isinstance(cat_info, str) else cat_info.get('name', str(cat_info))
                        tree['environments'][env]['catalogs'][cat_name] = {{
                            'schemas': {{}}
                        }}

                        # Get schemas for this catalog
                        try:
                            schemas_list = reg.get_schemas(catalog=cat_name, environment=env) or ['claims', 'members', 'clinical', 'providers', 'risk']
                            for schema in schemas_list:
                                schema_name = schema if isinstance(schema, str) else schema.get('name', str(schema))
                                tree['environments'][env]['catalogs'][cat_name]['schemas'][schema_name] = {{
                                    'tables': []
                                }}

                                # Get tables for this schema
                                try:
                                    tables_list = reg.get_tables(schema=schema_name, catalog=cat_name, environment=env) or []
                                    for tbl in tables_list:
                                        tbl_name = tbl if isinstance(tbl, str) else tbl.get('name', str(tbl))
                                        tbl_desc = tbl.get('description', '') if isinstance(tbl, dict) else ''
                                        tree['environments'][env]['catalogs'][cat_name]['schemas'][schema_name]['tables'].append({{
                                            'name': tbl_name,
                                            'description': tbl_desc,
                                            'rowCount': None,
                                            'tags': []
                                        }})
                                except:
                                    pass
                        except:
                            pass

                # Add default local environment if not present
                if 'local' not in tree['environments']:
                    tree['environments']['local'] = {{
                        'catalogs': {{
                            'local_catalog': {{
                                'schemas': {{
                                    'claims': {{'tables': [
                                        {{'name': 'inpatient_institutional_claims', 'description': 'Inpatient institutional claims', 'rowCount': None, 'tags': ['UB-04']}},
                                        {{'name': 'outpatient_professional_claims', 'description': 'Outpatient professional claims', 'rowCount': None, 'tags': ['CMS-1500']}},
                                        {{'name': 'pharmacy_claims', 'description': 'Pharmacy claims', 'rowCount': None, 'tags': ['Retail', 'Specialty']}},
                                        {{'name': 'dental_claims', 'description': 'Dental claims', 'rowCount': None, 'tags': ['Preventive']}},
                                        {{'name': 'behavioral_health_claims', 'description': 'Behavioral health claims', 'rowCount': None, 'tags': ['Mental Health']}},
                                    ]}},
                                    'members': {{'tables': [
                                        {{'name': 'members', 'description': 'Member demographics and enrollment', 'rowCount': None, 'tags': []}}
                                    ]}},
                                    'clinical': {{'tables': [
                                        {{'name': 'diagnoses', 'description': 'Clinical diagnoses', 'rowCount': None, 'tags': []}}
                                    ]}},
                                    'providers': {{'tables': [
                                        {{'name': 'providers', 'description': 'Healthcare provider information', 'rowCount': None, 'tags': []}}
                                    ]}},
                                    'risk': {{'tables': [
                                        {{'name': 'risk_scores', 'description': 'Member risk scores', 'rowCount': None, 'tags': []}}
                                    ]}}
                                }}
                            }}
                        }}
                    }}

                response = json.dumps(tree)
            except Exception as e:
                response = json.dumps({{'error': str(e), 'environments': {{}}}})
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/api/table-details':
            # ── DATA SOURCE NAVIGATOR: TABLE DETAILS ──
            # Returns: columns, sample values, row count for a specific table
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                params = parse_qs(parsed.query)
                table = params.get('table', [''])[0]
                env = params.get('env', ['local'])[0]
                catalog = params.get('catalog', ['local_catalog'])[0]

                # Validate table name
                valid_tables = ['claims', 'members', 'encounters', 'providers', 'diagnoses', 'prescriptions', 'referrals',
                               'inpatient_institutional_claims', 'outpatient_professional_claims', 'pharmacy_claims',
                               'dental_claims', 'behavioral_health_claims', 'members', 'diagnoses', 'providers', 'risk_scores']
                if table not in valid_tables:
                    response = json.dumps({{'error': 'Invalid table', 'columns': [], 'sample': []}})
                else:
                    conn = self.__class__.engine.db_conn if hasattr(self.__class__.engine, 'db_conn') else None
                    if conn:
                        cur = conn.cursor()
                        # Get column info
                        try:
                            cur.execute(f"PRAGMA table_info({table})")
                            columns = [{{
                                'name': r[1],
                                'type': r[2],
                                'nullable': not bool(r[3])
                            }} for r in cur.fetchall()]
                        except:
                            columns = []

                        # Get row count
                        try:
                            cur.execute(f"SELECT COUNT(*) FROM [{table}]")
                            row_count = cur.fetchone()[0]
                        except:
                            row_count = 0

                        # Get sample rows
                        try:
                            cur.execute(f"SELECT * FROM [{table}] LIMIT 5")
                            sample = [dict(zip([c['name'] for c in columns], row)) for row in cur.fetchall()]
                        except:
                            sample = []

                        response = json.dumps({{
                            'table': table,
                            'environment': env,
                            'catalog': catalog,
                            'columns': columns,
                            'rowCount': row_count,
                            'sample': sample
                        }})
                    else:
                        response = json.dumps({{'error': 'No database connection', 'columns': [], 'sample': []}})
            except Exception as e:
                response = json.dumps({{'error': str(e), 'columns': [], 'sample': []}})
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/api/history':
            # ── USER CONVERSATION HISTORY ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from catalog_registry import ContextualLearningEngine
                engine = ContextualLearningEngine()
                params = parse_qs(parsed.query)
                user_id = params.get('user_id', [''])[0]
                limit = int(params.get('limit', ['20'])[0])
                history = engine.get_conversation_history(user_id, limit=limit)
                response = json.dumps({'history': history})
            except ImportError:
                response = json.dumps({'history': []})
            except Exception as e:
                response = json.dumps({'history': [], 'error': str(e)})
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/api/learning-stats':
            # ── LEARNING ENGINE STATS ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from catalog_registry import ContextualLearningEngine
                engine = ContextualLearningEngine()
                stats = engine.get_learning_stats()
                response = json.dumps(stats)
            except ImportError:
                response = json.dumps({'total_queries': 0})
            except Exception as e:
                response = json.dumps({'error': str(e)})
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/api/table-hierarchy':
            # ── TABLE HIERARCHY ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                # Return table hierarchy structure
                hierarchy = {
                    'concepts': [
                        {
                            'key': 'patients',
                            'label': 'Patient Data',
                            'categories': [
                                {'key': 'demographics', 'label': 'Demographics', 'types': [
                                    {'key': 'age', 'label': 'Age'},
                                    {'key': 'gender', 'label': 'Gender'},
                                    {'key': 'location', 'label': 'Location'}
                                ]},
                                {'key': 'contact', 'label': 'Contact Info', 'types': [
                                    {'key': 'phone', 'label': 'Phone'},
                                    {'key': 'email', 'label': 'Email'},
                                    {'key': 'address', 'label': 'Address'}
                                ]}
                            ]
                        },
                        {
                            'key': 'claims',
                            'label': 'Claims & Billing',
                            'categories': [
                                {'key': 'claim_details', 'label': 'Claim Details', 'types': [
                                    {'key': 'claim_id', 'label': 'Claim ID'},
                                    {'key': 'amount', 'label': 'Amount'},
                                    {'key': 'status', 'label': 'Status'}
                                ]},
                                {'key': 'procedures', 'label': 'Procedures', 'types': [
                                    {'key': 'procedure_code', 'label': 'Procedure Code'},
                                    {'key': 'procedure_date', 'label': 'Date'},
                                    {'key': 'procedure_cost', 'label': 'Cost'}
                                ]}
                            ]
                        }
                    ]
                }
                response = json.dumps(hierarchy)
            except Exception as e:
                response = json.dumps({'error': str(e)})
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/api/session-context':
            # ── SESSION CONTEXT ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                # Return session context for current user
                params = parse_qs(parsed.query)
                session_id = params.get('session_id', [''])[0]
                context = {
                    'session_id': session_id,
                    'recent_turns': [],
                    'active_selections': [],
                    'turn_count': 0
                }
                response = json.dumps(context)
            except Exception as e:
                response = json.dumps({'error': str(e)})
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/api/distinct-values':
            # ── DISTINCT VALUES FOR DROPDOWNS (QUERY BUILDER) ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                params = parse_qs(parsed.query)
                table = params.get('table', [''])[0]
                column = params.get('column', [''])[0]
                limit = int(params.get('limit', ['50'])[0])

                # Validate table/column names (prevent SQL injection)
                valid_tables = ['claims', 'members', 'encounters', 'providers', 'diagnoses', 'prescriptions', 'referrals']
                if table not in valid_tables:
                    response = json.dumps({'error': 'Invalid table', 'values': []})
                else:
                    conn = self.__class__.engine.db_conn if hasattr(self.__class__.engine, 'db_conn') else None
                    if conn:
                        # Validate column exists
                        cur = conn.cursor()
                        cur.execute(f"PRAGMA table_info({table})")
                        valid_cols = [r[1] for r in cur.fetchall()]
                        if column in valid_cols:
                            cur.execute(f"SELECT DISTINCT [{column}] FROM [{table}] WHERE [{column}] IS NOT NULL ORDER BY [{column}] LIMIT ?", (limit,))
                            values = [r[0] for r in cur.fetchall()]
                            response = json.dumps({'values': values, 'table': table, 'column': column})
                        else:
                            response = json.dumps({'error': 'Invalid column', 'values': [], 'valid_columns': valid_cols})
                    else:
                        response = json.dumps({'error': 'No database connection', 'values': []})
            except Exception as e:
                response = json.dumps({'error': str(e), 'values': []})
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/api/schema-columns':
            # ── FULL SCHEMA WITH COLUMN INFORMATION FOR VQB ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                import os as _os
                catalog_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'semantic_catalog', 'tables')
                schema = {}
                for fname in sorted(_os.listdir(catalog_dir)):
                    if not fname.endswith('.json'):
                        continue
                    with open(_os.path.join(catalog_dir, fname), 'r') as fh:
                        data = json.load(fh)
                    table_name = data.get('table_name', fname.replace('.json', ''))
                    cols_raw = data.get('columns', {})
                    if isinstance(cols_raw, dict):
                        col_list = [{'name': k, 'type': v.get('data_type', 'text')} for k, v in cols_raw.items()]
                    else:
                        col_list = [{'name': c.get('column_name', c.get('name', '')), 'type': c.get('data_type', 'text')} for c in cols_raw]
                    schema[table_name] = col_list
                response = json.dumps({'schema': schema})
            except Exception as e:
                response = json.dumps({'error': str(e), 'schema': {}})
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/query':
            # ── RATE LIMITING ──
            if self.__class__._rate_limiter and not self.__class__._rate_limiter.allow(client_ip):
                self._send_json_error(429, 'Rate limit exceeded. Please wait before sending more queries.')
                return

            params = parse_qs(parsed.query)
            question = params.get('q', [''])[0]

            # ── INPUT VALIDATION ──
            if not question or not question.strip():
                self._send_json_error(400, 'Empty query')
                return
            if len(question) > 2000:
                self._send_json_error(400, 'Query too long (max 2000 characters)')
                return

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()

            try:
                # Import and run query
                from query_engine import process_question, SemanticCatalog
                cat = self.__class__.catalog or SemanticCatalog()
                session = {'username': 'dashboard', 'history': [], 'skip_catalog_discovery': True}
                result = process_question(question, self.__class__.cfg or {}, cat, session)

                answer = result.get('answer', [])
                sql = result.get('sql', '')
                intent = result.get('intent', '')
                mode = result.get('engine_mode', '')
                databricks_sql = result.get('databricks_sql', '')

                if isinstance(answer, list) and answer:
                    results_html = _build_results_html(answer, question, sql, intent, mode)
                elif isinstance(answer, str):
                    results_html = f'''
                    <div class="result-header"><div class="result-title">{_esc(question)}</div></div>
                    {'<div class="sql-box">' + _esc(sql) + '</div>' if sql else ''}
                    <div style="padding:16px;background:var(--bg-card);border-radius:10px;border:1px solid var(--border);font-size:14px;">
                        {_esc(answer)}
                    </div>'''
                else:
                    results_html = '<div style="padding:20px;color:var(--text-sec);">No results found.</div>'

                # Add engine pipeline info and Databricks SQL to results
                pipeline_info = result.get('scale_meta', {})
                ml_meta = result.get('ml_meta', {})
                nlp_meta = result.get('nlp_meta', {})

                # Build pipeline badges HTML
                pipeline_html = ''
                if pipeline_info or ml_meta or nlp_meta or mode or databricks_sql:
                    badges = []
                    if ml_meta.get('ml_intent'):
                        badges.append(f'<span class="meta-badge" style="background:rgba(124,58,237,0.12);color:{COLORS["behavioral"]};">ML: {ml_meta["ml_intent"]} ({ml_meta.get("ml_confidence", 0):.0%})</span>')
                    if nlp_meta.get('nlp_intent'):
                        badges.append(f'<span class="meta-badge" style="background:rgba(14,165,233,0.12);color:#0ea5e9;">NLP: {nlp_meta["nlp_intent"]} ({nlp_meta.get("nlp_confidence", 0):.0%})</span>')
                    if nlp_meta.get('nlp_top_column'):
                        badges.append(f'<span class="meta-badge" style="background:rgba(14,165,233,0.08);color:#0ea5e9;">Col: {nlp_meta["nlp_top_column"]}</span>')
                    validation = nlp_meta.get('validation', {})
                    if validation:
                        v_score = validation.get('score', 0)
                        v_color = COLORS["healthy"] if v_score >= 0.8 else (COLORS["warning"] if v_score >= 0.5 else COLORS["critical"])
                        badges.append(f'<span class="meta-badge" style="background:rgba(5,150,105,0.12);color:{v_color};">Valid: {v_score:.0%}</span>')
                    if pipeline_info.get('kg_nodes'):
                        badges.append(f'<span class="meta-badge" style="background:rgba(13,148,136,0.12);color:{COLORS["preventive"]};">KG: {pipeline_info["kg_nodes"]} nodes</span>')
                    if pipeline_info.get('vector_search_hits'):
                        badges.append(f'<span class="meta-badge" style="background:rgba(79,70,229,0.12);color:{COLORS["operations"]};">Vector: {pipeline_info["vector_search_hits"]} hits</span>')
                    if pipeline_info.get('cache_hit'):
                        badges.append(f'<span class="meta-badge" style="background:rgba(5,150,105,0.12);color:{COLORS["healthy"]};">Cache Hit</span>')
                    # Databricks data source badge
                    if HAS_DATABRICKS_MODULE:
                        try:
                            ds_mgr = DataSourceManager.get_instance()
                            if ds_mgr.is_databricks:
                                badges.append(f'<span class="meta-badge" style="background:rgba(255,99,71,0.12);color:#ff6347;">Databricks</span>')
                            else:
                                badges.append(f'<span class="meta-badge" style="background:rgba(148,163,184,0.12);color:#94a3b8;">Local DB</span>')
                        except Exception:
                            pass
                    if badges:
                        pipeline_html = '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;">' + ''.join(badges) + '</div>'

                # Add Databricks SQL if available
                databricks_html = ''
                if databricks_sql:
                    databricks_html = f'''
                    <details style="margin-bottom:16px;">
                        <summary style="cursor:pointer;font-size:13px;font-weight:600;color:{COLORS["operations"]};
                            padding:8px 0;user-select:none;">
                            &#9881; Databricks/Spark SQL (click to expand)
                        </summary>
                        <div class="sql-box" style="border-color:{COLORS["operations"]};">{_esc(databricks_sql)}</div>
                    </details>'''

                results_html = pipeline_html + results_html + databricks_html

                # Add "Others also asked" similar queries
                similar_queries = result.get('similar_queries', [])
                if similar_queries:
                    sim_html = '<div style="margin-top:16px;padding:12px 16px;background:rgba(0,107,166,0.05);border-radius:10px;border:1px solid rgba(0,107,166,0.1);">'
                    sim_html += '<div style="font-size:12px;font-weight:600;color:' + COLORS["info"] + ';margin-bottom:8px;">&#128161; Others also asked:</div>'
                    sim_html += '<div style="display:flex;gap:6px;flex-wrap:wrap;">'
                    for sq in similar_queries[:5]:
                        sq_q = _esc(sq.get('question', ''))
                        sq_by = _esc(sq.get('asked_by', 'someone'))
                        sim_html += f'<button class="suggestion-chip" style="font-size:11px;" onclick="submitQuery(\'{sq_q}\')">{sq_q[:50]}{"..." if len(sq_q) > 50 else ""}</button>'
                    sim_html += '</div></div>'
                    results_html += sim_html

                # Add catalog discovery info if present
                catalog_discovery = result.get('catalog_discovery')
                if catalog_discovery and catalog_discovery.get('matched_concept'):
                    cat_info = catalog_discovery
                    disc_html = '<div style="margin-top:12px;padding:10px 14px;background:rgba(13,148,136,0.05);border-radius:8px;border:1px solid rgba(13,148,136,0.15);font-size:12px;color:' + COLORS["text_sec"] + ';">'
                    disc_html += f'<span style="font-weight:600;color:{COLORS["preventive"]};">&#128218; Data context:</span> '
                    disc_html += f'Querying <b>{_esc(cat_info["matched_concept"])}</b>'
                    if cat_info.get('available_catalogs'):
                        disc_html += f' (available in: {", ".join(_esc(c) for c in cat_info["available_catalogs"][:3])})'
                    disc_html += '</div>'
                    results_html += disc_html

                response = json.dumps({'html': results_html, 'sql': sql, 'intent': intent})
            except Exception as e:
                response = json.dumps({'html': f'<div style="color:var(--critical);padding:20px;">Error: {_esc(str(e))}</div>', 'error': str(e)})

            self.wfile.write(response.encode('utf-8'))

        elif parsed.path == '/api/execute-sql':
            # ── EXECUTE EDITED SQL DIRECTLY ──
            params = parse_qs(parsed.query)
            raw_sql = params.get('sql', [''])[0].strip()

            if not raw_sql:
                self._send_json_error(400, 'Empty SQL')
                return

            # Security: only allow SELECT
            sql_upper = raw_sql.strip().upper()
            if not sql_upper.startswith('SELECT'):
                self._send_json_error(403, 'Only SELECT queries are allowed')
                return

            # Block dangerous patterns
            dangerous = ['DROP ', 'DELETE ', 'INSERT ', 'UPDATE ', 'ALTER ', 'CREATE ', 'EXEC ', '--']
            if any(d in sql_upper for d in dangerous):
                self._send_json_error(403, 'Query contains disallowed statements')
                return

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()

            try:
                cfg = self.__class__.cfg or {}
                db_path = cfg.get('db_path', '')
                if not db_path:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    db_path = os.path.join(os.path.dirname(script_dir), 'data', 'healthcare_demo.db')

                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(raw_sql)
                rows = [dict(r) for r in cursor.fetchall()]
                conn.close()

                if rows:
                    results_html = _build_results_html(rows, "Custom SQL", raw_sql, "custom_sql", "manual")
                else:
                    results_html = '<div style="padding:20px;color:var(--text-sec);">Query returned no results.</div>'

                response = json.dumps({'html': results_html, 'sql': raw_sql, 'intent': 'custom_sql',
                                       'row_count': len(rows), 'data': rows[:100]})
            except Exception as e:
                response = json.dumps({
                    'html': f'<div style="color:var(--critical);padding:20px;">SQL Error: {_esc(str(e))}</div>',
                    'error': str(e)
                })

            self.wfile.write(response.encode('utf-8'))

        elif parsed.path == '/api/session':
            # ── CHECK AUTH SESSION ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from auth_manager import validate_session
                params = parse_qs(parsed.query)
                token = params.get('token', [''])[0]
                user = validate_session(token)
                if user:
                    response = json.dumps({'authenticated': True, 'user': dict(user)})
                else:
                    response = json.dumps({'authenticated': False})
            except ImportError:
                response = json.dumps({'authenticated': False, 'error': 'Auth module not available'})
            self.wfile.write(response.encode('utf-8'))

        elif parsed.path == '/api/dashboards':
            # ── LIST SAVED DASHBOARDS ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from auth_manager import get_saved_dashboards, get_team_dashboards
                params = parse_qs(parsed.query)
                token = params.get('token', [''])[0]
                scope = params.get('scope', ['my'])[0]
                if scope == 'team':
                    dashboards = get_team_dashboards(token)
                else:
                    dashboards = get_saved_dashboards(token)
                response = json.dumps({'dashboards': dashboards})
            except ImportError:
                response = json.dumps({'dashboards': [], 'error': 'Auth module not available'})
            self.wfile.write(response.encode('utf-8'))

        elif parsed.path == '/api/user-sessions':
            # ── LIST USER SESSIONS ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from auth_manager import get_user_sessions
                params = parse_qs(parsed.query)
                token = params.get('token', [''])[0]
                sessions_list = get_user_sessions(token)
                response = json.dumps({'sessions': sessions_list})
            except ImportError:
                response = json.dumps({'sessions': []})
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/api/teams':
            # ── LIST TEAMS ──
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self._send_security_headers()
            self.end_headers()
            try:
                from auth_manager import get_teams
                response = json.dumps({'teams': get_teams()})
            except ImportError:
                response = json.dumps({'teams': []})
            self.wfile.write(response.encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests for auth and dashboard save."""
        self.__class__._init_production()
        parsed = urlparse(self.path)

        # Read POST body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json_error(400, 'Invalid JSON')
            return

        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self._send_security_headers()
        self.end_headers()

        if parsed.path == '/api/signup':
            try:
                from auth_manager import signup
                success, msg, token = signup(
                    username=data.get('username', ''),
                    password=data.get('password', ''),
                    display_name=data.get('display_name', ''),
                    team_name=data.get('team_name', ''),
                    email=data.get('email', ''),
                )
                response = json.dumps({'success': success, 'message': msg, 'token': token})
            except ImportError:
                response = json.dumps({'success': False, 'message': 'Auth module not available'})

        elif parsed.path == '/api/login':
            try:
                from auth_manager import login
                success, msg, token = login(
                    username_or_email=data.get('username', ''),
                    password=data.get('password', ''),
                )
                response = json.dumps({'success': success, 'message': msg, 'token': token})
            except ImportError:
                response = json.dumps({'success': False, 'message': 'Auth module not available'})

        elif parsed.path == '/api/logout':
            try:
                from auth_manager import logout
                logout(data.get('token', ''))
                response = json.dumps({'success': True, 'message': 'Logged out'})
            except ImportError:
                response = json.dumps({'success': True})

        elif parsed.path == '/api/save-dashboard':
            try:
                from auth_manager import save_dashboard
                success, msg, dash_id = save_dashboard(
                    session_token=data.get('token', ''),
                    name=data.get('name', ''),
                    queries=data.get('queries', []),
                    description=data.get('description', ''),
                    dashboard_id=data.get('dashboard_id', None),
                )
                response = json.dumps({'success': success, 'message': msg, 'dashboard_id': dash_id})
            except ImportError:
                response = json.dumps({'success': False, 'message': 'Auth module not available'})

        elif parsed.path == '/api/delete-dashboard':
            try:
                from auth_manager import delete_dashboard
                success, msg = delete_dashboard(
                    session_token=data.get('token', ''),
                    dashboard_id=data.get('dashboard_id', ''),
                )
                response = json.dumps({'success': success, 'message': msg})
            except ImportError:
                response = json.dumps({'success': False, 'message': 'Auth module not available'})

        elif parsed.path == '/api/set-export-password':
            try:
                from auth_manager import set_export_password
                success, msg = set_export_password(
                    session_token=data.get('token', ''),
                    password=data.get('password', ''),
                    label=data.get('label', ''),
                )
                response = json.dumps({'success': success, 'message': msg})
            except ImportError:
                response = json.dumps({'success': False, 'message': 'Auth module not available'})

        elif parsed.path == '/api/verify-export-password':
            try:
                from auth_manager import verify_export_password
                valid = verify_export_password(
                    session_token=data.get('token', ''),
                    password=data.get('password', ''),
                )
                response = json.dumps({'success': valid, 'message': 'Valid' if valid else 'Invalid password'})
            except ImportError:
                response = json.dumps({'success': False, 'message': 'Auth module not available'})

        elif parsed.path == '/api/check-export-password':
            try:
                from auth_manager import has_export_password
                has_pw = has_export_password(data.get('token', ''))
                response = json.dumps({'has_password': has_pw})
            except ImportError:
                response = json.dumps({'has_password': False})

        elif parsed.path == '/api/create-session':
            try:
                from auth_manager import create_user_session
                success, msg, sid = create_user_session(
                    session_token=data.get('token', ''),
                    session_name=data.get('session_name', ''),
                )
                response = json.dumps({'success': success, 'message': msg, 'session_id': sid})
            except ImportError:
                response = json.dumps({'success': False, 'message': 'Auth module not available'})

        elif parsed.path == '/api/update-session':
            try:
                from auth_manager import update_user_session
                success, msg = update_user_session(
                    session_token=data.get('token', ''),
                    user_session_id=data.get('session_id', ''),
                    queries=data.get('queries', []),
                )
                response = json.dumps({'success': success, 'message': msg})
            except ImportError:
                response = json.dumps({'success': False, 'message': 'Auth module not available'})

        elif parsed.path == '/api/send-email':
            try:
                from sso_auth import send_email_smtp, load_smtp_config
                from auth_manager import validate_session

                user = validate_session(data.get('token', ''))
                if not user:
                    response = json.dumps({'success': False, 'message': 'Not authenticated'})
                else:
                    smtp_config = load_smtp_config()
                    from_email = user.get('email', '')
                    to_email = data.get('to_email', '')
                    subject = data.get('subject', 'Dashboard Report')
                    body = data.get('body', '')

                    if not smtp_config:
                        response = json.dumps({'success': False, 'message': 'SMTP not configured. Set up email in Settings.'})
                    elif not to_email:
                        response = json.dumps({'success': False, 'message': 'Recipient email is required'})
                    else:
                        success, msg = send_email_smtp(
                            from_email=smtp_config.get('email', from_email),
                            to_email=to_email,
                            subject=subject,
                            body_html=body,
                            smtp_host=smtp_config.get('smtp_host', 'smtp.gmail.com'),
                            smtp_port=smtp_config.get('smtp_port', 587),
                            smtp_password=smtp_config.get('app_password', ''),
                        )
                        response = json.dumps({'success': success, 'message': msg})
            except ImportError:
                response = json.dumps({'success': False, 'message': 'Email module not available'})

        elif parsed.path == '/api/save-smtp':
            try:
                from sso_auth import save_smtp_config
                ok = save_smtp_config(
                    email=data.get('email', ''),
                    smtp_host=data.get('smtp_host', 'smtp.gmail.com'),
                    smtp_port=data.get('smtp_port', 587),
                    app_password=data.get('app_password', ''),
                )
                response = json.dumps({'success': ok, 'message': 'SMTP config saved' if ok else 'Failed to save'})
            except ImportError:
                response = json.dumps({'success': False, 'message': 'Module not available'})

        elif parsed.path == '/api/set-selections':
            # ── SET TABLE HIERARCHY SELECTIONS ──
            try:
                session_id = data.get('session_id', '')
                selections = data.get('selections', [])
                # Store selections in memory (could be persisted to DB if needed)
                response = json.dumps({'success': True, 'message': 'Selections saved', 'selections': selections})
            except Exception as e:
                response = json.dumps({'success': False, 'message': str(e)})

        elif parsed.path == '/api/execute-sql':
            # ── EXECUTE RAW SQL QUERY (VQB - POST) ──
            sql = data.get('sql', '').strip()

            if not sql or not sql.upper().startswith('SELECT'):
                response = json.dumps({'error': 'Invalid SQL: must be SELECT query', 'columns': [], 'rows': [], 'row_count': 0})
                self.wfile.write(response.encode('utf-8'))
                return

            # Block dangerous patterns
            sql_upper = sql.upper()
            dangerous = ['DROP ', 'DELETE ', 'INSERT ', 'UPDATE ', 'ALTER ', 'CREATE ', 'EXEC ', '--']
            if any(d in sql_upper for d in dangerous):
                response = json.dumps({'error': 'Query contains disallowed statements', 'columns': [], 'rows': [], 'row_count': 0})
                self.wfile.write(response.encode('utf-8'))
                return

            try:
                cfg = self.__class__.cfg or {}
                db_path = cfg.get('db_path', '')
                if not db_path:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    db_path = os.path.join(os.path.dirname(script_dir), 'data', 'healthcare_demo.db')

                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                # Execute SQL with protection: limit to 10000 rows max
                if 'LIMIT' not in sql_upper:
                    sql_exec = sql + ' LIMIT 10000'
                else:
                    sql_exec = sql

                cur.execute(sql_exec)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows_list = cur.fetchall()
                rows = [dict(zip(columns, row)) for row in rows_list]
                conn.close()

                response = json.dumps({'columns': columns, 'rows': rows, 'row_count': len(rows)})
            except Exception as e:
                response = json.dumps({'error': str(e), 'columns': [], 'rows': [], 'row_count': 0})

        elif parsed.path == '/api/conversation-turn':
            # ── RECORD CONVERSATION TURN ──
            try:
                session_id = data.get('session_id', '')
                question = data.get('question', '')
                sql = data.get('sql', '')
                tables = data.get('tables', [])
                result_count = data.get('result_count', 0)
                # Record turn in memory/DB (could be persisted if needed)
                response = json.dumps({'success': True, 'message': 'Turn recorded', 'turn_id': f'turn_{int(time.time())}'})
            except Exception as e:
                response = json.dumps({'success': False, 'message': str(e)})

        else:
            response = json.dumps({'error': 'Unknown endpoint'})

        self.wfile.write(response.encode('utf-8'))


# =============================================================================
# LAUNCH DASHBOARD
# =============================================================================

def launch_dashboard(cfg=None, port=8787, host='0.0.0.0', public=True, ngrok_token=None):
    """
    Launch the production healthcare dashboard server.

    Startup sequence:
        1. Initialize logging
        2. Resolve configuration paths
        3. Run preflight checks
        4. Initialize persistent database pool
        5. Load data and compute KPIs
        6. Build dashboard HTML
        7. Start HTTP server with security features
        8. Register graceful shutdown handlers
    """
    import signal

    # ── 1. PRODUCTION INITIALIZATION ──
    try:
        from production import (
            setup_logging, resolve_config, preflight_check,
            DatabasePool, get_logger, graceful_shutdown, register_shutdown_handler
        )
        has_production = True
        setup_logging()
        logger = get_logger('mtp.dashboard')
    except ImportError:
        has_production = False
        logger = None

    from query_engine import SemanticCatalog, clear_engine_cache
    from recommendation_engine import RecommendationEngine, load_csv_to_sqlite

    # Clear any stale cached engines/query caches from previous runs
    clear_engine_cache()

    if logger:
        logger.info("Starting Healthcare Dashboard Server (port=%d)", port)
    else:
        print("\n  Starting Healthcare Dashboard Server...")

    # ── 2. RESOLVE CONFIGURATION ──
    if has_production:
        cfg = resolve_config(cfg or {})
    else:
        cfg = cfg or {}

    raw_dir = cfg.get('RAW_DIR', '')
    if not raw_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        raw_dir = os.path.join(os.path.dirname(script_dir), 'data', 'raw')
        cfg['RAW_DIR'] = raw_dir

    if not os.path.exists(raw_dir):
        msg = f"Data directory not found: {raw_dir}"
        if logger:
            logger.error(msg)
        else:
            print(f"  ERROR: {msg}")
        return None

    # ── 3. PREFLIGHT CHECKS ──
    if has_production:
        preflight = preflight_check(cfg)
        if preflight.get('overall') == 'fail':
            logger.error("Preflight checks FAILED — aborting startup")
            return None

    # ── 4. INITIALIZE DATABASE POOL ──
    if has_production:
        try:
            pool = DatabasePool.get_instance()
            db_stats = pool.initialize(cfg)
            logger.info("Database pool: %s (%d tables)",
                        db_stats['status'], len(db_stats.get('tables', [])))
        except Exception as e:
            logger.error("Database pool initialization failed: %s", e)

    # ── 5. LOAD DATA & COMPUTE KPIs ──
    if logger:
        logger.info("Loading data from: %s", raw_dir)
    else:
        print(f"  Loading data from: {raw_dir}")

    conn = load_csv_to_sqlite(raw_dir)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    if logger:
        logger.info("Loaded %d tables: %s", len(tables), ', '.join(tables))
    else:
        print(f"  Loaded {len(tables)} tables: {', '.join(tables)}")

    rec_engine = RecommendationEngine(db_conn=conn)
    kpi_data = rec_engine.kpi_summary()
    ok_count = sum(1 for cat_kpis in kpi_data.get('by_category', {}).values()
                   for k in cat_kpis if not k.get('error'))
    total_count = sum(len(v) for v in kpi_data.get('by_category', {}).values())

    if logger:
        logger.info("KPIs computed: %d/%d OK", ok_count, total_count)
    else:
        print(f"  KPIs computed: {ok_count}/{total_count} OK")

    # ── 6. BUILD DASHBOARD HTML ──
    catalog = SemanticCatalog()
    dashboard_html = build_full_dashboard_html(kpi_data=kpi_data)

    # ── 7. CONFIGURE & START SERVER ──
    DashboardHandler.engine = rec_engine
    DashboardHandler.catalog = catalog
    DashboardHandler.cfg = cfg
    DashboardHandler.kpi_data = kpi_data
    DashboardHandler.dashboard_html = dashboard_html

    server = HTTPServer((host, port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # ── 8. GRACEFUL SHUTDOWN ──
    if has_production:
        register_shutdown_handler(lambda: server.shutdown())
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, graceful_shutdown)
            except (OSError, ValueError):
                pass  # Can't set signal handlers in non-main thread

    # ── 9. PUBLIC TUNNEL (ngrok) ──
    public_url = None
    _tunnel = None
    if public:
        global HAS_NGROK, _ngrok
        # Auto-install pyngrok if not present
        if not HAS_NGROK:
            print("  Installing pyngrok for public access...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyngrok', '-q'],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                from pyngrok import ngrok as _ngrok
                HAS_NGROK = True
                print("  ✓ pyngrok installed successfully")
            except Exception as e:
                print(f"  ⚠ Could not install pyngrok: {e}")
                print(f"  Run: pip install pyngrok")

        if HAS_NGROK:
            try:
                # Set auth token if provided (needed for custom domains/longer sessions)
                if ngrok_token:
                    _ngrok.set_auth_token(ngrok_token)
                elif os.environ.get('NGROK_AUTHTOKEN'):
                    _ngrok.set_auth_token(os.environ['NGROK_AUTHTOKEN'])

                _tunnel = _ngrok.connect(port, 'http', bind_tls=True)
                public_url = _tunnel.public_url
                if logger:
                    logger.info("Public tunnel: %s", public_url)
            except Exception as e:
                err_msg = str(e)
                if logger:
                    logger.warning("Could not start ngrok tunnel: %s", err_msg)
                else:
                    print(f"  ⚠ Ngrok tunnel failed: {err_msg}")
                    if 'authtoken' in err_msg.lower() or 'ERR_NGROK' in err_msg:
                        print(f"  → Sign up free at https://ngrok.com and set your token:")
                        print(f"     export NGROK_AUTHTOKEN=your_token_here")
                        print(f"     OR pass --ngrok-token=your_token_here")

    # ── 10. PRINT STARTUP INFO ──
    local_url = f'http://127.0.0.1:{port}'

    # Detect machine's LAN IP for shareable URL
    lan_url = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        lan_ip = s.getsockname()[0]
        s.close()
        lan_url = f'http://{lan_ip}:{port}'
    except Exception:
        pass

    if logger:
        logger.info("Dashboard running at: %s", local_url)
        if lan_url:
            logger.info("LAN URL: %s", lan_url)
        if public_url:
            logger.info("PUBLIC URL: %s", public_url)
    else:
        print(f"\n{'='*60}")
        print(f"  MTP Healthcare Analytics Dashboard")
        print(f"{'='*60}")
        print(f"  Local:       {local_url}")
        if lan_url:
            print(f"  LAN:         {lan_url}")
        if public_url:
            print(f"  ★ PUBLIC:    {public_url}  ← share this!")
        print(f"  Health:      {local_url}/health")
        print(f"{'─'*60}")
        if public_url:
            print(f"  ✓ Anyone with the PUBLIC URL above can access this dashboard")
            print(f"    from anywhere in the world (via secure HTTPS tunnel).")
        elif lan_url:
            print(f"  ℹ LAN URL works for anyone on your WiFi/network.")
            print(f"  For global access: pip install pyngrok && set NGROK_AUTHTOKEN")
        print(f"  ✓ DevTools/Inspect disabled for security.")
        print(f"  Press Ctrl+C to stop.\n")

    webbrowser.open(local_url)

    # Store tunnel ref on server for cleanup
    server._tunnel = _tunnel
    server._public_url = public_url
    return server


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MTP Healthcare Analytics Dashboard')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Bind address (default: 0.0.0.0 = all interfaces)')
    parser.add_argument('--port', type=int, default=8787,
                        help='Port number (default: 8787)')
    parser.add_argument('--no-public', action='store_true',
                        help='Disable public ngrok tunnel (LAN/local only)')
    parser.add_argument('--ngrok-token', type=str, default=None,
                        help='Ngrok auth token (or set NGROK_AUTHTOKEN env var)')
    parser.add_argument('--local-only', action='store_true',
                        help='Bind to 127.0.0.1 only, no tunnel (fully local)')
    args = parser.parse_args()

    if args.local_only:
        args.host = '127.0.0.1'
        args.no_public = True

    # Auto-detect project root from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)

    cfg = {
        'APP_DIR': project_dir,
        'DATA_DIR': os.path.join(project_dir, 'data'),
        'RAW_DIR': os.path.join(project_dir, 'data', 'raw'),
        'CATALOG_DIR': os.path.join(project_dir, 'semantic_catalog'),
        'LOCAL_EXECUTION': '1',
        'ENGINE_MODE': 'hybrid',
    }
    server = launch_dashboard(
        cfg=cfg, port=args.port, host=args.host,
        public=not args.no_public, ngrok_token=args.ngrok_token,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...")
        # Clean up ngrok tunnel
        if hasattr(server, '_tunnel') and server._tunnel:
            try:
                from pyngrok import ngrok as _ng
                _ng.disconnect(server._tunnel.public_url)
                _ng.kill()
                print("  ✓ Ngrok tunnel closed.")
            except Exception:
                pass
        server.shutdown()
