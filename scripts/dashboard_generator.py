import os
import json
import webbrowser
import tempfile
import html as html_lib
import math
import time
from typing import Dict, List, Any, Optional


THEME_CSS = """
:root {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --bg-card-hover: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #3b82f6;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --border: #334155;
    --shadow: 0 4px 24px rgba(0,0,0,0.3);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
}

.dashboard {
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px;
}

.header {
    text-align: center;
    margin-bottom: 32px;
    padding: 24px;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 16px;
    border: 1px solid var(--border);
}

.header h1 {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.header .subtitle {
    color: var(--text-secondary);
    font-size: 14px;
}

.grid {
    display: grid;
    gap: 20px;
    margin-bottom: 24px;
}

.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-4 { grid-template-columns: repeat(4, 1fr); }

@media (max-width: 900px) {
    .grid-3, .grid-4 { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 600px) {
    .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
}

.card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid var(--border);
    transition: transform 0.2s, box-shadow 0.2s;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow);
    border-color: var(--accent-blue);
}

.card-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 8px;
}

.card-value {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 4px;
}

.card-desc {
    font-size: 12px;
    color: var(--text-secondary);
}

.section {
    margin-bottom: 32px;
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
    color: var(--text-primary);
}

/* Alert styles */
.alert {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid var(--accent-red);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.alert-icon { font-size: 18px; }
.alert-text { font-size: 13px; color: var(--accent-red); }

/* Bar chart */
.bar-chart { padding: 16px 0; }

.bar-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    gap: 12px;
}

.bar-label {
    min-width: 120px;
    font-size: 13px;
    color: var(--text-secondary);
    text-align: right;
}

.bar-track {
    flex: 1;
    height: 28px;
    background: rgba(59, 130, 246, 0.1);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}

.bar-fill {
    height: 100%;
    border-radius: 6px;
    display: flex;
    align-items: center;
    padding: 0 10px;
    font-size: 12px;
    font-weight: 600;
    color: #fff;
    min-width: 40px;
    transition: width 1s ease;
}

.bar-value {
    min-width: 80px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    text-align: right;
}

/* Gauge */
.gauge-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 10px;
}

.gauge-svg { width: 140px; height: 80px; }

.gauge-label {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 6px;
    text-align: center;
}

/* Table */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.data-table th {
    background: var(--bg-primary);
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 2px solid var(--border);
    position: sticky;
    top: 0;
}

.data-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
}

.data-table tr:hover td {
    background: var(--bg-card-hover);
}

/* Recommendation card */
.rec-card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid var(--border);
    margin-bottom: 16px;
}

.rec-card:hover { border-color: var(--accent-blue); }

.rec-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.rec-title { font-size: 16px; font-weight: 600; }

.priority-badge {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
}

.priority-HIGH { background: rgba(239,68,68,0.15); color: var(--accent-red); }
.priority-MEDIUM { background: rgba(245,158,11,0.15); color: var(--accent-amber); }
.priority-LOW { background: rgba(16,185,129,0.15); color: var(--accent-green); }

.rec-metric {
    font-size: 13px;
    color: var(--accent-cyan);
    margin-bottom: 8px;
    font-family: 'SF Mono', Monaco, monospace;
}

.rec-insight {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 12px;
}

.rec-actions {
    list-style: none;
    padding: 0;
}

.rec-actions li {
    padding: 6px 0 6px 20px;
    font-size: 13px;
    color: var(--text-secondary);
    position: relative;
}

.rec-actions li::before {
    content: '\\2192';
    position: absolute;
    left: 0;
    color: var(--accent-blue);
}

.rec-impact {
    margin-top: 10px;
    padding: 8px 12px;
    background: rgba(16,185,129,0.08);
    border-radius: 6px;
    font-size: 12px;
    color: var(--accent-green);
}

/* Action rank */
.action-rank {
    display: flex;
    align-items: center;
    gap: 16px;
    background: var(--bg-card);
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid var(--border);
    margin-bottom: 12px;
}

.action-rank:hover { border-color: var(--accent-blue); }

.rank-number {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 16px;
    flex-shrink: 0;
}

.rank-1 { background: linear-gradient(135deg, #fbbf24, #f59e0b); color: #0f172a; }
.rank-2 { background: linear-gradient(135deg, #94a3b8, #64748b); color: #0f172a; }
.rank-3 { background: linear-gradient(135deg, #b45309, #92400e); color: #fef3c7; }
.rank-4, .rank-5 { background: var(--bg-card-hover); color: var(--text-secondary); }

.action-details { flex: 1; }
.action-title { font-size: 15px; font-weight: 600; }
.action-step { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }

/* Footer */
.footer {
    text-align: center;
    padding: 20px;
    color: var(--text-muted);
    font-size: 12px;
    border-top: 1px solid var(--border);
    margin-top: 32px;
}

/* Search/filter bar for tables */
.filter-bar {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
}

.filter-input {
    flex: 1;
    padding: 8px 14px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 13px;
    outline: none;
}

.filter-input:focus { border-color: var(--accent-blue); }

.filter-count {
    padding: 8px 14px;
    background: var(--bg-primary);
    border-radius: 8px;
    color: var(--text-muted);
    font-size: 13px;
    display: flex;
    align-items: center;
}
"""


def _esc(text):
    return html_lib.escape(str(text)) if text is not None else "N/A"


def _format_value(val, unit=""):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if abs(val) >= 1_000_000:
            return "${:,.1f}M".format(val / 1_000_000)
        elif unit == "$":
            return "${:,.2f}".format(val)
        else:
            return "{:,.2f}{}".format(val, unit)
    if isinstance(val, int):
        return "{:,}{}".format(val, unit)
    return str(val)


def _color_for_index(i):
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4",
              "#ef4444", "#ec4899", "#14b8a6", "#f97316", "#6366f1"]
    return colors[i % len(colors)]


def _gauge_svg(value, max_val, label, color="#3b82f6"):
    pct = min(max(value / max_val, 0), 1) if max_val > 0 else 0
    angle = 180 * pct
    end_x = 70 + 55 * math.cos(math.radians(180 - angle))
    end_y = 70 - 55 * math.sin(math.radians(180 - angle))
    large_arc = 1 if angle > 180 else 0

    return f'''<svg class="gauge-svg" viewBox="0 0 140 80">
        <path d="M 15 70 A 55 55 0 0 1 125 70" fill="none" stroke="#334155" stroke-width="10" stroke-linecap="round"/>
        <path d="M 15 70 A 55 55 0 {large_arc} 1 {end_x:.1f} {end_y:.1f}" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"/>
        <text x="70" y="65" text-anchor="middle" fill="{color}" font-size="18" font-weight="700">{_format_value(value)}</text>
    </svg>'''


def _bar_chart_html(data, label_key, value_key, max_val=None, color=None):
    if not data:
        return '<p style="color:var(--text-muted)">No data available</p>'

    vals = [float(r.get(value_key, 0) or 0) for r in data]
    if max_val is None:
        max_val = max(vals) if vals else 1

    rows_html = []
    for i, row in enumerate(data):
        label = _esc(row.get(label_key, f"Item {i+1}"))
        val = float(row.get(value_key, 0) or 0)
        pct = (val / max_val * 100) if max_val > 0 else 0
        c = color or _color_for_index(i)
        rows_html.append(f'''
        <div class="bar-row">
            <span class="bar-label">{label}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width:{pct:.1f}%;background:{c};">{_format_value(val)}</div>
            </div>
        </div>''')

    return '<div class="bar-chart">' + ''.join(rows_html) + '</div>'


def _table_html(data, columns=None, table_id="data-table"):
    if not data:
        return '<p style="color:var(--text-muted)">No data available</p>'

    if columns is None:
        if isinstance(data[0], dict):
            columns = list(data[0].keys())
        else:
            columns = [f"Col {i}" for i in range(len(data[0]))]

    header = ''.join(
        f'<th onclick="sortTable(\'{table_id}\', {i})" style="cursor:pointer;">'
        f'{_esc(c)} <span class="sort-icon">&#8597;</span></th>'
        for i, c in enumerate(columns)
    )

    rows = []
    for row in data:
        if isinstance(row, dict):
            cells = ''.join(f'<td>{_esc(row.get(c, ""))}</td>' for c in columns)
        else:
            cells = ''.join(f'<td>{_esc(v)}</td>' for v in row)
        rows.append(f'<tr>{cells}</tr>')

    col_toggles = ''.join(
        f'<label style="margin-right:10px;font-size:12px;color:var(--text-secondary);cursor:pointer;">'
        f'<input type="checkbox" checked onchange="toggleColumn(\'{table_id}\', {i}, this.checked)" '
        f'style="margin-right:4px;accent-color:var(--accent-blue);"> {_esc(c)}</label>'
        for i, c in enumerate(columns)
    )

    return f'''
    <div class="filter-bar">
        <input class="filter-input" type="text" placeholder="Search across all columns..." oninput="filterTable(this, '{table_id}')">
        <span class="filter-count" id="{table_id}-count">{len(data)} rows</span>
    </div>
    <div style="margin-bottom:8px;padding:6px 10px;background:var(--bg-primary);border-radius:6px;overflow-x:auto;white-space:nowrap;">
        <span style="font-size:11px;color:var(--text-muted);margin-right:8px;">Columns:</span>{col_toggles}
    </div>
    <div style="max-height:500px;overflow:auto;border-radius:8px;border:1px solid var(--border);">
        <table class="data-table" id="{table_id}">
            <thead><tr>{header}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>'''


def _rec_card_html(rec):
    priority = rec.get('priority', 'LOW')
    actions_html = ''.join(
        f'<li>{_esc(a)}</li>' for a in rec.get('actions', [])[:4]
    )
    impact_html = ''
    if rec.get('expected_impact'):
        impact_html = f'<div class="rec-impact">{_esc(rec["expected_impact"])}</div>'

    data_html = ''
    if rec.get('data') and isinstance(rec['data'], list) and len(rec['data']) > 0:
        data_html = _table_html(rec['data'][:5], table_id=f"rec-{abs(hash(rec.get('title', '')))}")

    return f'''
    <div class="rec-card">
        <div class="rec-header">
            <span class="rec-title">{_esc(rec.get('title', 'Recommendation'))}</span>
            <span class="priority-badge priority-{priority}">{priority}</span>
        </div>
        <div class="rec-metric">{_esc(rec.get('metric', ''))}</div>
        <div class="rec-insight">{_esc(rec.get('insight', ''))}</div>
        <ul class="rec-actions">{actions_html}</ul>
        {impact_html}
        {data_html}
    </div>'''


def _page_html(title, subtitle, body_html):
    timestamp = time.strftime("%B %d, %Y %I:%M %p")
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_esc(title)}</title>
    <style>{THEME_CSS}</style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>{_esc(title)}</h1>
            <div class="subtitle">{_esc(subtitle)} &mdash; Generated {timestamp}</div>
        </div>
        {body_html}
        <div class="footer">
            MTP Healthcare Analytics &bull; Built from Scratch &bull; Zero External Dependencies &bull; {timestamp}
        </div>
    </div>
    <script>
    /* ---- Filter ---- */
    function filterTable(input, tableId) {{
        const filter = input.value.toLowerCase();
        const table = document.getElementById(tableId);
        const rows = table.querySelectorAll('tbody tr');
        let visible = 0;
        rows.forEach(row => {{
            const text = row.textContent.toLowerCase();
            const show = text.includes(filter);
            row.style.display = show ? '' : 'none';
            if (show) visible++;
        }});
        const counter = document.getElementById(tableId + '-count');
        if (counter) counter.textContent = visible + ' / ' + rows.length + ' rows';
    }}

    /* ---- Sort ---- */
    const sortState = {{}};
    function sortTable(tableId, colIdx) {{
        const table = document.getElementById(tableId);
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const key = tableId + '_' + colIdx;
        sortState[key] = !sortState[key]; // toggle asc/desc
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

        // Update sort icons
        table.querySelectorAll('th .sort-icon').forEach((icon, i) => {{
            icon.innerHTML = i === colIdx ? (asc ? '&#9650;' : '&#9660;') : '&#8597;';
        }});
    }}

    /* ---- Column Toggle ---- */
    function toggleColumn(tableId, colIdx, show) {{
        const table = document.getElementById(tableId);
        table.querySelectorAll('tr').forEach(row => {{
            const cell = row.cells[colIdx];
            if (cell) cell.style.display = show ? '' : 'none';
        }});
    }}

    /* ---- Bar animation ---- */
    document.addEventListener('DOMContentLoaded', () => {{
        document.querySelectorAll('.bar-fill').forEach(bar => {{
            const w = bar.style.width;
            bar.style.width = '0%';
            requestAnimationFrame(() => {{ bar.style.width = w; }});
        }});
    }});
    </script>
</body>
</html>'''


def open_dashboard(html_content, filename="dashboard.html", dashboard_dir=None):
    if dashboard_dir is None:
        dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dashboard')
    os.makedirs(dashboard_dir, exist_ok=True)
    filepath = os.path.join(dashboard_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    webbrowser.open('file://' + os.path.abspath(filepath))
    return filepath


def kpi_dashboard(kpi_summary: Dict) -> str:
    alerts = kpi_summary.get('alerts', [])
    by_category = kpi_summary.get('by_category', {})

    alerts_html = ''
    if alerts:
        alert_items = ''.join(
            f'<div class="alert"><span class="alert-icon">&#9888;</span>'
            f'<span class="alert-text">{_esc(a.get("alert", ""))}</span></div>'
            for a in alerts
        )
        alerts_html = f'''
        <div class="section">
            <div class="section-title">Active Alerts ({len(alerts)})</div>
            {alert_items}
        </div>'''

    categories_html = ''
    category_colors = {
        'revenue': '#3b82f6', 'retention': '#10b981',
        'acquisition': '#8b5cf6', 'operations': '#f59e0b',
    }
    category_icons = {
        'revenue': '', 'retention': '',
        'acquisition': '', 'operations': '',
    }

    for cat, kpis in by_category.items():
        color = category_colors.get(cat, '#64748b')
        icon = category_icons.get(cat, '')

        cards = []
        for kpi in kpis:
            val = kpi.get('value', 'N/A')
            unit = kpi.get('unit', '')
            alert_mark = ' &#9888;' if kpi.get('alert') else ''

            if isinstance(val, (int, float)):
                display_val = _format_value(val, unit)
            elif isinstance(val, list):
                display_val = f"{len(val)} rows"
            else:
                display_val = str(val)

            err = kpi.get('error')
            if err:
                display_val = "Error"
                alert_mark = ' &#10060;'

            cards.append(f'''
            <div class="card">
                <div class="card-title">{_esc(kpi.get('kpi', 'Unknown'))}{alert_mark}</div>
                <div class="card-value" style="color:{color}">{display_val}</div>
                <div class="card-desc">{_esc(kpi.get('description', '')[:80])}</div>
            </div>''')

        grid_class = 'grid-3' if len(cards) >= 3 else 'grid-2'
        categories_html += f'''
        <div class="section">
            <div class="section-title">{icon} {cat.upper()} KPIs</div>
            <div class="grid {grid_class}">{''.join(cards)}</div>
        </div>'''

    body = alerts_html + categories_html
    page = _page_html("Healthcare KPI Dashboard",
                       f"{kpi_summary.get('total_kpis', 0)} KPIs tracked across {len(by_category)} categories",
                       body)
    return open_dashboard(page, "kpi_dashboard.html")


def revenue_dashboard(recs: List[Dict]) -> str:
    body = ''
    for rec in recs:
        body += _rec_card_html(rec)
    if not recs:
        body = '<div class="card"><div class="card-value" style="color:var(--accent-green)">All Clear</div><div class="card-desc">No revenue issues detected. All metrics within healthy ranges.</div></div>'

    page = _page_html("Revenue Optimization Dashboard",
                       f"{len(recs)} revenue improvement opportunities identified", body)
    return open_dashboard(page, "revenue_dashboard.html")


def retention_dashboard(recs: List[Dict]) -> str:
    body = ''
    for rec in recs:
        body += _rec_card_html(rec)
    if not recs:
        body = '<div class="card"><div class="card-value" style="color:var(--accent-green)">Healthy</div><div class="card-desc">Member retention metrics look good. No immediate concerns.</div></div>'

    page = _page_html("Member Retention Dashboard",
                       f"{len(recs)} retention improvement opportunities identified", body)
    return open_dashboard(page, "retention_dashboard.html")


def acquisition_dashboard(recs: List[Dict]) -> str:
    body = ''
    for rec in recs:
        body += _rec_card_html(rec)
    if not recs:
        body = '<div class="card"><div class="card-value" style="color:var(--accent-green)">On Track</div><div class="card-desc">No immediate acquisition gaps identified.</div></div>'

    page = _page_html("Customer Acquisition Dashboard",
                       f"{len(recs)} growth opportunities identified", body)
    return open_dashboard(page, "acquisition_dashboard.html")


def operations_dashboard(recs: List[Dict]) -> str:
    body = ''
    for rec in recs:
        body += _rec_card_html(rec)
    if not recs:
        body = '<div class="card"><div class="card-value" style="color:var(--accent-green)">Efficient</div><div class="card-desc">Operational metrics within acceptable ranges.</div></div>'

    page = _page_html("Operational Efficiency Dashboard",
                       f"{len(recs)} efficiency improvement opportunities", body)
    return open_dashboard(page, "operations_dashboard.html")


def actions_dashboard(actions: List[Dict]) -> str:
    body = ''
    for i, action in enumerate(actions, 1):
        priority = action.get('priority', 'LOW')
        body += f'''
        <div class="action-rank">
            <div class="rank-number rank-{min(i, 5)}">{i}</div>
            <div class="action-details">
                <div class="action-title">{_esc(action.get('action', 'Action'))}</div>
                <div class="action-step">First Step: {_esc(action.get('first_step', 'N/A'))}</div>
                <div style="margin-top:4px"><span class="priority-badge priority-{priority}">{priority}</span>
                    <span style="font-size:12px;color:var(--text-muted);margin-left:8px;">{_esc(action.get('metric', ''))}</span>
                </div>
            </div>
        </div>'''
    if not actions:
        body = '<div class="card"><div class="card-desc">No prioritized actions at this time.</div></div>'

    page = _page_html("Top Prioritized Actions",
                       f"{len(actions)} high-impact actions ranked by priority", body)
    return open_dashboard(page, "actions_dashboard.html")


def query_results_dashboard(question: str, sql: str, results: Any,
                            intent: str = "", mode: str = "") -> str:
    body = ''

    body += f'''
    <div class="card" style="margin-bottom:20px;">
        <div class="card-title">Query</div>
        <div style="font-size:16px;font-weight:600;margin-bottom:8px;">{_esc(question)}</div>
        <div style="display:flex;gap:8px;margin-bottom:8px;">
            <span class="priority-badge priority-MEDIUM">Intent: {_esc(intent)}</span>
            <span class="priority-badge priority-LOW">Mode: {_esc(mode)}</span>
        </div>
        <div style="font-family:'SF Mono',Monaco,monospace;font-size:12px;color:var(--accent-cyan);
                    background:var(--bg-primary);padding:10px;border-radius:6px;overflow-x:auto;white-space:pre-wrap;">
            {_esc(sql or 'No SQL generated')}
        </div>
    </div>'''

    if isinstance(results, list) and results:
        if isinstance(results[0], dict):
            columns = list(results[0].keys())

            numeric_cols = []
            cat_cols = []
            for col in columns:
                vals = [r.get(col) for r in results if r.get(col) is not None and str(r[col]).strip()]
                if not vals:
                    continue
                num_vals = []
                for v in vals:
                    try:
                        num_vals.append(float(str(v).replace(',', '')))
                    except (ValueError, TypeError):
                        break
                if len(num_vals) == len(vals) and num_vals:
                    numeric_cols.append((col, num_vals))
                else:
                    cat_cols.append(col)

            if cat_cols and numeric_cols and len(results) <= 50:
                label_col = cat_cols[0]
                val_col, val_data = numeric_cols[0]
                chart_data = []
                for row in results[:20]:
                    try:
                        chart_data.append({
                            'label': str(row.get(label_col, '')),
                            'value': float(str(row.get(val_col, 0)).replace(',', '')),
                        })
                    except (ValueError, TypeError):
                        pass
                if chart_data:
                    body += f'''
                    <div class="section">
                        <div class="section-title">Auto-Chart: {_esc(val_col)} by {_esc(label_col)}</div>
                        {_bar_chart_html(chart_data, 'label', 'value')}
                    </div>'''

            if numeric_cols:
                stats_cards = ''
                for col, vals in numeric_cols[:6]:
                    avg = sum(vals) / len(vals)
                    mn = min(vals)
                    mx = max(vals)
                    total = sum(vals)
                    stats_cards += f'''
                    <div class="card">
                        <div class="card-title">{_esc(col)}</div>
                        <div class="card-value" style="font-size:22px;color:var(--accent-cyan);">{_format_value(avg)}</div>
                        <div class="card-desc">Sum: {_format_value(total)} | Min: {_format_value(mn)} | Max: {_format_value(mx)} | Count: {len(vals)}</div>
                    </div>'''
                grid_class = 'grid-3' if len(numeric_cols) >= 3 else 'grid-2'
                body += f'''
                <div class="section">
                    <div class="section-title">Column Statistics</div>
                    <div class="grid {grid_class}">{stats_cards}</div>
                </div>'''

            body += f'''
            <div class="section">
                <div class="section-title">Data ({len(results)} rows) &mdash; click headers to sort, toggle columns above</div>
                {_table_html(results, table_id="query-results")}
            </div>'''

    elif isinstance(results, str):
        body += f'''
        <div class="card">
            <div class="card-title">Result</div>
            <div style="font-size:14px;white-space:pre-wrap;">{_esc(results)}</div>
        </div>'''
    else:
        body += '<div class="card"><div class="card-desc">No results returned.</div></div>'

    page = _page_html("Query Results", question, body)
    return open_dashboard(page, "query_results.html")


def metrics_dashboard(cache_stats: Dict, perf_stats: Dict) -> str:
    body = ''

    body += f'''
    <div class="section">
        <div class="section-title">Query Cache</div>
        <div class="grid grid-4">
            <div class="card">
                <div class="card-title">Cache Size</div>
                <div class="card-value" style="color:var(--accent-blue)">{cache_stats.get('size', 0)}</div>
                <div class="card-desc">of {cache_stats.get('capacity', 0)} capacity</div>
            </div>
            <div class="card">
                <div class="card-title">Hit Rate</div>
                <div class="card-value" style="color:var(--accent-green)">{_esc(cache_stats.get('hit_rate', '0%'))}</div>
                <div class="card-desc">Cache effectiveness</div>
            </div>
            <div class="card">
                <div class="card-title">Hits</div>
                <div class="card-value" style="color:var(--accent-cyan)">{cache_stats.get('hits', 0)}</div>
                <div class="card-desc">Queries served from cache</div>
            </div>
            <div class="card">
                <div class="card-title">Misses</div>
                <div class="card-value" style="color:var(--accent-amber)">{cache_stats.get('misses', 0)}</div>
                <div class="card-desc">Queries requiring computation</div>
            </div>
        </div>
    </div>'''

    latency = perf_stats.get('latency', {})
    body += f'''
    <div class="section">
        <div class="section-title">&#9889; Performance Metrics</div>
        <div class="grid grid-4">
            <div class="card">
                <div class="card-title">Total Queries</div>
                <div class="card-value" style="color:var(--accent-blue)">{perf_stats.get('total_queries', 0)}</div>
                <div class="card-desc">QPS: {_esc(perf_stats.get('queries_per_second', '0'))}</div>
            </div>
            <div class="card">
                <div class="card-title">Avg Latency</div>
                <div class="card-value" style="color:var(--accent-cyan)">{_esc(latency.get('mean', '0'))}ms</div>
                <div class="card-desc">p50: {_esc(latency.get('p50', '0'))}ms</div>
            </div>
            <div class="card">
                <div class="card-title">p95 Latency</div>
                <div class="card-value" style="color:var(--accent-amber)">{_esc(latency.get('p95', '0'))}ms</div>
                <div class="card-desc">p99: {_esc(latency.get('p99', '0'))}ms</div>
            </div>
            <div class="card">
                <div class="card-title">Errors</div>
                <div class="card-value" style="color:var(--accent-red)">{perf_stats.get('error_count', 0)}</div>
                <div class="card-desc">Uptime: {_esc(perf_stats.get('uptime_seconds', '0'))}s</div>
            </div>
        </div>
    </div>'''

    page = _page_html("System Performance Dashboard",
                       "Real-time system health and performance metrics", body)
    return open_dashboard(page, "metrics_dashboard.html")


if __name__ == '__main__':
    print("Dashboard Generator - Self Test")
    print("=" * 50)

    sample_kpi = {
        'total_kpis': 5,
        'alerts': [{'alert': 'Denial Rate 15.2% exceeds threshold of 10%', 'kpi': 'Denial Rate'}],
        'by_category': {
            'revenue': [
                {'kpi': 'Avg Revenue/Member', 'value': 2586.92, 'unit': '$', 'description': 'Average revenue per member'},
                {'kpi': 'Denial Rate', 'value': 15.2, 'unit': '%', 'description': 'Claim denial rate', 'alert': 'Too high'},
                {'kpi': 'Processing Time', 'value': 25.9, 'unit': ' days', 'description': 'Average days to process a claim'},
            ],
            'retention': [
                {'kpi': 'Readmission Rate', 'value': 8.3, 'unit': '%', 'description': 'Patient readmission within 30 days'},
                {'kpi': 'Utilization Rate', 'value': 77.6, 'unit': '%', 'description': 'Member service utilization rate'},
                {'kpi': 'Avg Encounters/Member', 'value': 3.2, 'unit': '', 'description': 'How frequently members use services'},
            ],
        }
    }
    path = kpi_dashboard(sample_kpi)
    print(f"  [OK] KPI Dashboard: {path}")

    sample_actions = [
        {'action': 'Reduce Claim Denials', 'priority': 'HIGH', 'first_step': 'Audit top 5 denial reasons', 'metric': 'Denial Rate: 15.2%'},
        {'action': 'Accelerate Processing', 'priority': 'HIGH', 'first_step': 'Auto-adjudicate clean claims', 'metric': 'Avg: 26 days'},
        {'action': 'Re-engage Inactive Members', 'priority': 'MEDIUM', 'first_step': 'Send wellness reminders', 'metric': '22% inactive'},
    ]
    path = actions_dashboard(sample_actions)
    print(f"  [OK] Actions Dashboard: {path}")

    sample_results = [
        {'KP_REGION': 'NW', 'total_claims': 450, 'avg_paid': 2800.50},
        {'KP_REGION': 'SO_CAL', 'total_claims': 380, 'avg_paid': 2650.25},
        {'KP_REGION': 'NO_CAL', 'total_claims': 410, 'avg_paid': 2720.00},
        {'KP_REGION': 'HI', 'total_claims': 290, 'avg_paid': 2580.75},
        {'KP_REGION': 'MAS', 'total_claims': 320, 'avg_paid': 2510.00},
    ]
    path = query_results_dashboard(
        "claims by region",
        "SELECT KP_REGION, COUNT(*) as total_claims, AVG(PAID_AMOUNT) as avg_paid FROM claims GROUP BY KP_REGION",
        sample_results, "aggregate", "hybrid")
    print(f"  [OK] Query Results: {path}")

    print("\nAll dashboards generated!")
