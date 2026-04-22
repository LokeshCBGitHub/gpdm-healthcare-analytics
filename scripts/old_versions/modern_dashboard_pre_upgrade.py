"""Modern healthcare dashboard. Flask app serving HTML/CSS/JS single-page application with intelligent NL query pipeline."""

import os
import sys
import ssl
import math
import logging
import numpy as np
from typing import Any

logger = logging.getLogger('gpdm.dashboard')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# SSL certificate paths
CERT_DIR = os.path.join(PROJECT_DIR, 'certs')
SSL_CERT = os.path.join(CERT_DIR, 'cert.pem')
SSL_KEY = os.path.join(CERT_DIR, 'key.pem')


def build_dashboard_html() -> str:
    """Build dashboard HTML with responsive design and modern styling."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KP Healthcare — GPDM Analytics</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <style>
        :root {
            --kp-blue: #004B87;
            --kp-blue-light: #0066CC;
            --kp-blue-pale: #E8F4FD;
            --kp-teal: #00A3B5;
            --kp-green: #48A23F;
            --kp-orange: #FF6B35;
            --kp-red: #D32F2F;
            --kp-gray-100: #F7F8FA;
            --kp-gray-200: #E8EBF0;
            --kp-gray-300: #C5CAD4;
            --kp-gray-500: #6B7280;
            --kp-gray-700: #374151;
            --kp-gray-900: #111827;
            --sidebar-width: 220px;
            --header-height: 52px;
            --radius: 8px;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.06);
            --shadow-md: 0 4px 6px rgba(0,0,0,0.07);
            --shadow-lg: 0 10px 25px rgba(0,0,0,0.1);
            --transition: all 0.2s ease;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background: var(--kp-gray-100); color: var(--kp-gray-900); line-height: 1.5; }

        /* ── Sidebar ── */
        .sidebar { position:fixed;left:0;top:0;bottom:0;width:var(--sidebar-width);background:var(--kp-blue);color:white;z-index:100;display:flex;flex-direction:column; }
        .sidebar-brand { padding:14px 18px;border-bottom:1px solid rgba(255,255,255,0.15); }
        .sidebar-brand h1 { font-size:15px;font-weight:600; }
        .sidebar-brand span { font-size:10px;opacity:0.7;display:block;margin-top:2px; }
        .sidebar-nav { flex:1;padding:8px 0;overflow-y:auto; }
        .nav-item { display:flex;align-items:center;padding:9px 18px;color:rgba(255,255,255,0.75);text-decoration:none;font-size:13px;cursor:pointer;transition:var(--transition);gap:10px; }
        .nav-item:hover { background:rgba(255,255,255,0.1);color:white; }
        .nav-item.active { background:rgba(255,255,255,0.15);color:white;font-weight:500;border-left:3px solid white;padding-left:15px; }
        .nav-icon { width:16px;text-align:center;flex-shrink:0; }
        .nav-section { padding:6px 18px 4px;font-size:10px;text-transform:uppercase;letter-spacing:1px;color:rgba(255,255,255,0.4);margin-top:8px; }
        .sidebar-footer { padding:10px 18px;border-top:1px solid rgba(255,255,255,0.15);font-size:10px;opacity:0.6; }
        .sidebar-footer .status-dot { display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--kp-green);margin-right:5px; }

        /* ── Main content (full width minus sidebar) ── */
        .main { margin-left:var(--sidebar-width);min-height:100vh; }
        .header { height:var(--header-height);background:white;border-bottom:1px solid var(--kp-gray-200);display:flex;align-items:center;padding:0 20px;position:sticky;top:0;z-index:50; }
        .header-title { font-size:15px;font-weight:600;color:var(--kp-gray-900); }
        .header-right { margin-left:auto;display:flex;align-items:center;gap:12px; }
        .badge { font-size:10px;padding:3px 8px;border-radius:12px;font-weight:500; }
        .badge-blue { background:var(--kp-blue-pale);color:var(--kp-blue); }
        .badge-green { background:#E8F5E9;color:var(--kp-green); }
        .badge-orange { background:#FFF3E0;color:var(--kp-orange); }
        .content { padding:20px; }

        .page { display:none; }
        .page.active { display:block; }

        /* ── Chat page — conversation thread layout ── */
        .chat-container { max-width:1200px;margin:0 auto;display:flex;flex-direction:column;height:calc(100vh - var(--header-height) - 40px); }
        .chat-thread { flex:1;overflow-y:auto;padding-bottom:16px;scroll-behavior:smooth; }
        .chat-thread::-webkit-scrollbar { width:6px; }
        .chat-thread::-webkit-scrollbar-thumb { background:var(--kp-gray-300);border-radius:3px; }
        .chat-empty { display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;
            color:var(--kp-gray-500);text-align:center;padding:40px; }
        .chat-empty h2 { font-size:22px;font-weight:600;color:var(--kp-gray-700);margin-bottom:8px; }
        .chat-empty p { font-size:13px;margin-bottom:20px; }
        .chat-input-area { background:white;border-radius:var(--radius);box-shadow:var(--shadow-md);padding:14px;
            flex-shrink:0;border-top:1px solid var(--kp-gray-200); }
        .chat-input-row { display:flex;gap:10px; }
        .chat-input { flex:1;padding:10px 14px;border:2px solid var(--kp-gray-200);border-radius:var(--radius);font-size:14px;outline:none;transition:var(--transition); }
        .chat-input:focus { border-color:var(--kp-blue-light); }
        .chat-btn { padding:10px 20px;background:var(--kp-blue);color:white;border:none;border-radius:var(--radius);font-size:13px;font-weight:500;cursor:pointer;transition:var(--transition);white-space:nowrap; }
        .chat-btn:hover { background:var(--kp-blue-light); }
        .chat-btn:disabled { opacity:0.5;cursor:not-allowed; }
        .chat-btn-followup { background:var(--kp-teal); }
        .chat-btn-followup:hover { background:#008a99; }
        .suggestions { display:flex;gap:6px;margin-top:10px;flex-wrap:wrap; }
        .suggestion-chip { padding:5px 12px;background:var(--kp-blue-pale);color:var(--kp-blue);border:none;border-radius:16px;font-size:12px;cursor:pointer;transition:var(--transition); }

        /* ── Question bubble ── */
        .q-bubble { display:flex;justify-content:flex-end;margin-bottom:4px;animation:fadeIn 0.2s ease; }
        .q-bubble-inner { max-width:75%;padding:10px 16px;background:var(--kp-blue);color:white;border-radius:16px 16px 4px 16px;font-size:13px;line-height:1.5; }
        @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        .suggestion-chip:hover { background:var(--kp-blue);color:white; }

        /* ── Result cards — more context, wider ── */
        .result-card { background:white;border-radius:var(--radius);box-shadow:var(--shadow-sm);margin-bottom:16px;overflow:hidden;border:1px solid var(--kp-gray-200); }
        .result-header { padding:12px 16px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--kp-gray-200);background:var(--kp-gray-100); }
        .result-header .query-text { font-size:13px;color:var(--kp-gray-700);font-weight:500; }
        .result-body { padding:16px; }

        /* ── SQL Editor ── */
        .sql-editor-area { border-top:1px solid var(--kp-gray-200);background:#FAFBFC; }
        .sql-toolbar { display:flex;align-items:center;gap:8px;padding:8px 14px;border-bottom:1px solid var(--kp-gray-200); }
        .sql-toolbar-btn { padding:4px 12px;font-size:11px;border-radius:4px;border:1px solid var(--kp-gray-300);background:white;cursor:pointer;transition:var(--transition); }
        .sql-toolbar-btn:hover { background:var(--kp-gray-100); }
        .sql-toolbar-btn.run { background:var(--kp-green);color:white;border-color:var(--kp-green); }
        .sql-toolbar-btn.run:hover { background:#3d8f36; }
        .sql-editor { width:100%;min-height:60px;max-height:200px;padding:10px 14px;font-family:'SF Mono',Monaco,Consolas,monospace;font-size:12px;border:none;outline:none;resize:vertical;background:transparent;color:var(--kp-gray-900);line-height:1.6; }
        .sql-readonly-badge { font-size:9px;background:#E8F5E9;color:var(--kp-green);padding:2px 6px;border-radius:3px;font-weight:600; }

        /* ── KPI cards ── */
        .kpi-grid { display:grid;grid-template-columns:repeat(auto-fill, minmax(220px, 1fr));gap:14px;margin-bottom:20px; }
        .kpi-card { background:white;border-radius:var(--radius);padding:14px 18px;box-shadow:var(--shadow-sm);border-left:4px solid var(--kp-blue);transition:var(--transition);cursor:pointer; }
        .kpi-card:hover { box-shadow:var(--shadow-md);transform:translateY(-1px); }
        .kpi-label { font-size:11px;color:var(--kp-gray-500);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px; }
        .kpi-value { font-size:26px;font-weight:700;color:var(--kp-gray-900);line-height:1.2; }

        /* ── Charts ── */
        .chart-container { background:white;border-radius:var(--radius);padding:16px;box-shadow:var(--shadow-sm);margin-bottom:14px; }
        .chart-title { font-size:13px;font-weight:600;color:var(--kp-gray-700);margin-bottom:12px; }
        .chart-canvas { width:100%;max-height:300px; }

        /* ── Data table with sortable columns ── */
        .data-table { width:100%;border-collapse:collapse;font-size:12px; }
        .data-table th { text-align:left;padding:8px 10px;background:var(--kp-gray-100);color:var(--kp-gray-700);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:0.3px;border-bottom:2px solid var(--kp-gray-200);cursor:pointer;user-select:none;white-space:nowrap; }
        .data-table th:hover { background:var(--kp-gray-200); }
        .data-table th .sort-arrow { font-size:10px;margin-left:3px;opacity:0.4; }
        .data-table th.sorted .sort-arrow { opacity:1;color:var(--kp-blue); }
        .data-table td { padding:7px 10px;border-bottom:1px solid var(--kp-gray-200);color:var(--kp-gray-700); }
        .data-table tr:hover td { background:var(--kp-blue-pale); }
        .data-table .numeric { text-align:right;font-variant-numeric:tabular-nums; }

        /* ── Loading ── */
        .loading { display:flex;align-items:center;justify-content:center;padding:30px;color:var(--kp-gray-500); }
        .spinner { width:20px;height:20px;border:3px solid var(--kp-gray-200);border-top-color:var(--kp-blue);border-radius:50%;animation:spin 0.8s linear infinite;margin-right:10px; }
        @keyframes spin { to { transform:rotate(360deg); } }

        /* ── Quality measures ── */
        .measure-card { background:white;border-radius:var(--radius);padding:16px;box-shadow:var(--shadow-sm);margin-bottom:10px; }
        .measure-name { font-size:14px;font-weight:600;color:var(--kp-gray-900); }
        .measure-desc { font-size:12px;color:var(--kp-gray-500);margin-top:3px; }
        .measure-bar { height:7px;background:var(--kp-gray-200);border-radius:4px;margin-top:10px;overflow:hidden; }
        .measure-fill { height:100%;border-radius:4px;transition:width 0.8s ease; }
        .measure-stats { display:flex;justify-content:space-between;margin-top:6px;font-size:11px;color:var(--kp-gray-500); }

        /* ── Explorer ── */
        .explorer-panel { background:white;border-radius:var(--radius);padding:16px;box-shadow:var(--shadow-sm);margin-bottom:14px; }
        .explorer-toolbar { display:flex;gap:10px;align-items:center;margin-bottom:14px;flex-wrap:wrap; }
        .explorer-select { padding:8px 12px;border:1px solid var(--kp-gray-300);border-radius:var(--radius);font-size:13px; }

        /* ── Custom Dashboard ── */
        .custom-dash-bar { display:flex;gap:10px;align-items:center;margin-bottom:16px;flex-wrap:wrap; }
        .custom-dash-tab { padding:6px 16px;background:white;border:1px solid var(--kp-gray-300);border-radius:var(--radius);font-size:12px;cursor:pointer;transition:var(--transition); }
        .custom-dash-tab:hover { border-color:var(--kp-blue); }
        .custom-dash-tab.active { background:var(--kp-blue);color:white;border-color:var(--kp-blue); }
        .widget-grid { display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:14px; }
        .widget-card { background:white;border-radius:var(--radius);box-shadow:var(--shadow-sm);overflow:hidden;border:1px solid var(--kp-gray-200); }
        .widget-header { padding:10px 14px;background:var(--kp-gray-100);border-bottom:1px solid var(--kp-gray-200);display:flex;align-items:center;gap:8px; }
        .widget-header input { flex:1;border:none;background:transparent;font-size:13px;font-weight:600;outline:none;color:var(--kp-gray-700); }
        .widget-body { padding:14px; }
        .widget-remove { cursor:pointer;color:var(--kp-gray-500);font-size:14px; }
        .widget-remove:hover { color:var(--kp-red); }

        /* ── Status ── */
        .status-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px; }
        .status-card { background:white;border-radius:var(--radius);padding:16px;box-shadow:var(--shadow-sm); }
        .status-card h3 { font-size:12px;text-transform:uppercase;letter-spacing:0.5px;color:var(--kp-gray-500);margin-bottom:10px; }
        .status-item { display:flex;justify-content:space-between;padding:5px 0;font-size:13px;border-bottom:1px solid var(--kp-gray-100); }
        .status-item:last-child { border-bottom:none; }
        .status-value { font-weight:600;color:var(--kp-gray-900); }

        /* ── Session items in sidebar ── */
        .sess-item { display:flex;align-items:center;padding:5px 12px 5px 18px;color:rgba(255,255,255,0.7);
            font-size:11px;cursor:pointer;transition:var(--transition);gap:6px;border-radius:4px;margin:1px 4px; }
        .sess-item:hover { background:rgba(255,255,255,0.12);color:white; }
        .sess-item.active { background:rgba(255,255,255,0.18);color:white;font-weight:500; }
        .sess-item .sess-title { flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }
        .sess-item .sess-del { opacity:0;font-size:13px;color:rgba(255,255,255,0.5);cursor:pointer;padding:0 2px; }
        .sess-item:hover .sess-del { opacity:1; }
        .sess-item .sess-del:hover { color:#e74c3c; }

        /* ── Collapsible result cards ── */
        .result-card .result-body.collapsed { display:none; }
        .result-card .collapse-btn { background:none;border:none;cursor:pointer;font-size:14px;color:var(--kp-gray-500);
            padding:0 4px;transition:transform 0.2s; }
        .result-card .collapse-btn.open { transform:rotate(180deg); }

        .toolbar-btn { padding:6px 16px; font-size:12px; border-radius:6px; cursor:pointer; font-weight:500; transition:all 0.2s ease; border:1px solid; }
        .toolbar-btn:hover { transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,0.1); }
        .toolbar-btn-primary { background:var(--kp-blue-pale); color:var(--kp-blue); border-color:var(--kp-blue); }
        .toolbar-btn-teal { background:#E0F7FA; color:#00838F; border-color:var(--kp-teal); }
        .toolbar-btn-danger { background:#FDE8E8; color:#c0392b; border-color:#c0392b; }

        @media (max-width:768px) {
            .sidebar { width:56px; }
            .sidebar-brand h1,.sidebar-brand span,.nav-item span,.nav-section { display:none; }
            .nav-item { justify-content:center;padding:10px; }
            .main { margin-left:56px; }
            .kpi-grid { grid-template-columns:repeat(2,1fr); }
            .widget-grid { grid-template-columns:1fr; }
        }
    </style>
</head>
<body>
    <nav class="sidebar">
        <div class="sidebar-brand">
            <h1>GPDM Analytics</h1>
            <span>KP Healthcare</span>
        </div>
        <div class="sidebar-nav">
            <div class="nav-section">Analytics</div>
            <a class="nav-item active" onclick="showPage('chat')" data-perm="insights">
                <span class="nav-icon">&#128172;</span> <span>Ask Analytics</span>
            </a>
            <a class="nav-item" onclick="showPage('dashboard')" data-perm="dashboards">
                <span class="nav-icon">&#128202;</span> <span>Dashboards</span>
            </a>
            <a class="nav-item" onclick="showPage('quality')" data-perm="insights,analytics_models">
                <span class="nav-icon">&#9989;</span> <span>Quality (HEDIS)</span>
            </a>
            <div class="nav-section">Data</div>
            <a class="nav-item" onclick="showPage('explore')" data-perm="sql_editor">
                <span class="nav-icon">&#128269;</span> <span>Explorer</span>
            </a>
            <a class="nav-item" onclick="showPage('system')" data-perm="insights,analytics_models">
                <span class="nav-icon">&#129504;</span> <span>Intelligence</span>
            </a>
            <a class="nav-item" onclick="showPage('forecast')" data-perm="insights,dashboards">
                <span class="nav-icon">&#128200;</span> <span>Forecasting</span>
            </a>

            <!-- ═══ SESSIONS ═══ -->
            <div class="nav-section" style="margin-top:12px;display:flex;align-items:center;justify-content:space-between;">
                <span>Sessions</span>
                <button onclick="startNewSession()" title="New Chat"
                    style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.3);color:white;
                    border-radius:4px;font-size:10px;padding:1px 8px;cursor:pointer;">+ New</button>
            </div>
            <div id="sessionsList" style="max-height:260px;overflow-y:auto;padding:0 4px;">
                <div style="padding:6px 18px;font-size:11px;color:rgba(255,255,255,0.4);">No sessions yet</div>
            </div>
        </div>
        <div class="sidebar-footer">
            <span class="status-dot"></span> System Online
        </div>
    </nav>

    <div class="main">
        <header class="header">
            <div class="header-title" id="pageTitle">Ask Analytics</div>
            <div class="header-right" id="headerToolbar">
                <button id="btnExport" onclick="exportCurrentCSV()" data-perm="export_csv"
                    class="toolbar-btn toolbar-btn-primary" title="Export last results as CSV">&#128190; Export CSV</button>
                <button id="btnEmail" onclick="openEmailModal()" data-perm="export_email"
                    class="toolbar-btn toolbar-btn-teal" title="Email results">&#9993; Email</button>
                <button id="btnAdmin" onclick="showAdminPanel()" data-perm="admin_panel"
                    class="toolbar-btn toolbar-btn-danger">&#9881; Admin</button>
                <span id="roleBadge" style="padding:4px 12px;border-radius:12px;color:#fff;
                    font-size:10px;font-weight:600;letter-spacing:0.3px;display:none;"></span>
                <span class="badge badge-green" id="statusBadge">Connected</span>
                <span class="badge badge-blue" id="learningBadge">0 patterns</span>
                <span id="userDisplayName" style="font-size:12px;font-weight:500;color:var(--kp-gray-700);"></span>
                <button id="btnLogout" onclick="doLogout()"
                    class="toolbar-btn toolbar-btn-danger">&#x27A1; Logout</button>
            </div>
        </header>

        <div class="content">
            <!-- ═══════════ CHAT PAGE ═══════════ -->
            <div class="page active" id="page-chat">
                <div class="chat-container">
                    <!-- Scrollable conversation thread -->
                    <div class="chat-thread" id="chatThread">
                        <div class="chat-empty" id="chatEmpty">
                            <h2>Healthcare Analytics</h2>
                            <p>Ask questions about claims, members, providers, costs, and more.</p>
                            <div class="suggestions" id="suggestions" style="justify-content:center">
                                <button class="suggestion-chip" onclick="askThis('What is the claim denial rate?')">Denial rate</button>
                                <button class="suggestion-chip" onclick="askThis('Show readmission rate')">Readmission rate</button>
                                <button class="suggestion-chip" onclick="askThis('Compare inpatient vs outpatient costs')">Inpatient vs Outpatient</button>
                                <button class="suggestion-chip" onclick="askThis('Top 10 providers by claim volume')">Top providers</button>
                                <button class="suggestion-chip" onclick="askThis('Show claims by provider specialty')">Claims by specialty</button>
                                <button class="suggestion-chip" onclick="askThis('What percentage of patients are female')">Demographics</button>
                            </div>
                        </div>
                        <div id="chatResults" style="display:none"></div>
                    </div>
                    <!-- Pinned input at bottom -->
                    <div class="chat-input-area">
                        <div class="chat-input-row">
                            <input class="chat-input" id="queryInput" type="text"
                                   placeholder="Ask a question about your healthcare data..."
                                   onkeydown="if(event.key==='Enter')askQuestion()">
                            <button class="chat-btn" onclick="askQuestion()" id="askBtn">Ask</button>
                            <button class="chat-btn chat-btn-followup" onclick="askFollowUp()" id="followBtn" style="display:none">Follow Up</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ═══════════ DASHBOARD PAGE (with custom dashboards) ═══════════ -->
            <div class="page" id="page-dashboard">
                <div class="custom-dash-bar" id="dashTabs" style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
                    <button class="custom-dash-tab active" onclick="switchDash('default')">Executive KPIs</button>
                    <button class="custom-dash-tab" onclick="addCustomDashboard()" style="border-style:dashed;color:var(--kp-gray-500)">+ New Dashboard</button>
                    <span style="flex:1"></span>
                    <button id="btnDashDownload" data-perm="export_csv" onclick="downloadDashboardCSV()" style="padding:5px 14px;font-size:12px;border-radius:6px;border:1px solid var(--kp-blue);background:var(--kp-blue-pale);color:var(--kp-blue);cursor:pointer;font-weight:500" title="Download dashboard KPIs as CSV">&#11015; Download Dashboard</button>
                    <button id="btnDashEmail" data-perm="export_email" onclick="emailDashboard()" style="padding:5px 14px;font-size:12px;border-radius:6px;border:1px solid var(--kp-teal);background:#E0F7FA;color:#00838F;cursor:pointer;font-weight:500" title="Email dashboard">&#9993; Email Dashboard</button>
                </div>
                <div id="dashContent">
                    <div id="alertBanner" style="display:none;background:#FEF3CD;border:1px solid #F2A900;border-radius:var(--radius);padding:10px 14px;margin-bottom:14px;font-size:12px;color:#856404"></div>
                    <div style="margin-bottom:20px">
                        <div style="font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--kp-gray-500);margin-bottom:8px">Revenue &amp; Financial</div>
                        <div class="kpi-grid" id="kpiRevenue"></div>
                    </div>
                    <div style="margin-bottom:20px">
                        <div style="font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--kp-gray-500);margin-bottom:8px">Clinical Quality &amp; Retention</div>
                        <div class="kpi-grid" id="kpiClinical"></div>
                    </div>
                    <div style="margin-bottom:20px">
                        <div style="font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--kp-gray-500);margin-bottom:8px">Operations &amp; Network</div>
                        <div class="kpi-grid" id="kpiOperations"></div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px">
                        <div class="chart-container"><div class="chart-title">Claim Status Distribution</div><canvas id="chartClaimStatus" class="chart-canvas"></canvas></div>
                        <div class="chart-container"><div class="chart-title">Monthly Claims Trend</div><canvas id="chartTrend" class="chart-canvas"></canvas></div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
                        <div class="chart-container"><div class="chart-title">Revenue by Region</div><canvas id="chartRegionRevenue" class="chart-canvas"></canvas></div>
                        <div class="chart-container"><div class="chart-title">Cost by Visit Type</div><canvas id="chartVisitCost" class="chart-canvas"></canvas></div>
                    </div>
                    <!-- KPI Deep Dive Panel (inline on dashboard) -->
                    <div id="kpiDeepDivePanel" style="display:none;margin-top:16px">
                        <div class="chart-container" style="border:2px solid var(--kp-blue-pale)">
                            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
                                <div class="chart-title" id="kpiDeepDiveTitle" style="margin-bottom:0;flex:1"></div>
                                <button onclick="document.getElementById('kpiDeepDivePanel').style.display='none'" style="background:none;border:none;font-size:18px;cursor:pointer;color:var(--kp-gray-500)">&times;</button>
                            </div>
                            <div id="kpiDeepDiveBody"></div>
                        </div>
                    </div>
                </div>
                <!-- Custom dashboard content area -->
                <div id="customDashContent" style="display:none">
                    <div style="display:flex;gap:10px;margin-bottom:14px;align-items:center">
                        <input id="customDashName" style="padding:8px 14px;border:1px solid var(--kp-gray-300);border-radius:var(--radius);font-size:14px;font-weight:600;flex:1;max-width:300px" placeholder="Dashboard name...">
                        <button class="sql-toolbar-btn run" onclick="saveCustomDashboard()">Save Dashboard</button>
                        <button class="sql-toolbar-btn" onclick="addWidget()">+ Add Widget</button>
                    </div>
                    <div class="widget-grid" id="widgetGrid"></div>
                </div>
            </div>

            <!-- ═══════════ QUALITY PAGE ═══════════ -->
            <div class="page" id="page-quality">
                <div id="qualityMeasures"></div>
            </div>

            <!-- ═══════════ EXPLORE PAGE ═══════════ -->
            <div class="page" id="page-explore">
                <div class="explorer-panel">
                    <div class="chart-title">Data Explorer</div>
                    <div class="explorer-toolbar">
                        <select id="tableSelect" onchange="loadTablePreview()" class="explorer-select">
                            <option value="">Select a table...</option>
                        </select>
                        <span style="font-size:12px;color:var(--kp-gray-500)" id="tableInfo"></span>
                    </div>
                    <div id="tablePreview"></div>
                </div>
                <div class="explorer-panel">
                    <div class="chart-title">Custom SQL Query</div>
                    <textarea id="explorerSQL" class="sql-editor" placeholder="SELECT * FROM claims LIMIT 20" style="border:1px solid var(--kp-gray-200);border-radius:4px;min-height:80px"></textarea>
                    <div style="margin-top:8px;display:flex;gap:8px;align-items:center">
                        <button class="sql-toolbar-btn run" onclick="runExplorerSQL()">Run Query</button>
                        <span class="sql-readonly-badge">READ ONLY</span>
                        <span id="explorerStatus" style="font-size:11px;color:var(--kp-gray-500)"></span>
                    </div>
                    <div id="explorerResults" style="margin-top:12px"></div>
                </div>
            </div>

            <!-- ═══════════ SYSTEM PAGE ═══════════ -->
            <div class="page" id="page-system">
                <div class="status-grid" id="systemStatus"></div>
            </div>

            <!-- ═══ FORECASTING PAGE ═══ -->
            <div class="page" id="page-forecast">
                <div style="margin-bottom:16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                    <select id="forecastMetric" onchange="loadSingleForecast()" style="padding:8px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;font-size:13px;min-width:220px;">
                        <option value="all">All Metrics Overview</option>
                    </select>
                    <select id="forecastPeriods" onchange="loadSingleForecast()" style="padding:8px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;font-size:13px;">
                        <option value="3">3 Months</option>
                        <option value="6" selected>6 Months</option>
                        <option value="12">12 Months</option>
                    </select>
                    <span id="forecastLoading" style="display:none;font-size:12px;color:var(--kp-gray-500);">Loading forecasts...</span>
                    <div style="margin-left:auto;font-size:11px;color:var(--kp-gray-500);display:flex;align-items:center;gap:4px;">
                        <span style="width:8px;height:8px;border-radius:50%;background:var(--kp-green);display:inline-block;"></span>
                        HIPAA Compliant &mdash; All models run on-premise
                    </div>
                </div>
                <div id="forecastContent" style="display:grid;grid-template-columns:1fr;gap:16px;">
                    <div style="text-align:center;padding:60px 20px;color:var(--kp-gray-500);">
                        Select a metric or view all forecasts. Models run locally on your server &mdash; no patient data leaves your network.
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- ═══════════ ACCESS CODE GATE ═══════════ -->
    <div id="accessCodeOverlay" style="position:fixed;inset:0;z-index:10000;background:linear-gradient(135deg,#001a3a 0%,#002855 50%,#003d7a 100%);display:flex;align-items:center;justify-content:center;">
        <div style="background:white;border-radius:12px;padding:40px;width:380px;box-shadow:0 20px 60px rgba(0,0,0,0.4);">
            <div style="text-align:center;margin-bottom:24px;">
                <div style="font-size:40px;margin-bottom:8px;">&#128274;</div>
                <div style="font-size:22px;font-weight:700;color:#002855;">GPDM Secure Access</div>
                <div style="font-size:13px;color:#666;margin-top:4px;">Enter your access code to continue</div>
            </div>
            <div id="accessCodeError" style="display:none;background:#FDEAEA;color:#e53e3e;padding:8px 12px;border-radius:6px;font-size:12px;margin-bottom:12px;"></div>
            <label style="font-size:12px;font-weight:600;color:#374151;display:block;margin-bottom:4px;">Access Code</label>
            <input id="accessCodeInput" type="password" placeholder="Enter access code"
                style="width:100%;padding:10px 12px;border:1px solid #D1D5DB;border-radius:6px;font-size:14px;margin-bottom:16px;box-sizing:border-box;"
                onkeydown="if(event.key==='Enter')verifyAccessCode()">
            <button onclick="verifyAccessCode()"
                style="width:100%;padding:10px;background:#002855;color:white;border:none;border-radius:6px;font-size:14px;font-weight:600;cursor:pointer;">
                Verify Access Code
            </button>
            <div style="text-align:center;margin-top:16px;font-size:11px;color:#999;">
                Contact your administrator for access credentials
            </div>
        </div>
    </div>

    <!-- ═══════════ LOGIN OVERLAY ═══════════ -->
    <div id="loginOverlay" style="position:fixed;inset:0;z-index:9999;background:linear-gradient(135deg,var(--kp-blue) 0%,#003366 100%);display:flex;align-items:center;justify-content:center;">
        <div style="background:white;border-radius:12px;padding:40px;width:380px;box-shadow:0 20px 60px rgba(0,0,0,0.3);">
            <div style="text-align:center;margin-bottom:24px;">
                <div style="width:48px;height:48px;border-radius:50%;background:var(--kp-blue);display:flex;align-items:center;justify-content:center;color:white;font-weight:800;font-size:20px;margin:0 auto 12px;">KP</div>
                <div style="font-size:28px;font-weight:700;color:var(--kp-blue);">GPDM Analytics</div>
                <div style="font-size:13px;color:var(--kp-gray-500);margin-top:4px;">KP Healthcare Healthcare Intelligence Platform</div>
            </div>
            <div id="loginError" style="display:none;background:#FDEAEA;color:var(--kp-red);padding:8px 12px;border-radius:6px;font-size:12px;margin-bottom:12px;"></div>
            <div id="loginForm">
                <label style="font-size:12px;font-weight:600;color:var(--kp-gray-700);display:block;margin-bottom:4px;">Username</label>
                <input id="loginUser" type="text" placeholder="Enter username" style="width:100%;padding:10px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;font-size:14px;margin-bottom:12px;" onkeydown="if(event.key==='Enter')doLogin()">
                <label style="font-size:12px;font-weight:600;color:var(--kp-gray-700);display:block;margin-bottom:4px;">Password</label>
                <input id="loginPass" type="password" placeholder="Enter password" style="width:100%;padding:10px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;font-size:14px;margin-bottom:16px;" onkeydown="if(event.key==='Enter')doLogin()">
                <button onclick="doLogin()" style="width:100%;padding:10px;background:var(--kp-blue);color:white;border:none;border-radius:6px;font-size:14px;font-weight:600;cursor:pointer;">Sign In</button>
                <div style="text-align:center;margin-top:12px;">
                    <span style="font-size:12px;color:var(--kp-gray-500);">Don't have an account?</span>
                    <a href="#" onclick="showSignup();return false;" style="font-size:12px;color:var(--kp-blue);font-weight:600;text-decoration:none;margin-left:4px;">Sign Up</a>
                </div>
            </div>
            <div id="signupForm" style="display:none;">
                <label style="font-size:12px;font-weight:600;color:var(--kp-gray-700);display:block;margin-bottom:4px;">Username</label>
                <input id="signupUser" type="text" placeholder="Choose a username" style="width:100%;padding:10px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;font-size:14px;margin-bottom:12px;">
                <label style="font-size:12px;font-weight:600;color:var(--kp-gray-700);display:block;margin-bottom:4px;">Password</label>
                <input id="signupPass" type="password" placeholder="Choose a password" style="width:100%;padding:10px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;font-size:14px;margin-bottom:12px;">
                <label style="font-size:12px;font-weight:600;color:var(--kp-gray-700);display:block;margin-bottom:4px;">Confirm Password</label>
                <input id="signupPass2" type="password" placeholder="Confirm password" style="width:100%;padding:10px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;font-size:14px;margin-bottom:16px;" onkeydown="if(event.key==='Enter')doSignup()">
                <button onclick="doSignup()" style="width:100%;padding:10px;background:var(--kp-green);color:white;border:none;border-radius:6px;font-size:14px;font-weight:600;cursor:pointer;">Create Account</button>
                <div style="text-align:center;margin-top:12px;">
                    <a href="#" onclick="showLogin();return false;" style="font-size:12px;color:var(--kp-blue);font-weight:600;text-decoration:none;">Back to Sign In</a>
                </div>
            </div>
        </div>
    </div>

    <script>
    const API = '/api';
    let sessionId = null;      // current chat session ID (from DB)
    let allSessions = [];       // loaded from server
    let chartInstances = {};
    let lastQuestion = '';
    let lastSQL = '';
    let lastData = null;
    let _cardResults = {};  // cardId → {columns, rows, question} for per-card download
    let _cardCounter = 0;
    let authToken = null;  // Auth token from login
    let currentUser = null;
    let userPermissions = [];
    let userRole = '';
    let userRoleLabel = '';

    // ─── Access Code Verification ───
    function verifyAccessCode() {
        const code = document.getElementById('accessCodeInput').value.trim();
        const errEl = document.getElementById('accessCodeError');
        if (!code) { errEl.style.display='block'; errEl.textContent='Please enter the access code'; return; }
        fetch('/api/verify-access-code', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ access_code: code })
        }).then(r => r.json()).then(data => {
            if (data.success) {
                document.getElementById('accessCodeOverlay').style.display = 'none';
                sessionStorage.setItem('gpdm_access_verified', '1');
            } else {
                errEl.style.display='block';
                errEl.textContent = data.message || 'Invalid access code';
                document.getElementById('accessCodeInput').value = '';
                document.getElementById('accessCodeInput').focus();
            }
        }).catch(e => { errEl.style.display='block'; errEl.textContent='Verification failed: ' + e.message; });
    }
    // Auto-hide if already verified this session
    if (sessionStorage.getItem('gpdm_access_verified') === '1') {
        document.getElementById('accessCodeOverlay').style.display = 'none';
    }

    function hasPerm(p) { return userPermissions.includes(p); }
    function applyPermissions() {
        console.log('[RBAC] applyPermissions called — role=' + userRole + ', perms=' + userPermissions.join(','));
        document.querySelectorAll('[data-perm]').forEach(el => {
            const required = el.dataset.perm.split(',');
            const allowed = required.some(p => hasPerm(p.trim()));
            el.style.display = allowed ? '' : 'none';
            if (allowed) el.style.removeProperty('display');
            console.log('[RBAC]   ' + (el.id||el.tagName) + ' perm=' + el.dataset.perm + ' => ' + (allowed?'SHOW':'HIDE'));
        });
        // Disable query input for non-insight roles
        const qi = document.getElementById('questionInput');
        if (qi && !hasPerm('insights')) {
            qi.placeholder = 'NL queries require Business or Admin role';
            qi.disabled = true;
        } else if (qi) {
            qi.placeholder = 'Ask anything about your healthcare data...';
            qi.disabled = false;
        }
        // Role badge
        const badge = document.getElementById('roleBadge');
        if (badge) {
            badge.textContent = userRoleLabel;
            const colors = {admin:'#c0392b',business:'#2980b9',dev:'#27ae60',viewer:'#8e44ad',member:'#2980b9'};
            badge.style.background = colors[userRole] || '#666';
            badge.style.display = userRole ? 'inline-block' : 'none';
        }
        // Always show Logout
        const logoutBtn = document.getElementById('btnLogout');
        if (logoutBtn) logoutBtn.style.display = '';
    }

    // ════════════════════════════════════════════════════════════
    // SESSION MANAGEMENT — persistent per-user chat sessions
    // ════════════════════════════════════════════════════════════

    async function loadSessions() {
        try {
            const resp = await authFetch(API + '/chat/sessions', {
                method: 'POST', headers: authHeaders(),
                body: JSON.stringify({token: authToken, action: 'list'}),
            });
            const d = await resp.json();
            if (d.success) {
                allSessions = d.sessions || [];
                renderSessionsList();
                // If no active session, create one
                if (!sessionId && allSessions.length > 0) {
                    switchSession(allSessions[0].session_id);
                } else if (!sessionId) {
                    await startNewSession();
                }
            }
        } catch(e) { console.warn('loadSessions error:', e); }
    }

    function renderSessionsList() {
        const el = document.getElementById('sessionsList');
        if (!el) return;
        if (allSessions.length === 0) {
            el.innerHTML = '<div style="padding:6px 18px;font-size:11px;color:rgba(255,255,255,0.4);">No sessions yet</div>';
            return;
        }
        el.innerHTML = allSessions.map(s => {
            const active = s.session_id === sessionId ? ' active' : '';
            const date = s.updated_at ? s.updated_at.split(' ')[0] : '';
            return '<div class="sess-item' + active + '" onclick="switchSession(\\'' + s.session_id + '\\')" title="' + (s.title||'').replace(/"/g,'') + '">'
                + '<span style="font-size:12px;">&#128488;</span>'
                + '<span class="sess-title">' + (s.title || 'New Chat') + '</span>'
                + '<span style="font-size:9px;color:rgba(255,255,255,0.4);white-space:nowrap;">' + date + '</span>'
                + '<span class="sess-del" onclick="event.stopPropagation();deleteSession(\\'' + s.session_id + '\\')" title="Delete">&#10005;</span>'
                + '</div>';
        }).join('');
    }

    async function startNewSession() {
        try {
            const resp = await authFetch(API + '/chat/sessions', {
                method: 'POST', headers: authHeaders(),
                body: JSON.stringify({token: authToken, action: 'create', title: 'New Chat'}),
            });
            const d = await resp.json();
            if (d.success) {
                sessionId = d.session_id;
                allSessions.unshift({session_id: d.session_id, title: d.title, updated_at: new Date().toISOString(), is_active: 1});
                renderSessionsList();
                // Reset chat to empty state
                document.getElementById('chatResults').innerHTML = '';
                document.getElementById('chatResults').style.display = 'none';
                document.getElementById('chatEmpty').style.display = 'flex';
                lastQuestion = '';
                lastSQL = '';
                lastData = null;
                document.getElementById('followBtn').style.display = 'none';
                showPage('chat');
                document.getElementById('queryInput').focus();
            }
        } catch(e) { console.warn('startNewSession error:', e); }
    }

    async function switchSession(sid) {
        if (sid === sessionId) return;
        sessionId = sid;
        renderSessionsList();
        // Load messages for this session
        try {
            const resp = await authFetch(API + '/chat/messages', {
                method: 'POST', headers: authHeaders(),
                body: JSON.stringify({token: authToken, action: 'list', session_id: sid}),
            });
            const d = await resp.json();
            const results = document.getElementById('chatResults');
            results.innerHTML = '';
            if (d.success && d.messages && d.messages.length > 0) {
                showChatResults();
                // Render chronologically: question bubble + answer card for each
                d.messages.forEach((msg, i) => {
                    // Question bubble
                    results.insertAdjacentHTML('beforeend',
                        '<div class="q-bubble"><div class="q-bubble-inner">' + (msg.question||'').replace(/</g,'&lt;') + '</div></div>');
                    // Answer card (collapsed for older, expanded for latest)
                    const isLast = (i === d.messages.length - 1);
                    const cardData = {
                        columns: msg.answer_columns || [],
                        rows: msg.answer_rows || [],
                        row_count: msg.row_count || 0,
                        narrative: msg.answer_narrative || '',
                        sql: msg.answer_sql || '',
                        intent: msg.intent || '',
                        confidence: {grade: msg.grade || ''},
                        strategy: msg.strategy || '',
                    };
                    const card = buildResultCard(msg.question, cardData, !isLast);
                    results.insertAdjacentHTML('beforeend', card);
                });
                applyPermissions();
                scrollToBottom();
                // Set context from last message
                const lastMsg = d.messages[d.messages.length - 1];
                lastQuestion = lastMsg.question;
                lastData = {columns: lastMsg.answer_columns, rows: lastMsg.answer_rows, narrative: lastMsg.answer_narrative};
                lastSQL = lastMsg.answer_sql || '';
                document.getElementById('followBtn').style.display = 'inline-block';
            } else {
                // Empty session — show empty state
                document.getElementById('chatEmpty').style.display = 'flex';
                document.getElementById('chatResults').style.display = 'none';
                lastQuestion = '';
                lastSQL = '';
                lastData = null;
                document.getElementById('followBtn').style.display = 'none';
            }
            // Show chat page
            showPage('chat');
        } catch(e) { console.warn('switchSession error:', e); }
    }

    async function deleteSession(sid) {
        if (!confirm('Delete this session and all its messages?')) return;
        try {
            await authFetch(API + '/chat/sessions', {
                method: 'POST', headers: authHeaders(),
                body: JSON.stringify({token: authToken, action: 'delete', session_id: sid}),
            });
            allSessions = allSessions.filter(s => s.session_id !== sid);
            if (sid === sessionId) {
                sessionId = null;
                document.getElementById('chatResults').innerHTML = '';
                if (allSessions.length > 0) {
                    switchSession(allSessions[0].session_id);
                } else {
                    await startNewSession();
                }
            }
            renderSessionsList();
        } catch(e) { console.warn('deleteSession error:', e); }
    }

    async function saveMessageToSession(question, data) {
        if (!sessionId) return;
        try {
            await authFetch(API + '/chat/messages', {
                method: 'POST', headers: authHeaders(),
                body: JSON.stringify({
                    token: authToken, action: 'save', session_id: sessionId,
                    question: question,
                    narrative: data.narrative || '',
                    sql: data.sql || '',
                    columns: data.columns || [],
                    rows: (data.rows || []).slice(0, 200),  // cap at 200 rows for storage
                    row_count: data.row_count || (data.rows||[]).length,
                    intent: data.intent || '',
                    grade: (data.confidence||{}).grade || '',
                    strategy: data.strategy || '',
                }),
            });
            // Update session title in sidebar if first message
            const sess = allSessions.find(s => s.session_id === sessionId);
            if (sess && sess.title === 'New Chat') {
                sess.title = question.substring(0, 50) + (question.length > 50 ? '...' : '');
                sess.updated_at = new Date().toISOString();
                renderSessionsList();
            }
        } catch(e) { console.warn('saveMessage error:', e); }
    }

    function toggleCard(btn) {
        const card = btn.closest('.result-card');
        const body = card.querySelector('.result-body');
        const extras = card.querySelectorAll('.sql-editor-area, .chart-canvas, div[style*="border-top"]');
        if (body.classList.contains('collapsed')) {
            body.classList.remove('collapsed');
            extras.forEach(el => el.style.display = '');
            btn.classList.add('open');
            btn.innerHTML = '&#9650;';
        } else {
            body.classList.add('collapsed');
            extras.forEach(el => el.style.display = 'none');
            btn.classList.remove('open');
            btn.innerHTML = '&#9660;';
        }
    }

    // ── Auth helpers ──
    function authHeaders() {
        const h = {'Content-Type': 'application/json'};
        if (authToken) h['Authorization'] = 'Bearer ' + authToken;
        return h;
    }

    // Authenticated fetch wrapper — adds auth header to all requests
    function authFetch(url, opts = {}) {
        if (!opts.headers) opts.headers = {};
        if (authToken) opts.headers['Authorization'] = 'Bearer ' + authToken;
        return fetch(url, opts).then(resp => {
            if (resp.status === 401) {
                // Token expired — show login again
                document.getElementById('loginOverlay').style.display = 'flex';
                authToken = null;
                const err = document.getElementById('loginError');
                err.textContent = 'Session expired — please sign in again';
                err.style.display = 'block';
            }
            return resp;
        });
    }

    function showSignup() {
        document.getElementById('loginForm').style.display = 'none';
        document.getElementById('signupForm').style.display = 'block';
        document.getElementById('loginError').style.display = 'none';
    }
    function showLogin() {
        document.getElementById('signupForm').style.display = 'none';
        document.getElementById('loginForm').style.display = 'block';
        document.getElementById('loginError').style.display = 'none';
    }

    async function doLogin() {
        const user = document.getElementById('loginUser').value.trim();
        const pass = document.getElementById('loginPass').value;
        if (!user || !pass) return;
        try {
            const resp = await fetch(API + '/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username: user, password: pass}),
            });
            const data = await resp.json();
            if (data.success && data.token) {
                authToken = data.token;
                currentUser = data.user || user;
                userPermissions = data.permissions || [];
                userRole = data.role || 'member';
                userRoleLabel = data.role_label || userRole;
                document.getElementById('loginOverlay').style.display = 'none';
                document.getElementById('userDisplayName').textContent = user;
                applyPermissions();
                loadSessions();  // Load persistent chat sessions
            } else {
                const err = document.getElementById('loginError');
                err.textContent = data.error || 'Invalid credentials';
                err.style.display = 'block';
            }
        } catch (e) {
            const err = document.getElementById('loginError');
            err.textContent = 'Connection error — is the server running?';
            err.style.display = 'block';
        }
    }

    async function doSignup() {
        const user = document.getElementById('signupUser').value.trim();
        const pass = document.getElementById('signupPass').value;
        const pass2 = document.getElementById('signupPass2').value;
        if (!user || !pass) return;
        if (pass !== pass2) {
            const err = document.getElementById('loginError');
            err.textContent = 'Passwords do not match';
            err.style.display = 'block';
            return;
        }
        try {
            const resp = await fetch(API + '/signup', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username: user, password: pass}),
            });
            const data = await resp.json();
            if (data.success && data.token) {
                authToken = data.token;
                currentUser = data.user || user;
                userPermissions = data.permissions || [];
                userRole = data.role || 'member';
                userRoleLabel = data.role_label || userRole;
                document.getElementById('loginOverlay').style.display = 'none';
                applyPermissions();
                loadSessions();  // Load persistent chat sessions
            } else {
                const err = document.getElementById('loginError');
                err.textContent = data.error || 'Signup failed';
                err.style.display = 'block';
            }
        } catch (e) {
            const err = document.getElementById('loginError');
            err.textContent = 'Connection error — is the server running?';
            err.style.display = 'block';
        }
    }
    let customDashboards = [];
    let activeCustomDash = null;

    // ── Navigation ──
    function showPage(name) {
        window.history.pushState({page: name}, '', '#' + name);
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        const page = document.getElementById('page-' + name);
        if (page) page.classList.add('active');
        const nav = document.querySelector('.nav-item[onclick*=\"' + name + '\"]');
        if (nav) nav.classList.add('active');
        const titles = { chat:'Ask Analytics', dashboard:'Dashboards', quality:'Quality Measures (HEDIS)', explore:'Data Explorer', system:'System Intelligence', forecast:'Forecasting' };
        document.getElementById('pageTitle').textContent = titles[name] || name;
        if (name === 'dashboard') loadDashboard();
        if (name === 'quality') loadQuality();
        if (name === 'explore') loadExplorer();
        if (name === 'system') loadSystem();
        if (name === 'forecast') loadForecast();
    }

    // Browser back/forward support
    window.addEventListener('popstate', function(e) {
        if (e.state && e.state.page) showPage(e.state.page);
        else showPage('chat');
    });

    // Initialize page from URL hash on load
    window.addEventListener('load', function() {
        const initialPage = window.location.hash.replace('#','') || 'chat';
        if (initialPage !== 'chat') showPage(initialPage);
    });

    // ── Chat ──
    function askThis(q) {
        document.getElementById('queryInput').value = q;
        askQuestion();
    }

    function askFollowUp() {
        const input = document.getElementById('queryInput');
        const q = input.value.trim();
        if (!q || !lastQuestion) return;
        // Prepend context so pipeline treats it as follow-up
        input.value = q;
        askQuestion(true);
    }

    function scrollToBottom() {
        const thread = document.getElementById('chatThread');
        if (thread) setTimeout(() => { thread.scrollTop = thread.scrollHeight; }, 80);
    }

    function showChatResults() {
        const empty = document.getElementById('chatEmpty');
        const results = document.getElementById('chatResults');
        if (empty) empty.style.display = 'none';
        if (results) results.style.display = 'block';
    }

    async function askQuestion(isFollowUp = false) {
        if (!hasPerm('insights')) {
            alert('Natural language queries require Business User or Admin role.');
            return;
        }
        const input = document.getElementById('queryInput');
        const q = input.value.trim();
        if (!q) return;

        // Ensure we have a session
        if (!sessionId) await startNewSession();

        const btn = document.getElementById('askBtn');
        const followBtn = document.getElementById('followBtn');
        btn.disabled = true;
        btn.textContent = 'Thinking...';
        if (followBtn) followBtn.disabled = true;

        const results = document.getElementById('chatResults');
        showChatResults();

        // Add question bubble (user message)
        results.insertAdjacentHTML('beforeend',
            '<div class="q-bubble"><div class="q-bubble-inner">' + q.replace(/</g,'&lt;') + '</div></div>');

        // Add loading indicator
        const loadingId = 'loading_' + Date.now();
        results.insertAdjacentHTML('beforeend',
            '<div class="result-card" id="' + loadingId + '" style="animation:fadeIn 0.2s ease"><div class="loading"><div class="spinner"></div> Analyzing...</div></div>');
        scrollToBottom();

        try {
            const payload = { question: q, session_id: sessionId };
            if (isFollowUp && lastQuestion) payload.context = lastQuestion;

            const resp = await authFetch(API + '/intelligent/query', {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            document.getElementById(loadingId)?.remove();

            lastQuestion = q;
            lastSQL = data.sql || '';
            lastData = data;

            // Append answer card (chronological — at bottom)
            const card = buildResultCard(q, data, false);
            results.insertAdjacentHTML('beforeend', card);
            applyPermissions();
            scrollToBottom();

            // Save to persistent session
            saveMessageToSession(q, data);

            // Render chart if applicable
            if (data.rows && data.rows.length > 1 && data.rows.length <= 30) {
                const lastCard = results.querySelector('.result-card:last-child');
                const cid = lastCard?.getAttribute('data-chart-id');
                if (cid) setTimeout(() => renderChart(cid, data), 100);
            }

            // Show follow-up button
            followBtn.style.display = 'inline-block';
            document.getElementById('learningBadge').textContent = (data.memory_recalls || 0) + ' patterns';

            // Bump this session to top of sidebar
            const sess = allSessions.find(s => s.session_id === sessionId);
            if (sess) {
                if (sess.title === 'New Chat') sess.title = q.substring(0, 50) + (q.length > 50 ? '...' : '');
                sess.updated_at = new Date().toISOString();
                allSessions.sort((a,b) => b.updated_at.localeCompare(a.updated_at));
                renderSessionsList();
            }

        } catch (err) {
            document.getElementById(loadingId)?.remove();
            results.insertAdjacentHTML('beforeend',
                '<div class="result-card" style="animation:fadeIn 0.2s ease"><div class="result-body" style="color:var(--kp-red)">Error: ' + err.message + '</div></div>');
            scrollToBottom();
        }

        btn.disabled = false;
        btn.textContent = 'Ask';
        if (followBtn) followBtn.disabled = false;
        input.value = '';
        input.focus();
    }

    function buildConceptCard(question, data) {
        const dims = data.dimensions || [];
        const label = data.label || 'Multi-Dimensional Analysis';
        const desc = data.description || '';
        const grade = (data.confidence || {}).grade || 'A';
        const latency = data.latency_ms || 0;
        const totalDims = data.total_dimensions || dims.length;
        const gradeColors = {A:'#48A23F',B:'#006BA6',C:'#FF6B35',D:'#D32F2F',F:'#7B0000'};

        // ── Header ──
        let html = '<div class="result-card" style="border:none;box-shadow:0 2px 12px rgba(0,0,0,0.08);border-radius:10px;overflow:hidden">';
        html += '<div style="background:linear-gradient(135deg,#003B71,#006BA6);color:white;padding:16px 20px">';
        html += '<div style="display:flex;justify-content:space-between;align-items:center">';
        html += '<div><div style="font-size:16px;font-weight:700">' + label + '</div>';
        html += '<div style="font-size:11px;opacity:0.85;margin-top:2px">' + desc + '</div></div>';
        html += '<div style="display:flex;gap:8px;align-items:center">';
        html += '<span style="background:rgba(255,255,255,0.2);padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600">' + totalDims + ' Dimensions</span>';
        html += '<span style="background:rgba(255,255,255,0.2);padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600">' + latency + 'ms</span>';
        html += '<div style="width:28px;height:28px;border-radius:50%;background:' + (gradeColors[grade]||'#666') + ';display:flex;align-items:center;justify-content:center;font-weight:800;font-size:14px">' + grade + '</div>';
        html += '</div></div></div>';

        // ── Dimension grid ──
        html += '<div style="padding:16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px">';

        for (let i = 0; i < dims.length; i++) {
            const dim = dims[i];
            const dimChartId = 'concept_chart_' + Date.now() + '_' + i;
            const rows = dim.rows || [];
            const cols = dim.columns || [];
            const chartType = dim.chart_type || 'bar';
            const isError = !!dim.error;

            html += '<div style="border:1px solid #E5E7EB;border-radius:8px;overflow:hidden;background:#fff">';

            // Dimension header
            html += '<div style="background:linear-gradient(135deg,#F8FAFF,#EEF2FF);padding:10px 14px;border-bottom:1px solid #E5E7EB">';
            html += '<div style="font-size:12px;font-weight:700;color:#1E3A5F">' + dim.label + '</div>';
            if (dim.row_count > 0) {
                html += '<span style="font-size:9px;color:#6B7280;margin-left:0">' + dim.row_count + ' categories</span>';
            }
            html += '</div>';

            if (isError) {
                html += '<div style="padding:14px;color:#D32F2F;font-size:11px">Unable to compute this dimension.</div>';
            } else if (rows.length > 0) {
                // Data table
                html += '<div style="max-height:200px;overflow-y:auto;padding:0">';
                html += '<table style="width:100%;border-collapse:collapse;font-size:11px">';
                // Header row
                html += '<thead><tr style="background:#F9FAFB;position:sticky;top:0">';
                for (const c of cols) {
                    html += '<th style="padding:6px 10px;text-align:left;font-weight:600;color:#374151;border-bottom:1px solid #E5E7EB;font-size:10px;text-transform:uppercase;letter-spacing:0.3px">' + c.replace(/_/g,' ') + '</th>';
                }
                html += '</tr></thead><tbody>';
                // Data rows (limit to 12)
                const displayRows = rows.slice(0, 12);
                for (let r = 0; r < displayRows.length; r++) {
                    const row = displayRows[r];
                    const bgColor = r % 2 === 0 ? '#fff' : '#F9FAFB';
                    html += '<tr style="background:' + bgColor + '">';
                    for (let c = 0; c < row.length; c++) {
                        const val = row[c];
                        const isNum = typeof val === 'number';
                        const formatted = isNum ? (val % 1 !== 0 ? val.toFixed(1) : val.toLocaleString()) : val;
                        const isPct = cols[c] && cols[c].toLowerCase().includes('pct');
                        html += '<td style="padding:5px 10px;color:#1F2937;border-bottom:1px solid #F3F4F6;' + (isNum ? 'text-align:right;font-variant-numeric:tabular-nums' : '') + '">' + formatted + (isPct ? '%' : '') + '</td>';
                    }
                    html += '</tr>';
                }
                if (rows.length > 12) {
                    html += '<tr><td colspan="' + cols.length + '" style="padding:4px 10px;font-size:10px;color:#9CA3AF;text-align:center">+ ' + (rows.length - 12) + ' more rows</td></tr>';
                }
                html += '</tbody></table></div>';

                // Inline mini-chart (horizontal bar for top 5)
                if (rows.length > 1 && cols.length >= 2) {
                    const maxVal = Math.max(...rows.slice(0,8).map(r => typeof r[1] === 'number' ? r[1] : 0));
                    html += '<div style="padding:8px 14px;border-top:1px solid #F3F4F6">';
                    for (let r = 0; r < Math.min(rows.length, 6); r++) {
                        const cat = rows[r][0];
                        const val = typeof rows[r][1] === 'number' ? rows[r][1] : 0;
                        const pctWidth = maxVal > 0 ? Math.round(val / maxVal * 100) : 0;
                        const colors = ['#006BA6','#48A23F','#FF6B35','#8B5CF6','#EC4899','#14B8A6'];
                        html += '<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">';
                        html += '<span style="font-size:9px;color:#6B7280;width:80px;text-align:right;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + cat + '</span>';
                        html += '<div style="flex:1;background:#F3F4F6;border-radius:3px;height:14px;overflow:hidden"><div style="width:' + pctWidth + '%;height:100%;background:' + colors[r % colors.length] + ';border-radius:3px;transition:width 0.5s"></div></div>';
                        html += '<span style="font-size:9px;color:#374151;font-weight:600;width:45px;font-variant-numeric:tabular-nums">' + val.toLocaleString() + '</span>';
                        html += '</div>';
                    }
                    html += '</div>';
                }
            }

            // Insight
            if (dim.insight && !isError) {
                html += '<div style="padding:8px 14px;border-top:1px solid #F3F4F6;font-size:10.5px;color:#4B5563;line-height:1.5;background:#FAFBFF">';
                html += '<span style="font-weight:600;color:#1E3A5F">Insight:</span> ' + dim.insight + '</div>';
            }

            // Medical context
            if (dim.medical_context && !isError) {
                html += '<div style="padding:6px 14px;font-size:9.5px;color:#6B7280;line-height:1.4;border-top:1px solid #F3F4F6;background:#F9FAFB">';
                html += '<span style="font-weight:600;color:#2D6A4F">Clinical Relevance:</span> ' + dim.medical_context + '</div>';
            }

            html += '</div>'; // close dimension card
        }

        html += '</div>'; // close grid

        // ── Follow-up suggestions ──
        const suggestions = data.suggestions || [];
        if (suggestions.length > 0) {
            html += '<div style="padding:12px 16px;border-top:1px solid var(--kp-gray-200);background:linear-gradient(135deg,#FAFBFF,#F5F7FF)">';
            html += '<div style="font-size:9px;font-weight:700;color:var(--kp-blue);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px">Explore Further — Cross-Dimensional Questions</div>';
            html += '<div style="display:flex;flex-wrap:wrap;gap:5px">';
            for (const s of suggestions.slice(0, 6)) {
                html += '<button class="suggestion-chip" style="font-size:10.5px;padding:4px 12px;background:var(--kp-blue-pale);border:1px solid #D0D9E8;border-radius:16px;line-height:1.4" onclick="askThis(\\'' + s.replace(/\\'/g, '') + '\\')">' + s + '</button>';
            }
            html += '</div></div>';
        }

        html += '</div>'; // close card
        return html;
    }

    function buildResultCard(question, data, startCollapsed) {
        // ── CONCEPT EXPANSION: Multi-dimensional dashboard ──
        if (data.is_concept && data.dimensions && data.dimensions.length > 0) {
            return buildConceptCard(question, data);
        }

        const chartId = 'chart_' + Date.now();
        const sqlId = 'sql_' + Date.now();
        const editorId = 'ed_' + Date.now();
        let bodyHTML = '';
        let insightHTML = '';

        // ── Insight section: narrative + confidence + anomalies + clinical ──
        const narrative = data.narrative || data.explanation || data.semantic_intent || '';
        const clinical = data.clinical_context;
        const strategy = data.strategy || '';
        const latency = data.latency_ms || '';
        const confidence = data.confidence || {};
        const anomalies = data.anomalies || [];
        const healActions = data.heal_actions || [];
        const cacheHit = data.cache_hit || false;
        const grade = confidence.grade || '';

        // Confidence grade badge
        const gradeColors = {A:'#48A23F',B:'#006BA6',C:'#FF6B35',D:'#D32F2F',F:'#7B0000'};
        if (grade) {
            insightHTML += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
                + '<div style="display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:50%;background:' + (gradeColors[grade]||'#666') + ';color:#fff;font-weight:800;font-size:16px">' + grade + '</div>'
                + '<div style="font-size:11px;color:var(--kp-gray-500)">Confidence: ' + ((confidence.overall||0)*100).toFixed(0) + '%'
                + (cacheHit ? ' &middot; <span style="color:var(--kp-teal);font-weight:600">CACHED</span>' : '')
                + '</div>'
                + '</div>';
        }

        // ── NARRATIVE: Split into structured sections ──
        if (narrative) {
            // Parse narrative into sections: summary, outlier alerts, recommendations, data notes
            const recMatch = narrative.match(/Recommended Actions:(.*?)$/s);
            const outlierMatch = narrative.match(/Outlier Alert.*?(?=Data Confidence|Data Completeness|Recommended Actions|Clinical Context|$)/s);
            const dataNote = narrative.match(/(Data Completeness Note:.*?)(?=Recommended Actions|Clinical Context|$)/s);
            const confAdv = narrative.match(/(Data Confidence Advisory:.*?)(?=Recommended Actions|$)/s);
            const clinCtx = narrative.match(/Clinical Context —(.*?)(?=Recommended Actions|Outlier Alert|$)/s);

            // Extract clean summary (everything before first special section)
            let summary = narrative
                .replace(/Outlier Alert.*$/s, '')
                .replace(/Data Completeness Note:.*$/s, '')
                .replace(/Data Confidence Advisory:.*$/s, '')
                .replace(/Note: Moderate confidence.*$/s, '')
                .replace(/Recommended Actions:.*$/s, '')
                .replace(/Clinical Context —.*$/s, '')
                .trim();

            // Executive Summary block
            if (summary) {
                insightHTML += '<div style="background:linear-gradient(135deg,#F0F6FF,#EBF0FF);border-radius:8px;padding:14px 16px;margin-bottom:10px;font-size:12.5px;color:#1E3A5F;line-height:1.7;border-left:4px solid var(--kp-blue)">'
                    + '<div style="font-size:10px;font-weight:700;color:var(--kp-blue);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px">Executive Summary</div>'
                    + summary + '</div>';
            }

            // Outlier Alert block
            if (outlierMatch) {
                const outlierText = outlierMatch[0].trim();
                const bullets = outlierText.split(/(?=Provider |Facility |Plan |Region |Member )/).filter(s => s.trim() && !s.startsWith('Outlier Alert'));
                const headerPart = outlierText.match(/Outlier Alert \((\d+) flagged\)/);
                const count = headerPart ? headerPart[1] : '';
                insightHTML += '<div style="background:#FFF8F0;border-radius:8px;padding:14px 16px;margin-bottom:10px;font-size:12px;border-left:4px solid #E65100">'
                    + '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px"><span style="font-size:10px;font-weight:700;color:#E65100;text-transform:uppercase;letter-spacing:0.8px">Outlier Alerts</span>'
                    + (count ? '<span style="background:#E65100;color:#fff;padding:1px 7px;border-radius:10px;font-size:9px;font-weight:700">' + count + ' flagged</span>' : '')
                    + '</div>';
                if (bullets.length > 0) {
                    insightHTML += '<div style="display:flex;flex-direction:column;gap:6px">';
                    for (const b of bullets.slice(0,3)) {
                        const isExtreme = b.includes('extreme');
                        const bgColor = isExtreme ? '#FFF0F0' : '#FFFAF5';
                        const borderColor = isExtreme ? '#D32F2F' : '#FF8A65';
                        insightHTML += '<div style="background:' + bgColor + ';border-radius:6px;padding:8px 12px;border-left:3px solid ' + borderColor + ';font-size:11.5px;line-height:1.5;color:#4A2800">' + b.trim() + '</div>';
                    }
                    insightHTML += '</div>';
                }
                insightHTML += '</div>';
            } else if (anomalies.length > 0) {
                // Fallback: use raw anomaly objects
                insightHTML += '<div style="background:#FFF8F0;border-radius:8px;padding:14px 16px;margin-bottom:10px;font-size:12px;border-left:4px solid #E65100">'
                    + '<div style="font-size:10px;font-weight:700;color:#E65100;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Outlier Alerts <span style="background:#E65100;color:#fff;padding:1px 7px;border-radius:10px;font-size:9px;font-weight:700">' + anomalies.length + '</span></div>'
                    + '<div style="display:flex;flex-direction:column;gap:6px">';
                for (const a of anomalies.slice(0,3)) {
                    const isHigh = a.severity === 'high';
                    insightHTML += '<div style="background:' + (isHigh?'#FFF0F0':'#FFFAF5') + ';border-radius:6px;padding:8px 12px;border-left:3px solid ' + (isHigh?'#D32F2F':'#FF8A65') + ';font-size:11.5px;line-height:1.5;color:#4A2800">' + a.message + '</div>';
                }
                insightHTML += '</div></div>';
            }

            // Data quality / completeness notes
            if (dataNote) {
                insightHTML += '<div style="background:#FFF9E6;border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:11.5px;color:#6B5900;border-left:4px solid #F59E0B;line-height:1.5">'
                    + '<span style="font-weight:700">&#9888; Data Completeness:</span> ' + dataNote[1].trim() + '</div>';
            }

            // Confidence advisory
            if (confAdv) {
                insightHTML += '<div style="background:#FFF0F0;border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:11.5px;color:#7B0000;border-left:4px solid #D32F2F;line-height:1.5">'
                    + '<span style="font-weight:700">&#9888; Confidence Advisory:</span> ' + confAdv[1].trim() + '</div>';
            }

            // Recommendations block
            if (recMatch) {
                const recText = recMatch[1].trim();
                const recs = recText.split(/\(\d+\)\s*/).filter(Boolean);
                insightHTML += '<div style="background:linear-gradient(135deg,#F0FFF4,#E8F5E9);border-radius:8px;padding:14px 16px;margin-bottom:10px;border-left:4px solid #48A23F">'
                    + '<div style="font-size:10px;font-weight:700;color:#2E7D32;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Recommended Actions</div>'
                    + '<div style="display:flex;flex-direction:column;gap:6px">';
                recs.forEach(function(rec, idx) {
                    rec = rec.replace(/^\|?\s*/, '').trim();
                    if (rec) {
                        insightHTML += '<div style="display:flex;gap:8px;align-items:flex-start;font-size:11.5px;line-height:1.5;color:#1B4332">'
                            + '<span style="background:#48A23F;color:#fff;min-width:20px;height:20px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0">' + (idx+1) + '</span>'
                            + '<span>' + rec + '</span></div>';
                    }
                });
                insightHTML += '</div></div>';
            }

            // Clinical context from narrative
            if (clinCtx) {
                insightHTML += '<div style="background:#F0FFF4;border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:11.5px;color:#2D6A4F;border-left:4px solid #52B788;line-height:1.5">'
                    + '<span style="font-weight:700">Clinical Context:</span> ' + clinCtx[1].trim() + '</div>';
            }
        }

        // Clinical context from structured data (fallback)
        if (!narrative && clinical && typeof clinical === 'object' && Object.keys(clinical).length > 0) {
            const items = Object.entries(clinical).filter(([k,v]) => v && typeof v !== 'object').slice(0, 4);
            if (items.length) {
                insightHTML += '<div style="background:#F0FFF4;border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:11.5px;color:#2D6A4F;border-left:4px solid #52B788">'
                    + '<strong>Clinical Context:</strong> ' + items.map(([k,v]) => '<span style="font-weight:600">' + k.replace(/_/g,' ').replace(/\\b\\w/g,l=>l.toUpperCase()) + ':</span> ' + v).join(' &middot; ') + '</div>';
            }
        }

        // ── BENCHMARK — industry comparison ──
        const bench = data.benchmark || {};
        if (bench.has_benchmark) {
            const benchColor = bench.comparison === 'better' ? '#48A23F' : bench.comparison === 'worse' ? '#D32F2F' : '#006BA6';
            const benchIcon = bench.comparison === 'better' ? '&#9650;' : bench.comparison === 'worse' ? '&#9660;' : '&#9654;';
            insightHTML += '<div style="background:linear-gradient(135deg,#F8F9FF,#EEF2FF);border-radius:6px;padding:10px 14px;margin-bottom:10px;font-size:12px;border-left:3px solid ' + benchColor + '">'
                + '<strong style="color:' + benchColor + '">' + benchIcon + ' Industry Benchmark:</strong> '
                + bench.message
                + (bench.percentile ? ' <span style="background:' + benchColor + '22;color:' + benchColor + ';padding:2px 6px;border-radius:3px;font-size:10px;font-weight:600">' + bench.percentile + '</span>' : '')
                + '</div>';
        }

        // ── MULTI-REASONING — strategies + insights ──
        const reasoning = data.reasoning || {};
        const rInsights = reasoning.insights || [];
        const rRisks = reasoning.risk_flags || [];
        const rOpps = reasoning.opportunities || [];
        const rActions = reasoning.recommended_actions || [];
        const allReasoningItems = [...rInsights, ...rRisks, ...rOpps, ...rActions].filter(Boolean);
        if (allReasoningItems.length > 0) {
            const stratBadges = (reasoning.strategies_applied || []).map(s => '<span style="background:#EEF2FF;color:#4338CA;padding:2px 6px;border-radius:3px;font-size:9px;font-weight:600;text-transform:uppercase">' + s + '</span>').join(' ');
            insightHTML += '<div style="background:#FAFBFF;border-radius:6px;padding:10px 14px;margin-bottom:10px;font-size:12px;border:1px solid #E0E7FF">'
                + '<div style="display:flex;align-items:center;gap:6px;margin-bottom:6px"><strong style="color:#4338CA">Multi-Reasoning Analysis</strong> ' + stratBadges + '</div>'
                + '<ul style="margin:0 0 0 16px;padding:0;list-style:disc;color:var(--kp-gray-700)">';
            for (const item of allReasoningItems.slice(0, 4)) {
                const isRisk = rRisks.includes(item);
                const isOpp = rOpps.includes(item);
                const itemColor = isRisk ? '#D32F2F' : isOpp ? '#48A23F' : '#374151';
                const prefix = isRisk ? '<span style="color:#D32F2F;font-weight:600">Risk:</span> ' : isOpp ? '<span style="color:#48A23F;font-weight:600">Opportunity:</span> ' : '';
                insightHTML += '<li style="margin-bottom:4px;color:' + itemColor + '">' + prefix + item + '</li>';
            }
            insightHTML += '</ul></div>';
        }

        // ── DATA GAP WARNING ──
        const gaps = data.data_gaps || {};
        if (gaps.has_issues && gaps.issues && gaps.issues.length > 0) {
            const gapSevColors = {critical:'#D32F2F',warning:'#FF6B35',info:'#006BA6'};
            const topSev = gaps.severity || 'info';
            insightHTML += '<div style="background:' + (topSev==='critical'?'#FFEBEE':topSev==='warning'?'#FFF8E1':'#E3F2FD') + ';border-radius:6px;padding:10px 14px;margin-bottom:10px;font-size:12px;border-left:3px solid ' + (gapSevColors[topSev]||'#666') + '">'
                + '<strong style="color:' + (gapSevColors[topSev]||'#666') + '">Data Quality Notice:</strong>'
                + '<ul style="margin:4px 0 0 16px;padding:0;list-style:disc">'
                + gaps.issues.slice(0,3).map(g => '<li style="margin-bottom:2px">' + g.message + '</li>').join('')
                + '</ul>'
                + (gaps.data_quality_score != null ? '<div style="margin-top:4px;font-size:10px;color:var(--kp-gray-500)">Data Quality Score: ' + gaps.data_quality_score + '/100</div>' : '')
                + '</div>';
        }

        // ── DEEP DIVES — precursor follow-up analyses ──
        const dives = data.deep_dives || [];
        if (dives.length > 0) {
            insightHTML += '<div style="margin-bottom:10px">'
                + '<div style="font-size:11px;font-weight:700;color:var(--kp-blue);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">Precursor Deep Dives</div>'
                + '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px">';
            for (const dive of dives.slice(0, 4)) {
                const dc = dive.columns || [];
                const dr = (dive.rows || []).slice(0, 5);
                insightHTML += '<div style="background:#fff;border:1px solid var(--kp-gray-200);border-radius:6px;padding:8px 10px;font-size:11px">'
                    + '<div style="font-weight:600;color:var(--kp-blue);margin-bottom:4px">' + dive.label + '</div>'
                    + '<table style="width:100%;border-collapse:collapse;font-size:10px"><thead><tr>'
                    + dc.map(c => '<th style="text-align:left;padding:2px 4px;border-bottom:1px solid #eee;color:var(--kp-gray-500)">' + c.replace(/_/g,' ') + '</th>').join('')
                    + '</tr></thead><tbody>'
                    + dr.map(r => '<tr>' + r.map(v => {
                        const vf = typeof v === 'number' ? v.toLocaleString(undefined,{maximumFractionDigits:1}) : (v==null?'':v);
                        return '<td style="padding:2px 4px;border-bottom:1px solid #f5f5f5">' + vf + '</td>';
                    }).join('') + '</tr>').join('')
                    + '</tbody></table>'
                    + '<div style="color:var(--kp-gray-500);font-size:9px;margin-top:3px">' + (dive.row_count||0) + ' rows</div>'
                    + '</div>';
            }
            insightHTML += '</div></div>';
        }

        // ── PREDICTIVE INSIGHTS — Business-friendly model results ──
        const statModels = data.stat_models || [];
        if (statModels.length > 0) {
            // Map model types to business-friendly headers and icons
            const businessLabels = {
                monte_carlo: {icon:'📊', label:'Projected Range', desc:'Where this metric is likely headed'},
                bayesian: {icon:'🎯', label:'Confidence Estimate', desc:'How confident we are in this rate'},
                time_series: {icon:'📈', label:'Forecast', desc:'Projected trajectory based on historical pattern'},
                bootstrap: {icon:'📐', label:'Confidence Range', desc:'Statistical confidence interval for this metric'},
                regression: {icon:'🔗', label:'Relationship Found', desc:'Correlation between variables'},
                risk_score: {icon:'⚠', label:'Risk Distribution', desc:'How records are distributed across risk tiers'},
                distribution: {icon:'📉', label:'Distribution Pattern', desc:'How values are spread across the dataset'},
                what_if: {icon:'💡', label:'Impact Scenarios', desc:'What happens if key metrics change'},
                markov_chain: {icon:'🔄', label:'State Transitions', desc:'How statuses change over time'},
            };

            insightHTML += '<div style="margin-bottom:10px">'
                + '<div style="font-size:11px;font-weight:700;color:#7C3AED;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">Predictive Insights</div>'
                + '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px">';

            for (const model of statModels.slice(0, 4)) {
                const typeColors = {
                    monte_carlo:'#7C3AED', bayesian:'#2563EB', time_series:'#059669',
                    markov_chain:'#DC2626', bootstrap:'#D97706', regression:'#4338CA',
                    risk_score:'#DC2626', distribution:'#7C3AED', what_if:'#EA580C'
                };
                const color = typeColors[model.type] || '#6B7280';
                const biz = businessLabels[model.type] || {icon:'📊', label: model.type, desc:''};

                let modelBody = '<div style="font-size:11px;color:var(--kp-gray-700);line-height:1.5">' + (model.message||'') + '</div>';

                // Business-friendly extra details
                if (model.type === 'monte_carlo' && model.confidence_interval_90) {
                    const ci = model.confidence_interval_90;
                    modelBody += '<div style="margin-top:6px;display:flex;gap:8px;font-size:10px">'
                        + '<div style="background:#F5F3FF;padding:4px 8px;border-radius:4px"><strong>Expected Range:</strong> ' + Number(ci[0]).toLocaleString() + ' – ' + Number(ci[1]).toLocaleString() + '</div>'
                        + '<div style="background:' + (model.prob_decline > 50 ? '#FEE2E2' : '#DCFCE7') + ';padding:4px 8px;border-radius:4px"><strong>Downside Risk:</strong> ' + model.prob_decline + '%</div>'
                        + '</div>';
                }
                if (model.type === 'regression' && model.significant) {
                    modelBody += '<div style="margin-top:6px;font-size:10px;background:#ECFDF5;padding:4px 8px;border-radius:4px;color:#166534">'
                        + '<strong>Confirmed relationship</strong> — ' + (model.equation||'') + ' (explains ' + ((model.r_squared||0)*100).toFixed(0) + '% of variation)'
                        + '</div>';
                } else if (model.type === 'regression' && !model.significant) {
                    modelBody += '<div style="margin-top:6px;font-size:10px;background:#F9FAFB;padding:4px 8px;border-radius:4px;color:#6B7280">'
                        + 'No statistically significant relationship found between these variables.'
                        + '</div>';
                }
                if (model.type === 'bayesian' && model.credible_interval_95) {
                    const ci95 = model.credible_interval_95;
                    modelBody += '<div style="margin-top:6px;font-size:10px;background:#EFF6FF;padding:4px 8px;border-radius:4px">'
                        + '<strong>True rate is 95% likely between</strong> ' + ci95[0] + '% and ' + ci95[1] + '%'
                        + '</div>';
                }
                if (model.type === 'what_if' && model.scenarios) {
                    modelBody += '<div style="margin-top:6px">';
                    for (const sc of model.scenarios.slice(0,3)) {
                        const impColor = sc.impact_pct > 0 ? '#059669' : '#DC2626';
                        modelBody += '<div style="font-size:10px;display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #f5f5f5">'
                            + '<span>' + sc.scenario + '</span>'
                            + '<span style="color:' + impColor + ';font-weight:600">' + (sc.impact_pct>0?'+':'') + sc.impact_pct + '% → ' + sc.projected.toLocaleString() + '</span>'
                            + '</div>';
                    }
                    modelBody += '</div>';
                }
                if (model.type === 'risk_score' && model.distribution) {
                    const rd = model.distribution;
                    const total = (rd.critical||0) + (rd.high||0) + (rd.medium||0) + (rd.low||0);
                    const critPct = total > 0 ? ((rd.critical||0)/total*100).toFixed(0) : 0;
                    const highPct = total > 0 ? ((rd.high||0)/total*100).toFixed(0) : 0;
                    modelBody += '<div style="margin-top:6px;display:flex;gap:6px;font-size:10px;flex-wrap:wrap">'
                        + '<span style="background:#FEE2E2;color:#991B1B;padding:2px 8px;border-radius:3px">Critical: ' + (rd.critical||0) + ' (' + critPct + '%)</span>'
                        + '<span style="background:#FFEDD5;color:#9A3412;padding:2px 8px;border-radius:3px">High: ' + (rd.high||0) + ' (' + highPct + '%)</span>'
                        + '<span style="background:#FEF9C3;color:#854D0E;padding:2px 8px;border-radius:3px">Medium: ' + (rd.medium||0) + '</span>'
                        + '<span style="background:#DCFCE7;color:#166534;padding:2px 8px;border-radius:3px">Low: ' + (rd.low||0) + '</span>'
                        + '</div>';
                }
                if (model.type === 'time_series' && model.forecast) {
                    const trend = model.trend_description || '';
                    const trendColor = trend.includes('increas') || trend.includes('upward') ? '#DC2626' : (trend.includes('decreas') || trend.includes('downward') ? '#059669' : '#6B7280');
                    modelBody += '<div style="margin-top:6px;font-size:10px;background:#ECFDF5;padding:4px 8px;border-radius:4px">'
                        + '<strong>Next periods:</strong> ' + model.forecast.map(v => Number(v).toLocaleString()).join(' → ')
                        + ' &nbsp; <span style="color:' + trendColor + ';font-weight:600">' + trend + '</span>'
                        + '</div>';
                }
                // Skip raw distribution stats (skew/kurtosis) — not business-meaningful
                // Instead show a simpler insight for distribution
                if (model.type === 'distribution' && model.statistics) {
                    const st = model.statistics;
                    const skew = parseFloat(st.skewness || 0);
                    let distInsight = 'Values are evenly distributed.';
                    if (skew > 1) distInsight = 'Distribution is right-skewed — most values are low with some high outliers.';
                    else if (skew < -1) distInsight = 'Distribution is left-skewed — most values are high with some low outliers.';
                    else if (st.iqr) distInsight = 'Typical range: ' + (st.q1||'') + ' to ' + (st.q3||'') + ' (middle 50% of values).';
                    modelBody += '<div style="margin-top:6px;font-size:10px;background:#F3F4F6;padding:4px 8px;border-radius:4px">'
                        + distInsight + '</div>';
                }

                insightHTML += '<div style="background:#FAFAFE;border:1px solid #E5E7EB;border-radius:6px;padding:10px 12px;border-top:3px solid ' + color + '">'
                    + '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">'
                    + '<span style="font-size:14px">' + biz.icon + '</span>'
                    + '<span style="font-size:11px;font-weight:700;color:var(--kp-gray-800)">' + biz.label + '</span>'
                    + '</div>'
                    + (biz.desc ? '<div style="font-size:9px;color:var(--kp-gray-500);margin-bottom:6px">' + biz.desc + '</div>' : '')
                    + modelBody
                    + '</div>';
            }
            insightHTML += '</div></div>';
        }

        // ── KPI Card (single value) ──
        if (data.rows?.length === 1 && data.columns?.length <= 3) {
            const val = data.rows[0][0];
            const formatted = typeof val === 'number' ? val.toLocaleString(undefined, {maximumFractionDigits: 2}) : val;
            const label = data.columns?.[0]?.replace(/_/g, ' ') || 'Result';
            let extras = '';
            for (let i = 1; i < (data.columns||[]).length; i++) {
                const v2 = data.rows[0][i];
                if (v2 != null) {
                    const f2 = typeof v2 === 'number' ? v2.toLocaleString(undefined, {maximumFractionDigits: 2}) : v2;
                    extras += '<div style="font-size:14px;color:var(--kp-gray-500);margin-top:3px">' + data.columns[i].replace(/_/g,' ') + ': ' + f2 + '</div>';
                }
            }
            bodyHTML = insightHTML + '<div style="text-align:center;padding:16px">'
                + '<div style="font-size:38px;font-weight:700;color:var(--kp-blue)">' + formatted + '</div>'
                + '<div style="font-size:13px;color:var(--kp-gray-500);margin-top:6px">' + label + '</div>'
                + extras + '</div>';
        }
        // ── Table + Chart ──
        else if (data.rows?.length > 0) {
            const showChart = data.rows.length <= 30 && data.rows.length > 1;
            bodyHTML = insightHTML;
            if (showChart) bodyHTML += '<canvas id="' + chartId + '" class="chart-canvas" style="margin-bottom:14px"></canvas>';

            const maxShow = Math.min(data.rows.length, 50);
            const cols = data.columns || [];
            const tableId = 'tbl_' + Date.now();
            bodyHTML += '<div style="overflow-x:auto"><table class="data-table" id="' + tableId + '">'
                + '<thead><tr>' + cols.map((c,ci) =>
                    '<th onclick="sortTable(\\'' + tableId + '\\',' + ci + ')" data-col="' + ci + '">' + c.replace(/_/g,' ') + ' <span class="sort-arrow">&#9650;</span></th>'
                ).join('') + '</tr></thead>'
                + '<tbody>' + data.rows.slice(0, maxShow).map(row =>
                    '<tr>' + row.map(v => {
                        const isNum = typeof v === 'number';
                        const fmt = isNum ? v.toLocaleString(undefined, {maximumFractionDigits: 2}) : (v == null ? '' : v);
                        return '<td class="' + (isNum ? 'numeric' : '') + '">' + fmt + '</td>';
                    }).join('') + '</tr>'
                ).join('') + '</tbody></table></div>';
            if (data.rows.length > maxShow) {
                bodyHTML += '<div style="text-align:center;padding:8px;color:var(--kp-gray-500);font-size:11px">Showing ' + maxShow + ' of ' + data.rows.length + ' rows</div>';
            }
        } else {
            bodyHTML = insightHTML + '<div style="text-align:center;padding:16px;color:var(--kp-gray-500)">No results found</div>';
        }

        // ── Follow-up suggestions (dynamic, context-aware) ──
        let followUpHTML = '';
        if (data.suggestions && data.suggestions.length > 0) {
            followUpHTML = '<div style="padding:10px 14px;border-top:1px solid var(--kp-gray-200);background:linear-gradient(135deg,#FAFBFF,#F5F7FF)">'
                + '<div style="font-size:9px;font-weight:700;color:var(--kp-blue);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px">Explore Further — Related Questions</div>'
                + '<div style="display:flex;flex-wrap:wrap;gap:5px">'
                + data.suggestions.slice(0, 6).map(s =>
                    '<button class="suggestion-chip" style="font-size:10.5px;padding:4px 12px;background:var(--kp-blue-pale);border:1px solid #D0D9E8;border-radius:16px;line-height:1.4" onclick="askThis(\\'' + s.replace(/'/g, '') + '\\')">' + s + '</button>'
                ).join('') + '</div></div>';
        }

        // ── SQL Editor (visible, editable, with Run button) ──
        let sqlSection = '';
        if (data.sql) {
            sqlSection = '<div class="sql-editor-area">'
                + '<div class="sql-toolbar">'
                + '<span style="font-size:11px;font-weight:600;color:var(--kp-gray-700)">Generated SQL</span>'
                + '<span class="sql-readonly-badge">READ ONLY</span>'
                + '<span style="flex:1"></span>'
                + '<button class="sql-toolbar-btn" onclick="copySql(\\'' + editorId + '\\')">Copy</button>'
                + '<button class="sql-toolbar-btn run" onclick="editAndRun(\\'' + editorId + '\\')">Edit &amp; Run</button>'
                + '</div>'
                + '<textarea id="' + editorId + '" class="sql-editor" rows="3">' + data.sql.replace(/</g,'&lt;') + '</textarea>'
                + '<div id="' + editorId + '_result"></div>';

            // Why this SQL? reasoning chain
            const whyId = 'why_' + Date.now();
            const explanation = data.explanation || '';
            const semanticIntent = data.semantic_intent || data.intent || '';
            const semanticColumns = data.semantic_columns || data.columns_resolved || [];
            const tablesUsed = data.tables_used || [];
            const strategy = data.strategy || '';
            const filters = data.filters || [];

            let whyHTML = '<div style="margin-top:6px;">';
            whyHTML += '<button onclick="document.getElementById(\\'' + whyId + '\\').style.display=document.getElementById(\\'' + whyId + '\\').style.display===\\'none\\'?\\'block\\':\\'none\\'" style="background:none;border:1px solid #D0E0F0;border-radius:6px;padding:4px 12px;font-size:11px;color:#0066CC;cursor:pointer;font-weight:600">&#128161; Why this SQL?</button>';
            whyHTML += '<div id="' + whyId + '" style="display:none;margin-top:8px;background:linear-gradient(135deg,#F8FAFF,#F0F4FF);border-radius:8px;padding:14px 16px;border-left:4px solid #0066CC;font-size:12px;line-height:1.8">';
            whyHTML += '<div style="font-size:10px;font-weight:700;color:#004B87;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Query Reasoning Chain</div>';
            if (semanticIntent) whyHTML += '<div><strong>Intent Detected:</strong> ' + semanticIntent + '</div>';
            if (strategy) whyHTML += '<div><strong>Strategy:</strong> ' + strategy + '</div>';
            if (tablesUsed.length) whyHTML += '<div><strong>Tables Selected:</strong> ' + tablesUsed.join(' → ') + '</div>';
            if (semanticColumns.length) {
                whyHTML += '<div><strong>Columns Matched:</strong> ';
                const cols = semanticColumns.slice(0, 5).map(c => typeof c === 'string' ? c : (c.column || c.name || JSON.stringify(c)));
                whyHTML += cols.join(', ');
                whyHTML += '</div>';
            }
            if (explanation) whyHTML += '<div style="margin-top:6px;color:#374151">' + explanation + '</div>';
            whyHTML += '</div></div>';
            sqlSection += whyHTML + '</div>';
        }

        // ── Insights Panel (statistical, probabilistic, ML analysis) ──
        let insightsSection = '';
        const insights = data.insights || {};
        if (insights && Object.keys(insights).length > 0) {
            const insId = 'insights_' + Date.now();
            insightsSection = '<div style="margin:0 14px 8px;">';
            insightsSection += '<button onclick="document.getElementById(\\'' + insId + '\\').style.display=document.getElementById(\\'' + insId + '\\').style.display===\\'none\\'?\\'block\\':\\'none\\'" style="background:none;border:1px solid #D0E8D0;border-radius:6px;padding:4px 12px;font-size:11px;color:#2E7D32;cursor:pointer;font-weight:600">&#128202; AI Insights</button>';
            insightsSection += '<div id="' + insId + '" style="display:none;margin-top:8px;background:linear-gradient(135deg,#F4FAF4,#E8F5E9);border-radius:8px;padding:14px 16px;border-left:4px solid #2E7D32;font-size:12px;line-height:1.8">';
            insightsSection += '<div style="font-size:10px;font-weight:700;color:#1B5E20;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Intelligent Analysis</div>';

            // Statistical summary
            const stats = insights.statistical_summary || {};
            if (Object.keys(stats).length > 0) {
                insightsSection += '<div style="margin-bottom:6px"><strong>Statistical Summary:</strong> ';
                const statKeys = Object.keys(stats).slice(0, 3);
                statKeys.forEach(k => {
                    const sv = stats[k];
                    if (sv && typeof sv === 'object' && sv.mean !== undefined) {
                        insightsSection += k + ' (mean: ' + (typeof sv.mean === 'number' ? sv.mean.toFixed(2) : sv.mean) + ') ';
                    }
                });
                insightsSection += '</div>';
            }

            // Anomalies
            const insightAnomalies = insights.anomalies || [];
            if (insightAnomalies.length > 0) {
                insightsSection += '<div style="margin-bottom:6px;color:#E65100"><strong>Anomalies Detected:</strong> ' + insightAnomalies.length + ' data points flagged</div>';
            }

            // Patterns
            const patterns = insights.patterns || [];
            if (patterns.length > 0) {
                insightsSection += '<div style="margin-bottom:6px"><strong>Patterns:</strong> ';
                patterns.slice(0, 3).forEach(p => {
                    insightsSection += (typeof p === 'string' ? p : (p.description || JSON.stringify(p)).substring(0, 80)) + ' ';
                });
                insightsSection += '</div>';
            }

            // Recommendations
            const recs = insights.recommendations || [];
            if (recs.length > 0) {
                insightsSection += '<div style="margin-top:8px;padding:8px 12px;background:rgba(255,255,255,0.7);border-radius:6px"><strong style="color:#1B5E20">Recommendations:</strong>';
                recs.slice(0, 3).forEach(r => {
                    insightsSection += '<div style="margin-top:4px;padding-left:12px;border-left:2px solid #4CAF50;font-size:11px">' + (typeof r === 'string' ? r : JSON.stringify(r)) + '</div>';
                });
                insightsSection += '</div>';
            }

            // Quality score
            const qs = insights.quality_score;
            if (qs !== undefined) {
                insightsSection += '<div style="margin-top:6px;font-size:10px;color:#666">Analysis Quality: <strong style="color:' + (qs > 0.7 ? '#2E7D32' : qs > 0.4 ? '#F57F17' : '#C62828') + '">' + (qs * 100).toFixed(0) + '%</strong></div>';
            }

            // Narrative
            const insNarrative = insights.narrative || '';
            if (insNarrative) {
                insightsSection += '<div style="margin-top:8px;font-style:italic;color:#374151;font-size:11px">' + String(insNarrative).substring(0, 300) + '</div>';
            }

            insightsSection += '</div></div>';
        }

        // ── Context bar (strategy, latency, row count, confidence, cache) ──
        let contextBar = '<div style="padding:6px 14px;border-top:1px solid var(--kp-gray-200);display:flex;flex-wrap:wrap;gap:12px;font-size:10px;color:var(--kp-gray-500);background:var(--kp-gray-100)">';
        if (strategy) contextBar += '<span>Strategy: ' + strategy + '</span>';
        if (latency) contextBar += '<span>Latency: ' + latency + 'ms</span>';
        contextBar += '<span>Rows: ' + (data.row_count || 0) + '</span>';
        if (data.intent) contextBar += '<span>Intent: ' + data.intent + '</span>';
        if (grade) contextBar += '<span>Grade: <strong style="color:' + (gradeColors[grade]||'#666') + '">' + grade + '</strong></span>';
        if (cacheHit) contextBar += '<span style="color:var(--kp-teal);font-weight:600">CACHED</span>';
        if (data.source) contextBar += '<span>Source: ' + data.source + '</span>';
        if (anomalies.length) contextBar += '<span style="color:var(--kp-orange)">' + anomalies.length + ' anomalies</span>';
        contextBar += '</div>';

        // Store per-card data for download
        const cardId = 'card_' + (++_cardCounter);
        _cardResults[cardId] = {columns: data.columns || [], rows: data.rows || [], question: question};

        const hasTableData = data.columns && data.rows && data.rows.length > 0;
        const dlBtn = hasTableData
            ? '<button data-perm="export_csv" onclick="downloadCardCSV(\\'' + cardId + '\\')" style="padding:2px 8px;font-size:10px;border-radius:4px;border:1px solid var(--kp-blue);background:var(--kp-blue-pale);cursor:pointer;color:var(--kp-blue);font-weight:500" title="Download CSV">&#11015; Download</button>'
            : '';
        const emBtn = hasTableData
            ? '<button data-perm="export_email" onclick="emailCardResults(\\'' + cardId + '\\')" style="padding:2px 8px;font-size:10px;border-radius:4px;border:1px solid var(--kp-teal);background:#E0F7FA;cursor:pointer;color:#00838F;font-weight:500" title="Email results">&#9993; Email</button>'
            : '';

        const collapseClass = startCollapsed ? ' collapsed' : '';
        const collapseIcon = startCollapsed ? '&#9660;' : '&#9650;';
        const collapseBtnClass = startCollapsed ? 'collapse-btn' : 'collapse-btn open';
        const hiddenStyle = startCollapsed ? ' style="display:none"' : '';

        return '<div class="result-card" data-chart-id="' + chartId + '">'
            + '<div class="result-header" style="cursor:pointer" onclick="toggleCard(this.querySelector(\\'button.collapse-btn\\'))">'
            + '<button class="' + collapseBtnClass + '" onclick="event.stopPropagation();toggleCard(this)">' + collapseIcon + '</button>'
            + '<span style="color:var(--kp-blue);font-weight:600">Q:</span>'
            + '<span class="query-text">' + question + '</span>'
            + '<div style="margin-left:auto;display:flex;gap:4px;align-items:center">'
            + dlBtn + emBtn
            + (grade ? '<span class="badge" style="background:' + (gradeColors[grade]||'#666') + ';color:#fff;font-weight:700">' + grade + '</span>' : '')
            + '<span class="badge badge-blue">' + (data.intent || '') + '</span>'
            + '<span class="badge badge-green">' + (data.row_count || 0) + ' rows</span>'
            + (cacheHit ? '<span class="badge" style="background:var(--kp-teal);color:#fff">CACHED</span>' : '')
            + (anomalies.length ? '<span class="badge" style="background:var(--kp-orange);color:#fff">' + anomalies.length + ' anomalies</span>' : '')
            + '</div></div>'
            + '<div class="result-body' + collapseClass + '">' + bodyHTML + '</div>'
            + (startCollapsed ? '' : followUpHTML)
            + (startCollapsed ? '' : sqlSection)
            + (startCollapsed ? '' : insightsSection)
            + (startCollapsed ? '' : contextBar)
            + '</div>';
    }

    // ── SQL Editor actions ──
    function copySql(editorId) {
        const el = document.getElementById(editorId);
        if (el) { navigator.clipboard.writeText(el.value); }
    }

    async function editAndRun(editorId) {
        const editor = document.getElementById(editorId);
        const resultDiv = document.getElementById(editorId + '_result');
        if (!editor || !resultDiv) return;
        const sql = editor.value.trim();
        if (!sql) return;

        resultDiv.innerHTML = '<div class="loading" style="padding:12px"><div class="spinner"></div> Running...</div>';
        try {
            const resp = await authFetch(API + '/run-sql', {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify({ sql }),
            });
            const data = await resp.json();
            if (data.error) {
                resultDiv.innerHTML = '<div style="padding:10px;color:var(--kp-red);font-size:12px">Error: ' + data.error + '</div>';
                return;
            }
            const cols = data.columns || [];
            const rows = (data.rows || []).slice(0, 100);
            const tid = 'rtbl_' + Date.now();
            resultDiv.innerHTML = '<div style="padding:10px;font-size:11px;color:var(--kp-green)">Query returned ' + data.row_count + ' rows' + (data.truncated ? ' (showing first 500)' : '') + '</div>'
                + '<div style="overflow-x:auto;max-height:300px;overflow-y:auto;padding:0 10px 10px">'
                + '<table class="data-table" id="' + tid + '"><thead><tr>'
                + cols.map((c,ci) => '<th onclick="sortTable(\\'' + tid + '\\',' + ci + ')">' + c.replace(/_/g,' ') + ' <span class="sort-arrow">&#9650;</span></th>').join('')
                + '</tr></thead><tbody>'
                + rows.map(row => '<tr>' + row.map(v => {
                    const isNum = typeof v === 'number';
                    return '<td class="' + (isNum ? 'numeric' : '') + '">' + (isNum ? v.toLocaleString(undefined,{maximumFractionDigits:2}) : (v==null?'':v)) + '</td>';
                }).join('') + '</tr>').join('')
                + '</tbody></table></div>';
        } catch(e) {
            resultDiv.innerHTML = '<div style="padding:10px;color:var(--kp-red);font-size:12px">Error: ' + e.message + '</div>';
        }
    }

    // ── Table sorting ──
    function sortTable(tableId, colIndex) {
        const table = document.getElementById(tableId);
        if (!table) return;
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const th = table.querySelectorAll('th')[colIndex];
        const asc = !th.classList.contains('sorted') || th.getAttribute('data-dir') === 'desc';

        // Reset all headers
        table.querySelectorAll('th').forEach(h => { h.classList.remove('sorted'); h.setAttribute('data-dir',''); });
        th.classList.add('sorted');
        th.setAttribute('data-dir', asc ? 'asc' : 'desc');
        th.querySelector('.sort-arrow').innerHTML = asc ? '&#9650;' : '&#9660;';

        rows.sort((a, b) => {
            let va = a.cells[colIndex]?.textContent.trim() || '';
            let vb = b.cells[colIndex]?.textContent.trim() || '';
            // Try numeric comparison
            const na = parseFloat(va.replace(/[,$%]/g, ''));
            const nb = parseFloat(vb.replace(/[,$%]/g, ''));
            if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
            return asc ? va.localeCompare(vb) : vb.localeCompare(va);
        });
        rows.forEach(r => tbody.appendChild(r));
    }

    // ── Chart rendering ──
    function renderChart(canvasId, data) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !data.rows || data.rows.length < 2) return;
        const labels = data.rows.map(r => r[0]);
        const values = data.rows.map(r => typeof r[1] === 'number' ? r[1] : r[0]);
        const chartType = data.dashboard?.chart_type || 'bar';
        const colors = ['#004B87','#00A3B5','#FF6B35','#48A23F','#D32F2F','#7C3AED','#EC4899','#F59E0B','#6366F1','#14B8A6'];
        let type = chartType === 'donut' ? 'doughnut' : (chartType === 'line' || chartType === 'area') ? 'line' : 'bar';
        const config = { type, data: { labels, datasets: [{ data: values, backgroundColor: type === 'doughnut' ? colors : colors[0]+'CC', borderColor: type==='line' ? colors[0] : undefined, fill: chartType==='area', tension:0.3, borderWidth: type==='line'?2:0, borderRadius: type==='bar'?4:0 }] }, options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{display:type==='doughnut'}}, scales: type==='doughnut' ? {} : { y:{beginAtZero:true,grid:{color:'#E8EBF0'}}, x:{grid:{display:false}} }, indexAxis: chartType==='horizontal_bar' ? 'y' : 'x' } };
        if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
        chartInstances[canvasId] = new Chart(canvas, config);
    }

    // ── Dashboard KPI formatting ──
    function fmtKPI(val, unit) {
        if (val == null || val === undefined) return '—';
        if (typeof val !== 'number') {
            if (Array.isArray(val) && val.length > 0) {
                const first = val[0];
                if (typeof first === 'object') {
                    const numKey = Object.keys(first).find(k => typeof first[k] === 'number');
                    if (numKey) return fmtKPI(first[numKey], unit) + ' (top)';
                }
                return val.length + ' items';
            }
            if (Array.isArray(val)) return '—';
            return String(val);
        }
        if (unit === '$') {
            if (Math.abs(val) >= 1e6) return '$' + (val/1e6).toFixed(1) + 'M';
            if (Math.abs(val) >= 1e3) return '$' + (val/1e3).toFixed(1) + 'K';
            return '$' + val.toFixed(2);
        }
        if (unit === '%') return val.toFixed(1) + '%';
        if (unit === 'days') return val.toFixed(1) + ' days';
        if (unit === 'ratio') return val.toFixed(3);
        if (val >= 1e6) return (val/1e6).toFixed(1) + 'M';
        if (val >= 1e3) return Math.round(val).toLocaleString();
        return val.toLocaleString();
    }

    function kpiCard(kpi) {
        const val = kpi.value;
        const unit = kpi.unit || '';
        const alert = kpi.alert;
        const name = kpi.kpi || '';
        const desc = kpi.description || '';
        const catColors = {revenue:'#286140', retention:'#006BA6', acquisition:'#007A7C', operations:'#1B4F9B'};
        let valColor = catColors[kpi.category] || '#006BA6';
        let alertHtml = '';
        if (alert) { valColor = '#C8102E'; alertHtml = '<span style="font-size:9px;background:#FEF3CD;color:#856404;padding:2px 5px;border-radius:3px;margin-left:4px">ALERT</span>'; }
        if (kpi.error) valColor = '#58595B';
        const displayVal = kpi.error ? 'Error' : fmtKPI(val, unit);
        const safeName = name.replace(/'/g, '');
        return '<div class="kpi-card" style="border-left-color:' + valColor + '" onclick="deepDiveKPI(\\'' + safeName + '\\')">'
            + '<div class="kpi-label">' + name + alertHtml + '</div>'
            + '<div class="kpi-value" style="color:' + valColor + '">' + displayVal + '</div>'
            + '<div style="font-size:10px;color:var(--kp-gray-500);margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + desc + '">' + desc + '</div>'
            + '</div>';
    }

    async function loadDashboard() {
        if (activeCustomDash) return; // Custom dashboard is active
        ['kpiRevenue','kpiClinical','kpiOperations'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = '<div class="kpi-card"><div class="kpi-label" style="color:var(--kp-gray-300)">Loading...</div><div class="kpi-value" style="color:var(--kp-gray-200)">—</div></div>'.repeat(3);
        });
        const [kpiResp, statusChart, trendChart, regionChart, visitChart] = await Promise.all([
            authFetch(API + '/kpis').then(r => r.json()).catch(() => null),
            authFetch(API + '/kpi-charts?chart=claim_status').then(r => r.json()).catch(() => null),
            authFetch(API + '/kpi-charts?chart=monthly_trend').then(r => r.json()).catch(() => null),
            authFetch(API + '/kpi-charts?chart=revenue_by_region').then(r => r.json()).catch(() => null),
            authFetch(API + '/kpi-charts?chart=visit_type_cost').then(r => r.json()).catch(() => null),
        ]);
        if (kpiResp && kpiResp.by_category) {
            const cats = kpiResp.by_category;
            const revEl = document.getElementById('kpiRevenue');
            if (revEl && cats.revenue) revEl.innerHTML = cats.revenue.map(k => kpiCard(k)).join('');
            const clinEl = document.getElementById('kpiClinical');
            if (clinEl && cats.retention) clinEl.innerHTML = cats.retention.map(k => kpiCard(k)).join('');
            const opsEl = document.getElementById('kpiOperations');
            if (opsEl) opsEl.innerHTML = [...(cats.operations||[]),...(cats.acquisition||[])].map(k => kpiCard(k)).join('');
            const banner = document.getElementById('alertBanner');
            const alerts = kpiResp.alerts || [];
            if (alerts.length) { banner.style.display='block'; banner.innerHTML='<strong>&#9888; '+alerts.length+' Alert'+(alerts.length>1?'s':'')+':</strong> '+alerts.map(a=>a.kpi+': '+(a.alert||'')).join(' &bull; '); }
            else banner.style.display='none';
        }
        if (statusChart?.rows) renderDashChart('chartClaimStatus', statusChart, 'doughnut');
        if (trendChart?.rows) renderDashChart('chartTrend', trendChart, 'line');
        if (regionChart?.rows) renderDashChart('chartRegionRevenue', regionChart, 'bar');
        if (visitChart?.rows) renderDashChart('chartVisitCost', visitChart, 'bar');
        // Load saved custom dashboards
        loadCustomDashList();
    }

    function renderDashChart(canvasId, data, type) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !data.rows) return;
        const labels = data.rows.map(r => String(r[0]).substring(0,18));
        const valIdx = data.columns ? Math.max(1, data.columns.findIndex(c => /count|total|revenue|amount|cost|avg|sum/i.test(c))) : 1;
        const values = data.rows.map(r => r[valIdx >= 0 ? valIdx : 1]);
        const colors = ['#004B87','#00A3B5','#FF6B35','#48A23F','#D32F2F','#7C3AED','#EC4899','#F59E0B','#1B4F9B','#007A7C'];
        if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
        chartInstances[canvasId] = new Chart(canvas, {
            type: type==='doughnut'?'doughnut':type==='line'?'line':'bar',
            data: { labels, datasets: [{ label: data.columns?.[valIdx]||'Value', data: values, backgroundColor: type==='doughnut'?colors:colors[0]+'CC', borderColor:type==='line'?colors[0]:undefined, fill:type==='line', tension:0.3, borderWidth:type==='line'?2:0, borderRadius:type==='bar'?4:0 }] },
            options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{display:type==='doughnut',position:'bottom'}}, scales:type==='doughnut'?{}:{y:{beginAtZero:true,grid:{color:'#E8EBF0'}},x:{grid:{display:false},ticks:{maxRotation:45}}} }
        });
    }

    // ── KPI Deep Dive (stays on dashboard page) ──
    async function deepDiveKPI(kpiName) {
        const panel = document.getElementById('kpiDeepDivePanel');
        const title = document.getElementById('kpiDeepDiveTitle');
        const body = document.getElementById('kpiDeepDiveBody');
        panel.style.display = 'block';
        title.textContent = kpiName + ' — Detail';
        body.innerHTML = '<div class="loading"><div class="spinner"></div> Loading details...</div>';
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Build a meaningful deep dive query based on KPI name
        const kn = kpiName.toLowerCase();
        let query = kpiName + ' by region';
        if (kn.includes('denial') || kn.includes('denied')) query = 'denial rate by plan type and region';
        else if (kn.includes('readmission')) query = 'readmission rate by provider specialty';
        else if (kn.includes('revenue')) query = 'total revenue by region';
        else if (kn.includes('cost') || kn.includes('encounter')) query = 'average cost per encounter by visit type';
        else if (kn.includes('utilization')) query = 'member utilization rate by region';
        else if (kn.includes('member') || kn.includes('growth')) query = 'member count by region and plan type';
        else if (kn.includes('provider')) query = 'provider count by specialty';
        else if (kn.includes('coverage') || kn.includes('service')) query = 'provider count by specialty';
        else if (kn.includes('appointment') || kn.includes('no-show') || kn.includes('noshow')) query = 'appointment no-show rate by department';
        else if (kn.includes('prescri')) query = 'prescription count by top 10 drugs';
        else if (kn.includes('referral')) query = 'referral count by specialty';

        try {
            const resp = await authFetch(API + '/intelligent/query', {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify({ question: query, session_id: sessionId }),
            });
            const data = await resp.json();
            let html = '';

            // Insight
            if (data.explanation) {
                html += '<div style="background:var(--kp-blue-pale);border-radius:6px;padding:10px 14px;margin-bottom:12px;font-size:12px;color:var(--kp-gray-700)">'
                    + '<strong style="color:var(--kp-blue)">Insight:</strong> ' + data.explanation + '</div>';
            }

            // Chart
            const drillChartId = 'drillChart_' + Date.now();
            if (data.rows && data.rows.length > 1 && data.rows.length <= 30) {
                html += '<canvas id="' + drillChartId + '" style="max-height:280px;margin-bottom:14px"></canvas>';
            }

            // Table with sortable columns
            if (data.rows && data.rows.length > 0) {
                const cols = data.columns || [];
                const rows = data.rows.slice(0, 50);
                const tid = 'drilltbl_' + Date.now();
                html += '<div style="overflow-x:auto;max-height:400px;overflow-y:auto">'
                    + '<table class="data-table" id="' + tid + '"><thead><tr>'
                    + cols.map((c, ci) => '<th onclick="sortTable(\\'' + tid + '\\',' + ci + ')">' + c.replace(/_/g, ' ') + ' <span class="sort-arrow">&#9650;</span></th>').join('')
                    + '</tr></thead><tbody>'
                    + rows.map(row => '<tr>' + row.map(v => {
                        const n = typeof v === 'number';
                        return '<td class="' + (n ? 'numeric' : '') + '">' + (n ? v.toLocaleString(undefined, {maximumFractionDigits: 2}) : (v == null ? '' : v)) + '</td>';
                    }).join('') + '</tr>').join('')
                    + '</tbody></table></div>';
                html += '<div style="font-size:10px;color:var(--kp-gray-500);margin-top:6px">' + (data.row_count || rows.length) + ' rows &middot; ' + (data.latency_ms || '') + 'ms</div>';
            }

            // SQL (editable)
            if (data.sql) {
                const edId = 'drillEd_' + Date.now();
                html += '<div style="margin-top:10px;border-top:1px solid var(--kp-gray-200);padding-top:8px">'
                    + '<div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">'
                    + '<span style="font-size:11px;font-weight:600;color:var(--kp-gray-700)">SQL</span>'
                    + '<span class="sql-readonly-badge">READ ONLY</span>'
                    + '<button class="sql-toolbar-btn run" onclick="editAndRun(\\'' + edId + '\\')">Edit &amp; Run</button>'
                    + '</div>'
                    + '<textarea id="' + edId + '" class="sql-editor" style="border:1px solid var(--kp-gray-200);border-radius:4px;min-height:50px">' + data.sql.replace(/</g, '&lt;') + '</textarea>'
                    + '<div id="' + edId + '_result"></div>'
                    + '</div>';
            }

            body.innerHTML = html;

            // Render chart
            if (data.rows && data.rows.length > 1 && data.rows.length <= 30) {
                setTimeout(() => renderChart(drillChartId, data), 100);
            }
        } catch (e) {
            body.innerHTML = '<div style="color:var(--kp-red);padding:10px">Error loading details: ' + e.message + '</div>';
        }
    }

    // ── Custom Dashboards ──
    async function loadCustomDashList() {
        try {
            const resp = await authFetch(API + '/custom-dashboards');
            customDashboards = await resp.json();
            if (!Array.isArray(customDashboards)) customDashboards = [];
        } catch(e) { customDashboards = []; }
        renderDashTabs();
    }

    function renderDashTabs() {
        const bar = document.getElementById('dashTabs');
        bar.innerHTML = '';
        // Default tab
        const defBtn = document.createElement('button');
        defBtn.className = 'custom-dash-tab' + (!activeCustomDash ? ' active' : '');
        defBtn.textContent = 'Executive KPIs';
        defBtn.onclick = () => switchDash('default');
        bar.appendChild(defBtn);
        // Custom tabs
        customDashboards.forEach(d => {
            const btn = document.createElement('button');
            btn.className = 'custom-dash-tab' + (activeCustomDash === d.id ? ' active' : '');
            btn.textContent = d.name;
            btn.onclick = () => switchDash(d.id);
            bar.appendChild(btn);
        });
        // Add new button
        const addBtn = document.createElement('button');
        addBtn.className = 'custom-dash-tab';
        addBtn.style.cssText = 'border-style:dashed;color:var(--kp-gray-500)';
        addBtn.textContent = '+ New Dashboard';
        addBtn.onclick = addCustomDashboard;
        bar.appendChild(addBtn);
    }

    function switchDash(id) {
        if (id === 'default') {
            activeCustomDash = null;
            document.getElementById('dashContent').style.display = 'block';
            document.getElementById('customDashContent').style.display = 'none';
            renderDashTabs();
            loadDashboard();
        } else {
            activeCustomDash = id;
            document.getElementById('dashContent').style.display = 'none';
            document.getElementById('customDashContent').style.display = 'block';
            renderDashTabs();
            const dash = customDashboards.find(d => d.id === id);
            if (dash) {
                document.getElementById('customDashName').value = dash.name;
                renderWidgets(dash.widgets || []);
            }
        }
    }

    function addCustomDashboard() {
        const id = 'dash_' + Date.now();
        const dash = { id, name: 'New Dashboard', widgets: [] };
        customDashboards.push(dash);
        activeCustomDash = id;
        document.getElementById('dashContent').style.display = 'none';
        document.getElementById('customDashContent').style.display = 'block';
        document.getElementById('customDashName').value = dash.name;
        renderWidgets([]);
        renderDashTabs();
    }

    function addWidget() {
        const dash = customDashboards.find(d => d.id === activeCustomDash);
        if (!dash) return;
        dash.widgets.push({ id: 'w_' + Date.now(), title: 'New Widget', sql: 'SELECT COUNT(*) as total FROM claims', chartType: 'bar' });
        renderWidgets(dash.widgets);
    }

    function renderWidgets(widgets) {
        const grid = document.getElementById('widgetGrid');
        if (!widgets.length) {
            grid.innerHTML = '<div style="text-align:center;padding:40px;color:var(--kp-gray-500)">No widgets yet. Click "+ Add Widget" to add a query widget.</div>';
            return;
        }
        grid.innerHTML = widgets.map(w =>
            '<div class="widget-card" id="wcard_' + w.id + '">'
            + '<div class="widget-header">'
            + '<input value="' + (w.title||'').replace(/"/g,'&quot;') + '" onchange="updateWidgetTitle(\\'' + w.id + '\\',this.value)" placeholder="Widget title...">'
            + '<span class="widget-remove" onclick="removeWidget(\\'' + w.id + '\\')" title="Remove">&#10005;</span>'
            + '</div>'
            + '<div class="widget-body">'
            + '<textarea class="sql-editor" id="wsql_' + w.id + '" style="border:1px solid var(--kp-gray-200);border-radius:4px;min-height:50px" placeholder="SELECT ...">' + (w.sql||'') + '</textarea>'
            + '<div style="margin-top:6px;display:flex;gap:6px">'
            + '<button class="sql-toolbar-btn run" onclick="runWidget(\\'' + w.id + '\\')">Run</button>'
            + '<select onchange="updateWidgetChart(\\'' + w.id + '\\',this.value)" style="padding:3px 8px;font-size:11px;border:1px solid var(--kp-gray-300);border-radius:4px">'
            + '<option value="bar"' + (w.chartType==='bar'?' selected':'') + '>Bar</option>'
            + '<option value="line"' + (w.chartType==='line'?' selected':'') + '>Line</option>'
            + '<option value="doughnut"' + (w.chartType==='doughnut'?' selected':'') + '>Doughnut</option>'
            + '<option value="table"' + (w.chartType==='table'?' selected':'') + '>Table Only</option>'
            + '</select>'
            + '<span class="sql-readonly-badge">READ ONLY</span>'
            + '</div>'
            + '<div id="wresult_' + w.id + '" style="margin-top:8px"></div>'
            + '</div></div>'
        ).join('');
        // Auto-run widgets that have SQL
        widgets.forEach(w => { if (w.sql) runWidget(w.id); });
    }

    function updateWidgetTitle(wid, val) {
        const dash = customDashboards.find(d => d.id === activeCustomDash);
        if (!dash) return;
        const w = dash.widgets.find(w => w.id === wid);
        if (w) w.title = val;
    }
    function updateWidgetChart(wid, val) {
        const dash = customDashboards.find(d => d.id === activeCustomDash);
        if (!dash) return;
        const w = dash.widgets.find(w => w.id === wid);
        if (w) { w.chartType = val; runWidget(wid); }
    }
    function removeWidget(wid) {
        const dash = customDashboards.find(d => d.id === activeCustomDash);
        if (!dash) return;
        dash.widgets = dash.widgets.filter(w => w.id !== wid);
        renderWidgets(dash.widgets);
    }

    async function runWidget(wid) {
        const dash = customDashboards.find(d => d.id === activeCustomDash);
        if (!dash) return;
        const w = dash.widgets.find(w => w.id === wid);
        const sqlEl = document.getElementById('wsql_' + wid);
        const resultEl = document.getElementById('wresult_' + wid);
        if (!w || !sqlEl || !resultEl) return;
        const sql = sqlEl.value.trim();
        w.sql = sql;
        if (!sql) { resultEl.innerHTML = ''; return; }
        resultEl.innerHTML = '<div class="loading" style="padding:8px"><div class="spinner"></div> Running...</div>';
        try {
            const resp = await authFetch(API + '/run-sql', { method:'POST', headers:authHeaders(), body:JSON.stringify({sql}) });
            const data = await resp.json();
            if (data.error) { resultEl.innerHTML = '<div style="color:var(--kp-red);font-size:11px;padding:4px">'+data.error+'</div>'; return; }
            const cols = data.columns||[];
            const rows = (data.rows||[]).slice(0,50);
            let html = '';
            // Render chart if applicable
            if (w.chartType !== 'table' && rows.length > 1 && rows.length <= 30) {
                const cid = 'wchart_' + wid + '_' + Date.now();
                html += '<canvas id="' + cid + '" style="max-height:200px;margin-bottom:8px"></canvas>';
                setTimeout(() => {
                    const canvas = document.getElementById(cid);
                    if (!canvas) return;
                    const labels = rows.map(r => String(r[0]).substring(0,16));
                    const values = rows.map(r => r[1]);
                    const colors = ['#004B87','#00A3B5','#FF6B35','#48A23F','#D32F2F','#7C3AED','#EC4899','#F59E0B'];
                    if (chartInstances[cid]) chartInstances[cid].destroy();
                    chartInstances[cid] = new Chart(canvas, { type:w.chartType==='doughnut'?'doughnut':w.chartType, data:{labels,datasets:[{data:values,backgroundColor:w.chartType==='doughnut'?colors:colors[0]+'CC',borderColor:w.chartType==='line'?colors[0]:undefined,tension:0.3,borderWidth:w.chartType==='line'?2:0,borderRadius:4}]}, options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:w.chartType==='doughnut'}},scales:w.chartType==='doughnut'?{}:{y:{beginAtZero:true},x:{grid:{display:false}}}} });
                }, 50);
            }
            // Table
            const tid = 'wtbl_' + wid;
            html += '<div style="overflow-x:auto;max-height:250px;overflow-y:auto"><table class="data-table" id="'+tid+'"><thead><tr>'
                + cols.map((c,ci) => '<th onclick="sortTable(\\''+tid+'\\','+ci+')">'+c.replace(/_/g,' ')+' <span class="sort-arrow">&#9650;</span></th>').join('')
                + '</tr></thead><tbody>'
                + rows.map(row => '<tr>'+row.map(v => { const n=typeof v==='number'; return '<td class="'+(n?'numeric':'')+'">'+(n?v.toLocaleString(undefined,{maximumFractionDigits:2}):(v==null?'':v))+'</td>'; }).join('')+'</tr>').join('')
                + '</tbody></table></div>';
            resultEl.innerHTML = html;
        } catch(e) { resultEl.innerHTML = '<div style="color:var(--kp-red);font-size:11px">'+e.message+'</div>'; }
    }

    async function saveCustomDashboard() {
        const dash = customDashboards.find(d => d.id === activeCustomDash);
        if (!dash) return;
        dash.name = document.getElementById('customDashName').value || 'Untitled';
        // Save SQL from textareas
        dash.widgets.forEach(w => {
            const el = document.getElementById('wsql_' + w.id);
            if (el) w.sql = el.value;
        });
        try {
            await authFetch(API + '/custom-dashboards', { method:'POST', headers:authHeaders(), body:JSON.stringify(dash) });
            renderDashTabs();
        } catch(e) { console.error('Save failed:', e); }
    }

    // ── Quality page ──
    async function loadQuality() {
        const container = document.getElementById('qualityMeasures');
        try {
            const resp = await authFetch(API + '/cms/quality_measures');
            const measures = await resp.json();
            container.innerHTML = Object.entries(measures).map(([id, m]) => {
                const pct = Math.round((m.benchmark||0) * 100);
                const color = pct >= 80 ? 'var(--kp-green)' : pct >= 60 ? 'var(--kp-orange)' : 'var(--kp-red)';
                return '<div class="measure-card">'
                    + '<div class="measure-name">' + id + ': ' + m.name + '</div>'
                    + '<div class="measure-desc">' + m.description + '</div>'
                    + '<div class="measure-bar"><div class="measure-fill" style="width:' + pct + '%;background:' + color + '"></div></div>'
                    + '<div class="measure-stats"><span>Benchmark: &#8805; ' + pct + '%</span><span>Higher is better</span></div>'
                    + '</div>';
            }).join('');
        } catch (err) {
            container.innerHTML = '<p style="color:var(--kp-gray-500)">Could not load quality measures</p>';
        }
    }

    // ── Explorer page with sortable columns ──
    let explorerSchema = null;
    async function loadExplorer() {
        try {
            const resp = await authFetch(API + '/schema');
            explorerSchema = await resp.json();
            const select = document.getElementById('tableSelect');
            select.innerHTML = '<option value="">Select a table...</option>'
                + Object.keys(explorerSchema).map(t =>
                    '<option value="' + t + '">' + t + ' (' + explorerSchema[t].rows + ' rows, ' + explorerSchema[t].columns + ' cols)</option>'
                ).join('');
        } catch(e) {}
    }

    async function loadTablePreview() {
        const table = document.getElementById('tableSelect').value;
        const preview = document.getElementById('tablePreview');
        const info = document.getElementById('tableInfo');
        if (!table) { preview.innerHTML = ''; info.textContent = ''; return; }

        info.textContent = explorerSchema?.[table] ? explorerSchema[table].rows + ' total rows, ' + explorerSchema[table].columns + ' columns' : '';

        // Use direct SQL for explorer (faster than intelligent pipeline)
        try {
            const resp = await authFetch(API + '/run-sql', {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify({ sql: 'SELECT * FROM ' + table + ' LIMIT 50' }),
            });
            const data = await resp.json();
            if (data.error) { preview.innerHTML = '<p style="color:var(--kp-red)">' + data.error + '</p>'; return; }
            const cols = data.columns || [];
            const rows = data.rows || [];
            const tid = 'exptbl_' + Date.now();
            preview.innerHTML = '<div style="overflow-x:auto;max-height:500px;overflow-y:auto">'
                + '<table class="data-table" id="' + tid + '"><thead><tr>'
                + cols.map((c,ci) => '<th onclick="sortTable(\\'' + tid + '\\',' + ci + ')">' + c + ' <span class="sort-arrow">&#9650;</span></th>').join('')
                + '</tr></thead><tbody>'
                + rows.map(row => '<tr>' + row.map(v => {
                    const n = typeof v === 'number';
                    return '<td class="' + (n?'numeric':'') + '">' + (n ? v.toLocaleString(undefined,{maximumFractionDigits:2}) : (v==null?'':v)) + '</td>';
                }).join('') + '</tr>').join('')
                + '</tbody></table></div>'
                + '<div style="margin-top:6px;display:flex;align-items:center;gap:12px"><span style="font-size:11px;color:var(--kp-gray-500)">Showing first ' + rows.length + ' of ' + (data.row_count||rows.length) + ' rows. Click column headers to sort.</span>'
                + (hasPerm('export_csv') ? '<button onclick="downloadTableById(\\'' + tid + '\\')" style="padding:2px 10px;font-size:10px;border-radius:4px;border:1px solid var(--kp-gray-300);background:white;cursor:pointer;color:var(--kp-blue)">&#11015; Download</button>' : '')
                + '</div>';
            // Pre-fill SQL editor with table query
            document.getElementById('explorerSQL').value = 'SELECT * FROM ' + table + ' LIMIT 50';
        } catch(e) {
            preview.innerHTML = '<p style="color:var(--kp-red)">Error: ' + e.message + '</p>';
        }
    }

    async function runExplorerSQL() {
        const sql = document.getElementById('explorerSQL').value.trim();
        const resultEl = document.getElementById('explorerResults');
        const statusEl = document.getElementById('explorerStatus');
        if (!sql) return;
        statusEl.textContent = 'Running...';
        resultEl.innerHTML = '<div class="loading" style="padding:12px"><div class="spinner"></div> Executing query...</div>';
        try {
            const resp = await authFetch(API + '/run-sql', { method:'POST', headers:authHeaders(), body:JSON.stringify({sql}) });
            const data = await resp.json();
            if (data.error) { resultEl.innerHTML = '<div style="color:var(--kp-red);font-size:12px;padding:8px">' + data.error + '</div>'; statusEl.textContent = 'Error'; return; }
            const cols = data.columns||[];
            const rows = (data.rows||[]).slice(0,200);
            const tid = 'exrsql_' + Date.now();
            statusEl.textContent = data.row_count + ' rows returned' + (data.truncated ? ' (truncated)' : '');
            resultEl.innerHTML = '<div style="overflow-x:auto;max-height:500px;overflow-y:auto">'
                + '<table class="data-table" id="'+tid+'"><thead><tr>'
                + cols.map((c,ci) => '<th onclick="sortTable(\\''+tid+'\\','+ci+')">'+c.replace(/_/g,' ')+' <span class="sort-arrow">&#9650;</span></th>').join('')
                + '</tr></thead><tbody>'
                + rows.map(row => '<tr>'+row.map(v => { const n=typeof v==='number'; return '<td class="'+(n?'numeric':'')+'">'+(n?v.toLocaleString(undefined,{maximumFractionDigits:2}):(v==null?'':v))+'</td>'; }).join('')+'</tr>').join('')
                + '</tbody></table></div>'
                + (hasPerm('export_csv') ? '<div style="margin-top:6px"><button onclick="downloadTableById(\\''+tid+'\\',\\'query_results\\')" style="padding:3px 12px;font-size:10px;border-radius:4px;border:1px solid var(--kp-gray-300);background:white;cursor:pointer;color:var(--kp-blue)">&#11015; Download CSV</button></div>' : '');
        } catch(e) { resultEl.innerHTML = '<div style="color:var(--kp-red);font-size:12px">'+e.message+'</div>'; statusEl.textContent = 'Error'; }
    }

    // ── System page ──
    async function loadSystem() {
        const container = document.getElementById('systemStatus');
        try {
            const resp = await authFetch(API + '/system/status');
            const status = await resp.json();
            container.innerHTML = '<div class="status-card"><h3>Neural Intelligence</h3>'
                + '<div class="status-item"><span>Embedding Vocab</span><span class="status-value">'+(status.neural?.embedding_vocab||0)+'</span></div>'
                + '<div class="status-item"><span>Hopfield Patterns</span><span class="status-value">'+(status.neural?.hopfield_patterns||0)+'</span></div>'
                + '<div class="status-item"><span>GNN Nodes</span><span class="status-value">'+(status.neural?.gnn_nodes||0)+'</span></div></div>'
                + '<div class="status-card"><h3>Learning Engine</h3>'
                + '<div class="status-item"><span>Total Interactions</span><span class="status-value">'+(status.learning?.session?.interactions||0)+'</span></div>'
                + '<div class="status-item"><span>Long-term Memories</span><span class="status-value">'+(status.learning?.long_term?.total_memories||0)+'</span></div>'
                + '<div class="status-item"><span>Success Rate</span><span class="status-value">'+Math.round((status.learning?.learning?.success_rate||0)*100)+'%</span></div></div>'
                + '<div class="status-card"><h3>Model Routing</h3>'
                + Object.entries(status.model_selector||{}).filter(([s,i])=>i.uses>0).map(([s,i]) =>
                    '<div class="status-item"><span>'+s+'</span><span class="status-value">'+i.uses+' uses ('+Math.round(i.success_rate*100)+'%)</span></div>'
                ).join('') + '</div>'
                + '<div class="status-card"><h3>CMS Knowledge Base</h3>'
                + Object.entries(status.cms_knowledge||{}).map(([k,v]) =>
                    '<div class="status-item"><span>'+k.replace(/_/g,' ')+'</span><span class="status-value">'+v+'</span></div>'
                ).join('') + '</div>'
                + '<div class="status-card"><h3>Schema</h3>'
                + '<div class="status-item"><span>Tables</span><span class="status-value">'+(status.schema?.tables||[]).length+'</span></div>'
                + '<div class="status-item"><span>Total Columns</span><span class="status-value">'+(status.schema?.total_columns||0)+'</span></div>'
                + '<div class="status-item"><span>Clinical Layer</span><span class="status-value">'+(status.clinical_active?'Active':'Inactive')+'</span></div></div>';
        } catch(e) {
            container.innerHTML = '<p style="color:var(--kp-red)">Could not load system status</p>';
        }
    }

    // ════════════════════════════════════════════════════════════
    // FORECASTING — HIPAA-compliant on-premise predictions
    // ════════════════════════════════════════════════════════════

    let _forecastCache = null;
    let _forecastMetricsLoaded = false;

    async function loadForecast() {
        // Populate metric dropdown on first load
        if (!_forecastMetricsLoaded) {
            try {
                const resp = await fetch(API + '/forecast/metrics', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({token: authToken})});
                const data = await resp.json();
                if (data.success) {
                    const sel = document.getElementById('forecastMetric');
                    Object.entries(data.metrics).forEach(([key, m]) => {
                        const opt = document.createElement('option');
                        opt.value = key;
                        opt.textContent = m.label;
                        sel.appendChild(opt);
                    });
                    _forecastMetricsLoaded = true;
                }
            } catch(e) { console.warn('Could not load forecast metrics', e); }
        }
        loadSingleForecast();
    }

    async function loadSingleForecast() {
        const metric = document.getElementById('forecastMetric').value;
        const periods = parseInt(document.getElementById('forecastPeriods').value);
        const container = document.getElementById('forecastContent');
        const loading = document.getElementById('forecastLoading');
        loading.style.display = '';
        container.style.opacity = '0.5';

        try {
            const body = {token: authToken, periods: periods};
            if (metric !== 'all') body.metric = metric;
            const resp = await fetch(API + '/forecast', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
            const data = await resp.json();
            if (!data.success) throw new Error(data.message || 'Forecast failed');

            if (metric === 'all') {
                renderAllForecasts(data.data, container);
            } else {
                renderSingleForecast(data.forecast, container);
            }
        } catch(e) {
            container.innerHTML = '<div style="text-align:center;padding:40px;color:var(--kp-red);">Forecast error: ' + e.message + '</div>';
        } finally {
            loading.style.display = 'none';
            container.style.opacity = '1';
        }
    }

    function renderAllForecasts(data, container) {
        if (!data || !data.forecasts) { container.innerHTML = '<p>No forecast data available.</p>'; return; }
        container.style.gridTemplateColumns = 'repeat(auto-fill, minmax(480px, 1fr))';
        let html = '';

        // HIPAA banner
        html += '<div style="grid-column:1/-1;background:linear-gradient(135deg,#e8f5e9,#f1f8e9);border:1px solid #c8e6c9;border-radius:8px;padding:12px 16px;display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
            + '<span style="font-size:18px;">&#128737;</span>'
            + '<div><strong style="font-size:13px;color:#2e7d32;">HIPAA Compliant Forecasting</strong>'
            + '<div style="font-size:11px;color:#558b2f;margin-top:2px;">' + (data.hipaa_statement || 'All models run on-premise.') + '</div></div></div>';

        Object.entries(data.forecasts).forEach(([key, f]) => {
            if (f.error) {
                html += '<div style="background:white;border:1px solid var(--kp-gray-200);border-radius:10px;padding:20px;">'
                    + '<div style="font-weight:600;font-size:14px;margin-bottom:8px;">' + f.label + '</div>'
                    + '<div style="color:var(--kp-gray-500);font-size:12px;">' + f.error + '</div></div>';
                return;
            }
            html += buildForecastCard(f, key);
        });
        container.innerHTML = html;
        // Draw charts after DOM is ready
        setTimeout(() => {
            Object.entries(data.forecasts).forEach(([key, f]) => {
                if (!f.error && f.historical_months) drawForecastChart(key, f);
            });
        }, 50);
    }

    function renderSingleForecast(f, container) {
        container.style.gridTemplateColumns = '1fr';
        if (f.error) {
            container.innerHTML = '<div style="text-align:center;padding:40px;color:var(--kp-gray-500);">' + f.error + '</div>';
            return;
        }
        let html = buildForecastCard(f, f.metric, true);
        // Add detailed model comparison
        html += buildModelComparison(f);
        container.innerHTML = html;
        setTimeout(() => drawForecastChart(f.metric, f, true), 50);
    }

    function buildForecastCard(f, key, expanded) {
        const dir = f.summary.direction;
        const dirIcon = dir === 'increase' ? '&#9650;' : dir === 'decrease' ? '&#9660;' : '&#9656;';
        const dirColor = dir === 'increase' ? (f.unit === 'dollars' && key.includes('cost') ? 'var(--kp-red)' : 'var(--kp-green)')
            : dir === 'decrease' ? (f.unit === 'dollars' && key.includes('revenue') ? 'var(--kp-red)' : 'var(--kp-green)')
            : 'var(--kp-gray-500)';
        const pct = f.summary.pct_change;
        const modelName = f.forecasts[f.recommended_model].explanation.name;

        let html = '<div style="background:white;border:1px solid var(--kp-gray-200);border-radius:10px;padding:20px;'
            + (expanded ? '' : '') + '">';
        // Header
        html += '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">'
            + '<div><div style="font-weight:600;font-size:14px;color:var(--kp-gray-900);">' + f.label + '</div>'
            + '<div style="font-size:11px;color:var(--kp-gray-500);margin-top:2px;">' + f.summary.data_points + ' months historical data</div></div>'
            + '<div style="text-align:right;">'
            + '<div style="font-size:20px;font-weight:700;color:' + dirColor + ';">' + dirIcon + ' ' + (pct > 0 ? '+' : '') + pct + '%</div>'
            + '<div style="font-size:11px;color:var(--kp-gray-500);">projected change</div></div></div>';
        // Chart canvas
        html += '<div style="position:relative;height:' + (expanded ? '280' : '200') + 'px;margin-bottom:12px;">'
            + '<canvas id="fcChart_' + key + '"></canvas></div>';
        // Narrative
        const narrative = f.narrative || '';
        const paragraphs = narrative.split('\\n\\n');
        html += '<div style="font-size:12px;line-height:1.6;color:var(--kp-gray-700);margin-bottom:10px;">';
        paragraphs.forEach(p => {
            if (p.trim()) html += '<p style="margin:0 0 8px;">' + p.trim() + '</p>';
        });
        html += '</div>';
        // Model badge
        html += '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
            + '<span style="font-size:11px;background:var(--kp-blue-pale,#E6F2FF);color:var(--kp-blue);padding:3px 10px;border-radius:12px;font-weight:500;">Model: ' + modelName + '</span>'
            + '<span style="font-size:11px;background:#e8f5e9;color:#2e7d32;padding:3px 10px;border-radius:12px;">&#128737; On-Premise</span>'
            + '</div>';
        html += '</div>';
        return html;
    }

    function buildModelComparison(f) {
        let html = '<div style="background:white;border:1px solid var(--kp-gray-200);border-radius:10px;padding:20px;margin-top:12px;">';
        html += '<h3 style="margin:0 0 12px;font-size:15px;color:var(--kp-gray-900);">Model Comparison &amp; Explanations</h3>';
        Object.entries(f.forecasts).forEach(([mkey, m]) => {
            const isRec = mkey === f.recommended_model;
            html += '<div style="border:1px solid ' + (isRec ? 'var(--kp-blue)' : 'var(--kp-gray-200)') + ';border-radius:8px;padding:14px;margin-bottom:10px;'
                + (isRec ? 'background:var(--kp-blue-pale,#E6F2FF);' : '') + '">';
            html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                + '<strong style="font-size:13px;">' + m.explanation.name + '</strong>';
            if (isRec) html += '<span style="font-size:10px;background:var(--kp-blue);color:white;padding:2px 8px;border-radius:10px;">RECOMMENDED</span>';
            html += '</div>';
            html += '<p style="margin:0 0 6px;font-size:12px;color:var(--kp-gray-700);line-height:1.5;">' + m.explanation.plain_english + '</p>';
            html += '<div style="font-size:11px;color:var(--kp-gray-500);"><strong>Best for:</strong> ' + m.explanation.strengths + '</div>';
            html += '<div style="font-size:11px;color:#2e7d32;margin-top:4px;">&#128737; ' + m.explanation.hipaa_note + '</div>';
            html += '</div>';
        });
        html += '</div>';
        return html;
    }

    function drawForecastChart(key, f, large) {
        const canvas = document.getElementById('fcChart_' + key);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const W = canvas.parentElement.offsetWidth;
        const H = large ? 280 : 200;
        canvas.width = W * 2;
        canvas.height = H * 2;
        canvas.style.width = W + 'px';
        canvas.style.height = H + 'px';
        ctx.scale(2, 2);

        const hist = f.historical_values;
        const fc = f.forecasts[f.recommended_model].values;
        const ciLow = f.confidence_interval.low;
        const ciHigh = f.confidence_interval.high;
        const allVals = hist.concat(fc).concat(ciHigh);
        const minVal = Math.min(...hist.concat(fc).concat(ciLow)) * 0.9;
        const maxVal = Math.max(...allVals) * 1.1;
        const totalPts = hist.length + fc.length;

        const padL = 50, padR = 20, padT = 15, padB = 35;
        const chartW = W - padL - padR;
        const chartH = H - padT - padB;

        function xPos(i) { return padL + (i / (totalPts - 1)) * chartW; }
        function yPos(v) { return padT + chartH - ((v - minVal) / (maxVal - minVal)) * chartH; }

        // Grid lines
        ctx.strokeStyle = '#f0f0f0';
        ctx.lineWidth = 0.5;
        for (let i = 0; i < 5; i++) {
            const y = padT + (i / 4) * chartH;
            ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
            const val = maxVal - (i / 4) * (maxVal - minVal);
            ctx.fillStyle = '#999';
            ctx.font = '10px -apple-system, sans-serif';
            ctx.textAlign = 'right';
            if (f.unit === 'dollars') ctx.fillText('$' + (val >= 1000 ? Math.round(val/1000) + 'k' : Math.round(val)), padL - 5, y + 3);
            else if (f.unit === 'percent') ctx.fillText(val.toFixed(1) + '%', padL - 5, y + 3);
            else ctx.fillText(Math.round(val).toLocaleString(), padL - 5, y + 3);
        }

        // Divider line at forecast start
        const divX = xPos(hist.length - 1);
        ctx.strokeStyle = '#ccc';
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(divX, padT); ctx.lineTo(divX, padT + chartH); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#999';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Forecast Start', divX, padT + chartH + 14);

        // Confidence interval band
        ctx.fillStyle = 'rgba(0, 107, 166, 0.1)';
        ctx.beginPath();
        for (let i = 0; i < fc.length; i++) {
            const x = xPos(hist.length + i);
            if (i === 0) ctx.moveTo(x, yPos(ciHigh[i]));
            else ctx.lineTo(x, yPos(ciHigh[i]));
        }
        for (let i = fc.length - 1; i >= 0; i--) {
            ctx.lineTo(xPos(hist.length + i), yPos(ciLow[i]));
        }
        ctx.closePath();
        ctx.fill();

        // Historical line
        ctx.strokeStyle = 'var(--kp-blue, #006BA6)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        hist.forEach((v, i) => {
            if (i === 0) ctx.moveTo(xPos(i), yPos(v));
            else ctx.lineTo(xPos(i), yPos(v));
        });
        ctx.stroke();

        // Forecast line (dashed)
        ctx.strokeStyle = '#006BA6';
        ctx.setLineDash([6, 3]);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(xPos(hist.length - 1), yPos(hist[hist.length - 1]));
        fc.forEach((v, i) => {
            ctx.lineTo(xPos(hist.length + i), yPos(v));
        });
        ctx.stroke();
        ctx.setLineDash([]);

        // Dots on forecast
        fc.forEach((v, i) => {
            ctx.fillStyle = '#006BA6';
            ctx.beginPath();
            ctx.arc(xPos(hist.length + i), yPos(v), 3, 0, Math.PI * 2);
            ctx.fill();
        });

        // X-axis labels (show a few)
        const allMonths = f.historical_months.concat(f.forecast_months);
        ctx.fillStyle = '#999';
        ctx.font = '9px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        const step = Math.max(1, Math.floor(totalPts / 8));
        for (let i = 0; i < totalPts; i += step) {
            if (allMonths[i]) ctx.fillText(allMonths[i], xPos(i), padT + chartH + 26);
        }

        // Legend
        ctx.font = '10px -apple-system, sans-serif';
        const legX = padL + 10;
        ctx.fillStyle = '#006BA6'; ctx.fillRect(legX, padT + 2, 14, 2);
        ctx.fillStyle = '#666'; ctx.textAlign = 'left'; ctx.fillText('Historical', legX + 18, padT + 6);
        ctx.setLineDash([4,2]); ctx.strokeStyle = '#006BA6'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(legX + 90, padT + 3); ctx.lineTo(legX + 104, padT + 3); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillText('Forecast', legX + 108, padT + 6);
        ctx.fillStyle = 'rgba(0,107,166,0.15)'; ctx.fillRect(legX + 170, padT, 14, 6);
        ctx.fillStyle = '#666'; ctx.fillText('95% CI', legX + 188, padT + 6);
    }

    // ── Logout ──
    async function doLogout() {
        try {
            await authFetch(API + '/logout', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({token: authToken}),
            });
        } catch(e) {}
        authToken = null;
        currentUser = null;
        userPermissions = [];
        userRole = '';
        userRoleLabel = '';
        document.getElementById('loginOverlay').style.display = 'flex';
        document.getElementById('userDisplayName').textContent = '';
    }

    // ════════════════════════════════════════════════════════════
    // ENCRYPTED DOWNLOAD — all downloads require password
    // ════════════════════════════════════════════════════════════

    let _pendingDownload = null;  // {columns, rows, filename}

    function showDownloadModal(columns, rows, filename) {
        if (!hasPerm('export_csv')) { alert('Export not available for your role.'); return; }
        if (!columns || !columns.length) { alert('No data to download.'); return; }
        _pendingDownload = {columns, rows: rows || [], filename: filename || 'report'};
        let modal = document.getElementById('downloadModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'downloadModal';
            modal.style.cssText = 'position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center;';
            modal.innerHTML = '<div style="background:white;border-radius:12px;padding:28px;width:420px;max-width:90vw;box-shadow:0 20px 60px rgba(0,0,0,0.3);">'
                + '<h3 style="margin:0 0 6px;color:var(--kp-gray-900);">&#128274; Encrypted Download</h3>'
                + '<p style="margin:0 0 16px;font-size:12px;color:var(--kp-gray-500);">All downloads are password-protected for security.</p>'
                + '<label style="font-size:12px;font-weight:600;color:var(--kp-gray-700);display:block;margin-bottom:4px;">Set Password</label>'
                + '<input id="dlPwd" type="password" placeholder="Enter password (min 4 chars)" style="width:100%;padding:10px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;margin-bottom:8px;box-sizing:border-box;font-size:13px;">'
                + '<label style="font-size:12px;font-weight:600;color:var(--kp-gray-700);display:block;margin-bottom:4px;">Confirm Password</label>'
                + '<input id="dlPwd2" type="password" placeholder="Confirm password" style="width:100%;padding:10px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;margin-bottom:8px;box-sizing:border-box;font-size:13px;">'
                + '<div id="dlError" style="color:#c0392b;font-size:12px;display:none;margin-bottom:8px;"></div>'
                + '<div id="dlInfo" style="font-size:11px;color:var(--kp-gray-500);margin-bottom:12px;"></div>'
                + '<div style="display:flex;gap:8px;">'
                + '<button id="dlSubmitBtn" onclick="doEncryptedDownload()" style="flex:1;padding:10px;background:var(--kp-blue);color:white;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500;">&#128274; Download Encrypted ZIP</button>'
                + '<button id="dlCancelBtn" style="padding:10px 16px;background:var(--kp-gray-100);border:1px solid var(--kp-gray-300);border-radius:6px;cursor:pointer;font-size:13px;">Cancel</button>'
                + '</div></div>';
            document.body.appendChild(modal);
            document.getElementById('dlCancelBtn').onclick = function(){ document.getElementById('downloadModal').style.display = 'none'; };
        }
        // Update info
        document.getElementById('dlInfo').textContent = _pendingDownload.rows.length + ' rows will be exported as encrypted ZIP';
        document.getElementById('dlPwd').value = '';
        document.getElementById('dlPwd2').value = '';
        document.getElementById('dlError').style.display = 'none';
        modal.style.display = 'flex';
        document.getElementById('dlPwd').focus();
    }

    async function doEncryptedDownload() {
        const pwd = document.getElementById('dlPwd').value;
        const pwd2 = document.getElementById('dlPwd2').value;
        const errEl = document.getElementById('dlError');
        if (!pwd || pwd.length < 4) { errEl.textContent = 'Password must be at least 4 characters'; errEl.style.display = 'block'; return; }
        if (pwd !== pwd2) { errEl.textContent = 'Passwords do not match'; errEl.style.display = 'block'; return; }
        if (!_pendingDownload) return;

        const btn = document.getElementById('dlSubmitBtn');
        btn.disabled = true; btn.textContent = 'Encrypting...';
        errEl.style.display = 'none';

        try {
            const resp = await authFetch(API + '/export/encrypted', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    token: authToken,
                    password: pwd,
                    columns: _pendingDownload.columns,
                    rows: _pendingDownload.rows,
                    filename: _pendingDownload.filename,
                }),
            });
            if (!resp.ok) throw new Error('Server error: ' + resp.status);
            const blob = await resp.blob();
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = _pendingDownload.filename + '_encrypted.zip';
            a.click();
            document.getElementById('downloadModal').style.display = 'none';
            showToast('Encrypted download complete');
        } catch(e) {
            errEl.textContent = 'Download failed: ' + e.message;
            errEl.style.display = 'block';
        } finally {
            btn.disabled = false;
            btn.innerHTML = '&#128274; Download Encrypted ZIP';
        }
    }

    // ── Export last query results ──
    function exportCurrentCSV() {
        if (!lastData || !lastData.columns) { alert('No results to export. Run a query first.'); return; }
        showDownloadModal(lastData.columns, lastData.rows, 'results_' + new Date().toISOString().slice(0,10));
    }

    // ── Download any HTML table by ID ──
    function downloadTableById(tableId, name) {
        const tbl = document.getElementById(tableId);
        if (!tbl) { alert('Table not found.'); return; }
        const cols = Array.from(tbl.querySelectorAll('thead th')).map(th => th.textContent.replace(/[\\u25B2\\u25BC]/g,'').trim());
        const rows = [];
        tbl.querySelectorAll('tbody tr').forEach(tr => {
            rows.push(Array.from(tr.querySelectorAll('td')).map(td => td.textContent));
        });
        showDownloadModal(cols, rows, (name || tableId) + '_' + new Date().toISOString().slice(0,10));
    }

    // ── Download per-card CSV ──
    function downloadCardCSV(cardId) {
        const d = _cardResults[cardId];
        if (!d || !d.columns.length) { alert('No data to download.'); return; }
        const fname = (d.question || 'results').replace(/[^a-zA-Z0-9]/g,'_').substring(0,40);
        showDownloadModal(d.columns, d.rows, fname);
    }

    // ── Download Dashboard KPIs ──
    function downloadDashboardCSV() {
        const cards = document.querySelectorAll('#page-dashboard .kpi-card');
        if (!cards.length) { alert('No dashboard data to download.'); return; }
        const columns = ['KPI', 'Value', 'Subtitle'];
        const rows = [];
        cards.forEach(card => {
            rows.push([
                (card.querySelector('.kpi-title') || {}).textContent || '',
                (card.querySelector('.kpi-value') || {}).textContent || '',
                (card.querySelector('.kpi-subtitle') || {}).textContent || '',
            ]);
        });
        showDownloadModal(columns, rows, 'dashboard_kpis_' + new Date().toISOString().slice(0,10));
    }

    // ── Email Modal ──
    function openEmailModal() {
        if (!hasPerm('export_email')) { alert('Email not available for your role.'); return; }
        if (!lastData || !lastData.columns) { alert('No results to email. Run a query first.'); return; }
        // Create modal if not exists
        let modal = document.getElementById('emailModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'emailModal';
            modal.style.cssText = 'position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center;';
            modal.innerHTML = `
                <div style="background:white;border-radius:12px;padding:24px;width:460px;max-width:90vw;box-shadow:0 20px 60px rgba(0,0,0,0.3);">
                    <h3 style="margin:0 0 16px;color:var(--kp-gray-900);">&#9993; Email Results</h3>
                    <input id="emTo" type="email" placeholder="Recipient email" style="width:100%;padding:8px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;margin-bottom:8px;box-sizing:border-box;">
                    <input id="emSubject" value="Healthcare Analytics Report" style="width:100%;padding:8px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;margin-bottom:8px;box-sizing:border-box;">
                    <textarea id="emBody" rows="2" placeholder="Add a note..." style="width:100%;padding:8px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;margin-bottom:8px;resize:vertical;box-sizing:border-box;font-family:inherit;">Please find the attached analytics report.</textarea>
                    <select id="emFormat" style="width:100%;padding:8px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;margin-bottom:8px;">
                        <option value="html">HTML Report (formatted)</option>
                        <option value="csv">CSV (data only)</option>
                    </select>
                    <label style="display:flex;align-items:center;gap:8px;font-size:13px;padding:8px 0;cursor:pointer;">
                        <input type="checkbox" id="emEncrypt" onchange="document.getElementById('emPwd').style.display=this.checked?'block':'none'">
                        &#128274; Password-protect attachment
                    </label>
                    <input id="emPwd" type="password" placeholder="Encryption password" style="display:none;width:100%;padding:8px 12px;border:1px solid var(--kp-gray-300);border-radius:6px;margin-bottom:8px;box-sizing:border-box;">
                    <div id="emError" style="color:#c0392b;font-size:12px;display:none;margin-bottom:8px;"></div>
                    <div style="display:flex;gap:8px;margin-top:12px;">
                        <button id="emSendBtn" onclick="doSendEmail()" style="flex:1;padding:8px;background:var(--kp-blue);color:white;border:none;border-radius:6px;cursor:pointer;font-size:13px;">&#9993; Send</button>
                        <button onclick="document.getElementById('emailModal').style.display='none'" style="flex:1;padding:8px;background:var(--kp-gray-100);border:1px solid var(--kp-gray-300);border-radius:6px;cursor:pointer;font-size:13px;">Cancel</button>
                    </div>
                </div>`;
            document.body.appendChild(modal);
        }
        modal.style.display = 'flex';
    }

    async function doSendEmail() {
        const to = document.getElementById('emTo').value.trim();
        const subject = document.getElementById('emSubject').value.trim();
        const body = document.getElementById('emBody').value.trim();
        const format = document.getElementById('emFormat').value;
        const encrypt = document.getElementById('emEncrypt').checked;
        const pwd = encrypt ? document.getElementById('emPwd').value : null;
        const errEl = document.getElementById('emError');

        if (!to || !to.includes('@')) { errEl.textContent='Enter a valid email'; errEl.style.display='block'; return; }
        if (encrypt && (!pwd || pwd.length < 4)) { errEl.textContent='Password must be 4+ chars'; errEl.style.display='block'; return; }

        const btn = document.getElementById('emSendBtn');
        btn.disabled = true; btn.textContent = 'Sending...';
        try {
            const resp = await authFetch(API + '/email/send', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    token: authToken, to_email: to, subject, body,
                    columns: lastData.columns, rows: lastData.rows,
                    format, password: pwd, narrative: lastData.narrative || '', sql: lastSQL,
                }),
            });
            const data = await resp.json();
            if (data.success) {
                document.getElementById('emailModal').style.display = 'none';
                showToast('Email sent to ' + to + (encrypt ? ' (encrypted)' : ''));
            } else { errEl.textContent = data.message; errEl.style.display = 'block'; }
        } catch(e) { errEl.textContent = 'Failed: '+e.message; errEl.style.display = 'block'; }
        finally { btn.disabled = false; btn.textContent = '✉ Send'; }
    }

    // ── Email per-card results ──
    function emailCardResults(cardId) {
        if (!hasPerm('export_email')) { alert('Email not available for your role.'); return; }
        const d = _cardResults[cardId];
        if (!d || !d.columns.length) { alert('No data to email.'); return; }
        // Temporarily swap lastData so the email modal uses this card's data
        const savedData = lastData;
        const savedSQL = lastSQL;
        lastData = {columns: d.columns, rows: d.rows, narrative: d.question};
        lastSQL = '';
        openEmailModal();
        // Restore after modal opens (email send will read lastData at send time)
        // We keep the swap active until the modal closes
    }

    // ── Email Dashboard ──
    function emailDashboard() {
        if (!hasPerm('export_email')) { alert('Email not available for your role.'); return; }
        const cards = document.querySelectorAll('#page-dashboard .kpi-card');
        if (!cards.length) { alert('No dashboard data to email.'); return; }
        const columns = ['KPI', 'Value', 'Subtitle'];
        const rows = [];
        cards.forEach(card => {
            const title = (card.querySelector('.kpi-title') || {}).textContent || '';
            const value = (card.querySelector('.kpi-value') || {}).textContent || '';
            const sub = (card.querySelector('.kpi-subtitle') || {}).textContent || '';
            rows.push([title, value, sub]);
        });
        lastData = {columns: columns, rows: rows, narrative: 'Executive Dashboard KPIs'};
        lastSQL = '';
        openEmailModal();
    }

    // ── Admin Panel ──
    function showAdminPanel() {
        if (!hasPerm('admin_panel')) { alert('Admin access required.'); return; }
        let modal = document.getElementById('adminModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'adminModal';
            modal.style.cssText = 'position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center;';
            modal.innerHTML = `
                <div style="background:white;border-radius:12px;padding:24px;width:700px;max-width:90vw;max-height:80vh;overflow-y:auto;box-shadow:0 20px 60px rgba(0,0,0,0.3);">
                    <h3 style="margin:0 0 16px;color:var(--kp-gray-900);">&#9881; Admin Panel</h3>
                    <div style="display:flex;gap:8px;margin-bottom:16px;">
                        <button onclick="adminLoadUsers()" style="padding:6px 14px;border-radius:4px;border:1px solid var(--kp-gray-300);background:var(--kp-blue);color:white;cursor:pointer;font-size:12px;">Users</button>
                        <button onclick="adminShowCreate()" style="padding:6px 14px;border-radius:4px;border:1px solid var(--kp-gray-300);background:white;cursor:pointer;font-size:12px;">Create User</button>
                    </div>
                    <div id="adminContent">Loading...</div>
                    <div style="margin-top:16px;text-align:right;">
                        <button onclick="document.getElementById('adminModal').style.display='none'"
                            style="padding:6px 16px;border-radius:4px;border:1px solid var(--kp-gray-300);background:white;cursor:pointer;">Close</button>
                    </div>
                </div>`;
            document.body.appendChild(modal);
        }
        modal.style.display = 'flex';
        adminLoadUsers();
    }

    async function adminLoadUsers() {
        const container = document.getElementById('adminContent');
        try {
            const resp = await authFetch(API + '/admin/users', {
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({token:authToken}),
            });
            const data = await resp.json();
            if (!data.success) { container.textContent = data.message; return; }
            const roleColors = {admin:'#c0392b',business:'#2980b9',dev:'#27ae60',viewer:'#8e44ad',member:'#2980b9'};
            let html = '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
                + '<tr style="background:var(--kp-gray-100);"><th style="padding:6px 8px;text-align:left;">User</th>'
                + '<th>Role</th><th>Last Login</th><th>Actions</th></tr>';
            data.users.forEach(u => {
                const opts = ['admin','business','dev','viewer'].map(r =>
                    '<option value="'+r+'"'+(r===u.role?' selected':'')+'>'+r.charAt(0).toUpperCase()+r.slice(1)+'</option>'
                ).join('');
                html += '<tr style="border-bottom:1px solid var(--kp-gray-200);">'
                    + '<td style="padding:6px 8px;"><strong>'+(u.display_name||u.username)+'</strong>'
                    + '<br><span style="color:var(--kp-gray-500);font-size:11px;">@'+u.username+'</span></td>'
                    + '<td><select onchange="adminSetRole(&apos;'+u.user_id+'&apos;,this.value)"'
                    + ' style="padding:3px 6px;border-radius:4px;border:1px solid var(--kp-gray-300);font-size:11px;'
                    + 'background:'+(roleColors[u.role]||'#666')+';color:#fff;">'+opts+'</select></td>'
                    + '<td style="font-size:11px;color:var(--kp-gray-500);">'+(u.last_login||'Never')+'</td>'
                    + '<td><button onclick="adminDelUser(&apos;'+u.user_id+'&apos;,&apos;'+u.username+'&apos;)"'
                    + ' style="font-size:11px;color:#c0392b;background:none;border:none;cursor:pointer;">Delete</button></td></tr>';
            });
            container.innerHTML = html + '</table>';
        } catch(e) { container.textContent = 'Error: '+e.message; }
    }

    async function adminSetRole(uid, role) {
        const resp = await authFetch(API + '/admin/update-role', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({token:authToken, user_id:uid, role}),
        });
        const data = await resp.json();
        showToast(data.message);
        adminLoadUsers();
    }

    async function adminDelUser(uid, name) {
        if (!confirm('Delete user "'+name+'"?')) return;
        const resp = await authFetch(API + '/admin/delete-user', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({token:authToken, user_id:uid}),
        });
        const data = await resp.json();
        showToast(data.message);
        adminLoadUsers();
    }

    function adminShowCreate() {
        document.getElementById('adminContent').innerHTML = `
            <input id="acUser" placeholder="Username" style="width:100%;padding:8px;border:1px solid var(--kp-gray-300);border-radius:6px;margin-bottom:6px;box-sizing:border-box;">
            <input id="acEmail" type="email" placeholder="Email (optional)" style="width:100%;padding:8px;border:1px solid var(--kp-gray-300);border-radius:6px;margin-bottom:6px;box-sizing:border-box;">
            <input id="acPass" type="password" placeholder="Password" style="width:100%;padding:8px;border:1px solid var(--kp-gray-300);border-radius:6px;margin-bottom:6px;box-sizing:border-box;">
            <select id="acRole" style="width:100%;padding:8px;border:1px solid var(--kp-gray-300);border-radius:6px;margin-bottom:6px;">
                <option value="viewer">Viewer</option><option value="dev">Developer</option>
                <option value="business" selected>Business User</option><option value="admin">Admin</option>
            </select>
            <div id="acError" style="color:#c0392b;font-size:12px;display:none;margin-bottom:6px;"></div>
            <button onclick="adminDoCreate()" style="width:100%;padding:8px;background:var(--kp-blue);color:white;border:none;border-radius:6px;cursor:pointer;">Create User</button>`;
    }

    async function adminDoCreate() {
        const user = document.getElementById('acUser').value.trim();
        const email = document.getElementById('acEmail').value.trim();
        const pass = document.getElementById('acPass').value;
        const role = document.getElementById('acRole').value;
        if (!user || !pass) { const e=document.getElementById('acError'); e.textContent='Username and password required'; e.style.display='block'; return; }
        const resp = await authFetch(API + '/admin/create-user', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({token:authToken, username:user, password:pass, email, role}),
        });
        const data = await resp.json();
        if (data.success) { showToast('User created: '+user); adminLoadUsers(); }
        else { const e=document.getElementById('acError'); e.textContent=data.message; e.style.display='block'; }
    }

    function showToast(msg) {
        const t = document.createElement('div');
        t.style.cssText = 'position:fixed;top:60px;right:20px;background:var(--kp-green);color:white;padding:10px 20px;border-radius:8px;font-size:13px;font-weight:500;z-index:99999;box-shadow:0 4px 12px rgba(0,0,0,0.2);';
        t.textContent = msg;
        document.body.appendChild(t);
        setTimeout(() => t.remove(), 3000);
    }

    // ── Initialize ──
    document.getElementById('queryInput').focus();

    // ── Load suggested questions from query tracker ──
    (async function loadSuggestions() {
        try {
            const resp = await fetch('/api/suggest');
            if (resp.ok) {
                const data = await resp.json();
                const container = document.getElementById('suggestions');
                if (container && data.frequently_asked && data.frequently_asked.length > 0) {
                    container.innerHTML = '';
                    data.frequently_asked.slice(0, 6).forEach(item => {
                        const btn = document.createElement('button');
                        btn.className = 'suggestion-chip';
                        btn.textContent = item.question || item;
                        btn.onclick = () => askThis(item.question || item);
                        container.appendChild(btn);
                    });
                }
            }
        } catch(e) { /* suggestions are optional */ }
    })();

    // ── DevTools protection (KP security requirement) ──
    document.addEventListener('contextmenu', e => e.preventDefault());
    document.addEventListener('keydown', function(e) {
        if (e.key === 'F12' ||
            (e.ctrlKey && e.shiftKey && (e.key === 'I' || e.key === 'J')) ||
            (e.ctrlKey && e.key === 'U')) {
            e.preventDefault();
            return false;
        }
    });
    </script>
</body>
</html>'''


def create_intelligent_server(db_path: str, host: str = '0.0.0.0', port: int = 5000) -> Any:
    """Create Flask server for intelligent dashboard with NL query API."""
    try:
        from flask import Flask, request, jsonify, Response
    except ImportError:
        logger.error("Flask not installed. Run: pip install flask")
        return None

    app = Flask(__name__)

    # ── Security Headers (KP compliance) ──
    @app.after_request
    def add_security_headers(response):
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'"
        )
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '0'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = (
            'camera=(), microphone=(), geolocation=(), accelerometer=(), gyroscope=(), magnetometer=()'
        )
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response

    # Initialize intelligent pipeline
    pipeline = None

    def get_pipeline():
        nonlocal pipeline
        if pipeline is None:
            pipeline = IntelligentPipeline(db_path, neural_dim=32)
        return pipeline

    @app.route('/')
    def index():
        return Response(build_dashboard_html(), content_type='text/html')

    @app.route('/api/intelligent/query', methods=['POST'])
    def intelligent_query():
        data = request.json
        question = data.get('question', '')
        session_id = data.get('session_id', 'default')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        p = get_pipeline()
        result = p.process(question, session_id=session_id)

        # Serialize numpy arrays
        if 'rows' in result:
            result['rows'] = [
                [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in row]
                for row in result['rows']
            ]

        # Remove non-serializable fields
        for key in ['clinical_context']:
            if key in result and isinstance(result[key], dict):
                for k, v in list(result[key].items()):
                    if hasattr(v, 'tolist'):
                        result[key][k] = v.tolist()

        # Ensure insights are JSON-serializable (convert sets, numpy, etc.)
        def _make_serializable(obj):
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_serializable(v) for v in obj]
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if hasattr(obj, '__float__'):
                try:
                    f = float(obj)
                    if math.isnan(f) or math.isinf(f):
                        return None
                    return f
                except (ValueError, TypeError):
                    return str(obj)
            return obj

        if 'insights' in result:
            result['insights'] = _make_serializable(result['insights'])
        if 'nlp_enrichment' in result:
            result['nlp_enrichment'] = _make_serializable(result['nlp_enrichment'])
        if 'gpdm_context' in result:
            result['gpdm_context'] = _make_serializable(result['gpdm_context'])

        return jsonify(result)

    @app.route('/api/schema')
    def schema():
        p = get_pipeline()
        return jsonify(p.sql_engine.semantic.get_schema_summary())

    @app.route('/api/cms/quality_measures')
    def quality_measures():
        from cms_data_loader import QUALITY_MEASURES
        return jsonify(QUALITY_MEASURES)

    @app.route('/api/system/status')
    def system_status():
        p = get_pipeline()
        status = p.get_system_status()
        return jsonify(status)

    return app


# Main

if __name__ == '__main__':
    os.chdir(SCRIPT_DIR)
    logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description='KP GPDM Intelligent Dashboard')
    parser.add_argument('--db', default='../data/healthcare_production.db',
                       help='Path to database')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    args = parser.parse_args()

    # Determine SSL availability
    ssl_context = None
    protocol = 'http'
    if os.path.exists(SSL_CERT) and os.path.exists(SSL_KEY):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        ssl_context.load_cert_chain(SSL_CERT, SSL_KEY)
        protocol = 'https'
    else:
        logger.warning("SSL certs not found at %s — falling back to HTTP", CERT_DIR)
        logger.warning("Generate certs: openssl req -x509 -newkey rsa:2048 "
                       "-keyout certs/key.pem -out certs/cert.pem -days 365 -nodes "
                       "-subj '/CN=localhost'")

    print(f"""
    ╔═══════════════════════════════════════════════════════╗
    ║  KP Healthcare GPDM — Intelligent Analytics        ║
    ║                                                       ║
    ║  Dashboard: {protocol}://localhost:{args.port}                    ║
    ║  API:       {protocol}://localhost:{args.port}/api                ║
    ║  Transport: {'TLS 1.2+ (HTTPS)' if ssl_context else 'HTTP (no certs)'}{"":>17}║
    ║  Headers:   HSTS, CSP, X-Frame, X-Content-Type       ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    app = create_intelligent_server(args.db, args.host, args.port)
    if app:
        app.run(host=args.host, port=args.port, debug=False, ssl_context=ssl_context)
    else:
        print("Install Flask: pip install flask --break-system-packages")
