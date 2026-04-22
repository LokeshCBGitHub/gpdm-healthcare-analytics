#!/usr/bin/env python3
"""
Live Dashboard Server — Pure Python HTTP Server
===============================================

Serves healthcare dashboards in real-time using the AnalyticalIntelligence engine.
- Landing page with links to all 16 dashboards
- Database selector (71K or 4M) with sticky header
- In-memory caching with "Regenerate" button
- Real-time progress indicators
- Professional healthcare theme

Usage:
    cd /sessions/great-gallant-allen/mnt/chatbot/mtp_demo
    python live_dashboard_server.py

Then visit: http://localhost:8050
"""

import sys
import os
import sqlite3
import threading
import time
import json
import html
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# Add scripts to path so we can import AnalyticalIntelligence
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from analytical_intelligence import AnalyticalIntelligence


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PORT = 8050
BASE_DIR = Path(__file__).parent.absolute()

DATABASES = {
    '71k': {
        'path': BASE_DIR / 'data' / 'healthcare_demo_71k_backup.db',
        'label': '71K Members',
        'color': '#1e3a5f',  # Dark navy
        'size': '18 MB'
    },
    '4m': {
        'path': BASE_DIR / 'data' / 'healthcare_demo.db',
        'label': '4M Members',
        'color': '#1b5e20',  # Dark green
        'size': '1.1 GB'
    }
}

DASHBOARDS = [
    ('executive_dashboard', 'Executive summary with KPIs, PMPM, MLR, Stars, regional scorecard'),
    ('financial_dashboard', 'Financial performance analysis'),
    ('demographics_dashboard', 'Demographic population age gender race ethnicity profile distribution breakdown'),
    ('utilization_dashboard', 'Utilization metrics and trends'),
    ('quality_dashboard', 'Quality metrics and HEDIS measures'),
    ('provider_dashboard', 'Provider performance analysis'),
    ('clinical_outcomes_dashboard', 'Clinical outcomes analysis'),
    ('claims_severity_dashboard', 'Claims severity analysis'),
    ('pharmacy_dashboard', 'Pharmacy analytics and medication insights'),
    ('referral_network_dashboard', 'Referral network analysis'),
    ('provider_network_dashboard', 'Provider network adequacy'),
    ('forecasting_dashboard', 'Forecasting and predictive analytics'),
    ('appointment_access_dashboard', 'Appointment access and scheduling'),
    ('membership_intelligence_dashboard', 'Membership intelligence and enrollment'),
    ('population_health_dashboard', 'Population health management'),
    ('revenue_cycle_dashboard', 'Revenue cycle management'),
]

# Global cache: {db: {dashboard_name: {html, metadata}}}
dashboard_cache = {}
cache_lock = threading.Lock()
generation_status = {}


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_db_size_bytes(db_path):
    """Get database file size in bytes."""
    try:
        return os.path.getsize(db_path)
    except:
        return 0


def format_size(bytes_val):
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def escape_html(s):
    """Escape HTML special characters."""
    return html.escape(str(s), quote=True)


def count_db_rows(db_path):
    """Count total rows across all tables in database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'gpdm_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        total_rows = 0
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                total_rows += cursor.fetchone()[0]
            except:
                pass

        conn.close()
        return total_rows
    except Exception as e:
        print(f"Error counting rows in {db_path}: {e}")
        return 0


def generate_dashboard_background(db_key):
    """Generate dashboard in background thread."""
    if db_key not in DATABASES:
        return

    db_info = DATABASES[db_key]
    db_path = db_info['path']

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return

    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting dashboard generation for {db_key}")
    print(f"{'='*70}")

    try:
        ai = AnalyticalIntelligence(str(db_path))

        # Initialize cache for this DB if needed
        if db_key not in dashboard_cache:
            with cache_lock:
                dashboard_cache[db_key] = {}

        total_dashboards = len(DASHBOARDS)

        for idx, (dashboard_name, question) in enumerate(DASHBOARDS, 1):
            status_key = f"{db_key}:{dashboard_name}"
            generation_status[status_key] = {
                'progress': (idx - 1) / total_dashboards * 100,
                'current': idx,
                'total': total_dashboards,
                'dashboard': dashboard_name,
                'status': 'generating'
            }

            print(f"[{idx}/{total_dashboards}] Generating {dashboard_name}...", end=' ', flush=True)

            try:
                t0 = time.time()
                result = ai.analyze(question)
                elapsed = round((time.time() - t0) * 1000)

                dashboard_html = result.get('dashboard_html', '')

                # Count rows analyzed (estimate from result)
                rows_analyzed = result.get('row_count', 0)

                with cache_lock:
                    dashboard_cache[db_key][dashboard_name] = {
                        'html': dashboard_html,
                        'generated_at': datetime.now().isoformat(),
                        'generation_time_ms': elapsed,
                        'rows_analyzed': rows_analyzed,
                        'db_size': format_size(get_db_size_bytes(db_path)),
                        'db_rows': count_db_rows(db_path),
                        'error': None
                    }

                print(f"OK ({elapsed}ms)")

            except Exception as e:
                error_msg = str(e)
                print(f"ERROR: {error_msg}")
                with cache_lock:
                    dashboard_cache[db_key][dashboard_name] = {
                        'html': '',
                        'generated_at': datetime.now().isoformat(),
                        'generation_time_ms': 0,
                        'rows_analyzed': 0,
                        'db_size': format_size(get_db_size_bytes(db_path)),
                        'db_rows': count_db_rows(db_path),
                        'error': error_msg
                    }

            generation_status[status_key]['progress'] = (idx / total_dashboards) * 100
            generation_status[status_key]['status'] = 'complete'

        print(f"\n{'='*70}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed {db_key} dashboard generation")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"FATAL ERROR during generation: {e}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
# HTML TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

def render_landing_page():
    """Render the professional landing page with all dashboard links."""
    dashboard_cards = '\n'.join(f'''
    <a href="/dashboard/{name}" class="dashboard-card">
        <div class="dashboard-icon">📊</div>
        <h3>{name.replace('_', ' ').title()}</h3>
        <p>{description}</p>
    </a>
    ''' for name, description in DASHBOARDS)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthcare Analytics Dashboard Hub</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            margin-bottom: 60px;
        }}

        .header h1 {{
            color: #1e3a5f;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 600;
        }}

        .header p {{
            color: #666;
            font-size: 1.1em;
            margin-bottom: 30px;
        }}

        .db-selector {{
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 40px;
        }}

        .db-button {{
            padding: 12px 24px;
            border: 2px solid #ddd;
            background: white;
            color: #333;
            font-size: 1em;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 500;
        }}

        .db-button:hover {{
            border-color: #1e3a5f;
            background: #f0f4f8;
        }}

        .db-button.active {{
            background: #1e3a5f;
            color: white;
            border-color: #1e3a5f;
        }}

        .dashboards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 25px;
            margin-top: 40px;
        }}

        .dashboard-card {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            text-decoration: none;
            color: #333;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            transition: all 0.3s ease;
            border-left: 5px solid #1e3a5f;
            cursor: pointer;
        }}

        .dashboard-card:hover {{
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
            transform: translateY(-4px);
            border-left-color: #0066cc;
        }}

        .dashboard-icon {{
            font-size: 2.5em;
            margin-bottom: 15px;
        }}

        .dashboard-card h3 {{
            color: #1e3a5f;
            margin-bottom: 10px;
            font-size: 1.2em;
        }}

        .dashboard-card p {{
            color: #666;
            font-size: 0.95em;
            line-height: 1.5;
        }}

        .footer {{
            text-align: center;
            margin-top: 60px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Healthcare Analytics Hub</h1>
            <p>Real-time dashboards powered by intelligent analysis</p>
        </div>

        <div class="db-selector">
            <button class="db-button active" onclick="selectDb('71k')">71K Members</button>
            <button class="db-button" onclick="selectDb('4m')">4M Members</button>
        </div>

        <div class="dashboards-grid">
            {dashboard_cards}
        </div>

        <div class="footer">
            <p>All dashboards refresh with real-time AnalyticalIntelligence analysis</p>
        </div>
    </div>

    <script>
        // Store selected DB in localStorage
        let selectedDb = localStorage.getItem('selectedDb') || '71k';

        function selectDb(db) {{
            selectedDb = db;
            localStorage.setItem('selectedDb', db);
            document.querySelectorAll('.db-button').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }}

        // Update all dashboard links to include ?db=<selected>
        document.querySelectorAll('.dashboard-card').forEach(card => {{
            card.addEventListener('click', (e) => {{
                const href = card.getAttribute('href');
                if (href && !href.includes('?')) {{
                    window.location.href = href + '?db=' + selectedDb;
                }}
            }});
        }});
    </script>
</body>
</html>'''


def render_dashboard_page(dashboard_name, db_key, is_generating=False, error_msg=None):
    """Render a dashboard page with DB selector header and content."""
    db_info = DATABASES.get(db_key, DATABASES['71k'])

    if error_msg:
        content_html = f'''
        <div style="padding: 40px; text-align: center;">
            <div style="background: #fee; color: #c33; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h2>Error Generating Dashboard</h2>
                <p>{escape_html(error_msg)}</p>
            </div>
        </div>
        '''
    elif is_generating:
        content_html = '''
        <div style="padding: 40px; text-align: center;">
            <div style="display: inline-block;">
                <div style="width: 50px; height: 50px; border: 4px solid #ddd; border-top-color: #1e3a5f; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 20px;"></div>
                <h2 style="color: #1e3a5f;">Generating Dashboard...</h2>
                <p style="color: #666; margin-top: 10px;">Analyzing data from {db_label}...</p>
            </div>
        </div>
        <style>
            @keyframes spin {{
                to {{ transform: rotate(360deg); }}
            }}
        </style>
        '''
    else:
        cached = dashboard_cache.get(db_key, {}).get(dashboard_name, {})
        dashboard_html = cached.get('html', '')
        metadata = {
            'generated_at': cached.get('generated_at', 'N/A'),
            'generation_time_ms': cached.get('generation_time_ms', 0),
            'db_size': cached.get('db_size', 'Unknown'),
            'db_rows': cached.get('db_rows', 0),
        }

        metadata_html = f'''
        <div style="background: #f8f9fa; padding: 15px 20px; font-size: 0.9em; color: #666; border-bottom: 1px solid #eee;">
            <span style="margin-right: 20px;">Generated: {metadata['generated_at']}</span>
            <span style="margin-right: 20px;">Time: {metadata['generation_time_ms']}ms</span>
            <span style="margin-right: 20px;">DB Size: {metadata['db_size']}</span>
            <span>Rows Analyzed: {metadata['db_rows']:,}</span>
        </div>
        '''

        content_html = metadata_html + dashboard_html

    db_label = db_info['label']

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dashboard_name.replace('_', ' ').title()} - Healthcare Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f7fa;
        }}

        .sticky-header {{
            position: sticky;
            top: 0;
            z-index: 1000;
            background: {db_info['color']};
            color: white;
            padding: 15px 30px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}

        .header-content {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header-left {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}

        .header-left a {{
            color: white;
            text-decoration: none;
            font-size: 1.2em;
            font-weight: 600;
        }}

        .header-left a:hover {{
            opacity: 0.8;
        }}

        .db-selector {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        .db-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}

        select {{
            padding: 8px 12px;
            border: none;
            border-radius: 4px;
            background: white;
            color: #333;
            font-size: 0.95em;
            cursor: pointer;
        }}

        .button {{
            padding: 8px 16px;
            background: white;
            color: {db_info['color']};
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.3s;
        }}

        .button:hover {{
            background: #f0f0f0;
        }}

        .dashboard-content {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
        }}
    </style>
</head>
<body>
    <div class="sticky-header">
        <div class="header-content">
            <div class="header-left">
                <a href="/">← Back</a>
                <h1 style="font-size: 1.3em; font-weight: 500;">{escape_html(dashboard_name.replace('_', ' ').title())}</h1>
            </div>
            <div class="db-selector">
                <span class="db-label">Database:</span>
                <select onchange="switchDb(this.value)">
                    <option value="71k" {'selected' if db_key == '71k' else ''}>71K Members</option>
                    <option value="4m" {'selected' if db_key == '4m' else ''}>4M Members</option>
                </select>
                <button class="button" onclick="regenerate()">Regenerate</button>
            </div>
        </div>
    </div>

    <div class="dashboard-content">
        {content_html}
    </div>

    <script>
        function switchDb(db) {{
            const currentDashboard = '{escape_html(dashboard_name)}';
            window.location.href = '/dashboard/' + currentDashboard + '?db=' + db;
        }}

        function regenerate() {{
            const currentDashboard = '{escape_html(dashboard_name)}';
            const currentDb = new URLSearchParams(window.location.search).get('db') || '71k';
            window.location.href = '/dashboard/' + currentDashboard + '?db=' + currentDb + '&regenerate=1';
        }}
    </script>
</body>
</html>'''


# ═══════════════════════════════════════════════════════════════════════════
# HTTP REQUEST HANDLER
# ═══════════════════════════════════════════════════════════════════════════

class DashboardHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for dashboard server."""

    def log_message(self, format, *args):
        """Custom logging."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {format % args}")

    def do_GET(self):
        """Handle GET requests."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        # Extract database selection (default to 71k)
        db_key = query_params.get('db', ['71k'])[0]
        if db_key not in DATABASES:
            db_key = '71k'

        # Route: /
        if path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(render_landing_page().encode('utf-8'))

        # Route: /dashboard/<name>
        elif path.startswith('/dashboard/'):
            dashboard_name = path.replace('/dashboard/', '').strip('/')

            # Validate dashboard name
            valid_names = [name for name, _ in DASHBOARDS]
            if dashboard_name not in valid_names:
                self.send_response(404)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<h1>Dashboard not found</h1>')
                return

            # Check if regenerate is requested
            force_regenerate = 'regenerate' in query_params

            # Check if we have cached version
            cached = dashboard_cache.get(db_key, {}).get(dashboard_name)

            if cached and not force_regenerate:
                # Serve from cache
                if cached.get('error'):
                    html_content = render_dashboard_page(dashboard_name, db_key, error_msg=cached['error'])
                else:
                    html_content = render_dashboard_page(dashboard_name, db_key, is_generating=False)

                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(html_content.encode('utf-8'))
            else:
                # Generate in background if not already generating
                status_key = f"{db_key}:{dashboard_name}"

                if status_key not in generation_status or force_regenerate:
                    generation_status[status_key] = {
                        'status': 'queued',
                        'progress': 0
                    }
                    thread = threading.Thread(
                        target=generate_dashboard_background,
                        args=(db_key,),
                        daemon=True
                    )
                    thread.start()

                # Show loading page
                html_content = render_dashboard_page(dashboard_name, db_key, is_generating=True)
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.send_header('Refresh', '2')  # Auto-refresh every 2 seconds
                self.end_headers()
                self.wfile.write(html_content.encode('utf-8'))

        # Route: /api/status
        elif path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            status_json = json.dumps(generation_status)
            self.wfile.write(status_json.encode('utf-8'))

        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>Not Found</h1>')

    def log_error(self, format, *args):
        """Suppress default error logging."""
        pass


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SERVER
# ═══════════════════════════════════════════════════════════════════════════

def start_server():
    """Start the HTTP server."""
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, DashboardHandler)

    print("\n" + "="*70)
    print("Healthcare Analytics Dashboard Server")
    print("="*70)
    print(f"Server running on http://localhost:{PORT}")
    print(f"Root directory: {BASE_DIR}")
    print("\nAvailable dashboards: {len(DASHBOARDS)}")
    for name, desc in DASHBOARDS:
        print(f"  - {name}")
    print("\nDatabases:")
    for db_key, db_info in DATABASES.items():
        exists = "✓" if db_info['path'].exists() else "✗"
        print(f"  [{exists}] {db_key}: {db_info['label']} ({db_info['size']})")
    print("="*70)
    print("Press Ctrl+C to stop server\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        httpd.server_close()


if __name__ == '__main__':
    start_server()
