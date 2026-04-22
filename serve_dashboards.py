#!/usr/bin/env python3
"""
Lightweight Dashboard Server
============================
Serves pre-generated healthcare dashboard HTML files.
Zero external dependencies — uses only Python standard library.

Usage:
    python3 serve_dashboards.py                  # port 8080
    python3 serve_dashboards.py --port 9090      # custom port

Then open: http://localhost:8080
"""

import http.server
import socketserver
import os
import sys
import argparse
import signal
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent


def build_index_html():
    """Auto-generate a hub page listing all available dashboards."""
    sections = {
        "71K Dataset Dashboards": "dashboards_71k",
        "4M Dataset Dashboards": "dashboards_4m",
        "Combined Dashboards": "dashboards",
    }

    cards = []
    for title, folder in sections.items():
        folder_path = BASE_DIR / folder
        if not folder_path.is_dir():
            continue
        files = sorted(f.name for f in folder_path.glob("*.html"))
        if not files:
            continue
        links = ""
        for f in files:
            nice = f.replace("_dashboard.html", "").replace("_", " ").title()
            if f == "index.html":
                nice = "Hub / Index"
            links += f'<li><a href="/{folder}/{f}">{nice}</a></li>\n'
        cards += [f"<h2>{title}</h2><ul>{links}</ul>"]

    # Also check for standalone HTML in the base directory
    standalone = sorted(f.name for f in BASE_DIR.glob("*.html") if f.name != "index.html")
    if standalone:
        links = ""
        for f in standalone:
            nice = f.replace("_dashboard.html", "").replace(".html", "").replace("_", " ").title()
            links += f'<li><a href="/{f}">{nice}</a></li>\n'
        cards += [f"<h2>Other Reports</h2><ul>{links}</ul>"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Healthcare Analytics Dashboard Hub</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f7fa; color: #1a1a2e; padding: 2rem; }}
  h1 {{ text-align:center; color:#002B5C; margin-bottom:0.5rem; font-size:1.8rem; }}
  .subtitle {{ text-align:center; color:#666; margin-bottom:2rem; font-size:0.95rem; }}
  .container {{ max-width:900px; margin:0 auto; }}
  h2 {{ color:#002B5C; border-bottom:2px solid #0066cc; padding-bottom:0.3rem; margin:1.5rem 0 0.8rem; }}
  ul {{ list-style:none; display:grid; grid-template-columns:repeat(auto-fill, minmax(280px,1fr)); gap:0.5rem; }}
  li a {{ display:block; padding:0.7rem 1rem; background:#fff; border-radius:8px;
          text-decoration:none; color:#0066cc; border:1px solid #e0e0e0;
          transition: all 0.15s ease; }}
  li a:hover {{ background:#002B5C; color:#fff; transform:translateY(-1px);
               box-shadow:0 2px 8px rgba(0,0,0,0.12); }}
  .footer {{ text-align:center; color:#999; font-size:0.8rem; margin-top:2rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>Healthcare Analytics Dashboard Hub</h1>
  <p class="subtitle">Select a dashboard to view</p>
  {''.join(cards)}
  <p class="footer">Served by serve_dashboards.py</p>
</div>
</body>
</html>"""


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves from BASE_DIR with a generated index."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        # Serve auto-generated index at root
        if self.path in ("/", "/index.html"):
            content = build_index_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
            return
        # Everything else: serve static files normally
        super().do_GET()

    def log_message(self, format, *args):
        # Cleaner logging
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="Serve healthcare dashboards")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()

    port = args.port

    # Allow quick restart without "Address already in use" error
    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(("0.0.0.0", port), DashboardHandler) as httpd:
            print(f"\n{'='*55}")
            print(f"  Healthcare Dashboard Server")
            print(f"  Serving files from: {BASE_DIR}")
            print(f"{'='*55}")
            print(f"\n  Open in browser:")
            print(f"    http://localhost:{port}")
            print(f"\n  Press Ctrl+C to stop\n")

            # Handle Ctrl+C gracefully
            signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
            httpd.serve_forever()

    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n  ERROR: Port {port} is already in use.")
            print(f"  Try: python3 serve_dashboards.py --port {port + 1}")
            print(f"  Or kill the existing process: lsof -ti:{port} | xargs kill -9\n")
        else:
            print(f"\n  ERROR: {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
