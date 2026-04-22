import os
import sys
import argparse
import threading
import time
import signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(description='Launch public healthcare dashboard')
    parser.add_argument('--port', type=int, default=8787, help='Local port (default: 8787)')
    parser.add_argument('--ngrok-token', type=str, default='', help='ngrok authtoken')
    parser.add_argument('--no-tunnel', action='store_true', help='Skip ngrok, local only')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  MTP Healthcare Analytics — Public Server")
    print(f"{'='*60}\n")

    print(f"[1/3] Starting dashboard server on port {args.port}...")
    from dashboard_server import launch_dashboard
    server_thread = threading.Thread(
        target=launch_dashboard,
        kwargs={'port': args.port},
        daemon=True
    )
    server_thread.start()
    time.sleep(2)
    print(f"  Local: http://localhost:{args.port}")

    if args.no_tunnel:
        print(f"\n  Dashboard running locally (no tunnel).")
        print(f"  Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            return

    print(f"\n[2/3] Starting ngrok tunnel...")
    tunnel_url = None

    try:
        from pyngrok import ngrok, conf

        if args.ngrok_token:
            conf.get_default().auth_token = args.ngrok_token

        tunnel = ngrok.connect(args.port, "http")
        tunnel_url = tunnel.public_url
        if tunnel_url.startswith('http://'):
            tunnel_url = tunnel_url.replace('http://', 'https://')

    except ImportError:
        import subprocess
        import json as _json

        if args.ngrok_token:
            subprocess.run(['ngrok', 'config', 'add-authtoken', args.ngrok_token],
                         capture_output=True)

        try:
            proc = subprocess.Popen(
                ['ngrok', 'http', str(args.port), '--log=stdout', '--log-format=json'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            for _ in range(30):
                time.sleep(1)
                try:
                    api_resp = subprocess.run(
                        ['curl', '-s', 'http://localhost:4040/api/tunnels'],
                        capture_output=True, text=True
                    )
                    data = _json.loads(api_resp.stdout)
                    tunnels = data.get('tunnels', [])
                    for t in tunnels:
                        if t.get('proto') == 'https':
                            tunnel_url = t['public_url']
                            break
                    if tunnel_url:
                        break
                except Exception:
                    continue

        except FileNotFoundError:
            print("\n  ERROR: ngrok not found!")
            print("  Install with: pip install pyngrok")
            print("  Or: brew install ngrok  (macOS)")
            print(f"\n  Dashboard still running locally: http://localhost:{args.port}")
            print(f"  Press Ctrl+C to stop.\n")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                return

    if tunnel_url:
        try:
            from sso_auth import _load_config, save_config
            config = _load_config()
            config['redirect_uri'] = f"{tunnel_url}/auth/callback"
            save_config(config)
            print(f"  SSO redirect URI updated to: {tunnel_url}/auth/callback")
        except Exception:
            pass

        print(f"\n[3/3] Public URL ready!")
        print(f"\n{'='*60}")
        print(f"  SHAREABLE LINK (anyone can access):")
        print(f"")
        print(f"    {tunnel_url}")
        print(f"")
        print(f"  Share this link with your team. They'll see the")
        print(f"  login page and can sign up / use SSO to access.")
        print(f"{'='*60}")
        print(f"\n  Local:  http://localhost:{args.port}")
        print(f"  Public: {tunnel_url}")
        print(f"\n  Press Ctrl+C to stop.\n")
    else:
        print(f"\n  Tunnel failed. Dashboard running locally only.")
        print(f"  Local: http://localhost:{args.port}")
        print(f"  Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except Exception:
            pass


if __name__ == '__main__':
    main()
