#!/usr/bin/env python3
"""GPDM Healthcare Analytics Server — Direct Launch (No Encryption)"""
import sys, os

_D = os.path.dirname(os.path.abspath(__file__))
os.chdir(_D)
_S = os.path.join(_D, 'scripts')

if _S not in sys.path:
    sys.path.insert(0, _S)

def main():
    try:
        import dashboard_server
        server = dashboard_server.launch_dashboard(
            cfg={'APP_DIR': _D, 'DATA_DIR': os.path.join(_D, 'data'),
                 'RAW_DIR': os.path.join(_D, 'data', 'raw'),
                 'CATALOG_DIR': os.path.join(_D, 'semantic_catalog'),
                 'LOCAL_EXECUTION': '1', 'ENGINE_MODE': 'hybrid'},
            port=8787, host='0.0.0.0', force_http=True)
        if server:
            server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.\n")

if __name__ == "__main__":
    main()
