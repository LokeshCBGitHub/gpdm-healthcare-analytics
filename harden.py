#!/usr/bin/env python3
"""
Run once after deployment to lock down the production server.
Removes all traces — helper scripts, readable code, git history.
After this, only start.py works.

Usage: python3 harden.py
"""
import os
import sys
import shutil
import py_compile
from pathlib import Path

BASE = Path(__file__).parent

print("Hardening production deployment...\n")

# 1. Remove helper scripts that reveal how the system works
remove_files = [
    'encrypt.py', 'unlock.py', 'status.py', 'harden.py',
    'protect.py',  # old version if exists
    '.gitignore', '.git',
]
for f in remove_files:
    p = BASE / f
    if p.is_file():
        os.remove(p)
        print(f"  Removed: {f}")
    elif p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
        print(f"  Removed: {f}/")

# 2. Compile file_guardian.py and start.py to .pyc, remove .py source
#    The .pyc bytecode is much harder to read than source
for script in ['file_guardian.py', 'start.py']:
    src = BASE / script
    if src.exists():
        # Compile to __pycache__
        py_compile.compile(str(src), cfile=str(BASE / f"{script}c"), doraise=True)
        os.remove(src)
        print(f"  Compiled {script} → {script}c (source removed)")

# 3. Create a minimal launcher that runs the compiled .pyc
launcher = BASE / 'start.py'
launcher.write_text('''#!/usr/bin/env python3
import importlib.util, sys, os
_d = os.path.dirname(os.path.abspath(__file__))
os.chdir(_d)
spec = importlib.util.spec_from_file_location("start", os.path.join(_d, "start.pyc"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.main()
''')
print("  Created minimal start.py launcher")

# 4. Lock down ALL file permissions
#    Owner-only: no group, no others can read/write/execute
for root, dirs, files in os.walk(BASE):
    for d in dirs:
        os.chmod(os.path.join(root, d), 0o700)
    for f in files:
        fp = os.path.join(root, f)
        if f == 'start.py':
            os.chmod(fp, 0o700)  # executable
        else:
            os.chmod(fp, 0o600)  # read-only by owner

print("  Set all permissions to owner-only (700/600)")

# 5. Remove any __pycache__ directories (Python cache can leak code)
for root, dirs, files in os.walk(BASE):
    for d in dirs:
        if d == '__pycache__':
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
print("  Cleared all __pycache__ directories")

# 6. Remove any .pyc files in scripts/ that might have been cached
for pyc in (BASE / 'scripts').rglob('*.pyc'):
    os.remove(pyc)
print("  Cleared any cached .pyc in scripts/")

print(f"\n{'='*50}")
print("  HARDENED. Only 'python3 start.py' works now.")
print("  All source code is compiled or encrypted.")
print("  All permissions locked to owner-only.")
print(f"{'='*50}\n")

# Self-destruct this script
os.remove(__file__)
