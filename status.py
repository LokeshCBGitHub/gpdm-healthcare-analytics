#!/usr/bin/env python3
"""Check encryption status without attempting decryption."""
from file_guardian import FileGuardian
from pathlib import Path

guardian = FileGuardian()
state = guardian._get_auth_state()

enc_count = len(list(guardian.scripts_dir.glob("*.py.enc")))
py_count = len(list(guardian.scripts_dir.glob("*.py")))

print("=" * 50)
print("  GPDM File Protection Status")
print("=" * 50)
print(f"  Encrypted files (.enc): {enc_count}")
print(f"  Plain Python files:     {py_count}")
print(f"  Failed attempts:        {state['attempts']}")
print(f"  System locked:          {state['locked']}")
print(f"  Salt file exists:       {guardian.salt_file.exists()}")
print(f"  Max attempts before destruct: {guardian.MAX_ATTEMPTS}")
print("=" * 50)

if state['locked']:
    print("\n  *** SYSTEM IS LOCKED — next decrypt triggers SELF-DESTRUCT ***")
elif state['attempts'] > 0:
    remaining = guardian.MAX_ATTEMPTS - state['attempts']
    print(f"\n  *** WARNING: {remaining} attempt(s) remaining ***")
elif enc_count > 0:
    print("\n  System is encrypted and secure.")
else:
    print("\n  System is NOT encrypted. Run encrypt.py to protect files.")
