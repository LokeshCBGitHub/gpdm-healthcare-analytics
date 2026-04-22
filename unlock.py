#!/usr/bin/env python3
"""
Decrypt all scripts — restores .py files from .enc.
Usage: python3 unlock.py
WARNING: 2 wrong passwords = ALL files permanently destroyed.
"""
import getpass
from file_guardian import FileGuardian
import json

print("=" * 60)
print("  GPDM File Decryption Tool")
print("  WARNING: 2 wrong passwords triggers SELF-DESTRUCT")
print("=" * 60)

guardian = FileGuardian()
state = guardian._get_auth_state()

if state.get("locked"):
    print("\n  SYSTEM IS LOCKED. Self-destruct will execute.")
    confirm = input("  Type 'YES' to proceed: ")
    if confirm == "YES":
        guardian.trigger_self_destruct()
    exit(1)

attempts = state.get("attempts", 0)
remaining = guardian.MAX_ATTEMPTS - attempts
print(f"\n  Remaining attempts: {remaining}")

password = getpass.getpass("\nEnter decryption password: ")
result = guardian.decrypt_all(password, write_to_disk=True)
print(f"\n{json.dumps(result, indent=2)}")

if "error" not in result:
    print(f"\nAll {result['decrypted']} files restored. Integrity: {result['integrity']}")
