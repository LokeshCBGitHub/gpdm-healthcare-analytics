#!/usr/bin/env python3
"""
Encrypt all scripts — run once to lock down the system.
Usage: python3 encrypt.py
"""
import getpass
from file_guardian import FileGuardian
import json

print("=" * 60)
print("  GPDM File Encryption Tool")
print("  AES-256 + PBKDF2HMAC (100K iterations)")
print("=" * 60)

password = getpass.getpass("\nEnter encryption password: ")
confirm = getpass.getpass("Confirm password: ")

if password != confirm:
    print("\nPasswords don't match. Aborting.")
    exit(1)

guardian = FileGuardian()
result = guardian.encrypt_all(password)
print(f"\n{json.dumps(result, indent=2)}")
print("\nAll scripts encrypted. Keep your password safe — 2 wrong attempts = PERMANENT deletion.")
