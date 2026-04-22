#!/usr/bin/env python3
"""
File Guardian - AES-256 Encrypted File Protection with Self-Destruct
====================================================================
- Encrypts all .py files in scripts/ (except dashboard_server.py)
- Uses AES-256 via Fernet with PBKDF2HMAC key derivation
- 2 wrong password attempts = PERMANENT deletion of ALL files
- Secure wipe: overwrite with random bytes before deletion
"""

import os
import sys
import json
import hashlib
import shutil
from pathlib import Path
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64


class FileGuardian:
    """AES-256 file protection with 2-attempt self-destruct."""

    EXCLUDED_FILES = {"dashboard_server.py", "__init__.py"}
    MAX_ATTEMPTS = 2

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.scripts_dir = self.base_dir / "scripts"
        self.auth_state_file = self.base_dir / ".auth_state"
        self.salt_file = self.base_dir / ".enc_salt"
        self.decrypted_cache = {}

    @staticmethod
    def derive_key(password: str, salt: bytes) -> bytes:
        """Derive AES-256 key from password using PBKDF2HMAC (100K iterations)."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100_000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def _get_auth_state(self) -> dict:
        """Read authentication state."""
        if not self.auth_state_file.exists():
            return {"attempts": 0, "locked": False}
        try:
            with open(self.auth_state_file, "r") as f:
                return json.load(f)
        except Exception:
            return {"attempts": 0, "locked": False}

    def _save_auth_state(self, attempts: int, locked: bool = False):
        """Save authentication state."""
        with open(self.auth_state_file, "w") as f:
            json.dump({"attempts": attempts, "locked": locked}, f)

    def _secure_wipe(self, path: Path):
        """Overwrite file with random bytes then delete."""
        try:
            if path.is_file():
                size = path.stat().st_size
                # 3-pass overwrite: random, zeros, random
                with open(path, "wb") as f:
                    f.write(os.urandom(max(size, 1)))
                with open(path, "wb") as f:
                    f.write(b'\x00' * max(size, 1))
                with open(path, "wb") as f:
                    f.write(os.urandom(max(size, 1)))
                os.remove(path)
        except Exception:
            try:
                os.remove(path)
            except Exception:
                pass

    def trigger_self_destruct(self):
        """
        FULL SELF-DESTRUCT: Permanently delete ALL protected files.
        3-pass secure wipe on every file, then remove directories.
        """
        print("\n" + "=" * 80)
        print("  SECURITY BREACH: 2 WRONG PASSWORDS — INITIATING SELF-DESTRUCT")
        print("=" * 80)

        targets = [
            self.scripts_dir,
            self.base_dir / "dashboards",
            self.base_dir / "dashboards_71k",
            self.base_dir / "dashboards_4m",
            self.base_dir / "config",
            self.base_dir / "semantic_catalog",
        ]

        destroyed = 0
        for target in targets:
            if not target.exists():
                continue
            if target.is_file():
                self._secure_wipe(target)
                destroyed += 1
            elif target.is_dir():
                for root, dirs, files in os.walk(target, topdown=False):
                    for f in files:
                        fp = Path(root) / f
                        self._secure_wipe(fp)
                        destroyed += 1
                    for d in dirs:
                        try:
                            os.rmdir(Path(root) / d)
                        except Exception:
                            pass
                shutil.rmtree(target, ignore_errors=True)

        # Also destroy database files
        for db in self.base_dir.glob("*.db"):
            if not db.is_symlink():
                self._secure_wipe(db)
                destroyed += 1
        for db in (self.base_dir / "data").glob("*.db"):
            if db.is_symlink():
                os.unlink(db)
            else:
                self._secure_wipe(db)
            destroyed += 1

        # Destroy auth state and salt
        for f in [self.auth_state_file, self.salt_file]:
            if f.exists():
                self._secure_wipe(f)

        print(f"  DESTROYED: {destroyed} files securely wiped")
        print("  ALL DATA HAS BEEN PERMANENTLY DELETED")
        print("=" * 80)
        sys.exit(1)

    def encrypt_all(self, password: str) -> dict:
        """
        Encrypt all .py files in scripts/ (except excluded ones).
        Returns dict with stats.
        """
        if not self.scripts_dir.exists():
            return {"error": f"Scripts directory not found: {self.scripts_dir}"}

        # Generate and save salt
        salt = os.urandom(32)
        with open(self.salt_file, "wb") as f:
            f.write(salt)

        key = self.derive_key(password, salt)
        cipher = Fernet(key)

        py_files = sorted(self.scripts_dir.glob("*.py"))
        encrypted = 0
        skipped = 0
        checksums = {}

        for py_file in py_files:
            if py_file.name in self.EXCLUDED_FILES:
                skipped += 1
                continue

            # Read original
            content = py_file.read_bytes()

            # Store SHA-256 checksum for integrity verification
            checksums[py_file.name] = hashlib.sha256(content).hexdigest()

            # Encrypt
            encrypted_content = cipher.encrypt(content)

            # Write .enc file
            enc_path = py_file.with_suffix('.py.enc')
            enc_path.write_bytes(encrypted_content)

            # Secure wipe original .py file
            self._secure_wipe(py_file)
            encrypted += 1

        # Save checksums for integrity verification on decrypt
        checksum_file = self.base_dir / ".checksums"
        checksum_data = cipher.encrypt(json.dumps(checksums).encode())
        checksum_file.write_bytes(checksum_data)

        # Reset auth state
        self._save_auth_state(0, False)

        return {
            "encrypted": encrypted,
            "skipped": skipped,
            "total": len(py_files),
            "salt_saved": True,
        }

    def decrypt_all(self, password: str, write_to_disk: bool = False) -> dict:
        """
        Decrypt all .py.enc files.
        If write_to_disk=True, writes .py files back. Otherwise stores in memory.
        Returns dict with stats.
        """
        state = self._get_auth_state()

        # Check if locked
        if state.get("locked", False):
            self.trigger_self_destruct()
            return {"error": "System locked — self-destruct triggered"}

        # Load salt
        if not self.salt_file.exists():
            return {"error": "Salt file not found — cannot derive key"}
        salt = self.salt_file.read_bytes()

        key = self.derive_key(password, salt)
        cipher = Fernet(key)

        enc_files = sorted(self.scripts_dir.glob("*.py.enc"))
        if not enc_files:
            return {"error": "No encrypted files found"}

        # Try to decrypt first file to validate password
        try:
            test_content = cipher.decrypt(enc_files[0].read_bytes())
        except InvalidToken:
            # WRONG PASSWORD
            attempts = state.get("attempts", 0) + 1
            remaining = self.MAX_ATTEMPTS - attempts

            if attempts >= self.MAX_ATTEMPTS:
                self._save_auth_state(attempts, True)
                self.trigger_self_destruct()
                return {"error": "Self-destruct triggered"}
            else:
                self._save_auth_state(attempts, False)
                print(f"WRONG PASSWORD. {remaining} attempt(s) remaining before self-destruct.")
                return {"error": f"Wrong password. {remaining} attempt(s) remaining."}

        # Password correct — decrypt all files
        decrypted = 0
        self.decrypted_cache = {}

        for enc_file in enc_files:
            try:
                content = cipher.decrypt(enc_file.read_bytes())
                original_name = enc_file.stem  # removes .enc, leaves .py
                if not original_name.endswith('.py'):
                    original_name = enc_file.name.replace('.enc', '')

                self.decrypted_cache[original_name] = content

                if write_to_disk:
                    out_path = enc_file.with_name(original_name)
                    out_path.write_bytes(content)
                    os.remove(enc_file)

                decrypted += 1
            except InvalidToken:
                print(f"WARNING: Could not decrypt {enc_file.name}")

        # Verify checksums if available
        checksum_file = self.base_dir / ".checksums"
        integrity_ok = True
        if checksum_file.exists():
            try:
                checksums = json.loads(cipher.decrypt(checksum_file.read_bytes()))
                for name, expected_hash in checksums.items():
                    if name in self.decrypted_cache:
                        actual_hash = hashlib.sha256(self.decrypted_cache[name]).hexdigest()
                        if actual_hash != expected_hash:
                            print(f"INTEGRITY FAIL: {name}")
                            integrity_ok = False
                if write_to_disk:
                    os.remove(checksum_file)
            except Exception:
                pass

        # Reset attempts on success
        self._save_auth_state(0, False)

        return {
            "decrypted": decrypted,
            "total_enc": len(enc_files),
            "integrity": "PASS" if integrity_ok else "FAIL",
            "mode": "disk" if write_to_disk else "memory",
        }

    def get_module(self, module_name: str):
        """Load a decrypted module from memory cache."""
        import importlib.util

        py_name = f"{module_name}.py"
        content = self.decrypted_cache.get(py_name)
        if content is None:
            raise ImportError(f"Module {module_name} not decrypted or not found")

        spec = importlib.util.spec_from_loader(module_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        exec(content, module.__dict__)
        return module


if __name__ == "__main__":
    import getpass

    guardian = FileGuardian()

    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        state = guardian._get_auth_state()
        enc_count = len(list(guardian.scripts_dir.glob("*.py.enc")))
        py_count = len(list(guardian.scripts_dir.glob("*.py")))
        print(f"Auth state: attempts={state['attempts']}, locked={state['locked']}")
        print(f"Encrypted files: {enc_count}")
        print(f"Plain .py files: {py_count}")
    else:
        password = getpass.getpass("Enter password: ") if len(sys.argv) < 2 else sys.argv[1]
        result = guardian.decrypt_all(password, write_to_disk=True)
        print(json.dumps(result, indent=2))
