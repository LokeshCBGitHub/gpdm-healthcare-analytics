from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

_log = logging.getLogger(__name__)


_DEFAULT_SCHEMA_PATH = os.path.join(
    (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "phi_schema.json",
)
_DEFAULT_VAULT_PATH = os.path.join(
    (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "phi_vault.db",
)

_DEFAULT_TYPE_PREFIX = "TKN"


def _load_hmac_key() -> Optional[bytes]:
    h = os.environ.get("GPDM_PHI_HMAC_KEY", "").strip()
    if not h:
        return None
    try:
        raw = bytes.fromhex(h)
    except ValueError:
        _log.error("GPDM_PHI_HMAC_KEY is not valid hex; PHI tokenization disabled")
        return None
    if len(raw) < 32:
        _log.error("GPDM_PHI_HMAC_KEY must be >= 32 bytes (64 hex chars); PHI tokenization disabled")
        return None
    return raw


def _load_aes_key() -> Optional[bytes]:
    h = os.environ.get("GPDM_PHI_AES_KEY", "").strip()
    if not h:
        return None
    try:
        raw = bytes.fromhex(h)
    except ValueError:
        _log.error("GPDM_PHI_AES_KEY is not valid hex; vault encryption disabled")
        return None
    if len(raw) != 32:
        _log.error("GPDM_PHI_AES_KEY must be exactly 32 bytes (64 hex chars); vault encryption disabled")
        return None
    return raw


def generate_keys() -> Dict[str, str]:
    return {
        "GPDM_PHI_HMAC_KEY": secrets.token_hex(32),
        "GPDM_PHI_AES_KEY": secrets.token_hex(32),
    }


class PhiSchema:

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._col_rules: List[Tuple[re.Pattern, str]] = []
        for rule in config.get("column_name_rules", []):
            try:
                self._col_rules.append((re.compile(rule["pattern"], re.I), rule["type"]))
            except re.error as e:
                _log.warning("Bad column-name regex %r: %s", rule.get("pattern"), e)

        self._value_rules: List[Tuple[re.Pattern, str]] = []
        for rule in config.get("value_patterns", []):
            try:
                self._value_rules.append((re.compile(rule["pattern"]), rule["type"]))
            except re.error as e:
                _log.warning("Bad value regex %r: %s", rule.get("pattern"), e)

        self._transforms: Dict[str, str] = dict(config.get("transforms", {}))

    @classmethod
    def load(cls, path: Optional[str] = None) -> "PhiSchema":
        path = path or _DEFAULT_SCHEMA_PATH
        if not os.path.exists(path):
            _log.warning("PHI schema not found at %s; using empty schema", path)
            return cls({})
        with open(path, "r") as f:
            return cls(json.load(f))

    def classify_column(self, col_name: str) -> Optional[str]:
        for pat, typ in self._col_rules:
            if pat.search(col_name or ""):
                return typ
        return None

    def classify_value(self, val: str) -> Optional[str]:
        if not val or not isinstance(val, str):
            return None
        for pat, typ in self._value_rules:
            if pat.search(val):
                return typ
        return None

    def transform_for(self, phi_type: str) -> str:
        return self._transforms.get(phi_type, "token")

    def all_types(self) -> List[str]:
        return sorted({t for _, t in self._col_rules} | {t for _, t in self._value_rules})


def _aes_encrypt(key: bytes, plaintext: bytes, aad: bytes = b"") -> Optional[bytes]:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = os.urandom(12)
        ct = AESGCM(key).encrypt(nonce, plaintext, aad or None)
        return nonce + ct
    except ImportError:
        return None


def _aes_decrypt(key: bytes, blob: bytes, aad: bytes = b"") -> Optional[bytes]:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce, ct = blob[:12], blob[12:]
        return AESGCM(key).decrypt(nonce, ct, aad or None)
    except ImportError:
        return None


class PhiTokenizer:

    def __init__(self,
                 schema: Optional[PhiSchema] = None,
                 vault_path: Optional[str] = None):
        self._schema = schema or PhiSchema.load()
        self._vault_path = vault_path or _DEFAULT_VAULT_PATH
        self._hmac_key = _load_hmac_key()
        self._aes_key = _load_aes_key()
        self._lock = threading.Lock()
        self._vault_initialized = False
        self._vault_conn: Optional[sqlite3.Connection] = None
        self._vault_uncommitted = 0
        self._vault_commit_every = int(os.environ.get("GPDM_PHI_VAULT_COMMIT_EVERY", "5000"))
        self._stats = {
            "tokens_issued": 0,
            "vault_writes": 0,
            "vault_reads": 0,
            "collisions_detected": 0,
        }
        if not self._hmac_key:
            _log.warning(
                "PHI_HMAC_KEY missing — tokens will use a RANDOM ephemeral key. "
                "DO NOT run this in production; joins will break across restarts. "
                "Run phi_tokenizer.generate_keys() to provision keys."
            )
            self._hmac_key = secrets.token_bytes(32)
            self._ephemeral = True
        else:
            self._ephemeral = False
        if not self._aes_key:
            _log.warning(
                "PHI_AES_KEY missing — vault writes will store an UNENCRYPTED "
                "plaintext column.  Detokenize() will refuse.  "
                "This is acceptable for dev only."
            )


    @property
    def ephemeral(self) -> bool:
        return self._ephemeral

    def classify_column(self, col_name: str) -> Optional[str]:
        return self._schema.classify_column(col_name)

    def is_sensitive_column(self, col_name: str) -> bool:
        return self._schema.classify_column(col_name) is not None

    def tokenize(self, value: Any, phi_type: str = _DEFAULT_TYPE_PREFIX,
                 source_table: str = "", source_col: str = "") -> str:
        if value is None:
            return None
        sval = str(value).strip()
        if not sval:
            return sval

        transform = self._schema.transform_for(phi_type)
        if transform == "year_only":
            return self._year_only(sval)
        if transform == "zip3":
            return self._zip3(sval)
        tok = self._hmac_token(sval, phi_type)
        self._vault_upsert(tok, phi_type, sval, source_table, source_col)
        return tok

    def mask_display(self, value: Any, phi_type: str = _DEFAULT_TYPE_PREFIX) -> str:
        if value is None:
            return ""
        sval = str(value)
        if len(sval) <= 4:
            return "*" * len(sval)
        return "*" * (len(sval) - 4) + sval[-4:]

    def detokenize(self, token: str) -> Optional[str]:
        if not self._aes_key:
            _log.error("detokenize: AES key missing; refusing")
            return None
        if self._ephemeral:
            _log.error("detokenize: running with ephemeral HMAC key; refusing")
            return None
        self._ensure_vault()
        conn = self._open_vault()
        try:
            row = conn.execute(
                "SELECT value_cipher FROM phi_vault WHERE token = ?", (token,)
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        blob = row[0]
        try:
            pt = _aes_decrypt(self._aes_key, blob, aad=token.encode("utf-8"))
        except Exception as e:
            _log.error("detokenize: AES decrypt failed: %s", e)
            return None
        if not pt:
            return None
        with self._lock:
            self._stats["vault_reads"] += 1
        return pt.decode("utf-8")

    def mask_text(self, text: str) -> str:
        if not text:
            return text
        out = text
        for pat, typ in self._schema._value_rules:
            def _sub(m, t=typ):
                return self.tokenize(m.group(0), phi_type=t,
                                     source_table="<inline>", source_col="<text>")
            out = pat.sub(_sub, out)
        return out

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)


    def mask_row(self, row: Dict[str, Any],
                 schema_hints: Dict[str, str],
                 source_table: str = "") -> Tuple[Dict[str, Any], List[str]]:
        out = dict(row)
        touched: List[str] = []
        for col, val in row.items():
            phi_type = schema_hints.get(col) or self.classify_column(col)
            if not phi_type:
                continue
            touched.append(col)
            out[col] = self.tokenize(val, phi_type=phi_type,
                                     source_table=source_table, source_col=col)
        return out, touched


    def _hmac_token(self, value: str, phi_type: str) -> str:
        norm = re.sub(r"\s+", "", value).strip()
        mac = hmac.new(self._hmac_key, f"{phi_type}|{norm}".encode("utf-8"),
                       hashlib.sha256).digest()
        b32 = base64.b32encode(mac).decode("ascii").rstrip("=")
        short = b32[:9]
        with self._lock:
            self._stats["tokens_issued"] += 1
        return f"{phi_type}-{short}"

    def _year_only(self, value: str) -> str:
        m = re.search(r"(\d{4})", value)
        return m.group(1) + "-01-01" if m else ""

    def _zip3(self, value: str) -> str:
        digits = re.sub(r"\D", "", value)
        return digits[:3] + "**" if len(digits) >= 3 else "***"

    def _ensure_vault(self) -> None:
        if self._vault_initialized:
            return
        os.makedirs(os.path.dirname(self._vault_path), exist_ok=True)
        conn = sqlite3.connect(self._vault_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS phi_vault (
                    token          TEXT PRIMARY KEY,
                    phi_type       TEXT NOT NULL,
                    value_cipher   BLOB,
                    value_preview  TEXT,      -- last-4 only, for debugging
                    source_table   TEXT,
                    source_col     TEXT,
                    first_seen     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hit_count      INTEGER DEFAULT 1
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_phi_vault_type ON phi_vault(phi_type)"
            )
            conn.commit()
        finally:
            conn.close()
        try:
            os.chmod(self._vault_path, 0o600)
        except OSError:
            pass
        self._vault_initialized = True

    def _open_vault(self) -> sqlite3.Connection:
        self._ensure_vault()
        conn = sqlite3.connect(self._vault_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _vault_upsert(self, token: str, phi_type: str, value: str,
                      source_table: str, source_col: str) -> None:
        self._ensure_vault()
        cipher = None
        if self._aes_key:
            cipher = _aes_encrypt(self._aes_key, value.encode("utf-8"),
                                  aad=token.encode("utf-8"))
        preview = "*" * max(0, len(value) - 4) + value[-4:]

        with self._lock:
            if self._vault_conn is None:
                self._vault_conn = sqlite3.connect(self._vault_path, timeout=30,
                                                   isolation_level="DEFERRED")
                self._vault_conn.execute("PRAGMA journal_mode=WAL")
                self._vault_conn.execute("PRAGMA synchronous=NORMAL")
            conn = self._vault_conn
            try:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO phi_vault "
                    "(token, phi_type, value_cipher, value_preview, source_table, source_col) "
                    "VALUES (?,?,?,?,?,?)",
                    (token, phi_type, cipher, preview, source_table, source_col),
                )
                if cur.rowcount == 0:
                    row = conn.execute(
                        "SELECT value_preview, phi_type FROM phi_vault WHERE token = ?",
                        (token,),
                    ).fetchone()
                    if row and (row[0] != preview or row[1] != phi_type):
                        self._stats["collisions_detected"] += 1
                        _log.error(
                            "PHI token collision on %s: existing preview=%s new=%s",
                            token, row[0], preview,
                        )
                    conn.execute(
                        "UPDATE phi_vault SET hit_count = hit_count + 1 WHERE token = ?",
                        (token,),
                    )
                else:
                    self._stats["vault_writes"] += 1
                self._vault_uncommitted += 1
                if self._vault_uncommitted >= self._vault_commit_every:
                    conn.commit()
                    self._vault_uncommitted = 0
            except sqlite3.Error as e:
                _log.error("vault upsert failed: %s", e)

    def flush_vault(self) -> None:
        with self._lock:
            if self._vault_conn is not None and self._vault_uncommitted:
                try:
                    self._vault_conn.commit()
                except sqlite3.Error as e:
                    _log.error("vault commit failed: %s", e)
                self._vault_uncommitted = 0


_instance: Optional[PhiTokenizer] = None
_instance_lock = threading.Lock()


def get_tokenizer() -> PhiTokenizer:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = PhiTokenizer()
        return _instance


if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="PHI tokenizer admin CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen-keys", help="Print a fresh pair of keys")
    t = sub.add_parser("tokenize", help="Tokenize a value")
    t.add_argument("type"); t.add_argument("value")
    d = sub.add_parser("detokenize", help="Reverse a token (requires both keys)")
    d.add_argument("token")
    s = sub.add_parser("stats", help="Show vault stats")

    args = p.parse_args()
    if args.cmd == "gen-keys":
        keys = generate_keys()
        for k, v in keys.items():
            print(f"export {k}={v}")
        sys.exit(0)

    tok = get_tokenizer()
    if args.cmd == "tokenize":
        print(tok.tokenize(args.value, phi_type=args.type))
    elif args.cmd == "detokenize":
        out = tok.detokenize(args.token)
        if out is None:
            print("(unable to detokenize)")
            sys.exit(2)
        print(out)
    elif args.cmd == "stats":
        print(json.dumps(tok.stats(), indent=2))
