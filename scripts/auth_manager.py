import os
import json
import sqlite3
import hashlib
import secrets
import time
import logging
from typing import Optional, Dict, Any, List, Tuple

try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False


MIN_PASSWORD_LENGTH = 4
MIN_USERNAME_LENGTH = 2
SESSION_EXPIRY_DAYS = 7
PBKDF2_ITERATIONS = 100_000

AUTH_DB_PATH = os.environ.get('AUTH_DB_PATH', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'auth.db'
))


def _get_conn():
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id         TEXT PRIMARY KEY,
            team_name       TEXT NOT NULL UNIQUE,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS users (
            user_id         TEXT PRIMARY KEY,
            username        TEXT NOT NULL UNIQUE COLLATE NOCASE,
            email           TEXT UNIQUE COLLATE NOCASE,
            password_hash   TEXT NOT NULL,
            display_name    TEXT,
            team_id         TEXT,
            role            TEXT DEFAULT 'member',
            sso_provider    TEXT DEFAULT '',
            sso_provider_id TEXT DEFAULT '',
            profile_picture TEXT DEFAULT '',
            created_at      TEXT DEFAULT (datetime('now')),
            last_login      TEXT,
            FOREIGN KEY (team_id) REFERENCES teams(team_id)
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_token   TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL,
            created_at      TEXT DEFAULT (datetime('now')),
            expires_at      TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS saved_dashboards (
            dashboard_id    TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL,
            name            TEXT NOT NULL,
            description     TEXT DEFAULT '',
            queries         TEXT NOT NULL,  -- JSON array of {question, sql, timestamp}
            created_at      TEXT DEFAULT (datetime('now')),
            updated_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS export_passwords (
            export_pw_id    TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL,
            password_hash   TEXT NOT NULL,
            label           TEXT DEFAULT '',
            created_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS user_sessions (
            user_session_id TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL,
            session_name    TEXT DEFAULT '',
            queries         TEXT NOT NULL DEFAULT '[]',
            created_at      TEXT DEFAULT (datetime('now')),
            updated_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_dashboards_user ON saved_dashboards(user_id);
        CREATE INDEX IF NOT EXISTS idx_users_team ON users(team_id);
        CREATE INDEX IF NOT EXISTS idx_export_pw_user ON export_passwords(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id);
    """)
    conn.commit()

    try:
        existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        if 'email' not in existing_cols:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        if 'sso_provider' not in existing_cols:
            conn.execute("ALTER TABLE users ADD COLUMN sso_provider TEXT DEFAULT ''")
        if 'sso_provider_id' not in existing_cols:
            conn.execute("ALTER TABLE users ADD COLUMN sso_provider_id TEXT DEFAULT ''")
        if 'profile_picture' not in existing_cols:
            conn.execute("ALTER TABLE users ADD COLUMN profile_picture TEXT DEFAULT ''")
        conn.commit()
    except Exception as e:
        logging.getLogger('gpdm.auth').warning('Schema migration: %s', e)

    conn.close()


def _hash_password(password: str) -> str:
    if HAS_BCRYPT:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    else:
        salt = secrets.token_hex(16)
        h = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), PBKDF2_ITERATIONS)
        return f"pbkdf2:{salt}:{h.hex()}"


def _verify_password(password: str, hashed: str) -> bool:
    if HAS_BCRYPT and hashed.startswith('$2'):
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    elif hashed.startswith('pbkdf2:'):
        _, salt, expected_hex = hashed.split(':', 2)
        h = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), PBKDF2_ITERATIONS)
        return secrets.compare_digest(h.hex(), expected_hex)
    return False

def create_team(team_name: str) -> str:
    team_id = f"team_{secrets.token_hex(8)}"
    conn = _get_conn()
    try:
        conn.execute("INSERT INTO teams (team_id, team_name) VALUES (?, ?)", (team_id, team_name))
        conn.commit()
        return team_id
    except sqlite3.IntegrityError:
        row = conn.execute("SELECT team_id FROM teams WHERE team_name = ?", (team_name,)).fetchone()
        return row['team_id'] if row else team_id
    finally:
        conn.close()


def get_teams() -> List[Dict]:
    conn = _get_conn()
    rows = conn.execute("""
        SELECT t.team_id, t.team_name, COUNT(u.user_id) as member_count
        FROM teams t LEFT JOIN users u ON t.team_id = u.team_id
        GROUP BY t.team_id
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def signup(username: str, password: str, display_name: str = "",
           team_name: str = "", role: str = "member",
           email: str = "") -> Tuple[bool, str, Optional[str]]:
    if not username or len(username) < MIN_USERNAME_LENGTH:
        return False, f"Username must be at least {MIN_USERNAME_LENGTH} characters", None
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters", None
    if email and '@' not in email:
        return False, "Invalid email address", None

    conn = _get_conn()
    try:
        existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            return False, "Username already exists", None

        if email:
            existing_email = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
            if existing_email:
                return False, "Email already registered", None

        team_id = None
        if team_name:
            team_row = conn.execute("SELECT team_id FROM teams WHERE team_name = ?", (team_name,)).fetchone()
            if team_row:
                team_id = team_row['team_id']
            else:
                team_id = f"team_{secrets.token_hex(8)}"
                conn.execute("INSERT INTO teams (team_id, team_name) VALUES (?, ?)", (team_id, team_name))

        user_id = f"user_{secrets.token_hex(8)}"
        password_hash = _hash_password(password)

        conn.execute(
            "INSERT INTO users (user_id, username, email, password_hash, display_name, team_id, role) VALUES (?,?,?,?,?,?,?)",
            (user_id, username, email or None, password_hash, display_name or username, team_id, role)
        )

        token = _create_session(conn, user_id)
        conn.commit()
        return True, "Account created successfully", token

    except Exception as e:
        conn.rollback()
        return False, f"Registration failed: {str(e)}", None
    finally:
        conn.close()


def login(username_or_email: str, password: str) -> Tuple[bool, str, Optional[str]]:
    conn = _get_conn()
    try:
        row = conn.execute("SELECT user_id, password_hash FROM users WHERE username = ?",
                           (username_or_email,)).fetchone()
        if not row and '@' in username_or_email:
            row = conn.execute("SELECT user_id, password_hash FROM users WHERE email = ?",
                               (username_or_email,)).fetchone()
        if not row:
            return False, "Invalid username/email or password", None

        if not _verify_password(password, row['password_hash']):
            return False, "Invalid username/email or password", None

        conn.execute("UPDATE users SET last_login = datetime('now') WHERE user_id = ?", (row['user_id'],))

        token = _create_session(conn, row['user_id'])
        conn.commit()
        return True, "Login successful", token

    except Exception as e:
        return False, f"Login failed: {str(e)}", None
    finally:
        conn.close()


def sso_login_or_create(email: str, display_name: str = "", provider: str = "",
                        provider_id: str = "", picture: str = "",
                        team_name: str = "") -> Tuple[bool, str, Optional[str]]:
    if not email:
        return False, "Email is required for SSO login", None

    conn = _get_conn()
    try:
        row = conn.execute("SELECT user_id FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            conn.execute("""
                UPDATE users SET sso_provider = ?, sso_provider_id = ?, profile_picture = ?,
                    last_login = datetime('now')
                WHERE user_id = ?
            """, (provider, provider_id, picture, row['user_id']))
            token = _create_session(conn, row['user_id'])
            conn.commit()
            return True, "SSO login successful", token
        else:
            username = email.split('@')[0]
            base_username = username
            counter = 1
            while conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
                username = f"{base_username}{counter}"
                counter += 1

            team_id = None
            if team_name:
                team_row = conn.execute("SELECT team_id FROM teams WHERE team_name = ?", (team_name,)).fetchone()
                if team_row:
                    team_id = team_row['team_id']
                else:
                    team_id = f"team_{secrets.token_hex(8)}"
                    conn.execute("INSERT INTO teams (team_id, team_name) VALUES (?, ?)", (team_id, team_name))

            user_id = f"user_{secrets.token_hex(8)}"
            password_hash = _hash_password(secrets.token_urlsafe(32))

            conn.execute("""
                INSERT INTO users (user_id, username, email, password_hash, display_name,
                    team_id, role, sso_provider, sso_provider_id, profile_picture)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (user_id, username, email, password_hash,
                  display_name or username, team_id, 'member',
                  provider, provider_id, picture))

            token = _create_session(conn, user_id)
            conn.commit()
            return True, "Account created via SSO", token

    except Exception as e:
        conn.rollback()
        return False, f"SSO login failed: {str(e)}", None
    finally:
        conn.close()


def logout(session_token: str) -> bool:
    conn = _get_conn()
    conn.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))
    conn.commit()
    conn.close()
    return True


def validate_session(session_token: str) -> Optional[Dict]:
    if not session_token:
        return None
    conn = _get_conn()
    row = conn.execute("""
        SELECT u.user_id, u.username, u.email, u.display_name, u.role,
               u.sso_provider, u.profile_picture,
               t.team_id, t.team_name, s.expires_at
        FROM sessions s
        JOIN users u ON s.user_id = u.user_id
        LEFT JOIN teams t ON u.team_id = t.team_id
        WHERE s.session_token = ? AND s.expires_at > datetime('now')
    """, (session_token,)).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def _create_session(conn, user_id: str) -> str:
    conn.execute("DELETE FROM sessions WHERE user_id = ? OR expires_at < datetime('now')", (user_id,))
    token = secrets.token_urlsafe(48)
    conn.execute(
        f"INSERT INTO sessions (session_token, user_id, expires_at) VALUES (?, ?, datetime('now', '+{SESSION_EXPIRY_DAYS} days'))",
        (token, user_id)
    )
    return token

def save_dashboard(session_token: str, name: str, queries: List[Dict],
                   description: str = "", dashboard_id: str = None) -> Tuple[bool, str, Optional[str]]:
    user = validate_session(session_token)
    if not user:
        return False, "Not authenticated", None

    if not name or not name.strip():
        return False, "Dashboard name is required", None

    conn = _get_conn()
    try:
        if dashboard_id:
            conn.execute("""
                UPDATE saved_dashboards
                SET name = ?, description = ?, queries = ?, updated_at = datetime('now')
                WHERE dashboard_id = ? AND user_id = ?
            """, (name.strip(), description, json.dumps(queries), dashboard_id, user['user_id']))
        else:
            dashboard_id = f"dash_{secrets.token_hex(8)}"
            conn.execute("""
                INSERT INTO saved_dashboards (dashboard_id, user_id, name, description, queries)
                VALUES (?, ?, ?, ?, ?)
            """, (dashboard_id, user['user_id'], name.strip(), description, json.dumps(queries)))

        conn.commit()
        return True, "Dashboard saved", dashboard_id
    except Exception as e:
        conn.rollback()
        return False, f"Save failed: {str(e)}", None
    finally:
        conn.close()


def get_saved_dashboards(session_token: str) -> List[Dict]:
    user = validate_session(session_token)
    if not user:
        return []

    conn = _get_conn()
    rows = conn.execute("""
        SELECT dashboard_id, name, description, queries, created_at, updated_at
        FROM saved_dashboards
        WHERE user_id = ?
        ORDER BY updated_at DESC
    """, (user['user_id'],)).fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        d['queries'] = json.loads(d['queries']) if d['queries'] else []
        result.append(d)
    return result


def get_team_dashboards(session_token: str) -> List[Dict]:
    user = validate_session(session_token)
    if not user or not user.get('team_id'):
        return []

    conn = _get_conn()
    rows = conn.execute("""
        SELECT sd.dashboard_id, sd.name, sd.description, sd.queries,
               sd.created_at, sd.updated_at, u.username, u.display_name
        FROM saved_dashboards sd
        JOIN users u ON sd.user_id = u.user_id
        WHERE u.team_id = ?
        ORDER BY sd.updated_at DESC
    """, (user['team_id'],)).fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        d['queries'] = json.loads(d['queries']) if d['queries'] else []
        result.append(d)
    return result


def delete_dashboard(session_token: str, dashboard_id: str) -> Tuple[bool, str]:
    user = validate_session(session_token)
    if not user:
        return False, "Not authenticated"

    conn = _get_conn()
    cursor = conn.execute(
        "DELETE FROM saved_dashboards WHERE dashboard_id = ? AND user_id = ?",
        (dashboard_id, user['user_id'])
    )
    conn.commit()
    conn.close()
    return (True, "Deleted") if cursor.rowcount > 0 else (False, "Dashboard not found")


def set_export_password(session_token: str, password: str, label: str = "") -> Tuple[bool, str]:
    user = validate_session(session_token)
    if not user:
        return False, "Not authenticated"
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Export password must be at least {MIN_PASSWORD_LENGTH} characters"

    conn = _get_conn()
    try:
        conn.execute("DELETE FROM export_passwords WHERE user_id = ?", (user['user_id'],))
        pw_id = f"epw_{secrets.token_hex(8)}"
        pw_hash = _hash_password(password)
        conn.execute(
            "INSERT INTO export_passwords (export_pw_id, user_id, password_hash, label) VALUES (?,?,?,?)",
            (pw_id, user['user_id'], pw_hash, label)
        )
        conn.commit()
        return True, "Export password set"
    except Exception as e:
        conn.rollback()
        return False, f"Failed: {str(e)}"
    finally:
        conn.close()


def verify_export_password(session_token: str, password: str) -> bool:
    user = validate_session(session_token)
    if not user:
        return False
    conn = _get_conn()
    row = conn.execute(
        "SELECT password_hash FROM export_passwords WHERE user_id = ?",
        (user['user_id'],)
    ).fetchone()
    conn.close()
    if not row:
        return False
    return _verify_password(password, row['password_hash'])


def has_export_password(session_token: str) -> bool:
    user = validate_session(session_token)
    if not user:
        return False
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM export_passwords WHERE user_id = ?",
        (user['user_id'],)
    ).fetchone()
    conn.close()
    return row is not None


def create_user_session(session_token: str, session_name: str = "") -> Tuple[bool, str, Optional[str]]:
    user = validate_session(session_token)
    if not user:
        return False, "Not authenticated", None
    conn = _get_conn()
    try:
        sid = f"usess_{secrets.token_hex(8)}"
        conn.execute(
            "INSERT INTO user_sessions (user_session_id, user_id, session_name) VALUES (?,?,?)",
            (sid, user['user_id'], session_name or f"Session {time.strftime('%Y-%m-%d %H:%M')}")
        )
        conn.commit()
        return True, "Session created", sid
    except Exception as e:
        conn.rollback()
        return False, f"Failed: {str(e)}", None
    finally:
        conn.close()


def update_user_session(session_token: str, user_session_id: str, queries: List[Dict]) -> Tuple[bool, str]:
    user = validate_session(session_token)
    if not user:
        return False, "Not authenticated"
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE user_sessions SET queries = ?, updated_at = datetime('now') WHERE user_session_id = ? AND user_id = ?",
            (json.dumps(queries), user_session_id, user['user_id'])
        )
        conn.commit()
        return True, "Session updated"
    except Exception as e:
        conn.rollback()
        return False, f"Failed: {str(e)}"
    finally:
        conn.close()


def get_user_sessions(session_token: str) -> List[Dict]:
    user = validate_session(session_token)
    if not user:
        return []
    conn = _get_conn()
    rows = conn.execute("""
        SELECT user_session_id, session_name, queries, created_at, updated_at
        FROM user_sessions
        WHERE user_id = ?
        ORDER BY updated_at DESC
    """, (user['user_id'],)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['queries'] = json.loads(d['queries']) if d['queries'] else []
        result.append(d)
    return result


ROLES = {
    'admin': {
        'label': 'Admin',
        'description': 'Full access + user management',
        'permissions': {
            'insights',
            'analytics_models',
            'dashboards',
            'sql_editor',
            'export_csv',
            'export_email',
            'admin_panel',
            'view_all_dashboards',
        },
    },
    'business': {
        'label': 'Business User',
        'description': 'Full analytics — insights, models, dashboards, export',
        'permissions': {
            'insights',
            'analytics_models',
            'dashboards',
            'sql_editor',
            'export_csv',
            'export_email',
        },
    },
    'dev': {
        'label': 'Developer',
        'description': 'Dashboards + SQL only — no insights or analytics models',
        'permissions': {
            'dashboards',
            'sql_editor',
            'export_csv',
        },
    },
    'viewer': {
        'label': 'Viewer',
        'description': 'Read-only — can view pre-built dashboards only',
        'permissions': {
            'dashboards',
        },
    },
    'member': {
        'label': 'Member',
        'description': 'Default — same as Business User',
        'permissions': {
            'insights',
            'analytics_models',
            'dashboards',
            'sql_editor',
            'export_csv',
            'export_email',
        },
    },
}


def get_user_permissions(role: str) -> set:
    role_def = ROLES.get(role, ROLES.get('member', {}))
    return set(role_def.get('permissions', set()))


def user_has_permission(user: Dict, permission: str) -> bool:
    role = user.get('role', 'member')
    return permission in get_user_permissions(role)


def get_roles_list() -> List[Dict]:
    return [
        {'id': k, 'label': v['label'], 'description': v['description'],
         'permissions': sorted(v['permissions'])}
        for k, v in ROLES.items() if k != 'member'
    ]


def admin_list_users(session_token: str) -> Tuple[bool, str, List[Dict]]:
    user = validate_session(session_token)
    if not user:
        return False, "Not authenticated", []
    if not user_has_permission(user, 'admin_panel'):
        return False, "Insufficient permissions", []

    conn = _get_conn()
    rows = conn.execute("""
        SELECT u.user_id, u.username, u.email, u.display_name, u.role,
               u.created_at, u.last_login, t.team_name
        FROM users u LEFT JOIN teams t ON u.team_id = t.team_id
        ORDER BY u.created_at DESC
    """).fetchall()
    conn.close()
    return True, "OK", [dict(r) for r in rows]


def admin_update_user_role(session_token: str, target_user_id: str, new_role: str) -> Tuple[bool, str]:
    user = validate_session(session_token)
    if not user:
        return False, "Not authenticated"
    if not user_has_permission(user, 'admin_panel'):
        return False, "Insufficient permissions"
    if new_role not in ROLES:
        return False, f"Invalid role: {new_role}"
    if target_user_id == user['user_id'] and new_role != 'admin':
        return False, "Cannot remove your own admin role"

    conn = _get_conn()
    try:
        conn.execute("UPDATE users SET role = ? WHERE user_id = ?", (new_role, target_user_id))
        conn.commit()
        return True, f"Role updated to {ROLES[new_role]['label']}"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def admin_delete_user(session_token: str, target_user_id: str) -> Tuple[bool, str]:
    user = validate_session(session_token)
    if not user:
        return False, "Not authenticated"
    if not user_has_permission(user, 'admin_panel'):
        return False, "Insufficient permissions"
    if target_user_id == user['user_id']:
        return False, "Cannot delete your own account"

    conn = _get_conn()
    try:
        target = conn.execute("SELECT username FROM users WHERE user_id = ?", (target_user_id,)).fetchone()
        if not target:
            return False, "User not found"
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (target_user_id,))
        conn.execute("DELETE FROM user_sessions WHERE user_id = ?", (target_user_id,))
        conn.execute("DELETE FROM saved_dashboards WHERE user_id = ?", (target_user_id,))
        conn.execute("DELETE FROM export_passwords WHERE user_id = ?", (target_user_id,))
        conn.execute("DELETE FROM users WHERE user_id = ?", (target_user_id,))
        conn.commit()
        return True, f"User '{target['username']}' deleted"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def admin_create_user(session_token: str, username: str, password: str,
                      display_name: str = "", role: str = "viewer",
                      email: str = "", team_name: str = "") -> Tuple[bool, str]:
    user = validate_session(session_token)
    if not user:
        return False, "Not authenticated"
    if not user_has_permission(user, 'admin_panel'):
        return False, "Insufficient permissions"

    success, msg, _ = signup(username, password, display_name, team_name, role, email)
    return success, msg


def get_smtp_config() -> Dict[str, Any]:
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'sgpdm_config.json'
    )
    config = {
        'host': os.environ.get('SGPDM_HOST', ''),
        'port': int(os.environ.get('SGPDM_PORT', '587')),
        'username': os.environ.get('SGPDM_USERNAME', ''),
        'password': os.environ.get('SGPDM_PASSWORD', ''),
        'from_email': os.environ.get('SGPDM_FROM', ''),
        'from_name': os.environ.get('SGPDM_FROM_NAME', 'Healthcare Analytics'),
        'use_tls': os.environ.get('SGPDM_TLS', 'true').lower() == 'true',
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_cfg = json.load(f)
            for k, v in file_cfg.items():
                if v:
                    config[k] = v
        except Exception:
            pass
    return config


def save_smtp_config(session_token: str, config: Dict) -> Tuple[bool, str]:
    user = validate_session(session_token)
    if not user or not user_has_permission(user, 'admin_panel'):
        return False, "Admin access required"

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'sgpdm_config.json'
    )
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True, "SGPDM configuration saved"
    except Exception as e:
        return False, str(e)


init_db()

try:
    _conn = _get_conn()
    existing_cols = {r[1] for r in _conn.execute("PRAGMA table_info(users)").fetchall()}
    if 'email' not in existing_cols:
        _conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        _conn.commit()
    _conn.close()
except Exception:
    pass
