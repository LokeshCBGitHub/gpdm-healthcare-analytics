from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)


_DEFAULT_PATH = os.path.join(
    (os.environ.get('GPDM_BASE_DIR', '') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'connections.yaml'
)


_ENV_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}')


def _interpolate_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, default if default is not None else '')
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        _log.warning("PyYAML not installed — using minimal fallback parser")
        return _load_yaml_minimal(path)


def _coerce_scalar(s: str) -> Any:
    s = s.strip()
    if not s:
        return ''
    if (s.startswith('"') and s.endswith('"')) or \
       (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    low = s.lower()
    if low == 'true':  return True
    if low == 'false': return False
    if low in ('null', '~'): return None
    try:
        if '.' in s: return float(s)
        return int(s)
    except ValueError:
        return s


def _load_yaml_minimal(path: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack = [(-1, root)]
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if '#' in line:
                in_q = False; qch = ''
                cut = -1
                for i, ch in enumerate(line):
                    if ch in ('"', "'"):
                        if not in_q:
                            in_q = True; qch = ch
                        elif ch == qch:
                            in_q = False
                    elif ch == '#' and not in_q:
                        cut = i; break
                if cut >= 0:
                    line = line[:cut]
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(' '))
            content = line.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1] if stack else root
            if content.startswith('- '):
                item = content[2:].strip()
                if isinstance(parent, list):
                    parent.append(_coerce_scalar(item))
                continue
            if ':' in content:
                key, _, rest = content.partition(':')
                key = key.strip()
                rest = rest.strip()
                if rest == '':
                    new_container: Any = {}
                    if isinstance(parent, dict):
                        parent[key] = new_container
                    stack.append((indent, new_container))
                else:
                    if isinstance(parent, dict):
                        parent[key] = _coerce_scalar(rest)
    return root


@lru_cache(maxsize=4)
def load_connections(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or os.environ.get('GPDM_CONNECTIONS_YAML', _DEFAULT_PATH)
    if not os.path.isfile(p):
        _log.warning("connections.yaml not found at %s — returning empty config", p)
        return {}
    try:
        raw = _load_yaml(p)
    except Exception as e:
        _log.error("failed to load connections.yaml: %s", e)
        return {}
    cfg = _interpolate_env(raw)
    _log.info("loaded connections.yaml from %s (keys=%s)", p, list(cfg.keys()))
    return cfg


def reload() -> Dict[str, Any]:
    load_connections.cache_clear()
    return load_connections()


def get_gateway_config() -> Dict[str, Any]:
    return load_connections().get('data_gateway', {}) or {}


def get_iics_config() -> Dict[str, Any]:
    return load_connections().get('iics', {}) or {}


def get_databricks_config() -> Dict[str, Any]:
    return load_connections().get('databricks_direct', {}) or {}


def get_sqlite_config() -> Dict[str, Any]:
    return load_connections().get('sqlite', {}) or {}


def get_logical_table(logical_name: str, backend: str) -> Optional[str]:
    tables = load_connections().get('tables', {}) or {}
    entry = tables.get(logical_name) or {}
    return entry.get(backend)


def is_backend_enabled(backend: str) -> bool:
    cfg = load_connections()
    if backend == 'iics':
        return bool((cfg.get('iics') or {}).get('enabled', False))
    if backend == 'databricks_direct':
        return bool((cfg.get('databricks_direct') or {}).get('enabled', False))
    if backend == 'sqlite':
        return bool((cfg.get('sqlite') or {}).get('enabled', True))
    return False


def priority_order() -> list:
    pr = (get_gateway_config().get('priority') or
          ['iics', 'databricks_direct', 'sqlite'])
    return [b for b in pr if is_backend_enabled(b)]


def status() -> Dict[str, Any]:
    cfg = load_connections()
    return {
        'loaded': bool(cfg),
        'path': os.environ.get('GPDM_CONNECTIONS_YAML', _DEFAULT_PATH),
        'priority': priority_order(),
        'iics_enabled': is_backend_enabled('iics'),
        'databricks_direct_enabled': is_backend_enabled('databricks_direct'),
        'sqlite_enabled': is_backend_enabled('sqlite'),
        'logical_tables': list((cfg.get('tables') or {}).keys()),
    }
