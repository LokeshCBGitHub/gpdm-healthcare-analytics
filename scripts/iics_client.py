from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import connection_config as _cc

_log = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    requests = None


class IICSError(Exception):
    pass


class IICSClient:

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._session_id: Optional[str] = None
        self._server_url: Optional[str] = None
        self._expires_at: float = 0.0
        self._cfg: Dict[str, Any] = {}

    def _load(self) -> Dict[str, Any]:
        self._cfg = _cc.get_iics_config() or {}
        return self._cfg

    def enabled(self) -> bool:
        return _cc.is_backend_enabled('iics')

    def _login(self) -> None:
        if requests is None:
            raise IICSError("python-requests not installed")
        cfg = self._load()
        url = cfg.get('login_url')
        sa = cfg.get('service_account') or {}
        user, pw = sa.get('user'), sa.get('password')
        if not url or not user or not pw:
            raise IICSError("IICS login_url / service_account missing")
        timeout = float(_cc.get_gateway_config().get('iics_timeout_s', 15))
        _log.info("IICS login → %s (user=%s)", url, user)
        resp = requests.post(
            url,
            json={'@type': 'login', 'username': user, 'password': pw},
            timeout=timeout,
            headers={'Accept': 'application/json',
                     'Content-Type': 'application/json'},
        )
        if resp.status_code != 200:
            raise IICSError(f"IICS login failed: {resp.status_code} {resp.text[:200]}")
        body = resp.json() if resp.content else {}
        self._session_id = body.get('icSessionId') or body.get('SessionId')
        self._server_url = body.get('serverUrl') or body.get('ServerUrl')
        self._expires_at = time.time() + 25 * 60
        if not self._session_id:
            raise IICSError(f"IICS login: no session id in response: {body}")

    def _ensure_session(self) -> None:
        with self._lock:
            if not self._session_id or time.time() >= self._expires_at:
                self._login()

    def _headers(self) -> Dict[str, str]:
        return {
            'icSessionId': self._session_id or '',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }

    def _with_retry(self, fn, *args, **kwargs):
        cfg = self._load()
        retry = cfg.get('retry') or {}
        max_attempts = int(retry.get('max_attempts', 3))
        base = float(retry.get('backoff_base_s', 1.5))
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except IICSError as e:
                last_exc = e
                if '401' in str(e) or '403' in str(e):
                    with self._lock:
                        self._session_id = None
                if attempt < max_attempts:
                    sleep_s = base ** attempt
                    _log.warning("IICS attempt %d failed: %s — retry in %.1fs",
                                 attempt, e, sleep_s)
                    time.sleep(sleep_s)
            except Exception as e:
                last_exc = IICSError(str(e))
                if attempt < max_attempts:
                    time.sleep(base ** attempt)
        raise last_exc or IICSError("IICS retry exhausted")

    def run_taskflow(self, logical_name: str,
                      params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.enabled():
            raise IICSError("IICS not enabled")
        cfg = self._load()
        taskflows = cfg.get('taskflows') or {}
        tf_name = taskflows.get(logical_name, logical_name)
        endpoint = cfg.get('taskflow_endpoint')
        if not endpoint:
            raise IICSError("taskflow_endpoint missing")
        url = endpoint.rstrip('/') + '/' + tf_name

        def _call() -> Dict[str, Any]:
            self._ensure_session()
            timeout = float(_cc.get_gateway_config().get('iics_timeout_s', 15))
            resp = requests.post(url, headers=self._headers(),
                                  json=params or {}, timeout=timeout)
            if resp.status_code >= 400:
                raise IICSError(f"taskflow {tf_name}: {resp.status_code} {resp.text[:200]}")
            return resp.json() if resp.content else {'status': 'ok'}

        return self._with_retry(_call)

    def query_dataset(self, logical_name: str,
                       params: Optional[Dict[str, Any]] = None,
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.enabled():
            raise IICSError("IICS not enabled")
        cfg = self._load()
        datasets = cfg.get('datasets') or {}
        ds_name = datasets.get(logical_name, logical_name)
        endpoint = cfg.get('query_endpoint')
        if not endpoint:
            raise IICSError("query_endpoint missing")

        body: Dict[str, Any] = {
            'dataset': ds_name,
            'connection': cfg.get('databricks_connection_name'),
            'params': params or {},
        }
        if limit is not None:
            body['limit'] = int(limit)

        def _call() -> List[Dict[str, Any]]:
            self._ensure_session()
            timeout = float(_cc.get_gateway_config().get('iics_timeout_s', 15))
            resp = requests.post(endpoint, headers=self._headers(),
                                  json=body, timeout=timeout)
            if resp.status_code >= 400:
                raise IICSError(f"query {ds_name}: {resp.status_code} {resp.text[:200]}")
            payload = resp.json() if resp.content else {}
            if isinstance(payload, list):
                return payload
            return payload.get('rows') or payload.get('data') or []

        return self._with_retry(_call)

    def health(self) -> Dict[str, Any]:
        info = {
            'enabled': self.enabled(),
            'requests_available': requests is not None,
            'has_session': bool(self._session_id),
            'server_url': self._server_url,
        }
        if not self.enabled() or requests is None:
            return info
        try:
            self._ensure_session()
            info['ok'] = True
        except Exception as e:
            info['ok'] = False
            info['error'] = str(e)
        return info


CLIENT = IICSClient()
