import os
import json
import secrets
import time
import hashlib
import base64
import logging
from urllib.parse import urlencode, parse_qs, urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError
from typing import Optional, Dict, Tuple

PKCE_VERIFIER_LENGTH = 128
STATE_EXPIRY_SECONDS = 600

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SSO_CONFIG_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'sso_config.json')

_PROVIDER_CONFIG = {
    'google': {
        'auth_url': 'https://accounts.google.com/o/oauth2/v2/auth',
        'token_url': 'https://oauth2.googleapis.com/token',
        'userinfo_url': 'https://www.googleapis.com/oauth2/v3/userinfo',
        'scope': 'openid email profile',
    },
    'microsoft': {
        'auth_url': 'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize',
        'token_url': 'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token',
        'userinfo_url': 'https://graph.microsoft.com/v1.0/me',
        'scope': 'openid email profile User.Read',
    },
}

_DEFAULT_CONFIG = {
    "google": {
        "client_id": "",
        "client_secret": "",
        "auth_url": _PROVIDER_CONFIG['google']['auth_url'],
        "token_url": _PROVIDER_CONFIG['google']['token_url'],
        "userinfo_url": _PROVIDER_CONFIG['google']['userinfo_url'],
        "scopes": _PROVIDER_CONFIG['google']['scope'],
    },
    "microsoft": {
        "client_id": "",
        "client_secret": "",
        "tenant_id": "common",
        "auth_url": _PROVIDER_CONFIG['microsoft']['auth_url'],
        "token_url": _PROVIDER_CONFIG['microsoft']['token_url'],
        "userinfo_url": _PROVIDER_CONFIG['microsoft']['userinfo_url'],
        "scopes": _PROVIDER_CONFIG['microsoft']['scope'],
    },
    "redirect_uri": "http://localhost:8787/auth/callback",
    "org_name": "KP Healthcare",
    "org_domain": "",
    "allowed_domains": [],
}


def _load_config() -> Dict:
    config = dict(_DEFAULT_CONFIG)
    if os.path.exists(SSO_CONFIG_PATH):
        try:
            with open(SSO_CONFIG_PATH, 'r') as f:
                file_config = json.load(f)
            for key, val in file_config.items():
                if isinstance(val, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(val)
                else:
                    config[key] = val
        except json.JSONDecodeError as e:
            logging.getLogger('gpdm.sso').warning('Failed to load SSO config (JSONDecodeError): %s', e)
        except (IOError, OSError) as e:
            logging.getLogger('gpdm.sso').warning('Failed to load SSO config (I/O error): %s', e)
    return config


def save_config(config: Dict) -> bool:
    try:
        os.makedirs(os.path.dirname(SSO_CONFIG_PATH), exist_ok=True)
        with open(SSO_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except (IOError, OSError) as e:
        logging.getLogger('gpdm.sso').warning('Failed to save SSO config: %s', e)
        return False


def is_sso_configured() -> Dict[str, bool]:
    config = _load_config()
    return {
        'google': bool(config.get('google', {}).get('client_id')),
        'microsoft': bool(config.get('microsoft', {}).get('client_id')),
        'any': bool(config.get('google', {}).get('client_id')) or bool(config.get('microsoft', {}).get('client_id')),
    }


def _generate_pkce() -> Tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:PKCE_VERIFIER_LENGTH]
    digest = hashlib.sha256(verifier.encode('ascii')).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
    return verifier, challenge

_pending_states: Dict[str, Dict] = {}


def generate_auth_url(provider: str, redirect_uri: str = None) -> Tuple[Optional[str], Optional[str]]:
    config = _load_config()
    prov_config = config.get(provider)
    if not prov_config or not prov_config.get('client_id'):
        return None, f"SSO provider '{provider}' is not configured"

    redirect = redirect_uri or config.get('redirect_uri', 'http://localhost:8787/auth/callback')
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(16)
    verifier, challenge = _generate_pkce()

    _pending_states[state] = {
        'provider': provider,
        'verifier': verifier,
        'nonce': nonce,
        'timestamp': time.time(),
        'redirect_uri': redirect,
    }

    now = time.time()
    expired = [k for k, v in _pending_states.items() if now - v['timestamp'] > STATE_EXPIRY_SECONDS]
    for k in expired:
        del _pending_states[k]

    if provider == 'google':
        params = {
            'client_id': prov_config['client_id'],
            'redirect_uri': redirect,
            'response_type': 'code',
            'scope': prov_config.get('scopes', _PROVIDER_CONFIG['google']['scope']),
            'state': state,
            'nonce': nonce,
            'code_challenge': challenge,
            'code_challenge_method': 'S256',
            'access_type': 'offline',
            'prompt': 'select_account',
        }
        return f"{prov_config['auth_url']}?{urlencode(params)}", state

    elif provider == 'microsoft':
        tenant = prov_config.get('tenant_id', 'common')
        auth_url = prov_config['auth_url'].replace('{tenant_id}', tenant)
        params = {
            'client_id': prov_config['client_id'],
            'redirect_uri': redirect,
            'response_type': 'code',
            'scope': prov_config.get('scopes', _PROVIDER_CONFIG['microsoft']['scope']),
            'state': state,
            'nonce': nonce,
            'code_challenge': challenge,
            'code_challenge_method': 'S256',
            'prompt': 'select_account',
        }
        return f"{auth_url}?{urlencode(params)}", state

    return None, f"Unknown provider: {provider}"


def handle_callback(code: str, state: str) -> Tuple[bool, str, Optional[Dict]]:
    if state not in _pending_states:
        return False, "Invalid or expired OAuth state", None

    state_data = _pending_states.pop(state)
    provider = state_data['provider']
    verifier = state_data['verifier']
    redirect_uri = state_data['redirect_uri']

    config = _load_config()
    prov_config = config.get(provider, {})

    try:
        token_data = _exchange_code(provider, code, verifier, redirect_uri, prov_config)
        if not token_data or 'access_token' not in token_data:
            return False, f"Token exchange failed: {token_data}", None

        user_info = _get_user_info(provider, token_data['access_token'], prov_config)
        if not user_info or not user_info.get('email'):
            return False, "Could not retrieve user email from provider", None

        allowed_domains_list = config.get('allowed_domains', [])
        if allowed_domains_list:
            allowed_domains = set(d.lower() for d in allowed_domains_list)
            email_domain = user_info['email'].split('@')[-1].lower()
            if email_domain not in allowed_domains:
                return False, f"Email domain '{email_domain}' is not allowed. Contact your administrator.", None

        user_info['provider'] = provider
        return True, "Authentication successful", user_info

    except Exception as e:
        return False, f"SSO authentication failed: {str(e)}", None


def _exchange_code(provider: str, code: str, verifier: str,
                   redirect_uri: str, prov_config: Dict) -> Optional[Dict]:
    if provider == 'google':
        token_url = prov_config.get('token_url', 'https://oauth2.googleapis.com/token')
    elif provider == 'microsoft':
        tenant = prov_config.get('tenant_id', 'common')
        token_url = prov_config.get('token_url', '').replace('{tenant_id}', tenant)
    else:
        return None

    data = urlencode({
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': redirect_uri,
        'client_id': prov_config['client_id'],
        'client_secret': prov_config.get('client_secret', ''),
        'code_verifier': verifier,
    }).encode('utf-8')

    try:
        req = Request(token_url, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'})
        resp = urlopen(req, timeout=10)
        return json.loads(resp.read().decode('utf-8'))
    except json.JSONDecodeError as e:
        logging.getLogger('gpdm.sso').warning('Token response JSON decode error: %s', e)
        return {'error': str(e)}
    except (IOError, OSError) as e:
        logging.getLogger('gpdm.sso').warning('Token exchange I/O error: %s', e)
        return {'error': str(e)}


def _get_user_info(provider: str, access_token: str, prov_config: Dict) -> Optional[Dict]:
    if provider == 'google':
        url = prov_config.get('userinfo_url', 'https://www.googleapis.com/oauth2/v3/userinfo')
    elif provider == 'microsoft':
        url = prov_config.get('userinfo_url', 'https://graph.microsoft.com/v1.0/me')
    else:
        return None

    try:
        req = Request(url, headers={'Authorization': f'Bearer {access_token}'})
        resp = urlopen(req, timeout=10)
        data = json.loads(resp.read().decode('utf-8'))

        if provider == 'google':
            return {
                'email': data.get('email', ''),
                'name': data.get('name', ''),
                'picture': data.get('picture', ''),
                'provider_id': data.get('sub', ''),
                'email_verified': data.get('email_verified', False),
            }
        elif provider == 'microsoft':
            return {
                'email': data.get('mail') or data.get('userPrincipalName', ''),
                'name': data.get('displayName', ''),
                'picture': '',
                'provider_id': data.get('id', ''),
                'email_verified': True,
            }
    except json.JSONDecodeError as e:
        logging.getLogger('gpdm.sso').warning('User info JSON decode error: %s', e)
        return None
    except (IOError, OSError) as e:
        logging.getLogger('gpdm.sso').warning('User info fetch I/O error: %s', e)
        return None


def send_email_smtp(from_email: str, to_email: str, subject: str, body_html: str,
                    smtp_host: str = 'sgpdm.gmail.com', smtp_port: int = 587,
                    smtp_password: str = '', attachment_path: str = None) -> Tuple[bool, str]:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    if not smtp_password:
        return False, "SGPDM password/app-password is required"

    try:
        msg = MIMEMultipart('mixed')
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body_html, 'html', 'utf-8'))

        if attachment_path and os.path.exists(attachment_path):
            filename = os.path.basename(attachment_path)
            with open(attachment_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
            msg.attach(part)

        server = smtplib.SGPDM(smtp_host, smtp_port)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(from_email, smtp_password)
        server.send_message(msg)
        server.quit()

        return True, f"Email sent to {to_email}"

    except smtplib.SGPDMAuthenticationError:
        return False, "SGPDM authentication failed. For Gmail, use an App Password (not your regular password)."
    except Exception as e:
        return False, f"Email failed: {str(e)}"

SGPDM_CONFIG_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'sgpdm_config.json')


def save_smtp_config(email: str, smtp_host: str, smtp_port: int, app_password: str) -> bool:
    try:
        config = {
            'email': email,
            'smtp_host': smtp_host,
            'smtp_port': smtp_port,
            'app_password': app_password,
        }
        with open(SGPDM_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        os.chmod(SGPDM_CONFIG_PATH, 0o600)
        return True
    except (IOError, OSError) as e:
        logging.getLogger('gpdm.sso').warning('Failed to save SGPDM config: %s', e)
        return False


def load_smtp_config() -> Optional[Dict]:
    if os.path.exists(SGPDM_CONFIG_PATH):
        try:
            with open(SGPDM_CONFIG_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.getLogger('gpdm.sso').warning('SGPDM config JSON decode error: %s', e)
        except (IOError, OSError) as e:
            logging.getLogger('gpdm.sso').warning('Failed to load SGPDM config: %s', e)
    return None

def get_google_account_creation_url(email: str = '') -> str:
    base = 'https://accounts.google.com/signup/v2/webcreateaccount'
    params = {'flowName': 'GlifWebSignIn', 'flowEntry': 'SignUp'}
    if email:
        params['Email'] = email
    return f"{base}?{urlencode(params)}"
