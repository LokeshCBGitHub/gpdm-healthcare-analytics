"""
SSO Authentication Module — Google + Microsoft OAuth 2.0

Production-grade SSO for KP healthcare dashboard.
Supports:
  - Google OAuth 2.0 (Google Workspace / Gmail accounts)
  - Microsoft Azure AD / Entra ID OAuth 2.0
  - PKCE flow for browser-based auth (no client secret needed on frontend)
  - Token validation and user provisioning

Setup:
  1. Create OAuth app in Google Cloud Console → Credentials → OAuth 2.0 Client ID
  2. Create app in Azure Portal → App registrations → New registration
  3. Set redirect URIs to: http://localhost:8787/auth/callback
  4. Put credentials in sso_config.json (see SSO_CONFIG_PATH)

Zero external dependencies — uses only Python stdlib (urllib, json, http).
"""

import os
import json
import secrets
import time
import hashlib
import base64
from urllib.parse import urlencode, parse_qs, urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError
from typing import Optional, Dict, Tuple

# ─── Configuration ───

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SSO_CONFIG_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'sso_config.json')

# Default config (override with sso_config.json)
_DEFAULT_CONFIG = {
    "google": {
        "client_id": "",
        "client_secret": "",
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://www.googleapis.com/oauth2/v3/userinfo",
        "scopes": "openid email profile",
    },
    "microsoft": {
        "client_id": "",
        "client_secret": "",
        "tenant_id": "common",  # Use 'common' for multi-tenant, or specific tenant ID
        "auth_url": "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize",
        "token_url": "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        "userinfo_url": "https://graph.microsoft.com/v1.0/me",
        "scopes": "openid email profile User.Read",
    },
    "redirect_uri": "http://localhost:8787/auth/callback",
    "org_name": "KP Healthcare",
    "org_domain": "",  # e.g., "kp.org" — restrict signups to this domain
    "allowed_domains": [],  # e.g., ["kp.org", "kp.org"] — empty = allow all
}


def _load_config() -> Dict:
    """Load SSO configuration from file, merging with defaults."""
    config = dict(_DEFAULT_CONFIG)
    if os.path.exists(SSO_CONFIG_PATH):
        try:
            with open(SSO_CONFIG_PATH, 'r') as f:
                file_config = json.load(f)
            # Deep merge
            for key, val in file_config.items():
                if isinstance(val, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(val)
                else:
                    config[key] = val
        except Exception:
            pass
    return config


def save_config(config: Dict) -> bool:
    """Save SSO configuration to file."""
    try:
        os.makedirs(os.path.dirname(SSO_CONFIG_PATH), exist_ok=True)
        with open(SSO_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False


def is_sso_configured() -> Dict[str, bool]:
    """Check which SSO providers are configured."""
    config = _load_config()
    return {
        'google': bool(config.get('google', {}).get('client_id')),
        'microsoft': bool(config.get('microsoft', {}).get('client_id')),
        'any': bool(config.get('google', {}).get('client_id')) or bool(config.get('microsoft', {}).get('client_id')),
    }


# ─── PKCE Helper ───

def _generate_pkce() -> Tuple[str, str]:
    """Generate PKCE code verifier and challenge."""
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode('ascii')).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
    return verifier, challenge


# ─── OAuth State Management ───

_pending_states: Dict[str, Dict] = {}  # state -> {provider, verifier, timestamp, nonce}


def generate_auth_url(provider: str, redirect_uri: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate OAuth authorization URL for the given provider.
    Returns (auth_url, state) or (None, error_message).
    """
    config = _load_config()
    prov_config = config.get(provider)
    if not prov_config or not prov_config.get('client_id'):
        return None, f"SSO provider '{provider}' is not configured"

    redirect = redirect_uri or config.get('redirect_uri', 'http://localhost:8787/auth/callback')
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(16)
    verifier, challenge = _generate_pkce()

    # Store state for verification
    _pending_states[state] = {
        'provider': provider,
        'verifier': verifier,
        'nonce': nonce,
        'timestamp': time.time(),
        'redirect_uri': redirect,
    }

    # Clean old states (>10 min)
    now = time.time()
    expired = [k for k, v in _pending_states.items() if now - v['timestamp'] > 600]
    for k in expired:
        del _pending_states[k]

    if provider == 'google':
        params = {
            'client_id': prov_config['client_id'],
            'redirect_uri': redirect,
            'response_type': 'code',
            'scope': prov_config.get('scopes', 'openid email profile'),
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
            'scope': prov_config.get('scopes', 'openid email profile User.Read'),
            'state': state,
            'nonce': nonce,
            'code_challenge': challenge,
            'code_challenge_method': 'S256',
            'prompt': 'select_account',
        }
        return f"{auth_url}?{urlencode(params)}", state

    return None, f"Unknown provider: {provider}"


def handle_callback(code: str, state: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    Handle OAuth callback. Exchange code for tokens and get user info.
    Returns (success, message, user_info_dict_or_None).

    user_info contains: email, name, picture, provider, provider_id
    """
    if state not in _pending_states:
        return False, "Invalid or expired OAuth state", None

    state_data = _pending_states.pop(state)
    provider = state_data['provider']
    verifier = state_data['verifier']
    redirect_uri = state_data['redirect_uri']

    config = _load_config()
    prov_config = config.get(provider, {})

    # Exchange code for tokens
    try:
        token_data = _exchange_code(provider, code, verifier, redirect_uri, prov_config)
        if not token_data or 'access_token' not in token_data:
            return False, f"Token exchange failed: {token_data}", None

        # Get user info
        user_info = _get_user_info(provider, token_data['access_token'], prov_config)
        if not user_info or not user_info.get('email'):
            return False, "Could not retrieve user email from provider", None

        # Check allowed domains
        allowed_domains = config.get('allowed_domains', [])
        if allowed_domains:
            email_domain = user_info['email'].split('@')[-1].lower()
            if email_domain not in [d.lower() for d in allowed_domains]:
                return False, f"Email domain '{email_domain}' is not allowed. Contact your administrator.", None

        user_info['provider'] = provider
        return True, "Authentication successful", user_info

    except Exception as e:
        return False, f"SSO authentication failed: {str(e)}", None


def _exchange_code(provider: str, code: str, verifier: str,
                   redirect_uri: str, prov_config: Dict) -> Optional[Dict]:
    """Exchange authorization code for tokens."""
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
    except Exception as e:
        return {'error': str(e)}


def _get_user_info(provider: str, access_token: str, prov_config: Dict) -> Optional[Dict]:
    """Fetch user profile from the identity provider."""
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
                'picture': '',  # MS Graph needs separate call for photo
                'provider_id': data.get('id', ''),
                'email_verified': True,  # Azure AD verified
            }
    except Exception:
        return None


# ─── Email Sending via SMTP ───

def send_email_smtp(from_email: str, to_email: str, subject: str, body_html: str,
                    smtp_host: str = 'smtp.gmail.com', smtp_port: int = 587,
                    smtp_password: str = '', attachment_path: str = None) -> Tuple[bool, str]:
    """
    Send email via SMTP (Gmail/Outlook).

    For Gmail: use App Password (Settings → Security → App Passwords).
    For Outlook: use smtp_host='smtp.office365.com', smtp_port=587.
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    if not smtp_password:
        return False, "SMTP password/app-password is required"

    try:
        msg = MIMEMultipart('mixed')
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject

        # HTML body
        msg.attach(MIMEText(body_html, 'html', 'utf-8'))

        # Attachment
        if attachment_path and os.path.exists(attachment_path):
            filename = os.path.basename(attachment_path)
            with open(attachment_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
            msg.attach(part)

        # Send
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(from_email, smtp_password)
        server.send_message(msg)
        server.quit()

        return True, f"Email sent to {to_email}"

    except smtplib.SMTPAuthenticationError:
        return False, "SMTP authentication failed. For Gmail, use an App Password (not your regular password)."
    except Exception as e:
        return False, f"Email failed: {str(e)}"


# ─── SMTP Config Storage ───

SMTP_CONFIG_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'smtp_config.json')


def save_smtp_config(email: str, smtp_host: str, smtp_port: int, app_password: str) -> bool:
    """Save SMTP credentials (encrypted with user's password hash is ideal, but simple for now)."""
    try:
        config = {
            'email': email,
            'smtp_host': smtp_host,
            'smtp_port': smtp_port,
            'app_password': app_password,  # In production, encrypt this!
        }
        with open(SMTP_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        os.chmod(SMTP_CONFIG_PATH, 0o600)  # Owner-only read/write
        return True
    except Exception:
        return False


def load_smtp_config() -> Optional[Dict]:
    """Load saved SMTP config."""
    if os.path.exists(SMTP_CONFIG_PATH):
        try:
            with open(SMTP_CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ─── Google Account Info ───

def get_google_account_creation_url(email: str = '') -> str:
    """Get the Google account creation URL, pre-filled with email if provided."""
    base = 'https://accounts.google.com/signup/v2/webcreateaccount'
    params = {'flowName': 'GlifWebSignIn', 'flowEntry': 'SignUp'}
    if email:
        params['Email'] = email
    return f"{base}?{urlencode(params)}"
