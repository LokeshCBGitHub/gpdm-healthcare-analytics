import re
import hashlib
import hmac
from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple


class SourceProtect:

    BLOCKED_AGENTS = {
        'sqlmap', 'nikto', 'nmap', 'masscan', 'nessus', 'acunetix',
        'burp', 'zap', 'w3af', 'wpscan', 'dirbuster', 'metasploit',
        'scrapy', 'requests', 'curl', 'wget', 'python-requests', 'go-http-client',
        'java/1.', 'ruby', 'perl', 'powershell', 'csharp', 'java',
        'bot', 'crawler', 'spider', 'scraper', 'monitor', 'validator'
    }

    RATE_LIMIT_WINDOW = 60
    RATE_LIMIT_MAX = 100

    def __init__(self, strict_mode: bool = True, log_violations: bool = True):
        self.strict_mode = strict_mode
        self.log_violations = log_violations
        self._request_cache: Dict[str, List[float]] = {}

    def get_security_headers(self) -> Dict[str, str]:
        headers = {
            'X-Content-Type-Options': 'nosniff',

            'X-Frame-Options': 'DENY',

            'X-XSS-Protection': '1; mode=block',

            'Referrer-Policy': 'no-referrer',

            'Permissions-Policy': (
                'camera=(), microphone=(), geolocation=(), '
                'accelerometer=(), ambient-light-sensor=(), '
                'gyroscope=(), magnetometer=(), payment=(), '
                'usb=(), vr=(), xr-spatial-tracking=()'
            ),

            'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',

            'Server': 'secure-server',

            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
        }

        csp_directives = [
            "default-src 'none'",
            "script-src 'self' https://cdnjs.cloudflare.com",
            "style-src 'self' https://cdnjs.cloudflare.com 'unsafe-inline'",
            "img-src 'self' data:",
            "font-src 'self' https://cdnjs.cloudflare.com",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "form-action 'self'",
            "base-uri 'self'",
            "object-src 'none'",
            "media-src 'none'",
            "worker-src 'self'",
            "upgrade-insecure-requests",
        ]

        headers['Content-Security-Policy'] = '; '.join(csp_directives)

        return headers

    def get_protection_script(self) -> str:
        js_code = '''
(function() {
    'use strict';

    // Detect if devtools is open and show warning overlay
    let devtoolsOpen = false;
    const checkDevtools = () => {
        const threshold = 160;
        if (window.outerHeight - window.innerHeight > threshold ||
            window.outerWidth - window.innerWidth > threshold) {
            if (!devtoolsOpen) {
                devtoolsOpen = true;
                showDevtoolsWarning();
            }
            return true;
        }
        if (devtoolsOpen) {
            devtoolsOpen = false;
            hideDevtoolsWarning();
        }
        return false;
    };

    // Show warning overlay if devtools detected
    const showDevtoolsWarning = () => {
        let overlay = document.getElementById('_dt_warning');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = '_dt_warning';
            overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);z-index:999999;display:flex;align-items:center;justify-content:center;font-family:monospace;color:red;font-size:18px;text-align:center;';
            overlay.innerHTML = '<div style="padding:20px;background:#111;border:2px solid red;">Developer tools are not permitted. This application is restricted.</div>';
            document.body.appendChild(overlay);
        }
    };

    const hideDevtoolsWarning = () => {
        const overlay = document.getElementById('_dt_warning');
        if (overlay) overlay.remove();
    };

    // Monitor devtools periodically
    setInterval(checkDevtools, 500);

    // Disable right-click context menu
    document.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        return false;
    });

    // Disable keyboard shortcuts for devtools/source viewing
    document.addEventListener('keydown', (e) => {
        // F12 - DevTools
        if (e.key === 'F12') {
            e.preventDefault();
            return false;
        }
        // Ctrl+Shift+I - DevTools (Windows/Linux)
        if (e.ctrlKey && e.shiftKey && e.key === 'I') {
            e.preventDefault();
            return false;
        }
        // Ctrl+Shift+J - Console (Windows/Linux)
        if (e.ctrlKey && e.shiftKey && e.key === 'J') {
            e.preventDefault();
            return false;
        }
        // Ctrl+U - View Source (Firefox)
        if (e.ctrlKey && e.key === 'u') {
            e.preventDefault();
            return false;
        }
        // Cmd+Shift+I - DevTools (Mac)
        if (e.metaKey && e.shiftKey && e.key === 'I') {
            e.preventDefault();
            return false;
        }
        // Cmd+Shift+J - Console (Mac)
        if (e.metaKey && e.shiftKey && e.key === 'J') {
            e.preventDefault();
            return false;
        }
        // Cmd+Shift+C - Element Picker (Mac)
        if (e.metaKey && e.shiftKey && e.key === 'C') {
            e.preventDefault();
            return false;
        }
        // Cmd+Shift+U - View Source (Mac)
        if (e.metaKey && e.shiftKey && e.key === 'U') {
            e.preventDefault();
            return false;
        }
        // Cmd+Option+I - DevTools (Mac)
        if (e.metaKey && e.altKey && e.key === 'I') {
            e.preventDefault();
            return false;
        }
        // Cmd+Option+J - Console (Mac)
        if (e.metaKey && e.altKey && e.key === 'J') {
            e.preventDefault();
            return false;
        }
        // Cmd+Option+U - View Source (Mac)
        if (e.metaKey && e.altKey && e.key === 'U') {
            e.preventDefault();
            return false;
        }
    });

    // Disable text selection on sensitive elements
    const protectElement = (selector) => {
        const elements = document.querySelectorAll(selector);
        elements.forEach((el) => {
            el.style.userSelect = 'none';
            el.style.webkitUserSelect = 'none';
            el.style.mozUserSelect = 'none';
            el.style.msUserSelect = 'none';
            el.addEventListener('selectstart', (e) => e.preventDefault());
            el.addEventListener('copy', (e) => e.preventDefault());
        });
    };
    protectElement('[data-protect], .protected, .api-key, .token, .secret');

    // Disable drag/drop on sensitive content
    document.addEventListener('dragstart', (e) => {
        if (e.target.tagName === 'IMG' || e.target.closest('[data-protect]')) {
            e.preventDefault();
            return false;
        }
    });

    // Clear console to prevent information leakage
    console.clear();

    // Override console methods to prevent data leakage via logs
    const noop = () => {};
    const originalWarn = console.warn;
    const originalError = console.error;

    console.log = noop;
    console.debug = noop;
    console.info = noop;
    console.warn = noop;
    console.error = noop;
    console.trace = noop;
    console.group = noop;
    console.groupEnd = noop;

    // Anti-debugging: trigger debugger on devtools open
    // (creates annoying pause when stepping through)
    const antiDebug = () => {
        debugger;
        setTimeout(antiDebug, 100);
    };
    antiDebug();

    // Prevent access to window properties that leak information
    Object.defineProperty(window, '__devtoolsEnabled', {
        get() { debugger; return false; },
        configurable: false
    });

    // Prevent eval and Function constructor abuse
    const preventEval = () => {
        window.eval = function() {
            throw new Error('eval is not allowed');
        };
        window.Function = function() {
            throw new Error('Function constructor is not allowed');
        };
        window.execScript = function() {
            throw new Error('execScript is not allowed');
        };
    };
    preventEval();

    // Disable sourcemap consumption
    if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
        window.__REACT_DEVTOOLS_GLOBAL_HOOK__.isDisabled = true;
    }
})();
'''
        return self._minify_js(js_code)

    def obfuscate_response(self, html: str) -> str:
        if not html:
            return html

        html = re.sub(r'', '', html)

        html = re.sub(r'//# sourceMappingURL=.*?(?=\n|$)', '', html)
        html = re.sub(r'/\*# sourceMappingURL=.*?\*/', '', html)

        def minify_script(match):
            script_content = match.group(1)
            return f'<script>{self._minify_js(script_content)}</script>'
        html = re.sub(r'<script\s*(?:type="text/javascript")?\s*>([\s\S]*?)</script>',
                      minify_script, html, flags=re.IGNORECASE)

        def minify_style(match):
            style_content = match.group(1)
            return f'<style>{self._minify_css(style_content)}</style>'
        html = re.sub(r'<style\s*(?:type="text/css")?\s*>([\s\S]*?)</style>',
                      minify_style, html, flags=re.IGNORECASE)

        protection_script = self.get_protection_script()
        head_injection = f'<script>{protection_script}</script>'

        if '<head>' in html.lower():
            html = re.sub(r'(<head[^>]*>)', rf'\1{head_injection}', html, flags=re.IGNORECASE)
        elif '<body>' in html.lower():
            html = re.sub(r'(<body[^>]*>)', rf'\1{head_injection}', html, flags=re.IGNORECASE)
        else:
            html = head_injection + html

        return html

    def sanitize_error(self, error_msg: str) -> str:
        if self.log_violations:
            self._log_security_event('ERROR_SANITIZED', {
                'original_message': error_msg[:500],
                'timestamp': datetime.utcnow().isoformat()
            })

        sanitized = re.sub(
            r'[/\\](?:home|var|tmp|opt|users|windows|system32|c:)[/\\][\w/\\.-]*\.py',
            '[file]',
            error_msg,
            flags=re.IGNORECASE
        )
        sanitized = re.sub(r'[/\\][\w/\\.-]+\.py', '[file]', sanitized)

        sanitized = re.sub(r'(?:line|:)\s*\d+', '[line]', sanitized, flags=re.IGNORECASE)

        sanitized = re.sub(r'(?:File|in|at)\s+"[^"]+",\s+line\s+\d+', '[trace]', sanitized)

        sanitized = re.sub(r'https?://[^\s]+', '[url]', sanitized)

        if len(sanitized) > 200 or 'traceback' in sanitized.lower():
            sanitized = 'An error occurred. Please contact support.'

        return sanitized or 'An internal error occurred.'

    def validate_request(self, headers: Dict[str, str],
                        client_ip: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        headers_lower = {k.lower(): v for k, v in headers.items()}

        user_agent = headers_lower.get('user-agent', '').lower()
        if user_agent:
            for blocked_agent in self.BLOCKED_AGENTS:
                if blocked_agent in user_agent:
                    if self.log_violations:
                        self._log_security_event('BLOCKED_USER_AGENT', {
                            'user_agent': user_agent,
                            'blocked_pattern': blocked_agent,
                            'client_ip': client_ip
                        })
                    return False, f'blocked_user_agent: {blocked_agent}'

        suspicious_headers = {
            'x-scanner': 'scanner header detected',
            'x-forwarded-for': 'proxy header suspicious',
            'x-forwarded-proto': 'protocol confusion attempt',
        }
        for header, reason in suspicious_headers.items():
            if header in headers_lower and len(headers_lower.get(header, '')) > 200:
                if self.log_violations:
                    self._log_security_event('SUSPICIOUS_HEADER', {
                        'header': header,
                        'client_ip': client_ip
                    })
                return False, f'suspicious_header: {reason}'

        if client_ip:
            now = datetime.utcnow().timestamp()
            if client_ip not in self._request_cache:
                self._request_cache[client_ip] = []

            cutoff = now - self.RATE_LIMIT_WINDOW
            self._request_cache[client_ip] = [
                ts for ts in self._request_cache[client_ip] if ts > cutoff
            ]

            if len(self._request_cache[client_ip]) >= self.RATE_LIMIT_MAX:
                if self.log_violations:
                    self._log_security_event('RATE_LIMIT_EXCEEDED', {
                        'client_ip': client_ip,
                        'requests': len(self._request_cache[client_ip]),
                        'limit': self.RATE_LIMIT_MAX
                    })
                return False, 'rate_limit_exceeded'

            self._request_cache[client_ip].append(now)

        return True, None

    def _minify_js(self, js: str) -> str:
        if not js:
            return js

        js = re.sub(r'//.*?(?=\n|$)', '', js)

        js = re.sub(r'/\*[\s\S]*?\*/', '', js)

        js = js.strip()

        js = re.sub(r'\s+', ' ', js)

        js = re.sub(r'\s*([{}()[\],;:=+\-*/%<>!&|?])\s*', r'\1', js)

        js = re.sub(r'(\w)(var|let|const|function|return|if|else|for|while)\s', r'\1 \2 ', js)

        return js

    def _minify_css(self, css: str) -> str:
        if not css:
            return css

        css = re.sub(r'/\*[\s\S]*?\*/', '', css)

        css = css.strip()

        css = re.sub(r'\s+', ' ', css)

        css = re.sub(r'\s*([{}:;,>+~])\s*', r'\1', css)

        return css

    def _log_security_event(self, event_type: str, details: Dict) -> None:
        timestamp = datetime.utcnow().isoformat()


def get_protect() -> SourceProtect:
    return SourceProtect(strict_mode=True, log_violations=True)


if __name__ == '__main__':
    protect = SourceProtect()

    print("=== Security Headers ===")
    for header, value in protect.get_security_headers().items():
        print(f"{header}: {value[:60]}..." if len(value) > 60 else f"{header}: {value}")

    print("\n=== Protection Script Size ===")
    script = protect.get_protection_script()
    print(f"Minified script: {len(script)} bytes")

    print("\n=== HTML Obfuscation Test ===")
    test_html = '''
    <!DOCTYPE html>
    <html>
    <head>
        
        <script>
            // Configuration
            const API_URL = "https://internal.api.local/v1/patient";
            const SECRET_KEY = "sk-1234567890abcdef";
        </script>
    </head>
    <body>
        <div data-protect>Sensitive Data</div>
    </body>
    </html>
    '''
    obfuscated = protect.obfuscate_response(test_html)
    print(f"Original: {len(test_html)} chars")
    print(f"Obfuscated: {len(obfuscated)} chars")
    print(f"Comments removed: {'API endpoint' not in obfuscated}")
    print(f"Script injected: {'__devtoolsEnabled' in obfuscated}")

    print("\n=== Error Sanitization Test ===")
    raw_error = "FileNotFoundError: [Errno 2] No such file or directory: '/var/www/app/config/db.py' at line 42 in init_database()"
    sanitized = protect.sanitize_error(raw_error)
    print(f"Raw: {raw_error}")
    print(f"Sanitized: {sanitized}")

    print("\n=== Request Validation Test ===")
    valid, reason = protect.validate_request({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Host': 'localhost:8787'
    })
    print(f"Normal request: valid={valid}, reason={reason}")

    invalid, reason = protect.validate_request({
        'User-Agent': 'sqlmap/1.4.9'
    })
    print(f"Malicious request: valid={invalid}, reason={reason}")
