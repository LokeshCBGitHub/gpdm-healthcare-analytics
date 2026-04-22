import io
import os
import csv
import json
import time
import struct
import hashlib
import hmac
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger('gpdm.email')


def _make_csv_bytes(columns: List[str], rows: List[list]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().encode('utf-8')


def _make_html_report(title: str, columns: List[str], rows: List[list],
                      narrative: str = "", sql: str = "") -> bytes:
    header = ''.join(f'<th style="padding:8px 12px;text-align:left;border-bottom:2px solid #002855;'
                     f'background:#f0f4f8;font-size:13px;">{c}</th>' for c in columns)
    tbody = ''
    for i, row in enumerate(rows[:500]):
        bg = '#ffffff' if i % 2 == 0 else '#f8f9fa'
        cells = ''.join(f'<td style="padding:6px 12px;border-bottom:1px solid #e8ecf0;font-size:13px;">{v}</td>'
                        for v in row)
        tbody += f'<tr style="background:{bg};">{cells}</tr>'

    narr_html = ''
    if narrative:
        narr_html = f'''<div style="background:#f0f7ff;border-left:4px solid #002855;padding:12px 16px;
            margin:16px 0;border-radius:0 6px 6px 0;font-size:14px;line-height:1.6;">
            {narrative}</div>'''

    sql_html = ''
    if sql and not sql.startswith('--'):
        sql_html = f'''<details style="margin:12px 0;"><summary style="cursor:pointer;font-size:12px;
            color:#666;">View SQL</summary><pre style="background:#1e1e2e;color:#cdd6f4;padding:12px;
            border-radius:6px;font-size:12px;overflow-x:auto;">{sql}</pre></details>'''

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;padding:20px;color:#1a1a2e;}}
table{{border-collapse:collapse;width:100%;margin:12px 0;}}
</style></head><body>
<div style="max-width:1000px;margin:0 auto;">
<h2 style="color:#002855;margin-bottom:4px;">{title}</h2>
<p style="color:#666;font-size:13px;">Generated {time.strftime('%B %d, %Y at %I:%M %p')} &bull; {len(rows)} rows</p>
{narr_html}{sql_html}
<table><thead><tr>{header}</tr></thead><tbody>{tbody}</tbody></table>
{f'<p style="color:#999;font-size:12px;">Showing first 500 of {len(rows)} rows</p>' if len(rows) > 500 else ''}
<hr style="border:none;border-top:1px solid #e0e0e0;margin:24px 0;">
<p style="color:#999;font-size:11px;">KP Healthcare Healthcare Analytics &bull; HIPAA Confidential</p>
</div></body></html>'''
    return html.encode('utf-8')


def _encrypt_zip(file_bytes: bytes, filename: str, password: str) -> bytes:
    try:
        import pyzipper
        buf = io.BytesIO()
        with pyzipper.AESZipFile(buf, 'w', compression=pyzipper.ZIP_DEFLATED,
                                  encryption=pyzipper.WZ_AES) as zf:
            zf.setpassword(password.encode('utf-8'))
            zf.writestr(filename, file_bytes)
        return buf.getvalue()
    except ImportError:
        pass

    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, file_bytes)
        zf.writestr('README.txt',
                     f'This file was exported from Healthcare Analytics.\n'
                     f'For encrypted exports, install pyzipper: pip install pyzipper\n'
                     f'Generated: {time.strftime("%Y-%m-%d %H:%M")}\n')
    return buf.getvalue()


def send_results_email(
    smtp_config: Dict[str, Any],
    to_email: str,
    subject: str,
    body_text: str,
    columns: List[str],
    rows: List[list],
    attachment_format: str = 'csv',
    password: Optional[str] = None,
    narrative: str = "",
    sql: str = "",
    sender_name: str = "",
) -> Tuple[bool, str]:
    if not sgpdm_config.get('host'):
        return False, "SGPDM not configured. Ask your admin to set up email in Settings."

    if not to_email or '@' not in to_email:
        return False, "Invalid recipient email address"

    try:
        if attachment_format == 'html':
            file_bytes = _make_html_report(subject, columns, rows, narrative, sql)
            orig_filename = 'report.html'
            mime_type = 'text/html'
        else:
            file_bytes = _make_csv_bytes(columns, rows)
            orig_filename = 'results.csv'
            mime_type = 'text/csv'

        if password:
            attachment_bytes = _encrypt_zip(file_bytes, orig_filename, password)
            attachment_name = orig_filename.rsplit('.', 1)[0] + '_encrypted.zip'
            mime_type = 'application/zip'
            body_text += '\n\nNote: The attachment is password-protected. Use the password shared with you to open it.'
        else:
            attachment_bytes = file_bytes
            attachment_name = orig_filename

        msg = MIMEMultipart()
        msg['From'] = f"{sgpdm_config.get('from_name', 'Healthcare Analytics')} <{smtp_config['from_email']}>"
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body_text, 'plain'))

        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment_bytes)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{attachment_name}"')
        msg.attach(part)

        host = smtp_config['host']
        port = int(sgpdm_config.get('port', 587))
        use_tls = sgpdm_config.get('use_tls', True)

        if use_tls:
            server = smtplib.SGPDM(host, port, timeout=30)
            server.starttls()
        else:
            server = smtplib.SGPDM(host, port, timeout=30)

        if sgpdm_config.get('username') and sgpdm_config.get('password'):
            server.login(smtp_config['username'], smtp_config['password'])

        server.sendmail(smtp_config['from_email'], [to_email], msg.as_string())
        server.quit()

        logger.info("Email sent to %s: %s (%d rows, encrypted=%s)",
                     to_email, subject, len(rows), bool(password))
        return True, f"Email sent to {to_email}"

    except smtplib.SGPDMAuthenticationError:
        return False, "SGPDM authentication failed. Check email credentials in Settings."
    except smtplib.SGPDMConnectError:
        return False, f"Could not connect to SGPDM server {sgpdm_config.get('host')}:{sgpdm_config.get('port')}"
    except Exception as e:
        logger.error("Email send failed: %s", e)
        return False, f"Failed to send email: {str(e)}"


def test_smtp_connection(smtp_config: Dict[str, Any]) -> Tuple[bool, str]:
    if not sgpdm_config.get('host'):
        return False, "SGPDM host not configured"
    try:
        host = smtp_config['host']
        port = int(sgpdm_config.get('port', 587))
        server = smtplib.SGPDM(host, port, timeout=10)
        if sgpdm_config.get('use_tls', True):
            server.starttls()
        if sgpdm_config.get('username') and sgpdm_config.get('password'):
            server.login(smtp_config['username'], smtp_config['password'])
        server.quit()
        return True, f"Connected to {host}:{port} successfully"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"
