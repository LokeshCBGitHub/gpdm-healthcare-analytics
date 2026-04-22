from __future__ import annotations

import datetime
import ipaddress
import logging
import os
import socket
import subprocess
from typing import List, Optional, Tuple

_log = logging.getLogger("gpdm.tls")


def _local_hostnames() -> List[str]:
    names = ["localhost"]
    try:
        names.append(socket.gethostname())
    except Exception:
        pass
    try:
        names.append(socket.getfqdn())
    except Exception:
        pass
    seen = set()
    out = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _local_ips() -> List[str]:
    ips = ["127.0.0.1", "::1"]
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        s6 = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        s6.connect(("2001:4860:4860::8888", 80))
        ips.append(s6.getsockname()[0])
        s6.close()
    except Exception:
        pass
    seen = set()
    out = []
    for ip in ips:
        if ip and ip not in seen:
            seen.add(ip)
            out.append(ip)
    return out


def _mint_with_cryptography(cert_path: str, key_path: str,
                            hostnames: List[str], ips: List[str]) -> bool:
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        return False

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, hostnames[0]),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "GPDM"),
    ])
    san_entries: List[x509.GeneralName] = [x509.DNSName(h) for h in hostnames]
    for ip in ips:
        try:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass
    now = datetime.datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None),
                       critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_encipherment=True,
                content_commitment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    pem_cert = cert.public_bytes(serialization.Encoding.PEM)
    pem_key = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    os.makedirs(os.path.dirname(cert_path), exist_ok=True)
    with open(cert_path, "wb") as f:
        f.write(pem_cert)
    with open(key_path, "wb") as f:
        f.write(pem_key)
    try:
        os.chmod(key_path, 0o600)
        os.chmod(cert_path, 0o644)
    except OSError:
        pass
    _log.info("Minted self-signed cert (cryptography) for %s",
              ", ".join(hostnames + ips))
    return True


def _mint_with_openssl(cert_path: str, key_path: str,
                       hostnames: List[str], ips: List[str]) -> bool:
    try:
        subprocess.run(["openssl", "version"], capture_output=True, check=True,
                       timeout=5)
    except Exception:
        return False
    san_parts: List[str] = []
    for i, h in enumerate(hostnames, 1):
        san_parts.append(f"DNS.{i} = {h}")
    for i, ip in enumerate(ips, 1):
        san_parts.append(f"IP.{i} = {ip}")
    san_block = "\n".join(san_parts)
    cnf_path = cert_path + ".cnf"
    with open(cnf_path, "w") as f:
        f.write(
            "[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=ext\n"
            "[dn]\nCN=" + hostnames[0] + "\nO=GPDM\n"
            "[ext]\nsubjectAltName=@san\nbasicConstraints=critical,CA:false\n"
            "keyUsage=critical,digitalSignature,keyEncipherment\n"
            "extendedKeyUsage=serverAuth\n"
            "[san]\n" + san_block + "\n"
        )
    try:
        os.makedirs(os.path.dirname(cert_path), exist_ok=True)
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-keyout", key_path, "-out", cert_path,
             "-days", "825", "-config", cnf_path],
            check=True, capture_output=True, timeout=30,
        )
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass
        _log.info("Minted self-signed cert (openssl) for %s",
                  ", ".join(hostnames + ips))
        return True
    except Exception as e:
        _log.warning("openssl mint failed: %s", e)
        return False
    finally:
        try:
            os.remove(cnf_path)
        except OSError:
            pass


def _trust_cert_in_os(cert_path: str) -> bool:
    import platform
    marker = cert_path + ".trusted"
    if os.path.exists(marker):
        return True

    system = platform.system()
    ok = False
    try:
        if system == "Darwin":
            r = subprocess.run(
                ["sudo", "-n", "security", "add-trusted-cert",
                 "-d", "-r", "trustRoot",
                 "-k", "/Library/Keychains/System.keychain",
                 cert_path],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                _log.info("Installing HTTPS cert into macOS trust store "
                          "(enter your Mac password if prompted)...")
                r = subprocess.run(
                    ["sudo", "security", "add-trusted-cert",
                     "-d", "-r", "trustRoot",
                     "-k", "/Library/Keychains/System.keychain",
                     cert_path],
                    timeout=60,
                )
            ok = r.returncode == 0

        elif system == "Windows":
            r = subprocess.run(
                ["certutil", "-addstore", "Root", cert_path],
                capture_output=True, text=True, timeout=30,
            )
            ok = r.returncode == 0

        elif system == "Linux":
            dest = None
            update_cmd = None
            if os.path.isdir("/usr/local/share/ca-certificates"):
                dest = "/usr/local/share/ca-certificates/gpdm-dashboard.crt"
                update_cmd = ["sudo", "update-ca-certificates"]
            elif os.path.isdir("/etc/pki/ca-trust/source/anchors"):
                dest = "/etc/pki/ca-trust/source/anchors/gpdm-dashboard.crt"
                update_cmd = ["sudo", "update-ca-trust"]
            if dest and update_cmd:
                subprocess.run(["sudo", "cp", cert_path, dest],
                               timeout=10, check=True)
                subprocess.run(update_cmd, timeout=30, check=True)
                ok = True

    except Exception as e:
        _log.warning("Could not auto-trust cert: %s (non-fatal — "
                     "you may need to accept the cert warning in the browser)", e)
        return False

    if ok:
        try:
            with open(marker, "w") as f:
                f.write("trusted\n")
        except OSError:
            pass
        _log.info("Cert installed in OS trust store — browsers will trust HTTPS.")
    else:
        _log.warning("Could not install cert in OS trust store. "
                     "You may see a browser warning; click Advanced -> Proceed.")
    return ok


def ensure_self_signed_cert(cert_dir: str,
                            hostnames: Optional[List[str]] = None,
                            ) -> Optional[Tuple[str, str]]:
    cert_path = os.path.join(cert_dir, "cert.pem")
    key_path = os.path.join(cert_dir, "key.pem")

    freshly_minted = False
    if os.path.exists(cert_path) and os.path.exists(key_path):
        _trust_cert_in_os(cert_path)
        return cert_path, key_path

    names = list(hostnames) if hostnames else _local_hostnames()
    ips = _local_ips()

    if _mint_with_cryptography(cert_path, key_path, names, ips):
        freshly_minted = True
    elif _mint_with_openssl(cert_path, key_path, names, ips):
        freshly_minted = True

    if freshly_minted:
        _trust_cert_in_os(cert_path)
        return cert_path, key_path

    _log.error(
        "TLS auto-cert failed: install the 'cryptography' wheel or an "
        "openssl binary on the host.  Falling back to plain HTTP.")
    return None
