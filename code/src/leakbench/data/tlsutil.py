"""Self-signed TLS material for the live experiment (RQ8).

The paper's live experiment runs a **real TLS tool-server**. Standing one up needs a
certificate; this module provisions a throwaway self-signed cert for
``127.0.0.1``/``localhost`` at run time, in a temporary directory, using whichever
of these is available:

1. the ``cryptography`` package (if the user installed it), or
2. the ``openssl`` command-line tool.

If neither is available the caller is expected to fall back to plain TCP and to
*record that fact* in the trace metadata --- never to claim TLS it did not run.
Nothing here is downloaded, and no key ever leaves the temporary directory (which
the caller removes).
"""

from __future__ import annotations

import shutil
import ssl
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_SUBJECT_CN = "localhost"
_SAN = "subjectAltName=IP:127.0.0.1,DNS:localhost"


@dataclass
class TLSMaterial:
    """Paths to a throwaway certificate/key pair plus how it was produced."""

    cert_path: Path
    key_path: Path
    directory: Path
    generator: str          # "cryptography" | "openssl"

    def cleanup(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


def _generate_with_cryptography(out_dir: Path) -> TLSMaterial | None:
    try:  # pragma: no cover - depends on an optional package
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        return None

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _SUBJECT_CN)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(__import__("ipaddress").IPv4Address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = out_dir / "cert.pem"
    key_path = out_dir / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return TLSMaterial(cert_path=cert_path, key_path=key_path, directory=out_dir,
                       generator="cryptography")


def _generate_with_openssl(out_dir: Path) -> TLSMaterial | None:
    exe = shutil.which("openssl")
    if exe is None:
        return None
    cert_path = out_dir / "cert.pem"
    key_path = out_dir / "key.pem"
    cmd = [
        exe, "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", str(key_path), "-out", str(cert_path),
        "-days", "1", "-nodes", "-subj", f"/CN={_SUBJECT_CN}",
        "-addext", _SAN,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not (cert_path.exists() and key_path.exists()):
        # Older/LibreSSL builds may reject -addext; retry without a SAN extension.
        cmd_no_san = [c for c in cmd if c not in ("-addext", _SAN)]
        try:
            proc = subprocess.run(cmd_no_san, capture_output=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0 or not (cert_path.exists() and key_path.exists()):
            return None
    return TLSMaterial(cert_path=cert_path, key_path=key_path, directory=out_dir,
                       generator="openssl")


def generate_self_signed(prefix: str = "leakbench_tls_") -> TLSMaterial | None:
    """Create throwaway TLS material, or ``None`` if no generator is available."""
    out_dir = Path(tempfile.mkdtemp(prefix=prefix))
    material = _generate_with_cryptography(out_dir) or _generate_with_openssl(out_dir)
    if material is None:
        shutil.rmtree(out_dir, ignore_errors=True)
    return material


def server_context(material: TLSMaterial) -> ssl.SSLContext:
    """A TLS-1.2+ server context for the throwaway certificate."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(material.cert_path), keyfile=str(material.key_path))
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def client_context(material: TLSMaterial) -> ssl.SSLContext:
    """A client context that trusts *only* this throwaway certificate."""
    return client_context_from_ca(material.cert_path)


def client_context_from_ca(ca_file: str | Path) -> ssl.SSLContext:
    """A client context pinned to one certificate file.

    Hostname checking is disabled because the peer is reached by IP (loopback, or a
    private address across a link); the certificate is still verified against the
    pinned CA file, so this is **not** an unverified context. Used both for the
    locally minted loopback certificate and for a certificate copied from a remote
    tool server (``docs/ec2_experiment_guide.md`` §5a).
    """
    path = Path(ca_file)
    if not path.exists():
        raise FileNotFoundError(
            f"TLS CA file {path} not found. Copy the certificate the tool server "
            f"printed, or pass use_tls=False to capture plaintext TCP."
        )
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=str(path))
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx
