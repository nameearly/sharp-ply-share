import os
import socket
from urllib.parse import urlparse, urlunparse


def _env_flag(name: str, default: bool = False) -> bool:
    try:
        v = os.getenv(name)
        if v is None:
            return bool(default)
        s = str(v).strip().lower()
        if not s:
            return bool(default)
        return s in ("1", "true", "yes", "y", "on")
    except Exception:
        return bool(default)


def _is_ip_literal(host: str) -> bool:
    try:
        import ipaddress

        ipaddress.ip_address(str(host))
        return True
    except Exception:
        return False


def _is_private_ip(host: str) -> bool:
    try:
        import ipaddress

        ip = ipaddress.ip_address(str(host))
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast)
    except Exception:
        return False


def _host_is_local(host: str) -> bool:
    try:
        h = str(host or "").strip().lower().strip(".")
        if not h:
            return True
        if h in ("localhost",):
            return True
        return False
    except Exception:
        return True


def validate_external_url(url: str, *, allow_private: bool | None = None) -> str:
    """Validate an externally-provided URL for safe fetching.

    Policy (default):
    - Only allow http(s)
    - Require a host
    - Block localhost and private/loopback/link-local IP literals

    Set env URL_ALLOW_PRIVATE=1 to allow private hosts.
    """

    u = str(url or "").strip()
    if not u:
        raise ValueError("empty url")

    p = urlparse(u)
    scheme = str(p.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"unsupported url scheme: {scheme}")

    netloc = str(p.netloc or "")
    if not netloc:
        raise ValueError("url missing host")

    host = netloc.split("@")[-1].split(":", 1)[0].strip().strip("[]")
    if not host:
        raise ValueError("url missing host")

    if allow_private is None:
        allow_private = _env_flag("URL_ALLOW_PRIVATE", False)

    if not allow_private:
        if _host_is_local(host):
            raise ValueError("disallowed host")
        if _is_ip_literal(host) and _is_private_ip(host):
            raise ValueError("disallowed private ip")

        # Optional DNS resolution (best-effort). Disabled by default.
        if _env_flag("URL_VALIDATE_DNS", False) and (not _is_ip_literal(host)):
            try:
                for res in socket.getaddrinfo(host, None):
                    ip = res[4][0]
                    if _is_private_ip(ip):
                        raise ValueError("disallowed private ip (dns)")
            except ValueError:
                raise
            except Exception:
                # If DNS fails, reject to be safe.
                raise ValueError("dns resolution failed")

    # Normalize: drop fragment
    p2 = p._replace(fragment="")
    return urlunparse(p2)


def safe_requests_get(url: str, *, timeout: float, stream: bool = False, headers: dict | None = None):
    """requests.get wrapper for externally-provided URLs."""

    import requests

    u2 = validate_external_url(url)
    return requests.get(u2, timeout=timeout, stream=bool(stream), headers=headers, allow_redirects=False)
