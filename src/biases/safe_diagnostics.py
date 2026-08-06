from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit


_URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_AUTH_HEADER_PATTERN = re.compile(
    r"(?i)\b(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_SECRET_PARAMETER_PATTERN = re.compile(
    r"(?i)\b(token|access_token|auth_token|api_key|signature|sig|x-amz-signature)"
    r"(\s*[:=]\s*)[^&\s,;]+"
)
_HF_TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9_-]+\b")
_LONG_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z0-9_-]{32,}(?![A-Za-z0-9])"
)


def _strip_url_credentials(match: re.Match[str]) -> str:
    raw_url = match.group(0)
    try:
        parts = urlsplit(raw_url)
        hostname = parts.hostname or ""
        if parts.port is not None:
            hostname = f"{hostname}:{parts.port}"
        return urlunsplit((parts.scheme, hostname, parts.path, "", ""))
    except (TypeError, ValueError):
        return "<redacted-url>"


def sanitize_exception_text(error: BaseException, *, max_length: int = 2000) -> str:
    """Return bounded exception text without URLs' credentials or auth material."""

    if max_length < 1:
        raise ValueError("max_length must be positive")
    message = str(error).replace("\r", " ").replace("\n", " ")
    message = _URL_PATTERN.sub(_strip_url_credentials, message)
    message = _AUTH_HEADER_PATTERN.sub(r"\1<redacted>", message)
    message = _BEARER_PATTERN.sub("Bearer <redacted>", message)
    message = _SECRET_PARAMETER_PATTERN.sub(r"\1\2<redacted>", message)
    message = _HF_TOKEN_PATTERN.sub("hf_<redacted>", message)
    message = _LONG_TOKEN_PATTERN.sub("<redacted-long-token>", message)
    return message[:max_length]
