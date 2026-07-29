from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def spec_sha256(spec: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(dict(spec)).encode("utf-8"))


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in sorted(paths, key=str)}
