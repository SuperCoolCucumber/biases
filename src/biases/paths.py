from __future__ import annotations

import os
from pathlib import Path


ARTIFACT_ROOT_ENV = "BIASES_ARTIFACT_ROOT"
PROJECT_DIR_NAME = "biases"


def default_artifact_root() -> Path:
    """Return the root for persistent non-code artifacts.

    Large datasets, model caches, and experiment outputs should not be committed
    to the repository. Override this with BIASES_ARTIFACT_ROOT on shared
    infrastructure or when using a large external volume.
    """

    configured = os.environ.get(ARTIFACT_ROOT_ENV)
    if configured:
        return Path(configured).expanduser()

    return Path("artifacts")


def artifact_path(*parts: str) -> Path:
    return default_artifact_root().joinpath(*parts)


def data_path(*parts: str) -> Path:
    return artifact_path("data", *parts)


def output_path(*parts: str) -> Path:
    return artifact_path("outputs", *parts)


def cache_path(*parts: str) -> Path:
    return artifact_path("cache", *parts)


def configure_artifact_environment() -> None:
    """Set cache-related environment defaults to the artifact root."""

    root = default_artifact_root()
    user = os.environ.get("USER") or Path.home().name
    job_id = os.environ.get("SLURM_JOB_ID") or "manual"
    tmp_root = Path(os.environ.get("TMPDIR", f"/tmp/{user}/{job_id}"))
    hf_home = root / "cache" / "huggingface"
    os.environ.setdefault("BIASES_ARTIFACT_ROOT", str(root))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("XDG_CACHE_HOME", str(root / "cache" / "xdg"))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(root / "cache" / "vllm"))
    os.environ.setdefault("TORCH_HOME", str(root / "cache" / "torch"))
    os.environ.setdefault("TMPDIR", str(tmp_root))
    os.environ.setdefault("TRITON_CACHE_DIR", str(tmp_root / "triton"))
