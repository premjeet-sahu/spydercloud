from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_VERSION = "0.2.0"
SENSITIVE_KEYS = {
    "authorization",
    "x_amz_security_token",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "access_key_id",
    "secret_access_key",
    "session_token",
    "api_key",
    "password",
}


@dataclass
class ProviderError(Exception):
    provider: str
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": sanitize(self.details),
        }


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            cleaned[key] = "[REDACTED]" if normalized in SENSITIVE_KEYS else sanitize(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


def credential_hint() -> dict[str, Any]:
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    session_token = os.environ.get("AWS_SESSION_TOKEN", "").strip()
    profile = os.environ.get("AWS_PROFILE", "").strip()

    if access_key and secret_key:
        return {
            "source": "environment_access_keys",
            "access_key": f"***{access_key[-4:]}" if len(access_key) >= 4 else "***",
            "temporary_credentials": bool(session_token),
        }
    if access_key or secret_key:
        return {"source": "incomplete_environment_access_keys"}
    if profile:
        return {"source": "aws_profile", "profile": profile}
    return {"source": "boto3_default_chain"}


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def system_diagnostics(project_root: Path | None = None) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "application_version": APP_VERSION,
        "python": sys.version.split()[0],
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "project_root": str(project_root.resolve()) if project_root else None,
        "packages": {
            "streamlit": _package_version("streamlit"),
            "boto3": _package_version("boto3"),
            "requests": _package_version("requests"),
        },
        "aws": {
            "region": os.environ.get("AWS_REGION", "us-east-1"),
            "model": os.environ.get("BEDROCK_MODEL_ID"),
            "credential_configuration": credential_hint(),
        },
        "jev": {"model": os.environ.get("JEV_MODEL", "jev-1.13.0")},
    }

