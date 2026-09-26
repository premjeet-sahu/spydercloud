from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .runtime import ProviderError

ROUTE_CRITERIA = {
    "billing": "Charges, invoices, payments, subscriptions, or refunds.",
    "technical": "Product defects, errors, integrations, outages, or technical troubleshooting.",
    "sales": "Pricing, quotes, demos, plans, or purchase evaluation.",
    "account": "Profile, access, password, account ownership, or account administration.",
    "other": "Anything that does not fit the other teams.",
}
ROUTES = tuple(ROUTE_CRITERIA)
QUESTION_TEXT = {
    "route": "Which team should handle the ticket based on its primary intent?",
    "urgent": (
        "Does the ticket indicate a time-critical business impact, an active outage, "
        "a hard deadline, or an explicit need for immediate action?"
    ),
    "refund_requested": "Does the customer explicitly ask for money to be returned?",
}


@dataclass
class Prediction:
    provider: str
    route: str
    route_confidence: float
    urgent_probability: float
    refund_probability: float
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    raw: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    route_rationale: str | None = None
    urgent_rationale: str | None = None
    refund_rationale: str | None = None

    @property
    def urgent(self) -> bool:
        return self.urgent_probability >= 0.5

    @property
    def refund_requested(self) -> bool:
        return self.refund_probability >= 0.5

    @property
    def decision_confidence(self) -> float:
        binary_confidences = (
            abs(2 * self.urgent_probability - 1),
            abs(2 * self.refund_probability - 1),
        )
        return min(self.route_confidence, *binary_confidences)


def _clamp(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def _format_probability_map(probabilities: dict[str, Any] | None) -> str:
    if not probabilities:
        return "No probability map returned."
    parts = []
    for label, probability in sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True):
        parts.append(f"{label}: {float(probability) * 100:.0f}%")
    return ", ".join(parts)


@contextmanager
def _without_aws_profile():
    profile = os.environ.pop("AWS_PROFILE", None)
    try:
        yield
    finally:
        if profile is not None:
            os.environ["AWS_PROFILE"] = profile


class RulesProvider:
    name = "rules"

    def predict(self, text: str) -> Prediction:
        started = time.perf_counter()
        lowered = text.lower()
        route_terms = {
            "billing": ("charge", "charged", "invoice", "refund", "payment", "subscription", "card"),
            "technical": ("error", "bug", "broken", "crash", "failing", "integration", "api", "login"),
            "sales": ("price", "pricing", "quote", "demo", "plan", "enterprise", "purchase"),
            "account": ("account", "email address", "password", "locked", "profile", "cancel"),
        }
        scores = {route: sum(term in lowered for term in terms) for route, terms in route_terms.items()}
        best_route = max(scores, key=scores.get)
        route = best_route if scores[best_route] else "other"
        tied = sum(value == scores[best_route] for value in scores.values()) > 1
        route_confidence = 0.55 if tied else (0.8 if scores[best_route] else 0.4)
        urgent = any(term in lowered for term in ("urgent", "asap", "immediately", "production down", "blocked"))
        refund = any(term in lowered for term in ("refund", "money back", "reverse the charge"))
        return Prediction(
            provider=self.name,
            route=route,
            route_confidence=route_confidence,
            urgent_probability=0.9 if urgent else 0.1,
            refund_probability=0.9 if refund else 0.1,
            latency_ms=(time.perf_counter() - started) * 1000,
            model="keyword-rules-v1",
        )


class JevProvider:
    name = "jev"
    endpoint = "https://jev-ai.pro/api/v1/systemone"

    def __init__(self) -> None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.api_key = (os.environ.get("JEV_AI_API_KEY") or os.environ.get("TYPESAFE_API_KEY", "")).strip()
        self.api_key_source = "JEV_AI_API_KEY" if os.environ.get("JEV_AI_API_KEY") else "TYPESAFE_API_KEY"
        self.model = os.environ.get("JEV_MODEL", "jev-latest")
        self.endpoint = os.environ.get("JEV_API_ENDPOINT", self.endpoint).strip()
        if not self.api_key or self.api_key == "replace-me":
            raise ProviderError(
                provider=self.name,
                code="missing_credentials",
                message="Set JEV_AI_API_KEY or TYPESAFE_API_KEY in .env before calling Jev.",
            )
        self.timeout = (
            float(os.environ.get("JEV_CONNECT_TIMEOUT_SECONDS", "5")),
            float(os.environ.get("JEV_READ_TIMEOUT_SECONDS", "30")),
        )
        retry = Retry(
            total=int(os.environ.get("JEV_MAX_RETRIES", "2")),
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("POST",),
            respect_retry_after_header=True,
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.requests = requests

    def predict(self, text: str) -> Prediction:
        payload = {
            "state": {"ticket_text": text},
            "model": self.model,
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": f"{QUESTION_TEXT['route']} Evaluate `ticket_text`.",
                    "criteria": ROUTE_CRITERIA,
                },
                "urgent": {
                    "type": "noul",
                    "instructions": f"{QUESTION_TEXT['urgent']} Evaluate `ticket_text`.",
                },
                "refund_requested": {
                    "type": "noul",
                    "instructions": f"{QUESTION_TEXT['refund_requested']} Evaluate `ticket_text`.",
                },
            },
        }
        started = time.perf_counter()
        try:
            response = self.session.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except self.requests.exceptions.Timeout as exc:
            raise ProviderError(
                provider=self.name,
                code="timeout",
                message="Jev did not respond within the configured timeout.",
                retryable=True,
                details={"timeout_seconds": self.timeout},
            ) from exc
        except self.requests.exceptions.ConnectionError as exc:
            raise ProviderError(
                provider=self.name,
                code="connection_error",
                message="Could not connect to the Jev API.",
                retryable=True,
            ) from exc

        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        retry_state = getattr(response.raw, "retries", None)
        retry_count = len(getattr(retry_state, "history", ()))
        if response.status_code >= 400:
            if response.status_code == 401:
                code, message, retryable = "authentication_failed", "The Jev API key was rejected.", False
            elif response.status_code == 403:
                code, message, retryable = "access_denied", "The Jev account cannot access this operation.", False
            elif response.status_code == 429:
                code, message, retryable = "throttled", "Jev rate-limited the request after retries.", True
            elif response.status_code >= 500:
                code, message, retryable = "service_unavailable", "Jev returned a service error after retries.", True
            else:
                code, message, retryable = "request_rejected", "Jev rejected the request.", False
            raise ProviderError(
                provider=self.name,
                code=code,
                message=message,
                retryable=retryable,
                details={
                    "http_status": response.status_code,
                    "request_id": request_id,
                    "retry_count": retry_count,
                    "endpoint": self.endpoint,
                    "model": self.model,
                    "api_key_source": self.api_key_source,
                    "response_body": response.text[:1000],
                },
            )
        try:
            body = response.json()
            answers = body["answers"]
            usage = body.get("usage", {})
            route_answer = answers["route"]
            urgent_answer = answers["urgent"]
            refund_answer = answers["refund_requested"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                provider=self.name,
                code="invalid_response",
                message="Jev returned a response that did not match the expected schema.",
                details={"http_status": response.status_code, "request_id": request_id},
            ) from exc
        elapsed = (time.perf_counter() - started) * 1000
        route_probability_map = route_answer.get("probabilities")
        route_rationale = (
            f"Jev returned route probabilities: {_format_probability_map(route_probability_map)}"
            if route_probability_map
            else f"Jev returned choice confidence {float(route_answer.get('confidence', 0.0)) * 100:.0f}%."
        )
        urgent_noul = _clamp(urgent_answer["noul"])
        refund_noul = _clamp(refund_answer["noul"])
        return Prediction(
            provider=self.name,
            route=route_answer["choice"],
            route_confidence=_clamp(route_answer["confidence"]),
            urgent_probability=urgent_noul,
            refund_probability=refund_noul,
            latency_ms=elapsed,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            model=body.get("model", self.model),
            raw=body,
            diagnostics={
                "http_status": response.status_code,
                "request_id": request_id,
                "retry_count": retry_count,
                "endpoint": self.endpoint,
                "model": self.model,
                "api_key_source": self.api_key_source,
            },
            route_rationale=route_rationale,
            urgent_rationale=f"Jev returned {urgent_noul * 100:.0f}% yes probability for urgency.",
            refund_rationale=f"Jev returned {refund_noul * 100:.0f}% yes probability for refund requested.",
        )


class BedrockProvider:
    name = "bedrock"

    def __init__(self) -> None:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import ProfileNotFound

        self.region = os.environ.get("AWS_REGION", "us-east-1")
        self.model = os.environ.get(
            "BEDROCK_MODEL_ID",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
        session_token = os.environ.get("AWS_SESSION_TOKEN", "").strip() or None
        profile = os.environ.get("AWS_PROFILE", "").strip() or None

        if bool(access_key) != bool(secret_key):
            raise ProviderError(
                provider=self.name,
                code="incomplete_credentials",
                message="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must both be set.",
            )

        if access_key and secret_key:
            with _without_aws_profile():
                session = boto3.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    aws_session_token=session_token,
                    region_name=self.region,
                )
            self.credential_source = "environment_access_keys"
        elif profile:
            try:
                session = boto3.Session(profile_name=profile, region_name=self.region)
                self.credential_source = f"profile:{profile}"
            except ProfileNotFound:
                with _without_aws_profile():
                    session = boto3.Session(region_name=self.region)
                self.credential_source = "default_chain_after_missing_profile"
        else:
            session = boto3.Session(region_name=self.region)
            self.credential_source = "boto3_default_chain"

        config = Config(
            connect_timeout=float(os.environ.get("BEDROCK_CONNECT_TIMEOUT_SECONDS", "5")),
            read_timeout=float(os.environ.get("BEDROCK_READ_TIMEOUT_SECONDS", "30")),
            retries={
                "max_attempts": int(os.environ.get("BEDROCK_MAX_ATTEMPTS", "3")),
                "mode": "standard",
            },
            user_agent_extra="jev-evaluation-poc/0.2.0",
        )
        self.client = session.client("bedrock-runtime", config=config)

    def predict(self, text: str) -> Prediction:
        schema = {
            "type": "object",
            "properties": {
                "route": {"type": "string", "enum": list(ROUTES)},
                "route_confidence": {"type": "number"},
                "urgent_probability": {"type": "number"},
                "refund_probability": {"type": "number"},
                "route_rationale": {"type": "string"},
                "urgent_rationale": {"type": "string"},
                "refund_rationale": {"type": "string"},
            },
            "required": [
                "route",
                "route_confidence",
                "urgent_probability",
                "refund_probability",
                "route_rationale",
                "urgent_rationale",
                "refund_rationale",
            ],
        }
        route_definitions = "\n".join(
            f"- {route}: {description}" for route, description in ROUTE_CRITERIA.items()
        )
        prompt = f"""Classify this support ticket using exactly the same questions as the comparison model.

Route definitions:
{route_definitions}

Questions:
1. {QUESTION_TEXT['route']}
2. {QUESTION_TEXT['urgent']} Return its yes probability as urgent_probability.
3. {QUESTION_TEXT['refund_requested']} Return its yes probability as refund_probability.
Also return route_confidence from 0 to 1 for the selected route.
Return concise rationales for route_rationale, urgent_rationale, and refund_rationale. Each rationale must be one short sentence grounded only in the ticket text.

Ticket:
{text}"""
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "classify_ticket",
                        "description": "Return the support ticket classification.",
                        "inputSchema": {"json": schema},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": "classify_ticket"}},
        }
        started = time.perf_counter()
        try:
            response = self.client.converse(
                modelId=self.model,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                toolConfig=tool_config,
                inferenceConfig={"maxTokens": 512, "temperature": 0.0},
            )
        except Exception as exc:
            raise self._provider_error(exc) from exc
        elapsed = (time.perf_counter() - started) * 1000
        metadata = response.get("ResponseMetadata", {})
        try:
            blocks = response["output"]["message"]["content"]
            tool_uses = [block["toolUse"] for block in blocks if "toolUse" in block]
            if not tool_uses:
                raise KeyError("toolUse")
            result = tool_uses[0]["input"]
            route = str(result["route"]).lower()
            if route not in ROUTES:
                raise ValueError(f"Unsupported route: {route}")
            usage = response.get("usage", {})
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                provider=self.name,
                code="invalid_response",
                message="Bedrock returned a response that did not match the classification schema.",
                details={
                    "request_id": metadata.get("RequestId"),
                    "http_status": metadata.get("HTTPStatusCode"),
                },
            ) from exc
        return Prediction(
            provider=self.name,
            route=route,
            route_confidence=_clamp(result["route_confidence"]),
            urgent_probability=_clamp(result["urgent_probability"]),
            refund_probability=_clamp(result["refund_probability"]),
            latency_ms=elapsed,
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            model=self.model,
            raw=response,
            diagnostics={
                "credential_source": self.credential_source,
                "request_id": metadata.get("RequestId"),
                "http_status": metadata.get("HTTPStatusCode"),
                "retry_attempts": metadata.get("RetryAttempts"),
            },
            route_rationale=str(result["route_rationale"]),
            urgent_rationale=str(result["urgent_rationale"]),
            refund_rationale=str(result["refund_rationale"]),
        )

    def _provider_error(self, exc: Exception) -> ProviderError:
        from botocore.exceptions import (
            ClientError,
            ConnectTimeoutError,
            EndpointConnectionError,
            NoCredentialsError,
            PartialCredentialsError,
            ReadTimeoutError,
        )

        if isinstance(exc, NoCredentialsError):
            return ProviderError(
                self.name,
                "missing_credentials",
                "No AWS credentials were found. Configure access keys, a profile, or an IAM role.",
            )
        if isinstance(exc, PartialCredentialsError):
            return ProviderError(
                self.name,
                "incomplete_credentials",
                "AWS credentials are incomplete.",
            )
        if isinstance(exc, (ConnectTimeoutError, ReadTimeoutError)):
            return ProviderError(
                self.name,
                "timeout",
                "Amazon Bedrock did not respond within the configured timeout.",
                retryable=True,
            )
        if isinstance(exc, EndpointConnectionError):
            return ProviderError(
                self.name,
                "connection_error",
                "Could not connect to Amazon Bedrock. Check the Region and network.",
                retryable=True,
            )
        if isinstance(exc, ClientError):
            response = exc.response
            error = response.get("Error", {})
            metadata = response.get("ResponseMetadata", {})
            aws_code = error.get("Code", "ClientError")
            throttling_codes = {"ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"}
            credential_codes = {"UnrecognizedClientException", "InvalidSignatureException", "ExpiredTokenException"}
            if aws_code in throttling_codes:
                code, message, retryable = "throttled", "Amazon Bedrock throttled the request after retries.", True
            elif aws_code in credential_codes:
                code, message, retryable = "authentication_failed", "AWS rejected the configured credentials.", False
            elif aws_code in {"AccessDeniedException", "UnauthorizedException"}:
                code, message, retryable = "access_denied", "AWS credentials lack access to the selected Bedrock model.", False
            elif aws_code in {"ResourceNotFoundException", "ValidationException"}:
                code, message, retryable = "invalid_model_or_request", "Check the Bedrock model ID, Region, and model access.", False
            elif aws_code in {"InternalServerException", "ServiceUnavailableException", "ModelTimeoutException"}:
                code, message, retryable = "service_unavailable", "Amazon Bedrock returned a service error after retries.", True
            else:
                code, message, retryable = "aws_client_error", "Amazon Bedrock rejected the request.", False
            return ProviderError(
                provider=self.name,
                code=code,
                message=message,
                retryable=retryable,
                details={
                    "aws_error_code": aws_code,
                    "aws_error_message": error.get("Message"),
                    "request_id": metadata.get("RequestId"),
                    "http_status": metadata.get("HTTPStatusCode"),
                    "retry_attempts": metadata.get("RetryAttempts"),
                    "credential_source": self.credential_source,
                },
            )
        return ProviderError(
            provider=self.name,
            code="unexpected_error",
            message="An unexpected Bedrock client error occurred.",
            details={"exception_type": type(exc).__name__, "credential_source": self.credential_source},
        )


def get_provider(name: str):
    normalized = name.strip().lower()
    if normalized == "rules":
        return RulesProvider()
    if normalized == "jev":
        return JevProvider()
    if normalized == "bedrock":
        return BedrockProvider()
    raise ValueError(f"Unknown provider {name!r}; choose rules, jev, or bedrock")


def load_dotenv(path: str = ".env") -> None:
    """Load a minimal KEY=VALUE file without adding another dependency."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().replace("\\_", "_")
            value = re.sub(r"^(['\"])(.*)\1$", r"\2", value.strip()).replace("\\_", "_")
            os.environ[key] = value

